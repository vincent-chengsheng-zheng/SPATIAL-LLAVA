"""
core/model/spatial_llava.py

Spatial-LLaVA: LLaVA model fine-tuned for spatial understanding.

Integrates:
- LLaVA-7B for vision-language understanding
- LoRA for efficient fine-tuning
- Regression head for bounding box prediction

The model processes image + text inputs and predicts normalized bounding boxes
[x_center, y_center, width, height] in [0, 1].

Usage:
    from core.model.spatial_llava import SpatialLLaVA

    model = SpatialLLaVA()
    outputs = model(images, input_ids)  # dict with 'bbox' key
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional
from transformers import LlavaForConditionalGeneration, AutoProcessor
from peft import get_peft_model, PeftModel

from .lora_config import get_lora_config
from .regression_head import RegressionHead


class SpatialLLaVA(nn.Module):
    """
    LLaVA-7B with LoRA adaptation and bounding box regression head.

    Architecture:
        Image + Text → LLaVA → LoRA → [LOC] token hidden state → Regression Head → BBox

    Args:
        model_id       : HuggingFace model ID (default: "llava-hf/llava-1.5-7b-hf")
        lora_rank      : LoRA adaptation rank (default: 16)
        freeze_vision  : Whether to freeze vision encoder (default: True)
        freeze_llm     : Whether to freeze LLM backbone (default: False, LoRA only)
    """

    def __init__(
        self,
        model_id: str = "llava-hf/llava-1.5-7b-hf",
        lora_rank: int = 16,
        freeze_vision: bool = True,
        freeze_llm: bool = False,
    ):
        super().__init__()

        self.model_id = model_id
        self.lora_rank = lora_rank

        # ── Load LLaVA model and processor ──────────────────────────────────────
        print(f"[SpatialLLaVA] Loading {model_id}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.llava = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,  # Use fp16 for memory efficiency
            device_map="auto",          # Auto GPU placement
            low_cpu_mem_usage=True,
        )

        # ── Freeze components ──────────────────────────────────────────────────
        if freeze_vision:
            self._freeze_vision_encoder()
            print("[SpatialLLaVA] Vision encoder frozen")

        if freeze_llm:
            self._freeze_llm_backbone()
            print("[SpatialLLaVA] LLM backbone frozen")

        # ── Apply LoRA adaptation ─────────────────────────────────────────────
        lora_config = get_lora_config(rank=lora_rank)
        self.llava = get_peft_model(self.llava, lora_config)
        print(f"[SpatialLLaVA] LoRA applied (rank={lora_rank})")

        # ── Add regression head ───────────────────────────────────────────────
        # Get hidden size from LLM config
        hidden_size = self.llava.config.text_config.hidden_size
        self.regression_head = RegressionHead(hidden_size=hidden_size)
        print(f"[SpatialLLaVA] Regression head added (hidden_size={hidden_size})")

        # ── Special token setup ───────────────────────────────────────────────
        # LLaVA uses <loc> token for spatial reasoning
        self.loc_token_id = self.processor.tokenizer.convert_tokens_to_ids("<loc>")
        if self.loc_token_id == self.processor.tokenizer.unk_token_id:
            # Fallback: try different token formats
            for token in ["<loc>", "[LOC]", "<LOC>"]:
                token_id = self.processor.tokenizer.convert_tokens_to_ids(token)
                if token_id != self.processor.tokenizer.unk_token_id:
                    self.loc_token_id = token_id
                    break

        if self.loc_token_id == self.processor.tokenizer.unk_token_id:
            raise ValueError("Could not find <loc> token in tokenizer vocabulary")

        print(f"[SpatialLLaVA] Using location token ID: {self.loc_token_id}")

    def _freeze_vision_encoder(self):
        """Freeze vision encoder parameters."""
        for param in self.llava.vision_tower.parameters():
            param.requires_grad = False

    def _freeze_llm_backbone(self):
        """Freeze LLM backbone parameters (LoRA will still train adapters)."""
        for param in self.llava.language_model.parameters():
            param.requires_grad = False

    def forward(
        self,
        images: Tensor,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass: image + text → bounding box prediction.

        Args:
            images         : Tensor (B, 3, 384, 384) — preprocessed images
            input_ids      : Tensor (B, seq_len) — tokenized text with <loc> token
            attention_mask : Tensor (B, seq_len) — attention mask (optional)

        Returns:
            Dict with:
                'bbox' : Tensor (B, 4) — predicted [x_c, y_c, w, h] in [0, 1]
                'loss' : Tensor (1,) — training loss (only during training)
        """
        batch_size = images.shape[0]

        # ── Prepare inputs for LLaVA ──────────────────────────────────────────
        # LLaVA expects pixel values, not preprocessed tensors
        # We need to convert back to pixel values for the vision encoder
        pixel_values = self._tensor_to_pixel_values(images)

        inputs = {
            "pixel_values": pixel_values,      # (B, 3, 336, 336) for CLIP
            "input_ids": input_ids,            # (B, seq_len)
        }
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask

        # ── Forward through LLaVA ────────────────────────────────────────────
        outputs = self.llava(**inputs, output_hidden_states=True)

        # ── Extract [LOC] token hidden state ────────────────────────────────
        # Get last hidden layer: (B, seq_len, hidden_size)
        last_hidden_states = outputs.hidden_states[-1]

        # Find [LOC] token positions: (B, seq_len) → (B,) indices
        loc_positions = (input_ids == self.loc_token_id).float().argmax(dim=1)

        # Extract hidden states at [LOC] positions: (B, hidden_size)
        loc_hidden_states = last_hidden_states[
            torch.arange(batch_size), loc_positions
        ]

        # ── Predict bounding boxes ───────────────────────────────────────────
        pred_bbox = self.regression_head(loc_hidden_states)  # (B, 4)

        result = {"bbox": pred_bbox}

        # Add loss if we have targets (during training)
        if hasattr(self, '_targets') and self._targets is not None:
            from ..loss.spatial_loss import spatial_loss
            result["loss"] = spatial_loss(pred_bbox, self._targets)

        return result

    def _tensor_to_pixel_values(self, images: Tensor) -> Tensor:
        """
        Convert preprocessed tensors back to pixel values for LLaVA vision encoder.

        LLaVA's vision encoder expects unnormalized pixel values in [0, 1],
        but our preprocessing normalizes with ImageNet stats.

        Args:
            images : Tensor (B, 3, 384, 384) — normalized with ImageNet stats

        Returns:
            Tensor (B, 3, 336, 336) — pixel values for CLIP vision encoder
        """
        # Reverse ImageNet normalization
        from ..data.preprocessing import IMAGENET_MEAN, IMAGENET_STD
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(images.device)

        pixel_values = images * std + mean  # Unnormalize

        # Resize to CLIP input size (336x336)
        pixel_values = torch.nn.functional.interpolate(
            pixel_values, size=(336, 336), mode='bilinear', align_corners=False
        )

        return pixel_values

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable and total parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total": total_params,
            "trainable": trainable_params,
            "trainable_percent": 100 * trainable_params / total_params,
        }

    def save_pretrained(self, path: str):
        """Save model and LoRA adapters."""
        self.llava.save_pretrained(path)
        torch.save(self.regression_head.state_dict(), f"{path}/regression_head.pth")

    def load_pretrained(self, path: str):
        """Load model and LoRA adapters."""
        self.llava = PeftModel.from_pretrained(self.llava, path)
        self.regression_head.load_state_dict(
            torch.load(f"{path}/regression_head.pth")
        )
