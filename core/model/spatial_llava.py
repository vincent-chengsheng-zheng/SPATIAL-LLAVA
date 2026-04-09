"""
core/model/spatial_llava.py

SpatialLLaVA: LLaVA backbone (optionally + LoRA) + MLP regression head.

Architecture:
    Image + Text
        → LlavaForConditionalGeneration (vision encoder + LLM)
        → hidden states of the [LOC] token (last token position)
        → RegressionHead (MLP)
        → [xc, yc, w, h]

How [LOC] token is extracted:
    We run a forward pass through LLaVA WITHOUT generating text.
    We take the hidden state at the LAST input token position (the [LOC] token
    is always appended at the end of the prompt), and feed it to the MLP head.

Two modes:
    use_lora=True  → step2_train_main.py    (LoRA + head trained)
    use_lora=False → step3_train_ablation.py (backbone frozen, head only)

Usage:
    from core.model.spatial_llava import SpatialLLaVA, load_model

    model, processor = load_model(use_lora=True, device="cuda")
    bbox = model(input_ids, attention_mask, pixel_values)  # (B, 4)
"""

import os
import sys

import torch
import torch.nn as nn
from torch import Tensor
from transformers import LlavaForConditionalGeneration, AutoProcessor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from core.paths import PATHS                        # noqa: E402
from core.model.regression_head import RegressionHead  # noqa: E402
from core.model.lora_config import apply_lora           # noqa: E402

# LLaVA model to use — hosted on HuggingFace
LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"

# Hidden dim of LLaVA-1.5-7B language model (Vicuna / LLaMA)
HIDDEN_DIM = 4096


# ── Model ─────────────────────────────────────────────────────────────────────

class SpatialLLaVA(nn.Module):
    """
    LLaVA backbone + optional LoRA + MLP regression head.

    Args:
        use_lora   : If True, inject LoRA into language model backbone.
                     If False, freeze backbone entirely (ablation mode).
        hidden_dim : LLM hidden state dimension (4096 for 7B).
        dropout    : Dropout for regression head.

    Forward input:
        input_ids      : LongTensor(B, seq_len)
        attention_mask : LongTensor(B, seq_len)
        pixel_values   : FloatTensor(B, 3, H, W)

    Forward output:
        Tensor(B, 4) — [xc, yc, w, h] in (0, 1)
    """

    def __init__(
        self,
        use_lora: bool = True,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_lora = use_lora
        self.hidden_dim = hidden_dim

        # ── Load LLaVA backbone ───────────────────────────────────────
        print(f"[SpatialLLaVA] Loading {LLAVA_MODEL_ID} ...")
        self.backbone = LlavaForConditionalGeneration.from_pretrained(
            LLAVA_MODEL_ID,
            torch_dtype=torch.float32,
            cache_dir=str(PATHS.weights),
            low_cpu_mem_usage=True,
        )
        self.backbone.gradient_checkpointing_enable()

        if use_lora:
            # ── LoRA mode: inject adapters, freeze everything else ────
            print("[SpatialLLaVA] Applying LoRA to language model ...")
            self.backbone.model.language_model = apply_lora(
                self.backbone.model.language_model
            )
            # Freeze vision tower and projector (only LM gets LoRA)
            for param in self.backbone.model.vision_tower.parameters():
                param.requires_grad = False
            for param in self.backbone.model.multi_modal_projector.parameters():
                param.requires_grad = False
        else:
            # ── Ablation mode: freeze entire backbone ─────────────────
            print("[SpatialLLaVA] Freezing entire backbone (ablation mode) ...")
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ── Regression head (always trainable) ───────────────────────
        self.head = RegressionHead(hidden_dim=hidden_dim, dropout=dropout)

        print(
            f"[SpatialLLaVA] Head params: {self.head.num_parameters():,}"
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        pixel_values: Tensor,
    ) -> Tensor:
        """
        Forward pass: image + text → bbox.

        What happens here:
            1. Run LLaVA forward (no generation) with output_hidden_states=True
            2. Extract hidden state at LAST token position ([LOC] token)
            3. Cast to float32 (backbone runs in float16)
            4. Pass through MLP regression head → [xc, yc, w, h]

        Args:
            input_ids      : (B, seq_len)
            attention_mask : (B, seq_len)
            pixel_values   : (B, 3, H, W)

        Returns:
            Tensor(B, 4) in (0, 1)
        """
        # Cast pixel_values to match backbone dtype (float16)
        pixel_values = pixel_values.to(
            dtype=next(self.backbone.parameters()).dtype
        )

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )

        # Last layer hidden states: (B, seq_len, hidden_dim)
        hidden_states = outputs.hidden_states[-1]

        # [LOC] is the last non-padding token
        # Use attention_mask to find last real token position
        seq_lens = attention_mask.sum(dim=1) - 1          # (B,)
        loc_hidden = hidden_states[
            torch.arange(hidden_states.size(0), device=hidden_states.device),
            seq_lens,
        ]                                                  # (B, hidden_dim)

        # Cast to float32 for head (head params are float32)
        loc_hidden = loc_hidden.float()

        return self.head(loc_hidden)                       # (B, 4)

    def trainable_parameters(self):
        """Return only trainable parameters (for optimizer)."""
        return [p for p in self.parameters() if p.requires_grad]


# ── Loader ────────────────────────────────────────────────────────────────────

def load_model(
    use_lora: bool = True,
    device: str = "cuda",
    dropout: float = 0.1,
):
    """
    Load SpatialLLaVA model + LlavaProcessor.

    Args:
        use_lora : True for main model, False for ablation
        device   : "cuda" or "cpu"
        dropout  : Head dropout

    Returns:
        (model, processor)
        model     : SpatialLLaVA on device
        processor : LlavaProcessor for tokenizing inputs
    """
    print(f"[load_model] use_lora={use_lora}  device={device}")

    processor = AutoProcessor.from_pretrained(
        LLAVA_MODEL_ID,
        cache_dir=str(PATHS.weights),
    )

    model = SpatialLLaVA(use_lora=use_lora, dropout=dropout)
    model = model.to(device)

    return model, processor
