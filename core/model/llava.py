"""
core/model/llava.py

Standard LLaVA-1.5-7B wrapper for baseline inference.

What this file does:
    1. Load pretrained LLaVA-1.5-7B from HuggingFace
    2. Wrap generation logic (image + text prompt → raw text output)
    3. Parse bbox coordinates from raw text output using regex
    4. Report parse success/failure clearly

Why this is the baseline:
    Standard LLaVA outputs bounding boxes as text, e.g.:
        "[0.32, 0.45, 0.18, 0.35]"
    This is fragile — the format varies, parsing fails ~20-40% of the time.
    Our Spatial-LLaVA (spatial_llava.py) fixes this with a direct MLP head.

Used by:
    pipeline/step1_baseline_inference.py
"""

import re
import os
import torch
import torch.nn as nn
from typing import Optional, List

from core.data.preprocessing import PROMPT_TEMPLATE


class StandardLLaVA(nn.Module):
    """
    Standard LLaVA-1.5-7B for baseline bbox prediction via text generation.

    What happens at init:
        1. Download/load LLaVA-1.5-7B from HuggingFace (~14GB, cached)
        2. Load the image processor + tokenizer (AutoProcessor)
        3. All weights are frozen — no training, no fine-tuning

    What happens at generate():
        1. Format prompt: PROMPT_TEMPLATE.format(text=referring_expression)
        2. Process image + text with AutoProcessor
        3. Run LLaVA generation (text output)
        4. Return raw text string

    What happens at parse_bbox():
        1. Search raw text for [x, y, w, h] pattern with regex
        2. Validate all values in [0, 1]
        3. Return list of 4 floats, or None if parsing fails

    Args:
        model_id : HuggingFace model ID (default: llava-hf/llava-1.5-7b-hf)
        hf_cache : Cache directory for model weights
    """

    def __init__(
        self,
        model_id: str = "llava-hf/llava-1.5-7b-hf",
        hf_cache: str = "~/SharedFolder/MDAIE/group6/hf_cache/",
    ):
        super().__init__()

        self.model_id = model_id
        hf_cache = os.path.expanduser(hf_cache)
        os.makedirs(hf_cache, exist_ok=True)

        try:
            from transformers import (
                LlavaForConditionalGeneration,
                AutoProcessor,
            )
        except ImportError as e:
            raise ImportError(
                f"[StandardLLaVA] transformers not installed: {e}\n"
                "  Run: pip install transformers"
            )

        # Load processor (tokenizer + image processor)
        print(f"  [StandardLLaVA] Loading processor: {model_id} ...")
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                cache_dir=hf_cache,
            )
        except Exception as e:
            raise RuntimeError(
                f"[StandardLLaVA] Failed to load processor "
                f"from {model_id}: {e}"
            )

        # Load model weights (~14GB, cached after first download)
        print("  [StandardLLaVA] Loading model weights (~14GB)...")
        print(f"  [StandardLLaVA] Cache: {hf_cache}")
        try:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                cache_dir=hf_cache,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"[StandardLLaVA] Failed to load model "
                f"from {model_id}: {e}"
            )

        # Freeze all weights — baseline does no training
        for param in self.model.parameters():
            param.requires_grad = False

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"  [StandardLLaVA] ✓ Loaded ({n_params/1e9:.1f}B params, "
              f"all frozen)")

    def to(self, device):
        """Move model to device."""
        self.model = self.model.to(device)
        self._device = device
        return self

    def eval(self):
        """Set model to eval mode."""
        self.model.eval()
        return self

    def generate(
        self,
        pil_image,
        text: str,
        max_new_tokens: int = 50,
    ) -> str:
        """
        Run LLaVA inference on one image + text prompt.

        What this does:
            1. Format prompt with PROMPT_TEMPLATE
            2. Process image + text with AutoProcessor
            3. Generate text tokens (greedy decoding)
            4. Decode only the NEW tokens (not the input prompt)
            5. Return raw text string

        Args:
            pil_image      : PIL.Image.Image (RGB, any resolution)
            text           : Raw referring expression from RefCOCO
            max_new_tokens : Max tokens to generate (50 is enough for coords)

        Returns:
            str : Raw LLaVA text output, e.g. "[0.32, 0.45, 0.18, 0.35]"

        Raises:
            ValueError  : If pil_image or text is None/empty
            RuntimeError: If generation fails
        """
        if pil_image is None:
            raise ValueError("[StandardLLaVA.generate] pil_image is None")
        if not text or not text.strip():
            raise ValueError(
                "[StandardLLaVA.generate] text is empty or None"
            )

        device = next(self.model.parameters()).device
        prompt = PROMPT_TEMPLATE.format(text=text)

        try:
            inputs = self.processor(
                text=prompt,
                images=pil_image,
                return_tensors="pt",
            ).to(device)
        except Exception as e:
            raise RuntimeError(
                f"[StandardLLaVA.generate] Processor failed for "
                f"text='{text[:50]}': {e}"
            )

        try:
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
        except Exception as e:
            raise RuntimeError(
                f"[StandardLLaVA.generate] Generation failed: {e}"
            )

        # Decode only newly generated tokens (exclude prompt)
        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][input_len:]

        try:
            raw_output = self.processor.decode(
                new_tokens, skip_special_tokens=True
            )
        except Exception as e:
            raise RuntimeError(
                f"[StandardLLaVA.generate] Decode failed: {e}"
            )

        return raw_output.strip()

    @staticmethod
    def parse_bbox(text: str) -> Optional[List[float]]:
        """
        Parse [xc, yc, w, h] bbox coordinates from LLaVA text output.

        What this does:
            1. Search for pattern: [num, num, num, num]
            2. Also handles formats without brackets (fallback)
            3. Validates all values are in [0, 1]
            4. If values > 1, tries dividing by 640 (pixel coords)
            5. Returns None if parsing fails (IoU will be 0)

        Args:
            text : Raw text output from LLaVA generate()

        Returns:
            List [xc, yc, w, h] in [0, 1] if successful, else None

        Examples:
            parse_bbox("[0.32, 0.45, 0.18, 0.35]") → [0.32, 0.45, 0.18, 0.35]
            parse_bbox("The box is 0.3, 0.4, 0.2, 0.1") → [0.3, 0.4, 0.2, 0.1]
            parse_bbox("I cannot determine...") → None
        """
        if not text:
            return None

        # Pattern 1: [x, y, w, h] with optional brackets
        pattern = (
            r'\[?\s*'
            r'(\d+\.?\d*)\s*,\s*'
            r'(\d+\.?\d*)\s*,\s*'
            r'(\d+\.?\d*)\s*,\s*'
            r'(\d+\.?\d*)\s*'
            r'\]?'
        )
        match = re.search(pattern, text)

        if not match:
            return None

        try:
            values = [float(match.group(i)) for i in range(1, 5)]
        except (ValueError, AttributeError):
            return None

        # All values in [0, 1] → valid normalized coords
        if all(0.0 <= v <= 1.0 for v in values):
            return values

        # Values in pixel range → try normalizing by 640
        if all(0 <= v <= 640 for v in values):
            norm = [v / 640.0 for v in values]
            if all(0.0 <= v <= 1.0 for v in norm):
                return norm

        # Values out of range → cannot parse
        return None

    def count_parameters(self) -> dict:
        """Return parameter count summary."""
        total = sum(p.numel() for p in self.model.parameters())
        return {
            "total": total,
            "trainable": 0,           # baseline: nothing is trainable
            "frozen": total,
            "total_billions": round(total / 1e9, 2),
        }
