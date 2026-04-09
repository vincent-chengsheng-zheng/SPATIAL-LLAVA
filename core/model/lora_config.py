"""
core/model/lora_config.py

LoRA configuration and injection for the LLaVA language model backbone.

What this file does:
    1. Define LoRA hyperparameters (rank, alpha, target modules)
    2. Apply PEFT LoRA to the LLaVA language model
    3. Print trainable parameter summary

Why LoRA:
    - Full fine-tuning of LLaVA-7B requires ~28GB VRAM
    - LoRA (rank=16) adds ~8M trainable params on top of frozen backbone
    - With A100-80GB we can afford rank=16 with batch_size=8

Target modules (LLaVA-1.5 / Vicuna / LLaMA attention):
    q_proj, v_proj  — standard LoRA targets (query and value projections)
    k_proj          — optional, adds more capacity

Usage:
    from core.model.lora_config import apply_lora, LORA_CONFIG

    model = apply_lora(llava_language_model)
"""

from dataclasses import dataclass
from typing import List

from peft import LoraConfig, get_peft_model, TaskType


# ── Default config ────────────────────────────────────────────────────────────

@dataclass
class LoRAConfig:
    """
    LoRA hyperparameters.

    Attributes:
        r              : LoRA rank — controls capacity. 16 is a good default.
        lora_alpha     : Scaling factor. Typically 2*r.
        lora_dropout   : Dropout on LoRA layers.
        target_modules : Which attention projections to adapt.
        bias           : Whether to train bias terms ("none" = no).
    """
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    bias: str = "none"

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj"]


# Singleton — import this in step2 and step3
LORA_CONFIG = LoRAConfig()


# ── Apply LoRA ────────────────────────────────────────────────────────────────

def apply_lora(language_model, config: LoRAConfig = None) -> object:
    """
    Inject LoRA adapters into the LLaVA language model backbone.

    What this does:
        1. Build a PeftConfig targeting q/v/k projection layers
        2. Wrap the language model with get_peft_model()
        3. Freeze all non-LoRA parameters automatically
        4. Print trainable param summary

    Args:
        language_model : The LLM part of LlavaForConditionalGeneration
                         (model.language_model)
        config         : LoRAConfig instance (default: LORA_CONFIG)

    Returns:
        PEFT-wrapped language model with LoRA adapters injected
        All backbone params are frozen; only LoRA + head params are trainable.

    Note:
        After calling this, the regression head params must be added
        separately to the optimizer — they are NOT part of the PEFT model.
    """
    if config is None:
        config = LORA_CONFIG

    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias=config.bias,
        inference_mode=False,
    )

    lora_model = get_peft_model(language_model, peft_config)
    _print_trainable_summary(lora_model, label="LLaVA backbone (LoRA)")
    return lora_model


def print_param_summary(model, label: str = "Model"):
    """Print total vs trainable parameter counts."""
    _print_trainable_summary(model, label)


def _print_trainable_summary(model, label: str):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    print(
        f"[LoRA] {label}\n"
        f"  Total params     : {total:,}\n"
        f"  Trainable params : {trainable:,} ({100*trainable/total:.2f}%)\n"
        f"  Frozen params    : {frozen:,}"
    )

    