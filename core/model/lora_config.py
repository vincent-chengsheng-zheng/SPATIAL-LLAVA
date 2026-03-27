"""
core/model/lora_config.py

LoRA (Low-Rank Adaptation) configuration for fine-tuning LLaVA-7B.

Defines which layers to adapt and with what rank/alpha settings.
These settings are passed directly to the PEFT library.

Why LoRA:
    LLaVA-7B has 7 billion parameters. Full fine-tuning requires
    ~28GB VRAM (fp32) or ~14GB (fp16) just for weights, plus
    optimizer states. With LoRA rank=16, we only train ~5M parameters
    (~0.07% of total), fitting comfortably in V100 32GB.

Usage:
    from core.model.lora_config import get_lora_config, LORA_TARGET_MODULES

    peft_config = get_lora_config(rank=16)
"""

from dataclasses import dataclass
from peft import LoraConfig, TaskType


# ── Target modules ────────────────────────────────────────────────────────────

# Attention projection layers in Vicuna-7B (LLaVA's LLM backbone).
# We only adapt query and value projections — this is standard practice
# and sufficient for most fine-tuning tasks.
# Adding k_proj and o_proj increases trainable params but rarely helps much.
LORA_TARGET_MODULES = ["q_proj", "v_proj"]


# ── Default hyperparameters ───────────────────────────────────────────────────

@dataclass
class LoRAHyperParams:
    """
    LoRA hyperparameters with sensible defaults for LLaVA-7B on V100.

    Attributes:
        rank        : LoRA rank r. Controls capacity of the adapter.
                      Higher rank = more params = more expressive but slower.
                      Typical values: 8, 16, 32. We use 16.
        alpha       : LoRA scaling factor. Usually set to 2*rank.
                      Controls how strongly LoRA updates influence the model.
        dropout     : Dropout applied to LoRA layers. Helps generalization.
        bias        : Whether to train bias terms. "none" is standard.
        target_modules : Which attention layers to adapt.
    """
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    target_modules: list = None

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = LORA_TARGET_MODULES


# ── Factory function ──────────────────────────────────────────────────────────

def get_lora_config(
    rank: int = 16,
    alpha: int = None,
    dropout: float = 0.05,
    target_modules: list = None,
) -> LoraConfig:
    """
    Build and return a PEFT LoraConfig for LLaVA-7B.

    Args:
        rank           : LoRA rank (default: 16)
        alpha          : LoRA alpha scaling (default: 2 * rank)
        dropout        : LoRA dropout rate (default: 0.05)
        target_modules : Layers to adapt (default: ["q_proj", "v_proj"])

    Returns:
        peft.LoraConfig — pass directly to get_peft_model()

    Example:
        from peft import get_peft_model
        lora_config = get_lora_config(rank=16)
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()
        # trainable params: 4,718,592 || all params: 7,065,440,256 || trainable%: 0.0668
    """
    if alpha is None:
        alpha = 2 * rank

    if target_modules is None:
        target_modules = LORA_TARGET_MODULES

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=target_modules,
    )


def get_lora_config_from_yaml(config: dict) -> LoraConfig:
    """
    Build LoraConfig from a config dict loaded from YAML.

    Expects the YAML to have a "lora" section:
        lora:
            rank: 16
            alpha: 32
            dropout: 0.05

    Args:
        config : Full training config dict (from config_coursework.yaml)

    Returns:
        peft.LoraConfig

    Example:
        import yaml
        with open("config_coursework.yaml") as f:
            config = yaml.safe_load(f)
        lora_config = get_lora_config_from_yaml(config)
    """
    lora_cfg = config.get("lora", {})
    return get_lora_config(
        rank=lora_cfg.get("rank", 16),
        alpha=lora_cfg.get("alpha", None),
        dropout=lora_cfg.get("dropout", 0.05),
    )
