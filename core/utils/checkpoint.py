"""
core/utils/checkpoint.py

Checkpoint save / load utilities for SpatialLLaVA training.

What this file provides:
    save_checkpoint()  — save model + optimizer + epoch + metrics
    load_checkpoint()  — restore from checkpoint (resume or eval)
    CheckpointManager  — tracks best val IoU, saves best + latest

Checkpoint format (one .pth file):
    {
        "epoch"        : int
        "model_state"  : state_dict (LoRA + head weights only)
        "optim_state"  : optimizer state_dict
        "scheduler_state": scheduler state_dict
        "metrics"      : dict  (val_iou, val_loss, etc.)
        "config"       : dict  (hyperparams for reproducibility)
    }

Why save LoRA + head only (not full backbone):
    - Full LLaVA-7B weights = ~14GB per checkpoint
    - LoRA adapters + head = ~50MB
    - We reload the frozen backbone from HuggingFace at inference time

Usage:
    from core.utils.checkpoint import CheckpointManager

    ckpt_mgr = CheckpointManager(ckpt_dir=PATHS.ckpt_main)
    ckpt_mgr.save(model, optimizer, scheduler, epoch=1, metrics={"val_iou": 0.45})
    ckpt_mgr.load_best(model, optimizer, scheduler)
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional

import torch


# ── Low-level save / load ─────────────────────────────────────────────────────

def save_checkpoint(
    path: str,
    model,
    optimizer,
    scheduler,
    epoch: int,
    metrics: Dict,
    config: Dict = None,
) -> None:
    """
    Save a training checkpoint.

    What is saved:
        - Trainable model params only (LoRA adapters + regression head)
        - Optimizer and scheduler states
        - Epoch number and metrics dict
        - Config dict for reproducibility

    Args:
        path      : Full file path to save (e.g. checkpoints/main/best.pth)
        model     : SpatialLLaVA instance
        optimizer : AdamW optimizer
        scheduler : LR scheduler
        epoch     : Current epoch (0-indexed)
        metrics   : Dict of metric values (val_iou, val_loss, ...)
        config    : Hyperparams dict (optional, for reproducibility)
    """
    # Save only trainable params to keep checkpoint small (~50MB vs ~14GB)
    trainable_state = {
        k: v for k, v in model.state_dict().items()
        if any(k.startswith(prefix) for prefix in ["head.", "backbone.language_model.base_model"])
        or "lora" in k.lower()
    }

    ckpt = {
        "epoch": epoch,
        "model_state": trainable_state,
        "optim_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
        "config": config or {},
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  [ckpt] Saved → {path}  ({size_mb:.1f} MB)")


def load_checkpoint(
    path: str,
    model,
    optimizer=None,
    scheduler=None,
    device: str = "cuda",
) -> Dict:
    """
    Load a checkpoint and restore model (+ optionally optimizer/scheduler).

    Args:
        path      : Path to .pth checkpoint file
        model     : SpatialLLaVA instance (must be initialized first)
        optimizer : Optional — if provided, restores optimizer state
        scheduler : Optional — if provided, restores scheduler state
        device    : Device to map tensors to

    Returns:
        Dict with keys: epoch, metrics, config
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[load_checkpoint] Not found: {path}")

    ckpt = torch.load(path, map_location=device)

    missing, unexpected = model.load_state_dict(
        ckpt["model_state"], strict=False
    )
    if missing:
        print(f"  [ckpt] Missing keys ({len(missing)}): {missing[:3]} ...")
    if unexpected:
        print(f"  [ckpt] Unexpected keys ({len(unexpected)}): {unexpected[:3]} ...")

    if optimizer and ckpt.get("optim_state"):
        optimizer.load_state_dict(ckpt["optim_state"])

    if scheduler and ckpt.get("scheduler_state"):
        scheduler.load_state_dict(ckpt["scheduler_state"])

    epoch = ckpt.get("epoch", 0)
    metrics = ckpt.get("metrics", {})
    print(
        f"  [ckpt] Loaded epoch={epoch}  "
        f"val_iou={metrics.get('val_iou', 'N/A')}"
    )
    return {"epoch": epoch, "metrics": metrics, "config": ckpt.get("config", {})}


# ── CheckpointManager ─────────────────────────────────────────────────────────

class CheckpointManager:
    """
    Manages best + latest checkpoints during training.

    Saves two files:
        <ckpt_dir>/best.pth    — best val IoU so far
        <ckpt_dir>/latest.pth  — most recent epoch

    Also saves a history JSON for plotting training curves:
        <ckpt_dir>/history.json

    Args:
        ckpt_dir : Path to checkpoint directory (e.g. PATHS.ckpt_main)
        config   : Hyperparams dict (saved inside each checkpoint)
    """

    def __init__(self, ckpt_dir, config: Dict = None):
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        self.best_iou: float = -1.0
        self.history = []

    @property
    def best_path(self) -> str:
        return str(self.ckpt_dir / "best.pth")

    @property
    def latest_path(self) -> str:
        return str(self.ckpt_dir / "latest.pth")

    def save(
        self,
        model,
        optimizer,
        scheduler,
        epoch: int,
        metrics: Dict,
    ) -> bool:
        """
        Save latest checkpoint. If val_iou improved, also save best.

        Args:
            model, optimizer, scheduler : as in save_checkpoint()
            epoch   : current epoch
            metrics : must contain "val_iou" key

        Returns:
            True if this is a new best checkpoint
        """
        val_iou = metrics.get("val_iou", 0.0)

        # Always save latest
        save_checkpoint(
            self.latest_path, model, optimizer, scheduler,
            epoch, metrics, self.config
        )

        # Save best if improved
        is_best = val_iou > self.best_iou
        if is_best:
            self.best_iou = val_iou
            save_checkpoint(
                self.best_path, model, optimizer, scheduler,
                epoch, metrics, self.config
            )
            print(f"  [ckpt] ★ New best IoU: {val_iou:.4f}")

        # Append to history
        self.history.append({"epoch": epoch, **metrics})
        self._save_history()

        return is_best

    def load_best(
        self,
        model,
        optimizer=None,
        scheduler=None,
        device: str = "cuda",
    ) -> Optional[Dict]:
        """Load best.pth if it exists."""
        if not os.path.exists(self.best_path):
            print(f"  [ckpt] No best checkpoint found at {self.best_path}")
            return None
        return load_checkpoint(
            self.best_path, model, optimizer, scheduler, device
        )

    def load_latest(
        self,
        model,
        optimizer=None,
        scheduler=None,
        device: str = "cuda",
    ) -> Optional[Dict]:
        """Load latest.pth for resuming training."""
        if not os.path.exists(self.latest_path):
            print(f"  [ckpt] No latest checkpoint found at {self.latest_path}")
            return None
        return load_checkpoint(
            self.latest_path, model, optimizer, scheduler, device
        )

    def _save_history(self):
        history_path = self.ckpt_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
            