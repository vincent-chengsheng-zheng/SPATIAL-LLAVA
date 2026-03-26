"""
core/utils/checkpoint.py

Utilities for saving and loading training checkpoints.
Supports resuming training from the last saved state.

Usage:
    from core.utils.checkpoint import save_checkpoint, load_checkpoint

    # Save
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=2,
        step=1500,
        best_iou=0.63,
        path="~/SharedFolder/checkpoints/coursework/step_1500.pth"
    )

    # Load
    epoch, step, best_iou = load_checkpoint(
        model=model,
        optimizer=optimizer,
        path="~/SharedFolder/checkpoints/coursework/step_1500.pth"
    )
"""

import os
import torch
import torch.nn as nn
from torch.optim import Optimizer


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    step: int,
    best_iou: float,
    path: str,
) -> None:
    """
    Save training state to disk.

    Saves model weights, optimizer state, and training progress.
    Creates parent directories automatically if they don't exist.

    Args:
        model     : The model being trained
        optimizer : The optimizer being used
        epoch     : Current epoch number (0-indexed)
        step      : Current global step number
        best_iou  : Best validation IoU seen so far
        path      : Full path to save the checkpoint file (.pth)

    Example:
        save_checkpoint(model, optimizer, epoch=2, step=1500,
                       best_iou=0.63, path="checkpoints/step_1500.pth")
    """
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save({
        "epoch": epoch,
        "step": step,
        "best_iou": best_iou,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)

    print(f"  ✓ Checkpoint saved → {path}")
    print(f"    epoch={epoch}, step={step}, best_iou={best_iou:.4f}")


def load_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    path: str,
) -> tuple:
    """
    Load training state from disk and resume training.

    If the checkpoint file does not exist, returns (0, 0, 0.0)
    so training starts from scratch without raising an error.

    Args:
        model     : The model to load weights into
        optimizer : The optimizer to restore state into
        path      : Full path to the checkpoint file (.pth)

    Returns:
        Tuple of (epoch, step, best_iou) — all ints/floats, not tensors.
        Returns (0, 0, 0.0) if checkpoint does not exist.

    Example:
        epoch, step, best_iou = load_checkpoint(model, optimizer, path)
        print(f"Resuming from epoch {epoch}, step {step}")
    """
    path = os.path.expanduser(path)

    if not os.path.exists(path):
        print(f"  ℹ No checkpoint found at {path}, starting from scratch.")
        return 0, 0, 0.0

    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]
    step = checkpoint["step"]
    best_iou = checkpoint["best_iou"]

    print(f"  ✓ Checkpoint loaded ← {path}")
    print(f"    epoch={epoch}, step={step}, best_iou={best_iou:.4f}")

    return epoch, step, best_iou


def save_best(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    step: int,
    current_iou: float,
    best_iou: float,
    output_dir: str,
) -> float:
    """
    Save checkpoint only if current_iou improves over best_iou.
    Always saves to 'best.pth' in output_dir.

    Args:
        model       : The model being trained
        optimizer   : The optimizer being used
        epoch       : Current epoch number
        step        : Current global step
        current_iou : IoU from the latest validation run
        best_iou    : Best IoU seen so far
        output_dir  : Directory to save best.pth into

    Returns:
        Updated best_iou (either current_iou if improved, or unchanged best_iou)

    Example:
        best_iou = save_best(model, optimizer, epoch, step,
                             current_iou=0.65, best_iou=0.63,
                             output_dir="~/SharedFolder/checkpoints/coursework")
    """
    if current_iou > best_iou:
        best_path = os.path.join(os.path.expanduser(output_dir), "best.pth")
        save_checkpoint(model, optimizer, epoch, step, current_iou, best_path)
        print(f"  🏆 New best IoU: {best_iou:.4f} → {current_iou:.4f}")
        return current_iou

    return best_iou
