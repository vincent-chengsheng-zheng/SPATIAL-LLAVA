"""
core/utils/metrics.py

Evaluation metrics for bounding box regression.

All bounding boxes are in normalized format: [x_center, y_center, width, height]
where all values are in [0, 1].

Usage:
    from core.utils.metrics import iou, rmse, mae, batch_iou

    pred = torch.tensor([[0.5, 0.5, 0.3, 0.4]])
    target = torch.tensor([[0.5, 0.5, 0.3, 0.4]])

    print(iou(pred, target))    # tensor([1.0])
    print(rmse(pred, target))   # tensor(0.0)
"""

import torch
from torch import Tensor


# ── IoU ───────────────────────────────────────────────────────────────────────

def iou(pred: Tensor, target: Tensor) -> Tensor:
    """
    Compute Intersection over Union for a batch of bounding boxes.

    Args:
        pred   : Tensor of shape (N, 4) — predicted [x_c, y_c, w, h]
        target : Tensor of shape (N, 4) — ground truth [x_c, y_c, w, h]

    Returns:
        Tensor of shape (N,) — IoU score per sample, values in [0, 1]
    """
    pred = pred.float()
    target = target.float()

    # Convert [x_center, y_center, w, h] → [x1, y1, x2, y2]
    pred_x1 = pred[:, 0] - pred[:, 2] / 2
    pred_y1 = pred[:, 1] - pred[:, 3] / 2
    pred_x2 = pred[:, 0] + pred[:, 2] / 2
    pred_y2 = pred[:, 1] + pred[:, 3] / 2

    target_x1 = target[:, 0] - target[:, 2] / 2
    target_y1 = target[:, 1] - target[:, 3] / 2
    target_x2 = target[:, 0] + target[:, 2] / 2
    target_y2 = target[:, 1] + target[:, 3] / 2

    # Intersection
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Union
    pred_area = pred[:, 2] * pred[:, 3]
    target_area = target[:, 2] * target[:, 3]
    union_area = pred_area + target_area - inter_area

    # Avoid division by zero
    iou_score = inter_area / union_area.clamp(min=1e-6)

    return iou_score


def mean_iou(pred: Tensor, target: Tensor) -> Tensor:
    """
    Compute mean IoU over a batch.

    Args:
        pred   : Tensor of shape (N, 4)
        target : Tensor of shape (N, 4)

    Returns:
        Scalar tensor — mean IoU
    """
    return iou(pred, target).mean()


# ── RMSE ──────────────────────────────────────────────────────────────────────

def rmse(pred: Tensor, target: Tensor) -> Tensor:
    """
    Compute Root Mean Square Error over all coordinates.

    Args:
        pred   : Tensor of shape (N, 4)
        target : Tensor of shape (N, 4)

    Returns:
        Scalar tensor — RMSE value
    """
    pred = pred.float()
    target = target.float()
    return torch.sqrt(torch.mean((pred - target) ** 2))


# ── MAE ───────────────────────────────────────────────────────────────────────

def mae(pred: Tensor, target: Tensor) -> Tensor:
    """
    Compute Mean Absolute Error over all coordinates.

    Args:
        pred   : Tensor of shape (N, 4)
        target : Tensor of shape (N, 4)

    Returns:
        Scalar tensor — MAE value
    """
    pred = pred.float()
    target = target.float()
    return torch.mean(torch.abs(pred - target))


# ── Summary dict (used in eval scripts) ───────────────────────────────────────

def compute_all_metrics(pred: Tensor, target: Tensor) -> dict:
    """
    Compute all metrics at once and return as a dict.
    Convenient for logging and saving to JSON.

    Args:
        pred   : Tensor of shape (N, 4)
        target : Tensor of shape (N, 4)

    Returns:
        dict with keys: mean_iou, rmse, mae
        All values are Python floats (not tensors).

    Example:
        metrics = compute_all_metrics(pred, target)
        # {'mean_iou': 0.67, 'rmse': 0.048, 'mae': 0.032}
    """
    return {
        "mean_iou": mean_iou(pred, target).item(),
        "rmse": rmse(pred, target).item(),
        "mae": mae(pred, target).item(),
    }
