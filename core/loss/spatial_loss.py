"""
core/loss/spatial_loss.py

Spatial regression loss for normalized bounding boxes.
Combines IoU loss + L1 regression for stable bbox training.

All boxes are in [x_center, y_center, width, height] normalized to [0, 1].

Usage:
    from core.loss.spatial_loss import spatial_loss

    loss = spatial_loss(pred_bbox, target_bbox)  # scalar
"""

import torch
from torch import Tensor


def _xywh_to_xyxy(boxes: Tensor) -> Tensor:
    """Convert [x_c, y_c, w, h] to [x1, y1, x2, y2]."""
    x_c, y_c, w, h = boxes.unbind(dim=-1)
    x1 = x_c - w / 2
    y1 = y_c - h / 2
    x2 = x_c + w / 2
    y2 = y_c + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def iou(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    """
    Batch IoU for normalized center+size boxes.

    Args:
        pred   : Tensor (B, 4) — predicted [x_c, y_c, w, h]
        target : Tensor (B, 4) — ground truth [x_c, y_c, w, h]
        eps    : Small value to avoid division by zero

    Returns:
        Tensor (B,) — IoU score per sample in [0, 1]
    """
    pred_xyxy = _xywh_to_xyxy(pred)
    target_xyxy = _xywh_to_xyxy(target)

    x1 = torch.max(pred_xyxy[..., 0], target_xyxy[..., 0])
    y1 = torch.max(pred_xyxy[..., 1], target_xyxy[..., 1])
    x2 = torch.min(pred_xyxy[..., 2], target_xyxy[..., 2])
    y2 = torch.min(pred_xyxy[..., 3], target_xyxy[..., 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter_area = inter_w * inter_h

    pred_area = (
        (pred_xyxy[..., 2] - pred_xyxy[..., 0]).clamp(min=0)
        * (pred_xyxy[..., 3] - pred_xyxy[..., 1]).clamp(min=0)
    )
    target_area = (
        (target_xyxy[..., 2] - target_xyxy[..., 0]).clamp(min=0)
        * (target_xyxy[..., 3] - target_xyxy[..., 1]).clamp(min=0)
    )
    union = pred_area + target_area - inter_area + eps

    return inter_area / union


def spatial_loss(
    pred: Tensor,
    target: Tensor,
    iou_weight: float = 1.0,
    l1_weight: float = 1.0,
    eps: float = 1e-7,
) -> Tensor:
    """
    Combined IoU + L1 loss for normalized bounding boxes.

    Args:
        pred       : Tensor (B, 4) — predicted [x_c, y_c, w, h]
        target     : Tensor (B, 4) — ground truth [x_c, y_c, w, h]
        iou_weight : Weight for IoU loss term (default: 1.0)
        l1_weight  : Weight for L1 loss term (default: 1.0)
        eps        : Small value to avoid division by zero

    Returns:
        Scalar tensor — mean loss over batch
    """
    assert pred.ndim == 2 and pred.size(1) == 4, "box tensors must be (B,4)"
    assert target.ndim == 2 and target.size(1) == 4, "box tensors must be (B,4)"
    assert pred.shape == target.shape, "pred and target must match shape"

    iou_score = iou(pred, target, eps=eps)
    iou_loss = 1.0 - iou_score

    l1 = torch.nn.functional.l1_loss(pred, target, reduction="none").mean(dim=1)

    loss = iou_weight * iou_loss + l1_weight * l1
    return loss.mean()


def smooth_l1_loss(pred: Tensor, target: Tensor, beta: float = 1.0) -> Tensor:
    """
    Smooth L1 loss for bounding box regression.

    Args:
        pred   : Tensor (B, 4)
        target : Tensor (B, 4)
        beta   : Threshold between L1 and L2 behavior (default: 1.0)

    Returns:
        Scalar tensor — mean smooth L1 loss
    """
    return torch.nn.functional.smooth_l1_loss(
        pred, target, reduction="mean", beta=beta
    )
