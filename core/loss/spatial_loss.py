"""
core/loss/spatial_loss.py

Loss function for bounding box regression.

What this file provides:
    SpatialLoss — weighted combination of SmoothL1 + IoU loss

Why two losses:
    SmoothL1  : coordinate-level regression, robust to outliers
    IoU loss  : shape-aware, directly optimizes the evaluation metric
                (IoU loss = 1 - IoU, lower = better)

Combined loss:
    loss = w_smooth * SmoothL1(pred, target) + w_iou * (1 - mean_IoU)

Usage:
    from core.loss.spatial_loss import SpatialLoss

    criterion = SpatialLoss(w_smooth=1.0, w_iou=1.0)
    loss = criterion(pred, target)   # scalar tensor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SpatialLoss(nn.Module):
    """
    Weighted SmoothL1 + IoU loss for normalized bbox regression.

    Args:
        w_smooth : Weight for SmoothL1 term (default 1.0)
        w_iou    : Weight for IoU loss term (default 1.0)
        beta     : SmoothL1 transition point (default 1.0)

    Input:
        pred   : Tensor(B, 4) — predicted [xc, yc, w, h] in (0, 1)
        target : Tensor(B, 4) — ground truth [xc, yc, w, h] in (0, 1)

    Output:
        Scalar tensor — combined loss value
    """

    def __init__(
        self,
        w_smooth: float = 1.0,
        w_iou: float = 1.0,
        beta: float = 1.0,
    ):
        super().__init__()
        self.w_smooth = w_smooth
        self.w_iou = w_iou
        self.beta = beta

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute combined loss.

        Args:
            pred   : Tensor(B, 4)
            target : Tensor(B, 4)

        Returns:
            Scalar tensor
        """
        pred = pred.float()
        target = target.float()

        # ── SmoothL1 ──────────────────────────────────────────────────
        smooth_loss = F.smooth_l1_loss(pred, target, beta=self.beta)

        # ── IoU loss ──────────────────────────────────────────────────
        iou_val = _batch_iou(pred, target)         # (B,)
        iou_loss = (1.0 - iou_val).mean()

        return self.w_smooth * smooth_loss + self.w_iou * iou_loss

    def __repr__(self):
        return (
            f"SpatialLoss(w_smooth={self.w_smooth}, "
            f"w_iou={self.w_iou}, beta={self.beta})"
        )


# ── Internal IoU (avoids circular import with metrics.py) ─────────────────────

def _batch_iou(pred: Tensor, target: Tensor) -> Tensor:
    """
    Compute per-sample IoU for a batch. [xc, yc, w, h] format.
    Kept here to avoid circular import with core.utils.metrics.
    """
    pred_x1 = pred[:, 0] - pred[:, 2] / 2
    pred_y1 = pred[:, 1] - pred[:, 3] / 2
    pred_x2 = pred[:, 0] + pred[:, 2] / 2
    pred_y2 = pred[:, 1] + pred[:, 3] / 2

    tgt_x1 = target[:, 0] - target[:, 2] / 2
    tgt_y1 = target[:, 1] - target[:, 3] / 2
    tgt_x2 = target[:, 0] + target[:, 2] / 2
    tgt_y2 = target[:, 1] + target[:, 3] / 2

    inter_x1 = torch.max(pred_x1, tgt_x1)
    inter_y1 = torch.max(pred_y1, tgt_y1)
    inter_x2 = torch.min(pred_x2, tgt_x2)
    inter_y2 = torch.min(pred_y2, tgt_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    pred_area = pred[:, 2] * pred[:, 3]
    tgt_area = target[:, 2] * target[:, 3]
    union_area = pred_area + tgt_area - inter_area

    return inter_area / union_area.clamp(min=1e-6)
