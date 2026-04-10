"""
core/loss/spatial_loss.py

Loss function for bounding box regression.

What this file provides:
    SpatialLoss — weighted combination of L1 + GIoU loss

Design reference:
    u-LLaVA (Xu et al., ECAI 2024): Lregion = β1*L1 + β2*GIoU, β1=β2=1.0
    GIoU (Rezatofighi et al., CVPR 2019): generalized IoU loss

Why L1 + GIoU (upgraded from SmoothL1 + IoU):
    L1       : coordinate-level regression, simple and effective
    GIoU loss: shape-aware, has gradient even when boxes don't overlap
               IoU loss has zero gradient when IoU=0 (non-overlapping boxes)
               GIoU = IoU - (C - U) / C, where C = smallest enclosing box area
               GIoU ∈ [-1, 1], always provides gradient signal

Combined loss:
    loss = w_l1 * L1(pred, target) + w_giou * (1 - mean_GIoU)

Usage:
    from core.loss.spatial_loss import SpatialLoss

    criterion = SpatialLoss(w_l1=1.0, w_giou=1.0)
    loss = criterion(pred, target)   # scalar tensor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SpatialLoss(nn.Module):
    """
    Weighted L1 + GIoU loss for normalized bbox regression.

    Follows u-LLaVA (Xu et al., ECAI 2024):
        Lregion = β1 * L1 + β2 * GIoU_loss, β1=β2=1.0

    Args:
        w_l1   : Weight for L1 term (default 1.0)
        w_giou : Weight for GIoU loss term (default 1.0)

    Input:
        pred   : Tensor(B, 4) — predicted [xc, yc, w, h] in (0, 1)
        target : Tensor(B, 4) — ground truth [xc, yc, w, h] in (0, 1)

    Output:
        Scalar tensor — combined loss value
    """

    def __init__(
        self,
        w_l1: float = 1.0,
        w_giou: float = 1.0,
    ):
        super().__init__()
        self.w_l1 = w_l1
        self.w_giou = w_giou

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute combined L1 + GIoU loss.

        Args:
            pred   : Tensor(B, 4)
            target : Tensor(B, 4)

        Returns:
            Scalar tensor
        """
        pred = pred.float()
        target = target.float()

        # ── L1 loss ───────────────────────────────────────────────────
        l1_loss = F.l1_loss(pred, target)

        # ── GIoU loss ─────────────────────────────────────────────────
        giou_val = _batch_giou(pred, target)        # (B,) in [-1, 1]
        giou_loss = (1.0 - giou_val).mean()         # in [0, 2]

        return self.w_l1 * l1_loss + self.w_giou * giou_loss

    def __repr__(self):
        return (
            f"SpatialLoss(w_l1={self.w_l1}, w_giou={self.w_giou})"
        )


# ── GIoU implementation ───────────────────────────────────────────────────────

def _batch_giou(pred: Tensor, target: Tensor) -> Tensor:
    """
    Compute per-sample GIoU for a batch. [xc, yc, w, h] format.

    GIoU = IoU - (C - U) / C
    where C = area of smallest enclosing box of pred and target
          U = union area

    Reference: Rezatofighi et al., CVPR 2019 (Algorithm 2)

    Args:
        pred   : Tensor(B, 4) [xc, yc, w, h]
        target : Tensor(B, 4) [xc, yc, w, h]

    Returns:
        Tensor(B,) GIoU values in [-1, 1]
    """
    # Convert [xc, yc, w, h] → [x1, y1, x2, y2]
    pred_x1 = pred[:, 0] - pred[:, 2] / 2
    pred_y1 = pred[:, 1] - pred[:, 3] / 2
    pred_x2 = pred[:, 0] + pred[:, 2] / 2
    pred_y2 = pred[:, 1] + pred[:, 3] / 2

    tgt_x1 = target[:, 0] - target[:, 2] / 2
    tgt_y1 = target[:, 1] - target[:, 3] / 2
    tgt_x2 = target[:, 0] + target[:, 2] / 2
    tgt_y2 = target[:, 1] + target[:, 3] / 2

    # Areas
    pred_area = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
    tgt_area  = (tgt_x2  - tgt_x1).clamp(min=0)  * (tgt_y2  - tgt_y1).clamp(min=0)

    # Intersection
    inter_x1 = torch.max(pred_x1, tgt_x1)
    inter_y1 = torch.max(pred_y1, tgt_y1)
    inter_x2 = torch.min(pred_x2, tgt_x2)
    inter_y2 = torch.min(pred_y2, tgt_y2)
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # Union
    union_area = pred_area + tgt_area - inter_area

    # IoU
    iou = inter_area / union_area.clamp(min=1e-6)

    # Smallest enclosing box C
    enc_x1 = torch.min(pred_x1, tgt_x1)
    enc_y1 = torch.min(pred_y1, tgt_y1)
    enc_x2 = torch.max(pred_x2, tgt_x2)
    enc_y2 = torch.max(pred_y2, tgt_y2)
    enc_area = (enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0)

    # GIoU = IoU - (C - U) / C
    giou = iou - (enc_area - union_area) / enc_area.clamp(min=1e-6)

    return giou


# ── Keep old IoU helper for backward compatibility ────────────────────────────

def _batch_iou(pred: Tensor, target: Tensor) -> Tensor:
    """Legacy IoU helper (kept for compatibility)."""
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
