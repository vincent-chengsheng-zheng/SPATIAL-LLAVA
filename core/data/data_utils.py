"""
core/data/data_utils.py

Utility functions shared across the data pipeline.

Includes:
- Collate function for DataLoader batching
- Dataset statistics computation
- Validation helpers

Usage:
    from core.data.data_utils import collate_fn, compute_dataset_stats
"""

import torch
from torch import Tensor
from typing import Dict, List


# ── DataLoader collation ───────────────────────────────────────────────────────

def collate_fn(samples: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """
    Custom collate function for RefCOCO DataLoader.

    Stacks individual sample dicts into a batched dict of tensors.

    Args:
        samples : List of dicts, each from RefCOCODataset.__getitem__()
                  Each dict has keys: "image", "input_ids", "bbox"

    Returns:
        {
            "image":     Tensor (B, 3, 384, 384)
            "input_ids": Tensor (B, 77)
            "bbox":      Tensor (B, 4)
        }

    Example:
        loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    """
    return {
        key: torch.stack([s[key] for s in samples], dim=0)
        for key in samples[0].keys()
    }


# ── Statistics ─────────────────────────────────────────────────────────────────

def compute_dataset_stats(dataset) -> dict:
    """
    Compute summary statistics for a RefCOCODataset.

    Used in stage_1_data_preparation.py to generate dataset_stats.json.

    Args:
        dataset : A RefCOCODataset instance

    Returns:
        {
            "num_samples":     int,
            "avg_bbox_width":  float,
            "avg_bbox_height": float,
            "avg_bbox_area":   float,
            "avg_text_length": float,
        }
    """
    widths, heights, areas, text_lengths = [], [], [], []

    for i in range(len(dataset)):
        sample = dataset[i]
        bbox = sample["bbox"]
        input_ids = sample["input_ids"]

        w = bbox[2].item()
        h = bbox[3].item()
        widths.append(w)
        heights.append(h)
        areas.append(w * h)
        text_lengths.append((input_ids != 0).sum().item())

    n = len(dataset)
    return {
        "num_samples": n,
        "avg_bbox_width": float(sum(widths) / n),
        "avg_bbox_height": float(sum(heights) / n),
        "avg_bbox_area": float(sum(areas) / n),
        "avg_text_length": float(sum(text_lengths) / n),
    }


# ── Validation helpers ─────────────────────────────────────────────────────────

def validate_bbox(bbox: Tensor) -> bool:
    """
    Check that a bounding box tensor is valid.

    A valid bbox must:
    - Have shape (4,)
    - All values in [0, 1]
    - Width > 0 and height > 0

    Args:
        bbox : Tensor of shape (4,) — [x_center, y_center, w, h]

    Returns:
        True if valid, False otherwise
    """
    if bbox.shape != (4,):
        return False
    if not (bbox >= 0).all() or not (bbox <= 1).all():
        return False
    if bbox[2].item() <= 0 or bbox[3].item() <= 0:
        return False
    return True


def validate_sample(sample: Dict[str, Tensor]) -> bool:
    """
    Check that a full dataset sample has correct shapes and dtypes.

    Args:
        sample : Dict with keys "image", "input_ids", "bbox"

    Returns:
        True if all checks pass, False otherwise
    """
    required_keys = {"image", "input_ids", "bbox"}
    if not required_keys.issubset(sample.keys()):
        return False
    if sample["image"].shape != (3, 384, 384):
        return False
    if sample["image"].dtype != torch.float32:
        return False
    if sample["input_ids"].shape != (77,):
        return False
    if sample["input_ids"].dtype != torch.long:
        return False
    if not validate_bbox(sample["bbox"]):
        return False
    return True
