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

from torch import Tensor
from typing import List, Dict


# ── DataLoader collation ───────────────────────────────────────────────────────

def collate_fn(samples: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """
    Custom collate function for RefCOCO DataLoader.

    Stacks individual sample dicts into a batched dict of tensors.
    Used when default collation is insufficient (e.g. variable length inputs).

    Args:
        samples : List of dicts, each from RefCOCODataset.__getitem__()
                  Each dict has keys: "image", "input_ids", "bbox"

    Returns:
        Dict with the same keys, but tensors have an extra batch dimension:
        {
            "image":     Tensor (B, 3, 384, 384)
            "input_ids": Tensor (B, 77)
            "bbox":      Tensor (B, 4)
        }

    Example:
        loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    """
    raise NotImplementedError(
        "Member 2: use torch.stack() on each key across samples. "
        "e.g. images = torch.stack([s['image'] for s in samples], dim=0)"
    )


# ── Statistics ─────────────────────────────────────────────────────────────────

def compute_dataset_stats(dataset) -> dict:
    """
    Compute and return summary statistics for a RefCOCODataset.

    Used in stage_1_data_preparation.py to generate dataset_stats.json.

    Args:
        dataset : A RefCOCODataset instance

    Returns:
        Dict with keys:
        {
            "num_samples":      int,
            "avg_bbox_width":   float,   # normalized
            "avg_bbox_height":  float,   # normalized
            "avg_bbox_area":    float,   # normalized
            "avg_text_length":  float,   # in tokens
        }
    """
    raise NotImplementedError(
        "Member 2: iterate over dataset, collect bbox widths/heights/areas "
        "and text lengths, compute means. "
        "Return as a plain Python dict (not tensors) so it can be json.dump()."
    )


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

    Example:
        assert validate_bbox(torch.tensor([0.5, 0.5, 0.3, 0.3]))   # True
        assert not validate_bbox(torch.tensor([0.5, 0.5, 0.0, 0.3]))  # False, w=0
    """
    raise NotImplementedError(
        "Member 2: check shape == (4,), all values in [0,1], "
        "bbox[2] > 0 (width), bbox[3] > 0 (height)."
    )


def validate_sample(sample: Dict[str, Tensor]) -> bool:
    """
    Check that a full dataset sample has the correct shapes and dtypes.

    Args:
        sample : Dict with keys "image", "input_ids", "bbox"

    Returns:
        True if all checks pass, False otherwise
    """
    raise NotImplementedError(
        "Member 2: check that: "
        "sample['image'].shape == (3, 384, 384), dtype float32; "
        "sample['input_ids'].shape == (77,), dtype long; "
        "sample['bbox'].shape == (4,) and validate_bbox() passes."
    )
