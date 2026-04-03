"""
pipeline/inspect_data.py

Inspect and visualize samples from the preprocessed RefCOCO dataset.
Run this after stage_1_data_preparation.py to verify data quality.

Usage:
    # Show basic statistics
    python pipeline/inspect_data.py \
        --data_dir ~/SharedFolder/MDAIE/group6/data/ \
        --mode stats

    # Show N random samples (saves images)
    python pipeline/inspect_data.py \
        --data_dir ~/SharedFolder/MDAIE/group6/data/ \
        --mode samples \
        --n 10 \
        --output_dir ~/SharedFolder/MDAIE/group6/results/data_inspection/

    # Show both stats and samples
    python pipeline/inspect_data.py \
        --data_dir ~/SharedFolder/MDAIE/group6/data/ \
        --mode all \
        --n 10
"""

import os
import sys
import pickle
import argparse
import random
import json

import torch
import numpy as np
from PIL import Image

# Add repo root to path
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from core.utils.visualization import draw_bboxes


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_pkl(path: str) -> list:
    """Load pickle file and return list of samples."""
    path = os.path.expanduser(path)
    print(f"  Loading: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"  Loaded {len(data):,} samples")
    return data


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """
    Convert preprocessed image tensor back to PIL Image for display.
    Reverses ImageNet normalization.
    """
    img = image_tensor.numpy().transpose(1, 2, 0)
    # Reverse ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    img = (img.clip(0, 1) * 255).astype(np.uint8)
    return Image.fromarray(img)


def bbox_to_pixels(bbox: torch.Tensor, img_w: int = 384, img_h: int = 384):
    """Convert normalized [x_c, y_c, w, h] to pixel [x1, y1, x2, y2]."""
    x_c, y_c, w, h = bbox.tolist()
    x1 = int((x_c - w / 2) * img_w)
    y1 = int((y_c - h / 2) * img_h)
    x2 = int((x_c + w / 2) * img_w)
    y2 = int((y_c + h / 2) * img_h)
    return x1, y1, x2, y2


# ── Statistics ────────────────────────────────────────────────────────────────

def print_stats(data_dir: str) -> None:
    """Print dataset statistics for all splits."""
    data_dir = os.path.expanduser(data_dir)

    # Load stats JSON if exists
    stats_path = os.path.join(data_dir, "dataset_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        print("\n── Dataset Stats (from dataset_stats.json) ──────────────")
        print(f"  Total samples  : {stats.get('total_samples', 'N/A'):,}")
        print(f"  Split sizes    : {stats.get('split_sizes', {})}")
        print(f"  Image size     : {stats.get('image_size', 384)}x{stats.get('image_size', 384)}")
        print(f"  Max text length: {stats.get('max_length', 77)} tokens")
        print(f"  LOC token      : {stats.get('loc_token', '[LOC]')}")
        if stats.get('max_samples_cap'):
            pct = stats['max_samples_cap'] * 100 // 120000
            print(f"  Max samples cap: {stats['max_samples_cap']:,} (~{pct}% of full dataset)")

    # Per-split detailed stats
    for split in ["train", "val", "test"]:
        pkl_path = os.path.join(data_dir, f"refcoco_{split}.pkl")
        if not os.path.exists(pkl_path):
            print(f"\n  ⚠ {split}.pkl not found, skipping")
            continue

        data = load_pkl(pkl_path)
        if not data:
            continue

        # Collect bbox stats
        widths = [s["bbox"][2].item() for s in data]
        heights = [s["bbox"][3].item() for s in data]
        areas = [w * h for w, h in zip(widths, heights)]

        # Text length stats (non-zero tokens)
        text_lengths = [(s["input_ids"] != 0).sum().item() for s in data]

        # Validate samples
        valid = sum(
            1 for s in data
            if s["image"].shape == (3, 384, 384)
            and s["input_ids"].shape == (77,)
            and s["bbox"].shape == (4,)
            and (s["bbox"] >= 0).all()
            and (s["bbox"] <= 1).all()
            and s["bbox"][2].item() > 0
            and s["bbox"][3].item() > 0
        )

        print(f"\n── {split.upper()} split ({len(data):,} samples) ──────────────")
        print(f"  Valid samples  : {valid:,} / {len(data):,} "
              f"({valid * 100 // len(data)}%)")
        print(f"  Image shape    : {data[0]['image'].shape} "
              f"dtype={data[0]['image'].dtype}")
        print(f"  input_ids shape: {data[0]['input_ids'].shape} "
              f"dtype={data[0]['input_ids'].dtype}")
        print(f"  bbox shape     : {data[0]['bbox'].shape} "
              f"dtype={data[0]['bbox'].dtype}")
        print(f"\n  BBox width     : min={min(widths):.3f} "
              f"mean={sum(widths)/len(widths):.3f} "
              f"max={max(widths):.3f}")
        print(f"  BBox height    : min={min(heights):.3f} "
              f"mean={sum(heights)/len(heights):.3f} "
              f"max={max(heights):.3f}")
        print(f"  BBox area      : min={min(areas):.4f} "
              f"mean={sum(areas)/len(areas):.4f} "
              f"max={max(areas):.4f}")
        print(f"\n  Text length    : min={min(text_lengths)} "
              f"mean={sum(text_lengths)/len(text_lengths):.1f} "
              f"max={max(text_lengths)} tokens")

        # Size distribution
        small = sum(1 for a in areas if a < 0.05)
        medium = sum(1 for a in areas if 0.05 <= a < 0.20)
        large = sum(1 for a in areas if a >= 0.20)
        print(f"\n  BBox size dist : "
              f"small(<5%)={small:,} "
              f"medium(5-20%)={medium:,} "
              f"large(>20%)={large:,}")


# ── Sample visualization ──────────────────────────────────────────────────────

def save_samples(
    data_dir: str,
    output_dir: str,
    n: int = 10,
    split: str = "train",
    seed: int = 42,
) -> None:
    """
    Save N random sample images with bounding boxes drawn on them.

    Each saved image shows:
    - The preprocessed image (384x384)
    - Ground truth bbox drawn in green

    Args:
        data_dir   : Directory containing pkl files
        output_dir : Where to save inspection images
        n          : Number of samples to save
        split      : Which split to inspect ("train", "val", "test")
        seed       : Random seed for sample selection
    """
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    pkl_path = os.path.join(os.path.expanduser(data_dir),
                            f"refcoco_{split}.pkl")
    data = load_pkl(pkl_path)

    indices = random.sample(range(len(data)), min(n, len(data)))

    print(f"\n── Saving {len(indices)} samples from '{split}' split ──────")
    for i, idx in enumerate(indices):
        sample = data[idx]
        pil_img = tensor_to_pil(sample["image"])
        bbox = sample["bbox"]

        # Draw ground truth bbox
        result = draw_bboxes(
            pil_img,
            pred_boxes=[tuple(bbox.tolist())],
            target_boxes=[tuple(bbox.tolist())],
            labels=[f"GT: [{bbox[0]:.2f},{bbox[1]:.2f},{bbox[2]:.2f},{bbox[3]:.2f}]"],
            colors=("lime", "lime"),
        )

        # Print sample info
        x1, y1, x2, y2 = bbox_to_pixels(bbox)
        text_len = (sample["input_ids"] != 0).sum().item()
        area = bbox[2].item() * bbox[3].item()

        print(f"  Sample {i:03d} (idx={idx}):")
        print(f"    bbox (normalized) : [{bbox[0]:.3f}, {bbox[1]:.3f}, "
              f"{bbox[2]:.3f}, {bbox[3]:.3f}]")
        print(f"    bbox (pixels)     : [{x1}, {y1}, {x2}, {y2}]")
        print(f"    bbox area         : {area:.4f} ({area*100:.1f}% of image)")
        print(f"    text tokens       : {text_len} non-zero tokens")

        fname = os.path.join(output_dir, f"{split}_sample_{i:03d}_idx{idx}.png")
        result.save(fname)
        print(f"    saved → {fname}")

    print(f"\n  ✅ {len(indices)} images saved to {output_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    data_dir = os.path.expanduser(args.data_dir)
    output_dir = os.path.expanduser(
        args.output_dir or
        os.path.join(
            os.path.dirname(data_dir.rstrip("/")),
            "..", "results", "data_inspection"
        )
    )

    print("=" * 60)
    print("  Spatial-LLaVA — Data Inspection")
    print(f"  Data dir : {data_dir}")
    print(f"  Mode     : {args.mode}")
    print("=" * 60)

    if args.mode in ("stats", "all"):
        print_stats(data_dir)

    if args.mode in ("samples", "all"):
        save_samples(
            data_dir=data_dir,
            output_dir=output_dir,
            n=args.n,
            split=args.split,
            seed=args.seed,
        )

    print("\n✅ Inspection complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect preprocessed RefCOCO dataset"
    )
    parser.add_argument(
        "--data_dir", type=str,
        default="~/SharedFolder/MDAIE/group6/data/",
        help="Directory containing refcoco_*.pkl files"
    )
    parser.add_argument(
        "--mode", choices=["stats", "samples", "all"],
        default="all",
        help="stats: print statistics only | "
             "samples: save sample images | "
             "all: both"
    )
    parser.add_argument(
        "--n", type=int, default=10,
        help="Number of samples to visualize (default: 10)"
    )
    parser.add_argument(
        "--split", choices=["train", "val", "test"],
        default="train",
        help="Which split to inspect (default: train)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Where to save sample images"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sample selection"
    )
    args = parser.parse_args()
    main(args)