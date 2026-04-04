"""
pipeline/inspect_data.py

Inspect and verify the preprocessed RefCOCO dataset (v3 format).

What this script checks:
    1. Stats mode:
       a. Dataset size and split sizes
       b. Bbox validity (all in [0,1], positive w/h)
       c. Bbox size distribution (small/medium/large objects)
       d. Text length distribution (short/medium/long descriptions)
       e. Image file accessibility (can all paths be opened?)

    2. Samples mode:
       a. Load image from image_path
       b. Draw ground truth bbox on image (green box)
       c. Print text description alongside
       d. Save as PNG for visual verification
       → Key check: does the green box match what the text describes?

    3. All mode: runs both stats and samples

Usage:
    # Quick stats check
    python pipeline/inspect_data.py --mode stats --data_dir /tmp/data/

    # Visual alignment check (10 samples)
    python pipeline/inspect_data.py --mode samples --n 10 --data_dir /tmp/data/

    # Full inspection
    python pipeline/inspect_data.py --mode all --n 10 --data_dir /tmp/data/

Output:
    Stats printed to terminal.
    Sample images saved to /tmp/data_inspection/ (or --output_dir).
    Each image shows: original photo + green bbox + text label.
"""

import os
import sys
import pickle
import argparse
import random
import json

from PIL import Image, ImageDraw

# Add repo root to path
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_pkl(path: str) -> list:
    """Load a v3 pkl file and return list of {image_path, text, bbox}."""
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        print(f"  ✗ File not found: {path}")
        return []
    size_mb = os.path.getsize(path) / 1e6
    print(f"  Loading: {path} ({size_mb:.1f} MB)")
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"  Loaded {len(data):,} samples")
    return data


def draw_sample(
    image_path: str,
    bbox_norm,
    text: str,
    max_display_size: int = 600,
) -> Image.Image:
    """
    Load the original COCO image and draw the bbox + text label on it.

    What this does:
        1. Open the original COCO JPEG (full resolution)
        2. Resize to max_display_size for easy viewing
        3. Convert normalized [xc, yc, w, h] bbox → pixel [x1, y1, x2, y2]
        4. Draw a thick green rectangle for the bbox
        5. Add the text label below the image

    Args:
        image_path       : Absolute path to COCO .jpg file
        bbox_norm        : Tensor or list [xc, yc, w, h] in [0, 1]
        text             : Raw referring expression string
        max_display_size : Resize image to this max dimension

    Returns:
        PIL Image with bbox drawn and text label added
    """
    # ── Load original image ────────────────────────────────────────
    pil_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = pil_img.size

    # Resize for display (keep aspect ratio)
    scale = min(max_display_size / orig_w, max_display_size / orig_h)
    disp_w = int(orig_w * scale)
    disp_h = int(orig_h * scale)
    pil_img = pil_img.resize((disp_w, disp_h), Image.BILINEAR)

    # ── Draw bbox ──────────────────────────────────────────────────
    draw = ImageDraw.Draw(pil_img)
    if hasattr(bbox_norm, "tolist"):
        xc, yc, bw, bh = bbox_norm.tolist()
    else:
        xc, yc, bw, bh = bbox_norm

    x1 = int((xc - bw / 2) * disp_w)
    y1 = int((yc - bh / 2) * disp_h)
    x2 = int((xc + bw / 2) * disp_w)
    y2 = int((yc + bh / 2) * disp_h)

    x1, x2 = max(0, x1), min(disp_w - 1, x2)
    y1, y2 = max(0, y1), min(disp_h - 1, y2)

    # Draw thick green box
    for thickness in range(3):
        draw.rectangle(
            [x1 - thickness, y1 - thickness,
             x2 + thickness, y2 + thickness],
            outline="lime",
        )

    # ── Add text label below image ────────────────────────────────
    label_h = 40
    canvas = Image.new("RGB", (disp_w, disp_h + label_h), (30, 30, 30))
    canvas.paste(pil_img, (0, 0))
    draw2 = ImageDraw.Draw(canvas)

    # Truncate long text for display
    display_text = text if len(text) <= 60 else text[:57] + "..."
    draw2.text((5, disp_h + 5), display_text, fill="lime")

    return canvas


# ── Statistics ────────────────────────────────────────────────────────────────

def print_stats(data_dir: str) -> None:
    """
    Print dataset statistics for all splits.

    Checks:
        1. Format version (must be v3_image_path)
        2. Per split: sample count, bbox validity, text length, size dist
        3. Image accessibility: sample 100 random paths and try to open them
    """
    data_dir = os.path.expanduser(data_dir)

    # ── Load stats JSON ────────────────────────────────────────────
    stats_path = os.path.join(data_dir, "dataset_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        print("\n── Dataset Overview ──────────────────────────────────────")
        print(f"  Format version : {stats.get('format_version', 'unknown')}")
        print(f"  Total samples  : {stats.get('total_samples', 'N/A'):,}")
        print(f"  Split sizes    : {stats.get('split_sizes', {})}")
        print(f"  Dataset source : {stats.get('refcoco_dataset', 'N/A')}")
        if stats.get("max_samples_cap"):
            print(f"  Max samples cap: {stats['max_samples_cap']:,}")

        fmt = stats.get("format_version", "")
        if fmt != "v3_image_path":
            print(f"\n  ⚠ Unexpected format: {fmt}")
            print("  Expected: v3_image_path")
            print("  Run: bash shared/scripts/download_data.sh --force")
            return
    else:
        print(f"  ⚠ dataset_stats.json not found in {data_dir}")

    # ── Per-split stats ────────────────────────────────────────────
    for split in ["train", "val", "test"]:
        pkl_path = os.path.join(data_dir, f"refcoco_{split}.pkl")
        data = load_pkl(pkl_path)
        if not data:
            continue

        print(f"\n── {split.upper()} split ({len(data):,} samples) ─────────")

        # Check format: must have image_path, text, bbox
        sample0 = data[0]
        has_image_path = "image_path" in sample0
        has_text = "text" in sample0
        has_bbox = "bbox" in sample0
        has_old_tensor = "image" in sample0
        has_old_ids = "input_ids" in sample0

        print("  Fields present :")
        print(f"    image_path : {'✅' if has_image_path else '❌ MISSING'}")
        print(f"    text       : {'✅' if has_text else '❌ MISSING'}")
        print(f"    bbox       : {'✅' if has_bbox else '❌ MISSING'}")
        if has_old_tensor or has_old_ids:
            print(
                "  ⚠ Old format fields detected (image/input_ids). "
                "Run --force to regenerate."
            )

        if not (has_image_path and has_text and has_bbox):
            print("  ✗ Skipping stats — wrong format")
            continue

        # BBox stats
        bboxes = [s["bbox"] for s in data]
        widths = [
            b[2].item() if hasattr(b[2], "item") else b[2]
            for b in bboxes
        ]
        heights = [
            b[3].item() if hasattr(b[3], "item") else b[3]
            for b in bboxes
        ]
        areas = [w * h for w, h in zip(widths, heights)]

        # Validity: bbox in [0,1], positive w/h
        valid_bbox = sum(
            1 for b in bboxes
            if all(0 <= v <= 1 for v in (
                b[0].item() if hasattr(b[0], "item") else b[0],
                b[1].item() if hasattr(b[1], "item") else b[1],
                b[2].item() if hasattr(b[2], "item") else b[2],
                b[3].item() if hasattr(b[3], "item") else b[3],
            ))
            and (b[2].item() if hasattr(b[2], "item") else b[2]) > 0
            and (b[3].item() if hasattr(b[3], "item") else b[3]) > 0
        )
        print(
            f"  Bbox validity  : {valid_bbox:,}/{len(data):,} "
            f"({'✅ 100%' if valid_bbox == len(data) else '⚠ some invalid'})"
        )

        # Text stats
        texts = [s["text"] for s in data]
        text_lens = [len(t.split()) for t in texts]
        empty_texts = sum(1 for t in texts if not t.strip())
        print(
            f"  Text length    : min={min(text_lens)} words  "
            f"mean={sum(text_lens)/len(text_lens):.1f}  "
            f"max={max(text_lens)}"
        )
        if empty_texts > 0:
            print(f"  ⚠ Empty texts  : {empty_texts}")

        # BBox size distribution
        small = sum(1 for a in areas if a < 0.05)
        medium = sum(1 for a in areas if 0.05 <= a < 0.20)
        large = sum(1 for a in areas if a >= 0.20)
        print(
            f"  BBox width     : min={min(widths):.3f}  "
            f"mean={sum(widths)/len(widths):.3f}  "
            f"max={max(widths):.3f}"
        )
        print(
            f"  BBox height    : min={min(heights):.3f}  "
            f"mean={sum(heights)/len(heights):.3f}  "
            f"max={max(heights):.3f}"
        )
        print(
            f"  BBox area      : min={min(areas):.4f}  "
            f"mean={sum(areas)/len(areas):.4f}  "
            f"max={max(areas):.4f}"
        )
        print(
            f"  Size dist      : "
            f"small(<5%)={small:,}  "
            f"medium(5-20%)={medium:,}  "
            f"large(>20%)={large:,}"
        )

        # Image accessibility check (sample 50 random)
        print("  Image access   : checking 50 random paths...")
        sample_indices = random.sample(range(len(data)), min(50, len(data)))
        accessible = 0
        missing_examples = []
        for idx in sample_indices:
            img_path = data[idx]["image_path"]
            if os.path.exists(img_path):
                accessible += 1
            else:
                missing_examples.append(img_path)

        if accessible == len(sample_indices):
            print("  ✅ All 50 sampled images accessible")
        else:
            print(
                f"  ⚠ {len(sample_indices) - accessible} / "
                f"{len(sample_indices)} images NOT found"
            )
            for p in missing_examples[:3]:
                print(f"    missing: {p}")


# ── Sample visualization ──────────────────────────────────────────────────────

def save_samples(
    data_dir: str,
    output_dir: str,
    n: int = 10,
    split: str = "train",
    seed: int = 42,
) -> None:
    """
    Save N random sample images for visual alignment verification.

    For each sample, saves a PNG showing:
        - Original COCO image (resized to 600px)
        - Green bounding box drawn on image
        - Text description shown below the image

    How to verify alignment:
        → Open the saved images
        → Check: does the green box surround the object described in the text?
        → Example: text="the woman in blue" → box should be around that woman

    Filename format: {split}_sample_{i:03d}_idx{dataset_idx}.png
    """
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    pkl_path = os.path.join(
        os.path.expanduser(data_dir), f"refcoco_{split}.pkl"
    )
    data = load_pkl(pkl_path)
    if not data:
        return

    indices = random.sample(range(len(data)), min(n, len(data)))

    print(f"\n── Saving {len(indices)} samples from '{split}' split ────────")
    print("  Verify: green box should match what the text describes.\n")

    saved = 0
    skipped = 0
    for i, idx in enumerate(indices):
        sample = data[idx]
        img_path = sample["image_path"]
        text = sample["text"]
        bbox = sample["bbox"]

        # Skip if image file missing
        if not os.path.exists(img_path):
            print(f"  ⚠ Sample {i:03d}: image not found: {img_path}")
            skipped += 1
            continue

        # Draw sample
        try:
            result = draw_sample(img_path, bbox, text)
        except Exception as e:
            print(f"  ⚠ Sample {i:03d}: failed to draw: {e}")
            skipped += 1
            continue

        # Compute pixel bbox for logging
        orig = Image.open(img_path)
        W, H = orig.size
        if hasattr(bbox, "tolist"):
            xc, yc, bw, bh = bbox.tolist()
        else:
            xc, yc, bw, bh = bbox
        x1 = int((xc - bw / 2) * W)
        y1 = int((yc - bh / 2) * H)
        x2 = int((xc + bw / 2) * W)
        y2 = int((yc + bh / 2) * H)
        area_pct = bw * bh * 100

        print(f"  Sample {i:03d} (idx={idx}):")
        print(f"    text    : \"{text}\"")
        print(
            f"    bbox    : pixels=[{x1},{y1},{x2},{y2}]  "
            f"norm=[{xc:.3f},{yc:.3f},{bw:.3f},{bh:.3f}]"
        )
        print(f"    area    : {area_pct:.1f}% of image")
        print(f"    img size: {W}x{H}")

        fname = os.path.join(
            output_dir, f"{split}_sample_{i:03d}_idx{idx}.png"
        )
        result.save(fname)
        print(f"    saved → {fname}\n")
        saved += 1

    print(
        f"  ✅ {saved} images saved to {output_dir}"
        + (f" ({skipped} skipped)" if skipped else "")
    )
    print("\n  → Download and open the PNG files to verify alignment.")
    print("  → Green box should enclose the object described in the text.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    data_dir = os.path.expanduser(args.data_dir)
    output_dir = os.path.expanduser(
        args.output_dir or "/tmp/data_inspection/"
    )

    print("=" * 60)
    print("  Spatial-LLaVA — Data Inspection (v3 format)")
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
        description="Inspect preprocessed RefCOCO dataset (v3 format)"
    )
    parser.add_argument(
        "--data_dir", type=str, default="/tmp/data/",
        help="Directory containing refcoco_*.pkl files (default: /tmp/data/)"
    )
    parser.add_argument(
        "--mode", choices=["stats", "samples", "all"], default="all",
        help=(
            "stats: print statistics | "
            "samples: save visual examples | "
            "all: both (default)"
        )
    )
    parser.add_argument(
        "--n", type=int, default=10,
        help="Number of samples to visualize (default: 10)"
    )
    parser.add_argument(
        "--split", choices=["train", "val", "test"], default="train",
        help="Which split to visualize (default: train)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Where to save sample images (default: /tmp/data_inspection/)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sample selection (default: 42)"
    )
    args = parser.parse_args()
    main(args)
