"""
pipeline/stage_1_data_preparation.py

Stage 1: Download RefCOCO annotations + COCO images, then preprocess.

Two-step process:
    1. Download RefCOCO annotations from HuggingFace (jxu124/refcoco, ~50MB)
    2. Download COCO train2014 images (~13GB) if not already present
    3. Preprocess into pickle files

Usage:
    # Full pipeline
    python pipeline/stage_1_data_preparation.py \\
        --output_dir ~/SharedFolder/MDAIE/group6/data/ \\
        --coco_dir   ~/SharedFolder/MDAIE/group6/coco/ \\
        --max_samples 60000

    # Skip COCO download if already done
    python pipeline/stage_1_data_preparation.py \\
        --output_dir ~/SharedFolder/MDAIE/group6/data/ \\
        --coco_dir   ~/SharedFolder/MDAIE/group6/coco/ \\
        --skip_coco_download

Output:
    ~/SharedFolder/MDAIE/group6/data/refcoco_train.pkl
    ~/SharedFolder/MDAIE/group6/data/refcoco_val.pkl
    ~/SharedFolder/MDAIE/group6/data/refcoco_test.pkl
    ~/SharedFolder/MDAIE/group6/data/dataset_stats.json
"""

import os
import sys
import json
import pickle
import argparse
import hashlib
import random
import subprocess
from typing import List, Dict

import torch
from PIL import Image
from transformers import AutoTokenizer

# Add repo root to path so core/ is importable
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from core.data.preprocessing import (  # noqa: E402
    preprocess_image_from_pil,
    preprocess_text,
    normalize_bbox,
    LOC_TOKEN,
    MAX_LENGTH,
)


# ── Constants ─────────────────────────────────────────────────────────────────

LLAVA_MODEL_ID = "liuhaotian/llava-v1.5-7b"
REFCOCO_DATASET = "jxu124/refcoco"
COCO_IMAGES_URL = (
    "http://images.cocodataset.org/zips/train2014.zip"
)
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_md5(path: str, chunk_size: int = 1 << 20) -> str:
    """Compute MD5 checksum of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_output(output_dir: str) -> bool:
    """Check that all 3 pkl files and stats JSON exist and are non-empty."""
    for split in ["train", "val", "test"]:
        p = os.path.join(output_dir, f"refcoco_{split}.pkl")
        if not os.path.exists(p):
            print(f"  ✗ Missing: {p}")
            return False
        if os.path.getsize(p) < 100:
            print(f"  ✗ Empty (0 samples): {p}")
            return False
    stats = os.path.join(output_dir, "dataset_stats.json")
    if not os.path.exists(stats):
        print(f"  ✗ Missing: {stats}")
        return False
    # Check stats reports non-zero samples
    with open(stats) as f:
        s = json.load(f)
    if s.get("total_samples", 0) == 0:
        print("  ✗ dataset_stats.json reports 0 total samples")
        return False
    return True


def resolve_image_path(raw: dict, coco_dir: str) -> str:
    """
    Build the full path to the COCO image for a given RefCOCO sample.

    jxu124/refcoco stores image_path as:
        "coco/train2014/COCO_train2014_000000581857.jpg"

    We resolve this against coco_dir:
        coco_dir/train2014/COCO_train2014_000000581857.jpg
    """
    image_path = raw.get("image_path", "")
    # Strip leading "coco/" prefix if present
    parts = image_path.replace("\\", "/").split("/")
    if parts and parts[0].lower() == "coco":
        parts = parts[1:]
    return os.path.join(os.path.expanduser(coco_dir), *parts)


def setup_tokenizer(hf_home: str) -> AutoTokenizer:
    """Load LLaVA tokenizer and add [LOC] special token."""
    print(f"\n[Stage 1] Loading tokenizer from {LLAVA_MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        LLAVA_MODEL_ID,
        cache_dir=hf_home,
        use_fast=False,
    )
    if LOC_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [LOC_TOKEN]}
        )
        print(
            f"  ✓ Added '{LOC_TOKEN}' to vocabulary "
            f"(new vocab size: {len(tokenizer)})"
        )
    else:
        print(f"  ✓ '{LOC_TOKEN}' already in vocabulary")
    return tokenizer


# ── Step 1: Download annotations ──────────────────────────────────────────────

def download_refcoco(hf_home: str):
    """Download RefCOCO annotations from HuggingFace."""
    print("\n[Stage 1] Downloading RefCOCO annotations from HuggingFace ...")
    from datasets import load_dataset

    os.environ["HF_HOME"] = hf_home
    dataset = load_dataset(REFCOCO_DATASET, cache_dir=hf_home)
    print(f"  ✓ Downloaded. Splits: {list(dataset.keys())}")
    for split, data in dataset.items():
        print(f"    {split}: {len(data):,} samples")
    return dataset


# ── Step 2: Download COCO images ──────────────────────────────────────────────

def download_coco_images(coco_dir: str) -> bool:
    """
    Download COCO train2014 images if not already present.

    Returns True if images are available (either downloaded or pre-existing).
    """
    coco_dir = os.path.expanduser(coco_dir)
    train2014_dir = os.path.join(coco_dir, "train2014")
    zip_path = os.path.join(coco_dir, "train2014.zip")

    # Already extracted
    if os.path.isdir(train2014_dir):
        n_imgs = len([
            f for f in os.listdir(train2014_dir)
            if f.endswith(".jpg")
        ])
        if n_imgs > 80000:
            print(f"\n[Stage 1] COCO images already present: "
                  f"{n_imgs:,} images in {train2014_dir}")
            return True
        else:
            print(f"\n[Stage 1] ⚠ train2014/ exists but only {n_imgs:,} "
                  "images found (expected ~83k). Re-downloading.")

    os.makedirs(coco_dir, exist_ok=True)

    # Download zip if not present
    if not os.path.exists(zip_path):
        print("\n[Stage 1] Downloading COCO train2014 images (~13GB) ...")
        print(f"  URL : {COCO_IMAGES_URL}")
        print(f"  Dest: {zip_path}")
        print("  This will take 30-60 minutes depending on network speed.")
        result = subprocess.run(
            ["wget", "-c", COCO_IMAGES_URL, "-O", zip_path],
            check=False,
        )
        if result.returncode != 0:
            print("  ✗ wget failed. Try manually:")
            print(f"    wget {COCO_IMAGES_URL} -O {zip_path}")
            return False
    else:
        print(f"\n[Stage 1] COCO zip already downloaded: {zip_path}")

    # Extract
    print(f"\n[Stage 1] Extracting {zip_path} ...")
    result = subprocess.run(
        ["unzip", "-q", zip_path, "-d", coco_dir],
        check=False,
    )
    if result.returncode != 0:
        print("  ✗ unzip failed.")
        return False

    n_imgs = len([
        f for f in os.listdir(train2014_dir)
        if f.endswith(".jpg")
    ])
    print(f"  ✓ Extracted {n_imgs:,} images to {train2014_dir}")
    return True


# ── Step 3: Preprocess ────────────────────────────────────────────────────────

def preprocess_sample(raw: dict, tokenizer, coco_dir: str) -> Dict:
    """
    Convert one raw RefCOCO sample into model-ready tensors.

    jxu124/refcoco fields used:
        image_path : relative path to COCO image
        sentences  : list of dicts with "raw" key
        bbox       : [x1, y1, x2, y2] pixel coords (xyxy format)
        raw_image_info : JSON string with width/height
    """
    # ── Image ──────────────────────────────────────────────────────
    img_full_path = resolve_image_path(raw, coco_dir)
    if not os.path.exists(img_full_path):
        raise FileNotFoundError(
            f"Image not found: {img_full_path}. "
            "Run with --coco_dir pointing to the COCO images directory."
        )
    pil_img = Image.open(img_full_path).convert("RGB")
    img_tensor = preprocess_image_from_pil(pil_img)

    # ── Text ───────────────────────────────────────────────────────
    sentences = raw.get("sentences", [])
    if isinstance(sentences, list) and len(sentences) > 0:
        sent = random.choice(sentences)
        prompt = sent["raw"] if isinstance(sent, dict) else str(sent)
    else:
        captions = raw.get("captions", [])
        prompt = captions[0] if captions else "find the object"

    input_ids = preprocess_text(prompt, tokenizer)

    # ── BBox ───────────────────────────────────────────────────────
    # jxu124/refcoco bbox is [x1, y1, x2, y2] pixel coords (xyxy)
    raw_bbox = raw["bbox"]
    x1, y1, x2, y2 = raw_bbox

    # Get image dimensions
    img_info_str = raw.get("raw_image_info", "{}")
    try:
        img_info = json.loads(img_info_str)
        img_w = img_info.get("width", pil_img.width)
        img_h = img_info.get("height", pil_img.height)
    except (json.JSONDecodeError, TypeError):
        img_w, img_h = pil_img.width, pil_img.height

    bbox_norm = normalize_bbox(
        [x1, y1, x2, y2], img_w=img_w, img_h=img_h
    )

    return {
        "image": img_tensor,
        "input_ids": input_ids,
        "bbox": bbox_norm,
    }


def preprocess_split(
    raw_split,
    tokenizer,
    split_name: str,
    coco_dir: str,
    max_samples: int = None,
) -> List[Dict]:
    """
    Preprocess all (or up to max_samples) samples in one split.
    """
    total_available = len(raw_split)
    total = (
        min(total_available, max_samples)
        if max_samples else total_available
    )
    errors = 0
    file_not_found = 0
    samples = []

    print(
        f"\n[Stage 1] Preprocessing '{split_name}' split "
        f"({total}/{total_available} samples) ..."
    )

    for i, raw in enumerate(raw_split):
        if max_samples is not None and len(samples) >= max_samples:
            break
        try:
            sample = preprocess_sample(raw, tokenizer, coco_dir)
            samples.append(sample)
        except FileNotFoundError as e:
            file_not_found += 1
            errors += 1
            if file_not_found <= 3:
                print(f"  ⚠ Image not found at index {i}: {e}")
            if file_not_found == 3:
                print("  ⚠ Further file-not-found errors suppressed.")
            continue
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ⚠ Error at index {i}: {e}")
            continue

        processed = len(samples)
        if processed % 5000 == 0 or processed == total:
            print(
                f"  [{processed}/{total}] processed, "
                f"errors: {errors} (missing images: {file_not_found})"
            )

    if file_not_found > 0:
        print(
            f"\n  ⚠ {file_not_found} images not found. "
            "Ensure COCO train2014 is downloaded to --coco_dir."
        )

    print(f"  ✓ Done: {len(samples)} samples ({errors} skipped)")
    return samples


# ── Step 4: Split (fallback) ──────────────────────────────────────────────────

def make_splits(all_samples: List[Dict]) -> Dict[str, List[Dict]]:
    """Shuffle and split samples 80/10/10 (used if no official splits)."""
    random.seed(SEED)
    random.shuffle(all_samples)
    n = len(all_samples)
    n_tr = int(n * SPLITS["train"])
    n_val = int(n * SPLITS["val"])
    return {
        "train": all_samples[:n_tr],
        "val": all_samples[n_tr: n_tr + n_val],
        "test": all_samples[n_tr + n_val:],
    }


# ── Step 5: Save ──────────────────────────────────────────────────────────────

def save_split(samples: List[Dict], output_dir: str, split: str) -> str:
    """Save samples to a pickle file. Returns file path."""
    path = os.path.join(output_dir, f"refcoco_{split}.pkl")
    print(
        f"\n[Stage 1] Saving '{split}' → {path} "
        f"({len(samples):,} samples) ..."
    )
    with open(path, "wb") as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_gb = os.path.getsize(path) / 1e9
    md5 = compute_md5(path)
    print(f"  ✓ Saved: {size_gb:.2f} GB  |  MD5: {md5}")
    return path


def save_stats(
    splits: Dict[str, List],
    output_dir: str,
    checksums: Dict,
    args,
) -> None:
    """Save dataset_stats.json."""
    stats = {
        "total_samples": sum(len(v) for v in splits.values()),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "split_ratios": SPLITS,
        "max_samples_cap": args.max_samples,
        "image_size": 384,
        "max_length": MAX_LENGTH,
        "loc_token": LOC_TOKEN,
        "llava_model": LLAVA_MODEL_ID,
        "refcoco_dataset": REFCOCO_DATASET,
        "coco_dir": args.coco_dir,
        "seed": SEED,
        "checksums": checksums,
    }
    path = os.path.join(output_dir, "dataset_stats.json")
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n[Stage 1] Stats saved → {path}")
    print(json.dumps(stats, indent=2))


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    output_dir = os.path.expanduser(args.output_dir)
    coco_dir = os.path.expanduser(args.coco_dir)
    hf_home = os.path.expanduser(
        args.hf_home or os.environ.get(
            "HF_HOME",
            "~/SharedFolder/MDAIE/group6/hf_cache"
        )
    )

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(hf_home, exist_ok=True)
    os.makedirs(coco_dir, exist_ok=True)

    random.seed(SEED)
    torch.manual_seed(SEED)

    pct = (
        f"~{args.max_samples * 100 // 120000}%"
        if args.max_samples else "100%"
    )

    print("=" * 60)
    print("  Stage 1: Data Preparation")
    print(f"  Output dir  : {output_dir}")
    print(f"  COCO dir    : {coco_dir}")
    print(f"  HF cache    : {hf_home}")
    print(
        f"  Max samples : "
        f"{args.max_samples or 'all'} ({pct} of dataset)"
    )
    print("=" * 60)

    # Skip if already complete
    if not args.force and verify_output(output_dir):
        print("\n✅ All output files already exist. Use --force to rerun.")
        return

    # ── Step 1: Download annotations ──────────────────────────────
    raw_dataset = download_refcoco(hf_home)

    # ── Step 2: Download COCO images ──────────────────────────────
    if args.skip_coco_download:
        print("\n[Stage 1] Skipping COCO download (--skip_coco_download)")
        train2014_dir = os.path.join(coco_dir, "train2014")
        if not os.path.isdir(train2014_dir):
            print(
                f"  ✗ {train2014_dir} not found. "
                "Remove --skip_coco_download or create the directory."
            )
            sys.exit(1)
    else:
        ok = download_coco_images(coco_dir)
        if not ok:
            print(
                "\n❌ COCO image download failed. "
                "Fix the issue and re-run."
            )
            sys.exit(1)

    # ── Step 3: Tokenizer ─────────────────────────────────────────
    tokenizer = setup_tokenizer(hf_home)

    # ── Step 4: Preprocess ────────────────────────────────────────
    if set(raw_dataset.keys()) >= {"train", "validation", "test"}:
        print("\n[Stage 1] Using official train/validation/test splits")
        train_max = args.max_samples
        val_max = (
            int(args.max_samples * 0.125) if args.max_samples else None
        )
        test_max = (
            int(args.max_samples * 0.125) if args.max_samples else None
        )
        processed = {
            "train": preprocess_split(
                raw_dataset["train"], tokenizer,
                "train", coco_dir, train_max
            ),
            "val": preprocess_split(
                raw_dataset["validation"], tokenizer,
                "val", coco_dir, val_max
            ),
            "test": preprocess_split(
                raw_dataset["test"], tokenizer,
                "test", coco_dir, test_max
            ),
        }
    else:
        print("\n[Stage 1] No official splits — pooling and splitting 80/10/10")
        all_raw = []
        for split_name, split_data in raw_dataset.items():
            all_raw.extend(list(split_data))
        cap = args.max_samples if args.max_samples else None
        all_samples = preprocess_split(
            all_raw, tokenizer, "all", coco_dir, cap
        )
        processed = make_splits(all_samples)

    # ── Step 5: Save pkl files ────────────────────────────────────
    checksums = {}
    for split, samples in processed.items():
        path = save_split(samples, output_dir, split)
        checksums[split] = compute_md5(path)

    save_stats(processed, output_dir, checksums, args)

    # ── Step 6: Final verification ────────────────────────────────
    print("\n[Stage 1] Verifying output files ...")
    if verify_output(output_dir):
        print("✅ Stage 1 complete! All files verified.")
    else:
        print("❌ Verification failed — check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 1: RefCOCO Data Preparation"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="~/SharedFolder/MDAIE/group6/data/",
        help="Where to save pickle files",
    )
    parser.add_argument(
        "--coco_dir",
        type=str,
        default="~/SharedFolder/MDAIE/group6/coco/",
        help=(
            "Directory containing COCO images. "
            "Expected layout: <coco_dir>/train2014/*.jpg"
        ),
    )
    parser.add_argument(
        "--hf_home",
        type=str,
        default=None,
        help="HuggingFace cache directory (default: $HF_HOME env var)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun even if output files already exist",
    )
    parser.add_argument(
        "--skip_coco_download",
        action="store_true",
        help=(
            "Skip downloading COCO images. "
            "Use if images are already in --coco_dir."
        ),
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help=(
            "Max training samples (e.g. 60000 for ~50%%). "
            "val/test are scaled to 12.5%% of this value. "
            "Default: all samples (~42,000 train)."
        ),
    )
    args = parser.parse_args()
    main(args)
