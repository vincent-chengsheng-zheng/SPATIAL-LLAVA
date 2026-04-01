"""
pipeline/stage_1_data_preparation.py

Stage 1: Download RefCOCO and preprocess into pickle files.

Usage:
    python pipeline/stage_1_data_preparation.py \
        --output_dir ~/SharedFolder/MDAIE/group6/data/

    # Use 50% of data (~60k samples, ~1 hour)
    python pipeline/stage_1_data_preparation.py \
        --output_dir ~/SharedFolder/MDAIE/group6/data/ \
        --max_samples 60000

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
from typing import List, Dict

# Add repo root to path so core/ is importable
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
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
    """Check that all 3 pkl files and stats JSON exist."""
    for split in ["train", "val", "test"]:
        p = os.path.join(output_dir, f"refcoco_{split}.pkl")
        if not os.path.exists(p):
            print(f"  ✗ Missing: {p}")
            return False
    stats = os.path.join(output_dir, "dataset_stats.json")
    if not os.path.exists(stats):
        print(f"  ✗ Missing: {stats}")
        return False
    return True


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
            {"additional_special_tokens": [LOC_TOKEN]})
        print(
            f"  ✓ Added '{LOC_TOKEN}' to vocabulary "
            f"(new vocab size: {len(tokenizer)})"
        )
    else:
        print(f"  ✓ '{LOC_TOKEN}' already in vocabulary")
    return tokenizer


# ── Step 1: Download ──────────────────────────────────────────────────────────

def download_refcoco(hf_home: str):
    """Download RefCOCO from HuggingFace datasets."""
    print("\n[Stage 1] Downloading RefCOCO from HuggingFace ...")
    from datasets import load_dataset

    os.environ["HF_HOME"] = hf_home
    dataset = load_dataset(REFCOCO_DATASET, cache_dir=hf_home)
    print(f"  ✓ Downloaded. Splits available: {list(dataset.keys())}")
    return dataset


# ── Step 2: Preprocess ────────────────────────────────────────────────────────

def preprocess_sample(raw: dict, tokenizer) -> Dict:
    """
    Convert one raw RefCOCO sample into model-ready tensors.

    Expected raw keys (from HuggingFace refcoco):
        - image     : PIL.Image
        - sentences : list of dicts with "raw" key
        - bbox      : [x1, y1, w, h] in pixel coords (COCO format)
        - width     : image width
        - height    : image height
    """
    pil_img = raw["image"].convert("RGB")
    img_tensor = preprocess_image_from_pil(pil_img)

    sentences = raw.get("sentences", raw.get("refs", []))
    if isinstance(sentences, list) and len(sentences) > 0:
        sent = random.choice(sentences)
        prompt = sent["raw"] if isinstance(sent, dict) else str(sent)
    else:
        prompt = str(raw.get("caption", "find the object"))

    input_ids = preprocess_text(prompt, tokenizer)

    img_w = raw.get("width", pil_img.width)
    img_h = raw.get("height", pil_img.height)
    raw_bbox = raw["bbox"]
    x1, y1, bw, bh = raw_bbox
    bbox_xyxy = [x1, y1, x1 + bw, y1 + bh]
    bbox_norm = normalize_bbox(bbox_xyxy, img_w=img_w, img_h=img_h)

    return {
        "image": img_tensor,
        "input_ids": input_ids,
        "bbox": bbox_norm,
    }


def preprocess_split(
    raw_split,
    tokenizer,
    split_name: str,
    max_samples: int = None,
) -> List[Dict]:
    """
    Preprocess all (or up to max_samples) samples in one split.

    Args:
        raw_split   : HuggingFace dataset split
        tokenizer   : LLaVA tokenizer
        split_name  : "train", "val", or "test"
        max_samples : Cap on number of samples to process. None = all.
    """
    samples = []
    total_available = len(raw_split)
    total = min(total_available, max_samples) if max_samples else total_available
    errors = 0

    print(
        f"\n[Stage 1] Preprocessing '{split_name}' split "
        f"({total}/{total_available} samples) ..."
    )

    for i, raw in enumerate(raw_split):
        if max_samples is not None and len(samples) >= max_samples:
            break
        try:
            sample = preprocess_sample(raw, tokenizer)
            samples.append(sample)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ⚠ Error at index {i}: {e}")
            continue

        if (len(samples)) % 5000 == 0 or len(samples) == total:
            print(
                f"  [{len(samples)}/{total}] processed, "
                f"errors so far: {errors}"
            )

    print(f"  ✓ Done: {len(samples)} samples ({errors} skipped)")
    return samples


# ── Step 3: Split ─────────────────────────────────────────────────────────────

def make_splits(all_samples: List[Dict]) -> Dict[str, List[Dict]]:
    """Shuffle and split samples into train/val/test (80/10/10)."""
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


# ── Step 4: Save ──────────────────────────────────────────────────────────────

def save_split(samples: List[Dict], output_dir: str, split: str) -> str:
    """Save samples to a pickle file. Returns file path."""
    path = os.path.join(output_dir, f"refcoco_{split}.pkl")
    print(
        f"\n[Stage 1] Saving '{split}' → {path} ({len(samples)} samples) ..."
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
    max_samples: int = None,
) -> None:
    """Save dataset_stats.json."""
    stats = {
        "total_samples": sum(len(v) for v in splits.values()),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "split_ratios": SPLITS,
        "max_samples_cap": max_samples,
        "image_size": 384,
        "max_length": MAX_LENGTH,
        "loc_token": LOC_TOKEN,
        "llava_model": LLAVA_MODEL_ID,
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
    hf_home = os.path.expanduser(
        args.hf_home or os.environ.get(
            "HF_HOME", "~/SharedFolder/MDAIE/group6/hf_cache"
        )
    )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(hf_home, exist_ok=True)

    random.seed(SEED)
    torch.manual_seed(SEED)

    pct = f"~{args.max_samples * 100 // 120000}%" if args.max_samples else "100%"

    print("=" * 60)
    print("  Stage 1: Data Preparation")
    print(f"  Output dir  : {output_dir}")
    print(f"  HF cache    : {hf_home}")
    print(f"  Max samples : {args.max_samples or 'all'} ({pct} of dataset)")
    print("=" * 60)

    if not args.force and verify_output(output_dir):
        print("\n✅ All output files already exist. Use --force to rerun.")
        return

    raw_dataset = download_refcoco(hf_home)
    tokenizer = setup_tokenizer(hf_home)

    if set(raw_dataset.keys()) >= {"train", "validation", "test"}:
        print("\n[Stage 1] Using official train/validation/test splits")
        train_max = args.max_samples
        val_max = int(args.max_samples * 0.125) if args.max_samples else None
        test_max = int(args.max_samples * 0.125) if args.max_samples else None
        processed = {
            "train": preprocess_split(
                raw_dataset["train"], tokenizer, "train", train_max
            ),
            "val": preprocess_split(
                raw_dataset["validation"], tokenizer, "val", val_max
            ),
            "test": preprocess_split(
                raw_dataset["test"], tokenizer, "test", test_max
            ),
        }
    else:
        print("\n[Stage 1] No official splits — pooling and splitting 80/10/10")
        all_raw = []
        for split_name, split_data in raw_dataset.items():
            all_raw.extend(list(split_data))
        cap = args.max_samples if args.max_samples else None
        all_samples = preprocess_split(all_raw, tokenizer, "all", cap)
        processed = make_splits(all_samples)

    checksums = {}
    for split, samples in processed.items():
        path = save_split(samples, output_dir, split)
        checksums[split] = compute_md5(path)

    save_stats(processed, output_dir, checksums, args.max_samples)

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
        "--max_samples",
        type=int,
        default=None,
        help=(
            "Max training samples to process (e.g. 60000 for ~50%%). "
            "val/test are scaled to 12.5%% of this value. "
            "Default: all samples (~120,000)."
        ),
    )
    args = parser.parse_args()
    main(args)
