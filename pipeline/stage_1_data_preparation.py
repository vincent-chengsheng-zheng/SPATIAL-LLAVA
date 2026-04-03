"""
pipeline/stage_1_data_preparation.py

Stage 1: Download RefCOCO annotations + COCO images, preprocess into pkl.

Key design: NO tokenizer here. Only image preprocessing + bbox normalization.
Tokenization happens at training time in refcoco_loader.py.
This avoids downloading the 500MB LLaVA tokenizer during data prep.

Sample format saved in pkl:
    {
        "image" : Tensor(3, 384, 384)  — preprocessed image
        "text"  : str                  — raw referring expression
        "bbox"  : Tensor(4,)           — [xc, yc, w, h] normalized
    }

Usage:
    bash shared/scripts/download_data.sh   # recommended

    # or manually:
    python pipeline/stage_1_data_preparation.py \\
        --output_dir /tmp/data/ \\
        --coco_dir   /tmp/coco/ \\
        --skip_coco_download
"""

import os
import sys
import json
import pickle
import argparse
import hashlib
import random
import zipfile
import urllib.request
from typing import List, Dict
from datasets import Dataset

import torch
from PIL import Image
import torchvision.transforms.v2 as transforms

# Add repo root to path so core/ is importable
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from core.data.preprocessing import (  # noqa: E402
    normalize_bbox,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


# ── Constants ─────────────────────────────────────────────────────────────────

REFCOCO_DATASET = "jxu124/refcoco"
COCO_IMAGES_URL = "http://images.cocodataset.org/zips/train2014.zip"
COCO_EXPECTED_SIZE = 13510573713
COCO_EXPECTED_IMGS = 82784
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 42

# Image transform (no tokenizer needed)
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_md5(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_output(output_dir: str) -> bool:
    """Check all pkl files exist, are non-empty, and have correct format."""
    for split in ["train", "val", "test"]:
        p = os.path.join(output_dir, f"refcoco_{split}.pkl")
        if not os.path.exists(p) or os.path.getsize(p) < 100:
            print(f"  ✗ Missing or empty: {p}")
            return False
    stats = os.path.join(output_dir, "dataset_stats.json")
    if not os.path.exists(stats):
        print(f"  ✗ Missing: {stats}")
        return False
    with open(stats) as f:
        s = json.load(f)
    if s.get("total_samples", 0) == 0:
        print("  ✗ dataset_stats.json reports 0 samples")
        return False
    # Check new format (text field, not input_ids)
    if not s.get("format_version") == "v2_raw_text":
        print("  ✗ Old format detected. Use --force to regenerate.")
        return False
    return True


def count_coco_images(train2014_dir: str) -> int:
    if not os.path.isdir(train2014_dir):
        return 0
    return sum(1 for f in os.listdir(train2014_dir) if f.endswith(".jpg"))


def resolve_image_path(raw: dict, coco_dir: str) -> str:
    image_path = raw.get("image_path", "")
    parts = image_path.replace("\\", "/").split("/")
    if parts and parts[0].lower() == "coco":
        parts = parts[1:]
    return os.path.join(os.path.expanduser(coco_dir), *parts)


# ── Step 1: Download annotations ──────────────────────────────────────────────

def download_refcoco(hf_home: str):
    """Download RefCOCO annotations from HuggingFace (~50MB, fast)."""
    print("\n[Stage 1] Downloading RefCOCO annotations from HuggingFace ...")
    from datasets import load_dataset

    os.environ["HF_HOME"] = hf_home
    dataset = load_dataset(REFCOCO_DATASET, cache_dir=hf_home)
    print(f"  ✓ Downloaded. Splits: {list(dataset.keys())}")
    for split, data in dataset.items():
        print(f"    {split}: {len(data):,} samples")
    return dataset


# ── Step 2: Download + extract COCO images ────────────────────────────────────

def _download_progress(block_num: int, block_size: int, total_size: int):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded * 100 / total_size, 100)
        bar = "#" * int(pct / 2)
        print(
            f"\r  [{bar:<50}] {pct:5.1f}%  "
            f"{downloaded / 1e9:.2f}/{total_size / 1e9:.2f} GB",
            end="", flush=True,
        )


def _extract_zip(zip_path: str, dest_dir: str) -> bool:
    """Extract zip using Python's zipfile. Skips already-extracted files."""
    print(f"\n[Stage 1] Extracting {zip_path} ...")
    train_dir = os.path.join(dest_dir, "train2014")
    os.makedirs(train_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = [m for m in zf.namelist() if m.endswith(".jpg")]
            total = len(members)
            extracted = 0
            for member in members:
                fname = os.path.basename(member)
                dest = os.path.join(train_dir, fname)
                if os.path.exists(dest):
                    continue
                zf.extract(member, dest_dir)
                extracted += 1
                if extracted % 10000 == 0:
                    done = count_coco_images(train_dir)
                    print(f"  [{done:,}/{total:,}] extracted...")
        n = count_coco_images(train_dir)
        print(f"  ✓ Done: {n:,} images")
        return True
    except Exception as e:
        print(f"  ✗ Extraction failed: {e}")
        return False


def download_coco_images(coco_dir: str) -> bool:
    """Download and extract COCO train2014 using only Python stdlib."""
    coco_dir = os.path.expanduser(coco_dir)
    train2014_dir = os.path.join(coco_dir, "train2014")
    zip_path = os.path.join(coco_dir, "train2014.zip")
    os.makedirs(coco_dir, exist_ok=True)

    n_imgs = count_coco_images(train2014_dir)
    if n_imgs >= COCO_EXPECTED_IMGS:
        print(
            f"\n[Stage 1] COCO images already present: "
            f"{n_imgs:,} images"
        )
        return True

    zip_size = os.path.getsize(zip_path) if os.path.exists(zip_path) else 0
    if zip_size != COCO_EXPECTED_SIZE:
        print("\n[Stage 1] Downloading COCO train2014 (~13.5 GB)...")
        try:
            urllib.request.urlretrieve(
                COCO_IMAGES_URL, zip_path, _download_progress
            )
            print()
        except Exception as e:
            print(f"\n  ✗ Download failed: {e}")
            return False

    ok = _extract_zip(zip_path, coco_dir)
    if ok:
        print("  Removing zip to free space...")
        os.remove(zip_path)
    return ok


# ── Step 3: Preprocess ────────────────────────────────────────────────────────

def preprocess_sample(raw: dict, coco_dir: str) -> Dict:
    """
    Convert one raw RefCOCO sample into a pkl-ready dict.

    Saves raw text string instead of tokenized input_ids.
    Tokenization happens at training time in RefCOCODataset.__getitem__().

    Returns:
        {
            "image" : Tensor(3, 384, 384)  preprocessed image
            "text"  : str                  raw referring expression
            "bbox"  : Tensor(4,)           normalized [xc, yc, w, h]
        }
    """
    # ── Image ──────────────────────────────────────────────────────
    img_path = resolve_image_path(raw, coco_dir)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    pil_img = Image.open(img_path).convert("RGB")
    img_tensor = IMAGE_TRANSFORM(pil_img)

    # ── Text: save raw string, tokenize later ──────────────────────
    sentences = raw.get("sentences", [])
    if isinstance(sentences, list) and len(sentences) > 0:
        sent = random.choice(sentences)
        text = sent["raw"] if isinstance(sent, dict) else str(sent)
    else:
        captions = raw.get("captions", [])
        text = captions[0] if captions else "find the object"

    # ── BBox: xyxy pixel → normalized xywh ─────────────────────────
    x1, y1, x2, y2 = raw["bbox"]
    try:
        img_info = json.loads(raw.get("raw_image_info", "{}"))
        img_w = img_info.get("width", pil_img.width)
        img_h = img_info.get("height", pil_img.height)
    except (json.JSONDecodeError, TypeError):
        img_w, img_h = pil_img.width, pil_img.height

    bbox_norm = normalize_bbox(
        [x1, y1, x2, y2], img_w=img_w, img_h=img_h
    )

    return {
        "image": img_tensor,   # float32 (3, 384, 384)
        "text": text,          # str — raw referring expression
        "bbox": bbox_norm,     # float32 (4,)
    }


def preprocess_split(
    raw_split,
    split_name: str,
    coco_dir: str,
    max_samples: int = None,
) -> List[Dict]:
    """Preprocess using HF map + multiprocessing for speed on A100."""
    total_available = len(raw_split)
    total = min(total_available, max_samples) if max_samples else total_available

    print(
        f"\n[Stage 1] Preprocessing '{split_name}' "
        f"({total}/{total_available} samples) with multiprocessing..."
    )

    # 转为 HF Dataset 以支持 map + num_proc
    if isinstance(raw_split, list):
        ds = Dataset.from_list(raw_split[:total])
    else:
        ds = raw_split.select(range(total)) if hasattr(raw_split, "select") else raw_split

    def process_fn(example):
        try:
            return preprocess_sample(example, coco_dir)
        except Exception as e:
            # 返回空 dict，让后面过滤
            print(f"  ⚠ Skipped sample: {e}")
            return {"image": None, "text": "", "bbox": None}

    # 关键加速参数（A100 上建议 16~32，根据你的 CPU 核心数调整）
    processed_ds = ds.map(
        process_fn,
        batched=False,          # 单样本处理（因为有随机选择句子）
        num_proc=24,            # ←←← 这里改大！推荐 16~32（A100 有很多核心）
        # remove_columns 可以根据需要加
    )

    # 过滤掉失败的样本
    samples = [ex for ex in processed_ds if ex["image"] is not None]

    print(f"  ✓ {len(samples)} samples processed successfully")
    return samples


# ── Step 4: Split (fallback) ──────────────────────────────────────────────────

def make_splits(all_samples: List[Dict]) -> Dict[str, List[Dict]]:
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
    path = os.path.join(output_dir, f"refcoco_{split}.pkl")
    print(f"\n[Stage 1] Saving '{split}' → {path} ({len(samples):,} samples)")
    with open(path, "wb") as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(path) / 1e6
    md5 = compute_md5(path)
    print(f"  ✓ {size_mb:.0f} MB  |  MD5: {md5}")
    return path


def save_stats(splits, output_dir, checksums, args) -> None:
    stats = {
        "format_version": "v2_raw_text",
        "total_samples": sum(len(v) for v in splits.values()),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "max_samples_cap": args.max_samples,
        "image_size": IMAGE_SIZE,
        "refcoco_dataset": REFCOCO_DATASET,
        "seed": SEED,
        "checksums": checksums,
        "note": "text field is raw str; tokenized in RefCOCODataset.__getitem__",
    }
    path = os.path.join(output_dir, "dataset_stats.json")
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n[Stage 1] Stats → {path}")
    print(json.dumps(stats, indent=2))


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    output_dir = os.path.expanduser(args.output_dir)
    coco_dir = os.path.expanduser(args.coco_dir)
    hf_home = os.path.expanduser(
        args.hf_home or os.environ.get("HF_HOME", "/tmp/hf_cache")
    )

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(hf_home, exist_ok=True)
    os.makedirs(coco_dir, exist_ok=True)

    random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 60)
    print("  Stage 1: Data Preparation (v2 — no tokenizer)")
    print(f"  Output dir  : {output_dir}")
    print(f"  COCO dir    : {coco_dir}")
    print(f"  HF cache    : {hf_home}")
    print(f"  Max samples : {args.max_samples or 'all'}")
    print("=" * 60)

    if not args.force and verify_output(output_dir):
        print("\n✅ All output files already exist. Use --force to rerun.")
        return

    # ── Download annotations ──────────────────────────────────────
    raw_dataset = download_refcoco(hf_home)

    # ── COCO images ───────────────────────────────────────────────
    if args.skip_coco_download:
        print("\n[Stage 1] Skipping COCO download (--skip_coco_download)")
        n = count_coco_images(os.path.join(coco_dir, "train2014"))
        if n == 0:
            print(f"  ✗ No images in {coco_dir}/train2014/")
            sys.exit(1)
        print(f"  ✓ Found {n:,} images")
    else:
        if not download_coco_images(coco_dir):
            print("\n❌ COCO download failed.")
            sys.exit(1)

    # ── Preprocess ────────────────────────────────────────────────
    if set(raw_dataset.keys()) >= {"train", "validation", "test"}:
        print("\n[Stage 1] Using official splits")
        train_max = args.max_samples
        val_max = int(args.max_samples * 0.125) if args.max_samples else None
        test_max = int(args.max_samples * 0.125) if args.max_samples else None
        processed = {
            "train": preprocess_split(
                raw_dataset["train"], "train", coco_dir, train_max
            ),
            "val": preprocess_split(
                raw_dataset["validation"], "val", coco_dir, val_max
            ),
            "test": preprocess_split(
                raw_dataset["test"], "test", coco_dir, test_max
            ),
        }
    else:
        all_raw = []
        for split_data in raw_dataset.values():
            all_raw.extend(list(split_data))
        all_samples = preprocess_split(
            all_raw, "all", coco_dir, args.max_samples
        )
        processed = make_splits(all_samples)

    # ── Save ──────────────────────────────────────────────────────
    checksums = {}
    for split, samples in processed.items():
        path = save_split(samples, output_dir, split)
        checksums[split] = compute_md5(path)

    save_stats(processed, output_dir, checksums, args)

    print("\n[Stage 1] Verifying ...")
    if verify_output(output_dir):
        print("✅ Stage 1 complete!")
    else:
        print("❌ Verification failed.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 1: RefCOCO Data Preparation (no tokenizer)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="/tmp/data/",
        help="Where to save pkl files (default: /tmp/data/)"
    )
    parser.add_argument(
        "--coco_dir", type=str, default="/tmp/coco/",
        help="COCO images dir (expected: <coco_dir>/train2014/*.jpg)"
    )
    parser.add_argument(
        "--hf_home", type=str, default=None,
        help="HuggingFace cache dir (default: /tmp/hf_cache)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rerun even if output files exist"
    )
    parser.add_argument(
        "--skip_coco_download", action="store_true",
        help="Skip COCO download/extract (images already in --coco_dir)"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Max train samples. val/test scaled to 12.5%%. Default: all."
    )
    args = parser.parse_args()
    main(args)
