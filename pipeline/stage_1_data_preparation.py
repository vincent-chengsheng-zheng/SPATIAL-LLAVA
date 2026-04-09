"""
pipeline/stage_1_data_preparation.py

Stage 1: Download RefCOCO annotations + COCO images, save metadata to pkl.

What this script does:
    1. Download RefCOCO annotations from HuggingFace (jxu124/refcoco, ~50MB)
    2. (Optional) Download + extract COCO train2014 images (~13.5GB)
    3. For each sample: resolve image path + extract text + normalize bbox
    4. Save lightweight pkl files (paths + text + bbox, NO image tensors)
    5. Image loading and transforms happen later in refcoco_loader.py

Why NO image tensors here:
    - Saving tensors = ~72GB pkl, slow IPC between processes
    - Saving paths   = ~5MB pkl, near-instant
    - DataLoader workers handle image loading in parallel during training

Sample format saved in pkl (v3):
    {
        "image_path" : str           absolute path to COCO .jpg file
        "text"       : str           raw referring expression
        "bbox"       : Tensor(4,)    normalized [xc, yc, w, h] in [0,1]
    }

Usage:
    bash shared/scripts/download_data.sh   # recommended

    # or manually:
    python pipeline/stage_1_data_preparation.py \\
        --output_dir data/ \\
        --coco_dir   data/coco/ \\
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

import torch

# ── sys.path must be set before any core/ imports ─────────────────────────────
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from core.paths import PATHS                  # noqa: E402
from core.data.preprocessing import normalize_bbox  # noqa: E402


# ── Constants ─────────────────────────────────────────────────────────────────

REFCOCO_DATASET = "jxu124/refcoco"
COCO_IMAGES_URL = "http://images.cocodataset.org/zips/train2014.zip"
COCO_EXPECTED_SIZE = 13510573713  # bytes, verified via curl -sI
COCO_EXPECTED_IMGS = 82783
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 42
FORMAT_VERSION = "v3_image_path"


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_md5(path: str, chunk_size: int = 1 << 20) -> str:
    """Compute MD5 checksum of a file for integrity verification."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_output(output_dir: str) -> bool:
    """
    Check that all 3 pkl files and stats JSON exist, are non-empty,
    and are in the correct v3 format (image_path, not tensor).
    """
    for split in ["train", "val", "test"]:
        p = os.path.join(output_dir, f"refcoco_{split}.pkl")
        if not os.path.exists(p) or os.path.getsize(p) < 100:
            print(f"  ✗ Missing or empty: {p}")
            return False
    stats_path = os.path.join(output_dir, "dataset_stats.json")
    if not os.path.exists(stats_path):
        print(f"  ✗ Missing: {stats_path}")
        return False
    with open(stats_path) as f:
        s = json.load(f)
    if s.get("total_samples", 0) == 0:
        print("  ✗ dataset_stats.json reports 0 samples")
        return False
    if s.get("format_version") != FORMAT_VERSION:
        print(
            f"  ✗ Wrong format: {s.get('format_version')} "
            f"(expected {FORMAT_VERSION}). Use --force to regenerate."
        )
        return False
    return True


def count_coco_images(train2014_dir: str) -> int:
    """Count .jpg files in a COCO train2014 directory."""
    if not os.path.isdir(train2014_dir):
        return 0
    return sum(1 for f in os.listdir(train2014_dir) if f.endswith(".jpg"))


def resolve_image_path(raw: dict, coco_dir: str) -> str:
    """
    Build the absolute path to the COCO image for a RefCOCO sample.

    jxu124/refcoco stores image_path as:
        "coco/train2014/COCO_train2014_000000581857.jpg"

    We strip "coco/" and join with coco_dir to get:
        <repo>/data/coco/train2014/COCO_train2014_000000581857.jpg
    """
    image_path = raw.get("image_path", "")
    parts = image_path.replace("\\", "/").split("/")
    if parts and parts[0].lower() == "coco":
        parts = parts[1:]
    return os.path.join(os.path.expanduser(coco_dir), *parts)


# ── Step 1: Download annotations ──────────────────────────────────────────────

def download_refcoco(hf_home: str):
    """
    Download RefCOCO annotations from HuggingFace.
    Only ~50MB, very fast. Cached after first download.
    """
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
    """Progress bar callback for urllib.request.urlretrieve."""
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
    """
    Extract COCO zip using Python's built-in zipfile (no unzip needed).
    Skips files that are already extracted (resume-safe).
    """
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
    """
    Download and extract COCO train2014 images using Python stdlib only.
    No wget or unzip required. Resume-safe for both download and extraction.
    """
    coco_dir = os.path.expanduser(coco_dir)
    train2014_dir = os.path.join(coco_dir, "train2014")
    # zip stays in /tmp to avoid filling the repo with a 13GB file
    zip_path = "/tmp/train2014.zip"
    os.makedirs(coco_dir, exist_ok=True)

    # Already extracted?
    n_imgs = count_coco_images(train2014_dir)
    if n_imgs >= COCO_EXPECTED_IMGS:
        print(f"\n[Stage 1] COCO images already present: {n_imgs:,}")
        return True

    # Download if zip is incomplete or missing
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
        print("  Removing zip from /tmp/ ...")
        os.remove(zip_path)
    return ok


# ── Step 3: Preprocess one sample ─────────────────────────────────────────────

def preprocess_sample(raw: dict, coco_dir: str) -> Dict:
    """
    Convert one raw RefCOCO annotation into a lightweight metadata dict.

    What this does:
        1. Resolve the absolute path to the COCO image file
        2. Verify the image file exists on disk
        3. Pick one random referring expression from the sentences list
        4. Parse image dimensions from raw_image_info JSON
        5. Convert bbox from [x1,y1,x2,y2] pixels → [xc,yc,w,h] normalized

    What this does NOT do (happens later in DataLoader):
        - Load the image into memory
        - Resize or normalize the image
        - Tokenize the text

    Args:
        raw      : One sample dict from jxu124/refcoco HuggingFace dataset
        coco_dir : Path to the COCO images directory (e.g. data/coco/)

    Returns:
        {
            "image_path" : str       absolute path to .jpg file
            "text"       : str       one referring expression (randomly chosen)
            "bbox"       : Tensor(4) normalized [xc, yc, w, h]
        }

    Raises:
        FileNotFoundError : If the image file is not found on disk
    """
    # ── 1. Resolve image path ──────────────────────────────────────
    img_path = resolve_image_path(raw, coco_dir)
    if not os.path.exists(img_path):
        raise FileNotFoundError(
            f"Image not found: {img_path}\n"
            "  Ensure COCO train2014 is extracted to --coco_dir"
        )

    # ── 2. Pick one referring expression ──────────────────────────
    sentences = raw.get("sentences", [])
    if isinstance(sentences, list) and len(sentences) > 0:
        sent = random.choice(sentences)
        text = sent["raw"] if isinstance(sent, dict) else str(sent)
    else:
        captions = raw.get("captions", [])
        text = captions[0] if captions else "find the object"

    # ── 3. Parse image dimensions ──────────────────────────────────
    try:
        img_info = json.loads(raw.get("raw_image_info", "{}"))
        img_w = img_info.get("width", 640)
        img_h = img_info.get("height", 480)
    except (json.JSONDecodeError, TypeError):
        img_w, img_h = 640, 480

    # ── 4. Normalize bbox from xyxy pixels → xywh normalized ──────
    # jxu124/refcoco stores bbox as [x1, y1, x2, y2] in pixel coords
    x1, y1, x2, y2 = raw["bbox"]
    bbox_norm = normalize_bbox(
        [x1, y1, x2, y2], img_w=img_w, img_h=img_h
    )

    return {
        "image_path": img_path,   # str  — loaded by DataLoader workers
        "text": text,             # str  — tokenized by RefCOCODataset
        "bbox": bbox_norm,        # Tensor(4) float32
    }


# ── Step 4: Preprocess a full split ───────────────────────────────────────────

def preprocess_split(
    raw_split,
    split_name: str,
    coco_dir: str,
    max_samples: int = None,
) -> List[Dict]:
    """
    Process all (or up to max_samples) samples from one dataset split.

    What this does:
        1. Iterate over raw RefCOCO samples
        2. Call preprocess_sample() on each
        3. Skip samples where the image file is missing
        4. Print progress every 5000 samples

    Args:
        raw_split   : HuggingFace dataset split (train/validation/test)
        split_name  : Human-readable name for logging
        coco_dir    : Path to COCO images directory
        max_samples : Cap on number of samples (None = all)

    Returns:
        List of sample dicts
    """
    total_available = len(raw_split)
    total = (
        min(total_available, max_samples) if max_samples else total_available
    )
    errors = 0
    missing = 0
    samples = []

    print(
        f"\n[Stage 1] Processing '{split_name}' "
        f"({total}/{total_available} samples) ..."
    )

    for i, raw in enumerate(raw_split):
        if max_samples is not None and len(samples) >= max_samples:
            break
        try:
            sample = preprocess_sample(raw, coco_dir)
            samples.append(sample)
        except FileNotFoundError as e:
            missing += 1
            errors += 1
            if missing <= 3:
                print(f"  ⚠ {e}")
            continue
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ⚠ Error at index {i}: {e}")
            continue

        n = len(samples)
        if n % 5000 == 0 or n == total:
            print(
                f"  [{n}/{total}] done "
                f"(errors={errors}, missing={missing})"
            )

    print(f"  ✓ {len(samples)} samples saved ({errors} skipped)")
    return samples


# ── Step 5: Split (fallback if no official splits) ────────────────────────────

def make_splits(all_samples: List[Dict]) -> Dict[str, List[Dict]]:
    """Shuffle and split 80/10/10 when no official splits are available."""
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


# ── Step 6: Save pkl files ────────────────────────────────────────────────────

def save_split(samples: List[Dict], output_dir: str, split: str) -> str:
    """
    Serialize samples to a pickle file.

    The pkl is tiny because we store paths (strings), not image tensors.
    Typical size: ~3MB for 42k samples (vs ~72GB for tensors).
    """
    path = os.path.join(output_dir, f"refcoco_{split}.pkl")
    print(
        f"\n[Stage 1] Saving '{split}' → {path} "
        f"({len(samples):,} samples) ..."
    )
    with open(path, "wb") as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_kb = os.path.getsize(path) / 1e3
    md5 = compute_md5(path)
    print(f"  ✓ {size_kb:.0f} KB  |  MD5: {md5}")
    return path


def save_stats(
    splits: Dict[str, List],
    output_dir: str,
    checksums: Dict,
    args,
) -> None:
    """Save dataset_stats.json with metadata about this preprocessing run."""
    stats = {
        "format_version": FORMAT_VERSION,
        "total_samples": sum(len(v) for v in splits.values()),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "max_samples_cap": args.max_samples,
        "refcoco_dataset": REFCOCO_DATASET,
        "seed": SEED,
        "checksums": checksums,
        "note": (
            "image_path is absolute path to COCO jpg. "
            "Image loading and transforms happen in RefCOCODataset.__getitem__"
        ),
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
        args.hf_home or os.environ.get("HF_HOME", str(PATHS.weights))
    )

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(hf_home, exist_ok=True)
    os.makedirs(coco_dir, exist_ok=True)

    random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 60)
    print("  Stage 1: Data Preparation (v3 — image paths only)")
    print(f"  Output dir  : {output_dir}")
    print(f"  COCO dir    : {coco_dir}")
    print(f"  HF cache    : {hf_home}")
    print(f"  Max samples : {args.max_samples or 'all'}")
    print("=" * 60)

    if not args.force and verify_output(output_dir):
        print("\n✅ All output files already exist. Use --force to rerun.")
        return

    # ── Download RefCOCO annotations ──────────────────────────────
    raw_dataset = download_refcoco(hf_home)

    # ── COCO images ───────────────────────────────────────────────
    if args.skip_coco_download:
        print("\n[Stage 1] Skipping COCO download (--skip_coco_download)")
        train2014_dir = os.path.join(coco_dir, "train2014")
        n = count_coco_images(train2014_dir)
        if n == 0:
            print(f"  ✗ No images in {train2014_dir}")
            sys.exit(1)
        print(f"  ✓ Found {n:,} images")
    else:
        if not download_coco_images(coco_dir):
            print("\n❌ COCO download failed.")
            sys.exit(1)

    # ── Preprocess splits ─────────────────────────────────────────
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
        print("\n[Stage 1] No official splits — pooling and splitting 80/10/10")
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

    # ── Verify ────────────────────────────────────────────────────
    print("\n[Stage 1] Verifying output ...")
    if verify_output(output_dir):
        print("✅ Stage 1 complete! pkl files ready in:", output_dir)
    else:
        print("❌ Verification failed — check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 1: RefCOCO Data Preparation (v3 — paths only)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(PATHS.data),
        help=f"Where to save pkl files (default: {PATHS.data})"
    )
    parser.add_argument(
        "--coco_dir", type=str, default=str(PATHS.coco),
        help=f"COCO images dir. Expected: <coco_dir>/train2014/*.jpg (default: {PATHS.coco})"
    )
    parser.add_argument(
        "--hf_home", type=str, default=None,
        help=f"HuggingFace cache dir (default: {PATHS.weights})"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rerun even if output files already exist"
    )
    parser.add_argument(
        "--skip_coco_download", action="store_true",
        help="Skip COCO download. Use if images are already in --coco_dir"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help=(
            "Max train samples (e.g. 30000). "
            "val/test scaled to 12.5%% of this. Default: all."
        ),
    )
    args = parser.parse_args()
    main(args)
