#!/bin/bash
# =============================================================
# shared/scripts/download_data.sh
#
# One-command data preparation. Everything runs in /tmp/ (local SSD).
# SharedFolder is only used for checkpoints/results/logs.
#
# Usage:
#   bash shared/scripts/download_data.sh           # Full pipeline
#   bash shared/scripts/download_data.sh --check   # Check status only
#   bash shared/scripts/download_data.sh --force   # Force re-preprocess
#
# Steps (auto-resumed if already done):
#   1. Download COCO zip to /tmp/train2014.zip
#   2. Extract to /tmp/coco/train2014/
#   3. Preprocess → pkl files in /tmp/data/
#   4. Train directly from /tmp/data/
# =============================================================

set -e

BASE_DIR=~/SharedFolder/MDAIE/group6
HF_CACHE=/tmp/hf_cache
REPO_DIR=~/spatial-llava
LOG_DIR=$BASE_DIR/logs

# Everything local
TMP_ZIP=/tmp/train2014.zip
TMP_COCO=/tmp/coco
TMP_DATA=/tmp/data

COCO_URL="http://images.cocodataset.org/zips/train2014.zip"
COCO_EXPECTED_SIZE=13510573713
COCO_EXPECTED_IMGS=82783   # 修正为实际数量（原脚本多写了1）

FORCE_FLAG=""
MODE="auto"
for arg in "$@"; do
    case $arg in
        --check) MODE="check" ;;
        --force) FORCE_FLAG="--force"; MODE="force" ;;
    esac
done

echo "======================================================"
echo " Spatial-LLaVA — Data Preparation"
echo " All data in : /tmp/ (local SSD, fast)"
echo " Persistent  : checkpoints/results/logs → SharedFolder"
echo " Mode        : $MODE"
echo " Started     : $(date)"
echo "======================================================"

# ── Helper: count images ───────────────────────────────────
count_imgs() {
    ls /tmp/coco/train2014/*.jpg 2>/dev/null | wc -l
}

# ── Status check ───────────────────────────────────────────
echo ""
echo "── Current Status ────────────────────────────────────"

# zip
ZIP_SIZE=$(stat -c%s $TMP_ZIP 2>/dev/null || echo 0)
if [[ $ZIP_SIZE -eq $COCO_EXPECTED_SIZE ]]; then
    echo "  ✅ zip  : $TMP_ZIP ($(du -sh $TMP_ZIP | cut -f1))"
elif [[ $ZIP_SIZE -gt 0 ]]; then
    echo "  ⚠  zip  : partial ($(du -sh $TMP_ZIP | cut -f1) / 13.5GB)"
else
    echo "  ❌ zip  : not found"
fi

# images
N_IMGS=$(count_imgs)
if [[ $N_IMGS -ge $COCO_EXPECTED_IMGS ]]; then
    echo "  ✅ imgs : $N_IMGS images in /tmp/coco/train2014/"
elif [[ $N_IMGS -gt 0 ]]; then
    echo "  ⚠  imgs : $N_IMGS / $COCO_EXPECTED_IMGS extracted"
else
    echo "  ❌ imgs : not extracted"
fi

# pkl
TRAIN_PKL=$TMP_DATA/refcoco_train.pkl
PKL_OK=false
if [[ -f "$TRAIN_PKL" && $(wc -c < "$TRAIN_PKL") -gt 100 ]]; then
    echo "  ✅ pkl  : $(du -sh $TMP_DATA/*.pkl 2>/dev/null | tr '\n' ' ')"
    PKL_OK=true
else
    echo "  ❌ pkl  : not found in /tmp/data/"
fi

# ── Check mode ─────────────────────────────────────────────
if [[ "$MODE" == "check" ]]; then
    echo ""
    if $PKL_OK; then
        echo "======================================================"
        echo " ✅ Ready to train!"
        echo " Data dir: $TMP_DATA"
        echo " Next: bash shared/scripts/start_training.sh main"
        echo "======================================================"
        exit 0
    else
        echo "======================================================"
        echo " ❌ Not ready. Run without --check to continue."
        echo "======================================================"
        exit 1
    fi
fi

# ── Already done ───────────────────────────────────────────
if $PKL_OK && [[ "$MODE" != "force" ]]; then
    echo ""
    echo "======================================================"
    echo " ✅ pkl files already in /tmp/data/. Nothing to do."
    echo " Next: bash shared/scripts/start_training.sh main"
    echo "======================================================"
    exit 0
fi

# ── Step 1: Download zip ───────────────────────────────────
echo ""
echo "[1/3] COCO zip download..."

# 优化后的下载逻辑：
# 只有当 zip 不完整 并且  images 也不足 时，才需要重新下载
if [[ $ZIP_SIZE -eq $COCO_EXPECTED_SIZE ]]; then
    echo "  ✅ Zip already complete, skipping download."
elif [[ $N_IMGS -ge $COCO_EXPECTED_IMGS ]]; then
    echo "  ✅ Images already extracted ($N_IMGS), skipping download and extraction."
else
    echo "  Downloading to $TMP_ZIP (~23 min at 8MB/s)..."
    wget -c $COCO_URL -O $TMP_ZIP
    echo "  ✅ Download complete."
fi

# ── Step 2: Extract ────────────────────────────────────────
echo ""
echo "[2/3] Extracting COCO images..."

N_IMGS=$(count_imgs)
if [[ $N_IMGS -ge $COCO_EXPECTED_IMGS ]]; then
    echo "  ✅ Already extracted: $N_IMGS images. Skipping extraction."
else
    echo "  Extracting $TMP_ZIP → $TMP_COCO ..."
    echo "  (using Python zipfile — no unzip needed)"
    mkdir -p $TMP_COCO

    python3 - << 'EOF'
import zipfile, os, sys

zip_path = "/tmp/train2014.zip"
dest_dir = "/tmp/coco"
train_dir = "/tmp/coco/train2014"

# Count already extracted
already = len([f for f in os.listdir(train_dir) if f.endswith(".jpg")]) \
    if os.path.isdir(train_dir) else 0
print(f"  Already extracted: {already:,} images")

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
            done = already + extracted
            print(f"  [{done:,}/{total:,}] extracted...")

done = len([f for f in os.listdir(train_dir) if f.endswith(".jpg")])
print(f"  ✅ Done: {done:,} images in {train_dir}")
EOF

    N_IMGS=$(count_imgs)
    echo "  Total images: $N_IMGS"

    echo "  Removing zip to free /tmp/ space..."
    rm -f $TMP_ZIP
    echo "  ✅ Zip removed."
fi

# ── Step 3: Preprocessing ─────────────────────────────────────
echo ""
echo "[3/3] Preprocessing RefCOCO → pkl..."
mkdir -p $TMP_DATA $HF_CACHE $LOG_DIR

TIMESTAMP=$(date +%Y%m%d_%H%M)
LOG_FILE=$LOG_DIR/stage1_${TIMESTAMP}.log
echo "  Log: $LOG_FILE"
echo ""

cd $REPO_DIR

python pipeline/stage_1_data_preparation.py \
    --output_dir $TMP_DATA \
    --coco_dir   $TMP_COCO \
    --hf_home    $HF_CACHE \
    --skip_coco_download \
    $FORCE_FLAG \
    2>&1 | tee $LOG_FILE

# ── Final check ────────────────────────────────────────────
echo ""
echo "── Final Verification ────────────────────────────────"
all_good=true
for f in "$TMP_DATA/refcoco_train.pkl" \
         "$TMP_DATA/refcoco_val.pkl" \
         "$TMP_DATA/refcoco_test.pkl" \
         "$TMP_DATA/dataset_stats.json"; do
    if [[ -f "$f" && $(wc -c < "$f") -gt 100 ]]; then
        echo "  ✅ $(basename $f) ($(du -sh $f | cut -f1))"
    else
        echo "  ❌ $(basename $f) — MISSING or EMPTY"
        all_good=false
    fi
done

echo ""
if $all_good; then
    echo "======================================================"
    echo " ✅ Data preparation complete!"
    echo ""
    echo " pkl location: $TMP_DATA  (local SSD)"
    echo " Note: re-run this script if container restarts."
    echo ""
    echo " Next: bash shared/scripts/start_training.sh main"
    echo "======================================================"
    exit 0
else
    echo "======================================================"
    echo " ❌ Something failed. Check log: $LOG_FILE"
    echo "======================================================"
    exit 1
fi
