#!/bin/bash
# =============================================================
# shared/scripts/download_data.sh
#
# Downloads COCO images to /tmp/ (local SSD), preprocesses,
# saves pkl files to SharedFolder. Cleans up /tmp/ after done.
#
# Usage:
#   bash shared/scripts/download_data.sh           # Full pipeline
#   bash shared/scripts/download_data.sh --check   # Check pkl files only
#   bash shared/scripts/download_data.sh --force   # Force re-preprocess
# =============================================================

set -e

BASE_DIR=~/SharedFolder/MDAIE/group6
DATA_DIR=$BASE_DIR/data
HF_CACHE=$BASE_DIR/hf_cache
REPO_DIR=~/spatial-llava
LOG_DIR=$BASE_DIR/logs

# Local SSD paths (fast, lost on container restart)
TMP_ZIP=/tmp/train2014.zip
TMP_COCO=/tmp/coco

MODE="auto"
FORCE_FLAG=""

for arg in "$@"; do
    case $arg in
        --check) MODE="check" ;;
        --force) MODE="force"; FORCE_FLAG="--force" ;;
    esac
done

echo "======================================================"
echo " Spatial-LLaVA — Data Preparation"
echo " pkl output : $DATA_DIR   (SharedFolder, persistent)"
echo " COCO tmp   : $TMP_COCO   (local SSD, fast)"
echo " Mode       : $MODE"
echo " Started    : $(date)"
echo "======================================================"

# ── Check pkl files ────────────────────────────────────────
echo ""
echo "[1/5] Checking pkl files..."

TRAIN_PKL=$DATA_DIR/refcoco_train.pkl
VAL_PKL=$DATA_DIR/refcoco_val.pkl
TEST_PKL=$DATA_DIR/refcoco_test.pkl
STATS_JSON=$DATA_DIR/dataset_stats.json

all_exist=true
for f in "$TRAIN_PKL" "$VAL_PKL" "$TEST_PKL" "$STATS_JSON"; do
    if [[ -f "$f" && $(wc -c < "$f") -gt 100 ]]; then
        size=$(du -sh "$f" | cut -f1)
        echo "  ✅ $(basename $f) ($size)"
    else
        echo "  ❌ $(basename $f) — NOT FOUND or EMPTY"
        all_exist=false
    fi
done

# ── Check only mode ────────────────────────────────────────
if [[ "$MODE" == "check" ]]; then
    echo ""
    if $all_exist; then
        echo "======================================================"
        echo " ✅ All pkl files present. Ready to train."
        echo " Next: bash shared/scripts/start_training.sh main"
        echo "======================================================"
        exit 0
    else
        echo "======================================================"
        echo " ❌ pkl files missing."
        echo " Run: bash shared/scripts/download_data.sh"
        echo "======================================================"
        exit 1
    fi
fi

# ── Already complete, not force ────────────────────────────
if $all_exist && [[ "$MODE" != "force" ]]; then
    echo ""
    echo "======================================================"
    echo " ✅ pkl files already complete. Skipping."
    echo " To re-preprocess: bash shared/scripts/download_data.sh --force"
    echo " Ready to train  : bash shared/scripts/start_training.sh main"
    echo "======================================================"
    exit 0
fi

# ── Clean up old COCO data from SharedFolder ──────────────
echo ""
echo "[2/5] Cleaning up old data from SharedFolder..."
OLD_COCO=$BASE_DIR/coco
if [[ -d "$OLD_COCO" ]]; then
    echo "  Removing $OLD_COCO (zip + extracted images no longer needed)..."
    rm -rf "$OLD_COCO"
    echo "  ✅ Removed $OLD_COCO"
else
    echo "  Nothing to clean up."
fi

# ── Download COCO zip to /tmp/ ─────────────────────────────
echo ""
echo "[3/5] Downloading COCO train2014 to local SSD..."
mkdir -p $TMP_COCO

TRAIN2014_DIR=$TMP_COCO/train2014
N_IMGS=$(ls $TRAIN2014_DIR/*.jpg 2>/dev/null | wc -l)

if [[ $N_IMGS -gt 80000 ]]; then
    echo "  ✅ COCO already extracted in /tmp/: $N_IMGS images"
else
    echo "  Downloading ~13.5GB to $TMP_ZIP ..."
    echo "  (local SSD → fast write, ~26 min at 8MB/s)"
    wget -c http://images.cocodataset.org/zips/train2014.zip \
        -O $TMP_ZIP

    echo ""
    echo "  Extracting to $TMP_COCO ..."
    unzip -q $TMP_ZIP -d $TMP_COCO

    N_IMGS=$(ls $TRAIN2014_DIR/*.jpg 2>/dev/null | wc -l)
    echo "  ✅ Extracted: $N_IMGS images"

    echo "  Removing zip from /tmp/ to free space..."
    rm -f $TMP_ZIP
fi

# ── Preprocess ─────────────────────────────────────────────
echo ""
echo "[4/5] Preprocessing RefCOCO..."
mkdir -p $DATA_DIR $HF_CACHE $LOG_DIR

TIMESTAMP=$(date +%Y%m%d_%H%M)
LOG_FILE=$LOG_DIR/stage1_${TIMESTAMP}.log

echo "  Log: $LOG_FILE"
echo ""

cd $REPO_DIR

python pipeline/stage_1_data_preparation.py \
    --output_dir $DATA_DIR \
    --coco_dir   $TMP_COCO \
    --hf_home    $HF_CACHE \
    --skip_coco_download \
    $FORCE_FLAG \
    2>&1 | tee $LOG_FILE

# ── Clean up /tmp/coco ─────────────────────────────────────
echo ""
echo "[5/5] Cleaning up /tmp/coco..."
rm -rf $TMP_COCO
echo "  ✅ /tmp/coco removed (pkl files saved to SharedFolder)"

# ── Verify ─────────────────────────────────────────────────
echo ""
echo "  Verifying pkl files..."
all_good=true
for f in "$TRAIN_PKL" "$VAL_PKL" "$TEST_PKL" "$STATS_JSON"; do
    if [[ -f "$f" && $(wc -c < "$f") -gt 100 ]]; then
        size=$(du -sh "$f" | cut -f1)
        echo "  ✅ $(basename $f) ($size)"
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
    echo " pkl files saved to: $DATA_DIR"
    echo " (persistent across container restarts)"
    echo ""
    echo " Next steps:"
    echo "   bash shared/scripts/start_training.sh main"
    echo "   bash shared/scripts/start_training.sh ablation"
    echo "======================================================"
    exit 0
else
    echo "======================================================"
    echo " ❌ Something failed. Check log:"
    echo "    $LOG_FILE"
    echo "======================================================"
    exit 1
fi
