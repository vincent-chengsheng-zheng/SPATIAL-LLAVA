#!/bin/bash
# =============================================================
# shared/scripts/download_data.sh
#
# Check if RefCOCO dataset exists, download if not.
#
# Usage:
#   bash shared/scripts/download_data.sh          # Check + download if needed
#   bash shared/scripts/download_data.sh --check  # Check only, no download
#   bash shared/scripts/download_data.sh --force  # Force re-download
# =============================================================

set -e

BASE_DIR=~/SharedFolder/MDAIE/group6
DATA_DIR=$BASE_DIR/data
HF_CACHE=$BASE_DIR/hf_cache
REPO_DIR=~/spatial-llava
LOG_DIR=$BASE_DIR/logs

MODE="auto"
if [[ "$1" == "--check" ]]; then MODE="check"; fi
if [[ "$1" == "--force" ]]; then MODE="force"; fi

echo "======================================================"
echo " Spatial-LLaVA — Data Check & Download"
echo " Data dir : $DATA_DIR"
echo " Mode     : $MODE"
echo " Started  : $(date)"
echo "======================================================"

# ── Check required files ───────────────────────────────────
echo ""
echo "[1/3] Checking dataset files..."

TRAIN_PKL=$DATA_DIR/refcoco_train.pkl
VAL_PKL=$DATA_DIR/refcoco_val.pkl
TEST_PKL=$DATA_DIR/refcoco_test.pkl
STATS_JSON=$DATA_DIR/dataset_stats.json

all_exist=true
for f in "$TRAIN_PKL" "$VAL_PKL" "$TEST_PKL" "$STATS_JSON"; do
    if [[ -f "$f" ]]; then
        size=$(du -sh "$f" | cut -f1)
        echo "  ✅ $(basename $f) ($size)"
    else
        echo "  ❌ $(basename $f) — NOT FOUND"
        all_exist=false
    fi
done

# ── Check only mode ────────────────────────────────────────
if [[ "$MODE" == "check" ]]; then
    echo ""
    if $all_exist; then
        echo "======================================================"
        echo " ✅ All dataset files present."
        echo " Ready to train: bash shared/scripts/start_training.sh main"
        echo "======================================================"
        exit 0
    else
        echo "======================================================"
        echo " ❌ Dataset incomplete."
        echo " Run: bash shared/scripts/download_data.sh"
        echo "======================================================"
        exit 1
    fi
fi

# ── Already exists and not force mode ─────────────────────
if $all_exist && [[ "$MODE" != "force" ]]; then
    echo ""
    echo "======================================================"
    echo " ✅ Dataset already complete. Skipping download."
    echo " To re-download: bash shared/scripts/download_data.sh --force"
    echo " Ready to train: bash shared/scripts/start_training.sh main"
    echo "======================================================"
    exit 0
fi

# ── Download ───────────────────────────────────────────────
echo ""
echo "[2/3] Starting download (~2-3 hours)..."

mkdir -p $DATA_DIR $HF_CACHE $LOG_DIR
TIMESTAMP=$(date +%Y%m%d_%H%M)
LOG_FILE=$LOG_DIR/stage1_${TIMESTAMP}.log

echo "  Log file: $LOG_FILE"
echo "  Monitor:  tail -f $LOG_FILE"
echo ""

cd $REPO_DIR

FORCE_FLAG=""
if [[ "$MODE" == "force" ]]; then FORCE_FLAG="--force"; fi

python pipeline/stage_1_data_preparation.py \
    --output_dir $DATA_DIR \
    --hf_home $HF_CACHE \
    $FORCE_FLAG \
    2>&1 | tee $LOG_FILE

# ── Verify ─────────────────────────────────────────────────
echo ""
echo "[3/3] Verifying downloaded files..."

all_good=true
for f in "$TRAIN_PKL" "$VAL_PKL" "$TEST_PKL" "$STATS_JSON"; do
    if [[ -f "$f" ]]; then
        size=$(du -sh "$f" | cut -f1)
        echo "  ✅ $(basename $f) ($size)"
    else
        echo "  ❌ $(basename $f) — MISSING"
        all_good=false
    fi
done

echo ""
if $all_good; then
    echo "======================================================"
    echo " ✅ Dataset download complete!"
    echo ""
    echo " Next steps:"
    echo "   bash shared/scripts/start_training.sh main"
    echo "   bash shared/scripts/start_training.sh ablation"
    echo "======================================================"
    exit 0
else
    echo "======================================================"
    echo " ❌ Download incomplete. Check log:"
    echo "    $LOG_FILE"
    echo "======================================================"
    exit 1
fi
