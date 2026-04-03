#!/bin/bash
# =============================================================
# shared/scripts/download_data.sh
#
# Downloads RefCOCO annotations + COCO images, then preprocesses.
#
# Usage:
#   bash shared/scripts/download_data.sh                  # Full pipeline
#   bash shared/scripts/download_data.sh --check          # Check only
#   bash shared/scripts/download_data.sh --force          # Force re-download
#   bash shared/scripts/download_data.sh --skip_coco      # Skip COCO download
#   bash shared/scripts/download_data.sh --extract_only   # Extract zip only
# =============================================================

set -e

BASE_DIR=~/SharedFolder/MDAIE/group6
DATA_DIR=$BASE_DIR/data
COCO_DIR=$BASE_DIR/coco
HF_CACHE=$BASE_DIR/hf_cache
REPO_DIR=~/spatial-llava
LOG_DIR=$BASE_DIR/logs

MODE="auto"
FORCE_FLAG=""
SKIP_COCO=""
EXTRACT_ONLY=false

for arg in "$@"; do
    case $arg in
        --check)        MODE="check" ;;
        --force)        MODE="force"; FORCE_FLAG="--force" ;;
        --skip_coco)    SKIP_COCO="--skip_coco_download" ;;
        --extract_only) EXTRACT_ONLY=true ;;
    esac
done

echo "======================================================"
echo " Spatial-LLaVA — Data Download & Preparation"
echo " Data dir : $DATA_DIR"
echo " COCO dir : $COCO_DIR"
echo " Mode     : $MODE"
echo " Started  : $(date)"
echo "======================================================"

# ── Extract only mode ──────────────────────────────────────
if $EXTRACT_ONLY; then
    ZIP_PATH=$COCO_DIR/train2014.zip
    echo ""
    echo "[extract_only] Extracting $ZIP_PATH ..."

    if [[ ! -f "$ZIP_PATH" ]]; then
        echo "  ❌ Zip not found: $ZIP_PATH"
        exit 1
    fi

    python3 - << EOF
import zipfile, os

zip_path = "$ZIP_PATH"
dest_dir = "$COCO_DIR"

print(f"  Zip size: {os.path.getsize(zip_path) / 1e9:.1f} GB")
with zipfile.ZipFile(zip_path, "r") as zf:
    members = zf.namelist()
    total = len(members)
    print(f"  Total files: {total:,}")
    for i, member in enumerate(members):
        zf.extract(member, dest_dir)
        if (i + 1) % 10000 == 0 or (i + 1) == total:
            print(f"  [{i+1:,}/{total:,}] extracted...")
print("  ✅ Extraction complete!")
EOF

    echo ""
    N=$(ls $COCO_DIR/train2014/*.jpg 2>/dev/null | wc -l)
    echo "  Images extracted: $N"
    echo ""
    echo "  Next: bash shared/scripts/download_data.sh --skip_coco"
    exit 0
fi

# ── Check required files ───────────────────────────────────
echo ""
echo "[1/4] Checking dataset files..."

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

# ── Check COCO images ──────────────────────────────────────
echo ""
echo "[2/4] Checking COCO images..."

TRAIN2014_DIR=$COCO_DIR/train2014
coco_ok=false
if [[ -d "$TRAIN2014_DIR" ]]; then
    N=$(ls $TRAIN2014_DIR/*.jpg 2>/dev/null | wc -l)
    if [[ $N -gt 80000 ]]; then
        echo "  ✅ COCO train2014: $N images"
        coco_ok=true
    else
        echo "  ⚠  COCO train2014 exists but only $N images (expected ~83k)"
    fi
else
    ZIP_PATH=$COCO_DIR/train2014.zip
    if [[ -f "$ZIP_PATH" ]]; then
        ZIP_SIZE=$(wc -c < "$ZIP_PATH")
        echo "  ⚠  train2014/ not extracted yet"
        echo "     Zip found: $(du -sh $ZIP_PATH | cut -f1)"
        echo "     Run: bash shared/scripts/download_data.sh --extract_only"
    else
        echo "  ❌ COCO train2014 not found: $TRAIN2014_DIR"
    fi
fi

# ── Check only mode ────────────────────────────────────────
if [[ "$MODE" == "check" ]]; then
    echo ""
    if $all_exist && $coco_ok; then
        echo "======================================================"
        echo " ✅ All files present. Ready to train."
        echo " Next: bash shared/scripts/start_training.sh main"
        echo "======================================================"
        exit 0
    else
        echo "======================================================"
        echo " ❌ Some files missing."
        if ! $coco_ok && [[ -f "$COCO_DIR/train2014.zip" ]]; then
            echo " Run: bash shared/scripts/download_data.sh --extract_only"
        else
            echo " Run: bash shared/scripts/download_data.sh"
        fi
        echo "======================================================"
        exit 1
    fi
fi

# ── Already complete, not force ────────────────────────────
if $all_exist && [[ "$MODE" != "force" ]]; then
    echo ""
    echo "======================================================"
    echo " ✅ Dataset already complete. Skipping."
    echo " To re-download: bash shared/scripts/download_data.sh --force"
    echo " Ready to train: bash shared/scripts/start_training.sh main"
    echo "======================================================"
    exit 0
fi

# ── Run stage_1 ────────────────────────────────────────────
echo ""
echo "[3/4] Running stage_1_data_preparation.py..."
mkdir -p $DATA_DIR $HF_CACHE $COCO_DIR $LOG_DIR

TIMESTAMP=$(date +%Y%m%d_%H%M)
LOG_FILE=$LOG_DIR/stage1_${TIMESTAMP}.log

echo "  Log: $LOG_FILE"
echo ""

cd $REPO_DIR

python pipeline/stage_1_data_preparation.py \
    --output_dir $DATA_DIR \
    --coco_dir   $COCO_DIR \
    --hf_home    $HF_CACHE \
    $FORCE_FLAG \
    $SKIP_COCO \
    2>&1 | tee $LOG_FILE

# ── Verify ─────────────────────────────────────────────────
echo ""
echo "[4/4] Verifying downloaded files..."

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
    echo " Next steps:"
    echo "   bash shared/scripts/start_training.sh main"
    echo "   bash shared/scripts/start_training.sh ablation"
    echo "======================================================"
    exit 0
else
    echo "======================================================"
    echo " ❌ Data preparation failed. Check log:"
    echo "    $LOG_FILE"
    echo "======================================================"
    exit 1
fi
