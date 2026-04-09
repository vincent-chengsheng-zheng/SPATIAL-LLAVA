#!/bin/bash
# =============================================================
# shared/scripts/download_data.sh
#
# Download COCO images and generate RefCOCO pkl files.
# Everything goes inside the repo under data/ and weights/.
#
# Usage:
#   bash shared/scripts/download_data.sh          # auto
#   bash shared/scripts/download_data.sh --check  # status only
#   bash shared/scripts/download_data.sh --force  # re-run preprocessing
# =============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

COCO_DIR="$REPO_DIR/data/coco"
COCO_TRAIN="$REPO_DIR/data/coco/train2014"
DATA_DIR="$REPO_DIR/data"
WEIGHTS_DIR="$REPO_DIR/weights"
LOG_DIR="$REPO_DIR/logs"
TMP_ZIP="/tmp/train2014.zip"   # zip stays in /tmp to save repo space

COCO_URL="http://images.cocodataset.org/zips/train2014.zip"
COCO_EXPECTED_SIZE=13510573713
COCO_EXPECTED_IMGS=82783

FORCE_FLAG=""
MODE="auto"
for arg in "$@"; do
    case $arg in
        --check) MODE="check" ;;
        --force) FORCE_FLAG="--force"; MODE="force" ;;
    esac
done

echo "======================================================"
echo " Spatial-LLaVA — Data Download"
echo " Repo    : $REPO_DIR"
echo " COCO    : $COCO_TRAIN"
echo " Data    : $DATA_DIR"
echo " Weights : $WEIGHTS_DIR"
echo " Mode    : $MODE"
echo " Started : $(date)"
echo "======================================================"

count_imgs() { find "$COCO_TRAIN" -maxdepth 1 -name "*.jpg" 2>/dev/null | wc -l; }

# ── Status check ──────────────────────────────────────────────────────────────
echo ""
echo "── Current Status ────────────────────────────────────"

ZIP_SIZE=$(stat -c%s $TMP_ZIP 2>/dev/null || echo 0)
if [[ $ZIP_SIZE -eq $COCO_EXPECTED_SIZE ]]; then
    echo "  ✅ zip  : $TMP_ZIP ($(du -sh $TMP_ZIP | cut -f1))"
elif [[ $ZIP_SIZE -gt 0 ]]; then
    echo "  ⚠  zip  : partial ($(du -sh $TMP_ZIP | cut -f1) / 13.5GB)"
else
    echo "  ❌ zip  : not found"
fi

N_IMGS=$(count_imgs)
if [[ $N_IMGS -ge $COCO_EXPECTED_IMGS ]]; then
    echo "  ✅ imgs : $N_IMGS images"
elif [[ $N_IMGS -gt 0 ]]; then
    echo "  ⚠  imgs : $N_IMGS / $COCO_EXPECTED_IMGS extracted"
else
    echo "  ❌ imgs : not extracted"
fi

TRAIN_PKL="$DATA_DIR/refcoco_train.pkl"
PKL_OK=false
if [[ -f "$TRAIN_PKL" && $(wc -c < "$TRAIN_PKL") -gt 100 ]]; then
    echo "  ✅ pkl  : $(du -sh $DATA_DIR/*.pkl 2>/dev/null | tr '\n' ' ')"
    PKL_OK=true
else
    echo "  ❌ pkl  : not found in $DATA_DIR"
fi

if [[ "$MODE" == "check" ]]; then
    echo ""
    $PKL_OK && echo "✅ Ready to train!" && exit 0 || echo "❌ Not ready." && exit 1
fi

if $PKL_OK && [[ "$MODE" != "force" ]]; then
    echo ""
    echo "✅ pkl files already exist. Nothing to do."
    echo "Next: python pipeline/step2_train_main.py"
    exit 0
fi

# ── Download COCO zip ─────────────────────────────────────────────────────────
echo ""
echo "[1/3] COCO zip download..."
mkdir -p "$COCO_TRAIN"

if [[ $N_IMGS -ge $COCO_EXPECTED_IMGS ]]; then
    echo "  ✅ Images already extracted, skipping download."
elif [[ $ZIP_SIZE -eq $COCO_EXPECTED_SIZE ]]; then
    echo "  ✅ Zip already complete, skipping download."
else
    echo "  Downloading to $TMP_ZIP (~13.5GB)..."
    wget -c $COCO_URL -O $TMP_ZIP
    echo "  ✅ Download complete."
fi

# ── Extract ───────────────────────────────────────────────────────────────────
echo ""
echo "[2/3] Extracting COCO images..."

N_IMGS=$(count_imgs)
if [[ $N_IMGS -ge $COCO_EXPECTED_IMGS ]]; then
    echo "  ✅ Already extracted: $N_IMGS images."
else
    python3 - << EOF
import zipfile, os

zip_path = "/tmp/train2014.zip"
dest_dir = "$COCO_DIR"
train_dir = "$COCO_TRAIN"
os.makedirs(train_dir, exist_ok=True)

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
            print(f"  [{already + extracted:,}/{total:,}] extracted...")

done = len([f for f in os.listdir(train_dir) if f.endswith(".jpg")])
print(f"  Done: {done:,} images")
EOF

    echo "  Removing zip from /tmp/ ..."
    rm -f $TMP_ZIP
fi

# ── Preprocessing ─────────────────────────────────────────────────────────────
echo ""
echo "[3/3] Preprocessing RefCOCO → pkl..."
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M)
LOG_FILE="$LOG_DIR/stage1_${TIMESTAMP}.log"

cd "$REPO_DIR"
python pipeline/stage_1_data_preparation.py \
    --output_dir "$DATA_DIR" \
    --coco_dir   "$COCO_DIR" \
    --skip_coco_download \
    $FORCE_FLAG \
    2>&1 | tee "$LOG_FILE"

# ── Final check ───────────────────────────────────────────────────────────────
echo ""
echo "── Final Verification ────────────────────────────────"
all_good=true
for f in "$DATA_DIR/refcoco_train.pkl" \
         "$DATA_DIR/refcoco_val.pkl" \
         "$DATA_DIR/refcoco_test.pkl" \
         "$DATA_DIR/dataset_stats.json"; do
    if [[ -f "$f" && $(wc -c < "$f") -gt 100 ]]; then
        echo "  ✅ $(basename $f) ($(du -sh $f | cut -f1))"
    else
        echo "  ❌ $(basename $f) — MISSING"
        all_good=false
    fi
done

echo ""
if $all_good; then
    echo "======================================================"
    echo " ✅ Data ready in $DATA_DIR"
    echo " Next: python pipeline/step2_train_main.py"
    echo "======================================================"
else
    echo "❌ Something failed. Log: $LOG_FILE"
    exit 1
fi
