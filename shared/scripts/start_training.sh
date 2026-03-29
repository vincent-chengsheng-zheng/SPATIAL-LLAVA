#!/bin/bash
# =============================================================
# shared/scripts/start_training.sh
#
# One-command training launcher for Member 3 and Member 4.
# Handles: env setup, data check, Stage 1 if needed, training.
#
# Usage:
#   bash shared/scripts/start_training.sh main      # LoRA + head (10 epochs)
#   bash shared/scripts/start_training.sh ablation  # head only  (3 epochs)
# =============================================================

set -e

MODE=${1:-""}
BASE_DIR=~/SharedFolder/MDAIE/group6
REPO_DIR=~/spatial-llava

# ── Validate argument ──────────────────────────────────────
if [[ "$MODE" != "main" && "$MODE" != "ablation" ]]; then
    echo ""
    echo "Usage: bash shared/scripts/start_training.sh [main|ablation]"
    echo ""
    echo "  main      → Full model: LoRA + regression head (10 epochs)"
    echo "              Used by: Member 4 (61.502 Enterprise)"
    echo "              Checkpoint: $BASE_DIR/checkpoints/main/"
    echo ""
    echo "  ablation  → Head only: regression head, LLM frozen (3 epochs)"
    echo "              Used by: Member 3 (51.511 GenAI ablation study)"
    echo "              Checkpoint: $BASE_DIR/checkpoints/ablation/"
    echo ""
    exit 1
fi

echo "======================================================"
echo " Spatial-LLaVA Training Launcher"
echo " Mode: $MODE"
echo " Base: $BASE_DIR"
echo "======================================================"

# ── Step 1: Environment variables ─────────────────────────
echo ""
echo "[1/5] Setting up environment..."

export SPATIAL_BASE=$BASE_DIR
export HF_HOME=$BASE_DIR/hf_cache
export SPATIAL_DATA=$BASE_DIR/data
export SPATIAL_CKPT=$BASE_DIR/checkpoints
export SPATIAL_RESULTS=$BASE_DIR/results
export SPATIAL_LOGS=$BASE_DIR/logs

mkdir -p $SPATIAL_LOGS

echo "      HF_HOME    = $HF_HOME"
echo "      SPATIAL_DATA = $SPATIAL_DATA"
echo "      SPATIAL_CKPT = $SPATIAL_CKPT"

# ── Step 2: Pull latest code ───────────────────────────────
echo ""
echo "[2/5] Pulling latest code from GitHub..."

cd $REPO_DIR
git pull
echo "      Code is up to date ✅"

# ── Step 3: Run Stage 0 environment check ─────────────────
echo ""
echo "[3/5] Running Stage 0 environment check..."

if ! python pipeline/stage_0_environment.py; then
    echo ""
    echo "❌ Stage 0 failed. Fix the issues above before training."
    exit 1
fi
echo "      Environment check passed ✅"

# ── Step 4: Check data, run Stage 1 if needed ─────────────
echo ""
echo "[4/5] Checking dataset..."

TRAIN_PKL=$SPATIAL_DATA/refcoco_train.pkl
VAL_PKL=$SPATIAL_DATA/refcoco_val.pkl
TEST_PKL=$SPATIAL_DATA/refcoco_test.pkl

if [[ -f "$TRAIN_PKL" && -f "$VAL_PKL" && -f "$TEST_PKL" ]]; then
    echo "      Dataset found ✅"
    echo "      $TRAIN_PKL"
    echo "      $VAL_PKL"
    echo "      $TEST_PKL"
else
    echo "      Dataset not found. Running Stage 1 to download RefCOCO..."
    echo "      This will take 2-3 hours. Logs: $SPATIAL_LOGS/stage1.log"
    echo ""

    nohup python pipeline/stage_1_data_preparation.py \
        --output_dir $SPATIAL_DATA \
        --hf_home $HF_HOME \
        > $SPATIAL_LOGS/stage1.log 2>&1

    # Wait for it to finish (stage1 runs synchronously here)
    if [[ -f "$TRAIN_PKL" && -f "$VAL_PKL" && -f "$TEST_PKL" ]]; then
        echo "      Dataset ready ✅"
    else
        echo "❌ Stage 1 failed. Check: $SPATIAL_LOGS/stage1.log"
        exit 1
    fi
fi

# ── Step 5: Launch training ────────────────────────────────
echo ""
echo "[5/5] Launching $MODE training..."

TIMESTAMP=$(date +%Y%m%d_%H%M)
LOG_FILE=$SPATIAL_LOGS/train_${MODE}_${TIMESTAMP}.log

if [[ "$MODE" == "main" ]]; then
    TRAIN_CMD="python courses/shared/train.py \
        --mode main \
        --config courses/shared/config.yaml \
        --data_dir $SPATIAL_DATA \
        --output_dir $SPATIAL_CKPT/main"
else
    TRAIN_CMD="python courses/shared/train.py \
        --mode ablation \
        --config courses/shared/config.yaml \
        --data_dir $SPATIAL_DATA \
        --output_dir $SPATIAL_CKPT/ablation"
fi

echo "      Log file: $LOG_FILE"
echo "      Monitor: tail -f $LOG_FILE"
echo ""

# Use tmux if available, otherwise nohup
if command -v tmux &> /dev/null; then
    SESSION="train_${MODE}_${TIMESTAMP}"
    tmux new-session -d -s $SESSION \
        "nohup $TRAIN_CMD > $LOG_FILE 2>&1; echo 'Training finished' >> $LOG_FILE"
    echo "      Started in tmux session: $SESSION"
    echo "      Attach with: tmux attach -t $SESSION"
else
    nohup $TRAIN_CMD > $LOG_FILE 2>&1 &
    echo "      Started with PID: $!"
    echo "      Monitor with: tail -f $LOG_FILE"
fi

echo ""
echo "======================================================"
echo " Training launched! ($MODE mode)"
echo ""
echo " Monitor progress:"
echo "   tail -f $LOG_FILE"
echo ""
echo " Checkpoints saved to:"
echo "   $SPATIAL_CKPT/$MODE/"
echo "======================================================"

