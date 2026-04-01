#!/bin/bash
# =============================================================
# shared/scripts/setup_cluster.sh
#
# Run this ONCE after first login to the SUTD AIMC Edu Cluster.
# Usage: bash shared/scripts/setup_cluster.sh
# =============================================================

set -e

BASE_DIR=~/SharedFolder/MDAIE/group6

echo "======================================================"
echo " Spatial-LLaVA — Cluster Setup (Group 6)"
echo " Base directory: $BASE_DIR"
echo "======================================================"

# ── Step 1: Persistent environment variables ───────────────
echo ""
echo "[1/5] Setting up environment variables..."

touch ~/.bashrc
if ! grep -q "SPATIAL_BASE" ~/.bashrc; then
    echo "export SPATIAL_BASE=$BASE_DIR" >> ~/.bashrc
    echo "export HF_HOME=$BASE_DIR/hf_cache" >> ~/.bashrc
    echo "export SPATIAL_DATA=$BASE_DIR/data" >> ~/.bashrc
    echo "export SPATIAL_CKPT=$BASE_DIR/checkpoints" >> ~/.bashrc
    echo "export SPATIAL_RESULTS=$BASE_DIR/results" >> ~/.bashrc
    echo "export SPATIAL_LOGS=$BASE_DIR/logs" >> ~/.bashrc
    echo "      Added all SPATIAL_* variables to ~/.bashrc"
else
    echo "      Variables already set in ~/.bashrc, skipping"
fi

export SPATIAL_BASE=$BASE_DIR
export HF_HOME=$BASE_DIR/hf_cache
export SPATIAL_DATA=$BASE_DIR/data
export SPATIAL_CKPT=$BASE_DIR/checkpoints
export SPATIAL_RESULTS=$BASE_DIR/results
export SPATIAL_LOGS=$BASE_DIR/logs

# ── Step 2: Create directory structure ─────────────────────
echo ""
echo "[2/5] Creating SharedFolder directories..."

mkdir -p $BASE_DIR/hf_cache
mkdir -p $BASE_DIR/data
mkdir -p $BASE_DIR/checkpoints/main
mkdir -p $BASE_DIR/checkpoints/ablation
mkdir -p $BASE_DIR/results/main
mkdir -p $BASE_DIR/results/ablation
mkdir -p $BASE_DIR/logs

echo "      $BASE_DIR/"
echo "      ├── hf_cache/             (HuggingFace downloads)"
echo "      ├── data/                 (RefCOCO ~50GB)"
echo "      ├── checkpoints/main/     (LoRA + head, 10 epochs)"
echo "      ├── checkpoints/ablation/ (head only, 3 epochs)"
echo "      ├── results/"
echo "      └── logs/"

# ── Step 3: Install dependencies from requirements.txt ─────
echo ""
echo "[3/5] Installing dependencies..."

pip install -r requirements.txt --upgrade --quiet
echo "      Dependencies installed ✅"

# ── Step 4: Verify GPU access ──────────────────────────────
echo ""
echo "[4/5] Verifying GPU access..."

python3 - <<'EOF'
import torch
if not torch.cuda.is_available():
    print("      ERROR: CUDA not available. Are you on a GPU node?")
    exit(1)
gpu = torch.cuda.get_device_properties(0)
vram_gb = gpu.total_memory / 1e9
print(f"      GPU: {gpu.name}")
print(f"      VRAM: {vram_gb:.1f} GB")
if vram_gb < 15:
    print("      WARNING: VRAM < 16GB, training may OOM.")
else:
    print("      VRAM sufficient ✅")
EOF

# ── Step 5: Verify SharedFolder is writable ────────────────
echo ""
echo "[5/5] Verifying SharedFolder is writable..."
TEST_FILE=$BASE_DIR/.setup_test
touch $TEST_FILE && rm $TEST_FILE
echo "      SharedFolder is writable ✅"

echo ""
echo "======================================================"
echo " Setup complete!"
echo ""
echo " Next: bash shared/scripts/start_training.sh main"
echo " Or:   bash shared/scripts/start_training.sh ablation"
echo ""
echo " NOTE: Run 'source ~/.bashrc' in new sessions."
echo "======================================================"
