#!/bin/bash
# =============================================================
# shared/scripts/setup_cluster.sh
#
# Run after each new login to the SUTD AIMC Edu Cluster.
# Usage:
#   cd /tmp && git clone <repo> spatial-llava && cd spatial-llava
#   source shared/scripts/setup_cluster.sh
#
# Steps:
#   1. Locate repo root (works from any working directory)
#   2. Create all runtime directories inside the repo
#   3. Install dependencies
#   4. Run Stage 0 environment check
# =============================================================

set -e

# ── Locate repo root ───────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "======================================================"
echo " Spatial-LLaVA — Cluster Setup"
echo " Repo root: $REPO_DIR"
echo " Started  : $(date)"
echo "======================================================"

# ── Step 1: Create runtime directories ────────────────────────────────────────
echo ""
echo "[1/3] Creating runtime directories..."

mkdir -p "$REPO_DIR/data/coco/train2014"
mkdir -p "$REPO_DIR/weights"
mkdir -p "$REPO_DIR/checkpoints/main"
mkdir -p "$REPO_DIR/checkpoints/ablation"
mkdir -p "$REPO_DIR/results/baseline"
mkdir -p "$REPO_DIR/results/main"
mkdir -p "$REPO_DIR/results/ablation"
mkdir -p "$REPO_DIR/logs"

echo "  $REPO_DIR/"
echo "  ├── data/coco/train2014/   (COCO images — gitignored)"
echo "  ├── weights/               (HF model weights — gitignored)"
echo "  ├── checkpoints/           (training ckpts — gitignored)"
echo "  ├── results/               (metrics + viz — git tracked)"
echo "  └── logs/                  (training logs — git tracked)"

# ── Step 2: Install dependencies ──────────────────────────────────────────────
echo ""
echo "[2/3] Installing dependencies..."

cd "$REPO_DIR"
pip install -r requirements.txt --upgrade --quiet
echo "  Dependencies installed ✅"

# ── Step 3: Stage 0 environment check ─────────────────────────────────────────
echo ""
echo "[3/3] Running Stage 0 environment check..."
echo ""

python pipeline/stage_0_environment.py

echo ""
echo "======================================================"
echo " Setup complete!"
echo " Next: bash shared/scripts/download_data.sh"
echo "======================================================"
