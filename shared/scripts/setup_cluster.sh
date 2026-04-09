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
#   3. Install dependencies (with progress)
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

echo "  ✅ Directories ready"

# ── Step 2: Install dependencies ──────────────────────────────────────────────
echo ""
echo "[2/3] Installing dependencies (this may take 3-5 min)..."
echo "      Showing latest package being installed:"
echo ""

cd "$REPO_DIR"

# --progress-bar off keeps output clean; grep shows one line per package
pip install -r requirements.txt --upgrade \
    --progress-bar off \
    2>&1 | grep -E "^(Collecting|Downloading|Installing|Successfully|already)" \
         | sed 's/^/  /'

echo ""
echo "  ✅ Dependencies installed"

# ── Step 3: Stage 0 environment check ─────────────────────────────────────────
echo ""
echo "[3/3] Running Stage 0 environment check..."
echo "      (imports torch + checks GPU, ~30s first time)"
echo ""

python pipeline/stage_0_environment.py

echo ""
echo "======================================================"
echo " ✅ Setup complete!"
echo " Next: bash shared/scripts/download_data.sh"
echo "======================================================"