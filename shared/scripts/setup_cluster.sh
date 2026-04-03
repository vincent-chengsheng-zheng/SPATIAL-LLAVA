#!/bin/bash
# =============================================================
# shared/scripts/setup_cluster.sh
#
# Run after each new login to the SUTD AIMC Edu Cluster.
# Usage: source shared/scripts/setup_cluster.sh
#
# Steps:
#   1. Set environment variables
#   2. Create SharedFolder directories
#   3. Install dependencies
#   4. Run Stage 0 environment check
# =============================================================

set -e

BASE_DIR=~/SharedFolder/MDAIE/group6
REPO_DIR=~/spatial-llava

echo "======================================================"
echo " Spatial-LLaVA — Cluster Setup (Group 6)"
echo " Base directory: $BASE_DIR"
echo "======================================================"

# ── Step 1: Persistent environment variables ───────────────
echo ""
echo "[1/4] Setting up environment variables..."

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
echo "[2/4] Creating SharedFolder directories..."

mkdir -p $BASE_DIR/hf_cache
mkdir -p $BASE_DIR/checkpoints/main
mkdir -p $BASE_DIR/checkpoints/ablation
mkdir -p $BASE_DIR/results/main
mkdir -p $BASE_DIR/results/ablation
mkdir -p $BASE_DIR/logs

echo "      $BASE_DIR/"
echo "      ├── hf_cache/             (HuggingFace cache)"
echo "      ├── checkpoints/main/     (LoRA + head, 10 epochs)"
echo "      ├── checkpoints/ablation/ (head only, 3 epochs)"
echo "      ├── results/"
echo "      └── logs/"

# ── Step 3: Install dependencies ───────────────────────────
echo ""
echo "[3/4] Installing dependencies..."

cd $REPO_DIR
pip install -r requirements.txt --upgrade --quiet
echo "      Dependencies installed ✅"

# ── Step 4: Stage 0 environment check ─────────────────────
echo ""
echo "[4/4] Running Stage 0 environment check..."
echo ""

python pipeline/stage_0_environment.py

# ── Done ───────────────────────────────────────────────────
# Stage 0 will print pass/fail and next steps
