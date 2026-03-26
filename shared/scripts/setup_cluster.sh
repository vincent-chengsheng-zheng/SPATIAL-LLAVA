#!/bin/bash
# =============================================================
# setup_cluster.sh
# Run this ONCE after first login to the SUTD AIMC Edu Cluster.
# Usage: bash shared/scripts/setup_cluster.sh
# =============================================================

set -e  # exit immediately on any error

echo "======================================================"
echo " Spatial-LLaVA — Cluster Setup"
echo "======================================================"

# ── Step 1: Persistent environment variables ───────────────
echo ""
echo "[1/5] Setting up environment variables..."

if ! grep -q "HF_HOME" ~/.bashrc; then
    echo 'export HF_HOME=~/SharedFolder/hf_cache' >> ~/.bashrc
    echo "      Added HF_HOME to ~/.bashrc"
else
    echo "      HF_HOME already set in ~/.bashrc, skipping"
fi

if ! grep -q "SPATIAL_LLAVA" ~/.bashrc; then
    echo 'export SPATIAL_DATA=~/SharedFolder/data' >> ~/.bashrc
    echo 'export SPATIAL_CKPT=~/SharedFolder/checkpoints' >> ~/.bashrc
    echo 'export SPATIAL_RESULTS=~/SharedFolder/results' >> ~/.bashrc
    echo 'export SPATIAL_LOGS=~/SharedFolder/logs' >> ~/.bashrc
    echo "      Added SPATIAL_* paths to ~/.bashrc"
fi

# Apply immediately for this session
export HF_HOME=~/SharedFolder/hf_cache
export SPATIAL_DATA=~/SharedFolder/data
export SPATIAL_CKPT=~/SharedFolder/checkpoints
export SPATIAL_RESULTS=~/SharedFolder/results
export SPATIAL_LOGS=~/SharedFolder/logs

# ── Step 2: Create SharedFolder directory structure ────────
echo ""
echo "[2/5] Creating SharedFolder directories..."

mkdir -p ~/SharedFolder/hf_cache
mkdir -p ~/SharedFolder/data
mkdir -p ~/SharedFolder/checkpoints/coursework
mkdir -p ~/SharedFolder/checkpoints/enterprise
mkdir -p ~/SharedFolder/results/coursework
mkdir -p ~/SharedFolder/results/enterprise
mkdir -p ~/SharedFolder/logs

echo "      SharedFolder structure created:"
echo "      ~/SharedFolder/"
echo "      ├── hf_cache/           (HuggingFace model downloads)"
echo "      ├── data/               (RefCOCO dataset ~50GB)"
echo "      ├── checkpoints/"
echo "      │   ├── coursework/     (3-epoch model weights)"
echo "      │   └── enterprise/     (10-epoch model weights)"
echo "      ├── results/"
echo "      │   ├── coursework/"
echo "      │   └── enterprise/"
echo "      └── logs/               (training logs)"

# ── Step 3: Install extra Python dependencies ──────────────
echo ""
echo "[3/5] Installing extra Python dependencies..."
echo "      (PyTorch + CUDA already pre-installed on cluster)"

pip install --quiet \
    peft==0.7.1 \
    transformers==4.35.2 \
    bitsandbytes==0.41.1 \
    gradio==4.17.0 \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    python-multipart==0.0.6 \
    pyyaml==6.0.1 \
    tqdm==4.66.1 \
    opencv-python-headless==4.8.1.78 \
    Pillow==10.1.0

echo "      Dependencies installed"

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
print(f"      GPU detected: {gpu.name}")
print(f"      VRAM: {vram_gb:.1f} GB")

if vram_gb < 15:
    print("      WARNING: VRAM < 16GB, training may OOM. Consider requesting a larger node.")
else:
    print("      VRAM sufficient for training ✅")
EOF

# ── Step 5: Verify SharedFolder is writable ────────────────
echo ""
echo "[5/5] Verifying SharedFolder is writable..."

TEST_FILE=~/SharedFolder/.setup_test
touch $TEST_FILE && rm $TEST_FILE
echo "      SharedFolder is writable ✅"

# ── Done ───────────────────────────────────────────────────
echo ""
echo "======================================================"
echo " Setup complete!"
echo ""
echo " Next steps:"
echo "   1. Run environment check:"
echo "      python pipeline/stage_0_environment.py"
echo ""
echo "   2. Download dataset (Member 2):"
echo "      python pipeline/stage_1_data_preparation.py \\"
echo "          --output_dir ~/SharedFolder/data/"
echo ""
echo " NOTE: Re-login or run 'source ~/.bashrc' to apply"
echo "       environment variables in new sessions."
echo "======================================================"