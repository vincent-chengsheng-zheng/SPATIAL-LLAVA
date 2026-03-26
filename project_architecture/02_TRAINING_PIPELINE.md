# 02_TRAINING_PIPELINE.md

## Complete Training Pipeline: 5 Stages

**Read time:** 45-60 minutes
**Audience:** All team members (read your section at minimum)
**Next document:** 03_DOCKER_STRATEGY.md

---

## Overview: What Are We Building?

```
Goal: Train Spatial-LLaVA model on RefCOCO dataset

Input: Image + Text prompt ("find the person on the left")
Output: Bounding box coordinates [x, y, w, h]

Architecture:
  Image ─────┐
             ├─→ LLaVA Vision Encoder (frozen) ─────┐
  Text ──────┤                                       ├─→ LLM + LoRA (trainable)
             │                                       │
             │   Generate [LOC] token ───────────────┤
             │                                       │
             │   Extract hidden state ─────────────→ MLP Regression Head (trainable)
             │                                       │
             └───────────────────────────────────────┘
                                                     ↓
                                          Output: [x, y, w, h]
                                                     ↓
                                          Compare with ground truth
                                                     ↓
                                          Calculate loss & backprop
```

---

## Compute Environment

Training runs on the **SUTD AI Mega Centre (AIMC) Education GPU Cluster**, NOT locally.

```
Local Mac (thin client)
  └─→ Browser / SSH
      └─→ JupyterHub (login2.gpucluster.sutd.edu.sg:8888)
          └─→ Kubernetes allocates:
              └─→ V100 GPU node (16/32GB VRAM, 4 vCPU, 24GB RAM)
```

**Key cluster facts:**
- GPU: NVIDIA Tesla V100, 16GB or 32GB VRAM
- PyTorch, CUDA 11.8/12+ already pre-installed — no Docker needed for training
- Only HOME directory is persistent across sessions (100GB)
- Large files (dataset, checkpoints, model weights) → `~/SharedFolder` (1TB+)
- Idle instances terminated after 2 hours → use `tmux` or `nohup`

**HuggingFace cache must be redirected** to SharedFolder to avoid filling HOME:
```bash
export HF_HOME=~/SharedFolder/hf_cache
```

---

## Storage Layout on Cluster

```
~/spatial-llava/                    ← git clone here (HOME, persistent)
~/SharedFolder/
    ├── hf_cache/                   ← HuggingFace model downloads (~14GB)
    ├── data/
    │   ├── refcoco_train.pkl       ← ~30GB
    │   ├── refcoco_val.pkl         ← ~4GB
    │   ├── refcoco_test.pkl        ← ~4GB
    │   └── dataset_stats.json
    ├── checkpoints/
    │   ├── coursework/             ← LoRA weights + regression head
    │   └── enterprise/
    ├── results/
    │   ├── coursework/
    │   └── enterprise/
    └── logs/                       ← Training logs (nohup output)
```

**Never put large files in:**
- `~/spatial-llava/data/`  (don't commit to git)
- `~/spatial-llava/checkpoints/` (too large for HOME)

---

## Stage Overview

| Stage | Duration | Purpose | Input | Output | Responsible |
|-------|----------|---------|-------|--------|-------------|
| **Stage 0** | 10 min | Check cluster environment | None | ✅ GPU ready | Member 1 |
| **Stage 1** | 2-3 hrs | Prepare data | Network | 📦 Pickle files in SharedFolder | Member 2 |
| **Stage 2** | 10 min | Init model | HuggingFace | 🤖 Model verified | Member 1 |
| **Stage 3** | 8-12 hrs | Train model | Data + Model | 💾 Checkpoints in SharedFolder | Members 3 & 4 |
| **Stage 4** | 1-2 hrs | Evaluate | Trained model | 📊 Metrics JSON | Members 3 & 4 |
| **Stage 5** | 30-60 min | Demo / Deploy | Evaluated model | 🚀 Gradio or FastAPI | Members 3 & 4 |

---

## Engineering Directory Structure

```
spatial-llava/                          ← git clone to ~/spatial-llava on cluster
│
├── pipeline/                           # Training pipeline scripts (stages 0-2)
│   ├── __init__.py
│   ├── stage_0_environment.py          # Check GPU, CUDA, SharedFolder access
│   ├── stage_1_data_preparation.py     # Download & preprocess RefCOCO
│   ├── stage_2_model_initialization.py # Verify LLaVA + LoRA + regression head loads
│   └── pipeline_config.yaml            # Shared config (paths point to SharedFolder)
│
├── core/                               # Reusable modules
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── spatial_llava.py            # LLaVA + [LOC] token + regression head
│   │   ├── regression_head.py          # MLP: hidden_state (4096) → [x, y, w, h]
│   │   └── lora_config.py              # LoRA rank, target_modules, alpha
│   ├── data/
│   │   ├── __init__.py
│   │   ├── refcoco_loader.py           # RefCOCO dataset loading from SharedFolder
│   │   ├── preprocessing.py            # Image resize (384x384), tokenization
│   │   └── data_utils.py               # Bbox normalization, [LOC] token handling
│   ├── loss/
│   │   ├── __init__.py
│   │   └── spatial_loss.py             # MSE regression loss + alignment loss
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py                  # IoU, RMSE, MAE
│       ├── checkpoint.py               # save/load checkpoint (resume support)
│       └── visualization.py            # Draw bounding boxes on images
│
├── courses/
│   │
│   ├── coursework_track/               # 51.511 Multimodal GenAI track
│   │   ├── __init__.py                 # Template 2: Generative Model Research Study
│   │   ├── config_coursework.yaml      # 3 epochs, batch=4, fp16=true
│   │   ├── train_coursework.py         # Stage 3a: training with resume support
│   │   ├── eval_coursework.py          # Stage 4a: 3-way ablation comparison (REQUIRED)
│   │   └── demo_gradio.py              # Stage 5a: Gradio UI showing 3 models side-by-side
│   │
│   └── enterprise_track/               # 61.502 Deep Learning for Enterprise track
│       ├── __init__.py
│       ├── config_enterprise.yaml      # 10 epochs, batch=8, fp16=true
│       ├── train_enterprise.py         # Stage 3b: training with resume support
│       ├── eval_enterprise.py          # Stage 4b: baseline comparison + benchmarks
│       └── deployment/
│           ├── __init__.py
│           ├── fastapi_server.py       # Stage 5b: REST API
│           └── requirements_prod.txt   # Lightweight inference dependencies
│
├── infrastructure/
│   └── docker/
│       ├── Dockerfile.dev              # Mac local dev (CPU-only, python:3.10-slim)
│       ├── Dockerfile.inference        # Production FastAPI deployment
│       ├── docker-compose.yml          # Local dev orchestration
│       └── .dockerignore
│
├── .github/
│   └── workflows/
│       └── pr_checks.yml               # Lint + unit tests + Docker build check
│
├── tests/
│   ├── __init__.py
│   ├── test_model.py                   # Model forward pass (CPU, no GPU needed)
│   ├── test_data_loader.py             # Data pipeline smoke test
│   └── test_metrics.py                 # IoU, RMSE calculation correctness
│
├── shared/
│   ├── scripts/
│   │   ├── setup_cluster.sh            # First-time cluster setup (export HF_HOME, mkdir)
│   │   └── download_refcoco.py         # Download dataset → SharedFolder
│   └── notebooks/
│       ├── 01_data_exploration.ipynb
│       └── 02_results_analysis.ipynb
│
├── docker-compose.yml                  # Root-level (single source of truth)
├── requirements.txt                    # Full deps (used in Docker + cluster pip install)
├── requirements_inference.txt          # Lightweight (FastAPI only)
├── .gitignore                          # Must exclude: data/, checkpoints/, *.pth, *.pkl
└── README.md
```

---

## Stage 0: Environment Check

**Duration:** 10 minutes
**Member:** 1
**Dependency:** None — run this first thing on the cluster

### Goal
Confirm the cluster node has GPU access, SharedFolder is mounted, and all imports work.

```python
# pipeline/stage_0_environment.py

import os, torch, shutil

checks = {
    "Python 3.10+":        sys.version_info >= (3, 10),
    "CUDA available":      torch.cuda.is_available(),
    "GPU VRAM >= 16GB":    torch.cuda.get_device_properties(0).total_memory >= 16e9,
    "SharedFolder exists": os.path.isdir(os.path.expanduser("~/SharedFolder")),
    "HF_HOME set":         "HF_HOME" in os.environ,
    "SharedFolder space":  shutil.disk_usage("~/SharedFolder").free >= 100e9,
}
```

### Expected Output on Cluster

```
✓ Python version: 3.10.x
✓ CUDA available: True
✓ GPU: NVIDIA Tesla V100-SXM2-32GB, 32GB VRAM
✓ SharedFolder mounted and accessible
✓ HF_HOME = ~/SharedFolder/hf_cache
✓ SharedFolder free space: ~900GB
✓ Environment check passed ✅
```

### Verification Command

```bash
export HF_HOME=~/SharedFolder/hf_cache
python pipeline/stage_0_environment.py
```

### Acceptance Criteria

- [ ] CUDA available, V100 detected
- [ ] VRAM ≥ 16GB
- [ ] `~/SharedFolder` accessible
- [ ] `HF_HOME` exported before running
- [ ] All imports (`torch`, `transformers`, `peft`) succeed

---

## Stage 1: Data Preparation

**Duration:** 2-3 hours
**Member:** 2
**Dependency:** Stage 0 complete, SharedFolder accessible

### Goal
Download RefCOCO, preprocess, save splits to `~/SharedFolder/data/`.

```python
# pipeline/stage_1_data_preparation.py

# All output goes to SharedFolder, NOT the repo directory
OUTPUT_DIR = os.path.expanduser("~/SharedFolder/data/")

steps:
1. Download RefCOCO from HuggingFace datasets (~50GB raw)
2. Verify checksum
3. Preprocess images → resize 384x384, normalize
4. Tokenize text with LLaVA tokenizer, handle [LOC] token
5. Normalize bbox → [0,1] range, format [x_center, y_center, w, h]
6. Split: train 80% / val 10% / test 10%
7. Save: refcoco_train.pkl, refcoco_val.pkl, refcoco_test.pkl
8. Save: dataset_stats.json
```

### Verification Command

```bash
export HF_HOME=~/SharedFolder/hf_cache
python pipeline/stage_1_data_preparation.py \
    --output_dir ~/SharedFolder/data/
```

### Acceptance Criteria

- [ ] 3 pickle files exist in `~/SharedFolder/data/`
- [ ] Total ~40-50GB
- [ ] 120,000 samples total
- [ ] Files loadable in Python

---

## Stage 2: Model Initialization

**Duration:** 10 minutes (plus ~30 min first-time download)
**Member:** 1
**Dependency:** Stage 0 complete, HF_HOME set

### Goal
Verify that LLaVA-7B loads correctly with LoRA and regression head attached.
This stage is a **verification step** — actual training happens in Stage 3.

### Architecture After Init

```
LLaVA-7B (liuhaotian/llava-v1.5-7b):
├── Vision Encoder (CLIP-ViT-L/14)     [FROZEN — 304M params]
├── Projection Layer                    [FROZEN]
├── LLM (Vicuna-7B)                     [BASE FROZEN]
│   └── LoRA adapters on q_proj, v_proj [TRAINABLE — ~4.7M params]
└── MLP Regression Head (NEW)           [TRAINABLE — ~0.5M params]
    └── 4096 → 512 → ReLU → 4
    └── Sigmoid output → [x, y, w, h] in [0,1]

Special token [LOC] added to vocabulary.
Hidden state at [LOC] position → input to regression head.

Total trainable: ~5.2M / 7B (0.07%)
```

### Expected GPU Memory (V100)

```
Model weights (fp16):    ~7.0 GB
LoRA adapters:           ~0.1 GB
Batch size 4 activations: ~1.0 GB
Optimizer state:          ~0.5 GB
─────────────────────────────────
Total estimate:          ~8.6 GB   ← well within V100 16/32GB ✅
```

### Verification Command

```bash
export HF_HOME=~/SharedFolder/hf_cache
python pipeline/stage_2_model_initialization.py \
    --lora_rank 16 \
    --output_dir ~/SharedFolder/checkpoints/ \
    --test_only        # just verify, don't save
```

### Acceptance Criteria

- [ ] Model loads without CUDA errors
- [ ] Trainable params ≈ 5M (< 10M)
- [ ] [LOC] token in vocabulary
- [ ] Forward pass produces shape (batch, 4) output
- [ ] GPU memory within expected range

---

## Stage 3: Training

**Duration:** 6-12 hours real time on V100
**Members:** 3 (coursework_track), 4 (enterprise_track)
**Dependency:** Stage 1 + Stage 2 complete

### Always Use tmux + nohup (Session Safety)

```bash
# Start tmux session — survives browser disconnect
tmux new -s training

# Run with nohup so it continues even if tmux dies
nohup python courses/coursework_track/train_coursework.py \
    --config courses/coursework_track/config_coursework.yaml \
    --data_dir ~/SharedFolder/data/ \
    --output_dir ~/SharedFolder/checkpoints/coursework/ \
    > ~/SharedFolder/logs/train_$(date +%Y%m%d_%H%M).log 2>&1 &

# Detach tmux: Ctrl+B then D
# Reattach later: tmux attach -t training
# Monitor: tail -f ~/SharedFolder/logs/train_*.log
```

### Checkpoint Resume (Critical)

Training scripts must support resuming from the last saved checkpoint:

```python
# core/utils/checkpoint.py

def save_checkpoint(model, optimizer, epoch, step, best_iou, path):
    torch.save({
        'epoch': epoch,
        'step': step,
        'best_iou': best_iou,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def load_checkpoint(model, optimizer, path):
    if os.path.exists(path):
        ckpt = torch.load(path, map_location='cuda')
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        print(f"Resumed from step {ckpt['step']}, best IoU {ckpt['best_iou']:.4f}")
        return ckpt['epoch'], ckpt['step'], ckpt['best_iou']
    return 0, 0, 0.0     # fresh start
```

### OOM & Stability Handling

```python
# In training loop:
for batch in dataloader:
    try:
        loss = train_step(batch)
    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            print(f"OOM at step {step}, skipping batch")
            continue
        raise e

    if step % 500 == 0:
        save_checkpoint(model, optimizer, epoch, step, best_iou,
                       f"{output_dir}/step_{step}.pth")
```

### Reproducibility

```python
# At top of every training script
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
```

### Configuration

#### coursework_track (51.511 GenAI — 3 epochs)

```yaml
# courses/coursework_track/config_coursework.yaml
training:
  num_epochs: 3
  batch_size: 4                      # V100 safe starting point
  gradient_accumulation_steps: 4     # effective batch = 16
  eval_interval: 500
  save_interval: 500
  seed: 42

optimizer:
  name: adamw
  learning_rate: 1.0e-4
  warmup_steps: 500
  gradient_clip: 1.0

hardware:
  device: cuda
  fp16: true                         # V100 supports fp16
  gradient_checkpointing: true       # saves VRAM at cost of speed

paths:
  data_dir: ~/SharedFolder/data/
  output_dir: ~/SharedFolder/checkpoints/coursework/
  log_dir: ~/SharedFolder/logs/
  hf_home: ~/SharedFolder/hf_cache/
```

#### enterprise_track (61.502 DL Enterprise — 10 epochs)

```yaml
# courses/enterprise_track/config_enterprise.yaml
training:
  num_epochs: 10
  batch_size: 8
  gradient_accumulation_steps: 2     # effective batch = 16
  eval_interval: 300
  save_interval: 300
  seed: 42

optimizer:
  name: adamw
  learning_rate: 1.0e-4
  warmup_steps: 1000
  gradient_clip: 1.0

hardware:
  device: cuda
  fp16: true
  gradient_checkpointing: true

paths:
  data_dir: ~/SharedFolder/data/
  output_dir: ~/SharedFolder/checkpoints/enterprise/
  log_dir: ~/SharedFolder/logs/
  hf_home: ~/SharedFolder/hf_cache/
```

### Expected Training Progress

```
Epoch 1/3:
  Step 0:    Loss=0.85, Val IoU=0.25
  Step 500:  Loss=0.42, Val IoU=0.48 ↑  [checkpoint saved]
  Step 1000: Loss=0.38, Val IoU=0.52 ↑
  Epoch 1:   Loss=0.32, Val IoU=0.57 ✓

Epoch 2/3:
  Step 2000: Loss=0.30, Val IoU=0.60 ↑
  Step 2500: Loss=0.28, Val IoU=0.63 ↑
  Epoch 2:   Loss=0.26, Val IoU=0.64 ✓

Epoch 3/3:
  Step 3500: Loss=0.25, Val IoU=0.65 ↑
  Epoch 3:   Loss=0.24, Val IoU=0.67 ✓

Best checkpoint: ~/SharedFolder/checkpoints/coursework/best.pth
```

### Acceptance Criteria

- [ ] Training starts without errors
- [ ] Loss decreasing, Val IoU increasing
- [ ] Checkpoints saved to SharedFolder every 500 steps
- [ ] `best.pth` updated whenever Val IoU improves
- [ ] Log file written to `~/SharedFolder/logs/`
- [ ] Coursework target: Val IoU ≥ 0.60
- [ ] Enterprise target: Val IoU ≥ 0.65

---

## Stage 4: Evaluation

**Duration:** 1-2 hours
**Members:** 3 (coursework), 4 (enterprise)
**Dependency:** Stage 3 complete

### ⚠️ Both tracks require 3-way ablation comparison

This is a hard requirement for **51.511 GenAI Template 2**.
Both `eval_coursework.py` and `eval_enterprise.py` must compare:

| Model | Description |
|-------|-------------|
| Baseline 1 | Standard LLaVA — outputs bbox as text tokens |
| Baseline 2 | Regression Head Only — LLM fully frozen, no LoRA |
| Ours | Spatial-LLaVA — LoRA + regression head |

### 4a: Coursework Evaluation (51.511)

```python
# courses/coursework_track/eval_coursework.py
# Outputs: results/coursework/metrics.json + failure_cases.json

runs = [
    {"name": "baseline_1_text_output",    "model": standard_llava},
    {"name": "baseline_2_head_only",      "model": head_only_model},
    {"name": "spatial_llava_ours",        "model": trained_model},
]

metrics_per_run = {
    "test_iou":          float,   # primary metric
    "test_rmse":         float,
    "test_mae":          float,
    "inference_time_ms": float,
    "failure_cases":     list,    # qualitative analysis
}
```

### 4b: Enterprise Evaluation (61.502)

Same 3-way comparison, plus additional benchmarks:

```python
# courses/enterprise_track/eval_enterprise.py
# Outputs: results/enterprise/baseline_comparison.json

additional_metrics = {
    "gpu_memory_gb":        float,
    "throughput_qps":       float,
    "improvement_vs_b1":    str,    # e.g. "+48.9%"
    "improvement_vs_b2":    str,
    "deployment_ready":     bool,
}
```

### Commands

```bash
# Coursework
python courses/coursework_track/eval_coursework.py \
    --model_path ~/SharedFolder/checkpoints/coursework/best.pth \
    --data_dir ~/SharedFolder/data/ \
    --output_dir ~/SharedFolder/results/coursework/

# Enterprise
python courses/enterprise_track/eval_enterprise.py \
    --model_path ~/SharedFolder/checkpoints/enterprise/best.pth \
    --data_dir ~/SharedFolder/data/ \
    --output_dir ~/SharedFolder/results/enterprise/ \
    --compare_baselines
```

### Acceptance Criteria

**Both tracks:**
- [ ] 3-way comparison table generated
- [ ] Failure cases documented (for paper/report)
- [ ] Results saved as JSON to SharedFolder

**Coursework (51.511):** Test IoU ≥ 0.60, RMSE ≤ 0.050
**Enterprise (61.502):** Test IoU ≥ 0.62, improvement vs baseline documented

---

## Stage 5: Demo / Deployment

**Duration:** 30-60 minutes
**Members:** 3 (coursework_track), 4 (enterprise_track)
**Dependency:** Stage 4 complete

### 5a: Gradio Demo — 51.511 GenAI (coursework_track)

Shows all 3 models side-by-side, required for Template 2 presentation.

```python
# courses/coursework_track/demo_gradio.py

import gradio as gr
from core.model import SpatialLLaVA

models = {
    "Standard LLaVA (baseline)": load_standard_llava(),
    "Head Only (ablation)":       load_head_only(),
    "Spatial-LLaVA (ours)":       load_spatial_llava(),
}

def predict_all(image, prompt):
    results = []
    for name, model in models.items():
        bbox = model.infer(image, prompt)
        annotated = draw_bbox(image, bbox, label=name)
        results.append(annotated)
    return results   # 3 annotated images side-by-side

demo = gr.Interface(
    fn=predict_all,
    inputs=[gr.Image(), gr.Textbox(label="Referring expression")],
    outputs=[gr.Image(label=name) for name in models.keys()],
    title="Spatial-LLaVA — Model Comparison",
)

# Launch on cluster (port-forward to view locally)
demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
```

```bash
# Run on cluster
python courses/coursework_track/demo_gradio.py \
    --coursework_ckpt ~/SharedFolder/checkpoints/coursework/best.pth

# Gradio will print a public share URL automatically
```

### 5b: FastAPI Server — 61.502 DL Enterprise (enterprise_track)

```python
# courses/enterprise_track/deployment/fastapi_server.py

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from core.model import SpatialLLaVA
import time, os

app = FastAPI(title="Spatial-LLaVA API", version="1.0.0")
model = SpatialLLaVA.load(os.environ.get("MODEL_PATH", "checkpoints/enterprise/best.pth"))

@app.post("/predict")
async def predict(image: UploadFile = File(...), prompt: str = Query(...)):
    from PIL import Image
    img = Image.open(image.file)
    t0 = time.time()
    bbox = model.infer(img, prompt)
    return JSONResponse({
        "bbox": bbox.tolist(),          # [x, y, w, h] normalized
        "inference_time_ms": (time.time() - t0) * 1000,
    })

@app.get("/health")
async def health():
    return {"status": "ok", "model": "spatial-llava-7b"}
```

```bash
# Run locally via Docker (Dockerfile.inference)
docker build -f infrastructure/docker/Dockerfile.inference -t spatial-llava:inference .
docker run -p 8000:8000 \
    -v ~/SharedFolder/checkpoints/enterprise/best.pth:/app/model.pth \
    -e MODEL_PATH=/app/model.pth \
    spatial-llava:inference

# Test
curl http://localhost:8000/health
curl -X POST "http://localhost:8000/predict" \
    -F "image=@test.jpg" \
    -F "prompt=find the person on the left"
```

### Acceptance Criteria

**Gradio (51.511):**
- [ ] 3 models shown side-by-side
- [ ] Works on live demo during Week 13 presentation
- [ ] Failure mode examples included

**FastAPI (61.502):**
- [ ] `/health` and `/predict` endpoints working
- [ ] Containerized via `Dockerfile.inference`
- [ ] Latency documented in report

---

## Team Responsibilities & Timeline

| Stage | Task | Member | Dependency | Deliverable |
|-------|------|--------|-----------|-------------|
| 0 | Cluster env check | 1 | — | ✅ V100 confirmed |
| 1 | Data prep → SharedFolder | 2 | Stage 0 | 📦 3 pkl files |
| 2 | Model init verify | 1 | Stage 0 | 🤖 Forward pass confirmed |
| 3a | Train coursework (3 epochs) | 3 | Stage 1-2 | 💾 best.pth in SharedFolder |
| 3b | Train enterprise (10 epochs) | 4 | Stage 1-2 | 💾 best.pth in SharedFolder |
| 4a | Eval coursework (3-way) | 3 | Stage 3a | 📊 metrics.json |
| 4b | Eval enterprise (3-way) | 4 | Stage 3b | 📊 baseline_comparison.json |
| 5a | Gradio demo (3 models) | 3 | Stage 4a | 🚀 Live demo URL |
| 5b | FastAPI + Docker | 4 | Stage 4b | 🚀 Containerized API |

### First-Time Cluster Setup (Member 1, once only)

```bash
# 1. Set up environment
echo 'export HF_HOME=~/SharedFolder/hf_cache' >> ~/.bashrc
source ~/.bashrc

# 2. Create SharedFolder structure
mkdir -p ~/SharedFolder/{hf_cache,data,checkpoints/{coursework,enterprise},results/{coursework,enterprise},logs}

# 3. Clone repo
git clone https://github.com/your-org/spatial-llava.git ~/spatial-llava
cd ~/spatial-llava

# 4. Install extra dependencies (PyTorch already pre-installed)
pip install peft transformers gradio bitsandbytes fastapi uvicorn python-multipart

# 5. Verify
python pipeline/stage_0_environment.py
```

---

## Quick Reference

```bash
# ── Cluster session start ──────────────────────────────────
export HF_HOME=~/SharedFolder/hf_cache
cd ~/spatial-llava
git pull

# ── Stages ────────────────────────────────────────────────
python pipeline/stage_0_environment.py
python pipeline/stage_1_data_preparation.py --output_dir ~/SharedFolder/data/
python pipeline/stage_2_model_initialization.py --test_only

# ── Training (always in tmux) ──────────────────────────────
tmux new -s train
nohup python courses/coursework_track/train_coursework.py \
    --config courses/coursework_track/config_coursework.yaml \
    > ~/SharedFolder/logs/train_cw.log 2>&1 &

# ── Monitor ───────────────────────────────────────────────
tail -f ~/SharedFolder/logs/train_cw.log

# ── Eval ──────────────────────────────────────────────────
python courses/coursework_track/eval_coursework.py \
    --model_path ~/SharedFolder/checkpoints/coursework/best.pth \
    --data_dir ~/SharedFolder/data/ \
    --output_dir ~/SharedFolder/results/coursework/

# ── Demo ──────────────────────────────────────────────────
python courses/coursework_track/demo_gradio.py \
    --coursework_ckpt ~/SharedFolder/checkpoints/coursework/best.pth
```