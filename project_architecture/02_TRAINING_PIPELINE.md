# 02_TRAINING_PIPELINE.md

## What Are We Building?

```
Input:  Image + Text prompt ("find the person on the left")
Output: Bounding box [x_center, y_center, width, height] normalized to [0, 1]

Architecture:
  Image + Text → LLaVA Vision Encoder (frozen)
                       ↓
              LLM + LoRA adapters (trainable)
                       ↓
           [LOC] token hidden state
                       ↓
          MLP Regression Head (trainable)
                       ↓
              [x, y, w, h] prediction
```

---

## Compute Environment

Training runs on **SUTD AIMC Education Cluster**, not locally.

```
Your Mac → Browser → JupyterHub (login2.gpucluster.sutd.edu.sg:8888)
                          → Kubernetes → V100 GPU node (32GB VRAM)
```

**Key facts:**
- PyTorch + CUDA already pre-installed, no Docker needed for training
- HOME directory: 100GB persistent storage — **code only**
- SharedFolder: 1TB+ shared storage — **all large files go here**
- Sessions idle >2 hours are terminated → always use `tmux`

---

## Storage Layout

```
~/spatial-llava/                         ← git clone (code only)
~/SharedFolder/MDAIE/group6/
    ├── hf_cache/                        ← HuggingFace model cache (~14GB)
    ├── data/                            ← RefCOCO dataset (~50GB)
    │   ├── refcoco_train.pkl
    │   ├── refcoco_val.pkl
    │   ├── refcoco_test.pkl
    │   └── dataset_stats.json
    ├── checkpoints/
    │   ├── main/                        ← LoRA + head, 10 epochs
    │   └── ablation/                    ← head only, 3 epochs
    ├── results/
    │   ├── main/
    │   └── ablation/
    └── logs/
```

---

## Training Strategy

Two training runs, one shared dataset:

| Run | What trains | Epochs | Purpose |
|-----|-------------|--------|---------|
| **main** | LoRA + regression head | 10 | Primary model for both courses |
| **ablation** | Regression head only (LLM frozen) | 3 | Ablation study for 51.511 |

**51.511 GenAI** uses epoch 3 checkpoint from main + ablation results for 3-way comparison.
**61.502 Enterprise** uses epoch 10 checkpoint from main for FastAPI deployment.

---

## Stage Overview

| Stage | Duration | What | Who | Command |
|-------|----------|------|-----|---------|
| 0 | 10 min | Cluster env check | Member 1 | `python pipeline/stage_0_environment.py` |
| 1 | 2-3 hrs | Download + preprocess RefCOCO | Member 2 | auto via start_training.sh |
| 3 | 8-12 hrs | Train main + ablation | Member 3, 4 | `bash shared/scripts/start_training.sh main` |
| 4 | 1-2 hrs | Evaluate 3-way comparison | Member 3, 4 | `python courses/shared/eval.py` |
| 5a | 30 min | Gradio demo (51.511) | Member 3 | `python courses/genai/demo_gradio.py` |
| 5b | 30 min | FastAPI server (61.502) | Member 4 | `python courses/enterprise/fastapi_server.py` |

---

## Directory Structure

```
spatial-llava/
├── core/                          # Reusable modules
│   ├── model/
│   │   ├── spatial_llava.py       # LLaVA + LoRA + regression head
│   │   ├── regression_head.py     # MLP: hidden_state → [x, y, w, h]
│   │   └── lora_config.py
│   ├── data/
│   │   ├── refcoco_loader.py
│   │   ├── preprocessing.py
│   │   └── data_utils.py
│   ├── loss/
│   │   └── spatial_loss.py        # IoU + L1 combined loss
│   └── utils/
│       ├── metrics.py             # IoU, RMSE, MAE
│       ├── checkpoint.py          # save/load/resume
│       └── visualization.py
│
├── pipeline/
│   ├── stage_0_environment.py     # Cluster env check
│   └── stage_1_data_preparation.py
│
├── courses/
│   ├── shared/
│   │   ├── train.py               # Unified training script (main + ablation)
│   │   ├── eval.py                # 3-way evaluation
│   │   └── config.yaml            # All training config
│   ├── genai/                     # 51.511 deliverables
│   │   ├── demo_gradio.py         # 3-model side-by-side demo
│   │   └── eval_gen_ai.py
│   └── enterprise/                # 61.502 deliverables
│       ├── fastapi_server.py
│       └── eval_enterprise.py
│
├── shared/scripts/
│   ├── setup_cluster.sh           # First-time cluster setup
│   └── start_training.sh          # One-command training launcher
│
├── infrastructure/docker/
│   ├── Dockerfile.dev             # Mac local dev (CPU-only)
│   └── Dockerfile.inference       # FastAPI production
│
├── tests/                         # Unit tests (267 passing, 95% coverage)
├── scripts/check.sh               # Local lint + test runner
└── .github/workflows/pr_checks.yml
```

---

## Quick Reference

```bash
# ── First time on cluster ──────────────────────────────────
git clone https://github.com/vincent-chengsheng-zheng/SPATIAL-LLAVA.git ~/spatial-llava
cd ~/spatial-llava
bash shared/scripts/setup_cluster.sh

# ── Every session ─────────────────────────────────────────
source ~/.bashrc
cd ~/spatial-llava && git pull

# ── Stage 0: Check environment ────────────────────────────
python pipeline/stage_0_environment.py

# ── Stage 3: Start training (auto-downloads data if needed)
bash shared/scripts/start_training.sh main
bash shared/scripts/start_training.sh ablation

# ── Monitor training ──────────────────────────────────────
tail -f ~/SharedFolder/MDAIE/group6/logs/train_main_*.log

# ── Stage 4: Evaluate ─────────────────────────────────────
python courses/shared/eval.py \
    --ours_ckpt ~/SharedFolder/MDAIE/group6/checkpoints/main/best.pth \
    --ablation_ckpt ~/SharedFolder/MDAIE/group6/checkpoints/ablation/best.pth \
    --output_dir ~/SharedFolder/MDAIE/group6/results/

# ── Stage 5a: Gradio demo (51.511) ────────────────────────
python courses/genai/demo_gradio.py \
    --ours_ckpt ~/SharedFolder/MDAIE/group6/checkpoints/main/best.pth \
    --ablation_ckpt ~/SharedFolder/MDAIE/group6/checkpoints/ablation/best.pth

# ── Stage 5b: FastAPI server (61.502) ─────────────────────
python courses/enterprise/fastapi_server.py \
    --model_path ~/SharedFolder/MDAIE/group6/checkpoints/main/best.pth
```

---

## Acceptance Criteria

| Milestone | Target |
|-----------|--------|
| Val IoU (main, epoch 3) | ≥ 0.60 |
| Val IoU (main, epoch 10) | ≥ 0.65 |
| Improvement vs baseline | ≥ 40% |
| Gradio demo | 3 models side-by-side, live |
| FastAPI | `/health` + `/predict` working, containerized |