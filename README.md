# Spatial-LLaVA: Visual Grounding via LLaVA + MLP Regression Head

A research project for 51.511 Multimodal Generative AI (Template 2: Generative Model Research Study).

We extend LLaVA-1.5-7B with a lightweight MLP regression head to perform visual grounding on RefCOCO — predicting bounding boxes from image-text pairs without relying on fragile text parsing.

---

## Results

Evaluated on RefCOCO test split (1,975 samples):

| Model | Val IoU | Test IoU | RMSE | MAE | Method |
|---|---|---|---|---|---|
| Baseline | 0.097 | 0.097 | 0.288 | 0.238 | Vanilla LLaVA + regex |
| Ablation | 0.267 | 0.284 | 0.224 | 0.177 | Frozen LLaVA + MLP head |
| **Main** | **0.357** | **0.386** | **0.172** | **0.119** | LoRA fine-tuned + MLP head |

Main model achieves **+297.5% IoU improvement** over baseline.

---

## Architecture

```
Image + Text
    → LLaVA-1.5-7B (vision encoder + LLM, optionally + LoRA)
    → hidden state of [LOC] token (last token, 4096-dim)
    → MLP Regression Head (4096 → 512 → 256 → 4)
    → Sigmoid → [xc, yc, w, h] ∈ (0, 1)
```

**Three model variants:**
- **Baseline** — vanilla LLaVA generates text, regex extracts coordinates. No training.
- **Ablation** — backbone frozen, only MLP head trained. Tests whether head alone is sufficient.
- **Main** — LoRA adapters (rank=16) injected into LLM + MLP head trained jointly.

**Training config:**
- Loss: L1 + GIoU
- Optimizer: AdamW + CosineAnnealingLR
- LR: 2e-4, batch size: 8, epochs: 10
- Training samples: 20,000 / Val: 1,000 / Test: 1,975
- VRAM peak: 44.5 GB (A100-SXM4-80GB)

---

## Repo Structure

```
spatial-llava/
├── core/
│   ├── model/
│   │   ├── spatial_llava.py      # SpatialLLaVA model + load_model()
│   │   ├── llava.py              # Baseline StandardLLaVA wrapper
│   │   ├── regression_head.py    # MLP head (4096→512→256→4)
│   │   └── lora_config.py        # LoRA configuration
│   ├── data/
│   │   ├── refcoco_loader.py     # Dataset for spatial models
│   │   ├── refcoco_loader_pil.py # Dataset for baseline (PIL)
│   │   └── preprocessing.py      # Prompt template + transforms
│   ├── loss/
│   │   └── spatial_loss.py       # L1 + GIoU loss
│   └── utils/
│       ├── metrics.py            # IoU, RMSE, MAE
│       ├── checkpoint.py         # Save/load helpers
│       └── visualization.py      # Bbox drawing utilities
├── pipeline/
│   ├── step1_baseline_inference.py  # Run baseline evaluation
│   ├── step2_train_main.py          # Train LoRA + head
│   ├── step3_train_ablation.py      # Train head only
│   └── step4_evaluate.py            # Evaluate all 3 models
├── demo/
│   └── demo_gradio.py            # Interactive Gradio demo
├── results/
│   ├── evaluation/               # Final test metrics (all 3 models)
│   ├── main/                     # Main model metrics + predictions
│   └── ablation/                 # Ablation metrics + predictions
├── logs/                         # Training logs
├── shared/
│   ├── notebooks/
│   │   └── training.ipynb        # Training notebook
│   └── scripts/
│       ├── setup_cluster.sh      # Cluster environment setup
│       └── download_data.sh      # COCO + RefCOCO data download
└── data/
    ├── refcoco_train.pkl         # 42,404 samples
    ├── refcoco_val.pkl           # 3,811 samples
    └── refcoco_test.pkl          # 1,975 samples
```

---

## Setup (SUTD GPU Cluster)

```bash
# Clone repo to /tmp/ (faster I/O)
git clone https://github.com/vincent-chengsheng-zheng/SPATIAL-LLAVA /tmp/SPATIAL-LLAVA
cd /tmp/SPATIAL-LLAVA

# Setup environment
source shared/scripts/setup_cluster.sh

# Download COCO images + RefCOCO pkl files (~13.5GB)
bash shared/scripts/download_data.sh

# Install flash-attn (required, ~5min)
pip install flash-attn --no-cache-dir --no-build-isolation

# Set offline mode after weights downloaded
export TRANSFORMERS_OFFLINE=1
```

> **Note:** `/tmp/` clears on cluster restart. Checkpoints are saved to `~/SPATIAL-LLAVA_APR10/` as backup.

---

## Training

```bash
# Step 1: Baseline (no training needed)
python pipeline/step1_baseline_inference.py

# Step 2: Train main model (LoRA + head, ~10 epochs)
python pipeline/step2_train_main.py

# Step 3: Train ablation (frozen backbone + head only)
python pipeline/step3_train_ablation.py

# Step 4: Evaluate all three models on test split
python pipeline/step4_evaluate.py
```

Results saved to `results/evaluation/comparison.json`.

---

## Gradio Demo

Interactive side-by-side comparison of all three models:

```bash
cd /tmp/SPATIAL-LLAVA
export TRANSFORMERS_OFFLINE=1
python demo/demo_gradio.py
```

Upload any image, enter a referring expression (e.g. "the person on the left"), and see bounding box predictions from all three models simultaneously.

> Running on SUTD cluster: open via `http://login2.gpucluster.sutd.edu.sg/user/<id>/vscode/proxy/7860/`

---

## References

- [LLaVA-1.5](https://arxiv.org/abs/2310.03744) — Haotian Liu et al., NeurIPS 2023
- [u-LLaVA](https://arxiv.org/abs/2311.05348) — Xu et al., ECAI 2024
- [PixelLLM](https://arxiv.org/abs/2312.09237) — Xu et al., CVPR 2024
- [GIoU Loss](https://arxiv.org/abs/1902.09630) — Rezatofighi et al., CVPR 2019
- [RefCOCO](https://arxiv.org/abs/1608.00272) — Yu et al., ECCV 2016
