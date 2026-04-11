"""
pipeline/step4_evaluate.py

Step 4: Evaluate and compare all three models on RefCOCO test split.

What this script does:
    1. Baseline  : Standard LLaVA + regex parsing (no training)
    2. Ablation  : Frozen LLaVA + MLP head (checkpoints/ablation/best.pth)
    3. Main      : LoRA + MLP head          (checkpoints/main/best.pth)
    All evaluated on RefCOCO test split (1,975 samples)

Output:
    results/evaluation/
        comparison.json     ← three-model comparison table
        baseline.json       ← baseline metrics
        ablation.json       ← ablation metrics
        main.json           ← main metrics

Usage:
    python pipeline/step4_evaluate.py
"""

import os, sys, json, time, argparse
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.paths import PATHS
from core.data.refcoco_loader import RefCOCODataset
from core.data.refcoco_loader_pil import RefCOCODatasetPIL
from core.model.spatial_llava import load_model
from core.model.llava import StandardLLaVA
from core.data.preprocessing import PROMPT_TEMPLATE
from core.utils.metrics import compute_all_metrics


# ── Logger ────────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self, log_path: str):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._file = open(log_path, "w")
        print(f"[Logger] Writing to {log_path}")

    def log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        self._file.write(line + "\n")
        self._file.flush()

    def close(self):
        self._file.close()


# ── Baseline ──────────────────────────────────────────────────────────────────

def evaluate_baseline(logger, device, batch_size=8):
    logger.log("\n" + "="*60)
    logger.log("  [1/3] Baseline: Standard LLaVA + regex")
    logger.log("="*60)

    model = StandardLLaVA(
        model_id="llava-hf/llava-1.5-7b-hf",
        hf_cache=str(PATHS.weights),
    )
    model.to(device)
    model.eval()

    test_ds = RefCOCODatasetPIL.from_split(
        data_dir=str(PATHS.data), split="test",
    )
    logger.log(f"  Test samples: {len(test_ds):,}")

    results = []
    parse_success = 0
    t0 = time.time()
    total = len(test_ds)

    for batch_start in range(0, total, batch_size):
        batch_end    = min(batch_start + batch_size, total)
        batch        = [test_ds[i] for i in range(batch_start, batch_end)]
        images       = [s["image"] for s in batch]
        texts        = [s["text"]  for s in batch]
        prompts      = [PROMPT_TEMPLATE.format(text=t) for t in texts]

        try:
            inputs = model.processor(
                text=prompts, images=images,
                return_tensors="pt", padding=True,
            ).to(device)
            with torch.no_grad():
                output_ids = model.model.generate(
                    **inputs, max_new_tokens=50, do_sample=False,
                )
            input_len = inputs["input_ids"].shape[1]
            for i, s in enumerate(batch):
                raw         = model.processor.decode(
                    output_ids[i][input_len:], skip_special_tokens=True
                ).strip()
                parsed_bbox = StandardLLaVA.parse_bbox(raw)
                parsed      = parsed_bbox is not None
                if parsed:
                    parse_success += 1
                results.append({
                    "pred_bbox": parsed_bbox or [0.0, 0.0, 0.0, 0.0],
                    "gt_bbox":   s["bbox"].tolist(),
                })
        except Exception:
            for s in batch:
                results.append({
                    "pred_bbox": [0.0, 0.0, 0.0, 0.0],
                    "gt_bbox":   s["bbox"].tolist(),
                })

        done = len(results)
        if done % 200 == 0 or done == total:
            logger.log(f"  [{done}/{total}] parse={parse_success/done*100:.1f}% "
                       f"elapsed={( time.time()-t0)/60:.1f}min")

    preds   = torch.tensor([r["pred_bbox"] for r in results])
    targets = torch.tensor([r["gt_bbox"]   for r in results])
    metrics = compute_all_metrics(preds, targets)
    metrics["parse_success_rate"] = round(parse_success / total, 4)
    metrics["model"] = "baseline"
    logger.log(f"  IoU={metrics['mean_iou']:.4f}  parse={metrics['parse_success_rate']*100:.1f}%")

    del model
    torch.cuda.empty_cache()
    return metrics


# ── Spatial Models ────────────────────────────────────────────────────────────

def evaluate_spatial_model(logger, device, use_lora, ckpt_dir, name, batch_size=8):
    logger.log("\n" + "="*60)
    logger.log(f"  Evaluating: {name}")
    logger.log("="*60)

    model, processor = load_model(use_lora=use_lora, device=device)

    ckpt_path = Path(ckpt_dir) / "best.pth"
    logger.log(f"  Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger.log(f"  Loaded epoch={ckpt.get('epoch','?')}")

    # Use RefCOCODataset directly with test split
    test_ds = RefCOCODataset(
        split="test",
        processor=processor,
        max_length=600,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, num_workers=2,
        pin_memory=True,
    )
    logger.log(f"  Test samples: {len(test_ds):,}")

    all_preds, all_targets = [], []
    t0 = time.time()
    total = len(test_ds)

    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            preds = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["pixel_values"].to(device),
            )
            all_preds.append(preds.cpu())
            all_targets.append(batch["bbox"])

            done = (step + 1) * batch_size
            if done % 400 == 0 or done >= total:
                logger.log(f"  [{min(done,total)}/{total}] "
                           f"elapsed={( time.time()-t0)/60:.1f}min")

    preds_t   = torch.cat(all_preds,   dim=0)
    targets_t = torch.cat(all_targets, dim=0)
    metrics   = compute_all_metrics(preds_t, targets_t)
    metrics["model"] = name
    logger.log(f"  IoU={metrics['mean_iou']:.4f}  RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}")

    del model, processor
    torch.cuda.empty_cache()
    return metrics


# ── Save ──────────────────────────────────────────────────────────────────────

def save_comparison(baseline, ablation, main, logger):
    out_dir = PATHS.results / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, m in [("baseline", baseline), ("ablation", ablation), ("main", main)]:
        with open(out_dir / f"{name}.json", "w") as f:
            json.dump(m, f, indent=2)

    comparison = {
        "evaluated_on": "RefCOCO test split (1,975 samples)",
        "timestamp": datetime.now().isoformat(),
        "results": {
            "baseline": {"iou": baseline["mean_iou"], "rmse": baseline["rmse"],
                         "mae": baseline["mae"],
                         "parse_rate": baseline.get("parse_success_rate")},
            "ablation": {"iou": ablation["mean_iou"], "rmse": ablation["rmse"],
                         "mae": ablation["mae"]},
            "main":     {"iou": main["mean_iou"],     "rmse": main["rmse"],
                         "mae": main["mae"]},
        },
        "improvements": {
            "ablation_vs_baseline": f"+{(ablation['mean_iou']/baseline['mean_iou']-1)*100:.1f}%",
            "main_vs_baseline":     f"+{(main['mean_iou']/baseline['mean_iou']-1)*100:.1f}%",
            "main_vs_ablation":     f"+{(main['mean_iou']/ablation['mean_iou']-1)*100:.1f}%",
        },
    }
    with open(out_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    logger.log("\n" + "="*60)
    logger.log("  FINAL RESULTS (RefCOCO Test Split)")
    logger.log("="*60)
    logger.log(f"  Baseline  IoU: {baseline['mean_iou']:.4f}")
    logger.log(f"  Ablation  IoU: {ablation['mean_iou']:.4f}  "
               f"({comparison['improvements']['ablation_vs_baseline']})")
    logger.log(f"  Main      IoU: {main['mean_iou']:.4f}  "
               f"({comparison['improvements']['main_vs_baseline']})")
    logger.log(f"  Saved → {out_dir}")
    logger.log("="*60)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    logger = Logger(str(PATHS.logs / f"step4_{timestamp}.log"))

    logger.log("="*60)
    logger.log("  Step 4: Final Evaluation — All Three Models")
    logger.log(f"  Device: {device}")
    logger.log("="*60)

    baseline = evaluate_baseline(logger, device, args.batch_size)
    ablation = evaluate_spatial_model(
        logger, device,
        use_lora=False, ckpt_dir=PATHS.ckpt_ablation,
        name="Ablation (Frozen + Head)",
        batch_size=args.batch_size,
    )
    main_m = evaluate_spatial_model(
        logger, device,
        use_lora=True, ckpt_dir=PATHS.ckpt_main,
        name="Main (LoRA + Head)",
        batch_size=args.batch_size,
    )
    save_comparison(baseline, ablation, main_m, logger)

    logger.log("\n✅ Step 4 complete!")
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    main(args)
