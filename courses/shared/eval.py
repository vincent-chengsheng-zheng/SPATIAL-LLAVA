"""
courses/shared/eval.py

Evaluation script for Spatial-LLaVA — runs all three models and generates
comparison metrics, visualizations, and inference speed benchmarks.

Three models evaluated:
    baseline  — Standard LLaVA (no training, text output parsed to bbox)
    ablation  — Head only (no LoRA, LLM frozen)
    ours      — Full Spatial-LLaVA (LoRA + regression head)

Usage:
    python courses/shared/eval.py \\
        --ours_ckpt  ~/SharedFolder/MDAIE/group6/checkpoints/main/best.pth \\
        --ablation_ckpt ~/SharedFolder/MDAIE/group6/checkpoints/ablation/best.pth \\
        --data_dir   ~/SharedFolder/MDAIE/group6/data/ \\
        --output_dir ~/SharedFolder/MDAIE/group6/results/

Output files:
    results/metrics_all.json        — all three models side by side
    results/comparison_table.json   — improvement percentages
    results/training_curves.json    — merged from training_log.json files
    results/examples/               — bbox visualizations per model
    results/failure_cases/          — cases where IoU < threshold
"""

import os
import json
import time
import argparse
from typing import Dict, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

from core.model.spatial_llava import SpatialLLaVA
from core.data.refcoco_loader import RefCOCODataset
from core.utils.metrics import compute_all_metrics
from core.utils.checkpoint import load_checkpoint
from core.utils.visualization import display_comparison
from core.loss.spatial_loss import iou as compute_iou


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference_batch(
    model: SpatialLLaVA,
    batch: Dict[str, Tensor],
    device: torch.device,
    use_fp16: bool = True,
) -> Tensor:
    """
    Run inference on one batch and return predicted bboxes.

    Args:
        model   : SpatialLLaVA model in eval mode
        batch   : Dict with "image", "input_ids", "bbox" tensors
        device  : Target device
        use_fp16: Whether to use fp16 autocast

    Returns:
        Tensor (B, 4) — predicted [x_c, y_c, w, h] in [0, 1]
    """
    images = batch["image"].to(device)
    input_ids = batch["input_ids"].to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(
            enabled=use_fp16 and device.type == "cuda"
        ):
            outputs = model(images, input_ids)

    return outputs["bbox"].cpu()


def collect_predictions(
    model: SpatialLLaVA,
    loader: DataLoader,
    device: torch.device,
    use_fp16: bool = True,
) -> tuple:
    """
    Run inference over the full dataset split.

    Returns:
        Tuple (preds, targets) — both Tensor (N, 4)
        Also returns per-sample inference time in ms
    """
    model.eval()
    all_preds = []
    all_targets = []
    total_time = 0.0
    total_samples = 0

    for batch in loader:
        t0 = time.time()
        preds = run_inference_batch(model, batch, device, use_fp16)
        elapsed = (time.time() - t0) * 1000   # ms
        total_time += elapsed
        total_samples += preds.shape[0]

        all_preds.append(preds)
        all_targets.append(batch["bbox"])

    avg_ms = total_time / max(total_samples, 1)
    return torch.cat(all_preds, dim=0), torch.cat(all_targets, dim=0), avg_ms


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_model_metrics(
    preds: Tensor,
    targets: Tensor,
    inference_time_ms: float,
) -> dict:
    """
    Compute full metric report for one model.

    Args:
        preds             : Tensor (N, 4) — predicted bboxes
        targets           : Tensor (N, 4) — ground truth bboxes
        inference_time_ms : Average inference time per sample in ms

    Returns:
        Dict with mean_iou, rmse, mae, inference_time_ms, num_samples
    """
    metrics = compute_all_metrics(preds, targets)
    metrics["inference_time_ms"] = round(inference_time_ms, 2)
    metrics["num_samples"] = preds.shape[0]
    return metrics


def compute_improvement(baseline_iou: float, model_iou: float) -> str:
    """
    Compute percentage improvement in IoU over baseline.

    Returns:
        String like "+48.9%" or "-5.2%" or "N/A"
    """
    if baseline_iou == 0:
        return "N/A"
    pct = (model_iou - baseline_iou) / baseline_iou * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def generate_comparison_table(results: Dict[str, dict]) -> dict:
    """
    Generate a comparison table from per-model metric dicts.

    Args:
        results : Dict mapping model_name → metrics dict
                  Must have keys: "baseline", "ablation", "ours"

    Returns:
        Dict with per-model metrics plus improvement percentages

    Raises:
        KeyError : If required model keys are missing
    """
    required = {"baseline", "ablation", "ours"}
    missing = required - set(results.keys())
    if missing:
        raise KeyError(f"Missing required model results: {missing}")

    baseline_iou = results["baseline"]["mean_iou"]
    ablation_iou = results["ablation"]["mean_iou"]
    ours_iou = results["ours"]["mean_iou"]

    table = {}
    for name, metrics in results.items():
        row = dict(metrics)
        if name == "ablation":
            row["improvement_vs_baseline"] = compute_improvement(
                baseline_iou, ablation_iou
            )
        elif name == "ours":
            row["improvement_vs_baseline"] = compute_improvement(
                baseline_iou, ours_iou
            )
            row["improvement_vs_ablation"] = compute_improvement(
                ablation_iou, ours_iou
            )
        table[name] = row

    return table


# ── Visualization ─────────────────────────────────────────────────────────────

def save_inference_examples(
    model_preds: Dict[str, Tensor],
    targets: Tensor,
    loader: DataLoader,
    output_dir: str,
    n_examples: int = 20,
    iou_threshold: float = 0.3,
) -> None:
    """
    Save side-by-side visualizations for all three models.

    Saves:
        output_dir/examples/     — random N examples
        output_dir/failures/     — cases where ours IoU < threshold

    Args:
        model_preds   : Dict of {model_name: predictions tensor (N, 4)}
        targets       : Ground truth tensor (N, 4)
        loader        : DataLoader (to get original images)
        output_dir    : Base results directory
        n_examples    : Number of examples to save
        iou_threshold : IoU below this is considered a failure case
    """
    examples_dir = os.path.join(output_dir, "examples")
    failures_dir = os.path.join(output_dir, "failure_cases")
    os.makedirs(examples_dir, exist_ok=True)
    os.makedirs(failures_dir, exist_ok=True)

    # Collect raw images from loader
    all_images = []
    for batch in loader:
        all_images.append(batch["image"])
    all_images = torch.cat(all_images, dim=0)

    # Compute IoU for "ours" to find failure cases
    ours_iou = compute_iou(model_preds["ours"], targets)

    saved_examples = 0
    saved_failures = 0

    # Random indices for examples
    n = targets.shape[0]
    indices = torch.randperm(n)[:n_examples].tolist()

    for idx in range(n):
        img_tensor = all_images[idx]
        # Denormalize image for display
        img_np = img_tensor.numpy().transpose(1, 2, 0)
        img_np = (img_np * 0.225 + 0.45).clip(0, 1)
        img_np = (img_np * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)

        target_box = tuple(targets[idx].tolist())
        sample_iou = ours_iou[idx].item()

        # Save example
        if saved_examples < n_examples and idx in indices:
            for model_name, preds in model_preds.items():
                pred_box = tuple(preds[idx].tolist())
                result = display_comparison(pil_img, pred_box, target_box)
                fname = f"example_{saved_examples:03d}_{model_name}.png"
                result.save(os.path.join(examples_dir, fname))
            saved_examples += 1

        # Save failure case
        if sample_iou < iou_threshold and saved_failures < n_examples:
            for model_name, preds in model_preds.items():
                pred_box = tuple(preds[idx].tolist())
                result = display_comparison(pil_img, pred_box, target_box)
                fname = (
                    f"failure_{saved_failures:03d}_"
                    f"iou{sample_iou:.2f}_{model_name}.png"
                )
                result.save(os.path.join(failures_dir, fname))
            saved_failures += 1

        if saved_examples >= n_examples and saved_failures >= n_examples:
            break

    print(f"  Saved {saved_examples} examples → {examples_dir}")
    print(f"  Saved {saved_failures} failure cases → {failures_dir}")


# ── Training curves ───────────────────────────────────────────────────────────

def load_training_curves(
    main_results_dir: str,
    ablation_results_dir: str,
) -> dict:
    """
    Load training_log.json files and merge into one dict for plotting.

    Returns:
        Dict with "main" and "ablation" training histories
    """
    curves = {}

    for mode, results_dir in [
        ("main", main_results_dir),
        ("ablation", ablation_results_dir),
    ]:
        log_path = os.path.join(
            os.path.expanduser(results_dir), "training_log.json"
        )
        if os.path.exists(log_path):
            with open(log_path) as f:
                curves[mode] = json.load(f)
            print(f"  Loaded training log: {log_path}")
        else:
            print(f"  ⚠ Training log not found: {log_path}")
            curves[mode] = {"mode": mode, "epochs": []}

    return curves


# ── IO ────────────────────────────────────────────────────────────────────────

def save_results(results: dict, output_dir: str, filename: str) -> str:
    """Save results dict as JSON. Returns saved file path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {path}")
    return path


def load_model_with_checkpoint(
    checkpoint_path: Optional[str],
    mode: str,
    device: torch.device,
) -> SpatialLLaVA:
    """
    Load SpatialLLaVA and optionally restore from checkpoint.

    Args:
        checkpoint_path : Path to .pth file, or None for baseline
        mode            : "main" or "ablation"
        device          : Target device

    Returns:
        SpatialLLaVA model on device in eval mode
    """
    from torch.optim import AdamW

    model = SpatialLLaVA(
        freeze_vision=True,
        freeze_llm=(mode == "ablation"),
    ).to(device)

    if checkpoint_path:
        path = os.path.expanduser(checkpoint_path)
        dummy_opt = AdamW(model.parameters(), lr=1e-4)
        load_checkpoint(model, dummy_opt, path)
        print(f"  Loaded checkpoint: {path}")

    model.eval()
    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda"
    output_dir = os.path.expanduser(args.output_dir)

    # Per-model result dirs
    baseline_dir = os.path.join(output_dir, "baseline")
    main_dir = os.path.join(output_dir, "main")
    ablation_dir = os.path.join(output_dir, "ablation")
    for d in [baseline_dir, main_dir, ablation_dir]:
        os.makedirs(d, exist_ok=True)

    print("=" * 60)
    print("  Spatial-LLaVA Evaluation")
    print(f"  Device     : {device}")
    print(f"  Output dir : {output_dir}")
    print("=" * 60)

    # ── Test DataLoader ───────────────────────────────────
    print("\n[1/5] Loading test set...")
    test_dataset = RefCOCODataset.from_config(
        {"data_dir": args.data_dir}, split="test"
    )
    test_loader = test_dataset.get_dataloader(
        batch_size=args.batch_size, shuffle=False
    )
    print(f"  Test samples: {len(test_dataset):,}")

    results = {}
    all_preds = {}

    # ── Baseline: Standard LLaVA ──────────────────────────
    print("\n[2/5] Evaluating Baseline (Standard LLaVA)...")
    baseline_model = load_model_with_checkpoint(None, "main", device)
    preds, targets, ms = collect_predictions(
        baseline_model, test_loader, device, use_fp16
    )
    results["baseline"] = compute_model_metrics(preds, targets, ms)
    all_preds["baseline"] = preds
    print(f"  IoU: {results['baseline']['mean_iou']:.4f} | "
          f"Latency: {ms:.1f}ms/sample")
    save_results(results["baseline"], baseline_dir, "metrics.json")
    del baseline_model
    torch.cuda.empty_cache()

    # ── Ablation: Head Only ────────────────────────────────
    print("\n[3/5] Evaluating Ablation (Head Only)...")
    ablation_model = load_model_with_checkpoint(
        args.ablation_ckpt, "ablation", device
    )
    preds, targets, ms = collect_predictions(
        ablation_model, test_loader, device, use_fp16
    )
    results["ablation"] = compute_model_metrics(preds, targets, ms)
    all_preds["ablation"] = preds
    print(f"  IoU: {results['ablation']['mean_iou']:.4f} | "
          f"Latency: {ms:.1f}ms/sample")
    save_results(results["ablation"], ablation_dir, "metrics.json")
    del ablation_model
    torch.cuda.empty_cache()

    # ── Ours: Full Spatial-LLaVA ──────────────────────────
    print("\n[4/5] Evaluating Ours (Spatial-LLaVA)...")
    our_model = load_model_with_checkpoint(
        args.ours_ckpt, "main", device
    )
    preds, _, ms = collect_predictions(
        our_model, test_loader, device, use_fp16
    )
    results["ours"] = compute_model_metrics(preds, targets, ms)
    all_preds["ours"] = preds
    print(f"  IoU: {results['ours']['mean_iou']:.4f} | "
          f"Latency: {ms:.1f}ms/sample")
    save_results(results["ours"], main_dir, "metrics.json")

    # ── Save combined results ──────────────────────────────
    save_results(results, output_dir, "metrics_all.json")

    table = generate_comparison_table(results)
    save_results(table, output_dir, "comparison_table.json")

    # ── Load and merge training curves ────────────────────
    print("\n[5/5] Saving visualizations and training curves...")
    curves = load_training_curves(main_dir, ablation_dir)
    save_results(curves, output_dir, "training_curves.json")

    # ── Visualizations ────────────────────────────────────
    if not args.no_viz:
        save_inference_examples(
            model_preds=all_preds,
            targets=targets,
            loader=test_loader,
            output_dir=output_dir,
            n_examples=args.n_examples,
        )

    del our_model

    # ── Summary ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Results Summary")
    print("=" * 60)
    header = f"  {'Model':12s} {'IoU':>8s} {'RMSE':>8s} {'MAE':>8s} {'ms/sample':>10s}"
    print(header)
    print("  " + "-" * 52)
    for name, m in results.items():
        print(
            f"  {name:12s} "
            f"{m['mean_iou']:8.4f} "
            f"{m['rmse']:8.4f} "
            f"{m['mae']:8.4f} "
            f"{m['inference_time_ms']:10.1f}"
        )
    print("=" * 60)
    print(f"\n  Improvement (ours vs baseline): "
          f"{table['ours']['improvement_vs_baseline']}")
    print(f"  Improvement (ours vs ablation): "
          f"{table['ours']['improvement_vs_ablation']}")
    print(f"\n  All results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Spatial-LLaVA 3-way Evaluation"
    )
    parser.add_argument("--ours_ckpt", type=str, required=True)
    parser.add_argument("--ablation_ckpt", type=str, required=True)
    parser.add_argument(
        "--data_dir", type=str,
        default="~/SharedFolder/MDAIE/group6/data/"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="~/SharedFolder/MDAIE/group6/results/"
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--n_examples", type=int, default=20,
        help="Number of visualization examples to save"
    )
    parser.add_argument(
        "--no_viz", action="store_true",
        help="Skip saving visualization examples"
    )
    args = parser.parse_args()
    main(args)
