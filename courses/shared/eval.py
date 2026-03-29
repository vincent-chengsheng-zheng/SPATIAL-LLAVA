"""
courses/shared/eval.py

Evaluation script for Spatial-LLaVA — runs all three ablation models
and generates comparison metrics for both courses.

Three models evaluated:
    baseline  — Standard LLaVA outputting bbox as text tokens
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
    results/inference_examples/     — bbox visualizations
"""

import os
import json
import argparse
import time
from typing import Dict, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from core.model.spatial_llava import SpatialLLaVA
from core.data.refcoco_loader import RefCOCODataset
from core.utils.metrics import compute_all_metrics
from core.utils.checkpoint import load_checkpoint
from core.utils.visualization import display_comparison


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
        with torch.cuda.amp.autocast(enabled=use_fp16 and device.type == "cuda"):
            outputs = model(images, input_ids)

    return outputs["bbox"].cpu()


def collect_predictions(
    model: SpatialLLaVA,
    loader: DataLoader,
    device: torch.device,
    use_fp16: bool = True,
) -> tuple:
    """
    Run inference over the full test set.

    Returns:
        Tuple (preds, targets) — both Tensor (N, 4)
    """
    model.eval()
    all_preds = []
    all_targets = []

    for batch in loader:
        preds = run_inference_batch(model, batch, device, use_fp16)
        all_preds.append(preds)
        all_targets.append(batch["bbox"])

    return torch.cat(all_preds, dim=0), torch.cat(all_targets, dim=0)


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
        inference_time_ms : Average inference time per sample

    Returns:
        Dict with mean_iou, rmse, mae, inference_time_ms, num_samples
    """
    metrics = compute_all_metrics(preds, targets)
    metrics["inference_time_ms"] = round(inference_time_ms, 2)
    metrics["num_samples"] = preds.shape[0]
    return metrics


def compute_improvement(
    baseline_iou: float,
    model_iou: float,
) -> str:
    """
    Compute percentage improvement in IoU over a baseline.

    Args:
        baseline_iou : IoU of the baseline model
        model_iou    : IoU of the model being compared

    Returns:
        String like "+48.9%" or "-5.2%"
    """
    if baseline_iou == 0:
        return "N/A"
    pct = (model_iou - baseline_iou) / baseline_iou * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def generate_comparison_table(
    results: Dict[str, dict],
) -> dict:
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
    model: SpatialLLaVA,
    loader: DataLoader,
    device: torch.device,
    output_dir: str,
    n_examples: int = 20,
    use_fp16: bool = True,
) -> None:
    """
    Save side-by-side pred vs target visualizations for N examples.

    Args:
        model      : SpatialLLaVA model
        loader     : DataLoader (test set)
        device     : Target device
        output_dir : Directory to save PNG files
        n_examples : Number of examples to save
        use_fp16   : Whether to use fp16 autocast
    """
    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    saved = 0

    for batch in loader:
        if saved >= n_examples:
            break

        preds = run_inference_batch(model, batch, device, use_fp16)
        images_np = batch["image"].numpy()
        targets = batch["bbox"]

        for i in range(len(preds)):
            if saved >= n_examples:
                break

            # Denormalize image for display
            img_np = images_np[i].transpose(1, 2, 0)
            img_np = (img_np * 0.225 + 0.45).clip(0, 1)
            img_np = (img_np * 255).astype("uint8")
            pil_img = Image.fromarray(img_np)

            pred_box = tuple(preds[i].tolist())
            target_box = tuple(targets[i].tolist())

            result = display_comparison(pil_img, pred_box, target_box)
            result.save(os.path.join(output_dir, f"example_{saved:03d}.png"))
            saved += 1

    print(f"  Saved {saved} inference examples to {output_dir}")


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
        checkpoint_path : Path to .pth file, or None for fresh model
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

    print("=" * 60)
    print("  Spatial-LLaVA Evaluation")
    print(f"  Device     : {device}")
    print(f"  Output dir : {output_dir}")
    print("=" * 60)

    # ── Test DataLoader ───────────────────────────────────────
    print("\n[1/4] Loading test set...")
    test_dataset = RefCOCODataset.from_config(
        {"data_dir": args.data_dir}, split="test"
    )
    test_loader = test_dataset.get_dataloader(
        batch_size=args.batch_size, shuffle=False
    )
    print(f"  Test samples: {len(test_dataset):,}")

    results = {}

    # ── Baseline: Standard LLaVA ──────────────────────────────
    print("\n[2/4] Evaluating Baseline (Standard LLaVA)...")
    baseline_model = load_model_with_checkpoint(None, "main", device)
    t0 = time.time()
    preds, targets = collect_predictions(baseline_model, test_loader,
                                         device, use_fp16)
    elapsed_ms = (time.time() - t0) / len(test_dataset) * 1000
    results["baseline"] = compute_model_metrics(preds, targets, elapsed_ms)
    print(f"  IoU: {results['baseline']['mean_iou']:.4f}")
    del baseline_model

    # ── Ablation: Head Only ────────────────────────────────────
    print("\n[3/4] Evaluating Ablation (Head Only)...")
    ablation_model = load_model_with_checkpoint(
        args.ablation_ckpt, "ablation", device
    )
    t0 = time.time()
    preds, targets = collect_predictions(ablation_model, test_loader,
                                         device, use_fp16)
    elapsed_ms = (time.time() - t0) / len(test_dataset) * 1000
    results["ablation"] = compute_model_metrics(preds, targets, elapsed_ms)
    print(f"  IoU: {results['ablation']['mean_iou']:.4f}")
    del ablation_model

    # ── Ours: Full Spatial-LLaVA ──────────────────────────────
    print("\n[4/4] Evaluating Ours (Spatial-LLaVA)...")
    our_model = load_model_with_checkpoint(
        args.ours_ckpt, "main", device
    )
    t0 = time.time()
    preds, targets = collect_predictions(our_model, test_loader,
                                         device, use_fp16)
    elapsed_ms = (time.time() - t0) / len(test_dataset) * 1000
    results["ours"] = compute_model_metrics(preds, targets, elapsed_ms)
    print(f"  IoU: {results['ours']['mean_iou']:.4f}")

    # ── Save results ──────────────────────────────────────────
    save_results(results, output_dir, "metrics_all.json")

    table = generate_comparison_table(results)
    save_results(table, output_dir, "comparison_table.json")

    # ── Visualizations ────────────────────────────────────────
    if not args.no_viz:
        viz_dir = os.path.join(output_dir, "inference_examples")
        save_inference_examples(our_model, test_loader, device,
                                viz_dir, n_examples=20,
                                use_fp16=use_fp16)

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Results Summary")
    print("=" * 60)
    for name, m in results.items():
        print(f"  {name:10s} IoU={m['mean_iou']:.4f}  "
              f"RMSE={m['rmse']:.4f}  "
              f"Latency={m['inference_time_ms']:.1f}ms")

    print(f"\n  Improvement (ours vs baseline): "
          f"{table['ours']['improvement_vs_baseline']}")
    print(f"  Improvement (ours vs ablation): "
          f"{table['ours']['improvement_vs_ablation']}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spatial-LLaVA Evaluation")
    parser.add_argument("--ours_ckpt", type=str, required=True)
    parser.add_argument("--ablation_ckpt", type=str, required=True)
    parser.add_argument("--data_dir", type=str,
                        default="~/SharedFolder/MDAIE/group6/data/")
    parser.add_argument("--output_dir", type=str,
                        default="~/SharedFolder/MDAIE/group6/results/")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--no_viz", action="store_true",
                        help="Skip saving visualization examples")
    args = parser.parse_args()
    main(args)
