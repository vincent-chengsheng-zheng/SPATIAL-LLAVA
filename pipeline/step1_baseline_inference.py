"""
pipeline/step1_baseline_inference.py

Step 1 of 4: Baseline inference with Standard LLaVA.

What this script does:
    1. Load pretrained LLaVA-1.5-7B (no modifications, no training)
    2. Load RefCOCO test split (1,975 samples) as PIL images
    3. For each sample:
        a. Format prompt: PROMPT_TEMPLATE.format(text=referring_expression)
        b. Run LLaVA text generation
        c. Parse text output → [xc, yc, w, h] with regex
        d. Record parse success/failure
    4. Compute metrics: mean_iou, rmse, mae, parse_success_rate
    5. Save 20 visualizations (green=GT, red=pred, FAILED if unparseable)
    6. Save results to results/baseline/ AND sync to SharedFolder

Why standard LLaVA is the baseline:
    LLaVA outputs bbox as text → fragile, imprecise, fails often.
    Our Spatial-LLaVA adds an MLP head → direct tensor output, much better.
    This comparison proves our contribution.

Expected results:
    mean_iou         : ~0.15-0.25 (low due to parse failures)
    parse_rate       : ~60-80% (20-40% of outputs cannot be parsed)
    inference_time   : ~400ms/sample on A100
    total_time       : ~13 minutes for 1,975 samples

Usage:
    python pipeline/step1_baseline_inference.py \\
        --data_dir   /tmp/data/ \\
        --output_dir results/baseline/

Output files:
    results/baseline/metrics.json       ← IoU/RMSE/MAE/parse_rate
    results/baseline/predictions.json   ← per-sample predictions
    results/baseline/examples/*.png     ← 20 visualizations
    → Also synced to ~/SharedFolder/MDAIE/group6/results/baseline/
"""

import os
import sys
import json
import time
import shutil
import argparse
import random

import torch

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from core.model.llava import StandardLLaVA               # noqa: E402
from core.data.refcoco_loader_pil import RefCOCODatasetPIL  # noqa: E402
from core.utils.metrics import compute_all_metrics        # noqa: E402
from core.utils.visualization import draw_comparison      # noqa: E402


SHARED_RESULTS = os.path.expanduser(
    "~/SharedFolder/MDAIE/group6/results/baseline/"
)


# ── Inference loop ────────────────────────────────────────────────────────────

def run_inference(
    model: StandardLLaVA,
    dataset: RefCOCODatasetPIL,
    max_new_tokens: int = 50,
) -> list:
    """
    Run LLaVA inference on all samples in the dataset.

    What this does:
        1. Iterate over dataset samples one by one (no DataLoader batching
           because LLaVA processor doesn't support batch PIL input easily)
        2. Call model.generate() for each sample
        3. Call StandardLLaVA.parse_bbox() to extract coordinates
        4. Record results, track parse rate and timing
        5. Print progress every 100 samples with ETA

    Args:
        model          : StandardLLaVA loaded and on device
        dataset        : RefCOCODatasetPIL test split
        max_new_tokens : Max tokens to generate (default: 50)

    Returns:
        List of dicts:
            {
                "idx"       : int   dataset index
                "pred_bbox" : list  [xc,yc,w,h] or [0,0,0,0] if failed
                "gt_bbox"   : list  [xc,yc,w,h] ground truth
                "raw_output": str   raw LLaVA text
                "parsed"    : bool  whether parse succeeded
            }

    Raises:
        RuntimeError : If model.generate() fails on first sample
    """
    results = []
    total = len(dataset)
    parse_success = 0
    t_start = time.time()

    print(f"  Running inference on {total:,} samples...")
    print("  Expected: ~400ms/sample → ~13 min total on A100")

    for i in range(total):

        # Load sample
        try:
            sample = dataset[i]
        except Exception as e:
            print(f"  ⚠ [step1] Failed to load sample idx={i}: {e}")
            results.append({
                "idx": i,
                "pred_bbox": [0.0, 0.0, 0.0, 0.0],
                "gt_bbox": [0.0, 0.0, 0.0, 0.0],
                "raw_output": f"LOAD_ERROR: {e}",
                "parsed": False,
            })
            continue

        pil_image = sample["image"]
        text = sample["text"]
        gt_bbox = sample["bbox"].tolist()

        # Run LLaVA generation
        try:
            raw_output = model.generate(
                pil_image=pil_image,
                text=text,
                max_new_tokens=max_new_tokens,
            )
        except Exception as e:
            print(f"  ⚠ [step1] generate() failed at idx={i}: {e}")
            raw_output = f"GENERATE_ERROR: {e}"

        # Parse bbox from text
        parsed_bbox = StandardLLaVA.parse_bbox(raw_output)
        parsed = parsed_bbox is not None

        if parsed:
            parse_success += 1
            pred_bbox = parsed_bbox
        else:
            pred_bbox = [0.0, 0.0, 0.0, 0.0]  # parse failed → IoU = 0

        results.append({
            "idx": i,
            "pred_bbox": pred_bbox,
            "gt_bbox": gt_bbox,
            "raw_output": raw_output,
            "parsed": parsed,
        })

        # Progress log every 100 samples
        if (i + 1) % 100 == 0 or (i + 1) == total:
            elapsed = time.time() - t_start
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (total - i - 1) / rate
            current_parse_rate = parse_success / (i + 1) * 100
            print(
                f"  [{i+1:,}/{total:,}]  "
                f"parse={current_parse_rate:.1f}%  "
                f"elapsed={elapsed/60:.1f}min  "
                f"ETA={eta/60:.1f}min"
            )

    return results


# ── Save visualizations ───────────────────────────────────────────────────────

def save_visualizations(
    dataset: RefCOCODatasetPIL,
    results: list,
    output_dir: str,
    n: int = 20,
    seed: int = 42,
) -> None:
    """
    Save N example visualizations showing pred bbox vs ground truth.

    What this does:
        1. Sample n/2 parsed + n/2 failed cases (balanced view)
        2. For each: load original image, draw green GT + red pred
        3. Add status label (PARSED ✓ or FAILED ✗) below image
        4. Save as PNG to output_dir/examples/

    How to read the output images:
        - Green box = ground truth (where object actually is)
        - Red box   = LLaVA prediction (only shown if parsing succeeded)
        - FAILED    = LLaVA text couldn't be parsed → IoU = 0
        - Big gap between red and green = why we need Spatial-LLaVA

    Args:
        dataset    : RefCOCODatasetPIL (to get image paths)
        results    : Output from run_inference()
        output_dir : Base results directory
        n          : Total number of examples to save
        seed       : Random seed for sample selection
    """
    examples_dir = os.path.join(output_dir, "examples")
    os.makedirs(examples_dir, exist_ok=True)

    random.seed(seed)

    # Balance parsed and failed
    parsed_results = [r for r in results if r["parsed"]]
    failed_results = [r for r in results if not r["parsed"]]

    n_parsed = min(n // 2, len(parsed_results))
    n_failed = min(n - n_parsed, len(failed_results))

    sampled = (
        random.sample(parsed_results, n_parsed) +
        random.sample(failed_results, n_failed)
    )
    random.shuffle(sampled)

    print(f"  Saving {len(sampled)} visualizations "
          f"({n_parsed} parsed, {n_failed} failed)...")

    saved = 0
    for r in sampled:
        idx = r["idx"]

        try:
            sample = dataset[idx]
        except Exception as e:
            print(f"  ⚠ [step1] Cannot load sample idx={idx} for viz: {e}")
            continue

        pil_image = sample["image"]
        text = sample["text"]

        try:
            canvas = draw_comparison(
                pil_image=pil_image,
                pred_bbox=r["pred_bbox"] if r["parsed"] else None,
                gt_bbox=r["gt_bbox"],
                text=text,
                parsed=r["parsed"],
                model_name="baseline",
            )
        except Exception as e:
            print(f"  ⚠ [step1] draw_comparison failed for idx={idx}: {e}")
            continue

        status = "parsed" if r["parsed"] else "failed"
        fname = os.path.join(
            examples_dir,
            f"baseline_{saved:03d}_{status}_idx{idx}.png"
        )
        try:
            canvas.save(fname)
            saved += 1
        except Exception as e:
            print(f"  ⚠ [step1] Cannot save image {fname}: {e}")

    print(f"  ✓ {saved} examples saved to {examples_dir}")
    print("    Green=GT, Red=Pred, FAILED=parse error → IoU=0")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  Step 1: Baseline Inference (Standard LLaVA)")
    print(f"  Device     : {device}")
    print(f"  Data dir   : {args.data_dir}")
    print(f"  Output dir : {output_dir}")
    print(f"  Model      : {args.model_id}")
    print(f"  HF cache   : {args.hf_cache}")
    print("=" * 60)

    # ── [1/5] Load model ──────────────────────────────────────────
    print("\n[1/5] Loading Standard LLaVA...")
    print("  First time: downloads ~14GB to SharedFolder/hf_cache/")
    print("  Subsequent: loads from cache (~2 min)")
    try:
        model = StandardLLaVA(
            model_id=args.model_id,
            hf_cache=args.hf_cache,
        )
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"\n❌ [step1] Failed to load model: {e}")
        sys.exit(1)

    params = model.count_parameters()
    print(f"  Total params : {params['total_billions']:.1f}B (all frozen)")

    # ── [2/5] Load dataset ────────────────────────────────────────
    print("\n[2/5] Loading test dataset...")
    try:
        test_ds = RefCOCODatasetPIL.from_split(
            data_dir=args.data_dir,
            split="test",
        )
    except Exception as e:
        print(f"\n❌ [step1] Failed to load dataset: {e}")
        sys.exit(1)

    print(f"  Test samples: {len(test_ds):,}")

    # ── [3/5] Run inference ───────────────────────────────────────
    print("\n[3/5] Running inference (~13 min)...")
    try:
        results = run_inference(
            model=model,
            dataset=test_ds,
            max_new_tokens=args.max_new_tokens,
        )
    except Exception as e:
        print(f"\n❌ [step1] Inference failed: {e}")
        sys.exit(1)

    # ── [4/5] Compute metrics ─────────────────────────────────────
    print("\n[4/5] Computing metrics...")
    try:
        preds = torch.tensor([r["pred_bbox"] for r in results])
        targets = torch.tensor([r["gt_bbox"] for r in results])
        metrics = compute_all_metrics(preds, targets)
    except Exception as e:
        print(f"\n❌ [step1] Metrics computation failed: {e}")
        sys.exit(1)

    n_parsed = sum(1 for r in results if r["parsed"])
    n_total = len(results)
    parse_rate = n_parsed / n_total

    metrics["parse_success_rate"] = round(parse_rate, 4)
    metrics["n_parsed"] = n_parsed
    metrics["n_failed"] = n_total - n_parsed
    metrics["num_samples"] = n_total
    metrics["model"] = "baseline_standard_llava"
    metrics["model_id"] = args.model_id
    metrics["description"] = (
        "Standard LLaVA-1.5-7B. Outputs bbox as text, parsed with regex. "
        "No fine-tuning. Failed parses → IoU = 0."
    )

    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    try:
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  ✓ Metrics saved: {metrics_path}")
    except Exception as e:
        print(f"  ⚠ [step1] Cannot save metrics: {e}")

    # Save predictions
    preds_path = os.path.join(output_dir, "predictions.json")
    try:
        with open(preds_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  ✓ Predictions saved: {preds_path}")
    except Exception as e:
        print(f"  ⚠ [step1] Cannot save predictions: {e}")

    # ── [5/5] Visualizations + sync ───────────────────────────────
    print("\n[5/5] Saving visualizations and syncing to SharedFolder...")
    save_visualizations(
        dataset=test_ds,
        results=results,
        output_dir=output_dir,
        n=20,
    )

    try:
        os.makedirs(SHARED_RESULTS, exist_ok=True)
        shutil.copytree(output_dir, SHARED_RESULTS, dirs_exist_ok=True)
        print(f"  ✓ Synced to {SHARED_RESULTS}")
    except Exception as e:
        print(f"  ⚠ [step1] SharedFolder sync failed: {e}")
        print("    Results are still saved locally in results/baseline/")

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Baseline Results Summary")
    print("=" * 60)
    print(f"  Mean IoU       : {metrics['mean_iou']:.4f}")
    print(f"  RMSE           : {metrics['rmse']:.4f}")
    print(f"  MAE            : {metrics['mae']:.4f}")
    print(
        f"  Parse rate     : {parse_rate*100:.1f}% "
        f"({n_parsed}/{n_total} samples)"
    )
    print(
        f"  Failed parse   : {n_total - n_parsed} samples "
        f"(counted as IoU=0)"
    )
    print("=" * 60)
    print("\n✅ Step 1 complete!")
    print(f"   Local results : {output_dir}")
    print(f"   SharedFolder  : {SHARED_RESULTS}")
    print("   Next: python pipeline/step2_train_main.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 1: Baseline inference with Standard LLaVA"
    )
    parser.add_argument(
        "--data_dir", type=str, default="/tmp/data/",
        help="Directory with refcoco_*.pkl (default: /tmp/data/)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/baseline/",
        help="Where to save results (default: results/baseline/)"
    )
    parser.add_argument(
        "--model_id", type=str, default="llava-hf/llava-1.5-7b-hf",
        help="HuggingFace model ID for LLaVA"
    )
    parser.add_argument(
        "--hf_cache", type=str,
        default="~/SharedFolder/MDAIE/group6/hf_cache/",
        help="HuggingFace cache dir (SharedFolder for persistence)"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=50,
        help="Max tokens to generate per sample (default: 50)"
    )
    args = parser.parse_args()
    main(args)
