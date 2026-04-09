"""
pipeline/step2_train_main.py

Step 2: Train SpatialLLaVA — LLaVA backbone + LoRA + MLP regression head.

What this script does:
    1. Load RefCOCO train/val data via refcoco_loader.py
    2. Load LLaVA-1.5-7B + inject LoRA + attach regression head
    3. Train for 10 epochs with AdamW + cosine LR schedule
    4. Save best checkpoint (by val IoU) to checkpoints/main/
    5. Save training metrics + 20 example visualizations

Expected results after 10 epochs:
    val IoU  ~ 0.60 - 0.65
    (vs baseline regex parsing: IoU ~ 0.097)

How to run:
    python pipeline/step2_train_main.py

    # Custom hyperparams:
    python pipeline/step2_train_main.py \\
        --epochs 10 --batch_size 8 --lr 2e-4 --lora_rank 16

    # Resume from latest checkpoint:
    python pipeline/step2_train_main.py --resume

    # Quick smoke test (100 train samples, 1 epoch):
    python pipeline/step2_train_main.py --max_samples 100 --epochs 1

Directory layout after completion:
    checkpoints/main/
        best.pth        ← best val IoU checkpoint
        latest.pth      ← most recent epoch
        history.json    ← per-epoch metrics for plotting
    results/main/
        metrics.json    ← final val metrics
        predictions.json← per-sample predictions on val set
        examples/       ← 20 PNG visualizations
    logs/
        step2_<timestamp>.log
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── sys.path ──────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.paths import PATHS                             # noqa: E402
from core.data.refcoco_loader import make_loaders        # noqa: E402
from core.model.spatial_llava import load_model          # noqa: E402
from core.loss.spatial_loss import SpatialLoss           # noqa: E402
from core.utils.metrics import compute_all_metrics       # noqa: E402
from core.utils.checkpoint import CheckpointManager      # noqa: E402
from core.utils.visualization import display_comparison  # noqa: E402


# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULTS = {
    "epochs": 10,
    "batch_size": 8,
    "lr": 2e-4,
    "lora_rank": 16,
    "weight_decay": 1e-2,
    "max_length": 128,
    "num_workers": 4,
    "log_every": 50,       # log loss every N batches
    "n_examples": 20,      # visualizations to save
    "seed": 42,
}


# ── Seeding ───────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Logging ───────────────────────────────────────────────────────────────────

class Logger:
    """Tee: write to stdout + log file simultaneously."""

    def __init__(self, log_path: str):
        self.log_path = log_path
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


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device: str,
    epoch: int,
    logger: Logger,
    log_every: int,
) -> Dict:
    """
    Run one training epoch.

    What happens:
        1. Forward pass → predicted bbox (B, 4)
        2. Compute SpatialLoss (SmoothL1 + IoU)
        3. Backward + optimizer step
        4. Log batch loss every log_every steps

    Returns:
        {"train_loss": float, "train_iou": float}
    """
    model.train()
    total_loss = 0.0
    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    t0 = time.time()

    for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        targets = batch["bbox"].to(device)

        optimizer.zero_grad()

        preds = model(input_ids, attention_mask, pixel_values)  # (B, 4)
        loss = criterion(preds, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        all_preds.append(preds.detach().cpu())
        all_targets.append(targets.detach().cpu())

        if (step + 1) % log_every == 0:
            elapsed = time.time() - t0
            logger.log(
                f"  Epoch {epoch+1} | step {step+1}/{len(loader)} | "
                f"loss={loss.item():.4f} | {elapsed:.1f}s"
            )

    all_preds_t = torch.cat(all_preds, dim=0)
    all_targets_t = torch.cat(all_targets, dim=0)
    metrics = compute_all_metrics(all_preds_t, all_targets_t)

    return {
        "train_loss": total_loss / len(loader),
        "train_iou": metrics["mean_iou"],
        "train_rmse": metrics["rmse"],
        "train_mae": metrics["mae"],
    }


# ── Validation loop ───────────────────────────────────────────────────────────

def validate(
    model,
    loader,
    criterion,
    device: str,
    epoch: int,
    logger: Logger,
) -> Dict:
    """
    Run validation: no grad, collect predictions, compute metrics.

    Returns:
        {"val_loss": float, "val_iou": float, "val_rmse": float, "val_mae": float}
        + raw preds/targets tensors for visualization
    """
    model.eval()
    total_loss = 0.0
    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    all_image_paths: List[str] = []
    all_texts: List[str] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            targets = batch["bbox"].to(device)

            preds = model(input_ids, attention_mask, pixel_values)
            loss = criterion(preds, targets)

            total_loss += loss.item()
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_image_paths.extend(batch["image_path"])
            all_texts.extend(batch["text"])

    all_preds_t = torch.cat(all_preds, dim=0)
    all_targets_t = torch.cat(all_targets, dim=0)
    metrics = compute_all_metrics(all_preds_t, all_targets_t)

    result = {
        "val_loss": total_loss / len(loader),
        "val_iou": metrics["mean_iou"],
        "val_rmse": metrics["rmse"],
        "val_mae": metrics["mae"],
        # Raw tensors for visualization (not serialized)
        "_preds": all_preds_t,
        "_targets": all_targets_t,
        "_image_paths": all_image_paths,
        "_texts": all_texts,
    }

    logger.log(
        f"  [Val] epoch={epoch+1}  "
        f"loss={result['val_loss']:.4f}  "
        f"IoU={result['val_iou']:.4f}  "
        f"RMSE={result['val_rmse']:.4f}  "
        f"MAE={result['val_mae']:.4f}"
    )
    return result


# ── Save results ──────────────────────────────────────────────────────────────

def save_results(
    val_metrics: Dict,
    history: List[Dict],
    results_dir: Path,
    n_examples: int,
    logger: Logger,
):
    """
    Save metrics JSON, predictions JSON, and example PNG visualizations.

    Args:
        val_metrics  : Last epoch val metrics dict (includes _preds, _targets, etc.)
        history      : List of per-epoch metric dicts
        results_dir  : PATHS.results_main
        n_examples   : How many PNG examples to save
        logger       : Logger instance
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = results_dir / "examples"
    examples_dir.mkdir(exist_ok=True)

    # ── metrics.json ──────────────────────────────────────────────────
    metrics_out = {k: v for k, v in val_metrics.items() if not k.startswith("_")}
    metrics_out["history"] = history
    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    logger.log(f"  Saved metrics → {metrics_path}")

    # ── predictions.json ──────────────────────────────────────────────
    preds_t = val_metrics["_preds"]
    targets_t = val_metrics["_targets"]
    image_paths = val_metrics["_image_paths"]
    texts = val_metrics["_texts"]

    predictions = []
    for i in range(len(preds_t)):
        predictions.append({
            "image_path": image_paths[i],
            "text": texts[i],
            "pred_bbox": preds_t[i].tolist(),
            "gt_bbox": targets_t[i].tolist(),
        })
    preds_path = results_dir / "predictions.json"
    with open(preds_path, "w") as f:
        json.dump(predictions, f, indent=2)
    logger.log(f"  Saved predictions → {preds_path}")

    # ── 20 example PNGs ───────────────────────────────────────────────
    indices = random.sample(range(len(preds_t)), min(n_examples, len(preds_t)))
    saved = 0
    for idx in indices:
        try:
            from PIL import Image
            img = Image.open(image_paths[idx]).convert("RGB")
            pred_box = tuple(preds_t[idx].tolist())
            gt_box = tuple(targets_t[idx].tolist())
            label = f"IoU={_single_iou(preds_t[idx], targets_t[idx]):.3f} | {texts[idx][:40]}"
            vis = display_comparison(img, pred_box, gt_box, label=label)
            out_path = examples_dir / f"example_{idx:04d}.png"
            vis.save(str(out_path))
            saved += 1
        except Exception as e:
            logger.log(f"  ⚠ Visualization failed for idx={idx}: {e}")
    logger.log(f"  Saved {saved} example PNGs → {examples_dir}")


def _single_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute IoU for a single bbox pair."""
    from core.utils.metrics import iou
    return iou(pred.unsqueeze(0), target.unsqueeze(0)).item()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_path = str(PATHS.logs / f"step2_{timestamp}.log")
    logger = Logger(log_path)

    # ── Config dict (saved in checkpoints) ────────────────────────────
    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lora_rank": args.lora_rank,
        "weight_decay": args.weight_decay,
        "max_length": args.max_length,
        "seed": args.seed,
        "use_lora": True,
        "model": "llava-hf/llava-1.5-7b-hf",
    }
    logger.log("=" * 60)
    logger.log("  Step 2: Train SpatialLLaVA (LLaVA + LoRA + Head)")
    logger.log(f"  Config: {json.dumps(config, indent=2)}")
    logger.log("=" * 60)

    # ── Model ─────────────────────────────────────────────────────────
    logger.log("[1/5] Loading model ...")
    model, processor = load_model(use_lora=True, device=device)

    # ── Data ──────────────────────────────────────────────────────────
    logger.log("[2/5] Loading data ...")
    train_loader, val_loader = make_loaders(
        processor=processor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length,
        max_samples=args.max_samples,
        val_max_samples=args.val_max_samples,
    )

    # ── Optimizer + Scheduler ─────────────────────────────────────────
    logger.log("[3/5] Setting up optimizer ...")
    optimizer = AdamW(
        model.trainable_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = SpatialLoss(w_smooth=1.0, w_iou=1.0)

    # ── Checkpoint manager ────────────────────────────────────────────
    ckpt_mgr = CheckpointManager(ckpt_dir=PATHS.ckpt_main, config=config)

    # ── Resume ────────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        logger.log("[3/5] Resuming from latest checkpoint ...")
        info = ckpt_mgr.load_latest(model, optimizer, scheduler, device)
        if info:
            start_epoch = info["epoch"] + 1
            ckpt_mgr.best_iou = info["metrics"].get("val_iou", -1.0)
            logger.log(f"  Resuming from epoch {start_epoch}")

    # ── Training ──────────────────────────────────────────────────────
    logger.log(f"[4/5] Training for {args.epochs} epochs ...")
    history = []
    val_metrics = {}

    for epoch in range(start_epoch, args.epochs):
        logger.log(f"\n{'='*40}")
        logger.log(f"  Epoch {epoch+1}/{args.epochs}")
        logger.log(f"{'='*40}")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, logger, args.log_every
        )
        logger.log(
            f"  [Train] loss={train_metrics['train_loss']:.4f}  "
            f"IoU={train_metrics['train_iou']:.4f}"
        )

        # Val
        val_metrics = validate(
            model, val_loader, criterion, device, epoch, logger
        )

        # LR step
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        logger.log(f"  LR → {lr_now:.2e}")

        # Epoch metrics (exclude raw tensors)
        epoch_metrics = {
            "epoch": epoch + 1,
            "lr": lr_now,
            **train_metrics,
            **{k: v for k, v in val_metrics.items() if not k.startswith("_")},
        }
        history.append(epoch_metrics)

        # Save checkpoint
        ckpt_mgr.save(model, optimizer, scheduler, epoch, epoch_metrics)

    # ── Save results ──────────────────────────────────────────────────
    logger.log("\n[5/5] Saving results ...")
    val_metrics["_history"] = history
    save_results(
        val_metrics=val_metrics,
        history=history,
        results_dir=PATHS.results_main,
        n_examples=args.n_examples,
        logger=logger,
    )

    logger.log("\n✅ Step 2 complete!")
    logger.log(f"  Best val IoU  : {ckpt_mgr.best_iou:.4f}")
    logger.log(f"  Checkpoint    : {ckpt_mgr.best_path}")
    logger.log(f"  Results       : {PATHS.results_main}")
    logger.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 2: Train SpatialLLaVA (LoRA + MLP head)"
    )
    parser.add_argument("--epochs",       type=int,   default=DEFAULTS["epochs"])
    parser.add_argument("--batch_size",   type=int,   default=DEFAULTS["batch_size"])
    parser.add_argument("--lr",           type=float, default=DEFAULTS["lr"])
    parser.add_argument("--lora_rank",    type=int,   default=DEFAULTS["lora_rank"])
    parser.add_argument("--weight_decay", type=float, default=DEFAULTS["weight_decay"])
    parser.add_argument("--max_length",   type=int,   default=DEFAULTS["max_length"])
    parser.add_argument("--num_workers",  type=int,   default=DEFAULTS["num_workers"])
    parser.add_argument("--log_every",    type=int,   default=DEFAULTS["log_every"])
    parser.add_argument("--n_examples",   type=int,   default=DEFAULTS["n_examples"])
    parser.add_argument("--seed",         type=int,   default=DEFAULTS["seed"])
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Cap on train samples (default: all ~42k)"
    )
    parser.add_argument(
        "--val_max_samples", type=int, default=None,
        help="Cap on val samples (default: all ~5k)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoints/main/latest.pth"
    )
    args = parser.parse_args()
    main(args)
