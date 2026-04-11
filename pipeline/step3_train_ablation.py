"""
pipeline/step3_train_ablation.py

Step 3: Ablation — frozen LLaVA backbone + MLP head only (no LoRA).

Purpose:
    Prove that LoRA contributes to performance by comparing:
        Step 3 (head only)    : IoU ~ 0.35-0.45
        Step 2 (LoRA + head)  : IoU ~ 0.45+
    The gap = LoRA's contribution to spatial representation quality.

What is different from step2_train_main.py:
    - use_lora=False          → backbone completely frozen, no LoRA adapters
    - Only head params trained → 2.2M vs 14.8M trainable params
    - ckpt_dir  → checkpoints/ablation/
    - results_dir → results/ablation/

Expected results after 3 epochs:
    val IoU ~ 0.35 - 0.45
    (vs main model with LoRA: IoU ~ 0.45+)

How to run:
    # From terminal:
    TRANSFORMERS_OFFLINE=1 python pipeline/step3_train_ablation.py

    # From notebook (with pre-loaded model):
    import argparse
    from pipeline.step3_train_ablation import main
    args = argparse.Namespace(epochs=3, batch_size=4, ...)
    main(args, model=model, processor=processor)

    # Resume:
    TRANSFORMERS_OFFLINE=1 python pipeline/step3_train_ablation.py --resume

Directory layout after completion:
    checkpoints/ablation/
        best.pth        ← best val IoU checkpoint
        latest.pth      ← most recent epoch
        history.json    ← per-epoch metrics
    results/ablation/
        metrics.json
        predictions.json
        examples/       ← 20 PNG visualizations
    logs/
        step3_<timestamp>.log
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

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

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
    "epochs": 3,
    "batch_size": 4,
    "lr": 2e-4,
    "weight_decay": 0.0,
    "max_length": 600,
    "num_workers": 2,
    "log_every": 200,
    "n_examples": 20,
    "seed": 42,
}


# ── Seeding ───────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Logger ────────────────────────────────────────────────────────────────────

class Logger:
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


# ── Train loop ────────────────────────────────────────────────────────────────

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device: str,
    epoch: int,
    total_epochs: int,
    logger: Logger,
    log_every: int,
) -> Dict:
    model.train()
    total_loss = 0.0
    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    t0 = time.time()

    for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values   = batch["pixel_values"].to(device)
        targets        = batch["bbox"].to(device)

        optimizer.zero_grad()
        preds = model(input_ids, attention_mask, pixel_values)
        loss  = criterion(preds, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        all_preds.append(preds.detach().cpu())
        all_targets.append(targets.detach().cpu())

        if (step + 1) % log_every == 0:
            elapsed = time.time() - t0
            step_done    = step + 1
            total_steps  = len(loader)
            epoch_pct    = step_done / total_steps * 100
            overall_done = epoch * total_steps + step_done
            overall_tot  = total_epochs * total_steps
            overall_pct  = overall_done / overall_tot * 100

            def _bar(pct, w=20):
                f = int(w * pct / 100)
                return f"[{'█'*f}{'░'*(w-f)}] {pct:5.1f}%"

            sps         = step_done / elapsed
            eta_epoch   = (total_steps - step_done) / sps
            eta_overall = (overall_tot - overall_done) / sps

            logger.log(
                f"  Epoch {epoch+1} | step {step_done}/{total_steps} | "
                f"loss={loss.item():.4f} | {elapsed:.1f}s\n"
                f"    Epoch   {_bar(epoch_pct)}  ETA: {eta_epoch/60:.1f}min\n"
                f"    Overall {_bar(overall_pct)}  ETA: {eta_overall/60:.1f}min"
            )

    all_preds_t   = torch.cat(all_preds, dim=0)
    all_targets_t = torch.cat(all_targets, dim=0)
    metrics = compute_all_metrics(all_preds_t, all_targets_t)

    return {
        "train_loss": total_loss / len(loader),
        "train_iou":  metrics["mean_iou"],
        "train_rmse": metrics["rmse"],
        "train_mae":  metrics["mae"],
    }


# ── Val loop ──────────────────────────────────────────────────────────────────

def validate(
    model,
    loader,
    criterion,
    device: str,
    epoch: int,
    logger: Logger,
) -> Dict:
    model.eval()
    total_loss = 0.0
    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    all_image_paths: List[str] = []
    all_texts: List[str] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values   = batch["pixel_values"].to(device)
            targets        = batch["bbox"].to(device)

            preds = model(input_ids, attention_mask, pixel_values)
            loss  = criterion(preds, targets)

            total_loss += loss.item()
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_image_paths.extend(batch["image_path"])
            all_texts.extend(batch["text"])

    all_preds_t   = torch.cat(all_preds, dim=0)
    all_targets_t = torch.cat(all_targets, dim=0)
    metrics = compute_all_metrics(all_preds_t, all_targets_t)

    result = {
        "val_loss": total_loss / len(loader),
        "val_iou":  metrics["mean_iou"],
        "val_rmse": metrics["rmse"],
        "val_mae":  metrics["mae"],
        "_preds":       all_preds_t,
        "_targets":     all_targets_t,
        "_image_paths": all_image_paths,
        "_texts":       all_texts,
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
    results_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = results_dir / "examples"
    examples_dir.mkdir(exist_ok=True)

    # metrics.json
    metrics_out = {k: v for k, v in val_metrics.items() if not k.startswith("_")}
    metrics_out["history"] = history
    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    logger.log(f"  Saved metrics → {metrics_path}")

    # predictions.json
    preds_t      = val_metrics["_preds"]
    targets_t    = val_metrics["_targets"]
    image_paths  = val_metrics["_image_paths"]
    texts        = val_metrics["_texts"]

    predictions = []
    for i in range(len(preds_t)):
        predictions.append({
            "image_path": image_paths[i],
            "text":       texts[i],
            "pred_bbox":  preds_t[i].tolist(),
            "gt_bbox":    targets_t[i].tolist(),
        })
    preds_path = results_dir / "predictions.json"
    with open(preds_path, "w") as f:
        json.dump(predictions, f, indent=2)
    logger.log(f"  Saved predictions → {preds_path}")

    # 20 example PNGs
    indices = random.sample(range(len(preds_t)), min(n_examples, len(preds_t)))
    saved = 0
    for idx in indices:
        try:
            from PIL import Image
            img      = Image.open(image_paths[idx]).convert("RGB")
            pred_box = tuple(preds_t[idx].tolist())
            gt_box   = tuple(targets_t[idx].tolist())
            iou_val  = _single_iou(preds_t[idx], targets_t[idx])
            label    = f"IoU={iou_val:.3f} | {texts[idx][:40]}"
            vis      = display_comparison(img, pred_box, gt_box, label=label)
            out_path = examples_dir / f"example_{idx:04d}.png"
            vis.save(str(out_path))
            saved += 1
        except Exception as e:
            logger.log(f"  ⚠ Visualization failed for idx={idx}: {e}")
    logger.log(f"  Saved {saved} example PNGs → {examples_dir}")


def _single_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    from core.utils.metrics import iou
    return iou(pred.unsqueeze(0), target.unsqueeze(0)).item()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args, model=None, processor=None):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_path  = str(PATHS.logs / f"step3_{timestamp}.log")
    logger    = Logger(log_path)

    config = {
        "epochs":       args.epochs,
        "batch_size":   args.batch_size,
        "lr":           args.lr,
        "weight_decay": args.weight_decay,
        "max_length":   args.max_length,
        "seed":         args.seed,
        "use_lora":     False,          # ← ablation: no LoRA
        "model":        "llava-hf/llava-1.5-7b-hf",
    }

    logger.log("=" * 60)
    logger.log("  Step 3: Ablation — frozen backbone + head only")
    logger.log(f"  Config: {json.dumps(config, indent=2)}")
    logger.log("=" * 60)

    # ── Model ──────────────────────────────────────────────────────────
    logger.log("[1/5] Loading model ...")
    if model is None or processor is None:
        model, processor = load_model(use_lora=False, device=device)
    else:
        logger.log("  Using pre-loaded model, skipping load.")

    # Verify backbone is frozen
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"  Trainable params: {trainable:,} (should be ~2.2M head only)")

    # ── Data ───────────────────────────────────────────────────────────
    logger.log("[2/5] Loading data ...")
    train_loader, val_loader = make_loaders(
        processor=processor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length,
        max_samples=args.max_samples,
        val_max_samples=args.val_max_samples,
    )

    # ── Optimizer ──────────────────────────────────────────────────────
    logger.log("[3/5] Setting up optimizer ...")
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = SpatialLoss(w_l1=1.0, w_giou=1.0)

    # ── Checkpoint manager ─────────────────────────────────────────────
    ckpt_mgr = CheckpointManager(ckpt_dir=PATHS.ckpt_ablation, config=config)

    # ── Resume ─────────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        logger.log("  Resuming from latest checkpoint ...")
        info = ckpt_mgr.load_latest(model, optimizer, scheduler, device)
        if info:
            start_epoch = info["epoch"] + 1
            ckpt_mgr.best_iou = info["metrics"].get("val_iou", -1.0)
            logger.log(f"  Resuming from epoch {start_epoch}")

    # ── Training ───────────────────────────────────────────────────────
    logger.log(f"[4/5] Training for {args.epochs} epochs ...")
    history    = []
    val_metrics = {}

    for epoch in range(start_epoch, args.epochs):
        logger.log(f"\n{'='*40}")
        logger.log(f"  Epoch {epoch+1}/{args.epochs}")
        logger.log(f"{'='*40}")

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, args.epochs, logger, args.log_every
        )
        logger.log(
            f"  [Train] loss={train_metrics['train_loss']:.4f}  "
            f"IoU={train_metrics['train_iou']:.4f}"
        )

        val_metrics = validate(
            model, val_loader, criterion, device, epoch, logger
        )

        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        logger.log(f"  LR → {lr_now:.2e}")

        epoch_metrics = {
            "epoch": epoch + 1,
            "lr":    lr_now,
            **train_metrics,
            **{k: v for k, v in val_metrics.items() if not k.startswith("_")},
        }
        history.append(epoch_metrics)
        ckpt_mgr.save(model, optimizer, scheduler, epoch, epoch_metrics)

    # ── Save results ───────────────────────────────────────────────────
    logger.log("\n[5/5] Saving results ...")
    val_metrics["_history"] = history
    save_results(
        val_metrics=val_metrics,
        history=history,
        results_dir=PATHS.results_ablation,
        n_examples=args.n_examples,
        logger=logger,
    )

    logger.log("\n✅ Step 3 complete!")
    logger.log(f"  Best val IoU  : {ckpt_mgr.best_iou:.4f}")
    logger.log(f"  Checkpoint    : {ckpt_mgr.best_path}")
    logger.log(f"  Results       : {PATHS.results_ablation}")
    logger.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 3: Ablation — frozen backbone + head only"
    )
    parser.add_argument("--epochs",          type=int,   default=DEFAULTS["epochs"])
    parser.add_argument("--batch_size",      type=int,   default=DEFAULTS["batch_size"])
    parser.add_argument("--lr",              type=float, default=DEFAULTS["lr"])
    parser.add_argument("--weight_decay",    type=float, default=DEFAULTS["weight_decay"])
    parser.add_argument("--max_length",      type=int,   default=DEFAULTS["max_length"])
    parser.add_argument("--num_workers",     type=int,   default=DEFAULTS["num_workers"])
    parser.add_argument("--log_every",       type=int,   default=DEFAULTS["log_every"])
    parser.add_argument("--n_examples",      type=int,   default=DEFAULTS["n_examples"])
    parser.add_argument("--seed",            type=int,   default=DEFAULTS["seed"])
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Cap on train samples (default: all ~42k)"
    )
    parser.add_argument(
        "--val_max_samples", type=int, default=500,
        help="Cap on val samples (default: 500)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoints/ablation/latest.pth"
    )
    args = parser.parse_args()
    main(args)
