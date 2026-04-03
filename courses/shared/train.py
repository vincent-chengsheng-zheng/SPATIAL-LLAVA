"""
courses/shared/train.py

Unified training script for both 61.502 and 51.511 courses.

Two modes:
    main     — LoRA + regression head (10 epochs) → used by both courses
    ablation — head only, LLM frozen   (3 epochs) → ablation for 51.511

Usage (on cluster):
    # Main training (Member 4 / 61.502)
    python courses/shared/train.py \\
        --mode main \\
        --config courses/shared/config.yaml \\
        --data_dir ~/SharedFolder/MDAIE/group6/data/ \\
        --output_dir ~/SharedFolder/MDAIE/group6/checkpoints/main/

    # Ablation training (Member 3 / 51.511)
    python courses/shared/train.py \\
        --mode ablation \\
        --config courses/shared/config.yaml \\
        --data_dir ~/SharedFolder/MDAIE/group6/data/ \\
        --output_dir ~/SharedFolder/MDAIE/group6/checkpoints/ablation/

    # Resume from checkpoint
    python courses/shared/train.py \\
        --mode main \\
        --resume ~/SharedFolder/MDAIE/group6/checkpoints/main/latest.pth
"""

import os
import json
import argparse
import random
import time

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

from core.model.spatial_llava import SpatialLLaVA
from core.data.refcoco_loader import RefCOCODataset
from core.loss.spatial_loss import spatial_loss
from core.utils.checkpoint import save_checkpoint, load_checkpoint, save_best
from core.utils.metrics import compute_all_metrics


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(config_path: str, mode: str) -> dict:
    """
    Load YAML config and return merged config for the given mode.

    Args:
        config_path : Path to config.yaml
        mode        : "main" or "ablation"

    Returns:
        Merged dict: base config overridden by mode-specific values
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    base = cfg.get("base", {})
    mode_cfg = cfg.get(mode, {})

    merged = {**base}
    for k, v in mode_cfg.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = {**merged[k], **v}
        else:
            merged[k] = v

    return merged


# ── Training log ──────────────────────────────────────────────────────────────

class TrainingLogger:
    """
    Logs training metrics per epoch to a JSON file.
    Used to generate loss curves and training history.

    Output format (training_log.json):
    {
        "mode": "main",
        "epochs": [
            {
                "epoch": 1,
                "train_loss": 0.45,
                "val_iou": 0.52,
                "val_rmse": 0.08,
                "val_mae": 0.06,
                "lr": 1e-4,
                "epoch_time_s": 1234.5,
                "timestamp": "2026-04-01 10:00:00"
            },
            ...
        ]
    }
    """

    def __init__(self, log_path: str, mode: str):
        self.log_path = log_path
        self.data = {"mode": mode, "epochs": []}
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_metrics: dict,
        lr: float,
        epoch_time_s: float,
    ) -> None:
        """Append one epoch record and save to disk."""
        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_iou": round(val_metrics.get("mean_iou", 0.0), 6),
            "val_rmse": round(val_metrics.get("rmse", 0.0), 6),
            "val_mae": round(val_metrics.get("mae", 0.0), 6),
            "lr": lr,
            "epoch_time_s": round(epoch_time_s, 1),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.data["epochs"].append(record)
        with open(self.log_path, "w") as f:
            json.dump(self.data, f, indent=2)

    def summary(self) -> dict:
        """Return best epoch summary."""
        if not self.data["epochs"]:
            return {}
        best = max(self.data["epochs"], key=lambda x: x["val_iou"])
        return {
            "best_epoch": best["epoch"],
            "best_val_iou": best["val_iou"],
            "best_val_rmse": best["val_rmse"],
            "total_epochs": len(self.data["epochs"]),
        }


# ── Dataset ───────────────────────────────────────────────────────────────────

def build_dataloaders(config: dict, data_dir: str):
    """Build train and val DataLoaders from config."""
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"].get("num_workers", 2)

    train_dataset = RefCOCODataset.from_config(
        {"data_dir": data_dir}, split="train"
    )
    val_dataset = RefCOCODataset.from_config(
        {"data_dir": data_dir}, split="val"
    )

    train_loader = train_dataset.get_dataloader(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = val_dataset.get_dataloader(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(config: dict, mode: str) -> SpatialLLaVA:
    """
    Build SpatialLLaVA model for the given training mode.

    main     : LoRA enabled, vision encoder frozen
    ablation : LoRA disabled, LLM fully frozen (head only)
    """
    lora_rank = config["model"].get("lora_rank", 16)

    model = SpatialLLaVA(
        model_id=config["model"].get("model_id", "llava-hf/llava-1.5-7b-hf"),
        lora_rank=lora_rank if mode == "main" else 0,
        freeze_vision=True,
        freeze_llm=(mode == "ablation"),
    )

    params = model.count_parameters()
    print(f"  Model ready: {params['trainable']:,} trainable params "
          f"({params['trainable_percent']:.3f}%)")

    return model


# ── Optimizer & Scheduler ─────────────────────────────────────────────────────

def build_optimizer_and_scheduler(model, config: dict, total_steps: int):
    """Build AdamW optimizer with linear warmup + cosine decay."""
    lr = config["optimizer"].get("learning_rate", 1e-4)
    warmup_steps = config["optimizer"].get("warmup_steps", 500)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=config["optimizer"].get("weight_decay", 0.01),
    )

    warmup = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
    )
    cosine = CosineAnnealingLR(
        optimizer, T_max=max(total_steps - warmup_steps, 1)
    )
    scheduler = SequentialLR(
        optimizer, [warmup, cosine], milestones=[warmup_steps]
    )

    return optimizer, scheduler


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    config: dict,
    epoch: int,
    global_step: int,
    device: torch.device,
    scaler,
) -> tuple:
    """
    Run one epoch of training.

    Returns:
        Tuple (global_step, avg_loss)
    """
    model.train()
    grad_clip = config["optimizer"].get("gradient_clip", 1.0)
    accum_steps = config["training"].get("gradient_accumulation_steps", 1)
    log_interval = config["training"].get("log_interval", 50)
    use_fp16 = config["hardware"].get("fp16", True)

    optimizer.zero_grad()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        targets = batch["bbox"].to(device)

        with torch.cuda.amp.autocast(enabled=use_fp16):
            outputs = model(images, input_ids)
            pred_bbox = outputs["bbox"]
            loss = spatial_loss(pred_bbox, targets)
            loss = loss / accum_steps

        if use_fp16 and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_loss += loss.item() * accum_steps
        num_batches += 1

        if (batch_idx + 1) % accum_steps == 0:
            if use_fp16 and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip
                )
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % log_interval == 0:
                print(f"  Epoch {epoch} | Step {global_step} | "
                      f"Loss {loss.item() * accum_steps:.4f} | "
                      f"LR {scheduler.get_last_lr()[0]:.2e}")

    avg_loss = total_loss / max(num_batches, 1)
    return global_step, avg_loss


def evaluate(
    model,
    loader,
    device: torch.device,
    use_fp16: bool = True,
) -> dict:
    """Run evaluation and return metrics dict."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            targets = batch["bbox"].to(device)

            with torch.cuda.amp.autocast(enabled=use_fp16):
                outputs = model(images, input_ids)

            all_preds.append(outputs["bbox"].cpu())
            all_targets.append(targets.cpu())

    preds = torch.cat(all_preds, dim=0)
    tgts = torch.cat(all_targets, dim=0)

    return compute_all_metrics(preds, tgts)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    # ── Setup ─────────────────────────────────────────────
    config = load_config(args.config, args.mode)
    set_seed(config.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = config["hardware"].get("fp16", True) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Results dir for training log
    results_dir = os.path.expanduser(
        args.results_dir or os.path.join(
            os.path.dirname(output_dir.rstrip("/")),
            "..", "results", args.mode
        )
    )
    os.makedirs(results_dir, exist_ok=True)

    log_dir = os.path.expanduser(
        args.log_dir or os.environ.get("SPATIAL_LOGS", output_dir)
    )
    os.makedirs(log_dir, exist_ok=True)

    num_epochs = config["training"]["num_epochs"]

    # ── Training logger ────────────────────────────────────
    log_path = os.path.join(results_dir, "training_log.json")
    training_logger = TrainingLogger(log_path, args.mode)

    print("=" * 60)
    print("  Spatial-LLaVA Training")
    print(f"  Mode       : {args.mode}")
    print(f"  Epochs     : {num_epochs}")
    print(f"  Device     : {device}")
    print(f"  FP16       : {use_fp16}")
    print(f"  Output     : {output_dir}")
    print(f"  Train log  : {log_path}")
    print("=" * 60)

    # ── Data ──────────────────────────────────────────────
    print("\n[1/4] Loading datasets...")
    train_loader, val_loader = build_dataloaders(config, args.data_dir)
    print(f"  Train: {len(train_loader.dataset):,} samples")
    print(f"  Val:   {len(val_loader.dataset):,} samples")

    # ── Model ─────────────────────────────────────────────
    print("\n[2/4] Building model...")
    model = build_model(config, args.mode).to(device)

    # ── Optimizer ─────────────────────────────────────────
    total_steps = num_epochs * len(train_loader)
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, config, total_steps
    )

    # ── Resume ────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    best_iou = 0.0

    if args.resume:
        resume_path = os.path.expanduser(args.resume)
        start_epoch, global_step, best_iou = load_checkpoint(
            model, optimizer, resume_path
        )
        start_epoch += 1

    # ── Training loop ──────────────────────────────────────
    print("\n[3/4] Training...")
    eval_interval = config["training"].get("eval_interval", 1)
    save_interval = config["training"].get("save_interval", 500)

    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        global_step, avg_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            config, epoch + 1, global_step, device, scaler
        )

        epoch_time = time.time() - t0

        # Save periodic checkpoint
        if global_step % save_interval == 0 or epoch == num_epochs - 1:
            ckpt_path = os.path.join(output_dir, f"epoch_{epoch + 1}.pth")
            save_checkpoint(
                model, optimizer, epoch, global_step, best_iou, ckpt_path
            )

        # Evaluate
        val_metrics = {}
        if (epoch + 1) % eval_interval == 0:
            val_metrics = evaluate(model, val_loader, device, use_fp16)
            current_iou = val_metrics["mean_iou"]
            print(f"  Val IoU: {current_iou:.4f} | "
                  f"RMSE: {val_metrics['rmse']:.4f} | "
                  f"MAE: {val_metrics['mae']:.4f} | "
                  f"Time: {epoch_time:.1f}s")

            best_iou = save_best(
                model, optimizer, epoch, global_step,
                current_iou, best_iou, output_dir
            )

        # Log epoch to training_log.json
        training_logger.log_epoch(
            epoch=epoch + 1,
            train_loss=avg_loss,
            val_metrics=val_metrics,
            lr=scheduler.get_last_lr()[0],
            epoch_time_s=epoch_time,
        )

    # ── Done ───────────────────────────────────────────────
    summary = training_logger.summary()
    print("\n[4/4] Training complete!")
    print(f"  Best Val IoU : {summary.get('best_val_iou', best_iou):.4f} "
          f"(epoch {summary.get('best_epoch', '?')})")
    print(f"  Training log : {log_path}")
    print(f"  Best checkpoint : {output_dir}/best.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spatial-LLaVA Training")
    parser.add_argument("--mode", choices=["main", "ablation"], required=True)
    parser.add_argument(
        "--config", type=str, default="courses/shared/config.yaml"
    )
    parser.add_argument(
        "--data_dir", type=str,
        default="~/SharedFolder/MDAIE/group6/data/"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="~/SharedFolder/MDAIE/group6/checkpoints/main/"
    )
    parser.add_argument(
        "--results_dir", type=str, default=None,
        help="Where to save training_log.json "
             "(default: ../results/<mode>/ relative to output_dir)"
    )
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()
    main(args)
