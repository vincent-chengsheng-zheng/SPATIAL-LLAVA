"""
pipeline/diagnose_nan.py

Systematic NaN diagnostic script.
Traces exactly where NaN first appears in the training loop.

Run:
    python pipeline/diagnose_nan.py
"""

import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.model.spatial_llava import load_model
from core.data.refcoco_loader import make_loaders
from core.loss.spatial_loss import SpatialLoss

SEP = "-" * 50

def check(name, tensor):
    if tensor is None:
        print(f"  {name}: None")
        return
    print(
        f"  {name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"nan={tensor.isnan().any().item()} "
        f"inf={tensor.isinf().any().item()} "
        f"min={tensor.float().min().item():.4f} "
        f"max={tensor.float().max().item():.4f}"
    )

# ── Load model ────────────────────────────────────────────────────────────────
print(SEP)
print("Loading model...")
model, processor = load_model(use_lora=True, device="cuda")
train_loader, _ = make_loaders(
    processor, batch_size=2, num_workers=0,
    max_length=600, max_samples=20
)
criterion = SpatialLoss()
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=1e-5
)

# ── Check initial weights ─────────────────────────────────────────────────────
print(SEP)
print("Initial weight check:")
for name, p in model.named_parameters():
    if p.requires_grad:
        print(
            f"  {name}: nan={p.isnan().any().item()} "
            f"inf={p.isinf().any().item()} "
            f"max={p.float().abs().max().item():.4f}"
        )

# ── Step-by-step training trace ───────────────────────────────────────────────
for step, batch in enumerate(train_loader):
    print(SEP)
    print(f"STEP {step}")

    # 1. Inputs
    print("[1] Inputs:")
    input_ids     = batch["input_ids"].cuda()
    attention_mask = batch["attention_mask"].cuda()
    pixel_values  = batch["pixel_values"].cuda()
    targets       = batch["bbox"].cuda()
    check("input_ids",      input_ids)
    check("attention_mask", attention_mask)
    check("pixel_values",   pixel_values)
    check("targets",        targets)

    # 2. Backbone hidden states (before head)
    print("[2] Backbone hidden states:")
    optimizer.zero_grad()
    pv_cast = pixel_values.to(dtype=next(model.backbone.parameters()).dtype)
    outputs = model.backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pv_cast,
        output_hidden_states=True,
        return_dict=True,
    )
    hs = outputs.hidden_states[-1]   # (B, seq_len, hidden_dim)
    check("hidden_states[-1]", hs)

    # 3. LOC token extraction
    print("[3] LOC token hidden state:")
    seq_lens = attention_mask.sum(dim=1) - 1
    loc_hidden = hs[
        torch.arange(hs.size(0), device=hs.device), seq_lens
    ]
    check("loc_hidden (float16)", loc_hidden)
    loc_hidden_f32 = loc_hidden.float()
    check("loc_hidden (float32)", loc_hidden_f32)

    # 4. Head output
    print("[4] Regression head output:")
    preds = model.head(loc_hidden_f32)
    check("preds", preds)

    # 5. Loss
    print("[5] Loss:")
    loss = criterion(preds, targets)
    print(f"  loss={loss.item()}")

    # 6. Backward
    print("[6] Backward:")
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            g = p.grad
            if g.isnan().any() or g.isinf().any():
                print(
                    f"  !! BAD GRAD: {name} "
                    f"nan={g.isnan().any().item()} "
                    f"inf={g.isinf().any().item()} "
                    f"max={g.float().abs().max().item():.4f}"
                )
            else:
                print(
                    f"  OK grad: {name} "
                    f"norm={g.float().norm().item():.4f} "
                    f"max={g.float().abs().max().item():.6f}"
                )

    # 7. Optimizer step + weight check
    print("[7] After optimizer step:")
    optimizer.step()
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.isnan().any() or p.isinf().any():
                print(
                    f"  !! BAD WEIGHT after step: {name} "
                    f"nan={p.isnan().any().item()} "
                    f"inf={p.isinf().any().item()}"
                )
            else:
                print(
                    f"  OK weight: {name} "
                    f"max={p.float().abs().max().item():.6f}"
                )

    if step >= 2:
        print(SEP)
        print("Done. Check above for first occurrence of nan/inf.")
        break

    