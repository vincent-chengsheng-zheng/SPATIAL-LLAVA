"""
pipeline/profile_step.py

Profile each substep of the training loop to identify bottlenecks.

Measures per-step:
    - data loading time
    - forward pass time
    - loss computation time
    - backward pass time
    - optimizer step time
    - total step time

Run:
    TRANSFORMERS_OFFLINE=1 python pipeline/profile_step.py
"""

import sys
import os
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.model.spatial_llava import load_model
from core.data.refcoco_loader import make_loaders
from core.loss.spatial_loss import SpatialLoss

N_STEPS = 10  # number of steps to profile

def cuda_time(func):
    """Run func, synchronize CUDA, return elapsed ms."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = func()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) * 1000
    return result, elapsed

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading model...")
model, processor = load_model(use_lora=True, device="cuda")
train_loader, _ = make_loaders(
    processor, batch_size=4, num_workers=0,
    max_length=600, max_samples=60
)
criterion = SpatialLoss()
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=1e-5
)

# ── Profile ───────────────────────────────────────────────────────────────────
print(f"\nProfiling {N_STEPS} steps...\n")
print(f"{'step':>4} {'data':>8} {'forward':>10} {'loss':>8} {'backward':>10} {'optim':>8} {'total':>8}")
print("-" * 60)

data_iter = iter(train_loader)
times = {k: [] for k in ["data", "forward", "loss", "backward", "optim", "total"]}

for step in range(N_STEPS):
    t_total_start = time.perf_counter()

    # 1. Data loading
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        batch = next(data_iter)
    t_data = (time.perf_counter() - t0) * 1000

    input_ids      = batch["input_ids"].cuda()
    attention_mask = batch["attention_mask"].cuda()
    pixel_values   = batch["pixel_values"].cuda()
    targets        = batch["bbox"].cuda()

    optimizer.zero_grad()

    # 2. Forward
    _, t_forward = cuda_time(
        lambda: model(input_ids, attention_mask, pixel_values)
    )
    preds = model(input_ids, attention_mask, pixel_values)

    # 3. Loss
    _, t_loss = cuda_time(lambda: criterion(preds, targets))
    loss = criterion(preds, targets)

    # 4. Backward
    _, t_backward = cuda_time(lambda: loss.backward())

    # 5. Optimizer
    _, t_optim = cuda_time(lambda: optimizer.step())

    torch.cuda.synchronize()
    t_total = (time.perf_counter() - t_total_start) * 1000

    times["data"].append(t_data)
    times["forward"].append(t_forward)
    times["loss"].append(t_loss)
    times["backward"].append(t_backward)
    times["optim"].append(t_optim)
    times["total"].append(t_total)

    print(
        f"{step:>4} "
        f"{t_data:>7.0f}ms "
        f"{t_forward:>9.0f}ms "
        f"{t_loss:>7.0f}ms "
        f"{t_backward:>9.0f}ms "
        f"{t_optim:>7.0f}ms "
        f"{t_total:>7.0f}ms"
    )

# ── Summary ───────────────────────────────────────────────────────────────────
print("-" * 60)
print("AVERAGE (excluding step 0 warmup):")

def avg(lst): return sum(lst[1:]) / len(lst[1:])

print(
    f"{'':>4} "
    f"{avg(times['data']):>7.0f}ms "
    f"{avg(times['forward']):>9.0f}ms "
    f"{avg(times['loss']):>7.0f}ms "
    f"{avg(times['backward']):>9.0f}ms "
    f"{avg(times['optim']):>7.0f}ms "
    f"{avg(times['total']):>7.0f}ms"
)

bottleneck = max(
    ["data", "forward", "loss", "backward", "optim"],
    key=lambda k: avg(times[k])
)
print(f"\nBottleneck: {bottleneck} ({avg(times[bottleneck]):.0f}ms / step)")
print(f"Total avg:  {avg(times['total']):.0f}ms = {avg(times['total'])/1000:.2f}s/step")
