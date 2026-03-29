"""
tests/test_train.py

Unit tests for courses/shared/train.py

Strategy:
    - Training requires GPU + 50GB data — too heavy for dev container.
    - We test: config loading, seed, optimizer/scheduler build,
      evaluate() logic, and argument parsing.
    - Training loop itself is tested with a tiny mock (2 steps).

Run with:
    pytest tests/test_train.py -v
"""

import pytest
import torch
import torch.nn as nn
import yaml

from torch.optim import AdamW
from torch.utils.data import DataLoader

from courses.shared.train import (
    set_seed,
    load_config,
    build_optimizer_and_scheduler,
    evaluate,
    train_one_epoch,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def config_file(tmp_path):
    """Write a minimal config.yaml for testing."""
    config = {
        "base": {
            "seed": 42,
            "training": {
                "batch_size": 2,
                "num_epochs": 3,
                "num_workers": 0,
                "gradient_accumulation_steps": 1,
                "log_interval": 1,
                "eval_interval": 1,
                "save_interval": 100,
            },
            "optimizer": {
                "learning_rate": 1e-4,
                "warmup_steps": 2,
                "weight_decay": 0.01,
                "gradient_clip": 1.0,
            },
            "hardware": {
                "fp16": False,
            },
            "model": {
                "model_id": "mock/llava",
                "lora_rank": 4,
            },
        },
        "main": {
            "training": {"num_epochs": 10},
        },
        "ablation": {
            "training": {"num_epochs": 3},
        },
    }
    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(config, f)
    return str(cfg_path)


@pytest.fixture
def base_config(config_file):
    return load_config(config_file, "main")


@pytest.fixture
def tiny_model():
    """A minimal nn.Module that mimics SpatialLLaVA's forward interface."""
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, images, input_ids, attention_mask=None):
            batch_size = images.shape[0]
            bbox = torch.sigmoid(self.linear(torch.randn(batch_size, 4)))
            return {"bbox": bbox}

    return TinyModel()


@pytest.fixture
def tiny_loader():
    """A DataLoader returning small fake batches."""
    n = 6
    images = torch.randn(n, 3, 384, 384)
    input_ids = torch.randint(0, 100, (n, 77))
    bboxes = torch.rand(n, 4)

    class DictDataset(torch.utils.data.Dataset):
        def __len__(self):
            return n

        def __getitem__(self, i):
            return {
                "image": images[i],
                "input_ids": input_ids[i],
                "bbox": bboxes[i],
            }

    return DataLoader(DictDataset(), batch_size=2, shuffle=False)


# ── set_seed tests ────────────────────────────────────────────────────────────

class TestSetSeed:

    def test_reproducible_tensors(self):
        set_seed(42)
        t1 = torch.randn(10)
        set_seed(42)
        t2 = torch.randn(10)
        assert torch.allclose(t1, t2), "Same seed should produce same tensors"

    def test_different_seeds_differ(self):
        set_seed(1)
        t1 = torch.randn(10)
        set_seed(2)
        t2 = torch.randn(10)
        assert not torch.allclose(t1, t2)


# ── load_config tests ─────────────────────────────────────────────────────────

class TestLoadConfig:

    def test_loads_base_config(self, config_file):
        cfg = load_config(config_file, "main")
        assert "training" in cfg
        assert "optimizer" in cfg
        assert "hardware" in cfg

    def test_mode_overrides_base(self, config_file):
        cfg_main = load_config(config_file, "main")
        cfg_ablation = load_config(config_file, "ablation")
        assert cfg_main["training"]["num_epochs"] == 10
        assert cfg_ablation["training"]["num_epochs"] == 3

    def test_base_values_preserved(self, config_file):
        cfg = load_config(config_file, "main")
        assert cfg["training"]["batch_size"] == 2

    def test_missing_mode_returns_base(self, config_file):
        """A mode with no overrides should just return base config."""
        cfg = load_config(config_file, "ablation")
        assert cfg["optimizer"]["learning_rate"] == 1e-4


# ── build_optimizer_and_scheduler tests ──────────────────────────────────────

class TestOptimizerAndScheduler:

    def test_returns_optimizer_and_scheduler(self, tiny_model, base_config):
        optimizer, scheduler = build_optimizer_and_scheduler(
            tiny_model, base_config, total_steps=100
        )
        assert isinstance(optimizer, AdamW)
        assert scheduler is not None

    def test_optimizer_has_correct_lr(self, tiny_model, base_config):
        optimizer, _ = build_optimizer_and_scheduler(
            tiny_model, base_config, total_steps=100
        )
        assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4 * 0.1,
                                                                rel=0.1)

    def test_scheduler_steps(self, tiny_model, base_config):
        """Scheduler should be steppable without errors."""
        optimizer, scheduler = build_optimizer_and_scheduler(
            tiny_model, base_config, total_steps=10
        )
        for _ in range(5):
            optimizer.step()
            scheduler.step()


# ── evaluate tests ────────────────────────────────────────────────────────────

class TestEvaluate:

    def test_returns_dict_with_metrics(self, tiny_model, tiny_loader):
        device = torch.device("cpu")
        metrics = evaluate(tiny_model, tiny_loader, device, use_fp16=False)
        assert set(metrics.keys()) == {"mean_iou", "rmse", "mae"}

    def test_all_values_are_floats(self, tiny_model, tiny_loader):
        device = torch.device("cpu")
        metrics = evaluate(tiny_model, tiny_loader, device, use_fp16=False)
        for key, val in metrics.items():
            assert isinstance(val, float), f"{key} should be float"

    def test_iou_in_range(self, tiny_model, tiny_loader):
        device = torch.device("cpu")
        metrics = evaluate(tiny_model, tiny_loader, device, use_fp16=False)
        assert 0.0 <= metrics["mean_iou"] <= 1.0

    def test_model_in_eval_mode_after(self, tiny_model, tiny_loader):
        """evaluate() should leave model in eval mode."""
        device = torch.device("cpu")
        evaluate(tiny_model, tiny_loader, device, use_fp16=False)
        assert not tiny_model.training


# ── train_one_epoch tests ─────────────────────────────────────────────────────

class TestTrainOneEpoch:

    def test_returns_incremented_step(self, tiny_model, tiny_loader,
                                      base_config):
        device = torch.device("cpu")
        optimizer, scheduler = build_optimizer_and_scheduler(
            tiny_model, base_config, total_steps=100
        )
        initial_step = 0
        new_step = train_one_epoch(
            tiny_model, tiny_loader, optimizer, scheduler,
            base_config, epoch=1, global_step=initial_step,
            device=device, scaler=None,
        )
        assert new_step > initial_step

    def test_model_in_train_mode_during(self, tiny_model, tiny_loader,
                                        base_config):
        """Model should be in train mode during training."""
        device = torch.device("cpu")
        optimizer, scheduler = build_optimizer_and_scheduler(
            tiny_model, base_config, total_steps=100
        )
        # Patch to check training mode
        original_forward = tiny_model.forward

        training_modes = []

        def patched_forward(images, input_ids, attention_mask=None):
            training_modes.append(tiny_model.training)
            return original_forward(images, input_ids, attention_mask)

        tiny_model.forward = patched_forward

        train_one_epoch(
            tiny_model, tiny_loader, optimizer, scheduler,
            base_config, epoch=1, global_step=0,
            device=device, scaler=None,
        )
        assert all(training_modes), "Model should be in train mode"

    def test_weights_change_after_training(self, tiny_model, tiny_loader,
                                           base_config):
        """Weights should change after one epoch."""
        device = torch.device("cpu")
        optimizer, scheduler = build_optimizer_and_scheduler(
            tiny_model, base_config, total_steps=100
        )
        weights_before = tiny_model.linear.weight.clone()
        train_one_epoch(
            tiny_model, tiny_loader, optimizer, scheduler,
            base_config, epoch=1, global_step=0,
            device=device, scaler=None,
        )
        weights_after = tiny_model.linear.weight
        assert not torch.allclose(weights_before, weights_after), \
            "Weights should update after training"
