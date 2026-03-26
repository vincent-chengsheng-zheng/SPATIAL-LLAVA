"""
tests/test_checkpoint.py

Unit tests for core/utils/checkpoint.py

Run with:
    pytest tests/test_checkpoint.py -v
"""

import os
import pytest
import torch
import torch.nn as nn
from torch.optim import Adam

from core.utils.checkpoint import save_checkpoint, load_checkpoint, save_best


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_model():
    """A minimal model for testing."""
    return nn.Linear(4, 4)


@pytest.fixture
def simple_optimizer(simple_model):
    """A minimal optimizer for testing."""
    return Adam(simple_model.parameters(), lr=1e-4)


@pytest.fixture
def checkpoint_path(tmp_path):
    """Temporary path for checkpoint files."""
    return str(tmp_path / "checkpoints" / "test_step_100.pth")


@pytest.fixture
def output_dir(tmp_path):
    """Temporary directory for best.pth."""
    d = tmp_path / "checkpoints" / "coursework"
    d.mkdir(parents=True)
    return str(d)


# ── save_checkpoint tests ─────────────────────────────────────────────────────

class TestSaveCheckpoint:

    def test_file_is_created(
        self, simple_model, simple_optimizer, checkpoint_path
    ):
        save_checkpoint(simple_model, simple_optimizer,
                        epoch=1, step=100, best_iou=0.5,
                        path=checkpoint_path)
        assert os.path.exists(checkpoint_path)

    def test_creates_parent_directories(
        self, simple_model, simple_optimizer, tmp_path
    ):
        deep_path = str(tmp_path / "a" / "b" / "c" / "ckpt.pth")
        save_checkpoint(simple_model, simple_optimizer,
                        epoch=0, step=0, best_iou=0.0,
                        path=deep_path)
        assert os.path.exists(deep_path)

    def test_saved_keys_are_correct(
        self, simple_model, simple_optimizer, checkpoint_path
    ):
        save_checkpoint(simple_model, simple_optimizer,
                        epoch=2, step=500, best_iou=0.63,
                        path=checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        assert set(ckpt.keys()) == {
            "epoch", "step", "best_iou",
            "model_state_dict", "optimizer_state_dict"
        }

    def test_saved_values_are_correct(
        self, simple_model, simple_optimizer, checkpoint_path
    ):
        save_checkpoint(simple_model, simple_optimizer,
                        epoch=3, step=1000, best_iou=0.71,
                        path=checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        assert ckpt["epoch"] == 3
        assert ckpt["step"] == 1000
        assert abs(ckpt["best_iou"] - 0.71) < 1e-5


# ── load_checkpoint tests ─────────────────────────────────────────────────────

class TestLoadCheckpoint:

    def test_returns_zero_when_no_file(
        self, simple_model, simple_optimizer, tmp_path
    ):
        missing_path = str(tmp_path / "nonexistent.pth")
        epoch, step, best_iou = load_checkpoint(
            simple_model, simple_optimizer, path=missing_path
        )
        assert epoch == 0
        assert step == 0
        assert best_iou == 0.0

    def test_restores_epoch_step_iou(
        self, simple_model, simple_optimizer, checkpoint_path
    ):
        save_checkpoint(simple_model, simple_optimizer,
                        epoch=5, step=2500, best_iou=0.68,
                        path=checkpoint_path)
        epoch, step, best_iou = load_checkpoint(
            simple_model, simple_optimizer, path=checkpoint_path
        )
        assert epoch == 5
        assert step == 2500
        assert abs(best_iou - 0.68) < 1e-5

    def test_restores_model_weights(
        self, simple_model, simple_optimizer, checkpoint_path
    ):
        with torch.no_grad():
            simple_model.weight.fill_(1.23)
        save_checkpoint(simple_model, simple_optimizer,
                        epoch=1, step=100, best_iou=0.5,
                        path=checkpoint_path)
        with torch.no_grad():
            simple_model.weight.fill_(9.99)
        load_checkpoint(simple_model, simple_optimizer, path=checkpoint_path)
        assert torch.allclose(
            simple_model.weight,
            torch.full_like(simple_model.weight, 1.23),
            atol=1e-5
        )

    def test_return_types_are_python_primitives(
        self, simple_model, simple_optimizer, checkpoint_path
    ):
        save_checkpoint(simple_model, simple_optimizer,
                        epoch=1, step=100, best_iou=0.5,
                        path=checkpoint_path)
        epoch, step, best_iou = load_checkpoint(
            simple_model, simple_optimizer, path=checkpoint_path
        )
        assert isinstance(epoch, int)
        assert isinstance(step, int)
        assert isinstance(best_iou, float)


# ── save_best tests ───────────────────────────────────────────────────────────

class TestSaveBest:

    def test_saves_when_iou_improves(
        self, simple_model, simple_optimizer, output_dir
    ):
        new_best = save_best(simple_model, simple_optimizer,
                             epoch=1, step=500,
                             current_iou=0.65, best_iou=0.60,
                             output_dir=output_dir)
        assert new_best == 0.65
        assert os.path.exists(os.path.join(output_dir, "best.pth"))

    def test_does_not_save_when_no_improvement(
        self, simple_model, simple_optimizer, output_dir
    ):
        new_best = save_best(simple_model, simple_optimizer,
                             epoch=1, step=500,
                             current_iou=0.55, best_iou=0.60,
                             output_dir=output_dir)
        assert new_best == 0.60
        assert not os.path.exists(os.path.join(output_dir, "best.pth"))

    def test_returns_updated_best_iou(
        self, simple_model, simple_optimizer, output_dir
    ):
        best = save_best(simple_model, simple_optimizer,
                         epoch=1, step=100,
                         current_iou=0.70, best_iou=0.65,
                         output_dir=output_dir)
        assert best == 0.70

    def test_returns_unchanged_best_iou_when_no_improvement(
        self, simple_model, simple_optimizer, output_dir
    ):
        best = save_best(simple_model, simple_optimizer,
                         epoch=1, step=100,
                         current_iou=0.60, best_iou=0.65,
                         output_dir=output_dir)
        assert best == 0.65
