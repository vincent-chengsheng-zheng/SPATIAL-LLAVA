"""
tests/test_metrics.py

Unit tests for core/utils/metrics.py

Run with:
    pytest tests/test_metrics.py -v
"""

import torch
import pytest
from core.utils.metrics import iou, mean_iou, rmse, mae, compute_all_metrics


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def perfect_pred():
    """Prediction exactly matches target."""
    boxes = torch.tensor([
        [0.5, 0.5, 0.4, 0.4],
        [0.2, 0.3, 0.2, 0.3],
    ])
    return boxes, boxes.clone()


@pytest.fixture
def no_overlap_pred():
    """Prediction and target have zero overlap."""
    pred = torch.tensor([[0.1, 0.1, 0.1, 0.1]])   # top-left
    target = torch.tensor([[0.9, 0.9, 0.1, 0.1]])  # bottom-right
    return pred, target


@pytest.fixture
def partial_pred():
    """Prediction partially overlaps target."""
    pred = torch.tensor([[0.5, 0.5, 0.4, 0.4]])
    target = torch.tensor([[0.6, 0.6, 0.4, 0.4]])
    return pred, target


# ── IoU tests ─────────────────────────────────────────────────────────────────

class TestIoU:

    def test_perfect_overlap_is_one(self, perfect_pred):
        pred, target = perfect_pred
        scores = iou(pred, target)
        assert torch.allclose(scores, torch.ones(2), atol=1e-5), \
            "Perfect overlap should give IoU = 1.0"

    def test_no_overlap_is_zero(self, no_overlap_pred):
        pred, target = no_overlap_pred
        scores = iou(pred, target)
        assert torch.allclose(scores, torch.zeros(1), atol=1e-5), \
            "No overlap should give IoU = 0.0"

    def test_partial_overlap_between_zero_and_one(self, partial_pred):
        pred, target = partial_pred
        scores = iou(pred, target)
        assert (scores > 0).all() and (scores < 1).all(), \
            "Partial overlap should give IoU in (0, 1)"

    def test_output_shape(self, perfect_pred):
        pred, target = perfect_pred
        scores = iou(pred, target)
        assert scores.shape == (2,), \
            f"Expected shape (2,), got {scores.shape}"

    def test_values_in_valid_range(self, partial_pred):
        pred, target = partial_pred
        scores = iou(pred, target)
        assert (scores >= 0).all() and (scores <= 1).all(), \
            "IoU must be in [0, 1]"

    def test_symmetry(self, partial_pred):
        """IoU(A, B) should equal IoU(B, A)."""
        pred, target = partial_pred
        assert torch.allclose(iou(pred, target), iou(target, pred), atol=1e-5), \
            "IoU should be symmetric"


class TestMeanIoU:

    def test_mean_iou_perfect(self, perfect_pred):
        pred, target = perfect_pred
        score = mean_iou(pred, target)
        assert torch.allclose(score, torch.tensor(1.0), atol=1e-5)

    def test_mean_iou_returns_scalar(self, partial_pred):
        pred, target = partial_pred
        score = mean_iou(pred, target)
        assert score.shape == torch.Size([]), \
            "mean_iou should return a scalar tensor"


# ── RMSE tests ────────────────────────────────────────────────────────────────

class TestRMSE:

    def test_perfect_pred_is_zero(self, perfect_pred):
        pred, target = perfect_pred
        score = rmse(pred, target)
        assert torch.allclose(score, torch.tensor(0.0), atol=1e-5), \
            "RMSE of perfect prediction should be 0"

    def test_rmse_is_nonnegative(self, partial_pred):
        pred, target = partial_pred
        score = rmse(pred, target)
        assert score >= 0, "RMSE must be non-negative"

    def test_rmse_returns_scalar(self, partial_pred):
        pred, target = partial_pred
        score = rmse(pred, target)
        assert score.shape == torch.Size([]), \
            "rmse should return a scalar tensor"

    def test_rmse_increases_with_error(self):
        """Larger prediction error should give larger RMSE."""
        target = torch.tensor([[0.5, 0.5, 0.4, 0.4]])
        small_error = torch.tensor([[0.55, 0.55, 0.4, 0.4]])
        large_error = torch.tensor([[0.8, 0.8, 0.4, 0.4]])
        assert rmse(small_error, target) < rmse(large_error, target), \
            "Larger error should produce larger RMSE"


# ── MAE tests ─────────────────────────────────────────────────────────────────

class TestMAE:

    def test_perfect_pred_is_zero(self, perfect_pred):
        pred, target = perfect_pred
        score = mae(pred, target)
        assert torch.allclose(score, torch.tensor(0.0), atol=1e-5), \
            "MAE of perfect prediction should be 0"

    def test_mae_is_nonnegative(self, partial_pred):
        pred, target = partial_pred
        score = mae(pred, target)
        assert score >= 0, "MAE must be non-negative"

    def test_mae_returns_scalar(self, partial_pred):
        pred, target = partial_pred
        score = mae(pred, target)
        assert score.shape == torch.Size([]), \
            "mae should return a scalar tensor"


# ── compute_all_metrics tests ─────────────────────────────────────────────────

class TestComputeAllMetrics:

    def test_returns_dict_with_correct_keys(self, perfect_pred):
        pred, target = perfect_pred
        result = compute_all_metrics(pred, target)
        assert set(result.keys()) == {"mean_iou", "rmse", "mae"}, \
            f"Unexpected keys: {result.keys()}"

    def test_values_are_python_floats(self, perfect_pred):
        pred, target = perfect_pred
        result = compute_all_metrics(pred, target)
        for key, value in result.items():
            assert isinstance(value, float), \
                f"{key} should be a Python float, got {type(value)}"

    def test_perfect_pred_values(self, perfect_pred):
        pred, target = perfect_pred
        result = compute_all_metrics(pred, target)
        assert abs(result["mean_iou"] - 1.0) < 1e-5
        assert abs(result["rmse"] - 0.0) < 1e-5
        assert abs(result["mae"] - 0.0) < 1e-5
