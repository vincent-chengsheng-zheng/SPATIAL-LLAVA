"""
tests/test_eval.py

Unit tests for courses/shared/eval.py

Tested (have logic worth testing):
    run_inference_batch()       — core inference logic
    compute_model_metrics()     — metric computation
    compute_improvement()       — percentage calculation
    generate_comparison_table() — table generation + validation

Not tested (pure IO or trivially thin wrappers):
    save_results()              — just json.dump
    save_inference_examples()   — pure IO + visualization
    collect_predictions()       — thin loop over run_inference_batch
    load_model_with_checkpoint()— requires real model weights
    main()                      — requires GPU + data

Run with:
    pytest tests/test_eval.py -v
"""

import pytest
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict


from courses.shared.eval import (
    run_inference_batch,
    compute_model_metrics,
    compute_improvement,
    generate_comparison_table,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_fake_model(batch_size: int = 2) -> nn.Module:
    """Model that returns fixed sigmoid output for any input."""
    class FakeModel(nn.Module):
        def forward(self, images, input_ids, attention_mask=None):
            b = images.shape[0]
            return {"bbox": torch.sigmoid(torch.randn(b, 4))}

    return FakeModel()


def make_batch(batch_size: int = 2) -> Dict[str, Tensor]:
    return {
        "image": torch.randn(batch_size, 3, 384, 384),
        "input_ids": torch.randint(0, 100, (batch_size, 77)),
        "bbox": torch.rand(batch_size, 4),
    }


def make_metrics(iou: float, rmse: float = 0.05, mae: float = 0.03) -> dict:
    return {
        "mean_iou": iou,
        "rmse": rmse,
        "mae": mae,
        "inference_time_ms": 130.0,
        "num_samples": 1000,
    }


# ── run_inference_batch tests ─────────────────────────────────────────────────

class TestRunInferenceBatch:

    def test_returns_tensor(self):
        model = make_fake_model()
        batch = make_batch(2)
        device = torch.device("cpu")
        result = run_inference_batch(model, batch, device, use_fp16=False)
        assert isinstance(result, torch.Tensor)

    def test_output_shape(self):
        model = make_fake_model()
        batch = make_batch(4)
        device = torch.device("cpu")
        result = run_inference_batch(model, batch, device, use_fp16=False)
        assert result.shape == (4, 4)

    def test_output_on_cpu(self):
        """Result should be on CPU regardless of device."""
        model = make_fake_model()
        batch = make_batch(2)
        device = torch.device("cpu")
        result = run_inference_batch(model, batch, device, use_fp16=False)
        assert result.device.type == "cpu"

    def test_values_in_range(self):
        """Sigmoid output should be in [0, 1]."""
        model = make_fake_model()
        batch = make_batch(8)
        device = torch.device("cpu")
        result = run_inference_batch(model, batch, device, use_fp16=False)
        assert (result >= 0).all() and (result <= 1).all()

    def test_no_grad_during_inference(self):
        """No gradients should be computed during inference."""
        model = make_fake_model()
        batch = make_batch(2)
        device = torch.device("cpu")
        result = run_inference_batch(model, batch, device, use_fp16=False)
        assert not result.requires_grad


# ── compute_model_metrics tests ───────────────────────────────────────────────

class TestComputeModelMetrics:

    def test_has_required_keys(self):
        preds = torch.rand(10, 4)
        targets = torch.rand(10, 4)
        metrics = compute_model_metrics(preds, targets, 130.0)
        required = {"mean_iou", "rmse", "mae", "inference_time_ms",
                    "num_samples"}
        assert required.issubset(set(metrics.keys()))

    def test_num_samples_correct(self):
        preds = torch.rand(42, 4)
        targets = torch.rand(42, 4)
        metrics = compute_model_metrics(preds, targets, 100.0)
        assert metrics["num_samples"] == 42

    def test_inference_time_stored(self):
        preds = torch.rand(10, 4)
        targets = torch.rand(10, 4)
        metrics = compute_model_metrics(preds, targets, 157.3)
        assert abs(metrics["inference_time_ms"] - 157.3) < 0.01

    def test_perfect_prediction(self):
        boxes = torch.rand(10, 4)
        metrics = compute_model_metrics(boxes, boxes, 100.0)
        assert abs(metrics["mean_iou"] - 1.0) < 1e-4
        assert abs(metrics["rmse"] - 0.0) < 1e-4

    def test_all_values_are_python_primitives(self):
        """Values must be serializable (no tensors)."""
        preds = torch.rand(10, 4)
        targets = torch.rand(10, 4)
        metrics = compute_model_metrics(preds, targets, 100.0)
        for k, v in metrics.items():
            assert isinstance(v, (int, float)), \
                f"{k} should be int or float, got {type(v)}"


# ── compute_improvement tests ─────────────────────────────────────────────────

class TestComputeImprovement:

    def test_positive_improvement(self):
        result = compute_improvement(baseline_iou=0.45, model_iou=0.67)
        assert result.startswith("+")
        assert "48.9" in result or "48" in result

    def test_negative_improvement(self):
        result = compute_improvement(baseline_iou=0.67, model_iou=0.45)
        assert result.startswith("-")

    def test_zero_improvement(self):
        result = compute_improvement(baseline_iou=0.5, model_iou=0.5)
        assert result == "+0.0%"

    def test_zero_baseline_returns_na(self):
        result = compute_improvement(baseline_iou=0.0, model_iou=0.5)
        assert result == "N/A"

    def test_returns_string(self):
        result = compute_improvement(0.4, 0.6)
        assert isinstance(result, str)
        assert "%" in result


# ── generate_comparison_table tests ──────────────────────────────────────────

class TestGenerateComparisonTable:

    def test_returns_dict(self):
        results = {
            "baseline": make_metrics(0.45),
            "ablation": make_metrics(0.58),
            "ours": make_metrics(0.67),
        }
        table = generate_comparison_table(results)
        assert isinstance(table, dict)

    def test_has_all_three_models(self):
        results = {
            "baseline": make_metrics(0.45),
            "ablation": make_metrics(0.58),
            "ours": make_metrics(0.67),
        }
        table = generate_comparison_table(results)
        assert set(table.keys()) == {"baseline", "ablation", "ours"}

    def test_ours_has_improvement_vs_baseline(self):
        results = {
            "baseline": make_metrics(0.45),
            "ablation": make_metrics(0.58),
            "ours": make_metrics(0.67),
        }
        table = generate_comparison_table(results)
        assert "improvement_vs_baseline" in table["ours"]

    def test_ours_has_improvement_vs_ablation(self):
        results = {
            "baseline": make_metrics(0.45),
            "ablation": make_metrics(0.58),
            "ours": make_metrics(0.67),
        }
        table = generate_comparison_table(results)
        assert "improvement_vs_ablation" in table["ours"]

    def test_baseline_has_no_improvement_field(self):
        """Baseline should not have improvement fields."""
        results = {
            "baseline": make_metrics(0.45),
            "ablation": make_metrics(0.58),
            "ours": make_metrics(0.67),
        }
        table = generate_comparison_table(results)
        assert "improvement_vs_baseline" not in table["baseline"]

    def test_missing_key_raises(self):
        results = {
            "baseline": make_metrics(0.45),
            "ours": make_metrics(0.67),
            # missing "ablation"
        }
        with pytest.raises(KeyError, match="ablation"):
            generate_comparison_table(results)

    def test_improvement_values_are_strings(self):
        results = {
            "baseline": make_metrics(0.45),
            "ablation": make_metrics(0.58),
            "ours": make_metrics(0.67),
        }
        table = generate_comparison_table(results)
        assert isinstance(table["ours"]["improvement_vs_baseline"], str)
        assert isinstance(table["ours"]["improvement_vs_ablation"], str)

    def test_positive_improvement_when_ours_better(self):
        results = {
            "baseline": make_metrics(0.45),
            "ablation": make_metrics(0.58),
            "ours": make_metrics(0.67),
        }
        table = generate_comparison_table(results)
        assert table["ours"]["improvement_vs_baseline"].startswith("+")
        assert table["ours"]["improvement_vs_ablation"].startswith("+")
