"""
tests/test_regression_head.py

Unit tests for core/model/regression_head.py

Run with:
    pytest tests/test_regression_head.py -v
"""

import torch
import pytest
from core.model.regression_head import RegressionHead


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def head():
    """Default regression head (hidden_size=4096)."""
    return RegressionHead(hidden_size=4096)


@pytest.fixture
def small_head():
    """Smaller head for faster tests."""
    return RegressionHead(hidden_size=64, intermediate=32)


@pytest.fixture
def batch_hidden(small_head):
    """Random batch of hidden states matching small_head input size."""
    return torch.randn(8, 64)


# ── Output shape tests ────────────────────────────────────────────────────────

class TestOutputShape:

    def test_single_sample(self, small_head):
        hidden = torch.randn(1, 64)
        out = small_head(hidden)
        assert out.shape == (1, 4), \
            f"Expected (1, 4), got {out.shape}"

    def test_batch(self, small_head, batch_hidden):
        out = small_head(batch_hidden)
        assert out.shape == (8, 4), \
            f"Expected (8, 4), got {out.shape}"

    def test_output_has_4_coordinates(self, small_head, batch_hidden):
        out = small_head(batch_hidden)
        assert out.shape[-1] == 4, \
            "Output must have exactly 4 coordinates [x, y, w, h]"


# ── Output range tests ────────────────────────────────────────────────────────

class TestOutputRange:

    def test_all_values_in_zero_one(self, small_head, batch_hidden):
        out = small_head(batch_hidden)
        assert (out >= 0).all() and (out <= 1).all(), \
            "All output values must be in [0, 1] due to Sigmoid"

    def test_output_range_with_extreme_inputs(self, small_head):
        """Sigmoid should clamp even extreme hidden states to [0, 1]."""
        large_hidden = torch.full((4, 64), fill_value=1000.0)
        out = small_head(large_hidden)
        assert (out >= 0).all() and (out <= 1).all()

        neg_hidden = torch.full((4, 64), fill_value=-1000.0)
        out = small_head(neg_hidden)
        assert (out >= 0).all() and (out <= 1).all()


# ── Architecture tests ────────────────────────────────────────────────────────

class TestArchitecture:

    def test_parameter_count_is_small(self):
        """Head should add negligible parameters on top of LLaVA-7B."""
        head = RegressionHead(hidden_size=4096, intermediate=512)
        params = head.count_parameters()
        assert params < 3_000_000, \
            f"Expected < 3M params, got {params:,}"

    def test_parameter_count_correct(self):
        """4096*512 + 512 + 512*4 + 4 = 2,099,716."""
        head = RegressionHead(hidden_size=4096, intermediate=512)
        expected = (4096 * 512 + 512) + (512 * 4 + 4)
        assert head.count_parameters() == expected, \
            f"Expected {expected:,} params, got {head.count_parameters():,}"

    def test_custom_hidden_size(self):
        head = RegressionHead(hidden_size=128, intermediate=64)
        hidden = torch.randn(2, 128)
        out = head(hidden)
        assert out.shape == (2, 4)

    def test_gradients_flow(self, small_head, batch_hidden):
        """Backprop must reach the head's parameters."""
        out = small_head(batch_hidden)
        loss = out.sum()
        loss.backward()
        for name, param in small_head.named_parameters():
            assert param.grad is not None, \
                f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), \
                f"NaN gradient for {name}"

    def test_dropout_off_in_eval_mode(self, small_head, batch_hidden):
        """In eval mode, two forward passes should give identical results."""
        small_head.eval()
        with torch.no_grad():
            out1 = small_head(batch_hidden)
            out2 = small_head(batch_hidden)
        assert torch.allclose(out1, out2), \
            "Eval mode should be deterministic (dropout disabled)"
