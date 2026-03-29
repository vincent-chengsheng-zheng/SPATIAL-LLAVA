"""
tests/test_spatial_loss.py

Unit tests for core/loss/spatial_loss.py

Run with:
    pytest tests/test_spatial_loss.py -v
"""

import pytest
import torch

from core.loss.spatial_loss import (
    _xywh_to_xyxy,
    iou,
    spatial_loss,
    smooth_l1_loss,
)


class TestXywhToXyxy:

    def test_single_box(self):
        boxes = torch.tensor([[0.5, 0.5, 0.4, 0.3]])
        expected = torch.tensor([[0.3, 0.35, 0.7, 0.65]])
        result = _xywh_to_xyxy(boxes)
        torch.testing.assert_close(result, expected)

    def test_batch(self):
        boxes = torch.tensor([
            [0.5, 0.5, 0.4, 0.3],
            [0.2, 0.3, 0.2, 0.4],
        ])
        expected = torch.tensor([
            [0.3, 0.35, 0.7, 0.65],
            [0.1, 0.1, 0.3, 0.5],
        ])
        torch.testing.assert_close(_xywh_to_xyxy(boxes), expected)

    def test_output_shape(self):
        boxes = torch.randn(8, 4)
        assert _xywh_to_xyxy(boxes).shape == (8, 4)


class TestIoU:

    def test_perfect_overlap(self):
        boxes = torch.tensor([[0.5, 0.5, 0.4, 0.3]])
        score = iou(boxes, boxes)
        assert torch.allclose(score, torch.tensor([1.0]), atol=1e-5)

    def test_no_overlap(self):
        pred = torch.tensor([[0.1, 0.1, 0.15, 0.15]])
        target = torch.tensor([[0.8, 0.8, 0.15, 0.15]])
        score = iou(pred, target)
        assert torch.allclose(score, torch.tensor([0.0]), atol=1e-5)

    def test_partial_overlap(self):
        """
        pred   [x_c=0.4, y_c=0.4, w=0.4, h=0.4] → [0.2, 0.2, 0.6, 0.6]
        target [x_c=0.5, y_c=0.5, w=0.4, h=0.4] → [0.3, 0.3, 0.7, 0.7]
        intersection: [0.3, 0.3, 0.6, 0.6] → area = 0.3*0.3 = 0.09
        union: 0.16 + 0.16 - 0.09 = 0.23
        iou: 0.09 / 0.23
        """
        pred = torch.tensor([[0.4, 0.4, 0.4, 0.4]])
        target = torch.tensor([[0.5, 0.5, 0.4, 0.4]])
        score = iou(pred, target)
        expected = torch.tensor([0.09 / 0.23])
        torch.testing.assert_close(score, expected, rtol=1e-3, atol=1e-5)

    def test_batch(self):
        """Both pred and target must be (B, 4)."""
        pred = torch.tensor([
            [0.5, 0.5, 0.4, 0.3],
            [0.1, 0.1, 0.15, 0.15],
        ])
        target = torch.tensor([
            [0.5, 0.5, 0.4, 0.3],
            [0.8, 0.8, 0.15, 0.15],
        ])
        scores = iou(pred, target)
        assert scores.shape == (2,)
        assert torch.allclose(scores[0], torch.tensor(1.0), atol=1e-5)
        assert torch.allclose(scores[1], torch.tensor(0.0), atol=1e-5)

    def test_output_in_range(self):
        pred = torch.rand(16, 4) * 0.5 + 0.1
        target = torch.rand(16, 4) * 0.5 + 0.1
        scores = iou(pred, target)
        assert (scores >= 0).all() and (scores <= 1 + 1e-5).all()

    def test_zero_sized_box(self):
        """Zero-area pred box → IoU should be near 0."""
        pred = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
        target = torch.tensor([[0.5, 0.5, 0.4, 0.3]])
        score = iou(pred, target)
        assert score.item() < 0.01

    def test_finite_with_tiny_boxes(self):
        """Tiny boxes should not produce NaN or inf."""
        pred = torch.tensor([[0.5, 0.5, 1e-8, 1e-8]])
        target = torch.tensor([[0.5, 0.5, 1e-8, 1e-8]])
        score = iou(pred, target, eps=1e-10)
        assert torch.isfinite(score).all()


class TestSpatialLoss:

    def test_perfect_match_is_zero(self):
        boxes = torch.tensor([[0.5, 0.5, 0.4, 0.3]])
        loss = spatial_loss(boxes, boxes)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)

    def test_positive_with_offset(self):
        pred = torch.tensor([[0.5, 0.5, 0.4, 0.3]])
        target = torch.tensor([[0.52, 0.48, 0.42, 0.32]])
        loss = spatial_loss(pred, target)
        assert 0 < loss.item() < 1.0

    def test_iou_only_weight(self):
        pred = torch.tensor([[0.5, 0.5, 0.4, 0.3]])
        target = torch.tensor([[0.6, 0.6, 0.3, 0.2]])
        loss = spatial_loss(pred, target, iou_weight=1.0, l1_weight=0.0)
        assert loss.item() >= 0

    def test_l1_only_weight(self):
        pred = torch.tensor([[0.5, 0.5, 0.4, 0.3]])
        target = torch.tensor([[0.6, 0.6, 0.3, 0.2]])
        loss = spatial_loss(pred, target, iou_weight=0.0, l1_weight=1.0)
        assert loss.item() >= 0

    def test_returns_scalar(self):
        pred = torch.rand(4, 4)
        target = torch.rand(4, 4)
        loss = spatial_loss(pred, target)
        assert loss.shape == torch.Size([])

    def test_batch(self):
        pred = torch.tensor([
            [0.5, 0.5, 0.4, 0.3],
            [0.1, 0.1, 0.2, 0.2],
        ])
        target = torch.tensor([
            [0.5, 0.5, 0.4, 0.3],
            [0.7, 0.7, 0.2, 0.2],
        ])
        loss = spatial_loss(pred, target)
        assert loss.item() > 0

    def test_shape_mismatch_raises(self):
        pred = torch.tensor([[0.5, 0.5, 0.4, 0.3]])
        target = torch.tensor([[0.5, 0.5, 0.4, 0.3], [0.2, 0.3, 0.2, 0.2]])
        with pytest.raises(AssertionError, match="pred and target must match shape"):
            spatial_loss(pred, target)

    def test_3d_tensor_raises(self):
        pred_3d = torch.tensor([[[0.5, 0.5, 0.4, 0.3]]])
        target_2d = torch.tensor([[0.5, 0.5, 0.4, 0.3]])
        with pytest.raises(AssertionError, match="box tensors must be"):
            spatial_loss(pred_3d, target_2d)

    def test_wrong_dim_raises(self):
        pred = torch.tensor([[0.5, 0.5, 0.4]])
        target = torch.tensor([[0.5, 0.5, 0.4, 0.3]])
        with pytest.raises(AssertionError, match="box tensors must be"):
            spatial_loss(pred, target)


class TestSmoothL1Loss:

    def test_perfect_match_is_zero(self):
        boxes = torch.tensor([[0.5, 0.5, 0.4, 0.3]])
        loss = smooth_l1_loss(boxes, boxes)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)

    def test_positive_with_offset(self):
        pred = torch.tensor([[0.6, 0.6, 0.5, 0.4]])
        target = torch.tensor([[0.5, 0.5, 0.4, 0.3]])
        assert smooth_l1_loss(pred, target).item() > 0

    def test_returns_scalar(self):
        pred = torch.rand(4, 4)
        target = torch.rand(4, 4)
        assert smooth_l1_loss(pred, target).shape == torch.Size([])
