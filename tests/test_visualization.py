
"""
tests/test_visualization.py

Unit tests for core/utils/visualization.py

Run with:
    pytest tests/test_visualization.py -v
"""

import pytest
import numpy as np
from PIL import Image

from core.utils.visualization import _xywh_to_xyxy, draw_bboxes, display_comparison


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_white_image(width: int = 100, height: int = 100) -> Image.Image:
    """Create a blank white RGB image for testing."""
    return Image.new("RGB", (width, height), color=(255, 255, 255))


def image_to_array(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array."""
    return np.array(img)


# ── _xywh_to_xyxy tests ───────────────────────────────────────────────────────

class TestXywhToXyxy:

    def test_center_box(self):
        x1, y1, x2, y2 = _xywh_to_xyxy((0.5, 0.5, 0.4, 0.2))
        assert abs(x1 - 0.3) < 1e-5
        assert abs(y1 - 0.4) < 1e-5
        assert abs(x2 - 0.7) < 1e-5
        assert abs(y2 - 0.6) < 1e-5

    def test_returns_four_values(self):
        result = _xywh_to_xyxy((0.5, 0.5, 0.3, 0.3))
        assert len(result) == 4

    def test_x2_greater_than_x1(self):
        x1, y1, x2, y2 = _xywh_to_xyxy((0.5, 0.5, 0.2, 0.2))
        assert x2 > x1
        assert y2 > y1

    def test_zero_size_box(self):
        x1, y1, x2, y2 = _xywh_to_xyxy((0.5, 0.5, 0.0, 0.0))
        assert x1 == x2
        assert y1 == y2

    def test_accepts_list(self):
        """Should accept list, not just tuple."""
        result = _xywh_to_xyxy([0.5, 0.5, 0.3, 0.3])
        assert len(result) == 4


# ── draw_bboxes tests ─────────────────────────────────────────────────────────

class TestDrawBboxes:

    def test_returns_pil_image(self):
        img = make_white_image()
        result = draw_bboxes(img, [(0.5, 0.5, 0.3, 0.3)], [(0.5, 0.5, 0.4, 0.4)])
        assert isinstance(result, Image.Image)

    def test_output_is_rgb(self):
        img = make_white_image()
        result = draw_bboxes(img, [(0.5, 0.5, 0.3, 0.3)], [(0.5, 0.5, 0.4, 0.4)])
        assert result.mode == "RGB"

    def test_output_size_matches_input(self):
        img = make_white_image(200, 150)
        result = draw_bboxes(img, [(0.5, 0.5, 0.3, 0.3)], [(0.5, 0.5, 0.4, 0.4)])
        assert result.size == (200, 150)

    def test_does_not_modify_input(self):
        """Input image should be unchanged after draw_bboxes."""
        img = make_white_image()
        original = image_to_array(img).copy()
        draw_bboxes(img, [(0.5, 0.5, 0.3, 0.3)], [(0.5, 0.5, 0.4, 0.4)])
        assert np.array_equal(image_to_array(img), original), \
            "Input image should not be modified"

    def test_boxes_are_drawn(self):
        """Drawing boxes on a white image should change some pixels."""
        img = make_white_image(200, 200)
        original = image_to_array(img).copy()
        result = draw_bboxes(img, [(0.5, 0.5, 0.4, 0.4)], [(0.5, 0.5, 0.3, 0.3)])
        result_arr = image_to_array(result)
        assert not np.array_equal(original, result_arr), \
            "Boxes should change at least some pixels"

    def test_grayscale_input_converted_to_rgb(self):
        """RGBA and L mode images should be converted to RGB."""
        img_l = Image.new("L", (100, 100), color=200)
        result = draw_bboxes(img_l, [(0.5, 0.5, 0.3, 0.3)], [(0.5, 0.5, 0.4, 0.4)])
        assert result.mode == "RGB"

        img_rgba = Image.new("RGBA", (100, 100), color=(255, 255, 255, 255))
        result = draw_bboxes(img_rgba, [(0.5, 0.5, 0.3, 0.3)], [(0.5, 0.5, 0.4, 0.4)])
        assert result.mode == "RGB"

    def test_multiple_boxes(self):
        img = make_white_image(300, 300)
        pred_boxes = [(0.2, 0.2, 0.2, 0.2), (0.7, 0.7, 0.2, 0.2)]
        target_boxes = [(0.2, 0.2, 0.3, 0.3), (0.7, 0.7, 0.3, 0.3)]
        result = draw_bboxes(img, pred_boxes, target_boxes)
        assert isinstance(result, Image.Image)
        assert result.size == (300, 300)

    def test_mismatched_lengths_raises(self):
        img = make_white_image()
        with pytest.raises(ValueError, match="same length"):
            draw_bboxes(img, [(0.5, 0.5, 0.3, 0.3)], [])

    def test_with_labels(self):
        img = make_white_image(200, 200)
        result = draw_bboxes(
            img,
            [(0.5, 0.5, 0.4, 0.4)],
            [(0.5, 0.5, 0.3, 0.3)],
            labels=["person"],
        )
        assert isinstance(result, Image.Image)

    def test_with_labels_shorter_than_boxes(self):
        """Labels list shorter than boxes should not crash."""
        img = make_white_image(200, 200)
        result = draw_bboxes(
            img,
            [(0.2, 0.2, 0.2, 0.2), (0.7, 0.7, 0.2, 0.2)],
            [(0.2, 0.2, 0.3, 0.3), (0.7, 0.7, 0.3, 0.3)],
            labels=["only one label"],
        )
        assert isinstance(result, Image.Image)

    def test_custom_colors(self):
        """Different colors should produce different pixel patterns."""
        img1 = make_white_image(200, 200)
        img2 = make_white_image(200, 200)
        box = [(0.5, 0.5, 0.4, 0.4)]

        result_red = draw_bboxes(img1, box, box, colors=("red", "lime"))
        result_blue = draw_bboxes(img2, box, box, colors=("blue", "yellow"))

        arr1 = image_to_array(result_red)
        arr2 = image_to_array(result_blue)
        assert not np.array_equal(arr1, arr2), \
            "Different colors should produce different outputs"

    def test_empty_box_lists(self):
        """Empty box lists should return unchanged image."""
        img = make_white_image()
        original = image_to_array(img).copy()
        result = draw_bboxes(img, [], [])
        assert np.array_equal(image_to_array(result), original)


# ── display_comparison tests ──────────────────────────────────────────────────

class TestDisplayComparison:

    def test_returns_pil_image(self):
        img = make_white_image()
        result = display_comparison(img, (0.5, 0.5, 0.3, 0.3), (0.5, 0.5, 0.4, 0.4))
        assert isinstance(result, Image.Image)

    def test_output_size_matches_input(self):
        img = make_white_image(320, 240)
        result = display_comparison(img, (0.5, 0.5, 0.3, 0.3), (0.5, 0.5, 0.4, 0.4))
        assert result.size == (320, 240)

    def test_draws_on_image(self):
        img = make_white_image(200, 200)
        original = image_to_array(img).copy()
        result = display_comparison(img, (0.5, 0.5, 0.4, 0.4), (0.5, 0.5, 0.3, 0.3))
        assert not np.array_equal(image_to_array(result), original)

    def test_perfect_overlap(self):
        """Identical pred and target should still draw without error."""
        img = make_white_image()
        box = (0.5, 0.5, 0.3, 0.3)
        result = display_comparison(img, box, box)
        assert isinstance(result, Image.Image)
