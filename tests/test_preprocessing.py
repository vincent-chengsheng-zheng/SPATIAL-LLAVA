"""
tests/test_preprocessing.py

Unit tests for core/data/preprocessing.py

Run with:
    pytest tests/test_preprocessing.py -v
Or without pytest:
    python tests/test_preprocessing.py
"""

from core.data.preprocessing import (
    preprocess_image,
    preprocess_image_from_pil,
    preprocess_text,
    normalize_bbox,
    denormalize_bbox,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MAX_LENGTH,
)
import unittest
import tempfile
from PIL import Image
import numpy as np
import torch
import sys
import os
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..")))


# ── Helpers ─────────────────────────────────────────────────────────────

def make_pil_image(width=640, height=480, mode="RGB"):
    """Create a random PIL image for testing."""
    array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    if mode == "L":
        array = array[:, :, 0]
    elif mode == "RGBA":
        alpha = np.full((height, width, 1), 255, dtype=np.uint8)
        array = np.concatenate([array, alpha], axis=-1)
    return Image.fromarray(array, mode=mode)


def make_temp_image(width=640, height=480, mode="RGB", fmt="JPEG"):
    """Save a random PIL image to a temp file and return the path."""
    img = make_pil_image(width, height, mode)
    if mode != "RGB":
        img = img.convert("RGB")  # JPEG doesn't support RGBA/L natively
    suffix = ".jpg" if fmt == "JPEG" else ".png"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    img.save(tmp.name, format=fmt)
    tmp.close()
    return tmp.name


class MockTokenizer:
    """
    Minimal tokenizer mock that mimics HuggingFace tokenizer behaviour.
    Returns deterministic input_ids based on whitespace split.
    """

    def __init__(self, max_length=77):
        self.max_length = max_length

    def __call__(self, text, padding=None, truncation=None,
                 max_length=None, return_tensors=None):
        tokens = text.split()
        length = max_length or self.max_length

        ids = [hash(t) % 30000 for t in tokens]

        # Truncate
        ids = ids[:length]
        # Pad
        ids += [0] * (length - len(ids))

        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}


# ── preprocess_image ────────────────────────────────────────────────────

class TestPreprocessImage(unittest.TestCase):

    def setUp(self):
        self.rgb_path = make_temp_image(640, 480, "RGB", "JPEG")
        self.png_path = make_temp_image(100, 100, "RGB", "PNG")

    def tearDown(self):
        for p in [self.rgb_path, self.png_path]:
            if os.path.exists(p):
                os.remove(p)

    # --- output shape & dtype ---

    def test_output_shape_standard(self):
        t = preprocess_image(self.rgb_path)
        self.assertEqual(t.shape, (3, IMAGE_SIZE, IMAGE_SIZE),
                         "Expected shape (3, 384, 384)")

    def test_output_dtype_float32(self):
        t = preprocess_image(self.rgb_path)
        self.assertEqual(t.dtype, torch.float32,
                         "Output tensor must be float32")

    def test_output_shape_small_png(self):
        """Small image should still be resized to 384x384."""
        t = preprocess_image(self.png_path)
        self.assertEqual(t.shape, (3, IMAGE_SIZE, IMAGE_SIZE))

    # --- normalization sanity ---

    def test_normalization_range(self):
        """
        After ImageNet normalization, values can be outside [0,1]
        but should be roughly within [-3, 3].
        """
        t = preprocess_image(self.rgb_path)
        self.assertGreater(t.min().item(), -4.0)
        self.assertLess(t.max().item(), 4.0)

    def test_normalization_mean_approx(self):
        """
        For a grey (128,128,128) image the per-channel normalized mean
        should be close to (0.5 - mean) / std for each channel.
        """
        grey_array = np.full((100, 100, 3), 128, dtype=np.uint8)
        img = Image.fromarray(grey_array)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(tmp.name)
        tmp.close()

        try:
            t = preprocess_image(tmp.name)
            for c, (m, s) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
                expected = (128 / 255.0 - m) / s
                actual = t[c].mean().item()
                self.assertAlmostEqual(actual, expected, places=1,
                                       msg=f"Channel {c}: expected ~{expected:.3f}, got {actual:.3f}")
        finally:
            os.remove(tmp.name)

    # --- error handling ---

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            preprocess_image("/nonexistent/path/image.jpg")

    # --- mode handling ---

    def test_rgba_image(self):
        """RGBA images should be converted to RGB without error."""
        rgba_array = np.random.randint(0, 256, (80, 80, 4), dtype=np.uint8)
        img = Image.fromarray(rgba_array, "RGBA")
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(tmp.name)
        tmp.close()
        try:
            t = preprocess_image(tmp.name)
            self.assertEqual(t.shape, (3, IMAGE_SIZE, IMAGE_SIZE))
        finally:
            os.remove(tmp.name)

    def test_grayscale_image(self):
        """Grayscale images should be converted to RGB without error."""
        grey = Image.fromarray(
            np.random.randint(0, 256, (80, 80), dtype=np.uint8), "L"
        )
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        grey.save(tmp.name)
        tmp.close()
        try:
            t = preprocess_image(tmp.name)
            self.assertEqual(t.shape, (3, IMAGE_SIZE, IMAGE_SIZE))
        finally:
            os.remove(tmp.name)


# ── preprocess_image_from_pil ───────────────────────────────────────────

class TestPreprocessImageFromPil(unittest.TestCase):

    def test_output_shape(self):
        img = make_pil_image(640, 480, "RGB")
        t = preprocess_image_from_pil(img)
        self.assertEqual(t.shape, (3, IMAGE_SIZE, IMAGE_SIZE))

    def test_output_dtype(self):
        img = make_pil_image(200, 200, "RGB")
        t = preprocess_image_from_pil(img)
        self.assertEqual(t.dtype, torch.float32)

    def test_rgba_input(self):
        img = make_pil_image(100, 100, "RGBA")
        t = preprocess_image_from_pil(img)
        self.assertEqual(t.shape, (3, IMAGE_SIZE, IMAGE_SIZE))

    def test_grayscale_input(self):
        img = make_pil_image(100, 100, "L")
        t = preprocess_image_from_pil(img)
        self.assertEqual(t.shape, (3, IMAGE_SIZE, IMAGE_SIZE))

    def test_consistent_with_preprocess_image(self):
        """Both functions should produce identical tensors for the same image."""
        img = make_pil_image(200, 200, "RGB")
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(tmp.name)
        tmp.close()
        try:
            t_path = preprocess_image(tmp.name)
            t_pil = preprocess_image_from_pil(img)
            self.assertTrue(
                torch.allclose(t_path, t_pil, atol=1e-5),
                "preprocess_image and preprocess_image_from_pil should return identical tensors"
            )
        finally:
            os.remove(tmp.name)

    def test_small_image(self):
        img = make_pil_image(10, 10, "RGB")
        t = preprocess_image_from_pil(img)
        self.assertEqual(t.shape, (3, IMAGE_SIZE, IMAGE_SIZE))

    def test_large_image(self):
        img = make_pil_image(2048, 2048, "RGB")
        t = preprocess_image_from_pil(img)
        self.assertEqual(t.shape, (3, IMAGE_SIZE, IMAGE_SIZE))


# ── preprocess_text ─────────────────────────────────────────────────────

class TestPreprocessText(unittest.TestCase):

    def setUp(self):
        self.tokenizer = MockTokenizer(max_length=MAX_LENGTH)

    def test_output_shape(self):
        ids = preprocess_text("the cat on the mat", self.tokenizer)
        self.assertEqual(ids.shape, (MAX_LENGTH,),
                         f"Expected shape ({MAX_LENGTH},)")

    def test_output_dtype(self):
        ids = preprocess_text("a dog near the fence", self.tokenizer)
        self.assertEqual(ids.dtype, torch.long,
                         "Output must be int64 (long)")

    def test_loc_token_in_prompt(self):
        """
        The formatted prompt must contain [LOC].
        We verify indirectly: tokenizing with vs without [LOC] gives
        different results when the mock hashes tokens.
        """
        ids_with = preprocess_text("the chair", self.tokenizer)
        # Manually tokenize without [LOC] to confirm difference
        raw = "Find the object referred to: the chair"
        enc = self.tokenizer(raw, padding="max_length", truncation=True,
                             max_length=MAX_LENGTH, return_tensors="pt")
        ids_without = enc["input_ids"].squeeze(0).long()
        self.assertFalse(
            torch.equal(ids_with, ids_without),
            "[LOC] token should change the tokenized output"
        )

    def test_empty_prompt(self):
        """Empty string prompt should still return correct shape."""
        ids = preprocess_text("", self.tokenizer)
        self.assertEqual(ids.shape, (MAX_LENGTH,))

    def test_long_prompt_truncated(self):
        """Very long prompts must be truncated to MAX_LENGTH."""
        long_prompt = "word " * 200
        ids = preprocess_text(long_prompt, self.tokenizer)
        self.assertEqual(ids.shape, (MAX_LENGTH,))

    def test_short_prompt_padded(self):
        """Short prompts must be padded to MAX_LENGTH with zeros."""
        ids = preprocess_text("hi", self.tokenizer)
        self.assertEqual(ids.shape, (MAX_LENGTH,))
        # Tail should be zero-padded
        self.assertTrue(
            (ids[-10:] == 0).all(),
            "Short prompts should be zero-padded at the end"
        )

    def test_prompt_format_contains_instruction(self):
        """
        Verify the formatted string starts with the required prefix.
        We do this by checking a known token from the prefix appears.
        """
        # 'Find' should always be in the formatted prompt
        ids_a = preprocess_text("apple", self.tokenizer)
        ids_b = preprocess_text("banana", self.tokenizer)
        # First token (from 'Find') should be identical in both
        self.assertEqual(ids_a[0].item(), ids_b[0].item(),
                         "Both prompts should start with the same prefix token")

    def test_deterministic(self):
        """Same input should always produce the same output."""
        ids1 = preprocess_text("the red ball", self.tokenizer)
        ids2 = preprocess_text("the red ball", self.tokenizer)
        self.assertTrue(torch.equal(ids1, ids2))


# ── normalize_bbox ──────────────────────────────────────────────────────

class TestNormalizeBbox(unittest.TestCase):

    def test_docstring_example(self):
        """Verify the exact example from the docstring."""
        bbox = normalize_bbox([100, 50, 300, 200], img_w=640, img_h=480)
        expected = torch.tensor(
            [0.3125, 0.2604, 0.3125, 0.3125], dtype=torch.float32)
        self.assertTrue(
            torch.allclose(bbox, expected, atol=1e-3),
            f"Expected {expected}, got {bbox}"
        )

    def test_output_shape(self):
        bbox = normalize_bbox([0, 0, 100, 100], img_w=200, img_h=200)
        self.assertEqual(bbox.shape, (4,))

    def test_output_dtype(self):
        bbox = normalize_bbox([0, 0, 100, 100], img_w=200, img_h=200)
        self.assertEqual(bbox.dtype, torch.float32)

    def test_full_image_bbox(self):
        """Bbox covering the full image → [0.5, 0.5, 1.0, 1.0]."""
        bbox = normalize_bbox([0, 0, 640, 480], img_w=640, img_h=480)
        expected = torch.tensor([0.5, 0.5, 1.0, 1.0], dtype=torch.float32)
        self.assertTrue(torch.allclose(bbox, expected, atol=1e-5))

    def test_center_pixel(self):
        """Single-pixel bbox at center."""
        bbox = normalize_bbox([320, 240, 321, 241], img_w=640, img_h=480)
        self.assertAlmostEqual(bbox[0].item(), 320.5 / 640, places=4)
        self.assertAlmostEqual(bbox[1].item(), 240.5 / 480, places=4)
        self.assertAlmostEqual(bbox[2].item(), 1.0 / 640, places=4)
        self.assertAlmostEqual(bbox[3].item(), 1.0 / 480, places=4)

    def test_values_in_unit_range(self):
        """All normalized values must be in [0, 1]."""
        bbox = normalize_bbox([10, 20, 200, 300], img_w=640, img_h=480)
        self.assertTrue((bbox >= 0).all() and (bbox <= 1).all(),
                        f"Values out of [0,1]: {bbox}")

    def test_clamping_overflow(self):
        """Bbox coords beyond image bounds should be clamped to [0,1]."""
        bbox = normalize_bbox([-50, -50, 700, 500], img_w=640, img_h=480)
        self.assertTrue((bbox >= 0).all() and (bbox <= 1).all())

    def test_square_image(self):
        bbox = normalize_bbox([0, 0, 50, 50], img_w=100, img_h=100)
        expected = torch.tensor([0.25, 0.25, 0.50, 0.50], dtype=torch.float32)
        self.assertTrue(torch.allclose(bbox, expected, atol=1e-5))

    def test_non_square_image(self):
        """Width and height normalization should use their respective axes."""
        bbox = normalize_bbox([0, 0, 100, 100], img_w=200, img_h=400)
        # x_c=50/200=0.25, y_c=50/400=0.125, w=100/200=0.5, h=100/400=0.25
        expected = torch.tensor([0.25, 0.125, 0.5, 0.25], dtype=torch.float32)
        self.assertTrue(torch.allclose(bbox, expected, atol=1e-5))


# ── denormalize_bbox ────────────────────────────────────────────────────

class TestDenormalizeBbox(unittest.TestCase):

    def test_output_type(self):
        bbox_norm = torch.tensor([0.5, 0.5, 0.3, 0.4], dtype=torch.float32)
        result = denormalize_bbox(bbox_norm, img_w=640, img_h=480)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)
        for v in result:
            self.assertIsInstance(v, int)

    def test_center_full(self):
        """[0.5, 0.5, 1.0, 1.0] → full image bbox [0, 0, W, H]."""
        bbox_norm = torch.tensor([0.5, 0.5, 1.0, 1.0], dtype=torch.float32)
        x1, y1, x2, y2 = denormalize_bbox(bbox_norm, img_w=640, img_h=480)
        self.assertEqual((x1, y1, x2, y2), (0, 0, 640, 480))

    def test_roundtrip(self):
        """normalize → denormalize should recover original bbox (within 1px)."""
        original = [100, 50, 300, 200]
        norm = normalize_bbox(original, img_w=640, img_h=480)
        x1, y1, x2, y2 = denormalize_bbox(norm, img_w=640, img_h=480)
        for got, exp in zip([x1, y1, x2, y2], original):
            self.assertAlmostEqual(got, exp, delta=1,
                                   msg=f"Round-trip failed: expected {exp}, got {got}")

    def test_clamping_low(self):
        """Normalized values that would go negative are clamped to 0."""
        bbox_norm = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
        x1, y1, x2, y2 = denormalize_bbox(bbox_norm, img_w=640, img_h=480)
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)

    def test_clamping_high(self):
        """Values that exceed image bounds are clamped to img_w / img_h."""
        bbox_norm = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
        x1, y1, x2, y2 = denormalize_bbox(bbox_norm, img_w=640, img_h=480)
        self.assertLessEqual(x2, 640)
        self.assertLessEqual(y2, 480)

    def test_small_box(self):
        bbox_norm = torch.tensor([0.5, 0.5, 0.1, 0.1], dtype=torch.float32)
        x1, y1, x2, y2 = denormalize_bbox(bbox_norm, img_w=100, img_h=100)
        self.assertLess(x1, x2)
        self.assertLess(y1, y2)

    def test_multiple_roundtrips(self):
        """Test several bboxes to ensure consistent roundtrip accuracy."""
        cases = [
            ([10, 10, 50, 50], 200, 200),
            ([0, 0, 640, 480], 640, 480),
            ([300, 200, 400, 350], 640, 480),
        ]
        for bbox, w, h in cases:
            norm = normalize_bbox(bbox, img_w=w, img_h=h)
            x1, y1, x2, y2 = denormalize_bbox(norm, img_w=w, img_h=h)
            for got, exp in zip([x1, y1, x2, y2], bbox):
                self.assertAlmostEqual(got, exp, delta=1,
                                       msg=f"Roundtrip failed for {bbox} on {w}x{h}: {got} != {exp}")


# ── Integration ─────────────────────────────────────────────────────────

class TestIntegration(unittest.TestCase):
    """End-to-end checks combining multiple preprocessing steps."""

    def test_image_and_bbox_pipeline(self):
        """
        Simulate loading an image + normalizing its annotation bbox.
        Both should be tensors of the right shape/dtype.
        """
        img_path = make_temp_image(640, 480, "RGB", "JPEG")
        try:
            img_tensor = preprocess_image(img_path)
            bbox_norm = normalize_bbox(
                [100, 50, 300, 200], img_w=640, img_h=480)

            self.assertEqual(img_tensor.shape, (3, 384, 384))
            self.assertEqual(bbox_norm.shape, (4,))
            self.assertEqual(img_tensor.dtype, torch.float32)
            self.assertEqual(bbox_norm.dtype, torch.float32)
        finally:
            os.remove(img_path)

    def test_text_and_image_pipeline(self):
        """
        Simulate a single sample: image tensor + tokenized prompt.
        """
        tokenizer = MockTokenizer()
        img = make_pil_image(320, 240, "RGB")
        img_tensor = preprocess_image_from_pil(img)
        ids = preprocess_text("the person on the left", tokenizer)

        self.assertEqual(img_tensor.shape, (3, 384, 384))
        self.assertEqual(ids.shape, (MAX_LENGTH,))

    def test_full_sample_pipeline(self):
        """
        Full pipeline: image + text + bbox → all tensors correct.
        """
        tokenizer = MockTokenizer()
        img = make_pil_image(800, 600, "RGBA")  # intentional RGBA

        img_tensor = preprocess_image_from_pil(img)
        ids = preprocess_text("the red car in the background", tokenizer)
        bbox_norm = normalize_bbox([200, 100, 600, 500], img_w=800, img_h=600)
        x1, y1, x2, y2 = denormalize_bbox(bbox_norm, img_w=800, img_h=600)

        self.assertEqual(img_tensor.shape, (3, 384, 384))
        self.assertEqual(ids.shape, (MAX_LENGTH,))
        self.assertEqual(bbox_norm.shape, (4,))
        self.assertIsInstance(x1, int)

        # Sanity: denormalized coords should be within original image
        self.assertGreaterEqual(x1, 0)
        self.assertLessEqual(x2, 800)
        self.assertGreaterEqual(y1, 0)
        self.assertLessEqual(y2, 600)


# ── Entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
