"""
tests/test_data_utils.py

Unit tests for core/data/data_utils.py

Run with:
    pytest tests/test_data_utils.py -v
"""

import torch
from unittest.mock import MagicMock

from core.data.data_utils import collate_fn, compute_dataset_stats, validate_bbox, validate_sample


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_sample(
    image_shape=(3, 384, 384),
    input_ids_shape=(77,),
    bbox=(0.5, 0.5, 0.3, 0.3),
    image_dtype=torch.float32,
    input_ids_dtype=torch.long,
):
    """Build a valid sample dict."""
    return {
        "image": torch.zeros(image_shape, dtype=image_dtype),
        "input_ids": torch.zeros(input_ids_shape, dtype=input_ids_dtype),
        "bbox": torch.tensor(list(bbox), dtype=torch.float32),
    }


def make_mock_dataset(n: int = 5):
    """Build a mock dataset with n identical valid samples."""
    sample = make_sample()
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=n)
    dataset.__getitem__ = MagicMock(return_value=sample)
    return dataset


# ── collate_fn tests ──────────────────────────────────────────────────────────

class TestCollateFn:

    def test_returns_dict(self):
        samples = [make_sample() for _ in range(4)]
        batch = collate_fn(samples)
        assert isinstance(batch, dict)

    def test_has_correct_keys(self):
        samples = [make_sample() for _ in range(4)]
        batch = collate_fn(samples)
        assert set(batch.keys()) == {"image", "input_ids", "bbox"}

    def test_image_batch_shape(self):
        samples = [make_sample() for _ in range(4)]
        batch = collate_fn(samples)
        assert batch["image"].shape == (4, 3, 384, 384)

    def test_input_ids_batch_shape(self):
        samples = [make_sample() for _ in range(4)]
        batch = collate_fn(samples)
        assert batch["input_ids"].shape == (4, 77)

    def test_bbox_batch_shape(self):
        samples = [make_sample() for _ in range(4)]
        batch = collate_fn(samples)
        assert batch["bbox"].shape == (4, 4)

    def test_single_sample(self):
        samples = [make_sample()]
        batch = collate_fn(samples)
        assert batch["image"].shape == (1, 3, 384, 384)

    def test_preserves_dtypes(self):
        samples = [make_sample() for _ in range(2)]
        batch = collate_fn(samples)
        assert batch["image"].dtype == torch.float32
        assert batch["input_ids"].dtype == torch.long
        assert batch["bbox"].dtype == torch.float32

    def test_preserves_values(self):
        """Values should be unchanged after batching."""
        sample = make_sample(bbox=(0.1, 0.2, 0.3, 0.4))
        batch = collate_fn([sample])
        assert torch.allclose(batch["bbox"][0], torch.tensor([0.1, 0.2, 0.3, 0.4]))


# ── validate_bbox tests ───────────────────────────────────────────────────────

class TestValidateBbox:

    def test_valid_bbox(self):
        assert validate_bbox(torch.tensor([0.5, 0.5, 0.3, 0.3])) is True

    def test_invalid_wrong_shape(self):
        assert validate_bbox(torch.tensor([0.5, 0.5, 0.3])) is False
        assert validate_bbox(torch.tensor([0.5, 0.5, 0.3, 0.3, 0.1])) is False

    def test_invalid_negative_value(self):
        assert validate_bbox(torch.tensor([-0.1, 0.5, 0.3, 0.3])) is False

    def test_invalid_value_over_one(self):
        assert validate_bbox(torch.tensor([0.5, 0.5, 1.1, 0.3])) is False

    def test_invalid_zero_width(self):
        assert validate_bbox(torch.tensor([0.5, 0.5, 0.0, 0.3])) is False

    def test_invalid_zero_height(self):
        assert validate_bbox(torch.tensor([0.5, 0.5, 0.3, 0.0])) is False

    def test_boundary_values(self):
        """Values exactly at [0, 1] boundary should be valid."""
        assert validate_bbox(torch.tensor([0.0, 0.0, 0.1, 0.1])) is True
        assert validate_bbox(torch.tensor([0.9, 0.9, 0.1, 0.1])) is True

    def test_very_small_box(self):
        """Tiny but nonzero box should be valid."""
        assert validate_bbox(torch.tensor([0.5, 0.5, 1e-6, 1e-6])) is True


# ── validate_sample tests ─────────────────────────────────────────────────────

class TestValidateSample:

    def test_valid_sample(self):
        assert validate_sample(make_sample()) is True

    def test_missing_key(self):
        sample = make_sample()
        del sample["bbox"]
        assert validate_sample(sample) is False

    def test_wrong_image_shape(self):
        sample = make_sample(image_shape=(3, 224, 224))
        assert validate_sample(sample) is False

    def test_wrong_image_dtype(self):
        sample = make_sample(image_dtype=torch.float64)
        assert validate_sample(sample) is False

    def test_wrong_input_ids_shape(self):
        sample = make_sample(input_ids_shape=(50,))
        assert validate_sample(sample) is False

    def test_wrong_input_ids_dtype(self):
        sample = make_sample(input_ids_dtype=torch.float32)
        assert validate_sample(sample) is False

    def test_invalid_bbox_fails(self):
        sample = make_sample(bbox=(0.5, 0.5, 0.0, 0.3))
        assert validate_sample(sample) is False

    def test_extra_keys_allowed(self):
        """Extra keys in sample dict should not cause failure."""
        sample = make_sample()
        sample["extra"] = torch.tensor([1.0])
        assert validate_sample(sample) is True


# ── compute_dataset_stats tests ───────────────────────────────────────────────

class TestComputeDatasetStats:

    def test_returns_dict(self):
        dataset = make_mock_dataset(n=5)
        stats = compute_dataset_stats(dataset)
        assert isinstance(stats, dict)

    def test_has_correct_keys(self):
        dataset = make_mock_dataset(n=5)
        stats = compute_dataset_stats(dataset)
        expected_keys = {
            "num_samples", "avg_bbox_width", "avg_bbox_height",
            "avg_bbox_area", "avg_text_length"
        }
        assert set(stats.keys()) == expected_keys

    def test_num_samples_correct(self):
        dataset = make_mock_dataset(n=10)
        stats = compute_dataset_stats(dataset)
        assert stats["num_samples"] == 10

    def test_values_are_python_floats(self):
        """All values except num_samples must be Python floats for json.dump()."""
        dataset = make_mock_dataset(n=5)
        stats = compute_dataset_stats(dataset)
        assert isinstance(stats["avg_bbox_width"], float)
        assert isinstance(stats["avg_bbox_height"], float)
        assert isinstance(stats["avg_bbox_area"], float)
        assert isinstance(stats["avg_text_length"], float)

    def test_bbox_stats_correct(self):
        """With fixed bbox [0.5, 0.5, 0.3, 0.4], stats should be exact."""
        sample = make_sample(bbox=(0.5, 0.5, 0.3, 0.4))
        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=3)
        dataset.__getitem__ = MagicMock(return_value=sample)

        stats = compute_dataset_stats(dataset)
        assert abs(stats["avg_bbox_width"] - 0.3) < 1e-5
        assert abs(stats["avg_bbox_height"] - 0.4) < 1e-5
        assert abs(stats["avg_bbox_area"] - 0.12) < 1e-5
