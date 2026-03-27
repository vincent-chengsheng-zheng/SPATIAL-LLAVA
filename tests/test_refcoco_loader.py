"""
tests/test_refcoco_loader.py

Unit tests for core/data/refcoco_loader.py

Run with:
    pytest tests/test_refcoco_loader.py -v
Or:
    python tests/test_refcoco_loader.py
"""

from core.data.refcoco_loader import RefCOCODataset, load_splits
import unittest
import tempfile
import pickle
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

def make_fake_sample():
    """Create one fake preprocessed sample matching expected tensor shapes."""
    return {
        "image": torch.rand(3, 384, 384, dtype=torch.float32),
        "input_ids": torch.randint(0, 30000, (77,), dtype=torch.long),
        "bbox": torch.rand(4, dtype=torch.float32).clamp(0, 1),
    }


def make_fake_pkl(num_samples: int = 20) -> str:
    """
    Write a fake pickle file to a temp path and return the path.
    Caller is responsible for cleanup.
    """
    samples = [make_fake_sample() for _ in range(num_samples)]
    tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
    with open(tmp.name, "wb") as f:
        pickle.dump(samples, f)
    tmp.close()
    return tmp.name


def make_fake_pkl_raw_lists(num_samples: int = 10) -> str:
    """
    Fake pkl where tensors are stored as plain Python lists (not torch.Tensor).
    Tests that __getitem__ converts them correctly.
    """
    samples = [
        {
            "image": torch.rand(3, 384, 384).tolist(),
            "input_ids": torch.randint(0, 30000, (77,)).tolist(),
            "bbox": torch.rand(4).clamp(0, 1).tolist(),
        }
        for _ in range(num_samples)
    ]
    tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
    with open(tmp.name, "wb") as f:
        pickle.dump(samples, f)
    tmp.close()
    return tmp.name


# ── RefCOCODataset._load ────────────────────────────────────────────────

class TestLoad(unittest.TestCase):

    def setUp(self):
        self.pkl_path = make_fake_pkl(num_samples=10)

    def tearDown(self):
        if os.path.exists(self.pkl_path):
            os.remove(self.pkl_path)

    def test_loads_successfully(self):
        ds = RefCOCODataset(self.pkl_path, split="train")
        self.assertEqual(len(ds), 10)

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            RefCOCODataset("/nonexistent/path/refcoco_train.pkl")

    def test_empty_pkl_raises(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        with open(tmp.name, "wb") as f:
            pickle.dump([], f)
        tmp.close()
        try:
            with self.assertRaises(AssertionError):
                RefCOCODataset(tmp.name)
        finally:
            os.remove(tmp.name)

    def test_missing_key_raises(self):
        """Samples missing required keys should raise AssertionError."""
        bad_samples = [{"image": torch.rand(
            3, 384, 384)}]  # missing input_ids, bbox
        tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        with open(tmp.name, "wb") as f:
            pickle.dump(bad_samples, f)
        tmp.close()
        try:
            with self.assertRaises(AssertionError):
                RefCOCODataset(tmp.name)
        finally:
            os.remove(tmp.name)

    def test_non_list_pkl_raises(self):
        """Pickle files containing non-list objects should raise AssertionError."""
        tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        with open(tmp.name, "wb") as f:
            pickle.dump({"not": "a list"}, f)
        tmp.close()
        try:
            with self.assertRaises(AssertionError):
                RefCOCODataset(tmp.name)
        finally:
            os.remove(tmp.name)

    def test_split_label_stored(self):
        ds = RefCOCODataset(self.pkl_path, split="val")
        self.assertEqual(ds.split, "val")

    def test_tilde_expansion(self):
        """pkl_path with ~ should expand correctly (uses os.path.expanduser)."""
        # We can't truly test ~ expansion without HOME, so just verify
        # that the stored path has no tilde after init
        ds = RefCOCODataset(self.pkl_path, split="train")
        self.assertNotIn("~", ds.pkl_path)


# ── RefCOCODataset.__len__ ──────────────────────────────────────────────

class TestLen(unittest.TestCase):

    def test_len_matches_samples(self):
        for n in [1, 10, 100]:
            pkl = make_fake_pkl(n)
            try:
                ds = RefCOCODataset(pkl)
                self.assertEqual(len(ds), n,
                                 f"Expected len={n}, got {len(ds)}")
            finally:
                os.remove(pkl)


# ── RefCOCODataset.__getitem__ ──────────────────────────────────────────

class TestGetItem(unittest.TestCase):

    def setUp(self):
        self.pkl_path = make_fake_pkl(num_samples=15)
        self.ds = RefCOCODataset(self.pkl_path, split="train")

    def tearDown(self):
        os.remove(self.pkl_path)

    # --- keys ---

    def test_returns_dict(self):
        sample = self.ds[0]
        self.assertIsInstance(sample, dict)

    def test_has_required_keys(self):
        sample = self.ds[0]
        self.assertIn("image", sample)
        self.assertIn("input_ids", sample)
        self.assertIn("bbox", sample)

    # --- shapes ---

    def test_image_shape(self):
        self.assertEqual(self.ds[0]["image"].shape, (3, 384, 384))

    def test_input_ids_shape(self):
        self.assertEqual(self.ds[0]["input_ids"].shape, (77,))

    def test_bbox_shape(self):
        self.assertEqual(self.ds[0]["bbox"].shape, (4,))

    # --- dtypes ---

    def test_image_dtype_float32(self):
        self.assertEqual(self.ds[0]["image"].dtype, torch.float32)

    def test_input_ids_dtype_long(self):
        self.assertEqual(self.ds[0]["input_ids"].dtype, torch.long)

    def test_bbox_dtype_float32(self):
        self.assertEqual(self.ds[0]["bbox"].dtype, torch.float32)

    # --- values ---

    def test_bbox_values_in_unit_range(self):
        for i in range(len(self.ds)):
            bbox = self.ds[i]["bbox"]
            self.assertTrue((bbox >= 0).all() and (bbox <= 1).all(),
                            f"Sample {i}: bbox out of [0,1]: {bbox}")

    def test_first_and_last_accessible(self):
        """Index 0 and -1 (last) must both work."""
        _ = self.ds[0]
        _ = self.ds[len(self.ds) - 1]

    def test_all_indices_accessible(self):
        for i in range(len(self.ds)):
            sample = self.ds[i]
            self.assertIn("image", sample)

    # --- list → tensor conversion ---

    def test_raw_lists_converted_to_tensors(self):
        """Items stored as Python lists must be converted to tensors."""
        pkl = make_fake_pkl_raw_lists(num_samples=5)
        try:
            ds = RefCOCODataset(pkl)
            sample = ds[0]
            self.assertIsInstance(sample["image"], torch.Tensor)
            self.assertIsInstance(sample["input_ids"], torch.Tensor)
            self.assertIsInstance(sample["bbox"], torch.Tensor)
        finally:
            os.remove(pkl)

    def test_raw_list_dtypes_correct(self):
        """Converted list samples must still have correct dtypes."""
        pkl = make_fake_pkl_raw_lists(num_samples=5)
        try:
            ds = RefCOCODataset(pkl)
            sample = ds[0]
            self.assertEqual(sample["image"].dtype, torch.float32)
            self.assertEqual(sample["input_ids"].dtype, torch.long)
            self.assertEqual(sample["bbox"].dtype, torch.float32)
        finally:
            os.remove(pkl)


# ── RefCOCODataset.get_dataloader ───────────────────────────────────────

class TestGetDataloader(unittest.TestCase):

    def setUp(self):
        self.pkl_path = make_fake_pkl(num_samples=20)
        self.ds = RefCOCODataset(self.pkl_path, split="train")

    def tearDown(self):
        os.remove(self.pkl_path)

    def test_returns_dataloader(self):
        from torch.utils.data import DataLoader
        loader = self.ds.get_dataloader(batch_size=4)
        self.assertIsInstance(loader, DataLoader)

    def test_batch_shape_image(self):
        loader = self.ds.get_dataloader(
            batch_size=4, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        self.assertEqual(batch["image"].shape, (4, 3, 384, 384))

    def test_batch_shape_input_ids(self):
        loader = self.ds.get_dataloader(
            batch_size=4, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        self.assertEqual(batch["input_ids"].shape, (4, 77))

    def test_batch_shape_bbox(self):
        loader = self.ds.get_dataloader(
            batch_size=4, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        self.assertEqual(batch["bbox"].shape, (4, 4))

    def test_batch_dtypes(self):
        loader = self.ds.get_dataloader(
            batch_size=2, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        self.assertEqual(batch["image"].dtype, torch.float32)
        self.assertEqual(batch["input_ids"].dtype, torch.long)
        self.assertEqual(batch["bbox"].dtype, torch.float32)

    def test_batch_size_1(self):
        loader = self.ds.get_dataloader(
            batch_size=1, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        self.assertEqual(batch["image"].shape[0], 1)

    def test_full_epoch_iterable(self):
        """All batches should be iterable without error."""
        loader = self.ds.get_dataloader(
            batch_size=4, shuffle=False, num_workers=0)
        count = sum(1 for _ in loader)
        self.assertGreater(count, 0)

    def test_shuffle_false_deterministic(self):
        """shuffle=False should give same order on two passes."""
        loader = self.ds.get_dataloader(
            batch_size=4, shuffle=False, num_workers=0)
        pass1 = [b["input_ids"][0, 0].item() for b in loader]
        pass2 = [b["input_ids"][0, 0].item() for b in loader]
        self.assertEqual(pass1, pass2)


# ── RefCOCODataset.from_config ──────────────────────────────────────────

class TestFromConfig(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.pkl_path = os.path.join(self.tmp_dir, "refcoco_train.pkl")
        samples = [make_fake_sample() for _ in range(8)]
        with open(self.pkl_path, "wb") as f:
            pickle.dump(samples, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir)

    def test_from_config_train(self):
        config = {"data_dir": self.tmp_dir}
        ds = RefCOCODataset.from_config(config, split="train")
        self.assertEqual(len(ds), 8)

    def test_from_config_split_label(self):
        config = {"data_dir": self.tmp_dir}
        ds = RefCOCODataset.from_config(config, split="train")
        self.assertEqual(ds.split, "train")

    def test_from_config_missing_file_raises(self):
        config = {"data_dir": self.tmp_dir}
        with self.assertRaises(FileNotFoundError):
            RefCOCODataset.from_config(
                config, split="val")  # val pkl not created


# ── load_splits ─────────────────────────────────────────────────────────

class TestLoadSplits(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        for split in ["train", "val", "test"]:
            samples = [make_fake_sample() for _ in range(5)]
            path = os.path.join(self.tmp_dir, f"refcoco_{split}.pkl")
            with open(path, "wb") as f:
                pickle.dump(samples, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir)

    def test_returns_all_three_splits(self):
        splits = load_splits(self.tmp_dir)
        self.assertIn("train", splits)
        self.assertIn("val", splits)
        self.assertIn("test", splits)

    def test_each_split_is_dataset(self):
        splits = load_splits(self.tmp_dir)
        for name, ds in splits.items():
            self.assertIsInstance(ds, RefCOCODataset,
                                  f"Split '{name}' is not a RefCOCODataset")

    def test_each_split_has_correct_label(self):
        splits = load_splits(self.tmp_dir)
        for name, ds in splits.items():
            self.assertEqual(ds.split, name)

    def test_each_split_has_samples(self):
        splits = load_splits(self.tmp_dir)
        for name, ds in splits.items():
            self.assertEqual(len(ds), 5, f"Split '{name}' has wrong length")

    def test_custom_splits_subset(self):
        """load_splits with explicit splits list should only load those."""
        splits = load_splits(self.tmp_dir, splits=["train", "val"])
        self.assertIn("train", splits)
        self.assertIn("val", splits)
        self.assertNotIn("test", splits)

    def test_missing_split_raises(self):
        load_splits(self.tmp_dir, splits=["train"])
        # This should work fine; requesting a nonexistent one raises
        with self.assertRaises(FileNotFoundError):
            load_splits(self.tmp_dir, splits=["nonexistent"])


# ── Entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
