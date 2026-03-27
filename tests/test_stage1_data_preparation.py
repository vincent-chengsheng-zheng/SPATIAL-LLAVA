"""
tests/test_stage1_data_preparation.py

Unit tests for pipeline/stage_1_data_preparation.py

Tests all helper functions WITHOUT downloading real data or hitting HuggingFace.
Run with:
    pytest tests/test_stage1_data_preparation.py -v
Or:
    python tests/test_stage1_data_preparation.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import pickle
import random
import shutil
import tempfile
import unittest

import torch
from PIL import Image
import numpy as np


# ── Import helpers from stage_1 directly ─────────────────────────────────────

from pipeline.stage_1_data_preparation import (
    compute_md5,
    verify_output,
    make_splits,
    save_split,
    save_stats,
    preprocess_sample,
    SPLITS,
    SEED,
    LOC_TOKEN,
)


# ── Fake tokenizer ────────────────────────────────────────────────────────────

class MockTokenizer:
    """Minimal tokenizer mock — same as in test_preprocessing.py."""
    def __init__(self, max_length=77):
        self.max_length = max_length
        self._vocab = {}

    def get_vocab(self):
        return self._vocab

    def add_special_tokens(self, d):
        for tok in d.get("additional_special_tokens", []):
            self._vocab[tok] = len(self._vocab)

    def __call__(self, text, padding=None, truncation=None,
                 max_length=None, return_tensors=None):
        length = max_length or self.max_length
        ids    = [hash(t) % 30000 for t in text.split()][:length]
        ids   += [0] * (length - len(ids))
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}


# ── Fake raw sample ───────────────────────────────────────────────────────────

def make_raw_sample(img_w=640, img_h=480, sentences=None):
    """
    Build a fake raw HuggingFace RefCOCO sample dict.
    """
    arr = np.random.randint(0, 256, (img_h, img_w, 3), dtype=np.uint8)
    pil = Image.fromarray(arr, "RGB")

    if sentences is None:
        sentences = [{"raw": "the person on the left"}]

    return {
        "image":     pil,
        "sentences": sentences,
        "bbox":      [100.0, 50.0, 200.0, 150.0],   # COCO format [x1,y1,w,h]
        "width":     img_w,
        "height":    img_h,
    }


def make_fake_processed_samples(n: int = 20):
    """Return n fake preprocessed sample dicts (already tensor form)."""
    return [
        {
            "image":     torch.rand(3, 384, 384, dtype=torch.float32),
            "input_ids": torch.randint(0, 30000, (77,), dtype=torch.long),
            "bbox":      torch.rand(4, dtype=torch.float32).clamp(0, 1),
        }
        for _ in range(n)
    ]


# ── compute_md5 ───────────────────────────────────────────────────────────────

class TestComputeMd5(unittest.TestCase):

    def test_returns_string(self):
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(b"hello world")
        tmp.close()
        try:
            result = compute_md5(tmp.name)
            self.assertIsInstance(result, str)
        finally:
            os.remove(tmp.name)

    def test_known_md5(self):
        """MD5 of b'hello world' is a known constant."""
        import hashlib
        expected = hashlib.md5(b"hello world").hexdigest()
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(b"hello world")
        tmp.close()
        try:
            self.assertEqual(compute_md5(tmp.name), expected)
        finally:
            os.remove(tmp.name)

    def test_different_files_different_md5(self):
        tmp1 = tempfile.NamedTemporaryFile(delete=False)
        tmp2 = tempfile.NamedTemporaryFile(delete=False)
        tmp1.write(b"aaa"); tmp1.close()
        tmp2.write(b"bbb"); tmp2.close()
        try:
            self.assertNotEqual(compute_md5(tmp1.name), compute_md5(tmp2.name))
        finally:
            os.remove(tmp1.name); os.remove(tmp2.name)

    def test_same_content_same_md5(self):
        tmp1 = tempfile.NamedTemporaryFile(delete=False)
        tmp2 = tempfile.NamedTemporaryFile(delete=False)
        tmp1.write(b"same"); tmp1.close()
        tmp2.write(b"same"); tmp2.close()
        try:
            self.assertEqual(compute_md5(tmp1.name), compute_md5(tmp2.name))
        finally:
            os.remove(tmp1.name); os.remove(tmp2.name)

    def test_md5_length(self):
        """MD5 hex digest is always 32 characters."""
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(b"data"); tmp.close()
        try:
            self.assertEqual(len(compute_md5(tmp.name)), 32)
        finally:
            os.remove(tmp.name)


# ── verify_output ─────────────────────────────────────────────────────────────

class TestVerifyOutput(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def _touch(self, filename):
        open(os.path.join(self.tmp_dir, filename), "w").close()

    def test_returns_false_when_nothing_exists(self):
        self.assertFalse(verify_output(self.tmp_dir))

    def test_returns_false_missing_val(self):
        self._touch("refcoco_train.pkl")
        self._touch("refcoco_test.pkl")
        self._touch("dataset_stats.json")
        self.assertFalse(verify_output(self.tmp_dir))

    def test_returns_false_missing_stats(self):
        self._touch("refcoco_train.pkl")
        self._touch("refcoco_val.pkl")
        self._touch("refcoco_test.pkl")
        self.assertFalse(verify_output(self.tmp_dir))

    def test_returns_true_when_all_exist(self):
        for f in ["refcoco_train.pkl", "refcoco_val.pkl",
                  "refcoco_test.pkl", "dataset_stats.json"]:
            self._touch(f)
        self.assertTrue(verify_output(self.tmp_dir))


# ── preprocess_sample ─────────────────────────────────────────────────────────

class TestPreprocessSample(unittest.TestCase):

    def setUp(self):
        self.tokenizer = MockTokenizer(max_length=77)

    def test_returns_dict_with_required_keys(self):
        raw    = make_raw_sample()
        result = preprocess_sample(raw, self.tokenizer)
        self.assertIn("image",     result)
        self.assertIn("input_ids", result)
        self.assertIn("bbox",      result)

    def test_image_shape(self):
        raw    = make_raw_sample()
        result = preprocess_sample(raw, self.tokenizer)
        self.assertEqual(result["image"].shape, (3, 384, 384))

    def test_image_dtype(self):
        raw    = make_raw_sample()
        result = preprocess_sample(raw, self.tokenizer)
        self.assertEqual(result["image"].dtype, torch.float32)

    def test_input_ids_shape(self):
        raw    = make_raw_sample()
        result = preprocess_sample(raw, self.tokenizer)
        self.assertEqual(result["input_ids"].shape, (77,))

    def test_input_ids_dtype(self):
        raw    = make_raw_sample()
        result = preprocess_sample(raw, self.tokenizer)
        self.assertEqual(result["input_ids"].dtype, torch.long)

    def test_bbox_shape(self):
        raw    = make_raw_sample()
        result = preprocess_sample(raw, self.tokenizer)
        self.assertEqual(result["bbox"].shape, (4,))

    def test_bbox_dtype(self):
        raw    = make_raw_sample()
        result = preprocess_sample(raw, self.tokenizer)
        self.assertEqual(result["bbox"].dtype, torch.float32)

    def test_bbox_values_in_unit_range(self):
        raw    = make_raw_sample()
        result = preprocess_sample(raw, self.tokenizer)
        bbox   = result["bbox"]
        self.assertTrue((bbox >= 0).all() and (bbox <= 1).all(),
            f"bbox out of [0,1]: {bbox}")

    def test_coco_bbox_conversion(self):
        """
        COCO bbox [x1,y1,w,h]=[0,0,640,480] on a 640x480 image
        should normalise to [0.5, 0.5, 1.0, 1.0].
        """
        raw = make_raw_sample(img_w=640, img_h=480)
        raw["bbox"] = [0.0, 0.0, 640.0, 480.0]
        result = preprocess_sample(raw, self.tokenizer)
        expected = torch.tensor([0.5, 0.5, 1.0, 1.0])
        self.assertTrue(
            torch.allclose(result["bbox"], expected, atol=1e-4),
            f"Expected {expected}, got {result['bbox']}"
        )

    def test_multiple_sentences_picks_one(self):
        """When multiple sentences exist, one is picked (no crash)."""
        sentences = [
            {"raw": "the cat on the left"},
            {"raw": "a white cat"},
            {"raw": "the animal near the wall"},
        ]
        raw    = make_raw_sample(sentences=sentences)
        result = preprocess_sample(raw, self.tokenizer)
        self.assertEqual(result["input_ids"].shape, (77,))

    def test_rgba_image_handled(self):
        """RGBA images should be converted to RGB without error."""
        arr = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
        pil = Image.fromarray(arr, "RGBA")
        raw = make_raw_sample()
        raw["image"] = pil
        result = preprocess_sample(raw, self.tokenizer)
        self.assertEqual(result["image"].shape, (3, 384, 384))

    def test_small_image_resized(self):
        """Very small images must still be resized to 384x384."""
        arr = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        pil = Image.fromarray(arr, "RGB")
        raw = make_raw_sample(img_w=10, img_h=10)
        raw["image"] = pil
        result = preprocess_sample(raw, self.tokenizer)
        self.assertEqual(result["image"].shape, (3, 384, 384))

    def test_deterministic_given_fixed_seed(self):
        """Same sample + seed → same input_ids (sentence selection is random)."""
        raw = make_raw_sample(sentences=[{"raw": "only one sentence"}])
        random.seed(0)
        r1 = preprocess_sample(raw, self.tokenizer)
        random.seed(0)
        r2 = preprocess_sample(raw, self.tokenizer)
        self.assertTrue(torch.equal(r1["input_ids"], r2["input_ids"]))


# ── make_splits ───────────────────────────────────────────────────────────────

class TestMakeSplits(unittest.TestCase):

    def test_returns_three_keys(self):
        samples = make_fake_processed_samples(100)
        splits  = make_splits(samples)
        self.assertIn("train", splits)
        self.assertIn("val",   splits)
        self.assertIn("test",  splits)

    def test_total_preserved(self):
        samples = make_fake_processed_samples(100)
        splits  = make_splits(samples)
        total   = sum(len(v) for v in splits.values())
        self.assertEqual(total, 100)

    def test_approximate_ratios(self):
        samples = make_fake_processed_samples(1000)
        splits  = make_splits(samples)
        self.assertAlmostEqual(len(splits["train"]) / 1000, SPLITS["train"], delta=0.02)
        self.assertAlmostEqual(len(splits["val"])   / 1000, SPLITS["val"],   delta=0.02)
        self.assertAlmostEqual(len(splits["test"])  / 1000, SPLITS["test"],  delta=0.02)

    def test_no_overlap(self):
        """Each sample must appear in exactly one split."""
        samples = make_fake_processed_samples(100)
        # Tag each sample with an index
        for i, s in enumerate(samples):
            s["_idx"] = i
        splits  = make_splits(samples)
        all_idx = (
            [s["_idx"] for s in splits["train"]] +
            [s["_idx"] for s in splits["val"]]   +
            [s["_idx"] for s in splits["test"]]
        )
        self.assertEqual(len(all_idx), len(set(all_idx)),
            "Some samples appear in more than one split")

    def test_reproducible_with_seed(self):
        """Two calls with the same data produce the same split order."""
        samples1 = make_fake_processed_samples(50)
        samples2 = [dict(s) for s in samples1]  # shallow copy
        sp1 = make_splits(samples1)
        sp2 = make_splits(samples2)
        self.assertEqual(len(sp1["train"]), len(sp2["train"]))
        self.assertEqual(len(sp1["val"]),   len(sp2["val"]))

    def test_small_dataset(self):
        """Even with 3 samples the function should not crash."""
        samples = make_fake_processed_samples(3)
        splits  = make_splits(samples)
        total   = sum(len(v) for v in splits.values())
        self.assertEqual(total, 3)


# ── save_split ────────────────────────────────────────────────────────────────

class TestSaveSplit(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_file_created(self):
        samples = make_fake_processed_samples(5)
        save_split(samples, self.tmp_dir, "train")
        self.assertTrue(os.path.exists(
            os.path.join(self.tmp_dir, "refcoco_train.pkl")))

    def test_file_loadable(self):
        samples = make_fake_processed_samples(5)
        path    = save_split(samples, self.tmp_dir, "val")
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        self.assertEqual(len(loaded), 5)

    def test_returns_path_string(self):
        samples = make_fake_processed_samples(3)
        path    = save_split(samples, self.tmp_dir, "test")
        self.assertIsInstance(path, str)
        self.assertTrue(path.endswith(".pkl"))

    def test_content_roundtrip(self):
        """Saved and reloaded samples must match the originals."""
        samples = make_fake_processed_samples(4)
        path    = save_split(samples, self.tmp_dir, "train")
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        for orig, reloaded in zip(samples, loaded):
            self.assertTrue(torch.equal(orig["bbox"], reloaded["bbox"]))

    def test_all_three_splits_saved(self):
        for split in ["train", "val", "test"]:
            save_split(make_fake_processed_samples(3), self.tmp_dir, split)
        for split in ["train", "val", "test"]:
            self.assertTrue(os.path.exists(
                os.path.join(self.tmp_dir, f"refcoco_{split}.pkl")))


# ── save_stats ────────────────────────────────────────────────────────────────

class TestSaveStats(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def _make_splits_dict(self, n=10):
        return {
            "train": make_fake_processed_samples(int(n * 0.8)),
            "val":   make_fake_processed_samples(int(n * 0.1)),
            "test":  make_fake_processed_samples(int(n * 0.1)),
        }

    def test_json_file_created(self):
        splits = self._make_splits_dict()
        save_stats(splits, self.tmp_dir, checksums={})
        self.assertTrue(os.path.exists(
            os.path.join(self.tmp_dir, "dataset_stats.json")))

    def test_json_is_valid(self):
        splits = self._make_splits_dict()
        save_stats(splits, self.tmp_dir, checksums={})
        path = os.path.join(self.tmp_dir, "dataset_stats.json")
        with open(path) as f:
            data = json.load(f)
        self.assertIsInstance(data, dict)

    def test_json_has_required_keys(self):
        splits = self._make_splits_dict(100)
        save_stats(splits, self.tmp_dir, checksums={"train": "abc"})
        with open(os.path.join(self.tmp_dir, "dataset_stats.json")) as f:
            data = json.load(f)
        for key in ["total_samples", "split_sizes", "checksums", "seed"]:
            self.assertIn(key, data, f"Missing key: {key}")

    def test_total_samples_correct(self):
        splits = self._make_splits_dict(100)
        save_stats(splits, self.tmp_dir, checksums={})
        with open(os.path.join(self.tmp_dir, "dataset_stats.json")) as f:
            data = json.load(f)
        expected = sum(len(v) for v in splits.values())
        self.assertEqual(data["total_samples"], expected)

    def test_checksums_stored(self):
        splits    = self._make_splits_dict()
        checksums = {"train": "abc123", "val": "def456", "test": "ghi789"}
        save_stats(splits, self.tmp_dir, checksums=checksums)
        with open(os.path.join(self.tmp_dir, "dataset_stats.json")) as f:
            data = json.load(f)
        self.assertEqual(data["checksums"], checksums)


# ── Integration ───────────────────────────────────────────────────────────────

class TestIntegration(unittest.TestCase):
    """
    End-to-end: simulate the full Stage 1 pipeline using fake data.
    No HuggingFace download, no real tokenizer.
    """

    def setUp(self):
        self.tmp_dir   = tempfile.mkdtemp()
        self.tokenizer = MockTokenizer()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_full_pipeline_fake_data(self):
        """Preprocess → split → save → verify → load."""
        # 1. Preprocess fake raw samples
        raw_samples = [make_raw_sample() for _ in range(30)]
        processed   = [preprocess_sample(r, self.tokenizer) for r in raw_samples]
        self.assertEqual(len(processed), 30)

        # 2. Split
        splits = make_splits(processed)
        self.assertEqual(sum(len(v) for v in splits.values()), 30)

        # 3. Save pkl files
        checksums = {}
        for split, samples in splits.items():
            path             = save_split(samples, self.tmp_dir, split)
            checksums[split] = compute_md5(path)

        # 4. Save stats
        save_stats(splits, self.tmp_dir, checksums)

        # 5. Verify all files exist
        self.assertTrue(verify_output(self.tmp_dir))

        # 6. Load back and check shapes
        for split in ["train", "val", "test"]:
            path = os.path.join(self.tmp_dir, f"refcoco_{split}.pkl")
            with open(path, "rb") as f:
                loaded = pickle.load(f)
            self.assertGreater(len(loaded), 0)
            sample = loaded[0]
            self.assertEqual(sample["image"].shape,     (3, 384, 384))
            self.assertEqual(sample["input_ids"].shape, (77,))
            self.assertEqual(sample["bbox"].shape,      (4,))

    def test_checksums_match_saved_files(self):
        """MD5 saved in stats must match the actual files on disk."""
        processed = [preprocess_sample(make_raw_sample(), self.tokenizer)
                     for _ in range(10)]
        splits    = make_splits(processed)
        checksums = {}
        for split, samples in splits.items():
            path             = save_split(samples, self.tmp_dir, split)
            checksums[split] = compute_md5(path)

        save_stats(splits, self.tmp_dir, checksums)

        with open(os.path.join(self.tmp_dir, "dataset_stats.json")) as f:
            stats = json.load(f)

        for split, expected_md5 in stats["checksums"].items():
            path       = os.path.join(self.tmp_dir, f"refcoco_{split}.pkl")
            actual_md5 = compute_md5(path)
            self.assertEqual(actual_md5, expected_md5,
                f"Checksum mismatch for split '{split}'")

    def test_loader_works_after_save(self):
        """Saved pkl files must be loadable by RefCOCODataset."""
        from core.data.refcoco_loader import RefCOCODataset

        processed = [preprocess_sample(make_raw_sample(), self.tokenizer)
                     for _ in range(12)]
        splits    = make_splits(processed)
        for split, samples in splits.items():
            save_split(samples, self.tmp_dir, split)

        ds     = RefCOCODataset(
            os.path.join(self.tmp_dir, "refcoco_train.pkl"), split="train"
        )
        sample = ds[0]
        self.assertEqual(sample["image"].shape,     (3, 384, 384))
        self.assertEqual(sample["input_ids"].shape, (77,))
        self.assertEqual(sample["bbox"].shape,      (4,))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
    