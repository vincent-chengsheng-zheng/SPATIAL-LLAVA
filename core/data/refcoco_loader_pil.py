"""
core/data/refcoco_loader_pil.py

RefCOCO dataset loader returning PIL Images for baseline inference.

What this file does:
    1. Load pkl files (v3 format: image_path + text + bbox)
    2. __getitem__ returns PIL Image (not tensor)
    3. LLaVA processor requires PIL Images, not tensors
    4. No tokenization here — LLaVA processor handles it internally

Why separate from refcoco_loader.py:
    - Training loader returns tensors + input_ids (for DataLoader batching)
    - Baseline loader returns PIL + raw text (for LLaVA processor)
    - Different output format, different use case
    - Keeps training code clean and unaffected

Output format per sample:
    {
        "image"      : PIL.Image.Image  original COCO image (any resolution)
        "text"       : str              raw referring expression
        "bbox"       : Tensor(4,)       ground truth [xc, yc, w, h] in [0,1]
        "image_path" : str              absolute path (for visualization)
        "idx"        : int              dataset index (for predictions.json)
    }

Usage:
    from core.data.refcoco_loader_pil import RefCOCODatasetPIL

    test_ds = RefCOCODatasetPIL("/tmp/data/refcoco_test.pkl")
    sample = test_ds[0]
    # sample["image"] is PIL.Image, ready for LLaVA processor
"""

import os
import pickle
from typing import Dict

from PIL import Image
from torch.utils.data import Dataset


class RefCOCODatasetPIL(Dataset):
    """
    RefCOCO dataset returning PIL Images for LLaVA baseline inference.

    What happens at init:
        1. Load pkl file into memory (lightweight — only paths and strings)
        2. Verify format is v3 (image_path field present)

    What happens at __getitem__:
        1. Get sample from pkl (image_path, text, bbox)
        2. Open JPEG with PIL at original resolution
        3. Return dict with PIL Image + metadata

    Args:
        pkl_path : Path to refcoco_{split}.pkl (v3 format)
        split    : "train", "val", or "test" (for logging only)
    """

    def __init__(self, pkl_path: str, split: str = "test"):
        self.split = split
        self.pkl_path = os.path.expanduser(pkl_path)
        self.data = self._load_pkl()

    def _load_pkl(self) -> list:
        """
        Load pkl file and verify it is v3 format.

        What this does:
            1. Check file exists
            2. Deserialize with pickle
            3. Verify first sample has image_path field (v3 format)
            4. Return list of sample dicts

        Raises:
            FileNotFoundError : If pkl file does not exist
            ValueError        : If pkl is not v3 format
            RuntimeError      : If pkl cannot be loaded
        """
        if not os.path.exists(self.pkl_path):
            raise FileNotFoundError(
                f"[RefCOCODatasetPIL] pkl not found: {self.pkl_path}\n"
                "  Run: bash shared/scripts/download_data.sh"
            )

        try:
            print(f"  [RefCOCODatasetPIL] Loading {self.split}: "
                  f"{self.pkl_path}")
            with open(self.pkl_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            raise RuntimeError(
                f"[RefCOCODatasetPIL] Failed to load pkl "
                f"{self.pkl_path}: {e}"
            )

        if len(data) == 0:
            raise ValueError(
                f"[RefCOCODatasetPIL] pkl is empty: {self.pkl_path}"
            )

        # Verify v3 format
        sample0 = data[0]
        if "image_path" not in sample0:
            raise ValueError(
                f"[RefCOCODatasetPIL] Wrong format — 'image_path' not found.\n"
                f"  Fields present: {list(sample0.keys())}\n"
                f"  Expected v3 format with 'image_path', 'text', 'bbox'.\n"
                f"  Run: bash shared/scripts/download_data.sh --force"
            )

        print(f"  [RefCOCODatasetPIL] ✓ Loaded {len(data):,} samples "
              f"(v3 format)")
        return data

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """
        Return one sample as PIL Image + metadata.

        What this does:
            1. Get sample dict from pkl
            2. Open JPEG at original resolution with PIL
            3. Convert to RGB (handles grayscale, RGBA)
            4. Return dict with PIL Image and metadata

        Args:
            idx : Sample index (0 to len-1)

        Returns:
            {
                "image"      : PIL.Image.Image  RGB, original resolution
                "text"       : str              raw referring expression
                "bbox"       : Tensor(4,)       ground truth [xc,yc,w,h]
                "image_path" : str              absolute path to JPEG
                "idx"        : int              dataset index
            }

        Raises:
            IndexError        : If idx is out of range
            FileNotFoundError : If image file does not exist on disk
            RuntimeError      : If image cannot be opened
        """
        if idx < 0 or idx >= len(self.data):
            raise IndexError(
                f"[RefCOCODatasetPIL] Index {idx} out of range "
                f"(dataset size: {len(self.data)})"
            )

        sample = self.data[idx]
        img_path = sample["image_path"]
        text = sample["text"]
        bbox = sample["bbox"]

        # Validate fields
        if not img_path:
            raise ValueError(
                f"[RefCOCODatasetPIL] Empty image_path at idx={idx}"
            )
        if not text:
            raise ValueError(
                f"[RefCOCODatasetPIL] Empty text at idx={idx}"
            )

        # Load image
        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"[RefCOCODatasetPIL] Image not found at idx={idx}: "
                f"{img_path}\n"
                f"  COCO images may have been deleted from /tmp/.\n"
                f"  Run: bash shared/scripts/download_data.sh"
            )

        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(
                f"[RefCOCODatasetPIL] Cannot open image at idx={idx}: "
                f"{img_path}: {e}"
            )

        return {
            "image": pil_img,           # PIL.Image — for LLaVA processor
            "text": text,               # str — raw referring expression
            "bbox": bbox.float(),       # Tensor(4,) — ground truth
            "image_path": img_path,     # str — for visualization
            "idx": idx,                 # int — for predictions.json
        }

    @classmethod
    def from_split(
        cls,
        data_dir: str,
        split: str = "test",
    ) -> "RefCOCODatasetPIL":
        """
        Convenience factory: build dataset from data_dir + split name.

        Args:
            data_dir : Directory containing refcoco_*.pkl files
            split    : "train", "val", or "test"

        Returns:
            RefCOCODatasetPIL instance

        Raises:
            FileNotFoundError : If pkl file not found in data_dir

        Example:
            test_ds = RefCOCODatasetPIL.from_split("/tmp/data/", "test")
        """
        data_dir = os.path.expanduser(data_dir)
        pkl_path = os.path.join(data_dir, f"refcoco_{split}.pkl")

        if not os.path.exists(pkl_path):
            raise FileNotFoundError(
                f"[RefCOCODatasetPIL.from_split] pkl not found: {pkl_path}\n"
                f"  Run: bash shared/scripts/download_data.sh"
            )

        return cls(pkl_path=pkl_path, split=split)
