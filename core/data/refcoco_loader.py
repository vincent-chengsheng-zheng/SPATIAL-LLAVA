"""
core/data/refcoco_loader.py

PyTorch Dataset for RefCOCO training and validation.

What this file provides:
    RefCOCODataset   — reads pkl v3, loads images, tokenizes text for LLaVA
    make_loaders()   — convenience function to build train/val DataLoaders

pkl v3 format (one sample):
    {
        "image_path": str           absolute path to COCO .jpg
        "text":       str           referring expression
        "bbox":       Tensor(4,)    normalized [xc, yc, w, h] in [0, 1]
    }

Usage:
    from core.data.refcoco_loader import make_loaders

    train_loader, val_loader = make_loaders(
        processor=processor,
        batch_size=8,
        num_workers=4,
    )
"""

import pickle
import sys
import os
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ── sys.path guard ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from core.paths import PATHS  # noqa: E402

# Prompt template — must match baseline exactly
PROMPT_TEMPLATE = "Find the object referred to: {text} [LOC]"


# ── Dataset ───────────────────────────────────────────────────────────────────

class RefCOCODataset(Dataset):
    """
    PyTorch Dataset for RefCOCO grounding task.

    What __getitem__ returns:
        {
            "input_ids"      : LongTensor(seq_len,)   tokenized prompt
            "attention_mask" : LongTensor(seq_len,)
            "pixel_values"   : FloatTensor(3, H, W)   preprocessed image
            "bbox"           : FloatTensor(4,)         [xc, yc, w, h] in [0,1]
            "image_path"     : str                     for debugging
            "text"           : str                     raw expression
        }

    Args:
        split      : "train" | "val" | "test"
        processor  : LlavaProcessor (handles image + text jointly)
        max_length : token sequence length (default 128)
        max_samples: cap on dataset size (None = all)
    """

    def __init__(
        self,
        split: str,
        processor,
        max_length: int = 128,
        max_samples: Optional[int] = None,
    ):
        self.split = split
        self.processor = processor
        self.max_length = max_length

        pkl_path = PATHS.pkl(split)
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"pkl not found: {pkl_path}\n"
                "  Run: bash shared/scripts/download_data.sh"
            )

        with open(pkl_path, "rb") as f:
            samples = pickle.load(f)

        if max_samples is not None:
            samples = samples[:max_samples]

        self.samples: List[Dict] = samples
        print(f"[RefCOCODataset] {split}: {len(self.samples):,} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # ── Load image ────────────────────────────────────────────────
        image_path = sample["image_path"]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(
                f"[RefCOCODataset] Failed to load image: {image_path}\n{e}"
            )

        # ── Format prompt ─────────────────────────────────────────────
        text = sample["text"]
        prompt = PROMPT_TEMPLATE.format(text=text)

        # ── Tokenize (image + text jointly via LlavaProcessor) ────────
        encoding = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        # processor returns batch dim — squeeze it out
        input_ids = encoding["input_ids"].squeeze(0)           # (seq_len,)
        attention_mask = encoding["attention_mask"].squeeze(0)  # (seq_len,)
        pixel_values = encoding["pixel_values"].squeeze(0)      # (3, H, W)

        # ── BBox ──────────────────────────────────────────────────────
        bbox = sample["bbox"]
        if not isinstance(bbox, Tensor):
            bbox = torch.tensor(bbox, dtype=torch.float32)
        else:
            bbox = bbox.float()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "bbox": bbox,
            "image_path": image_path,
            "text": text,
        }


# ── Collate ───────────────────────────────────────────────────────────────────

def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate: stack tensors, keep strings as lists.
    """
    return {
        "input_ids":      torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "pixel_values":   torch.stack([b["pixel_values"] for b in batch]),
        "bbox":           torch.stack([b["bbox"] for b in batch]),
        "image_path":     [b["image_path"] for b in batch],
        "text":           [b["text"] for b in batch],
    }


# ── Convenience builder ───────────────────────────────────────────────────────

def make_loaders(
    processor,
    batch_size: int = 8,
    num_workers: int = 4,
    max_length: int = 128,
    max_samples: Optional[int] = None,
    val_max_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and val DataLoaders.

    Args:
        processor       : LlavaProcessor instance
        batch_size      : samples per batch
        num_workers     : DataLoader worker processes
        max_length      : token sequence length
        max_samples     : cap on train samples (None = all)
        val_max_samples : cap on val samples (None = all)

    Returns:
        (train_loader, val_loader)
    """
    train_ds = RefCOCODataset(
        split="train",
        processor=processor,
        max_length=max_length,
        max_samples=max_samples,
    )
    val_ds = RefCOCODataset(
        split="val",
        processor=processor,
        max_length=max_length,
        max_samples=val_max_samples,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(
        f"[make_loaders] train={len(train_ds):,}  val={len(val_ds):,}  "
        f"batch={batch_size}  workers={num_workers}"
    )
    return train_loader, val_loader