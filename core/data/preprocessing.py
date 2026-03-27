"""
core/data/preprocessing.py

Image and text preprocessing for Spatial-LLaVA.

Handles:
- Image: resize to 384x384, normalize to ImageNet stats
- Text: tokenize with LLaVA tokenizer, insert [LOC] trigger token
- Bbox: convert pixel coords → normalized [x_center, y_center, w, h]

Usage:
    from core.data.preprocessing import preprocess_image, preprocess_text, normalize_bbox

    image_tensor = preprocess_image("path/to/image.jpg")
    input_ids    = preprocess_text("find the person on the left", tokenizer)
    bbox_norm    = normalize_bbox([x1, y1, x2, y2], img_w=640, img_h=480)
"""

import torch
from torch import Tensor
from typing import List, Tuple
from PIL import Image
import torchvision.transforms as transforms


# ── Image ─────────────────────────────────────────────────────────────────────

IMAGE_SIZE = 384

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def preprocess_image(image_path: str) -> Tensor:
    """
    Load an image from disk and preprocess it for LLaVA.

    Steps:
        1. Open image with PIL
        2. Resize to (384, 384)
        3. Convert to RGB (handles grayscale, RGBA)
        4. Normalize with ImageNet mean/std
        5. Return as float32 tensor (3, 384, 384)

    Args:
        image_path : Absolute path to image file (JPEG, PNG, etc.)

    Returns:
        Tensor of shape (3, 384, 384), dtype=float32

    Raises:
        FileNotFoundError : If image_path does not exist
    """
    import os
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    pil_image = Image.open(image_path).convert("RGB")
    return _image_transform(pil_image)


def preprocess_image_from_pil(pil_image) -> Tensor:
    """
    Preprocess a PIL Image object (used during inference).

    Same steps as preprocess_image() but accepts PIL Image directly
    instead of a file path. Used in Gradio demo and FastAPI server.

    Args:
        pil_image : PIL.Image.Image object

    Returns:
        Tensor of shape (3, 384, 384), dtype=float32
    """
    pil_image = pil_image.convert("RGB")
    return _image_transform(pil_image)


# ── Text ──────────────────────────────────────────────────────────────────────

LOC_TOKEN  = "[LOC]"
MAX_LENGTH = 77


def preprocess_text(prompt: str, tokenizer) -> Tensor:
    """
    Tokenize a referring expression and append the [LOC] trigger token.

    The [LOC] token signals to the model where to extract the hidden state
    that gets passed to the regression head.

    Prompt format:
        "Find the object referred to: <prompt> [LOC]"

    Args:
        prompt    : Raw referring expression, e.g. "the person on the left"
        tokenizer : LLaVA tokenizer (with [LOC] already added to vocab)

    Returns:
        Tensor of shape (MAX_LENGTH,), dtype=int64 (long)
        Padded/truncated to MAX_LENGTH=77 tokens.

    Example:
        ids = preprocess_text("the cat on the mat", tokenizer)
        # ids.shape == (77,)
    """
    formatted = f"Find the object referred to: {prompt} {LOC_TOKEN}"

    encoded = tokenizer(
        formatted,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    # encoded["input_ids"] shape: (1, MAX_LENGTH) → squeeze to (MAX_LENGTH,)
    return encoded["input_ids"].squeeze(0).long()


# ── Bounding box ──────────────────────────────────────────────────────────────

def normalize_bbox(
    bbox_xyxy: List[float],
    img_w: int,
    img_h: int,
) -> Tensor:
    """
    Convert a bounding box from pixel [x1, y1, x2, y2] format to
    normalized [x_center, y_center, width, height] format.

    All output values are in [0, 1].

    Args:
        bbox_xyxy : [x1, y1, x2, y2] in pixel coordinates
        img_w     : Original image width in pixels
        img_h     : Original image height in pixels

    Returns:
        Tensor of shape (4,), dtype=float32
        Values: [x_center/img_w, y_center/img_h, w/img_w, h/img_h]

    Example:
        bbox = normalize_bbox([100, 50, 300, 200], img_w=640, img_h=480)
        # tensor([0.3125, 0.2604, 0.3125, 0.3125])
    """
    x1, y1, x2, y2 = bbox_xyxy

    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    w        = x2 - x1
    h        = y2 - y1

    normalized = torch.tensor(
        [x_center / img_w, y_center / img_h, w / img_w, h / img_h],
        dtype=torch.float32,
    )

    return normalized.clamp(0.0, 1.0)


def denormalize_bbox(
    bbox_norm: Tensor,
    img_w: int,
    img_h: int,
) -> Tuple[int, int, int, int]:
    """
    Convert normalized [x_center, y_center, w, h] back to pixel [x1, y1, x2, y2].

    Inverse of normalize_bbox(). Used for visualization and evaluation.

    Args:
        bbox_norm : Tensor of shape (4,) — [x_c, y_c, w, h] in [0, 1]
        img_w     : Target image width in pixels
        img_h     : Target image height in pixels

    Returns:
        Tuple (x1, y1, x2, y2) in pixel coordinates (integers)

    Example:
        x1, y1, x2, y2 = denormalize_bbox(tensor([0.5, 0.5, 0.3, 0.4]),
                                           img_w=640, img_h=480)
    """
    x_c, y_c, w, h = bbox_norm.tolist()

    x1 = int((x_c - w / 2) * img_w)
    y1 = int((y_c - h / 2) * img_h)
    x2 = int((x_c + w / 2) * img_w)
    y2 = int((y_c + h / 2) * img_h)

    x1 = max(0, min(x1, img_w))
    x2 = max(0, min(x2, img_w))
    y1 = max(0, min(y1, img_h))
    y2 = max(0, min(y2, img_h))

    return x1, y1, x2, y2
