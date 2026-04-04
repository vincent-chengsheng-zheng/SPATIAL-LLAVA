"""
core/data/preprocessing.py

Image and text preprocessing utilities for Spatial-LLaVA.

What this file provides:
    1. IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD  — image transform constants
    2. LOC_TOKEN, MAX_LENGTH                    — text tokenization constants
    3. PROMPT_TEMPLATE                          — shared prompt format for all models
    4. normalize_bbox()                         — pixel xyxy → normalized xywh
    5. denormalize_bbox()                       — normalized xywh → pixel xyxy
    6. preprocess_image()                       — load image from path → tensor
    7. preprocess_image_from_pil()              — PIL Image → tensor
    8. preprocess_text()                        — text → tokenized tensor

Used by:
    - stage_1_data_preparation.py  (normalize_bbox)
    - refcoco_loader.py            (IMAGE_TRANSFORM, LOC_TOKEN, PROMPT_TEMPLATE)
    - refcoco_loader_pil.py        (PROMPT_TEMPLATE)
    - core/model/llava.py          (PROMPT_TEMPLATE)
"""

import torch
from torch import Tensor
from typing import List, Tuple
from torchvision import transforms


# ── Image constants ───────────────────────────────────────────────────────────

IMAGE_SIZE = 384
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── Text constants ────────────────────────────────────────────────────────────

LOC_TOKEN = "[LOC]"
MAX_LENGTH = 77

# Shared prompt template used by ALL models (baseline, ablation, ours)
# Carefully worded to maximize LLaVA parse success rate
# {text} is the raw referring expression from RefCOCO
PROMPT_TEMPLATE = (
    "USER: <image>\n"
    "Locate the following object in the image and provide its bounding box "
    "as normalized coordinates [x_center, y_center, width, height] where "
    "all values are between 0 and 1.\n"
    "Object: {text}\n"
    "Respond ONLY with four numbers in square brackets, "
    "e.g. [0.5, 0.3, 0.2, 0.4].\n"
    "ASSISTANT:"
)

# Prompt used by SpatialLLaVA (adds [LOC] trigger token)
SPATIAL_PROMPT_TEMPLATE = (
    "Find the object referred to: {text} " + LOC_TOKEN
)


# ── Image preprocessing ───────────────────────────────────────────────────────

def preprocess_image(image_path: str) -> Tensor:
    """
    Load image from disk and preprocess for SpatialLLaVA training.

    What this does:
        1. Open JPEG with PIL
        2. Convert to RGB
        3. Resize to 384x384
        4. Convert to tensor
        5. Normalize with ImageNet stats

    Args:
        image_path : Absolute path to JPEG file

    Returns:
        Tensor(3, 384, 384) float32

    Raises:
        FileNotFoundError : If image_path does not exist
        RuntimeError      : If image cannot be opened or transformed
    """
    from PIL import Image

    if not image_path:
        raise ValueError("[preprocessing] image_path is empty or None")

    try:
        pil_img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"[preprocessing] Image not found: {image_path}"
        )
    except Exception as e:
        raise RuntimeError(
            f"[preprocessing] Failed to open image {image_path}: {e}"
        )

    try:
        return IMAGE_TRANSFORM(pil_img)
    except Exception as e:
        raise RuntimeError(
            f"[preprocessing] Transform failed for {image_path}: {e}"
        )


def preprocess_image_from_pil(pil_image) -> Tensor:
    """
    Preprocess a PIL Image for SpatialLLaVA training.

    Same as preprocess_image() but accepts a PIL Image directly.
    Used in Gradio demo and FastAPI server.

    Args:
        pil_image : PIL.Image.Image (any mode, will be converted to RGB)

    Returns:
        Tensor(3, 384, 384) float32

    Raises:
        ValueError  : If pil_image is None
        RuntimeError: If transform fails
    """
    if pil_image is None:
        raise ValueError("[preprocessing] pil_image is None")

    try:
        pil_image = pil_image.convert("RGB")
        return IMAGE_TRANSFORM(pil_image)
    except Exception as e:
        raise RuntimeError(
            f"[preprocessing] preprocess_image_from_pil failed: {e}"
        )


# ── Text preprocessing ────────────────────────────────────────────────────────

def preprocess_text(prompt: str, tokenizer) -> Tensor:
    """
    Tokenize a referring expression for SpatialLLaVA training.

    What this does:
        1. Format: "Find the object referred to: {prompt} [LOC]"
        2. Tokenize with padding to MAX_LENGTH=77
        3. Return input_ids as int64 tensor

    Args:
        prompt    : Raw referring expression, e.g. "the woman in blue"
        tokenizer : LLaVA tokenizer (with [LOC] already in vocab)

    Returns:
        Tensor(77,) int64

    Raises:
        ValueError  : If prompt is empty
        RuntimeError: If tokenization fails
    """
    if not prompt or not prompt.strip():
        raise ValueError(
            "[preprocessing] preprocess_text got empty prompt"
        )
    if tokenizer is None:
        raise ValueError("[preprocessing] tokenizer is None")

    try:
        formatted = SPATIAL_PROMPT_TEMPLATE.format(text=prompt)
        encoded = tokenizer(
            formatted,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        return encoded["input_ids"].squeeze(0).long()
    except Exception as e:
        raise RuntimeError(
            f"[preprocessing] tokenization failed for prompt '{prompt}': {e}"
        )


# ── Bounding box utilities ────────────────────────────────────────────────────

def normalize_bbox(
    bbox_xyxy: List[float],
    img_w: int,
    img_h: int,
) -> Tensor:
    """
    Convert bbox from pixel [x1,y1,x2,y2] to normalized [xc,yc,w,h].

    What this does:
        1. Compute center: xc = (x1+x2)/2, yc = (y1+y2)/2
        2. Compute size:   w = x2-x1, h = y2-y1
        3. Divide by image dimensions
        4. Clamp to [0, 1]

    Args:
        bbox_xyxy : [x1, y1, x2, y2] in pixel coordinates
        img_w     : Image width in pixels
        img_h     : Image height in pixels

    Returns:
        Tensor(4,) float32: [xc, yc, w, h] all in [0, 1]

    Raises:
        ValueError : If bbox has wrong length or image dims are invalid
    """
    if len(bbox_xyxy) != 4:
        raise ValueError(
            f"[preprocessing] normalize_bbox expects 4 values, "
            f"got {len(bbox_xyxy)}: {bbox_xyxy}"
        )
    if img_w <= 0 or img_h <= 0:
        raise ValueError(
            f"[preprocessing] Invalid image dimensions: "
            f"w={img_w}, h={img_h}"
        )

    x1, y1, x2, y2 = bbox_xyxy
    xc = (x1 + x2) / 2.0 / img_w
    yc = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h

    result = torch.tensor([xc, yc, w, h], dtype=torch.float32)
    result = result.clamp(0.0, 1.0)
    return result


def denormalize_bbox(
    bbox_norm: Tensor,
    img_w: int,
    img_h: int,
) -> Tuple[int, int, int, int]:
    """
    Convert normalized [xc,yc,w,h] back to pixel [x1,y1,x2,y2].

    Inverse of normalize_bbox(). Used for visualization.

    Args:
        bbox_norm : Tensor(4,) [xc, yc, w, h] in [0, 1]
        img_w     : Target image width in pixels
        img_h     : Target image height in pixels

    Returns:
        Tuple (x1, y1, x2, y2) in pixel coordinates (integers)

    Raises:
        ValueError : If bbox_norm has wrong shape
    """
    if bbox_norm.shape != (4,):
        raise ValueError(
            f"[preprocessing] denormalize_bbox expects Tensor(4,), "
            f"got shape {bbox_norm.shape}"
        )

    xc, yc, w, h = bbox_norm.tolist()
    x1 = int((xc - w / 2) * img_w)
    y1 = int((yc - h / 2) * img_h)
    x2 = int((xc + w / 2) * img_w)
    y2 = int((yc + h / 2) * img_h)

    x1 = max(0, min(x1, img_w))
    y1 = max(0, min(y1, img_h))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))

    return x1, y1, x2, y2
