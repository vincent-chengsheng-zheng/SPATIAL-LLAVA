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

from torch import Tensor
from typing import List, Tuple


# ── Image ─────────────────────────────────────────────────────────────────────

IMAGE_SIZE = 384

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
    raise NotImplementedError(
        "Member 2: implement using PIL + torchvision.transforms. "
        "Use transforms.Compose([Resize, ToTensor, Normalize(IMAGENET_MEAN, IMAGENET_STD)]). "
        "Return tensor of shape (3, 384, 384)."
    )


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
    raise NotImplementedError(
        "Member 2: same as preprocess_image() but skip the file load step. "
        "Accept PIL Image directly."
    )


# ── Text ──────────────────────────────────────────────────────────────────────

LOC_TOKEN = "[LOC]"
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
    raise NotImplementedError(
        "Member 2: format the prompt as 'Find the object referred to: {prompt} [LOC]', "
        "then tokenize with padding='max_length', truncation=True, max_length=MAX_LENGTH. "
        "Return input_ids as a long tensor."
    )


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
    raise NotImplementedError(
        "Member 2: compute x_center=(x1+x2)/2, y_center=(y1+y2)/2, "
        "w=x2-x1, h=y2-y1, then divide by img_w and img_h respectively. "
        "Clamp all values to [0, 1]. Return float32 tensor."
    )


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
    raise NotImplementedError(
        "Member 2: reverse of normalize_bbox(). "
        "x_c, y_c, w, h = bbox_norm. "
        "x1 = int((x_c - w/2) * img_w), etc. "
        "Clamp to [0, img_w] and [0, img_h]."
    )
