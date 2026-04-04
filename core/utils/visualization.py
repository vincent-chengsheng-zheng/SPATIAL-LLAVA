"""
core/utils/visualization.py

Visualization utilities for Spatial-LLaVA.

What this file provides:
    1. draw_bboxes()       — draw pred + gt boxes on a PIL image
    2. display_comparison() — side-by-side comparison for eval
    3. draw_comparison()   — baseline vs ground truth with status label
                             (NEW: used by step1_baseline_inference.py)

Color convention (used across ALL visualization functions):
    Green (lime) = Ground Truth bbox
    Red          = Predicted bbox (when parse failed: nothing drawn)
    White text   = Description label
    Gray text    = Secondary info
"""

from PIL import Image, ImageDraw
from typing import Optional, Tuple, List


# ── Existing functions ────────────────────────────────────────────────────────

def draw_bboxes(
    image,
    pred_boxes: Optional[List[Tuple]] = None,
    target_boxes: Optional[List[Tuple]] = None,
    labels: Optional[List[str]] = None,
    colors: Tuple[str, str] = ("red", "lime"),
) -> Image.Image:
    """
    Draw predicted and ground truth bounding boxes on an image.

    What this does:
        1. Convert image to PIL if needed
        2. Draw each pred box in colors[0] (default: red)
        3. Draw each target box in colors[1] (default: lime/green)
        4. Optionally add text labels

    Args:
        image       : PIL Image or numpy array
        pred_boxes  : List of (xc, yc, w, h) normalized tuples (predictions)
        target_boxes: List of (xc, yc, w, h) normalized tuples (ground truth)
        labels      : Optional text labels for each box
        colors      : (pred_color, target_color) as color name strings

    Returns:
        PIL Image with boxes drawn

    Raises:
        ValueError  : If image is None
        RuntimeError: If drawing fails
    """
    if image is None:
        raise ValueError("[draw_bboxes] image is None")

    if not isinstance(image, Image.Image):
        try:
            import numpy as np
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            else:
                raise ValueError(
                    f"[draw_bboxes] Unsupported image type: {type(image)}"
                )
        except Exception as e:
            raise RuntimeError(f"[draw_bboxes] Image conversion failed: {e}")

    img = image.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size

    def _to_pixels(xc, yc, bw, bh):
        x1 = max(0, int((xc - bw / 2) * W))
        y1 = max(0, int((yc - bh / 2) * H))
        x2 = min(W - 1, int((xc + bw / 2) * W))
        y2 = min(H - 1, int((yc + bh / 2) * H))
        return x1, y1, x2, y2

    # Draw predictions
    if pred_boxes:
        for i, box in enumerate(pred_boxes):
            try:
                xc, yc, bw, bh = box
                x1, y1, x2, y2 = _to_pixels(xc, yc, bw, bh)
                for t in range(3):
                    draw.rectangle(
                        [x1 - t, y1 - t, x2 + t, y2 + t],
                        outline=colors[0]
                    )
                if labels and i < len(labels):
                    draw.text((x1, max(0, y1 - 15)), labels[i], fill=colors[0])
            except Exception as e:
                raise RuntimeError(
                    f"[draw_bboxes] Failed to draw pred box {i}: {e}"
                )

    # Draw ground truth
    if target_boxes:
        for i, box in enumerate(target_boxes):
            try:
                xc, yc, bw, bh = box
                x1, y1, x2, y2 = _to_pixels(xc, yc, bw, bh)
                for t in range(3):
                    draw.rectangle(
                        [x1 - t, y1 - t, x2 + t, y2 + t],
                        outline=colors[1]
                    )
            except Exception as e:
                raise RuntimeError(
                    f"[draw_bboxes] Failed to draw target box {i}: {e}"
                )

    return img


def display_comparison(
    image,
    pred_box: Tuple,
    target_box: Tuple,
    label: str = "",
) -> Image.Image:
    """
    Create a side-by-side comparison image for evaluation.

    What this does:
        1. Draw ground truth (green) and prediction (red) on same image
        2. Add label text below the image
        3. Used by eval.py for all three models

    Args:
        image      : PIL Image
        pred_box   : (xc, yc, w, h) normalized prediction
        target_box : (xc, yc, w, h) normalized ground truth
        label      : Optional text label to add below image

    Returns:
        PIL Image with both boxes and label

    Raises:
        ValueError  : If image, pred_box, or target_box is None
        RuntimeError: If drawing fails
    """
    if image is None:
        raise ValueError("[display_comparison] image is None")
    if pred_box is None:
        raise ValueError("[display_comparison] pred_box is None")
    if target_box is None:
        raise ValueError("[display_comparison] target_box is None")

    try:
        result = draw_bboxes(
            image,
            pred_boxes=[pred_box],
            target_boxes=[target_box],
            colors=("red", "lime"),
        )
    except Exception as e:
        raise RuntimeError(
            f"[display_comparison] draw_bboxes failed: {e}"
        )

    if label:
        W, H = result.size
        label_h = 30
        canvas = Image.new("RGB", (W, H + label_h), (20, 20, 20))
        canvas.paste(result, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((5, H + 5), label[:80], fill="white")
        return canvas

    return result


def draw_comparison(
    pil_image: Image.Image,
    pred_bbox: Optional[List[float]],
    gt_bbox: List[float],
    text: str,
    parsed: bool,
    model_name: str = "baseline",
) -> Image.Image:
    """
    Draw baseline comparison: ground truth (green) + prediction (red).

    What this does:
        1. Load original COCO image
        2. Draw green box for ground truth
        3. Draw red box for prediction (only if parsed=True)
        4. Add text label and status (PARSED/FAILED) below image
        5. Return PIL Image ready to save as PNG

    How to interpret the output:
        - Green box = where the object actually is (ground truth)
        - Red box   = where LLaVA thought the object was
        - FAILED    = LLaVA output could not be parsed -> IoU = 0
        - Big gap between red and green = why we need Spatial-LLaVA

    Args:
        pil_image  : PIL.Image.Image original COCO image
        pred_bbox  : [xc, yc, w, h] predicted or None if parse failed
        gt_bbox    : [xc, yc, w, h] ground truth
        text       : Referring expression string
        parsed     : True if pred_bbox was successfully parsed
        model_name : Label prefix for the image

    Returns:
        PIL Image with annotations and label area

    Raises:
        ValueError  : If pil_image or gt_bbox is None
        RuntimeError: If drawing fails
    """
    if pil_image is None:
        raise ValueError("[draw_comparison] pil_image is None")
    if gt_bbox is None:
        raise ValueError("[draw_comparison] gt_bbox is None")

    try:
        img = pil_image.copy()
        draw = ImageDraw.Draw(img)
        W, H = img.size

        def _draw_box(box, color):
            xc, yc, bw, bh = box
            x1 = max(0, int((xc - bw / 2) * W))
            y1 = max(0, int((yc - bh / 2) * H))
            x2 = min(W - 1, int((xc + bw / 2) * W))
            y2 = min(H - 1, int((yc + bh / 2) * H))
            for t in range(3):
                draw.rectangle(
                    [x1 - t, y1 - t, x2 + t, y2 + t], outline=color
                )

        # Ground truth: always draw (green)
        _draw_box(gt_bbox, "lime")

        # Prediction: only draw if parsed (red)
        if parsed and pred_bbox is not None:
            _draw_box(pred_bbox, "red")

    except Exception as e:
        raise RuntimeError(
            f"[draw_comparison] Failed to draw boxes: {e}"
        )

    # Label area below image
    label_h = 60
    try:
        canvas = Image.new("RGB", (W, H + label_h), (20, 20, 20))
        canvas.paste(img, (0, 0))
        draw2 = ImageDraw.Draw(canvas)

        status = "PARSED OK" if parsed else "FAILED"
        status_color = "lime" if parsed else "red"
        prefix = f"[{model_name.upper()}] [{status}]"

        draw2.text((5, H + 2), f"{prefix} {text[:50]}", fill=status_color)
        draw2.text(
            (5, H + 22),
            "Green=GroundTruth  Red=Prediction",
            fill="gray"
        )
        draw2.text(
            (5, H + 40),
            "FAILED: LLaVA couldn't output parseable coords -> IoU=0",
            fill="gray"
        )
    except Exception as e:
        raise RuntimeError(
            f"[draw_comparison] Failed to add label area: {e}"
        )

    return canvas
