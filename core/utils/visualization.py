"""
core/utils/visualization.py

Utilities for drawing predicted and ground truth bounding boxes onto images.
Used for debugging and result visualization.

Usage:
    from core.utils.visualization import draw_bboxes, display_comparison

    img = Image.open("image.jpg")
    result = draw_bboxes(img, pred_boxes=[(0.5, 0.5, 0.3, 0.2)],
                                target_boxes=[(0.5, 0.5, 0.4, 0.3)])
    result.save("output.jpg")
"""

from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


def _xywh_to_xyxy(
    box: Sequence[float],
) -> Tuple[float, float, float, float]:
    """
    Convert [x_center, y_center, w, h] to [x1, y1, x2, y2].

    Args:
        box : Sequence of 4 floats in normalized [0, 1] space

    Returns:
        Tuple (x1, y1, x2, y2)
    """
    x, y, w, h = box
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


def draw_bboxes(
    image: Image.Image,
    pred_boxes: List[Sequence[float]],
    target_boxes: List[Sequence[float]],
    labels: Optional[List[str]] = None,
    colors: Tuple[str, str] = ("red", "lime"),
    thickness: int = 2,
) -> Image.Image:
    """
    Draw predicted and ground truth bounding boxes onto an image.

    Both pred_boxes and target_boxes are in normalized
    [x_center, y_center, width, height] format with values in [0, 1].

    Args:
        image       : PIL Image (any mode, converted to RGB internally)
        pred_boxes  : List of predicted boxes [(x_c, y_c, w, h), ...]
        target_boxes: List of ground truth boxes [(x_c, y_c, w, h), ...]
        labels      : Optional list of label strings, one per box pair
        colors      : (pred_color, target_color) as color name strings
        thickness   : Border line width in pixels

    Returns:
        New PIL Image with boxes drawn (does not modify the input)

    Raises:
        ValueError : If pred_boxes and target_boxes have different lengths
    """
    if len(pred_boxes) != len(target_boxes):
        raise ValueError("pred_boxes and target_boxes must have same length")

    img = image.convert("RGB")
    img_w, img_h = img.size
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    for i, (pred, target) in enumerate(zip(pred_boxes, target_boxes)):
        px1, py1, px2, py2 = _xywh_to_xyxy(pred)
        tx1, ty1, tx2, ty2 = _xywh_to_xyxy(target)

        # Convert normalized coords to pixel coords
        pred_coords = (px1 * img_w, py1 * img_h, px2 * img_w, py2 * img_h)
        target_coords = (tx1 * img_w, ty1 * img_h, tx2 * img_w, ty2 * img_h)

        draw.rectangle(pred_coords, outline=colors[0], width=thickness)
        draw.rectangle(target_coords, outline=colors[1], width=thickness)

        if labels is not None and i < len(labels):
            draw.text(
                (pred_coords[0], max(pred_coords[1] - 12, 0)),
                labels[i],
                fill=colors[0],
                font=font,
            )

    return img


def display_comparison(
    image: Image.Image,
    pred_box: Sequence[float],
    target_box: Sequence[float],
) -> Image.Image:
    """
    Draw a single predicted vs ground truth box onto an image.

    Convenience wrapper around draw_bboxes() for single-sample visualization.

    Args:
        image      : PIL Image
        pred_box   : Predicted box (x_c, y_c, w, h) normalized
        target_box : Ground truth box (x_c, y_c, w, h) normalized

    Returns:
        New PIL Image with boxes drawn
    """
    return draw_bboxes(image, [pred_box], [target_box])
