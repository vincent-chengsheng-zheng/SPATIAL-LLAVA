"""
demo/demo_gradio.py

Gradio demo: side-by-side visual grounding comparison across 3 models.
  - Baseline  : Standard LLaVA + regex parsing  (IoU=0.097)
  - Ablation  : Frozen LLaVA + MLP head         (IoU=0.284)
  - Main      : LoRA fine-tuned + MLP head      (IoU=0.386)

Run on cluster:
    cd /tmp/SPATIAL-LLAVA
    pip install gradio --break-system-packages
    export TRANSFORMERS_OFFLINE=1
    python demo/demo_gradio.py

Then open the public URL printed in the terminal (share=True).
"""

import os
import sys
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import gradio as gr

from core.model.spatial_llava import load_model
from core.model.llava import StandardLLaVA
from core.data.preprocessing import PROMPT_TEMPLATE
from core.paths import PATHS


# ── Config ────────────────────────────────────────────────────────────────────

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_ABLATION = "/tmp/SPATIAL-LLAVA/checkpoints/ablation/best.pth"
CKPT_MAIN     = "/tmp/SPATIAL-LLAVA/checkpoints/main/best.pth"

COLORS = {
    "Baseline": "#FF4444",
    "Ablation": "#FF9900",
    "Main":     "#00CC66",
}


# ── Model Loading ─────────────────────────────────────────────────────────────

def load_all_models():
    print("\n[Demo] Loading models... (~1-2 min)")

    print("[Demo] 1/3 Baseline...")
    baseline = StandardLLaVA(
        model_id="llava-hf/llava-1.5-7b-hf",
        hf_cache=str(PATHS.weights),
    )
    baseline.to(DEVICE).eval()

    print("[Demo] 2/3 Ablation...")
    ablation_model, ablation_processor = load_model(use_lora=False, device=DEVICE)
    ckpt = torch.load(CKPT_ABLATION, map_location=DEVICE)
    ablation_model.load_state_dict(ckpt["model_state"], strict=False)
    ablation_model.eval()

    print("[Demo] 3/3 Main (LoRA)...")
    main_model, main_processor = load_model(use_lora=True, device=DEVICE)
    ckpt = torch.load(CKPT_MAIN, map_location=DEVICE)
    main_model.load_state_dict(ckpt["model_state"], strict=False)
    main_model.eval()

    print("[Demo] All models loaded.\n")
    return baseline, ablation_model, ablation_processor, main_model, main_processor


baseline_model, ablation_model, ablation_processor, main_model, main_processor = load_all_models()


# ── Inference ─────────────────────────────────────────────────────────────────

def run_baseline(image: Image.Image, text: str):
    prompt = PROMPT_TEMPLATE.format(text=text)
    inputs = baseline_model.processor(
        text=[prompt], images=[image],
        return_tensors="pt", padding=True,
    ).to(DEVICE)
    with torch.no_grad():
        output_ids = baseline_model.model.generate(
            **inputs, max_new_tokens=50, do_sample=False,
        )
    raw = baseline_model.processor.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()
    bbox = StandardLLaVA.parse_bbox(raw)
    return (bbox if bbox else [0.0, 0.0, 0.0, 0.0]), raw


def run_spatial(model, processor, image: Image.Image, text: str):
    prompt = PROMPT_TEMPLATE.format(text=text)
    inputs = processor(
        text=[prompt], images=[image],
        return_tensors="pt", padding=True,
    ).to(DEVICE)
    with torch.no_grad():
        pred = model(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["pixel_values"],
        )
    return pred[0].cpu().tolist()


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_bbox(image: Image.Image, bbox: list, color: str, label: str) -> Image.Image:
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    W, H = img.size
    xc, yc, w, h = bbox
    x1 = max(0, int((xc - w / 2) * W))
    y1 = max(0, int((yc - h / 2) * H))
    x2 = min(W, int((xc + w / 2) * W))
    y2 = min(H, int((yc + h / 2) * H))

    for offset in range(3):
        draw.rectangle(
            [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
            outline=color, width=1,
        )

    label_text = f" {label} "
    text_bbox  = draw.textbbox((x1, y1), label_text)
    text_h     = text_bbox[3] - text_bbox[1]
    label_y    = max(0, y1 - text_h - 4)
    draw.rectangle(
        [x1, label_y, x1 + (text_bbox[2] - text_bbox[0]), label_y + text_h + 4],
        fill=color,
    )
    draw.text((x1, label_y + 2), label_text, fill="white")
    return img


def compute_iou(pred: list, gt: list) -> float:
    def to_xyxy(b):
        xc, yc, w, h = b
        return xc - w/2, yc - h/2, xc + w/2, yc + h/2
    px1, py1, px2, py2 = to_xyxy(pred)
    gx1, gy1, gx2, gy2 = to_xyxy(gt)
    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (px2-px1)*(py2-py1) + (gx2-gx1)*(gy2-gy1) - inter
    return round(inter / union, 4) if union > 0 else 0.0


# ── Gradio Logic ──────────────────────────────────────────────────────────────

def predict(image, text, gt_x, gt_y, gt_w, gt_h):
    if image is None:
        return None, None, None, "Please upload an image."
    if not text or not text.strip():
        return None, None, None, "Please enter a text description."

    image = Image.fromarray(image).convert("RGB")

    baseline_bbox, baseline_raw = run_baseline(image, text)
    ablation_bbox = run_spatial(ablation_model, ablation_processor, image, text)
    main_bbox     = run_spatial(main_model, main_processor, image, text)

    img_baseline = draw_bbox(image, baseline_bbox, COLORS["Baseline"], "Baseline")
    img_ablation = draw_bbox(image, ablation_bbox, COLORS["Ablation"], "Ablation")
    img_main     = draw_bbox(image, main_bbox,     COLORS["Main"],     "Main")

    info = [
        f"Text: {text}",
        f"",
        f"Baseline  bbox: [{', '.join(f'{v:.3f}' for v in baseline_bbox)}]",
        f"Ablation  bbox: [{', '.join(f'{v:.3f}' for v in ablation_bbox)}]",
        f"Main      bbox: [{', '.join(f'{v:.3f}' for v in main_bbox)}]",
    ]

    has_gt = all(v is not None for v in [gt_x, gt_y, gt_w, gt_h])
    if has_gt:
        gt_bbox = [float(gt_x), float(gt_y), float(gt_w), float(gt_h)]
        info += [
            f"",
            f"IoU vs ground truth:",
            f"  Baseline : {compute_iou(baseline_bbox, gt_bbox):.4f}",
            f"  Ablation : {compute_iou(ablation_bbox, gt_bbox):.4f}",
            f"  Main     : {compute_iou(main_bbox,     gt_bbox):.4f}",
        ]

    info += [f"", f"Baseline raw output: {baseline_raw}"]
    return img_baseline, img_ablation, img_main, "\n".join(info)


# ── UI ────────────────────────────────────────────────────────────────────────

DESCRIPTION = """
# Spatial-LLaVA: Visual Grounding Comparison

Upload an image and describe an object. All three models predict a bounding box.

| Model | Val IoU | Test IoU | Method |
|---|---|---|---|
| 🔴 Baseline | 0.097 | 0.097 | Vanilla LLaVA + regex |
| 🟠 Ablation | 0.267 | 0.284 | Frozen LLaVA + MLP head |
| 🟢 Main     | 0.357 | 0.386 | LoRA fine-tuned + MLP head |
"""

with gr.Blocks(title="Spatial-LLaVA Demo") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Input Image", type="numpy")
            text_input  = gr.Textbox(
                label="Referring Expression",
                placeholder='e.g. "the person on the left"',
            )
            gr.Markdown("**Optional: Ground Truth BBox** `[xc, yc, w, h]` normalised 0–1")
            with gr.Row():
                gt_x = gr.Number(label="xc", value=None, precision=3)
                gt_y = gr.Number(label="yc", value=None, precision=3)
                gt_w = gr.Number(label="w",  value=None, precision=3)
                gt_h = gr.Number(label="h",  value=None, precision=3)
            run_btn = gr.Button("Run Prediction", variant="primary")

        with gr.Column(scale=2):
            with gr.Row():
                out_baseline = gr.Image(label="🔴 Baseline")
                out_ablation = gr.Image(label="🟠 Ablation")
                out_main     = gr.Image(label="🟢 Main")
            info_box = gr.Textbox(label="Details", lines=12)

    run_btn.click(
        fn=predict,
        inputs=[image_input, text_input, gt_x, gt_y, gt_w, gt_h],
        outputs=[out_baseline, out_ablation, out_main, info_box],
    )

    gr.Examples(
        examples=[
            ["results/ablation/examples/example_0025.png", "the person on the left"],
            ["results/main/examples/example_0089.png",     "the red object"],
            ["results/ablation/examples/example_0604.png", "the man wearing a hat"],
        ],
        inputs=[image_input, text_input],
    )

if __name__ == "__main__":
    demo.launch(inline=True)

