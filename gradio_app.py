# ── Monkeypatch to fix a known bug in gradio_client with Pydantic v2 ──────────
try:
    import gradio_client.utils
    orig_json_schema_to_python_type = getattr(gradio_client.utils, "_json_schema_to_python_type", None)
    orig_get_type = getattr(gradio_client.utils, "get_type", None)

    def patched_json_schema_to_python_type(schema, defs=None):
        if isinstance(schema, bool):
            return "any"
        if orig_json_schema_to_python_type:
            return orig_json_schema_to_python_type(schema, defs)
        return "any"

    def patched_get_type(schema):
        if isinstance(schema, bool):
            return "any"
        if orig_get_type:
            return orig_get_type(schema)
        return "any"

    if hasattr(gradio_client.utils, "_json_schema_to_python_type"):
        gradio_client.utils._json_schema_to_python_type = patched_json_schema_to_python_type
    if hasattr(gradio_client.utils, "get_type"):
        gradio_client.utils.get_type = patched_get_type
except Exception:
    pass
# ─────────────────────────────────────────────────────────────────────────────

import gradio as gr
import cv2
import numpy as np
import json
import os
import time
from PIL import Image

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.detector import PatternDetector
from src.utils import visualize_results

# ── Preset Configs ────────────────────────────────────────────────────────────
PRESETS = {
    "🔌 Fuse (Example 1)": {
        "tm_thresh": 0.50, "recall_thresh": 0.70, "rec_nodil_thresh": 0.55,
        "asymmetry_thresh": 0.0, "score_thresh": 0.0,
        "scales_str": "0.08, 0.09, 0.10, 0.11",
        "angles": [0, 90], "exclude_tables": True,
        "drawing": "examples/fuse_drawing.png",
        "template": "examples/fuse_pattern.png",
    },
    "⬛ Resistor (Example 2)": {
        "tm_thresh": 0.60, "recall_thresh": 0.70, "rec_nodil_thresh": 0.35,
        "asymmetry_thresh": 0.0, "score_thresh": 0.0,
        "scales_str": "0.11, 0.12",
        "angles": [0, 90], "exclude_tables": True,
        "drawing": "examples/resistor_drawing.png",
        "template": "examples/resistor_pattern.png",
    },
    "🔺 Diode (Example 3)": {
        "tm_thresh": 0.50, "recall_thresh": 0.70, "rec_nodil_thresh": 0.50,
        "asymmetry_thresh": 0.0, "score_thresh": 0.75,
        "scales_str": "0.04, 0.05, 0.06",
        "angles": [0, 90, 180, 270], "exclude_tables": True,
        "drawing": "examples/diode_drawing.png",
        "template": "examples/diode_pattern.png",
    },
    "📂 Custom Upload": {
        "tm_thresh": 0.50, "recall_thresh": 0.70, "rec_nodil_thresh": 0.55,
        "asymmetry_thresh": 0.0, "score_thresh": 0.0,
        "scales_str": "0.08, 0.09, 0.10, 0.11",
        "angles": [0, 90], "exclude_tables": True,
        "drawing": None, "template": None,
    },
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_preset(preset_name):
    """Returns preset images and parameters when user selects a preset."""
    cfg = PRESETS[preset_name]
    drawing_img = None
    template_img = None
    if cfg["drawing"] and cfg["template"]:
        drawing_path = os.path.join(BASE_DIR, cfg["drawing"])
        template_path = os.path.join(BASE_DIR, cfg["template"])
        drawing_img = np.array(Image.open(drawing_path).convert("RGB"))
        template_img = np.array(Image.open(template_path).convert("RGB"))
    angles_str = ", ".join(str(a) for a in cfg["angles"])
    return (
        drawing_img,
        template_img,
        cfg["tm_thresh"],
        cfg["recall_thresh"],
        cfg["rec_nodil_thresh"],
        cfg["asymmetry_thresh"],
        cfg["score_thresh"],
        cfg["scales_str"],
        angles_str,
        cfg["exclude_tables"],
    )


def run_detection(
    drawing_input,
    template_input,
    preset_name,
    tm_thresh,
    recall_thresh,
    rec_nodil_thresh,
    asymmetry_thresh,
    score_thresh,
    scales_str,
    angles_str,
    remove_text,
    exclude_tables,
):
    """Core detection function called when user clicks Run."""
    # ── Resolve images ─────────────────────────────────────────────────────
    cfg = PRESETS.get(preset_name, PRESETS["📂 Custom Upload"])

    if drawing_input is None:
        if cfg["drawing"]:
            drawing_bgr = cv2.imread(os.path.join(BASE_DIR, cfg["drawing"]))
        else:
            return None, "❌ Please upload a Drawing image.", "{}"
    else:
        drawing_bgr = cv2.cvtColor(np.array(drawing_input), cv2.COLOR_RGB2BGR)

    if template_input is None:
        if cfg["template"]:
            template_bgr = cv2.imread(os.path.join(BASE_DIR, cfg["template"]))
        else:
            return None, "❌ Please upload a Template (pattern) image.", "{}"
    else:
        template_bgr = cv2.cvtColor(np.array(template_input), cv2.COLOR_RGB2BGR)

    # ── Parse scales & angles ──────────────────────────────────────────────
    try:
        scales = [float(s.strip()) for s in scales_str.split(",") if s.strip()]
    except ValueError:
        return None, "❌ Invalid scales format. Use comma-separated numbers e.g. `0.08, 0.10`", "{}"

    try:
        angles = [int(a.strip()) for a in angles_str.split(",") if a.strip()]
    except ValueError:
        return None, "❌ Invalid angles format. Use comma-separated integers e.g. `0, 90, 180`", "{}"

    if not scales:
        return None, "❌ Scales list is empty.", "{}"
    if not angles:
        return None, "❌ Angles list is empty.", "{}"

    # ── Run detector ───────────────────────────────────────────────────────
    detector = PatternDetector(
        tm_thresh=tm_thresh,
        recall_thresh=recall_thresh,
        rec_nodil_thresh=rec_nodil_thresh,
        asymmetry_thresh=asymmetry_thresh,
    )

    start = time.time()
    detections = detector.detect(
        drawing_bgr,
        template_bgr,
        scales=scales,
        angles=angles,
        binarize_thresh=240,
        remove_text=remove_text,
    )
    elapsed = time.time() - start

    # ── Post-filters ───────────────────────────────────────────────────────
    if exclude_tables:
        detections = [d for d in detections if d["box"][0] < 1050]
    if score_thresh > 0.0:
        detections = [d for d in detections if d["score"] >= score_thresh]

    # ── Visualize ──────────────────────────────────────────────────────────
    vis_bgr = visualize_results(drawing_bgr, detections)
    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    vis_pil = Image.fromarray(vis_rgb)

    # ── Summary text ───────────────────────────────────────────────────────
    count = len(detections)
    status = (
        f"✅ Detected **{count}** instance{'s' if count != 1 else ''} "
        f"in **{elapsed:.2f}s**"
        if count > 0
        else f"⚠️ No patterns detected in {elapsed:.2f}s — try adjusting thresholds or scales."
    )

    # ── JSON output ────────────────────────────────────────────────────────
    formatted = []
    for idx, d in enumerate(detections):
        box = d["box"]
        formatted.append({
            "id": idx,
            "box": {"x1": int(box[0]), "y1": int(box[1]), "x2": int(box[2]), "y2": int(box[3])},
            "width": int(box[2] - box[0]),
            "height": int(box[3] - box[1]),
            "confidence": round(float(d["score"]), 4),
            "scale": round(float(d["scale"]), 3),
            "angle_deg": int(d["angle"]),
        })
    json_out = json.dumps(formatted, indent=2)

    return vis_pil, status, json_out


# ── Custom CSS ────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

* { font-family: 'Outfit', sans-serif !important; }

body, .gradio-container {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%) !important;
    min-height: 100vh;
}

.gr-panel, .gr-box, .gr-form, .panel {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(12px);
}

.gr-button-primary {
    background: linear-gradient(135deg, #00C9FF 0%, #92FE9D 100%) !important;
    color: #0d0d0d !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 32px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(0,201,255,0.3) !important;
}

.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0,201,255,0.5) !important;
}

.gr-button-secondary {
    background: rgba(255,255,255,0.07) !important;
    color: #ccc !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
}

label, .gr-block-label span {
    color: #a0aec0 !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}

input, textarea, select {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

.gr-slider input[type=range]::-webkit-slider-thumb {
    background: #00C9FF !important;
}

h1.title {
    font-size: 2.4rem !important;
    font-weight: 700;
    background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 4px;
}

.subtitle {
    text-align: center;
    color: #718096;
    font-size: 1rem;
    margin-bottom: 20px;
}

.output-status {
    padding: 12px 18px;
    border-radius: 10px;
    background: rgba(0,201,255,0.08);
    border: 1px solid rgba(0,201,255,0.2);
    color: #e2e8f0;
    font-size: 0.95rem;
}
"""

# ── Build Gradio UI ───────────────────────────────────────────────────────────
with gr.Blocks(
    css=CUSTOM_CSS,
    title="Zero-Shot CAD Pattern Detector",
    theme=gr.themes.Base(
        primary_hue="cyan",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Outfit"), "sans-serif"],
    ),
) as demo:

    # Header
    gr.HTML("""
        <h1 class="title">🎯 Zero-Shot CAD Pattern Detector</h1>
        <p class="subtitle">Detect any symbol in complex technical drawings using zero-shot pattern matching — no training required.</p>
    """)

    with gr.Row():
        # ── Left Column: Inputs ───────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Preset or Custom Upload")
            preset_dd = gr.Dropdown(
                choices=list(PRESETS.keys()),
                value="🔌 Fuse (Example 1)",
                label="Choose a preset or upload your own images",
                interactive=True,
            )

            with gr.Row():
                drawing_input = gr.Image(
                    label="📐 Drawing Image",
                    type="pil",
                    sources=["upload"],
                    height=220,
                )
                template_input = gr.Image(
                    label="🔍 Query Symbol (Pattern)",
                    type="pil",
                    sources=["upload"],
                    height=220,
                )

            gr.Markdown("### 🎛️ Matching Thresholds")
            with gr.Row():
                tm_thresh    = gr.Slider(0.10, 1.00, value=0.50, step=0.05, label="Coarse Match (TM)")
                recall_thresh = gr.Slider(0.10, 1.00, value=0.70, step=0.05, label="Recall (Dilated)")
            with gr.Row():
                rec_nodil_thresh = gr.Slider(0.10, 1.00, value=0.55, step=0.05, label="Recall (Raw)")
                score_thresh     = gr.Slider(0.00, 1.00, value=0.00, step=0.05, label="Combined Score Min")
            asymmetry_thresh = gr.Slider(0.00, 1.00, value=0.00, step=0.05, label="Asymmetry Threshold")

            gr.Markdown("### 📐 Scale & Rotation Grid")
            scales_str = gr.Textbox(
                value="0.08, 0.09, 0.10, 0.11",
                label="Scales (comma-separated)",
                placeholder="e.g. 0.08, 0.09, 0.10, 0.11",
            )
            angles_str = gr.Textbox(
                value="0, 90",
                label="Rotation Angles in degrees (comma-separated)",
                placeholder="e.g. 0, 90, 180, 270",
            )

            gr.Markdown("### 🛡️ Noise & Area Filters")
            with gr.Row():
                remove_text    = gr.Checkbox(value=True,  label="Remove Text & Noise (CC Filter)")
                exclude_tables = gr.Checkbox(value=True,  label="Exclude BOM Table (x ≥ 1050)")

            run_btn = gr.Button("🚀  Run Pattern Detection", variant="primary", size="lg")

        # ── Right Column: Outputs ─────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Detection Results")
            status_out = gr.Markdown(
                value="*Results will appear here after running detection.*",
                elem_classes=["output-status"],
            )
            vis_out = gr.Image(
                label="Annotated Output — Detected Instances with Bounding Boxes",
                type="pil",
                height=450,
                interactive=False,
            )
            json_out = gr.Code(
                label="📦 Detections JSON (id, box, confidence, scale, angle)",
                language="json",
                lines=15,
            )

    # ── Wire preset → auto-fill params ───────────────────────────────────
    preset_outputs = [
        drawing_input, template_input,
        tm_thresh, recall_thresh, rec_nodil_thresh,
        asymmetry_thresh, score_thresh,
        scales_str, angles_str, exclude_tables,
    ]
    preset_dd.change(fn=load_preset, inputs=preset_dd, outputs=preset_outputs)

    # ── Wire run button ───────────────────────────────────────────────────
    run_btn.click(
        fn=run_detection,
        inputs=[
            drawing_input, template_input, preset_dd,
            tm_thresh, recall_thresh, rec_nodil_thresh,
            asymmetry_thresh, score_thresh,
            scales_str, angles_str,
            remove_text, exclude_tables,
        ],
        outputs=[vis_out, status_out, json_out],
    )

    # ── Pre-load the first preset on startup ─────────────────────────────
    demo.load(
        fn=load_preset,
        inputs=preset_dd,
        outputs=preset_outputs,
    )

if __name__ == "__main__":
    demo.launch()
