import streamlit as st
import cv2
import numpy as np
import time
import os
import sys
import json
from PIL import Image

# Add current folder to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.detector import PatternDetector
from src.utils import visualize_results

st.set_page_config(
    page_title="Zero-Shot CAD Pattern Detector",
    page_icon="🎯",
    layout="wide"
)

# Custom premium styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #888;
        margin-bottom: 2rem;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #00C9FF 0%, #92FE9D 100%);
        color: black !important;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 201, 255, 0.4);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎯 Zero-Shot CAD Pattern Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect any symbol in complex technical drawings instantly using zero-shot pattern matching.</div>', unsafe_allow_html=True)

# Helper function to read image safely
def load_image(uploaded_file):
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img
    return None

# Load preset helper
def load_preset(name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    examples_dir = os.path.join(base_dir, "examples")
    
    drawing_path = os.path.join(examples_dir, f"{name}_drawing.png")
    template_path = os.path.join(examples_dir, f"{name}_pattern.png")
    
    drawing = cv2.imread(drawing_path)
    template = cv2.imread(template_path)
    
    # Fallback to UTF-8 safe reading if cv2.imread returns None on Windows paths
    if drawing is None and os.path.exists(drawing_path):
        drawing = cv2.imdecode(np.fromfile(drawing_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if template is None and os.path.exists(template_path):
        template = cv2.imdecode(np.fromfile(template_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        
    return drawing, template

# Sidebar section for configurations
st.sidebar.markdown("### 🛠️ Configuration & Inputs")

# Presets Selectbox
preset_option = st.sidebar.selectbox(
    "Choose a preset or upload custom images:",
    ["Custom Upload", "Fuse (Example 1)", "Resistor (Example 2)", "Diode (Example 3)"]
)

# Parameters mapping based on preset
default_params = {
    "tm_thresh": 0.50,
    "recall_thresh": 0.70,
    "rec_nodil_thresh": 0.55,
    "asymmetry_thresh": 0.0,
    "score_thresh": 0.0,
    "scales_str": "0.08, 0.09, 0.10, 0.11",
    "angles_list": [0, 90],
    "exclude_tables": True
}

if "Fuse" in preset_option:
    default_params = {
        "tm_thresh": 0.50,
        "recall_thresh": 0.70,
        "rec_nodil_thresh": 0.55,
        "asymmetry_thresh": 0.0,
        "score_thresh": 0.0,
        "scales_str": "0.08, 0.09, 0.10, 0.11",
        "angles_list": [0, 90],
        "exclude_tables": True
    }
elif "Resistor" in preset_option:
    default_params = {
        "tm_thresh": 0.60,
        "recall_thresh": 0.70,
        "rec_nodil_thresh": 0.35,
        "asymmetry_thresh": 0.0,
        "score_thresh": 0.0,
        "scales_str": "0.11, 0.12",
        "angles_list": [0, 90],
        "exclude_tables": True
    }
elif "Diode" in preset_option:
    default_params = {
        "tm_thresh": 0.50,
        "recall_thresh": 0.70,
        "rec_nodil_thresh": 0.50,
        "asymmetry_thresh": 0.0,
        "score_thresh": 0.75,
        "scales_str": "0.04, 0.05, 0.06",
        "angles_list": [0, 90, 180, 270],
        "exclude_tables": True
    }

# File Uploaders (if Custom Upload)
drawing_img = None
template_img = None

if preset_option == "Custom Upload":
    drawing_file = st.sidebar.file_uploader("Upload Drawing Image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    template_file = st.sidebar.file_uploader("Upload Query Symbol Image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    
    if drawing_file:
        drawing_img = load_image(drawing_file)
    if template_file:
        template_img = load_image(template_file)
else:
    preset_name = preset_option.split(" ")[0].lower()
    drawing_img, template_img = load_preset(preset_name)

# Sliders for Threshold Tuning
st.sidebar.markdown("### 🎛️ Matching Thresholds")
tm_thresh = st.sidebar.slider("Coarse Match Threshold (TM)", 0.10, 1.00, default_params["tm_thresh"], 0.05)
recall_thresh = st.sidebar.slider("Recall Threshold (Dilated)", 0.10, 1.00, default_params["recall_thresh"], 0.05)
rec_nodil_thresh = st.sidebar.slider("Recall Threshold (Raw)", 0.10, 1.00, default_params["rec_nodil_thresh"], 0.05)
asymmetry_thresh = st.sidebar.slider("Asymmetry Threshold (Diode)", 0.00, 1.00, default_params["asymmetry_thresh"], 0.05)
score_thresh = st.sidebar.slider("Combined Score Threshold", 0.00, 1.00, default_params["score_thresh"], 0.05)

# Scale Grid
st.sidebar.markdown("### 📐 Scale & Rotation Grid")
scales_input = st.sidebar.text_input("Scales (comma-separated list):", default_params["scales_str"])
try:
    scales = [float(s.strip()) for s in scales_input.split(",") if s.strip()]
except ValueError:
    st.sidebar.error("Invalid scales input. Use numbers separated by commas.")
    scales = [0.10]

# Rotation Angles Multi-Select
all_angles = [0, 45, 90, 135, 180, 225, 270, 315]
selected_angles = st.sidebar.multiselect("Rotation Angles (degrees):", all_angles, default=default_params["angles_list"])

# Exclusion Filters
st.sidebar.markdown("### 🛡️ Noise & Area Filtering")
remove_text = st.sidebar.checkbox("Remove Text and Noise (CC Filter)", value=True)
exclude_tables = st.sidebar.checkbox("Exclude Title Block & BOM Table (x >= 1050)", value=default_params["exclude_tables"])

# Main Dashboard Layout
col_preview, col_controls = st.columns([1, 1])

with col_preview:
    st.markdown("### 🖼️ Inputs Preview")
    if drawing_img is not None:
        st.write(f"Drawing dimensions: **{drawing_img.shape[1]}x{drawing_img.shape[0]}**")
        # Resize for preview to prevent huge rendering
        preview_h = 400
        aspect = drawing_img.shape[1] / drawing_img.shape[0]
        preview_w = int(preview_h * aspect)
        st.image(cv2.cvtColor(cv2.resize(drawing_img, (preview_w, preview_h)), cv2.COLOR_BGR2RGB), caption="CAD Drawing Preview", use_column_width=False)
    else:
        st.info("Please upload a drawing image or select a preset.")

with col_controls:
    if template_img is not None:
        st.markdown("### 🔍 Query Symbol Pattern")
        st.image(cv2.cvtColor(template_img, cv2.COLOR_BGR2RGB), caption="Cropped Query Symbol", width=120)
    else:
        st.info("Please upload a query template symbol.")

# Run Detection
if st.button("🚀 Run Pattern Detection"):
    if drawing_img is None or template_img is None:
        st.error("Both drawing and template images must be loaded.")
    else:
        with st.spinner("Processing drawing... this may take up to 10 seconds on CPU."):
            detector = PatternDetector(
                tm_thresh=tm_thresh,
                recall_thresh=recall_thresh,
                rec_nodil_thresh=rec_nodil_thresh,
                asymmetry_thresh=asymmetry_thresh
            )
            
            start_time = time.time()
            detections = detector.detect(
                drawing_img,
                template_img,
                scales=scales,
                angles=selected_angles,
                binarize_thresh=240,
                remove_text=remove_text
            )
            
            # Apply Table Region filter if checked
            if exclude_tables:
                detections = [d for d in detections if d['box'][0] < 1050]
                
            # Filter by combined score
            if score_thresh > 0.0:
                detections = [d for d in detections if d['score'] >= score_thresh]
                
            elapsed = time.time() - start_time
            
        st.success(f"Finished in **{elapsed:.2f} seconds**! Detected **{len(detections)}** instances.")
        
        # Visualize
        vis_img = visualize_results(drawing_img, detections)
        
        # Save output image to a temp location for download
        output_pil = Image.fromarray(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        
        st.markdown("### 📊 Detection Results")
        st.image(output_pil, caption="Detected Instances with Bounding Boxes & Confidence Scores", use_column_width=True)
        
        # Export Detections
        if len(detections) > 0:
            # Format detections for JSON / CSV
            formatted_detections = []
            for idx, d in enumerate(detections):
                box = d['box']
                formatted_detections.append({
                    "id": idx,
                    "box_x1": int(box[0]),
                    "box_y1": int(box[1]),
                    "box_x2": int(box[2]),
                    "box_y2": int(box[3]),
                    "width": int(box[2] - box[0]),
                    "height": int(box[3] - box[1]),
                    "confidence": round(float(d['score']), 4),
                    "scale": round(float(d['scale']), 2),
                    "angle": int(d['angle'])
                })
            
            col_tbl, col_json = st.columns([1, 1])
            with col_tbl:
                st.markdown("#### Bounding Box Table")
                st.dataframe(formatted_detections)
            with col_json:
                st.markdown("#### Export JSON")
                st.code(json.dumps(formatted_detections, indent=2), language="json")
        else:
            st.warning("No patterns detected. Try lowering the thresholds or adjusting the scale range in the sidebar.")
