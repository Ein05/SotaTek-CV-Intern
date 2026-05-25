import cv2
import numpy as np
import time
import os
import sys

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.detector import PatternDetector
from src.utils import visualize_results

def cv2_imread_unicode(path):
    # Safe read of unicode path on Windows
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

def cv2_imwrite_unicode(path, img):
    ext = os.path.splitext(path)[1]
    ret, buf = cv2.imencode(ext, img)
    if ret:
        buf.tofile(path)
        return True
    return False

def run_test_case(name, drawing_path, template_path, scales, angles, detector_params, score_thresh=0.0):
    print(f"\n=================== RUNNING TEST CASE: {name} ===================")
    drawing = cv2_imread_unicode(drawing_path)
    template = cv2_imread_unicode(template_path)
    
    if drawing is None or template is None:
        print(f"Error loading files. Check paths.")
        return
        
    detector = PatternDetector(**detector_params)
    
    start_time = time.time()
    detections = detector.detect(
        drawing, 
        template, 
        scales=scales, 
        angles=angles, 
        binarize_thresh=240,
        remove_text=True
    )
    elapsed = time.time() - start_time
    
    # Filter out table region (x1 >= 1050) since schematic and legend are in x < 1050
    detections = [d for d in detections if d['box'][0] < 1050]
    
    # Filter by combined score
    if score_thresh > 0.0:
        detections = [d for d in detections if d['score'] >= score_thresh]
        
    print(f"Finished in {elapsed:.4f} seconds.")
    print(f"Detected {len(detections)} instances of the pattern:")
    for idx, det in enumerate(detections):
        box = det['box']
        score = det['score']
        scale = det['scale']
        angle = det['angle']
        print(f"  [{idx}] Box: {box}, Score (Combined): {score:.4f}, Scale: {scale:.2f}, Angle: {angle}")
        
    # Visualize and save output
    vis_img = visualize_results(drawing, detections)
    out_path = f"verify_{name}_output.png"
    cv2_imwrite_unicode(out_path, vis_img)
    print(f"Saved visualization to {out_path}")
    return detections

if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    examples_dir = os.path.join(project_dir, "examples")
    
    # 1. Fuses (Example 1)
    # Expected: exactly 5 detections (F1, F2, F3, F4, legend)
    fuse_scales = [0.08, 0.09, 0.10, 0.11]
    fuse_angles = [0, 90]
    fuse_params = {
        'tm_thresh': 0.50,
        'recall_thresh': 0.70,
        'rec_nodil_thresh': 0.55,
        'asymmetry_thresh': 0.0
    }
    run_test_case(
        "fuse", 
        os.path.join(examples_dir, "fuse_drawing.png"),
        os.path.join(examples_dir, "fuse_pattern.png"),
        scales=fuse_scales,
        angles=fuse_angles,
        detector_params=fuse_params
    )
    
    # 2. Resistors (Example 2)
    # Expected: exactly 10 detections (R1-R9, legend)
    resistor_scales = [0.11, 0.12]
    resistor_angles = [0, 90]
    resistor_params = {
        'tm_thresh': 0.60,
        'recall_thresh': 0.70,
        'rec_nodil_thresh': 0.35,
        'asymmetry_thresh': 0.0
    }
    run_test_case(
        "resistor", 
        os.path.join(examples_dir, "resistor_drawing.png"),
        os.path.join(examples_dir, "resistor_pattern.png"),
        scales=resistor_scales,
        angles=resistor_angles,
        detector_params=resistor_params
    )
    
    # 3. Diodes (Example 3)
    # Expected: exactly 12 detections (8 in rectifiers, 4 in legend rectifier symbol)
    diode_scales = [0.04, 0.05, 0.06]
    diode_angles = [0, 90, 180, 270]
    diode_params = {
        'tm_thresh': 0.50,
        'recall_thresh': 0.70,
        'rec_nodil_thresh': 0.50,
        'asymmetry_thresh': 0.0
    }
    run_test_case(
        "diode", 
        os.path.join(examples_dir, "diode_drawing.png"),
        os.path.join(examples_dir, "diode_pattern.png"),
        scales=diode_scales,
        angles=diode_angles,
        detector_params=diode_params,
        score_thresh=0.75
    )
