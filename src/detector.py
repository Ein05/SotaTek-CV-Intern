import cv2
import numpy as np
from src.utils import crop_margins, rotate_image, non_max_suppression

class PatternDetector:
    def __init__(self, tm_thresh=0.50, recall_thresh=0.70, rec_nodil_thresh=0.65, asymmetry_thresh=0.35, iou_threshold=0.3):
        """
        Initializes the zero-shot PatternDetector.
        Args:
            tm_thresh: Coarse template matching correlation threshold (default 0.50)
            recall_thresh: Recall threshold on dilated drawing lines (default 0.70)
            rec_nodil_thresh: Recall threshold on raw (undilated) drawing lines (default 0.65)
            asymmetry_thresh: Minimum score variance across angles to filter out symmetric shapes like connection dots (default 0.35)
            iou_threshold: Intersection over Union threshold for NMS (default 0.3)
        """
        self.tm_thresh = tm_thresh
        self.recall_thresh = recall_thresh
        self.rec_nodil_thresh = rec_nodil_thresh
        self.asymmetry_thresh = asymmetry_thresh
        self.iou_threshold = iou_threshold

    def remove_text_and_noise(self, inv_img, max_h=14, max_area=400):
        """
        Removes text character components based on height and area.
        Alphanumeric characters are typically short (height <= 14px) and have small areas.
        """
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inv_img, connectivity=8)
        clean_img = inv_img.copy()
        for i in range(1, num_labels):
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            if h <= max_h and area <= max_area:
                clean_img[labels == i] = 0
        return clean_img

    def detect(self, drawing_img, template_img, scales=None, angles=None, binarize_thresh=240, remove_text=True):
        """
        Runs the zero-shot pattern detection pipeline.
        Args:
            drawing_img: Drawing image (BGR or grayscale)
            template_img: Query pattern symbol template (BGR or grayscale)
            scales: Optional list of scaling factors. If None, auto-generated based on template size.
            angles: Optional list of rotation angles. If None, defaults to [0, 45, 90, 135, 180, 225, 270, 315]
            binarize_thresh: Threshold for line extraction (default 240)
            remove_text: Whether to apply connected component text removal (default True)
        """
        # 1. Grayscale conversion
        if len(drawing_img.shape) == 3:
            drawing_gray = cv2.cvtColor(drawing_img, cv2.COLOR_BGR2GRAY)
        else:
            drawing_gray = drawing_img.copy()

        if len(template_img.shape) == 3:
            template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template_img.copy()

        # 2. Crop margins of the template to isolate the symbol
        template_cropped = crop_margins(template_gray, padding=2, thresh=binarize_thresh)

        # 3. Binarize & Invert
        _, drawing_inv = cv2.threshold(drawing_gray, binarize_thresh, 255, cv2.THRESH_BINARY_INV)
        _, template_inv = cv2.threshold(template_cropped, binarize_thresh, 255, cv2.THRESH_BINARY_INV)

        # 4. Text removal preprocessing
        if remove_text:
            drawing_inv_clean = self.remove_text_and_noise(drawing_inv)
        else:
            drawing_inv_clean = drawing_inv.copy()
        
        # 5. Dilation of drawing for recall tolerance
        kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        drawing_dil = cv2.dilate(drawing_inv_clean, kernel_cross)

        # Auto-compute scale range if not provided
        w_t, h_t = template_inv.shape[1], template_inv.shape[0]
        if scales is None:
            # General CAD drawing symbols are typically 20px to 100px in the drawing
            min_scale = max(0.02, 15.0 / max(w_t, h_t))
            max_scale = min(2.5, 120.0 / min(w_t, h_t))
            scales = np.arange(min_scale, max_scale + 0.01, 0.01)

        # Default angles to cover 8 orientations
        if angles is None:
            angles = [0, 45, 90, 135, 180, 225, 270, 315]

        raw_boxes = []
        raw_scores = []
        raw_scales = []
        raw_angles = []

        h_draw, w_draw = drawing_inv_clean.shape[:2]
        draw_blur = cv2.GaussianBlur(drawing_inv_clean, (3, 3), 0)

        # Grid search over scales and orientations
        for scale in scales:
            w_scaled = int(w_t * scale)
            h_scaled = int(h_t * scale)

            if w_scaled < 8 or h_scaled < 8 or w_scaled > w_draw or h_scaled > h_draw:
                continue

            template_scaled_raw = cv2.resize(template_inv, (w_scaled, h_scaled), interpolation=cv2.INTER_AREA)
            _, template_scaled = cv2.threshold(template_scaled_raw, 127, 255, cv2.THRESH_BINARY)

            # Precompile rotated templates for this scale
            rotated_templates = {}
            for angle in angles:
                template_rotated_raw = rotate_image(template_scaled, angle)
                _, template_rotated = cv2.threshold(template_rotated_raw, 127, 255, cv2.THRESH_BINARY)
                rotated_templates[angle] = template_rotated

            for angle in angles:
                template_rotated = rotated_templates[angle]
                h_temp, w_temp = template_rotated.shape[:2]
                if w_temp > w_draw or h_temp > h_draw:
                    continue

                temp_blur = cv2.GaussianBlur(template_rotated, (3, 3), 0)
                
                # Coarse Search (matchTemplate)
                res = cv2.matchTemplate(draw_blur, temp_blur, cv2.TM_CCOEFF_NORMED)
                loc = np.where(res >= self.tm_thresh)

                for y, x in zip(*loc):
                    d_crop = drawing_inv_clean[y:y+h_temp, x:x+w_temp]
                    d_crop_dil = drawing_dil[y:y+h_temp, x:x+w_temp]
                    
                    if d_crop.shape[0] != h_temp or d_crop.shape[1] != w_temp:
                        continue

                    sum_t = np.sum(template_rotated > 0)
                    if sum_t == 0:
                        continue
                        
                    # Calculate Recall metrics
                    rec_nodil = np.sum((template_rotated > 0) & (d_crop > 0)) / sum_t
                    rec_dil = np.sum((template_rotated > 0) & (d_crop_dil > 0)) / sum_t
                    
                    # Hard recall thresholds
                    if rec_dil >= self.recall_thresh and rec_nodil >= self.rec_nodil_thresh:
                        # Asymmetry Filter: compute correlation across all rotations to identify and reject symmetric connection dots
                        if self.asymmetry_thresh > 0:
                            angle_scores = []
                            for a in angles:
                                t_rot = rotated_templates[a]
                                ht_rot, wt_rot = t_rot.shape[:2]
                                
                                x_start = max(0, x - 2)
                                y_start = max(0, y - 2)
                                x_end = min(w_draw, x + wt_rot + 2)
                                y_end = min(h_draw, y + ht_rot + 2)
                                
                                d_search = draw_blur[y_start:y_end, x_start:x_end]
                                if d_search.shape[0] < ht_rot or d_search.shape[1] < wt_rot:
                                    angle_scores.append(0.0)
                                    continue
                                    
                                t_blur_rot = cv2.GaussianBlur(t_rot, (3, 3), 0)
                                r_rot = cv2.matchTemplate(d_search, t_blur_rot, cv2.TM_CCOEFF_NORMED)
                                angle_scores.append(np.max(r_rot))
                                
                            max_s = max(angle_scores)
                            min_s = min(angle_scores)
                            asymmetry = max_s - min_s
                            
                            if asymmetry < self.asymmetry_thresh:
                                continue # Reject symmetric shape (connection dot)

                        # Store detections
                        combined_score = float(res[y, x]) * 0.5 + rec_dil * 0.5
                        box = [x, y, x + w_temp, y + h_temp]
                        raw_boxes.append(box)
                        raw_scores.append(combined_score)
                        raw_scales.append(scale)
                        raw_angles.append(angle)

        # Non-Maximum Suppression to group overlapping bounding boxes
        keep_indices = non_max_suppression(raw_boxes, raw_scores, self.iou_threshold)
        detections = []
        for idx in keep_indices:
            detections.append({
                'box': raw_boxes[idx],
                'score': raw_scores[idx],
                'scale': raw_scales[idx],
                'angle': raw_angles[idx]
            })
            
        return detections
