import cv2
import numpy as np

def crop_margins(img, padding=2, thresh=240):
    """
    Crops white margins around a black-on-white symbol.
    Inverts the image to find the bounding box of foreground (black) pixels.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
        
    # Invert so background is black and drawing lines are white
    _, thresh_img = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    
    # Find non-zero coordinates
    coords = cv2.findNonZero(thresh_img)
    if coords is None:
        return img # Return original if blank
        
    x, y, w, h = cv2.boundingRect(coords)
    
    # Add a tiny padding if possible
    h_img, w_img = img.shape[:2]
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(w_img, x + w + padding)
    y_end = min(h_img, y + h + padding)
    
    return img[y_start:y_end, x_start:x_end]

def rotate_image(image, angle):
    """
    Rotates an image by a given angle (in degrees) without cropping the corners.
    Fills the background with black (0).
    """
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # Get rotation matrix (invoking cv2.getRotationMatrix2D)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    
    # Calculate new width and height to prevent cropping
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # Adjust the translation component of the matrix
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # Perform the rotation
    return cv2.warpAffine(image, M, (nW, nH), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def non_max_suppression(boxes, scores, iou_threshold=0.3):
    """
    Applies Non-Maximum Suppression (NMS) to eliminate overlapping redundant bounding boxes.
    Args:
        boxes: List or array of [xmin, ymin, xmax, ymax]
        scores: List or array of scores for each box
        iou_threshold: Intersection over Union threshold for overlapping boxes
    Returns:
        indices: Indices of the boxes that are kept
    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes, dtype=float)
    scores = np.array(scores, dtype=float)

    # Coordinates
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Calculate areas
    areas = (x2 - x1) * (y2 - y1)
    
    # Sort indices by score descending
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        # Get intersection coordinates
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # Intersection width and height
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        # Calculate IoU
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # Keep boxes with IoU less than threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

def visualize_results(drawing, detections, border_color=(0, 230, 0), thickness=2):
    """
    Draws bounding boxes and confidence scores onto the drawing image.
    detections: list of dicts with keys: 'box' (xmin, ymin, xmax, ymax), 'score', 'angle', 'scale'
    """
    output = drawing.copy()
    
    # Draw translucent fill overlay first
    overlay = output.copy()
    for det in detections:
        x1, y1, x2, y2 = det['box']
        cv2.rectangle(overlay, (x1, y1), (x2, y2), border_color, -1)
    
    # Blend overlay with original for a sleek glassmorphic effect
    alpha = 0.15
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    
    # Draw borders and labels
    for det in detections:
        x1, y1, x2, y2 = det['box']
        score = det['score']
        angle = det['angle']
        
        # Bounding box border
        cv2.rectangle(output, (x1, y1), (x2, y2), border_color, thickness)
        
        # Sleek text label
        label = f"{score:.2f} ({angle}deg)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thick = 1
        
        # Draw background label box
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, font_thick)
        y_text = max(y1, label_h + 5)
        cv2.rectangle(output, (x1, y_text - label_h - 4), (x1 + label_w + 4, y_text + baseline - 2), border_color, -1)
        
        # Write text in contrast color (white or black depending on border_color brightness)
        # Assuming border_color is bright green, black text (0,0,0) is very readable
        cv2.putText(output, label, (x1 + 2, y_text - 2), font, font_scale, (0, 0, 0), font_thick, cv2.LINE_AA)
        
    return output
