import cv2
import numpy as np

def calibrate(points, ticks, perspective=False):
    """Generates the perspective transformation matrix based on ruler points."""
    x1, x2, y1, y2 = ticks
    if not (0 < x1 < x2 and 0 < y1 < y2):
        raise ValueError("Invalid ruler ticks")
    src = np.float32(points)
    hull = cv2.convexHull(src)
    if len(hull) != 4 or cv2.contourArea(hull) < 100:
        raise ValueError("Invalid calibration points")
    dst = np.float32([
        [x1, 0],
        [x2, 0],
        [0, y1],
        [0, y2]
    ])
    if perspective:
        matrix = cv2.getPerspectiveTransform(src, dst)
    else:
        design = np.column_stack((src, np.ones(4)))
        axes = np.linalg.lstsq(design, dst, rcond=None)[0]
        matrix = np.vstack((axes.T, [0, 0, 1]))
    if np.linalg.matrix_rank(matrix) != 3:
        raise ValueError("Invalid calibration matrix")
    return matrix

def refine_outline(image, outline):
    """Refines the segmentation outline using GrabCut."""
    x, y, w, h = cv2.boundingRect(np.float32(outline))
    radius = max(3, round(min(w, h) * 0.06))
    x0 = max(0, x - radius - 2)
    y0 = max(0, y - radius - 2)
    roi = image[y0:y + h + radius + 2, x0:x + w + radius + 2]
    
    mask = np.zeros(roi.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.int32(outline - [x0, y0])], 1)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    core = cv2.erode(mask, kernel)
    band = cv2.dilate(mask, kernel)
    
    if not core.any() or band.all():
        return outline
        
    labels = np.full_like(mask, cv2.GC_BGD)
    labels[band > 0] = cv2.GC_PR_BGD
    labels[mask > 0] = cv2.GC_PR_FGD
    labels[core > 0] = cv2.GC_FGD
    
    cv2.grabCut(
        roi, labels, None,
        np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64),
        3, cv2.GC_INIT_WITH_MASK
    )
    
    contours, _ = cv2.findContours(labels & 1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return outline
        
    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < 0.8 * cv2.contourArea(np.float32(outline)):
        return outline
    return best.reshape(-1, 2) + [x0, y0]

def rectangular_outline(image):
    """Fallback method to find rectangular outlines using Canny edge detection."""
    smooth = cv2.GaussianBlur(image, (5, 5), 0)
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)
    edges = cv2.Canny(lab[:, :, 0], 25, 75)
    masks = [cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))]
    
    for channel in cv2.split(lab):
        _, mask = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks.extend([mask, cv2.bitwise_not(mask)])
        
    h, w = image.shape[:2]
    best = None
    best_score = 0
    
    for mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if not 0.04 * h * w < area < 0.9 * h * w:
                continue
            quad = cv2.approxPolyDP(contour, 0.015 * cv2.arcLength(contour, True), True)
            if len(quad) != 4 or not cv2.isContourConvex(quad):
                continue
            x, y, bw, bh = cv2.boundingRect(quad)
            if min(x, y, w - x - bw, h - y - bh) < 2:
                continue
            a, b = cv2.minAreaRect(quad)[1]
            if min(a, b) < 0.18 * max(a, b):
                continue
            fill = area / (a * b)
            if 0.89 < fill < 1.08 and area * fill > best_score:
                best = quad.reshape(4, 2)
                best_score = area * fill
    return best

def measure(points, matrix):
    """Calculates final real-world dimensions using the transformation matrix."""
    contour = np.float32(points).reshape(-1, 1, 2)
    denominators = contour[:, 0] @ matrix[2, :2] + matrix[2, 2]
    if denominators.min() <= 0 <= denominators.max():
        raise ValueError("Outline crosses calibration horizon")
    
    metric = cv2.perspectiveTransform(contour, matrix)
    rect = cv2.minAreaRect(metric)
    length, width = sorted(rect[1], reverse=True)
    if width <= 0:
        raise ValueError("Invalid object dimensions")
    return length, width