"""
Segmentation module for Kannada inscription characters.
Identifies and segments individual characters or character groups.
"""

import cv2
import numpy as np


def segment_characters(image, min_area=100, padding=5, max_area_ratio=0.8):
    """
    Segment characters from the preprocessed image.
    
    Args:
        image: Preprocessed image (grayscale or binary)
        min_area: Minimum area for a valid character region
        padding: Padding around bounding boxes (pixels)
        max_area_ratio: Maximum area as ratio of image size (to filter out full-image detections)
    
    Returns:
        List of bounding boxes [(x, y, w, h), ...]
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Check if image is mostly white (needs inversion for segmentation)
    mean_val = np.mean(gray)
    if mean_val > 127:
        # Image is bright, invert for processing
        gray = cv2.bitwise_not(gray)
    
    # Apply binary threshold with Otsu for better automatic thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply morphological opening to remove small noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    # Find connected components instead of contours for better accuracy
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
    
    # Calculate image area for filtering
    img_area = gray.shape[0] * gray.shape[1]
    max_area = img_area * max_area_ratio
    
    # Extract bounding boxes from connected components
    bounding_boxes = []
    
    for i in range(1, num_labels):  # Skip background (label 0)
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Filter based on area (minimum and maximum)
        if min_area <= area <= max_area:
            # Also filter by aspect ratio (characters shouldn't be too elongated)
            aspect_ratio = w / h if h > 0 else 0
            
            # Accept reasonable aspect ratios
            if 0.1 < aspect_ratio < 10:
                # Add padding
                x_pad = max(0, x - padding)
                y_pad = max(0, y - padding)
                w_pad = min(gray.shape[1] - x_pad, w + 2 * padding)
                h_pad = min(gray.shape[0] - y_pad, h + 2 * padding)
                
                bounding_boxes.append((x_pad, y_pad, w_pad, h_pad))
    
    # If no boxes found, try with more aggressive preprocessing
    if len(bounding_boxes) == 0:
        # Try erosion to separate touching characters
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        eroded = cv2.erode(opened, kernel_erode, iterations=1)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded, connectivity=8)
        
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            
            if min_area <= area <= max_area:
                aspect_ratio = w / h if h > 0 else 0
                if 0.1 < aspect_ratio < 10:
                    x_pad = max(0, x - padding)
                    y_pad = max(0, y - padding)
                    w_pad = min(gray.shape[1] - x_pad, w + 2 * padding)
                    h_pad = min(gray.shape[0] - y_pad, h + 2 * padding)
                    bounding_boxes.append((x_pad, y_pad, w_pad, h_pad))
    
    # Sort bounding boxes from left to right, top to bottom
    bounding_boxes = sort_boxes(bounding_boxes)
    
    return bounding_boxes


def sort_boxes(boxes, line_threshold=20):
    """
    Sort bounding boxes from left to right, top to bottom (reading order).
    This ensures proper sentence formation.
    
    Args:
        boxes: List of bounding boxes [(x, y, w, h), ...]
        line_threshold: Threshold to determine if boxes are on the same line
    
    Returns:
        Sorted list of bounding boxes (left to right, top to bottom)
    """
    if not boxes:
        return boxes
    
    # Sort by y-coordinate first (top to bottom)
    boxes_sorted = sorted(boxes, key=lambda b: b[1])
    
    # Group boxes into lines based on vertical position
    lines = []
    current_line = [boxes_sorted[0]]
    
    for box in boxes_sorted[1:]:
        # Check if box is on the same line (similar y-coordinate)
        if abs(box[1] - current_line[0][1]) <= line_threshold:
            current_line.append(box)
        else:
            # Sort current line by x-coordinate (left to right)
            current_line.sort(key=lambda b: b[0])
            lines.append(current_line)
            current_line = [box]
    
    # Add last line and sort it
    current_line.sort(key=lambda b: b[0])
    lines.append(current_line)
    
    # Flatten lines to get final sorted list (left to right, top to bottom)
    sorted_boxes = [box for line in lines for box in line]
    
    return sorted_boxes


def draw_bounding_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on the image.
    
    Args:
        image: Input image
        boxes: List of bounding boxes [(x, y, w, h), ...]
        color: Box color (BGR)
        thickness: Line thickness
    
    Returns:
        Image with bounding boxes drawn
    """
    # Convert to BGR if grayscale
    if len(image.shape) == 2:
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        output = image.copy()
    
    # Draw each box
    for i, (x, y, w, h) in enumerate(boxes):
        # Draw rectangle
        cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)
        
        # Add label
        label = str(i + 1)
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y - 5, label_size[1])
        
        cv2.putText(
            output,
            label,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )
    
    return output


def extract_character_images(image, boxes):
    """
    Extract individual character images based on bounding boxes.
    
    Args:
        image: Input image
        boxes: List of bounding boxes [(x, y, w, h), ...]
    
    Returns:
        List of character images
    """
    character_images = []
    
    for x, y, w, h in boxes:
        char_img = image[y:y+h, x:x+w]
        character_images.append(char_img)
    
    return character_images


def refine_segmentation(image, boxes, merge_threshold=10):
    """
    Refine segmentation by merging nearby boxes that might be parts of the same character.
    
    Args:
        image: Input image
        boxes: List of bounding boxes [(x, y, w, h), ...]
        merge_threshold: Distance threshold for merging boxes
    
    Returns:
        Refined list of bounding boxes
    """
    if len(boxes) <= 1:
        return boxes
    
    refined_boxes = []
    merged = [False] * len(boxes)
    
    for i in range(len(boxes)):
        if merged[i]:
            continue
        
        x1, y1, w1, h1 = boxes[i]
        
        # Try to merge with subsequent boxes
        for j in range(i + 1, len(boxes)):
            if merged[j]:
                continue
            
            x2, y2, w2, h2 = boxes[j]
            
            # Check if boxes are close enough
            horizontal_distance = abs((x1 + w1/2) - (x2 + w2/2)) - (w1 + w2)/2
            vertical_distance = abs((y1 + h1/2) - (y2 + h2/2)) - (h1 + h2)/2
            
            if horizontal_distance < merge_threshold and vertical_distance < merge_threshold:
                # Merge boxes
                x_min = min(x1, x2)
                y_min = min(y1, y2)
                x_max = max(x1 + w1, x2 + w2)
                y_max = max(y1 + h1, y2 + h2)
                
                x1, y1 = x_min, y_min
                w1, h1 = x_max - x_min, y_max - y_min
                
                merged[j] = True
        
        refined_boxes.append((x1, y1, w1, h1))
    
    return refined_boxes


def segment_lines(image, threshold=50):
    """
    Segment text into lines (horizontal segmentation).
    
    Args:
        image: Preprocessed image
        threshold: Threshold for binarization
    
    Returns:
        List of line bounding boxes [(x, y, w, h), ...]
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply binary threshold
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Calculate horizontal projection
    horizontal_projection = np.sum(binary, axis=1)
    
    # Find lines based on projection
    in_line = False
    line_start = 0
    lines = []
    
    for i, value in enumerate(horizontal_projection):
        if value > 0 and not in_line:
            line_start = i
            in_line = True
        elif value == 0 and in_line:
            lines.append((0, line_start, gray.shape[1], i - line_start))
            in_line = False
    
    # Add last line if needed
    if in_line:
        lines.append((0, line_start, gray.shape[1], len(horizontal_projection) - line_start))
    
    return lines