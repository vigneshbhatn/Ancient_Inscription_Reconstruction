"""
OCR module for Kannada inscription recognition.
Uses EasyOCR with placeholder for fine-tuned model.
"""

import easyocr
import numpy as np
import cv2


# Global reader instance (initialized once for efficiency)
_reader = None


def get_reader(languages=['kn'], gpu=True):
    """
    Get or create EasyOCR reader instance.
    
    Args:
        languages: List of language codes ('kn' for Kannada)
        gpu: Whether to use GPU acceleration
    
    Returns:
        EasyOCR Reader instance
    """
    global _reader
    
    if _reader is None:
        _reader = easyocr.Reader(languages, gpu=gpu)
    
    return _reader


def run_easyocr(image, bounding_boxes=None, detail=1):
    """
    Run EasyOCR on the image or specific regions.
    
    Args:
        image: Input image (grayscale or BGR)
        bounding_boxes: Optional list of bounding boxes [(x, y, w, h), ...]
                       If provided, OCR will run on each region separately
        detail: Level of detail in results (0, 1, or 2)
                0: Only text
                1: Text with confidence
                2: Text, confidence, and coordinates
    
    Returns:
        List of dictionaries with OCR results
        [{'text': str, 'confidence': float, 'bbox': tuple}, ...]
    """
    reader = get_reader()
    
    # Convert to RGB if needed (EasyOCR expects RGB)
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    results = []
    
    if bounding_boxes is None:
        # Run OCR on entire image
        ocr_results = reader.readtext(image_rgb, detail=1)
        
        for detection in ocr_results:
            bbox, text, confidence = detection
            results.append({
                'text': text,
                'confidence': confidence,
                'bbox': bbox
            })
    else:
        # Run OCR on each region
        for i, (x, y, w, h) in enumerate(bounding_boxes):
            # Extract region
            region = image_rgb[y:y+h, x:x+w]
            
            # Skip very small regions
            if region.size == 0 or w < 10 or h < 10:
                continue
            
            # Run OCR on region
            ocr_results = reader.readtext(region, detail=1)
            
            if ocr_results:
                # Take the best result
                best_result = max(ocr_results, key=lambda x: x[2])  # Max confidence
                bbox, text, confidence = best_result
                
                results.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': (x, y, w, h),
                    'region_id': i
                })
            else:
                # No text detected in region
                results.append({
                    'text': '',
                    'confidence': 0.0,
                    'bbox': (x, y, w, h),
                    'region_id': i
                })
    
    return results


def run_custom_model(image, bounding_boxes=None, model_path=None):
    """
    Placeholder for your fine-tuned model.
    Replace this function with your actual model implementation.
    
    Args:
        image: Input image
        bounding_boxes: Optional list of bounding boxes
        model_path: Path to your fine-tuned model weights
    
    Returns:
        List of dictionaries with OCR results
    """
    # TODO: Implement your fine-tuned model here
    # Example structure:
    # 1. Load your model
    # 2. Preprocess image/regions
    # 3. Run inference
    # 4. Post-process results
    # 5. Return in same format as run_easyocr
    
    print("Custom model not implemented yet. Using EasyOCR as fallback.")
    return run_easyocr(image, bounding_boxes)


def post_process_text(text):
    """
    Post-process OCR text to fix common errors.
    
    Args:
        text: Raw OCR text
    
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    cleaned = ' '.join(text.split())
    
    # Add custom corrections for common OCR errors in Kannada
    # Example replacements (customize based on your data)
    replacements = {
        # Add common OCR mistakes here
        # 'wrong': 'correct',
    }
    
    for wrong, correct in replacements.items():
        cleaned = cleaned.replace(wrong, correct)
    
    return cleaned


def batch_ocr(images, bounding_boxes_list=None):
    """
    Run OCR on multiple images in batch.
    
    Args:
        images: List of images
        bounding_boxes_list: Optional list of bounding box lists
    
    Returns:
        List of OCR results for each image
    """
    all_results = []
    
    for i, image in enumerate(images):
        if bounding_boxes_list is not None:
            boxes = bounding_boxes_list[i]
        else:
            boxes = None
        
        results = run_easyocr(image, boxes)
        all_results.append(results)
    
    return all_results


def visualize_ocr_results(image, results):
    """
    Visualize OCR results on the image.
    
    Args:
        image: Input image
        results: OCR results from run_easyocr
    
    Returns:
        Image with OCR results visualized
    """
    # Convert to BGR if grayscale
    if len(image.shape) == 2:
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        output = image.copy()
    
    for result in results:
        text = result['text']
        confidence = result['confidence']
        bbox = result['bbox']
        
        if isinstance(bbox, tuple) and len(bbox) == 4:
            # Simple bbox format (x, y, w, h)
            x, y, w, h = bbox
            points = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
        else:
            # EasyOCR format (list of points)
            points = [(int(p[0]), int(p[1])) for p in bbox]
        
        # Draw polygon
        pts = np.array(points, dtype=np.int32)
        cv2.polylines(output, [pts], True, (0, 255, 0), 2)
        
        # Draw text and confidence
        label = f"{text} ({confidence:.2f})"
        cv2.putText(
            output,
            label,
            (points[0][0], points[0][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )
    
    return output


def filter_low_confidence(results, threshold=0.5):
    """
    Filter OCR results by confidence threshold.
    
    Args:
        results: OCR results
        threshold: Minimum confidence score
    
    Returns:
        Filtered results
    """
    return [r for r in results if r['confidence'] >= threshold]