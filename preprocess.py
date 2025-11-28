"""
Preprocessing module for Kannada inscription images.
Based on the v8 preprocessing pipeline with real-time parameter adjustment.
"""

import cv2
import numpy as np


def remove_dots(image, min_size=20, max_aspect_ratio=2.5, circularity_area_threshold=50):
    """
    Remove dots from the inscription image (works on binary images).
    
    Args:
        image: Input binary image as numpy array
        min_size: Minimum area to keep (dots are typically smaller)
        max_aspect_ratio: Max width/height ratio for dots
        circularity_area_threshold: Max area for highly circular components
    
    Returns:
        Image with dots removed
    """
    # Work directly on binary image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Ensure binary image (white text on black background)
    # Check if image is already binary
    unique_vals = np.unique(gray)
    if len(unique_vals) > 2:
        # Not binary, apply threshold
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    else:
        binary = gray.copy()
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Create output image
    output = np.zeros_like(binary)
    
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Calculate aspect ratio
        if height > 0:
            aspect_ratio = width / height
        else:
            aspect_ratio = 0
        
        # Calculate circularity (how close to a circle)
        perimeter = cv2.arcLength(np.argwhere(labels == i), True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        else:
            circularity = 0
        
        # Keep component if:
        # 1. Area is larger than min_size
        # 2. OR it's not too circular and not too small
        # 3. OR it's elongated (not a dot-like shape)
        
        keep = False
        
        # Large enough components are always kept
        if area >= circularity_area_threshold:
            keep = True
        # Medium sized components - check shape
        elif area >= min_size:
            # Keep if it's elongated (not circular/square-like)
            if aspect_ratio > max_aspect_ratio or aspect_ratio < 1/max_aspect_ratio:
                keep = True
            # Keep if it's not very circular
            elif circularity < 0.7:
                keep = True
        
        if keep:
            output[labels == i] = 255
    
    return output


def preprocess_image_v8(image_cv, apply_clahe, clahe_clip, clahe_grid,
                        invert_grayscale, denoise_method, blur_ksize,
                        nl_h, nl_template, nl_search,
                        adaptive_method_option, adaptive_block, adaptive_c,
                        opening_iter, erosion_iter, dilation_iter,
                        apply_dot_removal, dot_min_size, dot_max_aspect_ratio, 
                        dot_circularity_threshold):
    """
    Complete preprocessing pipeline for inscription images.
    
    Args:
        image_cv: Input image in BGR format (from cv2)
        apply_clahe: Whether to apply CLAHE contrast enhancement
        clahe_clip: CLAHE clip limit
        clahe_grid: CLAHE grid size
        invert_grayscale: Whether to invert grayscale (for light text on dark)
        denoise_method: "Gaussian Blur", "Median Blur", or "Non-Local Means"
        blur_ksize: Kernel size for blur methods (must be odd)
        nl_h: Filter strength for Non-Local Means
        nl_template: Template window size for NLM
        nl_search: Search window size for NLM
        adaptive_method_option: "Gaussian" or "Mean" for adaptive threshold
        adaptive_block: Block size for adaptive threshold (must be odd)
        adaptive_c: Constant subtracted in adaptive threshold
        opening_iter: Number of morphological opening iterations
        erosion_iter: Number of erosion iterations
        dilation_iter: Number of dilation iterations
        apply_dot_removal: Whether to apply dot removal (LAST step)
        dot_min_size: Minimum dot size for removal
        dot_max_aspect_ratio: Max aspect ratio for dots
        dot_circularity_threshold: Circularity threshold for dots
    
    Returns:
        Tuple of (display_gray, display_clahe, denoised, binarized, post_morph, final_output)
    """
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    
    # Optional Invert
    gray_proc = cv2.bitwise_not(gray) if invert_grayscale else gray
    
    # Optional CLAHE Step (before denoising)
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
        gray_proc = clahe.apply(gray_proc)
    
    # 2. Denoising Step (Choice of 3 methods)
    denoised = None
    if denoise_method == "Median Blur":
        denoised = cv2.medianBlur(gray_proc, blur_ksize)
    elif denoise_method == "Gaussian Blur":
        denoised = cv2.GaussianBlur(gray_proc, (blur_ksize, blur_ksize), 0)
    elif denoise_method == "Non-Local Means":
        # This is slow, so we show a spinner in the main app
        denoised = cv2.fastNlMeansDenoising(
            gray_proc, 
            None, 
            h=nl_h, 
            templateWindowSize=nl_template,
            searchWindowSize=nl_search
        )
    
    # 3. Binarization (Adaptive Threshold)
    adaptive_method_cv = (cv2.ADAPTIVE_THRESH_GAUSSIAN_C if adaptive_method_option == "Gaussian" 
                         else cv2.ADAPTIVE_THRESH_MEAN_C)
    
    binarized = cv2.adaptiveThreshold(
        denoised, 
        255, 
        adaptive_method_cv,
        cv2.THRESH_BINARY_INV, 
        adaptive_block, 
        adaptive_c
    )
    
    # 4. Post-Processing (Morphological Operations)
    post_morph = binarized
    kernel = np.ones((3, 3), np.uint8)
    
    if opening_iter > 0:
        post_morph = cv2.morphologyEx(binarized, cv2.MORPH_OPEN, kernel, iterations=opening_iter)
    
    if erosion_iter > 0:
        post_morph = cv2.erode(post_morph, kernel, iterations=erosion_iter)
    
    if dilation_iter > 0:
        post_morph = cv2.dilate(post_morph, kernel, iterations=dilation_iter)
    
    # 5. Dot Removal (LAST STEP - after all other processing)
    final_output = post_morph
    if apply_dot_removal:
        final_output = remove_dots(post_morph, dot_min_size, dot_max_aspect_ratio, dot_circularity_threshold)
    
    # Prepare display images
    display_gray = gray if not invert_grayscale else cv2.bitwise_not(gray)
    display_clahe = None
    if apply_clahe:
        clahe_obj = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
        display_clahe = clahe_obj.apply(display_gray)
    
    return display_gray, display_clahe, denoised, binarized, post_morph, final_output


def preprocess_simple(image, method="adaptive"):
    """
    Simple preprocessing for quick results.
    
    Args:
        image: Input image
        method: "adaptive", "otsu", or "simple"
    
    Returns:
        Binary image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Binarization
    if method == "adaptive":
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
    elif method == "otsu":
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:  # simple
        _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
    
    return binary


def enhance_contrast(image, method="clahe", clip_limit=2.0, grid_size=8):
    """
    Enhance contrast of the image.
    
    Args:
        image: Input grayscale image
        method: "clahe" or "histogram_eq"
        clip_limit: CLAHE clip limit
        grid_size: CLAHE grid size
    
    Returns:
        Contrast-enhanced image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    if method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
        enhanced = clahe.apply(gray)
    else:  # histogram equalization
        enhanced = cv2.equalizeHist(gray)
    
    return enhanced


def denoise_image(image, method="gaussian", **kwargs):
    """
    Apply denoising to the image.
    
    Args:
        image: Input image
        method: "gaussian", "median", "bilateral", or "nlm"
        **kwargs: Method-specific parameters
    
    Returns:
        Denoised image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    if method == "gaussian":
        ksize = kwargs.get('ksize', 5)
        denoised = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    elif method == "median":
        ksize = kwargs.get('ksize', 5)
        denoised = cv2.medianBlur(gray, ksize)
    elif method == "bilateral":
        d = kwargs.get('d', 9)
        sigma_color = kwargs.get('sigma_color', 75)
        sigma_space = kwargs.get('sigma_space', 75)
        denoised = cv2.bilateralFilter(gray, d, sigma_color, sigma_space)
    elif method == "nlm":
        h = kwargs.get('h', 10)
        template = kwargs.get('template', 7)
        search = kwargs.get('search', 21)
        denoised = cv2.fastNlMeansDenoising(gray, None, h, template, search)
    else:
        denoised = gray
    
    return denoised


def binarize_image(image, method="adaptive", **kwargs):
    """
    Binarize the image.
    
    Args:
        image: Input grayscale image
        method: "adaptive", "otsu", "simple"
        **kwargs: Method-specific parameters
    
    Returns:
        Binary image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    if method == "adaptive":
        adaptive_method = kwargs.get('adaptive_method', cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        block_size = kwargs.get('block_size', 11)
        c = kwargs.get('c', 2)
        binary = cv2.adaptiveThreshold(
            gray, 255, adaptive_method, cv2.THRESH_BINARY_INV, block_size, c
        )
    elif method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:  # simple
        threshold_val = kwargs.get('threshold', 127)
        _, binary = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY_INV)
    
    return binary


def morphological_operations(image, operation="open", kernel_size=3, iterations=1):
    """
    Apply morphological operations.
    
    Args:
        image: Input binary image
        operation: "open", "close", "erode", "dilate"
        kernel_size: Size of structuring element
        iterations: Number of iterations
    
    Returns:
        Processed image
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    if operation == "open":
        result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif operation == "close":
        result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif operation == "erode":
        result = cv2.erode(image, kernel, iterations=iterations)
    elif operation == "dilate":
        result = cv2.dilate(image, kernel, iterations=iterations)
    else:
        result = image
    
    return result