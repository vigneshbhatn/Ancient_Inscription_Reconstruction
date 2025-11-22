import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
from dotenv import load_dotenv
from preprocess import preprocess_image_v8
from segment import segment_characters, draw_bounding_boxes
from ocr_model import run_easyocr
from gemini_reconstruct import reconstruct_with_gemini

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')

# Page config
st.set_page_config(
    page_title="Kannada Inscription Reconstruction",
    layout="wide"
)

# Initialize session state
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'segmented_image' not in st.session_state:
    st.session_state.segmented_image = None
if 'bounding_boxes' not in st.session_state:
    st.session_state.bounding_boxes = []
if 'ocr_results' not in st.session_state:
    st.session_state.ocr_results = []
if 'reconstructed_text' not in st.session_state:
    st.session_state.reconstructed_text = ""

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    
    # CLAHE Settings
    st.subheader("0. Contrast Enhancement (Optional)")
    apply_clahe = st.checkbox("Apply CLAHE First", value=False, 
                             help="Enhances local contrast BEFORE denoising")
    clahe_clip = st.slider("CLAHE Clip Limit", 1.0, 10.0, 2.0, 0.5, key="clahe_clip")
    clahe_grid = st.slider("CLAHE Grid Size", 2, 16, 8, key="clahe_grid")
    
    st.divider()
    
    # Image Type
    st.subheader("1. Image Type")
    invert_grayscale = st.checkbox("Invert Grayscale (Light text on dark)", value=False, key="invert_grayscale")
    
    st.divider()
    
    # Denoising Method
    st.subheader("2. Denoising Method")
    denoise_method = st.selectbox(
        "Select Denoising Filter",
        ["Gaussian Blur", "Median Blur", "Non-Local Means"],
        index=0,
        key="denoise_method"
    )
    
    if denoise_method in ["Gaussian Blur", "Median Blur"]:
        blur_ksize = st.slider("Blur Kernel Size", 3, 31, 15, 2,
                              help="Must be odd. Use larger value to remove texture",
                              key="blur_ksize")
        nl_h, nl_template, nl_search = 10, 7, 21
    else:  # Non-Local Means
        st.info("NLM is powerful but SLOW. Be patient after changing sliders.")
        nl_h = st.slider("Filter Strength (h)", 1.0, 30.0, 10.0, 0.5,
                        help="Higher h = more noise removal but less detail",
                        key="nl_h")
        nl_template = st.slider("Template Size", 3, 11, 7, 2, 
                               help="Must be odd",
                               key="nl_template")
        nl_search = st.slider("Search Window Size", 5, 41, 21, 2,
                             help="Must be odd. Larger is slower",
                             key="nl_search")
        blur_ksize = 5
    
    st.divider()
    
    # Binarization
    st.subheader("3. Binarization (Adaptive)")
    adaptive_method_option = st.selectbox("Adaptive Method", ["Gaussian", "Mean"], 
                                         index=0, key="adaptive_method")
    adaptive_block = st.slider("Block Size", 3, 255, 31, 2, 
                              help="Must be odd",
                              key="adaptive_block")
    adaptive_c = st.slider("Constant (C)", -30, 30, 10, key="adaptive_c")
    
    st.divider()
    
    # Post-Processing
    st.subheader("4. Post-Processing (Clean & Fix)")
    opening_iter = st.slider("Opening (Removes noise)", 0, 10, 0,
                            help="Set to 1-2 to remove speckles",
                            key="opening_iter")
    erosion_iter = st.slider("Erosion (Thinner lines)", 0, 10, 0,
                            help="Fixes bloated characters",
                            key="erosion_iter")
    dilation_iter = st.slider("Dilation (Thicker lines)", 0, 10, 0,
                             help="Fixes broken characters",
                             key="dilation_iter")
    
    st.divider()
    
    # Segmentation Parameters
    st.subheader("5. Segmentation Parameters")
    min_area = st.slider(
        "Minimum Character Area",
        min_value=50,
        max_value=5000,
        value=500,
        help="Minimum area for a valid character region",
        key="min_area"
    )
    
    max_area_ratio = st.slider(
        "Maximum Area Ratio",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Maximum area as fraction of image",
        key="max_area_ratio"
    )
    
    seg_padding = st.slider(
        "Bounding Box Padding",
        min_value=0,
        max_value=20,
        value=5,
        help="Padding around each character region",
        key="seg_padding"
    )
    
    st.divider()
    
    # API Configuration
    st.subheader("6. API Configuration")
    
    # Check if API key is in environment
    if GEMINI_API_KEY:
        st.success("Gemini API Key loaded from .env")
        gemini_api_key = GEMINI_API_KEY
    else:
        st.warning("No API key found in .env file")
        gemini_api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Enter your Google Gemini API key"
        )

# Main content area
st.title("Reconstruction of ancient kannada inscriptions")
st.write("Ancient Kannada inscription analysis using computer vision and AI")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Image Upload & Processing")
    
    # Image upload
    uploaded_file = st.file_uploader(
        "Upload Inscription Image",
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp']
    )
    
    if uploaded_file is not None:
        # Load image
        pil_image = Image.open(uploaded_file).convert('RGB')
        original_image_np = np.array(pil_image)
        original_image_cv = cv2.cvtColor(original_image_np, cv2.COLOR_RGB2BGR)
        st.session_state.original_image = original_image_cv
        
        st.success("Image uploaded successfully")
        
        # Real-time preprocessing
        st.subheader("Step 1: Preprocessing Pipeline (Real-time)")
        
        # Show spinner for NLM method
        if denoise_method == "Non-Local Means":
            with st.spinner('Applying Non-Local Means Denoising... This may take a moment...'):
                disp_gray, disp_clahe, denoised, binarized, final_output = preprocess_image_v8(
                    original_image_cv, apply_clahe, clahe_clip, clahe_grid,
                    invert_grayscale, denoise_method, blur_ksize,
                    nl_h, nl_template, nl_search,
                    adaptive_method_option, adaptive_block, adaptive_c,
                    opening_iter, erosion_iter, dilation_iter
                )
        else:
            disp_gray, disp_clahe, denoised, binarized, final_output = preprocess_image_v8(
                original_image_cv, apply_clahe, clahe_clip, clahe_grid,
                invert_grayscale, denoise_method, blur_ksize,
                nl_h, nl_template, nl_search,
                adaptive_method_option, adaptive_block, adaptive_c,
                opening_iter, erosion_iter, dilation_iter
            )
        
        st.session_state.processed_image = final_output
        
        # Display final processed image
        st.image(final_output, caption="Processed Image", width=200)
        
        # Step 2: Real-time Segmentation
        st.subheader("Step 2: Character Segmentation (Real-time)")
        
        with st.spinner("Segmenting characters..."):
            bounding_boxes = segment_characters(
                st.session_state.processed_image,
                min_area=min_area,
                padding=seg_padding,
                max_area_ratio=max_area_ratio
            )
            
            st.session_state.bounding_boxes = bounding_boxes
            
            if len(bounding_boxes) == 0:
                st.warning("No character regions found. Try adjusting: Decrease minimum character area, Increase maximum area ratio, Adjust preprocessing parameters")
                st.session_state.segmented_image = None
            else:
                # Draw bounding boxes with left-to-right numbering
                st.session_state.segmented_image = draw_bounding_boxes(
                    st.session_state.processed_image,
                    bounding_boxes
                )
                st.success(f"Found {len(bounding_boxes)} character regions (numbered left to right)")
        
        # Display segmented image
        if st.session_state.segmented_image is not None:
            st.image(
                st.session_state.segmented_image,
                caption=f"Segmented Image ({len(st.session_state.bounding_boxes)} regions, left to right)",
                width=200
            )

with col2:
    st.header("OCR & Reconstruction")
    
    if st.session_state.processed_image is not None and len(st.session_state.bounding_boxes) > 0:
        st.info(f"Image processed | Regions detected: {len(st.session_state.bounding_boxes)} (numbered left to right)")
    
    if st.session_state.segmented_image is not None and len(st.session_state.bounding_boxes) > 0:
        # Step 3: OCR
        st.subheader("Step 3: OCR Recognition")
        
        if st.button("Run EasyOCR", width='stretch'):
            with st.spinner("Running OCR..."):
                ocr_results = run_easyocr(
                    st.session_state.processed_image,
                    st.session_state.bounding_boxes
                )
                st.session_state.ocr_results = ocr_results
                st.success("OCR complete")
                st.rerun()
        
        # Display OCR results
        if st.session_state.ocr_results:
            st.write("**OCR Results:**")
            
            ocr_df_data = []
            for i, result in enumerate(st.session_state.ocr_results):
                ocr_df_data.append({
                    "Region": i + 1,
                    "Text": result['text'],
                    "Confidence": f"{result['confidence']:.2%}"
                })
            
            st.dataframe(ocr_df_data, width='stretch')
            
            # Extracted text
            extracted_text = " ".join([r['text'] for r in st.session_state.ocr_results])
            st.text_area("Extracted Text", extracted_text, height=100)
            
            # Step 4: Gemini Reconstruction
            st.subheader("Step 4: AI Reconstruction")
            
            if not gemini_api_key:
                st.warning("Please add GEMINI_API_KEY to your .env file or enter it in the sidebar")
            else:
                if st.button("Reconstruct with Gemini", width='stretch'):
                    with st.spinner("Reconstructing text with Gemini AI..."):
                        reconstructed = reconstruct_with_gemini(
                            extracted_text,
                            gemini_api_key
                        )
                        st.session_state.reconstructed_text = reconstructed
                        st.success("Reconstruction complete")
                        st.rerun()
            
            # Display reconstructed text
            if st.session_state.reconstructed_text:
                st.write("**Reconstructed Text:**")
                st.markdown(st.session_state.reconstructed_text)
                
        
        
    else:
        st.info("Please complete the preprocessing and segmentation steps first")


