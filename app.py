from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import cv2
import numpy as np
import uuid
import os

app = FastAPI()

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_image(image_path, output_path):
    # Read image
    img = cv2.imread(image_path)

    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Noise removal (median + bilateral)
    median = cv2.medianBlur(gray, 3)
    denoised = cv2.bilateralFilter(median, 9, 75, 75)

    # Step 3: Contrast Enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)

    # Save processed image
    cv2.imwrite(output_path, enhanced)
    return output_path

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # Save uploaded file
    file_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}.png")
    output_path = os.path.join(OUTPUT_DIR, f"{file_id}_processed.png")

    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Process image
    processed_path = preprocess_image(input_path, output_path)

    return FileResponse(processed_path, media_type="image/png")
