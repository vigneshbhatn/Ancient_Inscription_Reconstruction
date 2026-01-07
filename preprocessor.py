import cv2
import numpy as np
import os

# ---------- Setup ----------
# Input file (change this to your actual image path)
input_path = "pic.jpg"

# Output folder
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# ---------- Step 1: Read Image ----------
image = cv2.imread(input_path)

# Keep full image size (not cropped)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ---------- Step 2: Noise Removal ----------
# Median blur removes salt-and-pepper noise
median = cv2.medianBlur(gray, 3)

# Bilateral filter preserves edges while smoothing texture
denoised = cv2.bilateralFilter(median, d=9, sigmaColor=75, sigmaSpace=75)

# ---------- Step 3: Contrast Enhancement ----------
# Histogram Equalization
equalized = cv2.equalizeHist(denoised)

# CLAHE (adaptive histogram equalization)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
enhanced = clahe.apply(equalized)

# ---------- Save Preprocessing Results ----------
cv2.imwrite(os.path.join(output_dir, "01_gray.jpg"), gray)
cv2.imwrite(os.path.join(output_dir, "02_denoised.jpg"), denoised)
cv2.imwrite(os.path.join(output_dir, "03_equalized.jpg"), equalized)
cv2.imwrite(os.path.join(output_dir, "04_enhanced.jpg"), enhanced)

# ---------- Step 4: Thresholding for Segmentation ----------
_, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Invert if needed (white text on black)
binary = cv2.bitwise_not(binary)
cv2.imwrite(os.path.join(output_dir, "05_binary.jpg"), binary)

# ---------- Step 5: Find Contours (Character Segmentation) ----------
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

char_regions = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    # Filter by reasonable size (tune for your images!)
    if 15 < w < 150 and 15 < h < 150:
        char_regions.append((x, y, w, h))

# Sort top-to-bottom, then left-to-right
char_regions = sorted(char_regions, key=lambda b: (b[1]//50, b[0]))

# ---------- Step 6: Draw Boxes & Save Characters ----------
segmented_preview = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

for i, (x, y, w, h) in enumerate(char_regions):
    cv2.rectangle(segmented_preview, (x, y), (x+w, y+h), (0,255,0), 2)
    char_img = binary[y:y+h, x:x+w]
    cv2.imwrite(os.path.join(output_dir, f"char_{i:03d}.png"), char_img)

cv2.imwrite(os.path.join(output_dir, "06_segmented_preview.jpg"), segmented_preview)

print(f"✅ Processing complete! Check the '{output_dir}' folder for results.")
