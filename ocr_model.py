import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import json
import os

# ---------------------------------------------------------
# 1. MODEL ARCHITECTURE
# Must match the class used in your training notebook exactly
# ---------------------------------------------------------
class KannadaOCRModel(nn.Module):
    def __init__(self, num_classes):
        super(KannadaOCRModel, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ---------------------------------------------------------
# 2. SETUP & HELPERS
# ---------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Caching variables so we don't reload model on every call
_CACHED_MODEL = None
_IDX_TO_CHAR = None

def load_resources(model_dir="models"):
    """
    Loads the trained model weights and character mapping.
    """
    global _CACHED_MODEL, _IDX_TO_CHAR
    
    if _CACHED_MODEL is not None and _IDX_TO_CHAR is not None:
        return _CACHED_MODEL, _IDX_TO_CHAR

    # Paths
    model_path = os.path.join(model_dir, "best_kannada_ocr_model.pth")
    mapping_path = os.path.join(model_dir, "char_mappings.json")

    # 1. Load Mappings
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Mapping file not found at {mapping_path}. Export it from your notebook!")
    
    with open(mapping_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # JSON converts integer keys to strings, so we convert them back
        _IDX_TO_CHAR = {int(k): v for k, v in data['idx_to_char'].items()}
        
    num_classes = len(_IDX_TO_CHAR)
    
    # 2. Load Model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")
        
    model = KannadaOCRModel(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    # Handle checkpoint dictionary vs direct model save
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(DEVICE)
    model.eval()
    
    _CACHED_MODEL = model
    print(f"Loaded model with {num_classes} classes on {DEVICE}")
    
    return _CACHED_MODEL, _IDX_TO_CHAR

def preprocess_crop(img_crop):
    """
    Transforms a cv2 image crop to the format expected by the model.
    Matches the validation transforms from your notebook.
    """
    # Convert BGR (OpenCV) to RGB
    img_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_img = Image.fromarray(img_rgb)
    
    # Define Transforms (Must match your training 'val_transform')
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Add batch dimension (1, C, H, W) and move to device
    return transform(pil_img).unsqueeze(0).to(DEVICE)

# ---------------------------------------------------------
# 3. MAIN INFERENCE FUNCTION
# ---------------------------------------------------------
def run_custom_model(image, bounding_boxes=None, model_path=None):
    """
    Main entry point called by app.py.
    
    Args:
        image: Full input image (OpenCV/numpy array)
        bounding_boxes: List of (x, y, w, h) tuples
        model_path: Ignored (uses load_resources defaults), kept for compatibility
    
    Returns:
        List of dicts: [{'text': char, 'confidence': float, 'bbox': tuple}, ...]
    """
    # Load model and mappings
    model, idx_to_char = load_resources()
    
    results = []
    
    if not bounding_boxes:
        return results

    # Loop through each bounding box
    for i, (x, y, w, h) in enumerate(bounding_boxes):
        # Skip tiny boxes (noise)
        if w < 5 or h < 5:
            continue
            
        # 1. Crop the character region
        # Handle edge cases to ensure we don't crop outside image
        y1, y2 = max(0, y), min(image.shape[0], y+h)
        x1, x2 = max(0, x), min(image.shape[1], x+w)
        
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            continue

        # 2. Preprocess
        input_tensor = preprocess_crop(roi)

        # 3. Prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            
            # Apply Softmax to get confidence percentages
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, predicted_idx = torch.max(probs, 1)
            
            class_id = predicted_idx.item()
            confidence = conf.item()
            
            # Map ID to Character
            predicted_char = idx_to_char.get(class_id, "?")

        # 4. Format Result
        results.append({
            'text': predicted_char,
            'confidence': confidence,
            'bbox': (x, y, w, h),
            'region_id': i
        })

    return results