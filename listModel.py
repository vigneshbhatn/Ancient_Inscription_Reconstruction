import google.generativeai as genai
import os

# Configure API key
API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
if API_KEY == "YOUR_API_KEY_HERE":
    print("Warning: API_KEY not set. Please set the 'GEMINI_API_KEY' environment variable.")
    print("Some models may not be accessible without a valid API key.\n")

genai.configure(api_key=API_KEY)

# Free tier models typically include: gemini-1.5-flash, gemini-1.5-pro, gemini-pro
# Exclude premium/ultra models
FREE_TIER_KEYWORDS = ["flash", "pro", "gemini-1.5", "gemini-pro"]
EXCLUDE_KEYWORDS = ["ultra", "2.0"]  # Exclude premium models

def is_free_tier(model_name: str) -> bool:
    """Check if model is likely in free tier"""
    model_lower = model_name.lower()
    # Exclude premium models
    if any(keyword in model_lower for keyword in EXCLUDE_KEYWORDS):
        return False
    # Include common free tier models
    return any(keyword in model_lower for keyword in FREE_TIER_KEYWORDS)

print("=" * 60)
print("FREE TIER MODELS - generateContent Support")
print("=" * 60)
generate_content_models = []
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        if is_free_tier(m.name):
            generate_content_models.append(m.name)
            print(f"  {m.name}")

if not generate_content_models:
    print("  No free tier models found with generateContent support.")

print("\n" + "=" * 60)
print("FREE TIER MODELS - embedContent Support")
print("=" * 60)
embed_content_models = []
for m in genai.list_models():
    if "embedContent" in m.supported_generation_methods:
        if is_free_tier(m.name):
            embed_content_models.append(m.name)
            print(f"  {m.name}")

if not embed_content_models:
    print("  No free tier models found with embedContent support.")

print("\n" + "=" * 60)
print(f"Summary: {len(generate_content_models)} generateContent models, {len(embed_content_models)} embedContent models")
print("=" * 60)