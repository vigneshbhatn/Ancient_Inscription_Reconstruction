import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# 1. SETUP
# Replace this with your actual Groq API Key
API_KEY = os.getenv('GROQ_API_KEY', '')
MODEL = "llama-3.3-70b-versatile"

def run_test():
    print(f"Testing model: {MODEL}...")

    # 2. SAMPLE DATA
    # This is a simulated 'damaged' OCR text (9th-15th century style)
    # Original meaning: "Svasti Shri Jayabhyudaya Shaka Varsha..." (Prosperity! In the victorious Saka year...)
    # Corrupted version (dots represent missing/unclear chars):
    sample_ocr = "ಸ್ವ..ತಿಶ್ರೀ ಜಯಾ..ದ ತಕವ.. ೧೪..೩೧ ಪ್ರ..ವ ಸಂವ..ರ"

    print("\n--- Input (Corrupted OCR) ---")
    print(sample_ocr)

    # 3. CALL GROQ API
    try:
        client = Groq(api_key=API_KEY)
        
        prompt = f"""
        You are an expert in ancient Kannada inscriptions (9th-15th century).
        Reconstruct this corrupted text. Output ONLY the corrected Kannada string.
        
        Corrupted Text: {sample_ocr}
        """

        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL,
            temperature=0.1,
        )

        # 4. SHOW RESULT
        result = completion.choices[0].message.content
        print("\n--- Output (Reconstructed) ---")
        print(result)
        print("\nTest Successful! ✅")

    except Exception as e:
        print(f"\nTest Failed ❌: {e}")

if __name__ == "__main__":
    # Ensure library is installed: pip install groq
    run_test()