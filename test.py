import os
import random
import time
import math
from typing import List, Tuple
# Import your existing module
from gemini_reconstruct import reconstruct_with_gemini

# --- CONFIGURATION ---
INPUT_FILE = "groundtruth.txt"
API_KEY = os.getenv("GEMINI_API_KEY")
DAMAGE_LEVEL = 0.15  # 15% of characters will be damaged
DELAY_BETWEEN_CALLS = 2  # Seconds to wait (to avoid rate limits)

class MetricsCalculator:
    """
    Calculates standard OCR evaluation metrics without external dependencies.
    Implements Levenshtein distance for CER and WER.
    """
    
    @staticmethod
    def levenshtein(s1: List, s2: List) -> int:
        """
        Calculates edit distance between two sequences (chars or words).
        """
        if len(s1) < len(s2):
            return MetricsCalculator.levenshtein(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

    @staticmethod
    def calculate_cer(reference: str, hypothesis: str) -> float:
        """Character Error Rate = Edit Distance / Length of Reference"""
        if not reference: return 0.0 if not hypothesis else 1.0
        ref_chars = list(reference)
        hyp_chars = list(hypothesis)
        dist = MetricsCalculator.levenshtein(ref_chars, hyp_chars)
        return dist / len(ref_chars)

    @staticmethod
    def calculate_wer(reference: str, hypothesis: str) -> float:
        """Word Error Rate = Edit Distance / Word Count of Reference"""
        if not reference: return 0.0 if not hypothesis else 1.0
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        dist = MetricsCalculator.levenshtein(ref_words, hyp_words)
        return dist / len(ref_words) if len(ref_words) > 0 else 1.0

class OCRSimulator:
    """Simulates common OCR errors found in ancient inscriptions."""
    
    def __init__(self, damage_rate=0.15):
        self.damage_rate = damage_rate
        # Common confusables in OCR (visual similarity)
        self.noise_chars = ['_', '.', ',', '|', '?', '!', '%', ';']
    
    def damage_text(self, text: str) -> str:
        chars = list(text)
        num_damage = int(len(chars) * self.damage_rate)
        indices = random.sample(range(len(chars)), num_damage)
        
        for idx in indices:
            action = random.choice(['delete', 'substitute', 'noise'])
            
            if action == 'delete':
                chars[idx] = '' # Remove character
            elif action == 'substitute':
                # Simulate misread character
                chars[idx] = random.choice(self.noise_chars) 
            elif action == 'noise':
                # Add random noise
                chars[idx] = chars[idx] + random.choice(self.noise_chars)
                
        return "".join(chars)

def run_test_suite():
    if not API_KEY:
        print("❌ Error: GEMINI_API_KEY environment variable not found.")
        return

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found.")
        return

    print(f"🔹 Starting Restoration Test using Model: gemini-1.5-flash")
    print(f"🔹 Damage Level: {DAMAGE_LEVEL * 100}%")
    print("="*60)

    simulator = OCRSimulator(damage_rate=DAMAGE_LEVEL)
    results = []
    
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    total_cer = 0
    total_wer = 0

    for i, original_text in enumerate(lines):
        print(f"\nProcessing Sentence {i+1}/{len(lines)}...")
        
        # 1. Simulate OCR Damage
        damaged_text = simulator.damage_text(original_text)
        
        # 2. Reconstruct using your Gemini pipeline
        start_time = time.time()
        restored_text = reconstruct_with_gemini(damaged_text, API_KEY)
        duration = time.time() - start_time
        
        # Clean up output for metric calculation (trim whitespace)
        restored_text = restored_text.strip()
        
        # 3. Calculate Metrics
        cer = MetricsCalculator.calculate_cer(original_text, restored_text)
        wer = MetricsCalculator.calculate_wer(original_text, restored_text)
        accuracy = (1 - cer) * 100
        
        total_cer += cer
        total_wer += wer
        
        # 4. Print Individual Result
        print(f"Original : {original_text}")
        print(f"Damaged  : {damaged_text}")
        print(f"Restored : {restored_text}")
        print(f"--> Accuracy: {accuracy:.2f}% | CER: {cer:.2f} | Time: {duration:.2f}s")
        
        results.append({
            "id": i+1,
            "original": original_text,
            "restored": restored_text,
            "cer": cer
        })
        
        time.sleep(DELAY_BETWEEN_CALLS)

    # --- FINAL REPORT ---
    avg_cer = total_cer / len(lines)
    avg_wer = total_wer / len(lines)
    avg_acc = (1 - avg_cer) * 100

    print("\n" + "="*60)
    print("FINAL TEST REPORT")
    print("="*60)
    print(f"Total Sentences Processed : {len(lines)}")
    print(f"Average Character Error Rate (CER) : {avg_cer:.4f} (Lower is better)")
    print(f"Average Word Error Rate (WER)      : {avg_wer:.4f} (Lower is better)")
    print(f"Overall System Accuracy            : {avg_acc:.2f}%")
    print("="*60)

if __name__ == "__main__":
    run_test_suite()