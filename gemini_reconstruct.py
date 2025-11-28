"""
Gemini API integration for reconstructing ancient Kannada inscriptions.
Uses Google's Gemini API to correct and reconstruct OCR text.
"""

import google.generativeai as genai
import time
from typing import Dict, List, Optional


def configure_gemini(api_key: str):
    """
    Configure Gemini API with the provided API key.
    
    Args:
        api_key: Google Gemini API key
    """
    genai.configure(api_key=api_key)


def reconstruct_with_gemini(
    ocr_text: str,
    api_key: str,
    context: Optional[str] = None,
    model_name: str = "gemini-2.5-flash"
) -> str:
    """
    Reconstruct ancient Kannada text using Gemini API.
    
    Args:
        ocr_text: Raw OCR extracted text
        api_key: Gemini API key
        context: Optional additional context about the inscription
        model_name: Gemini model to use
    
    Returns:
        Reconstructed and corrected text
    """
    # Configure API
    configure_gemini(api_key)
    
    # Create model instance
    model = genai.GenerativeModel(model_name)
    
    # Build prompt
    prompt = build_reconstruction_prompt(ocr_text, context)
    
    try:
        # Generate response
        response = model.generate_content(prompt)
        
        # Extract text from response
        reconstructed_text = response.text
        
        return reconstructed_text
    
    except Exception as e:
        return f"Error during reconstruction: {str(e)}"


def build_reconstruction_prompt(ocr_text: str, context: Optional[str] = None) -> str:
    """
    Build a detailed prompt for Gemini to reconstruct the inscription.
    
    Args:
        ocr_text: Raw OCR text
        context: Optional context
    
    Returns:
        Formatted prompt string
    """
    base_prompt = f"""You are an expert paleographer and historian specializing in ancient Kannada inscriptions.You are helping in reconstructing ancient inscriptions belonging to the the region of Karnataka. These inscriptions are of the time period between 9th to 15th century AD and in the Halegannada language.Do not include any information which of after 15th century, also do not display any current affairs and modern kannada history. Also do not include any information related to inscriptions after 15th century. 

Your task is to reconstruct and correct the following text that was extracted from an ancient Kannada inscription using OCR. The OCR may contain errors due to:
- Weathering and damage to the stone
- Dots and artifacts in the image
- Character segmentation issues
- Archaic script forms

OCR Output:
{ocr_text}
"""
    
    if context:
        base_prompt += f"\nAdditional Context:\n{context}\n"
    
    base_prompt += """
Please provide ONLY the corrected and reconstructed Kannada text. Do not include any explanations, translations, transliterations, or analysis.

Output only the reconstructed Kannada sentence(s).
"""
    
    return base_prompt


def reconstruct_with_context(
    ocr_text: str,
    api_key: str,
    historical_period: Optional[str] = None,
    location: Optional[str] = None,
    inscription_type: Optional[str] = None
) -> str:
    """
    Reconstruct text with additional contextual information.
    
    Args:
        ocr_text: Raw OCR text
        api_key: Gemini API key
        historical_period: Time period of inscription (e.g., "10th century CE")
        location: Location of inscription (e.g., "Hampi, Karnataka")
        inscription_type: Type of inscription (e.g., "royal decree", "temple inscription")
    
    Returns:
        Reconstructed text
    """
    context_parts = []
    
    if historical_period:
        context_parts.append(f"Historical Period: {historical_period}")
    if location:
        context_parts.append(f"Location: {location}")
    if inscription_type:
        context_parts.append(f"Type: {inscription_type}")
    
    context = "\n".join(context_parts) if context_parts else None
    
    return reconstruct_with_gemini(ocr_text, api_key, context)


def batch_reconstruct(
    ocr_texts: List[str],
    api_key: str,
    delay: float = 1.0
) -> List[str]:
    """
    Reconstruct multiple OCR texts in batch.
    
    Args:
        ocr_texts: List of OCR text strings
        api_key: Gemini API key
        delay: Delay between API calls (seconds) to avoid rate limiting
    
    Returns:
        List of reconstructed texts
    """
    results = []
    
    for i, ocr_text in enumerate(ocr_texts):
        print(f"Processing text {i+1}/{len(ocr_texts)}...")
        
        reconstructed = reconstruct_with_gemini(ocr_text, api_key)
        results.append(reconstructed)
        
        # Add delay to avoid rate limiting
        if i < len(ocr_texts) - 1:
            time.sleep(delay)
    
    return results


def compare_reconstructions(
    ocr_text: str,
    api_key: str,
    num_variations: int = 3
) -> List[str]:
    """
    Generate multiple reconstruction variations for comparison.
    
    Args:
        ocr_text: Raw OCR text
        api_key: Gemini API key
        num_variations: Number of variations to generate
    
    Returns:
        List of different reconstruction attempts
    """
    configure_gemini(api_key)
    model = genai.GenerativeModel("gemini-pro")
    
    variations = []
    
    for i in range(num_variations):
        prompt = build_reconstruction_prompt(ocr_text)
        prompt += f"\n\nThis is attempt {i+1}. Provide an independent reconstruction."
        
        try:
            response = model.generate_content(prompt)
            variations.append(response.text)
            time.sleep(1)  # Rate limiting
        except Exception as e:
            variations.append(f"Error in variation {i+1}: {str(e)}")
    
    return variations


def interactive_reconstruction(
    ocr_text: str,
    api_key: str,
    user_feedback: Optional[str] = None
) -> str:
    """
    Interactive reconstruction that incorporates user feedback.
    
    Args:
        ocr_text: Raw OCR text
        api_key: Gemini API key
        user_feedback: User's feedback or corrections to previous attempt
    
    Returns:
        Refined reconstruction (Kannada text only)
    """
    configure_gemini(api_key)
    model = genai.GenerativeModel("gemini-pro")
    
    prompt = build_reconstruction_prompt(ocr_text)
    
    if user_feedback:
        prompt += f"""
\nUser Feedback on Previous Reconstruction:
{user_feedback}

Please refine the reconstruction based on this feedback. Output only the corrected Kannada text.
"""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error during interactive reconstruction: {str(e)}"


def extract_confidence_score(reconstruction_text: str) -> Optional[float]:
    """
    Extract confidence score from the reconstruction text if available.
    
    Args:
        reconstruction_text: Gemini's reconstruction output
    
    Returns:
        Confidence score (0-1) or None if not found
    """
    # Look for confidence indicators in the text
    import re
    
    patterns = [
        r'confidence[:\s]+(\d+)%',
        r'confidence[:\s]+(\d+\.\d+)',
        r'certainty[:\s]+(\d+)%',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, reconstruction_text.lower())
        if match:
            value = float(match.group(1))
            # Convert percentage to decimal if needed
            return value / 100 if value > 1 else value
    
    return None


def format_reconstruction_output(reconstruction: str) -> Dict[str, str]:
    """
    Parse and format the reconstruction output into structured sections.
    
    Args:
        reconstruction: Raw reconstruction text from Gemini
    
    Returns:
        Dictionary with parsed sections
    """
    sections = {
        'corrected_text': '',
        'transliteration': '',
        'translation': '',
        'analysis': ''
    }
    
    # Simple parsing based on common section headers
    lines = reconstruction.split('\n')
    current_section = None
    
    for line in lines:
        line_lower = line.lower().strip()
        
        if 'corrected' in line_lower or 'reconstructed' in line_lower:
            current_section = 'corrected_text'
        elif 'transliteration' in line_lower:
            current_section = 'transliteration'
        elif 'translation' in line_lower:
            current_section = 'translation'
        elif 'analysis' in line_lower or 'explanation' in line_lower:
            current_section = 'analysis'
        elif current_section and line.strip():
            sections[current_section] += line + '\n'
    
    # Clean up sections
    for key in sections:
        sections[key] = sections[key].strip()
    
    return sections