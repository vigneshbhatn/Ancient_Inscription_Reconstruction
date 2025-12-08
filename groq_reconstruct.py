"""
Groq API integration for reconstructing ancient Kannada inscriptions.
Uses Groq's Llama-3.1-70b-versatile model to correct and reconstruct OCR text.
"""

from groq import Groq
import time
from typing import Dict, List, Optional

# Default model configuration
DEFAULT_MODEL = "llama-3.3-70b-versatile"

def get_groq_client(api_key: str) -> Groq:
    """
    Initialize Groq client with the provided API key.
    
    Args:
        api_key: Groq API key
    Returns:
        Groq client instance
    """
    return Groq(api_key=api_key)


def reconstruct_with_groq(
    ocr_text: str,
    api_key: str,
    context: Optional[str] = None,
    model_name: str = DEFAULT_MODEL
) -> str:
    """
    Reconstruct ancient Kannada text using Groq API.
    
    Args:
        ocr_text: Raw OCR extracted text
        api_key: Groq API key
        context: Optional additional context about the inscription
        model_name: Groq model to use (default: llama-3.1-70b-versatile)
    
    Returns:
        Reconstructed and corrected text
    """
    try:
        # Create client instance
        client = get_groq_client(api_key)
        
        # Build prompt
        prompt = build_reconstruction_prompt(ocr_text, context)
        
        # Generate response
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model_name,
            temperature=0,  # Low temperature for more deterministic/factual output
        )
        
        # Extract text from response
        reconstructed_text = chat_completion.choices[0].message.content
        
        return reconstructed_text
    
    except Exception as e:
        return f"Error during reconstruction: {str(e)}"


def build_reconstruction_prompt(ocr_text: str, context: Optional[str] = None) -> str:
    """
    Strict academic prompt that forces the AI to justify its changes.
    """
    base_prompt = f"""You are a strict epigraphical assistant for ancient Kannada inscriptions (Halegannada). 
Your goal is ACCURACY, not creativity. Do not invent names, places, or events that are not supported by the visible text fragments.

Task:
1. Analyze the OCR text.
2. Identify standard epigraphical patterns (e.g., 'Svasti Shri' preambles, standard titles).
3. Reconstruct ONLY the missing letters based on these standard patterns.
4. Translate the result.

OCR Input:
{ocr_text}
"""

    if context:
        base_prompt += f"\nContext: {context}\n"

    base_prompt += """
Output Format:
**Literal Reading:**
[Transcribe exactly what is visible, using '...' for illegible parts]

**Reconstructed Text:**
[The full text with your corrections]

**Translation:**
[English translation]

**Justification:**
[Explain WHY you reconstructed specific words. Example: "Inferred 'Rajadhiraja' from visible fragment 'Ra..dhi..ja' as it is a standard title."]

If the text is too damaged to make sense, state: "Insufficient data for reconstruction."
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
        api_key: Groq API key
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
    
    return reconstruct_with_groq(ocr_text, api_key, context)


def batch_reconstruct(
    ocr_texts: List[str],
    api_key: str,
    delay: float = 1.0
) -> List[str]:
    """
    Reconstruct multiple OCR texts in batch.
    
    Args:
        ocr_texts: List of OCR text strings
        api_key: Groq API key
        delay: Delay between API calls (seconds) to avoid rate limiting
    
    Returns:
        List of reconstructed texts
    """
    results = []
    
    for i, ocr_text in enumerate(ocr_texts):
        print(f"Processing text {i+1}/{len(ocr_texts)}...")
        
        reconstructed = reconstruct_with_groq(ocr_text, api_key)
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
        api_key: Groq API key
        num_variations: Number of variations to generate
    
    Returns:
        List of different reconstruction attempts
    """
    client = get_groq_client(api_key)
    
    variations = []
    
    for i in range(num_variations):
        prompt = build_reconstruction_prompt(ocr_text)
        prompt += f"\n\nThis is attempt {i+1}. Provide an independent reconstruction."
        
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=DEFAULT_MODEL,
                temperature=0.7  # Higher temperature for variations
            )
            variations.append(chat_completion.choices[0].message.content)
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
        api_key: Groq API key
        user_feedback: User's feedback or corrections to previous attempt
    
    Returns:
        Refined reconstruction (Kannada text only)
    """
    client = get_groq_client(api_key)
    
    prompt = build_reconstruction_prompt(ocr_text)
    
    if user_feedback:
        prompt += f"""
\nUser Feedback on Previous Reconstruction:
{user_feedback}

Please refine the reconstruction based on this feedback. Output only the corrected Kannada text.
"""
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=DEFAULT_MODEL
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error during interactive reconstruction: {str(e)}"


def extract_confidence_score(reconstruction_text: str) -> Optional[float]:
    """
    Extract confidence score from the reconstruction text if available.
    
    Args:
        reconstruction_text: Reconstruction output
    
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
        reconstruction: Raw reconstruction text
    
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