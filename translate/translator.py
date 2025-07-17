# translate/translator.py
from transformers import pipeline

try:
    translator_hi_to_en = pipeline("translation", model="Helsinki-NLP/opus-mt-hi-en")
    translator_en_to_hi = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")
except Exception as e:
    # This will print the actual error during pipeline loading
    print(f"Error loading translation pipelines: {e}")
    # You might want to re-raise or handle this more gracefully
    raise

def translate_hi_to_en(text: str) -> str:
    """Translate Hindi to English."""
    try:
        translated = translator_hi_to_en(text, max_length=512)
        return translated[0]['translation_text']
    except Exception as e:
        return f"Translation Error (Hindi to English): {str(e)}"

def translate_en_to_hi(text: str) -> str:
    """Translate English to Hindi."""
    try:
        translated = translator_en_to_hi(text, max_length=512)
        return translated[0]['translation_text']
    except Exception as e:
        return f"Translation Error (English to Hindi): {str(e)}"