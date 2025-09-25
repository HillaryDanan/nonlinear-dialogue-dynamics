"""Check what models are actually available"""

import os
from dotenv import load_dotenv

load_dotenv()

def check_openai():
    """List available OpenAI models"""
    import openai
    try:
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        # Remove any proxy settings if they exist
        models = client.models.list()
        print("OpenAI Models Available:")
        gpt_models = [m.id for m in models if 'gpt' in m.id]
        for model in sorted(gpt_models):
            print(f"  - {model}")
    except Exception as e:
        print(f"OpenAI error: {e}")

def check_anthropic():
    """Check Anthropic models"""
    import anthropic
    print("\nAnthropic Models (September 2025):")
    print("  - claude-3-5-sonnet-20241022")  # Current model
    print("  - claude-3-5-haiku-20241022")   # Faster, cheaper
    print("  - claude-3-opus-20240229")       # Most capable
    
def check_google():
    """Check Google models"""
    import google.generativeai as genai
    try:
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        print("\nGoogle Models Available:")
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                print(f"  - {model.name}")
    except Exception as e:
        print(f"Google error: {e}")

if __name__ == "__main__":
    check_openai()
    check_anthropic()
    check_google()