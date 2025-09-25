"""Test API connections before running experiment"""

import os
from dotenv import load_dotenv
import openai
import anthropic
import google.generativeai as genai

load_dotenv()

def test_apis():
    print("Testing API connections...")
    
    # Test OpenAI
    try:
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'API working'"}],
            max_tokens=10
        )
        print("✓ OpenAI API working")
    except Exception as e:
        print(f"✗ OpenAI API failed: {e}")
    
    # Test Anthropic
    try:
        client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": "Say 'API working'"}],
            max_tokens=10
        )
        print("✓ Anthropic API working")
    except Exception as e:
        print(f"✗ Anthropic API failed: {e}")
    
    # Test Google
    try:
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content("Say 'API working'")
        print("✓ Google API working")
    except Exception as e:
        print(f"✗ Google API failed: {e}")

if __name__ == "__main__":
    test_apis()
