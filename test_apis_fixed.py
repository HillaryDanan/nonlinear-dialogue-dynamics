"""Test API connections with correct models (September 2025)"""

import os
from dotenv import load_dotenv

load_dotenv()

def test_apis():
    print("Testing API connections with current models...")
    
    # Test OpenAI
    try:
        import openai
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Still available
            messages=[{"role": "user", "content": "Say 'API working'"}],
            max_tokens=10
        )
        print(f"✓ OpenAI API working: {response.choices[0].message.content}")
    except Exception as e:
        print(f"✗ OpenAI API failed: {e}")
        # Try gpt-4o-mini as fallback
        try:
            import openai
            client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Newer, cheaper alternative
                messages=[{"role": "user", "content": "Say 'API working'"}],
                max_tokens=10
            )
            print(f"✓ OpenAI API working with gpt-4o-mini: {response.choices[0].message.content}")
        except Exception as e2:
            print(f"✗ OpenAI still failed: {e2}")
    
    # Test Anthropic with current models
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",  # Current sonnet model
            messages=[{"role": "user", "content": "Say 'API working'"}],
            max_tokens=10
        )
        print(f"✓ Anthropic API working: {response.content[0].text}")
    except Exception as e:
        print(f"✗ Anthropic API failed: {e}")
        # Try haiku as cheaper alternative
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
            response = client.messages.create(
                model="claude-3-5-haiku-20241022",  # Cheaper, faster
                messages=[{"role": "user", "content": "Say 'API working'"}],
                max_tokens=10
            )
            print(f"✓ Anthropic API working with haiku: {response.content[0].text}")
        except Exception as e2:
            print(f"✗ Anthropic still failed: {e2}")
    
    # Test Google with current models
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        
        # Try gemini-1.5-flash first (cheaper, faster)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Say 'API working'")
        print(f"✓ Google API working with gemini-1.5-flash: {response.text}")
    except Exception as e:
        print(f"✗ Google API failed: {e}")
        # Try gemini-1.5-pro as alternative
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
            model = genai.GenerativeModel('gemini-1.5-pro')
            response = model.generate_content("Say 'API working'")
            print(f"✓ Google API working with gemini-1.5-pro: {response.text}")
        except Exception as e2:
            print(f"✗ Google still failed: {e2}")

if __name__ == "__main__":
    test_apis()