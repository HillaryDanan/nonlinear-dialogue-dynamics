#!/usr/bin/env python3
"""Dead simple API test"""

import os
from dotenv import load_dotenv
load_dotenv()

print("Testing each API directly...\n")

# 1. Test Anthropic
print("ANTHROPIC:")
try:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        messages=[{"role": "user", "content": "Say 'working'"}],
        max_tokens=10
    )
    print(f"✓ Response: {response.content[0].text}")
    print(f"✓ Tokens: {response.usage.input_tokens + response.usage.output_tokens}")
except Exception as e:
    print(f"✗ Error: {e}")

# 2. Test Google
print("\nGOOGLE:")
try:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("Say 'working'")
    print(f"✓ Response: {response.text}")
except Exception as e:
    print(f"✗ Error: {e}")

# 3. Test OpenAI
print("\nOPENAI:")
try:
    import openai
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'working'"}],
        max_tokens=10
    )
    print(f"✓ Response: {response.choices[0].message.content}")
    print(f"✓ Tokens: {response.usage.total_tokens}")
except Exception as e:
    print(f"✗ Error: {e}")
