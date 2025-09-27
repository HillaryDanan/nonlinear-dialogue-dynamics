"""Test the new OpenAI key"""
import os
from dotenv import load_dotenv
from openai import OpenAI

# Reload environment
load_dotenv(override=True)

api_key = os.getenv('OPENAI_API_KEY')
print(f"Testing key: {api_key[:8]}...{api_key[-4:]}")

try:
    client = OpenAI(api_key=api_key)
    
    # Simple test
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'working'"}],
        max_tokens=10
    )
    
    print(f"✅ SUCCESS! New key works!")
    print(f"Response: {response.choices[0].message.content}")
    print(f"Tokens used: {response.usage.total_tokens}")
    print(f"Model: {response.model}")
    
except Exception as e:
    print(f"❌ Still broken: {e}")
    if "organization" in str(e).lower():
        print("\nMake sure you created the key while logged into tide-analysis org!")
