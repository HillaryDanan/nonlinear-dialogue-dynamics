"""
Find ANY fucking OpenAI model that works
=========================================
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import time

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Try these models from your Tier 1 access
models_to_try = [
    "gpt-3.5-turbo",           # 200,000 TPM
    "gpt-3.5-turbo-0125",      # 200,000 TPM
    "gpt-3.5-turbo-1106",      # 200,000 TPM
    "gpt-4o-mini",             # 200,000 TPM
    "gpt-4o-mini-2024-07-18",  # Specific version
    "babbage-002",             # 250,000 TPM - completion model
    "davinci-002",             # 250,000 TPM - completion model
]

print("Testing OpenAI models to find one that fucking works...\n")

for model in models_to_try:
    print(f"Trying {model}...", end=" ")
    
    try:
        if "babbage" in model or "davinci" in model:
            # Completion models use different API
            response = client.completions.create(
                model=model,
                prompt="Say hello",
                max_tokens=5
            )
            text = response.choices[0].text
        else:
            # Chat models
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Say hello"}],
                max_tokens=5
            )
            text = response.choices[0].message.content
        
        print(f"✅ WORKS! Response: {text}")
        print(f"   Model to use: {model}")
        break
        
    except Exception as e:
        error_str = str(e)
        if "quota" in error_str:
            print(f"❌ Quota error")
        elif "model" in error_str:
            print(f"❌ Model not found")
        else:
            print(f"❌ {error_str[:50]}")
    
    time.sleep(1)  # Pause between attempts

else:
    print("\n❌ No OpenAI models work. Something is fucked with your account.")
    print("\nPossible issues:")
    print("1. Your API key might be from a different organization")
    print("2. Your organization might have restrictions")
    print("3. Check if this key matches the org showing $117 budget")
