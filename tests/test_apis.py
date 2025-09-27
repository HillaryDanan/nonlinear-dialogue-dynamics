#!/usr/bin/env python3
"""Quick API test"""

import os
from dotenv import load_dotenv

# Load .env
load_dotenv()

print("Environment Check:")
print("="*50)

# Check each key
keys = {
    'ANTHROPIC_API_KEY': 'Anthropic',
    'GOOGLE_API_KEY': 'Google',
    'OPENAI_API_KEY': 'OpenAI'
}

for key, name in keys.items():
    value = os.getenv(key)
    if value:
        # Show first/last 4 chars only for security
        masked = f"{value[:4]}...{value[-4:]}" if len(value) > 8 else "***"
        print(f"✓ {name:12} {masked}")
    else:
        print(f"✗ {name:12} NOT FOUND")

print("\nNow testing models...")
import asyncio
import sys
sys.path.append('src/core')
from model_interface import test_models

asyncio.run(test_models())
