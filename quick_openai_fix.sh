#!/bin/bash

# Quick fix for OpenAI proxy issue

echo "======================================"
echo "QUICK OPENAI FIX ATTEMPT"
echo "======================================"

# 1. Clear proxy environment variables for this session
unset HTTP_PROXY
unset HTTPS_PROXY
unset http_proxy
unset https_proxy
unset ALL_PROXY
unset all_proxy

echo "✓ Cleared proxy variables"

# 2. Reinstall OpenAI with clean cache
echo "Reinstalling OpenAI library..."
python3 -m pip uninstall openai -y > /dev/null 2>&1
python3 -m pip cache purge > /dev/null 2>&1
python3 -m pip install openai==1.45.0 --no-cache-dir

# 3. Test it
echo ""
echo "Testing OpenAI..."
python3 -c "
import os
from dotenv import load_dotenv
load_dotenv()

# Clear any proxy settings
for var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    if var in os.environ:
        del os.environ[var]

try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': 'test'}],
        max_tokens=5
    )
    print('✓ OpenAI WORKS!')
    print(f'Response: {response.choices[0].message.content}')
except Exception as e:
    print(f'✗ Still broken: {e}')
"

echo ""
echo "======================================"
echo "If still broken, run:"
echo "python3 fix_and_proceed.py"
echo "======================================"