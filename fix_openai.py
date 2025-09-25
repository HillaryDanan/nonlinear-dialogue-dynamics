"""
Fix for OpenAI proxy issue
The error suggests environment variables or config is adding unwanted proxy args
"""

import os
import sys

def diagnose_openai_issue():
    """Figure out what's wrong with OpenAI setup"""
    
    print("Diagnosing OpenAI issue...")
    
    # Check for proxy-related environment variables
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 
                  'NO_PROXY', 'no_proxy', 'ALL_PROXY', 'all_proxy']
    
    print("\nProxy-related environment variables:")
    for var in proxy_vars:
        value = os.environ.get(var)
        if value:
            print(f"  {var} = {value}")
    
    # Check OpenAI version
    try:
        import openai
        print(f"\nOpenAI library version: {openai.__version__}")
    except Exception as e:
        print(f"Error checking version: {e}")
    
    # Try different initialization methods
    print("\nTrying different OpenAI initialization methods...")
    
    # Method 1: Basic initialization
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        )
        print("✓ Method 1: Basic initialization works")
        
        # Test it
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say test"}],
            max_tokens=5
        )
        print(f"✓ API call successful: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"✗ Method 1 failed: {e}")
        
    # Method 2: Try legacy openai style
    try:
        import openai
        openai.api_key = os.getenv('OPENAI_API_KEY')
        
        # Remove any proxy config if it exists
        if hasattr(openai, 'proxy'):
            delattr(openai, 'proxy')
        if hasattr(openai, 'proxies'):
            delattr(openai, 'proxies')
            
        print("✓ Method 2: Legacy style setup")
        
    except Exception as e:
        print(f"✗ Method 2 failed: {e}")
    
    # Method 3: Clean environment and retry
    try:
        # Temporarily clear proxy vars
        old_vars = {}
        for var in proxy_vars:
            if var in os.environ:
                old_vars[var] = os.environ[var]
                del os.environ[var]
        
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        print("✓ Method 3: Works with proxy vars removed")
        
        # Restore vars
        for var, value in old_vars.items():
            os.environ[var] = value
            
    except Exception as e:
        print(f"✗ Method 3 failed: {e}")
        # Restore vars even on failure
        for var, value in old_vars.items():
            os.environ[var] = value

if __name__ == "__main__":
    diagnose_openai_issue()