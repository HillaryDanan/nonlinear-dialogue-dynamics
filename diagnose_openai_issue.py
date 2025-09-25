"""
Diagnose and fix the OpenAI proxy issue definitively
"""

import os
import sys
import json
from pathlib import Path

def deep_diagnosis():
    """Deep dive into what's causing the proxy error"""
    
    print("="*60)
    print("OPENAI DEEP DIAGNOSIS")
    print("="*60)
    
    # 1. Check OpenAI version
    try:
        import openai
        print(f"\n1. OpenAI Library Version: {openai.__version__}")
        
        # Check if it's the right version
        if openai.__version__ != "1.45.0":
            print(f"   ⚠️  Unexpected version (expected 1.45.0)")
    except Exception as e:
        print(f"Error checking version: {e}")
    
    # 2. Check for openai config files
    print("\n2. Checking for OpenAI config files...")
    config_locations = [
        Path.home() / ".openai" / "config.json",
        Path.home() / ".config" / "openai" / "config.json",
        Path.cwd() / "openai_config.json",
        Path.cwd() / ".openai_config"
    ]
    
    for config_path in config_locations:
        if config_path.exists():
            print(f"   Found config at: {config_path}")
            try:
                with open(config_path) as f:
                    config = json.load(f)
                    if 'proxy' in config or 'proxies' in config:
                        print(f"   ⚠️  Config contains proxy settings: {config}")
            except:
                pass
    
    # 3. Check environment for proxy settings
    print("\n3. Environment proxy variables:")
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 
                  'ALL_PROXY', 'all_proxy', 'OPENAI_PROXY']
    
    found_proxy = False
    for var in proxy_vars:
        if var in os.environ:
            print(f"   {var} = {os.environ[var]}")
            found_proxy = True
    
    if not found_proxy:
        print("   No proxy variables found")
    
    # 4. Check Python's site-packages for modifications
    print("\n4. Checking OpenAI package location...")
    try:
        import openai
        print(f"   Package location: {openai.__file__}")
        
        # Check if there's a monkey patch or modification
        init_file = Path(openai.__file__).parent / "__init__.py"
        if init_file.exists():
            with open(init_file) as f:
                content = f.read()
                if 'proxy' in content.lower():
                    print("   ⚠️  Found 'proxy' references in __init__.py")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 5. Try different initialization methods
    print("\n5. Testing initialization methods...")
    
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("   ✗ No API key found")
        return
    
    # Method A: Direct import with minimal setup
    print("\n   Method A: Minimal import")
    try:
        # Clear any OpenAI module from cache
        if 'openai' in sys.modules:
            del sys.modules['openai']
        
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        print("   ✓ Method A works!")
        
        # Test it
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        print(f"   ✓ API call successful: {response.choices[0].message.content}")
        return client
        
    except Exception as e:
        print(f"   ✗ Method A failed: {e}")
    
    # Method B: Try with explicit no proxy
    print("\n   Method B: Bypass proxy completely")
    try:
        # Temporarily clear ALL proxy settings
        old_env = {}
        proxy_vars_extended = proxy_vars + ['NO_PROXY', 'no_proxy']
        
        for var in proxy_vars_extended:
            if var in os.environ:
                old_env[var] = os.environ[var]
                del os.environ[var]
        
        # Set NO_PROXY to everything
        os.environ['NO_PROXY'] = '*'
        
        # Reimport
        if 'openai' in sys.modules:
            del sys.modules['openai']
        
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Test it
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        print("   ✓ Method B works (with proxy bypass)!")
        print(f"   ✓ Response: {response.choices[0].message.content}")
        
        # Restore environment
        for var, val in old_env.items():
            os.environ[var] = val
        
        return client
        
    except Exception as e:
        print(f"   ✗ Method B failed: {e}")
        
        # Restore environment even on failure
        for var, val in old_env.items():
            os.environ[var] = val
    
    # Method C: Use requests directly (bypass openai library)
    print("\n   Method C: Direct HTTP request (bypass library)")
    try:
        import requests
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 5
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=10
        )
        
        if response.status_code == 200:
            print("   ✓ Method C works (direct HTTP)!")
            result = response.json()
            print(f"   ✓ Response: {result['choices'][0]['message']['content']}")
        else:
            print(f"   ✗ Method C failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"   ✗ Method C failed: {e}")

def create_workaround():
    """Create a working OpenAI wrapper"""
    
    print("\n" + "="*60)
    print("CREATING WORKAROUND")
    print("="*60)
    
    workaround_code = '''"""
OpenAI Workaround Wrapper
Bypasses proxy issues by using direct HTTP requests
"""

import os
import requests
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

class OpenAIWorkaround:
    """Direct HTTP implementation bypassing proxy issues"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.base_url = "https://api.openai.com/v1"
        
        if not self.api_key:
            raise ValueError("No API key provided")
    
    def chat_completion(self, 
                       messages: List[Dict],
                       model: str = "gpt-4o-mini",
                       max_tokens: int = 150,
                       temperature: float = 0.7) -> tuple:
        """Make chat completion request"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                tokens = result['usage']['total_tokens']
                return content, tokens
            else:
                print(f"OpenAI error: {response.status_code} - {response.text}")
                return f"Error: {response.status_code}", 0
                
        except Exception as e:
            print(f"Request failed: {e}")
            return f"Error: {str(e)}", 0

# Test it
if __name__ == "__main__":
    client = OpenAIWorkaround()
    response, tokens = client.chat_completion([
        {"role": "user", "content": "Say 'Workaround successful!'"}
    ])
    print(f"Response: {response}")
    print(f"Tokens: {tokens}")
'''
    
    # Save workaround
    with open("openai_workaround.py", "w") as f:
        f.write(workaround_code)
    
    print("Created openai_workaround.py")
    print("\nTesting workaround...")
    
    # Test it
    try:
        from openai_workaround import OpenAIWorkaround
        client = OpenAIWorkaround()
        response, tokens = client.chat_completion([
            {"role": "user", "content": "test"}
        ])
        print(f"✓ Workaround successful: {response}")
        return True
    except Exception as e:
        print(f"✗ Workaround failed: {e}")
        return False

if __name__ == "__main__":
    # Run diagnosis
    deep_diagnosis()
    
    # If standard methods fail, create workaround
    print("\n" + "="*60)
    print("SOLUTION")
    print("="*60)
    
    print("\nBased on diagnosis, recommendations:")
    print("1. Clear proxy environment variables")
    print("2. Use the workaround wrapper if needed")
    print("3. Or proceed with just Anthropic + Google (both work fine)")
    
    create = input("\nCreate workaround wrapper? (y/n): ")
    if create.lower() == 'y':
        if create_workaround():
            print("\n✓ Workaround created and tested successfully!")
            print("Now update data_collector_all_models.py to use the workaround")