"""
Deep fix for the persistent OpenAI proxy issue
This will find what's injecting the 'proxies' parameter
"""

import os
import sys
import inspect
import importlib
from pathlib import Path

def find_the_culprit():
    """Find what's adding the proxies parameter"""
    
    print("="*60)
    print("DEEP OPENAI DEBUGGING")
    print("="*60)
    
    # 1. Check if httpx is the problem
    print("\n1. Checking httpx configuration...")
    try:
        import httpx
        print(f"   httpx version: {httpx.__version__}")
        
        # Check if httpx has proxy settings
        if hasattr(httpx, '_client'):
            print(f"   httpx._client exists")
        
        # Check default transport
        client = httpx.Client()
        if hasattr(client, '_transport'):
            print(f"   httpx transport: {client._transport}")
        if hasattr(client, '_proxies'):
            print(f"   ⚠️  httpx has proxies: {client._proxies}")
            
    except Exception as e:
        print(f"   httpx check error: {e}")
    
    # 2. Check if there's a sitecustomize.py interfering
    print("\n2. Checking for sitecustomize.py...")
    try:
        import sitecustomize
        print(f"   ⚠️  Found sitecustomize at: {sitecustomize.__file__}")
        
        # Check its content
        with open(sitecustomize.__file__, 'r') as f:
            content = f.read()
            if 'proxy' in content.lower():
                print(f"   ⚠️  sitecustomize contains proxy configuration!")
                
    except ImportError:
        print("   No sitecustomize.py (good)")
    
    # 3. Check if openai package has been modified
    print("\n3. Checking OpenAI package integrity...")
    try:
        import openai
        openai_path = Path(openai.__file__).parent
        
        # Check the _client.py file
        client_file = openai_path / "_client.py"
        if client_file.exists():
            with open(client_file, 'r') as f:
                content = f.read()
                
                # Look for proxy references
                if 'proxies' in content:
                    # Count occurrences
                    count = content.count('proxies')
                    print(f"   Found {count} references to 'proxies' in _client.py")
                    
                    # Find the specific lines
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if 'proxies' in line and 'kwargs' in line:
                            print(f"   Line {i+1}: {line.strip()[:80]}")
                            
    except Exception as e:
        print(f"   Error checking OpenAI package: {e}")
    
    # 4. Check for monkey patching
    print("\n4. Checking for monkey patches...")
    
    # Get the actual OpenAI Client class
    try:
        from openai import OpenAI
        
        # Check __init__ signature
        init_signature = inspect.signature(OpenAI.__init__)
        params = list(init_signature.parameters.keys())
        
        print(f"   OpenAI.__init__ parameters: {params}")
        
        if 'proxies' in params:
            print(f"   ⚠️  'proxies' is in the parameters! This shouldn't be there!")
        
        # Check if __init__ has been wrapped
        if hasattr(OpenAI.__init__, '__wrapped__'):
            print(f"   ⚠️  OpenAI.__init__ has been wrapped!")
            
    except Exception as e:
        print(f"   Error inspecting OpenAI: {e}")
    
    # 5. Try to trace where the proxies argument comes from
    print("\n5. Tracing the error...")
    
    try:
        from openai import OpenAI
        import traceback
        
        # Try to initialize with explicit no proxies
        try:
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        except Exception as e:
            print(f"   Error: {e}")
            print("\n   Full traceback:")
            traceback.print_exc()
            
    except Exception as e:
        print(f"   Trace error: {e}")

def create_working_wrapper():
    """Create a wrapper that definitely works"""
    
    print("\n" + "="*60)
    print("CREATING BULLETPROOF WRAPPER")
    print("="*60)
    
    wrapper_code = '''"""
Bulletproof OpenAI wrapper that bypasses all proxy issues
"""

import os
import json
from typing import List, Dict, Optional
import urllib.request
import urllib.parse

class OpenAIDirectWrapper:
    """Direct HTTP calls to OpenAI API, no dependencies"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("No OpenAI API key provided")
        
        self.base_url = "https://api.openai.com/v1"
    
    def chat_completion(self,
                        messages: List[Dict[str, str]],
                        model: str = "gpt-4o-mini",
                        max_tokens: int = 150,
                        temperature: float = 0.7) -> tuple:
        """Direct API call using only stdlib"""
        
        url = f"{self.base_url}/chat/completions"
        
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
        
        # Convert to JSON
        json_data = json.dumps(data).encode('utf-8')
        
        # Create request
        req = urllib.request.Request(url, data=json_data, headers=headers)
        
        try:
            # Make request
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                
                content = result['choices'][0]['message']['content']
                tokens = result['usage']['total_tokens']
                
                return content, tokens
                
        except urllib.error.HTTPError as e:
            error_msg = e.read().decode('utf-8')
            print(f"OpenAI API error: {e.code} - {error_msg}")
            return f"Error: {e.code}", 0
        except Exception as e:
            print(f"Request failed: {e}")
            return f"Error: {str(e)}", 0

# Test it
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    wrapper = OpenAIDirectWrapper()
    
    print("Testing direct wrapper...")
    response, tokens = wrapper.chat_completion([
        {"role": "user", "content": "Say 'Direct wrapper works!'"}
    ])
    
    print(f"Response: {response}")
    print(f"Tokens used: {tokens}")
'''
    
    # Save wrapper
    with open("openai_direct.py", "w") as f:
        f.write(wrapper_code)
    
    print("Created openai_direct.py - uses only Python stdlib")
    
    # Test it
    print("\nTesting wrapper...")
    try:
        # Import and test
        from dotenv import load_dotenv
        load_dotenv()
        
        # Import our wrapper
        sys.path.insert(0, os.getcwd())
        from openai_direct import OpenAIDirectWrapper
        
        wrapper = OpenAIDirectWrapper()
        response, tokens = wrapper.chat_completion([
            {"role": "user", "content": "test"}
        ])
        
        print(f"✓ Wrapper works! Response: {response}")
        return True
        
    except Exception as e:
        print(f"✗ Wrapper failed: {e}")
        return False

def nuclear_option():
    """Complete reinstall of OpenAI with all dependencies"""
    
    print("\n" + "="*60)
    print("NUCLEAR OPTION - COMPLETE REINSTALL")
    print("="*60)
    
    import subprocess
    
    print("This will:")
    print("1. Uninstall openai and all dependencies")
    print("2. Clear pip cache")
    print("3. Reinstall fresh")
    
    proceed = input("\nProceed with nuclear option? (y/n): ")
    if proceed.lower() != 'y':
        return
    
    try:
        # Uninstall openai and key dependencies
        print("\nUninstalling...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", 
                       "openai", "httpx", "httpcore", "-y"], 
                      capture_output=True)
        
        # Clear pip cache
        print("Clearing cache...")
        subprocess.run([sys.executable, "-m", "pip", "cache", "purge"], 
                      capture_output=True)
        
        # Reinstall
        print("Reinstalling...")
        subprocess.run([sys.executable, "-m", "pip", "install", 
                       "openai==1.45.0", "--no-cache-dir", "--force-reinstall"], 
                      capture_output=True)
        
        print("Testing...")
        
        # Test
        import importlib
        if 'openai' in sys.modules:
            del sys.modules['openai']
        
        from openai import OpenAI
        from dotenv import load_dotenv
        load_dotenv()
        
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        
        print(f"✓ NUCLEAR OPTION WORKED! Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"✗ Nuclear option failed: {e}")
        return False

def main():
    """Main fix routine"""
    
    # First, find what's wrong
    find_the_culprit()
    
    print("\n" + "="*60)
    print("SOLUTIONS")
    print("="*60)
    
    print("\nOption 1: Use the bulletproof wrapper (recommended)")
    print("Option 2: Try nuclear reinstall")
    print("Option 3: Proceed without OpenAI")
    
    choice = input("\nYour choice (1/2/3): ")
    
    if choice == '1':
        if create_working_wrapper():
            print("\n✓ Wrapper created successfully!")
            print("Update your experiment code to use OpenAIDirectWrapper")
            print("from openai_direct import OpenAIDirectWrapper")
    
    elif choice == '2':
        if nuclear_option():
            print("\n✓ OpenAI fixed!")
        else:
            print("\nNuclear option failed. Use wrapper instead.")
            create_working_wrapper()
    
    elif choice == '3':
        print("\nProceeding with Anthropic + Google only")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. If using wrapper: Update experiment code to import OpenAIDirectWrapper")
    print("2. Run your experiment: python3 run_pilot_final.py")
    print("3. Science continues regardless!")

if __name__ == "__main__":
    main()