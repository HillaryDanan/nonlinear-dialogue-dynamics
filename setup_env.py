"""
Setup and verify .env file for API keys
"""

import os
from pathlib import Path

def setup_env_file():
    """Interactive setup for .env file"""
    
    print("\n" + "="*60)
    print("API KEY SETUP")
    print("="*60)
    
    env_path = Path(".env")
    
    # Check if .env exists
    if env_path.exists():
        print(f"✓ .env file found at: {env_path.absolute()}")
        print("\nCurrent contents (keys masked):")
        
        with open(env_path, 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    if value and len(value) > 10:
                        masked = value[:7] + "..." + value[-4:]
                    else:
                        masked = "NOT SET"
                    print(f"  {key} = {masked}")
        
        update = input("\nUpdate API keys? (y/n): ")
        if update.lower() != 'y':
            return
    else:
        print(f"✗ No .env file found")
        create = input("\nCreate .env file? (y/n): ")
        if create.lower() != 'y':
            print("Cannot proceed without .env file")
            return
    
    # Collect API keys
    print("\nEnter your API keys (or press Enter to skip):")
    
    keys = {}
    
    # OpenAI
    print("\n1. OpenAI API Key")
    print("   Get from: https://platform.openai.com/api-keys")
    openai_key = input("   OPENAI_API_KEY: ").strip()
    if openai_key:
        keys['OPENAI_API_KEY'] = openai_key
    
    # Anthropic
    print("\n2. Anthropic API Key")
    print("   Get from: https://console.anthropic.com/settings/keys")
    anthropic_key = input("   ANTHROPIC_API_KEY: ").strip()
    if anthropic_key:
        keys['ANTHROPIC_API_KEY'] = anthropic_key
    
    # Google
    print("\n3. Google API Key")
    print("   Get from: https://makersuite.google.com/app/apikey")
    google_key = input("   GOOGLE_API_KEY: ").strip()
    if google_key:
        keys['GOOGLE_API_KEY'] = google_key
    
    # Write to .env
    if keys:
        with open(env_path, 'w') as f:
            f.write("# API Keys for Nonlinear Dialogue Dynamics Study\n\n")
            for key, value in keys.items():
                f.write(f"{key}={value}\n")
        
        print(f"\n✓ Saved {len(keys)} keys to .env")
    else:
        print("\n⚠️  No keys entered")
    
    # Test the keys
    print("\n" + "="*60)
    print("TESTING API CONNECTIONS")
    print("="*60)
    
    from dotenv import load_dotenv
    load_dotenv()
    
    # Test OpenAI
    if os.getenv('OPENAI_API_KEY'):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            print("✓ OpenAI connection successful")
        except Exception as e:
            print(f"✗ OpenAI error: {e}")
    else:
        print("⚠️  OpenAI key not set")
    
    # Test Anthropic
    if os.getenv('ANTHROPIC_API_KEY'):
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
            response = client.messages.create(
                model="claude-3-5-haiku-20241022",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            print("✓ Anthropic connection successful")
        except Exception as e:
            print(f"✗ Anthropic error: {e}")
    else:
        print("⚠️  Anthropic key not set")
    
    # Test Google
    if os.getenv('GOOGLE_API_KEY'):
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content("test")
            print("✓ Google connection successful")
        except Exception as e:
            print(f"✗ Google error: {e}")
    else:
        print("⚠️  Google key not set")
    
    print("\n" + "="*60)
    print("Setup complete!")
    print("Next: python3 run_comprehensive_pilot.py test")

if __name__ == "__main__":
    setup_env_file()