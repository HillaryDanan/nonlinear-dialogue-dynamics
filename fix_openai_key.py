"""
Fix OpenAI API key issue
"""

import os
from pathlib import Path
from dotenv import load_dotenv

def diagnose_api_keys():
    """Check all API keys are properly loaded"""
    
    print("API Key Diagnostic")
    print("="*50)
    
    # Try loading .env from different locations
    possible_paths = [
        Path(".env"),
        Path("./.env"),
        Path(__file__).parent / ".env",
        Path.home() / "Desktop/nonlinear-dialogue-dynamics/.env"
    ]
    
    env_found = False
    for path in possible_paths:
        if path.exists():
            print(f"✓ Found .env at: {path}")
            load_dotenv(path)
            env_found = True
            break
    
    if not env_found:
        print("✗ No .env file found!")
        print("\nCreate .env file with:")
        print("OPENAI_API_KEY=sk-...")
        print("ANTHROPIC_API_KEY=sk-ant-...")
        print("GOOGLE_API_KEY=...")
        return False
    
    # Check each API key
    keys_status = {
        "OPENAI_API_KEY": os.getenv('OPENAI_API_KEY'),
        "ANTHROPIC_API_KEY": os.getenv('ANTHROPIC_API_KEY'),
        "GOOGLE_API_KEY": os.getenv('GOOGLE_API_KEY')
    }
    
    print("\nAPI Keys Status:")
    all_good = True
    
    for key_name, key_value in keys_status.items():
        if key_value:
            # Mask the key for security
            masked = key_value[:7] + "..." + key_value[-4:] if len(key_value) > 11 else "***"
            print(f"✓ {key_name}: {masked}")
        else:
            print(f"✗ {key_name}: NOT FOUND")
            all_good = False
    
    # Test OpenAI specifically
    if keys_status["OPENAI_API_KEY"]:
        print("\nTesting OpenAI connection...")
        try:
            from openai import OpenAI
            
            # Try with explicit api_key parameter
            client = OpenAI(api_key=keys_status["OPENAI_API_KEY"])
            
            # Test with actual API call
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say test"}],
                max_tokens=5
            )
            print(f"✓ OpenAI works: {response.choices[0].message.content}")
            
        except Exception as e:
            print(f"✗ OpenAI failed: {e}")
            
            # Try to identify the issue
            if "Invalid API key" in str(e):
                print("\n⚠️  Your OpenAI API key is invalid or expired")
                print("   Get a new one at: https://platform.openai.com/api-keys")
            elif "quota" in str(e).lower():
                print("\n⚠️  You may have exceeded your OpenAI quota")
            else:
                print(f"\n⚠️  Unexpected error: {e}")
    
    return all_good

def create_env_template():
    """Create template .env file"""
    
    template = """# API Keys for Nonlinear Dialogue Dynamics Study

# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic  
ANTHROPIC_API_KEY=sk-ant-...

# Google
GOOGLE_API_KEY=...
"""
    
    env_path = Path(".env.template")
    with open(env_path, 'w') as f:
        f.write(template)
    
    print(f"\nCreated template at: {env_path}")
    print("Copy to .env and add your actual keys")

if __name__ == "__main__":
    if diagnose_api_keys():
        print("\n✓ All API keys are properly configured!")
    else:
        print("\n✗ Some API keys are missing")
        create_env_template()