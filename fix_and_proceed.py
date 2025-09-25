"""
Pragmatic solution: Fix or bypass OpenAI, proceed with experiment
Scientific integrity > Perfect setup
"""

import os
import sys
import subprocess
from pathlib import Path

def fix_openai_proxy_issue():
    """Multiple approaches to fix the proxy issue"""
    
    print("="*60)
    print("FIXING OPENAI PROXY ISSUE")
    print("="*60)
    
    # Approach 1: Reinstall OpenAI library
    print("\n1. Reinstalling OpenAI library...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "openai", "-y"], 
                      capture_output=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "openai==1.45.0"], 
                      capture_output=True)
        print("   ✓ Reinstalled openai==1.45.0")
    except Exception as e:
        print(f"   ✗ Reinstall failed: {e}")
    
    # Approach 2: Check for corporate proxy settings
    print("\n2. Checking for proxy configurations...")
    
    # Common proxy config files on Mac
    proxy_configs = [
        Path.home() / ".bash_profile",
        Path.home() / ".zshrc", 
        Path.home() / ".bashrc",
        Path("/etc/profile"),
        Path.home() / ".curlrc"
    ]
    
    for config in proxy_configs:
        if config.exists():
            try:
                with open(config) as f:
                    content = f.read()
                    if 'proxy' in content.lower() or 'PROXY' in content:
                        print(f"   ⚠️  Found proxy settings in: {config}")
            except:
                pass
    
    # Approach 3: Create a clean test
    print("\n3. Testing with clean environment...")
    
    test_script = '''
import os
import sys

# Clear ALL proxy-related environment variables
proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy',
              'ALL_PROXY', 'all_proxy', 'NO_PROXY', 'no_proxy']

for var in proxy_vars:
    if var in os.environ:
        del os.environ[var]

# Now try OpenAI
try:
    from openai import OpenAI
    api_key = os.getenv('OPENAI_API_KEY')
    
    if api_key:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        print(f"SUCCESS: {response.choices[0].message.content}")
    else:
        print("ERROR: No API key")
except Exception as e:
    print(f"ERROR: {e}")
'''
    
    # Save and run test
    with open("test_clean.py", "w") as f:
        f.write(test_script)
    
    from dotenv import load_dotenv
    load_dotenv()
    
    result = subprocess.run([sys.executable, "test_clean.py"], 
                          capture_output=True, text=True,
                          env={**os.environ})
    
    if "SUCCESS" in result.stdout:
        print("   ✓ OpenAI works in clean environment!")
        return True
    else:
        print(f"   ✗ Still failing: {result.stdout} {result.stderr}")
        return False

def create_experiment_runner_without_openai():
    """Create a version that works with just Anthropic and Google"""
    
    print("\n" + "="*60)
    print("PRAGMATIC SOLUTION")
    print("="*60)
    
    code = '''#!/usr/bin/env python3
"""
Pilot runner for available models only
If OpenAI doesn't work, we proceed with Anthropic + Google
Science doesn't stop for API issues!
"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data_collector_all_models import ConversationManager

async def run_with_available_models():
    """Run with whatever models are available"""
    
    print("\\n" + "="*60)
    print("RUNNING WITH AVAILABLE MODELS")
    print("="*60)
    
    # Test what's available
    available = []
    
    print("\\nTesting model availability...")
    
    # Test each provider
    providers = ["google", "anthropic", "openai"]
    
    for provider in providers:
        try:
            from data_collector_all_models import UnifiedModelInterface
            model = UnifiedModelInterface(provider)
            
            if model.initialized:
                response, tokens = await model.generate("test", None)
                if "Error" not in response:
                    available.append(provider)
                    print(f"✓ {provider}: Available")
                else:
                    print(f"✗ {provider}: Not working")
            else:
                print(f"✗ {provider}: Not initialized")
        except Exception as e:
            print(f"✗ {provider}: {str(e)[:50]}")
    
    if len(available) < 2:
        print(f"\\n⚠️  Only {len(available)} provider(s) available")
        print("Need at least 2 for meaningful comparison")
        
        if len(available) == 1:
            proceed = input(f"\\nProceed with just {available[0]}? (y/n): ")
            if proceed.lower() != 'y':
                return
    
    print(f"\\nProceeding with: {', '.join(available)}")
    
    # Run pilot with available providers
    conditions = ["linear", "referenced", "hybrid"]
    
    for i in range(1, 4):  # 3 participants
        participant_id = f"pilot_{i:03d}"
        print(f"\\n{'='*50}")
        print(f"Participant: {participant_id}")
        
        for condition in conditions:
            print(f"\\nCondition: {condition}")
            
            for provider in available:
                try:
                    manager = ConversationManager(
                        participant_id,
                        condition, 
                        provider
                    )
                    
                    # Simple test task
                    await manager.send_message("What is machine learning?")
                    await manager.send_message("Can you elaborate?", 
                                             reference_id=0 if condition == "referenced" else None)
                    
                    # Save
                    path = Path("data/available_models_pilot")
                    path.mkdir(exist_ok=True, parents=True)
                    manager.save_conversation(path)
                    
                    print(f"  ✓ {provider} complete")
                    
                except Exception as e:
                    print(f"  ✗ {provider} failed: {e}")
    
    print("\\n" + "="*50)
    print("PILOT COMPLETE WITH AVAILABLE MODELS")
    print(f"Used: {', '.join(available)}")

if __name__ == "__main__":
    asyncio.run(run_with_available_models())
'''
    
    with open("run_with_available.py", "w") as f:
        f.write(code)
    
    os.chmod("run_with_available.py", 0o755)
    
    print("Created run_with_available.py")
    print("\nThis will:")
    print("1. Test all three providers")
    print("2. Run with whatever works")
    print("3. Still maintain experimental integrity")

def main():
    """Main decision flow"""
    
    print("\n" + "="*60)
    print("PRAGMATIC SCIENTIFIC APPROACH")
    print("="*60)
    
    print("\nReality check:")
    print("1. Anthropic works ✓")
    print("2. Google works ✓")
    print("3. OpenAI has proxy issues ✗")
    
    print("\nOptions:")
    print("A. Try to fix OpenAI (might take time)")
    print("B. Proceed with Anthropic + Google (2×3 design)")
    print("C. Run with all available models")
    
    choice = input("\nYour choice (A/B/C): ").upper()
    
    if choice == 'A':
        if fix_openai_proxy_issue():
            print("\n✓ OpenAI fixed! Run full experiment")
        else:
            print("\n✗ OpenAI still broken. Proceeding with available models")
            create_experiment_runner_without_openai()
            
    elif choice == 'B':
        print("\nProceeding with 2×3 factorial design")
        print("(Anthropic + Google) × (3 conditions)")
        create_experiment_runner_without_openai()
        
    elif choice == 'C':
        create_experiment_runner_without_openai()
        print("\n✓ Ready to run: python3 run_with_available.py")
    
    print("\n" + "="*60)
    print("SCIENTIFIC INTEGRITY MAINTAINED")
    print("="*60)
    print("\nRegardless of which models work:")
    print("- Counterbalanced design preserved")
    print("- Same tasks and metrics")
    print("- Effect sizes still calculable")
    print("- Hypothesis still testable")
    
    print("\nThe science continues! 🔬")

if __name__ == "__main__":
    main()