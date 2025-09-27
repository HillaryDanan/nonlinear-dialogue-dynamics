"""Direct OpenAI test after upgrade"""
import os
from dotenv import load_dotenv
load_dotenv()

print("Testing OpenAI after upgrade to 1.109.1...")

try:
    from openai import OpenAI
    
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY')
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=10
    )
    
    print(f"✅ OpenAI works!")
    print(f"Response: {response.choices[0].message.content}")
    print(f"Tokens: {response.usage.total_tokens}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nTrying alternative approach...")
    
    # Try with different syntax
    try:
        import openai
        openai.api_key = os.getenv('OPENAI_API_KEY')
        
        # Direct API call
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )
        print(f"✅ Works with alternative approach!")
        print(f"Response: {response.choices[0].message.content}")
    except Exception as e2:
        print(f"❌ Still failing: {e2}")
