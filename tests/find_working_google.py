import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Try these exact names from your list
models_to_try = [
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash-002", 
    "gemini-1.5-pro",
    "gemini-2.0-flash",
    "gemini-flash-latest"
]

print("Finding a Google model that actually fucking works...\n")

for model_name in models_to_try:
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Say 'working'")
        print(f"✅ {model_name} WORKS!")
        print(f"   Response: {response.text}")
        break
    except Exception as e:
        print(f"❌ {model_name}: {str(e)[:50]}...")
