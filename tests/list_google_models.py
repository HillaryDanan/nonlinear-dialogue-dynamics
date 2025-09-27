import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

print("Available Google models:")
print("="*40)
for model in genai.list_models():
    print(f"Name: {model.name}")
    if 'generateContent' in model.supported_generation_methods:
        print(f"  ✓ Supports generateContent")
    print(f"  Methods: {model.supported_generation_methods}")
    print()
