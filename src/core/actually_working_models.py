"""
THE ACTUALLY FUCKING WORKING MODELS
====================================
Tested and verified September 26, 2025
"""

import os
import time
import asyncio
from typing import Tuple, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelStatus:
    anthropic: bool = False
    google: bool = False
    openai: bool = False


class WorkingModels:
    """Only what actually works, with proper rate limiting"""
    
    def __init__(self):
        self.status = ModelStatus()
        self.anthropic = None
        self.google = None
        self.openai = None
        
        # Rate limiting trackers
        self.last_openai_time = 0
        self.openai_delay = 0.5  # Start conservative
        
        self._initialize_all()
    
    def _initialize_all(self):
        """Initialize all models"""
        
        # ANTHROPIC - Works perfectly
        try:
            import anthropic
            key = os.getenv('ANTHROPIC_API_KEY')
            if key:
                self.anthropic = anthropic.Anthropic(api_key=key)
                self.status.anthropic = True
                print("✅ Anthropic: Ready")
        except Exception as e:
            print(f"❌ Anthropic: {e}")
        
        # GOOGLE - Use gemini-2.0-flash (confirmed working)
        try:
            import google.generativeai as genai
            key = os.getenv('GOOGLE_API_KEY')
            if key:
                genai.configure(api_key=key)
                self.google = genai.GenerativeModel('gemini-2.0-flash')
                # Quick test
                self.google.generate_content("test")
                self.status.google = True
                print("✅ Google: Ready (gemini-2.0-flash)")
        except Exception as e:
            print(f"❌ Google: {e}")
        
        # OPENAI - With aggressive rate limiting
        try:
            from openai import OpenAI
            key = os.getenv('OPENAI_API_KEY')
            if key:
                self.openai = OpenAI(api_key=key)
                self.status.openai = True
                print("✅ OpenAI: Ready (with rate limiting)")
        except Exception as e:
            print(f"❌ OpenAI: {e}")
    
    def generate_anthropic(self, prompt: str, max_tokens: int = 150) -> Tuple[str, int]:
        """Anthropic generation"""
        if not self.status.anthropic:
            return "Anthropic not available", 0
        
        try:
            response = self.anthropic.messages.create(
                model="claude-3-5-haiku-20241022",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            text = response.content[0].text
            tokens = response.usage.input_tokens + response.usage.output_tokens
            return text, tokens
        except Exception as e:
            return f"Error: {e}", 0
    
    def generate_google(self, prompt: str) -> Tuple[str, int]:
        """Google generation with gemini-2.0-flash"""
        if not self.status.google:
            return "Google not available", 0
        
        try:
            response = self.google.generate_content(prompt)
            text = response.text
            tokens = len(prompt.split()) + len(text.split())
            return text, int(tokens * 1.3)
        except Exception as e:
            return f"Error: {e}", 0
    
    def generate_openai(self, prompt: str, max_tokens: int = 150) -> Tuple[str, int]:
        """OpenAI with aggressive rate limiting for Tier 1"""
        if not self.status.openai:
            return "OpenAI not available", 0
        
        # Enforce rate limit delay
        elapsed = time.time() - self.last_openai_time
        if elapsed < self.openai_delay:
            time.sleep(self.openai_delay - elapsed)
        
        try:
            response = self.openai.chat.completions.create(
                model="gpt-3.5-turbo",  # Cheaper and higher limits than gpt-4o-mini
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            text = response.choices[0].message.content
            tokens = response.usage.total_tokens
            
            self.last_openai_time = time.time()
            # Successful call - can reduce delay slightly
            self.openai_delay = max(0.3, self.openai_delay * 0.95)
            
            return text, tokens
            
        except Exception as e:
            if "rate" in str(e).lower():
                # Rate limit hit - increase delay
                self.openai_delay = min(2.0, self.openai_delay * 1.5)
                print(f"⚠️ OpenAI rate limit - increasing delay to {self.openai_delay:.1f}s")
                time.sleep(self.openai_delay)
                # Retry once
                try:
                    response = self.openai.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens
                    )
                    return response.choices[0].message.content, response.usage.total_tokens
                except:
                    pass
            
            return f"Error: {e}", 0


def test_all():
    """Test everything"""
    print("\n" + "="*60)
    print("TESTING ALL THREE MODELS")
    print("="*60)
    
    models = WorkingModels()
    test_prompt = "What is the capital of France?"
    
    print(f"\nActive models: {sum([models.status.anthropic, models.status.google, models.status.openai])}/3")
    
    # Test each
    if models.status.anthropic:
        response, tokens = models.generate_anthropic(test_prompt, 20)
        print(f"\n✅ Anthropic: {response[:50]}... [{tokens} tokens]")
    
    if models.status.google:
        response, tokens = models.generate_google(test_prompt)
        print(f"✅ Google: {response[:50]}... [{tokens} tokens]")
    
    if models.status.openai:
        print("\nTesting OpenAI with rate limiting...")
        for i in range(3):
            response, tokens = models.generate_openai(test_prompt, 20)
            if "Error" not in response:
                print(f"✅ OpenAI test {i+1}: {response[:30]}... [{tokens} tokens]")
            else:
                print(f"❌ OpenAI test {i+1}: {response}")
            time.sleep(0.5)  # Extra safety


if __name__ == "__main__":
    test_all()
