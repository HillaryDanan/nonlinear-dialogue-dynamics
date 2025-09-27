"""
THE ACTUALLY FUCKING WORKING MODEL INTERFACE
=============================================
Tested and verified - no bullshit
"""

import os
import asyncio
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    provider: str
    model_name: str
    max_tokens: int = 150
    temperature: float = 0.7
    rate_limit_delay: float = 1.0


class WorkingModels:
    """Only the shit that actually works"""
    
    def __init__(self):
        self.anthropic_client = None
        self.google_model = None
        self.openai_client = None
        self._initialize()
    
    def _initialize(self):
        """Initialize each model properly"""
        
        # ANTHROPIC - We know this works
        try:
            import anthropic
            key = os.getenv('ANTHROPIC_API_KEY')
            if key:
                self.anthropic_client = anthropic.Anthropic(api_key=key)
                logger.info("✅ Anthropic ready")
        except Exception as e:
            logger.error(f"❌ Anthropic: {e}")
        
        # GOOGLE - Use the models/ prefix!
        try:
            import google.generativeai as genai
            key = os.getenv('GOOGLE_API_KEY')
            if key:
                genai.configure(api_key=key)
                # Use one of the working models with proper prefix
                self.google_model = genai.GenerativeModel('models/gemini-1.5-flash')
                logger.info("✅ Google ready (gemini-1.5-flash)")
        except Exception as e:
            logger.error(f"❌ Google: {e}")
        
        # OPENAI - Should work with new version
        try:
            from openai import OpenAI
            key = os.getenv('OPENAI_API_KEY')
            if key:
                self.openai_client = OpenAI(api_key=key)
                logger.info("✅ OpenAI ready")
        except Exception as e:
            logger.error(f"❌ OpenAI: {e}")
    
    def generate_anthropic(self, prompt: str, max_tokens: int = 150) -> Tuple[str, int]:
        """Generate with Anthropic"""
        if not self.anthropic_client:
            return "Anthropic not available", 0
        
        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-5-haiku-20241022",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            
            text = response.content[0].text
            # FIXED: Use correct attribute names
            tokens = response.usage.input_tokens + response.usage.output_tokens
            
            return text, tokens
        except Exception as e:
            logger.error(f"Anthropic error: {e}")
            return f"Error: {e}", 0
    
    def generate_google(self, prompt: str) -> Tuple[str, int]:
        """Generate with Google"""
        if not self.google_model:
            return "Google not available", 0
        
        try:
            response = self.google_model.generate_content(prompt)
            text = response.text
            tokens = len(prompt.split()) + len(text.split())
            return text, int(tokens * 1.3)
        except Exception as e:
            logger.error(f"Google error: {e}")
            return f"Error: {e}", 0
    
    def generate_openai(self, prompt: str, max_tokens: int = 150) -> Tuple[str, int]:
        """Generate with OpenAI"""
        if not self.openai_client:
            return "OpenAI not available", 0
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            
            text = response.choices[0].message.content
            tokens = response.usage.total_tokens
            
            return text, tokens
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return f"Error: {e}", 0


async def test_everything():
    """Final test of all models"""
    print("\n" + "="*60)
    print("FINAL TEST - ALL MODELS")
    print("="*60)
    
    models = WorkingModels()
    
    test_prompts = [
        "What is 2+2?",
        "Name the capital of France.",
        "Explain quantum mechanics in one sentence."
    ]
    
    # Test Anthropic
    print("\n🤖 ANTHROPIC:")
    for prompt in test_prompts:
        response, tokens = models.generate_anthropic(prompt, max_tokens=50)
        if "Error" not in response:
            print(f"  ✅ {prompt[:20]}... → {response[:50]}... [{tokens} tokens]")
        else:
            print(f"  ❌ {prompt[:20]}... → {response}")
    
    # Test Google
    print("\n🌐 GOOGLE:")
    for prompt in test_prompts:
        response, tokens = models.generate_google(prompt)
        if "Error" not in response:
            print(f"  ✅ {prompt[:20]}... → {response[:50]}... [{tokens} tokens]")
        else:
            print(f"  ❌ {prompt[:20]}... → {response}")
    
    # Test OpenAI
    print("\n🧠 OPENAI:")
    for prompt in test_prompts:
        response, tokens = models.generate_openai(prompt, max_tokens=50)
        if "Error" not in response:
            print(f"  ✅ {prompt[:20]}... → {response[:50]}... [{tokens} tokens]")
        else:
            print(f"  ❌ {prompt[:20]}... → {response}")
    
    print("\n" + "="*60)
    print("READY FOR FOLLOW-UP STUDIES!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(test_everything())
