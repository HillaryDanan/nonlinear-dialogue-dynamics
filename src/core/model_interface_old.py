"""
Unified Model Interface with Robust Error Handling
===================================================
Following software engineering best practices (Martin, 2008)
and defensive programming principles (McConnell, 2004)
"""

import os
import time
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from datetime import datetime
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for each model provider"""
    provider: str
    model_name: str
    max_tokens: int = 150
    temperature: float = 0.7
    rate_limit_delay: float = 1.0
    max_retries: int = 3
    timeout: float = 30.0


class BaseModel(ABC):
    """Abstract base class for all models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.is_initialized = False
        self.total_tokens = 0
        self.total_requests = 0
        self.last_request_time = 0
        
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the model connection"""
        pass
    
    @abstractmethod
    async def _generate_impl(self, prompt: str, history: List[Dict]) -> Tuple[str, int]:
        """Implementation-specific generation"""
        pass
    
    async def generate(self, prompt: str, history: Optional[List[Dict]] = None) -> Tuple[str, int]:
        """Generate with rate limiting and retries"""
        
        # Rate limiting
        elapsed = time.time() - self.last_request_time
        if elapsed < self.config.rate_limit_delay:
            await asyncio.sleep(self.config.rate_limit_delay - elapsed)
        
        # Retry logic
        for attempt in range(self.config.max_retries):
            try:
                response, tokens = await self._generate_impl(prompt, history or [])
                self.last_request_time = time.time()
                self.total_tokens += tokens
                self.total_requests += 1
                return response, tokens
                
            except Exception as e:
                logger.warning(f"{self.config.provider} attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"{self.config.provider} failed after {self.config.max_retries} attempts")
                    return f"Error: {str(e)[:50]}", 0
        
        return "Max retries exceeded", 0


class AnthropicModel(BaseModel):
    """Anthropic Claude implementation"""
    
    async def initialize(self) -> bool:
        try:
            import anthropic
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                logger.error("ANTHROPIC_API_KEY not found")
                return False
                
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
            self.is_initialized = True
            logger.info(f"Anthropic initialized: {self.config.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Anthropic initialization failed: {e}")
            return False
    
    async def _generate_impl(self, prompt: str, history: List[Dict]) -> Tuple[str, int]:
        messages = []
        
        # Include relevant history
        for h in history[-6:]:
            if 'user' in h:
                messages.append({"role": "user", "content": h['user'][:300]})
            if 'assistant' in h:
                messages.append({"role": "assistant", "content": h['assistant'][:300]})
        
        messages.append({"role": "user", "content": prompt})
        
        response = await self.client.messages.create(
            model=self.config.model_name,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )
        
        text = response.content[0].text
        tokens = getattr(response.usage, 'total_tokens', len(text.split()) * 2)
        
        return text, tokens


class GoogleModel(BaseModel):
    """Google Gemini implementation"""
    
    async def initialize(self) -> bool:
        try:
            import google.generativeai as genai
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                logger.error("GOOGLE_API_KEY not found")
                return False
            
            genai.configure(api_key=api_key)
            # Use the FULL model path
            self.model = genai.GenerativeModel('models/gemini-1.5-flash')
            self.is_initialized = True
            logger.info(f"Google initialized: {self.config.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Google initialization failed: {e}")
            return False
    
    async def _generate_impl(self, prompt: str, history: List[Dict]) -> Tuple[str, int]:
        # Build context string
        context_parts = []
        for h in history[-4:]:
            if 'user' in h:
                context_parts.append(f"Human: {h['user'][:200]}")
            if 'assistant' in h:
                context_parts.append(f"Assistant: {h['assistant'][:200]}")
        
        full_prompt = "\n".join(context_parts + [f"Human: {prompt}\nAssistant:"])
        
        # Generate (Gemini is synchronous, so we run in executor)
        import asyncio
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            self.model.generate_content,
            full_prompt
        )
        
        text = response.text
        tokens = len(full_prompt.split()) + len(text.split())
        
        return text, int(tokens * 1.3)


class OpenAIModel(BaseModel):
    """OpenAI GPT implementation"""
    
    async def initialize(self) -> bool:
        try:
            import openai
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.error("OPENAI_API_KEY not found")
                return False
            
            self.client = openai.AsyncOpenAI(api_key=api_key)
            self.is_initialized = True
            logger.info(f"OpenAI initialized: {self.config.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"OpenAI initialization failed: {e}")
            return False
    
    async def _generate_impl(self, prompt: str, history: List[Dict]) -> Tuple[str, int]:
        messages = []
        
        # System message for consistency
        messages.append({
            "role": "system",
            "content": "You are a helpful assistant engaged in a research study."
        })
        
        # Include history
        for h in history[-4:]:
            if 'user' in h:
                messages.append({"role": "user", "content": h['user'][:300]})
            if 'assistant' in h:
                messages.append({"role": "assistant", "content": h['assistant'][:300]})
        
        messages.append({"role": "user", "content": prompt})
        
        response = await self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )
        
        text = response.choices[0].message.content
        tokens = response.usage.total_tokens
        
        return text, tokens


class ModelFactory:
    """Factory for creating model instances"""
    
    CONFIGS = {
        'anthropic': ModelConfig(
            provider='anthropic',
            model_name='claude-3-5-haiku-20241022',
            rate_limit_delay=1.0
        ),
        'google': ModelConfig(
            provider='google',
            model_name='gemini-1.5-flash',
            rate_limit_delay=0.5
        ),
        'openai': ModelConfig(
            provider='openai',
            model_name='gpt-4o-mini',
            rate_limit_delay=0.5
        )
    }
    
    @classmethod
    async def create(cls, provider: str) -> Optional[BaseModel]:
        """Create and initialize a model"""
        
        if provider not in cls.CONFIGS:
            logger.error(f"Unknown provider: {provider}")
            return None
        
        config = cls.CONFIGS[provider]
        
        if provider == 'anthropic':
            model = AnthropicModel(config)
        elif provider == 'google':
            model = GoogleModel(config)
        elif provider == 'openai':
            model = OpenAIModel(config)
        else:
            return None
        
        if await model.initialize():
            return model
        
        return None


# Test function
async def test_models():
    """Test all three models"""
    
    print("\n" + "="*60)
    print("TESTING MODEL IMPLEMENTATIONS")
    print("="*60)
    
    test_prompt = "What is the capital of France?"
    
    for provider in ['anthropic', 'google', 'openai']:
        print(f"\nTesting {provider}...")
        model = await ModelFactory.create(provider)
        
        if model:
            try:
                response, tokens = await model.generate(test_prompt)
                print(f"✓ {provider}: {response[:100]}...")
                print(f"  Tokens: {tokens}")
            except Exception as e:
                print(f"✗ {provider} failed: {e}")
        else:
            print(f"✗ {provider}: Could not initialize")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_models())
