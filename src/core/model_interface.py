"""
Unified Model Interface with Robust Error Handling
===================================================
FIXED VERSION WITH PROPER ENV LOADING
"""

import os
import sys
import time
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from datetime import datetime
import asyncio
from pathlib import Path

# CRITICAL FIX: Load .env file BEFORE anything else
from dotenv import load_dotenv

# Try multiple locations for .env file
env_locations = [
    Path.cwd() / '.env',
    Path(__file__).parent.parent.parent / '.env',
    Path('/Users/hillarylevinson/Desktop/nonlinear-dialogue-dynamics/.env')
]

env_loaded = False
for env_path in env_locations:
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded .env from: {env_path}")
        env_loaded = True
        break

if not env_loaded:
    print("⚠️ Warning: No .env file found, trying system environment variables")

# Now verify keys are loaded
print("\nAPI Key Status:")
print(f"  ANTHROPIC_API_KEY: {'✓ Found' if os.getenv('ANTHROPIC_API_KEY') else '✗ Missing'}")
print(f"  GOOGLE_API_KEY: {'✓ Found' if os.getenv('GOOGLE_API_KEY') else '✗ Missing'}")
print(f"  OPENAI_API_KEY: {'✓ Found' if os.getenv('OPENAI_API_KEY') else '✗ Missing'}")
print()

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
        
        # Retry logic with exponential backoff
        for attempt in range(self.config.max_retries):
            try:
                response, tokens = await self._generate_impl(prompt, history or [])
                self.last_request_time = time.time()
                self.total_tokens += tokens
                self.total_requests += 1
                return response, tokens
                
            except Exception as e:
                logger.warning(f"{self.config.provider} attempt {attempt + 1}/{self.config.max_retries} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    wait_time = min(2 ** attempt, 10)  # Cap at 10 seconds
                    logger.info(f"Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"{self.config.provider} failed after {self.config.max_retries} attempts")
                    return f"Error after {self.config.max_retries} attempts: {str(e)[:50]}", 0
        
        return "Max retries exceeded", 0


class AnthropicModel(BaseModel):
    """Anthropic Claude implementation"""
    
    async def initialize(self) -> bool:
        try:
            import anthropic
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                logger.error("ANTHROPIC_API_KEY not found in environment")
                return False
            
            # Check key format
            if not api_key.startswith('sk-'):
                logger.warning("ANTHROPIC_API_KEY doesn't start with 'sk-', might be invalid")
                
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
            self.is_initialized = True
            logger.info(f"✓ Anthropic initialized: {self.config.model_name}")
            
            # Test the connection
            try:
                test_response = await self.client.messages.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": "Test"}],
                    max_tokens=10
                )
                logger.info("✓ Anthropic connection verified")
            except Exception as e:
                logger.warning(f"Anthropic test failed: {e}")
                
            return True
            
        except Exception as e:
            logger.error(f"Anthropic initialization failed: {e}")
            return False
    
    async def _generate_impl(self, prompt: str, history: List[Dict]) -> Tuple[str, int]:
        messages = []
        
        # Build conversation history
        for h in history[-6:]:  # Keep last 6 exchanges for context
            if 'user' in h:
                messages.append({"role": "user", "content": h['user'][:500]})
            if 'assistant' in h:
                messages.append({"role": "assistant", "content": h['assistant'][:500]})
        
        messages.append({"role": "user", "content": prompt})
        
        response = await self.client.messages.create(
            model=self.config.model_name,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )
        
        text = response.content[0].text if hasattr(response.content[0], 'text') else str(response.content[0])
        
        # Get token count
        tokens = 0
        if hasattr(response, 'usage'):
            tokens = response.usage.total_tokens
        else:
            # Estimate
            tokens = len(prompt.split()) + len(text.split())
        
        return text, tokens


class GoogleModel(BaseModel):
    """Google Gemini implementation - FIXED"""
    
    async def initialize(self) -> bool:
        try:
            import google.generativeai as genai
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                logger.error("GOOGLE_API_KEY not found in environment")
                return False
            
            genai.configure(api_key=api_key)
            
            # CRITICAL FIX: Use proper model name
            # Try both formats
            try:
                self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
                logger.info("Using gemini-1.5-flash-latest")
            except:
                try:
                    self.model = genai.GenerativeModel('models/gemini-1.5-flash')
                    logger.info("Using models/gemini-1.5-flash")
                except:
                    self.model = genai.GenerativeModel('gemini-pro')
                    logger.info("Falling back to gemini-pro")
            
            self.is_initialized = True
            logger.info(f"✓ Google initialized: {self.config.model_name}")
            
            # Test connection
            try:
                test = self.model.generate_content("Test")
                logger.info("✓ Google connection verified")
            except Exception as e:
                logger.warning(f"Google test failed: {e}")
                
            return True
            
        except Exception as e:
            logger.error(f"Google initialization failed: {e}")
            return False
    
    async def _generate_impl(self, prompt: str, history: List[Dict]) -> Tuple[str, int]:
        # Build conversation context
        context_parts = []
        
        for h in history[-4:]:  # Google has smaller context window
            if 'user' in h:
                context_parts.append(f"Human: {h['user'][:300]}")
            if 'assistant' in h:
                context_parts.append(f"Assistant: {h['assistant'][:300]}")
        
        if context_parts:
            full_prompt = "\n".join(context_parts) + f"\nHuman: {prompt}\nAssistant:"
        else:
            full_prompt = prompt
        
        # Gemini is synchronous, run in executor
        import asyncio
        loop = asyncio.get_event_loop()
        
        try:
            response = await loop.run_in_executor(
                None,
                self.model.generate_content,
                full_prompt,
                dict(
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_tokens
                )
            )
            
            text = response.text
            # Estimate tokens
            tokens = len(full_prompt.split()) + len(text.split())
            
            return text, int(tokens * 1.3)
            
        except Exception as e:
            logger.error(f"Google generation error: {e}")
            raise


class OpenAIModel(BaseModel):
    """OpenAI GPT implementation - FIXED"""
    
    async def initialize(self) -> bool:
        try:
            import openai
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.error("OPENAI_API_KEY not found in environment")
                return False
            
            # Check key format
            if not api_key.startswith('sk-'):
                logger.warning("OPENAI_API_KEY doesn't start with 'sk-', might be invalid")
            
            self.client = openai.AsyncOpenAI(api_key=api_key)
            self.is_initialized = True
            logger.info(f"✓ OpenAI initialized: {self.config.model_name}")
            
            # Test connection
            try:
                test = await self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": "Test"}],
                    max_tokens=10
                )
                logger.info("✓ OpenAI connection verified")
            except Exception as e:
                logger.warning(f"OpenAI test failed: {e}")
                
            return True
            
        except Exception as e:
            logger.error(f"OpenAI initialization failed: {e}")
            return False
    
    async def _generate_impl(self, prompt: str, history: List[Dict]) -> Tuple[str, int]:
        messages = [
            {"role": "system", "content": "You are a helpful assistant participating in a research study."}
        ]
        
        # Add history
        for h in history[-4:]:  # Keep conversation manageable
            if 'user' in h:
                messages.append({"role": "user", "content": h['user'][:500]})
            if 'assistant' in h:
                messages.append({"role": "assistant", "content": h['assistant'][:500]})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                presence_penalty=0.0,
                frequency_penalty=0.0
            )
            
            text = response.choices[0].message.content
            tokens = response.usage.total_tokens if hasattr(response, 'usage') else len(text.split()) * 2
            
            return text, tokens
            
        except openai.RateLimitError as e:
            logger.warning(f"OpenAI rate limit hit: {e}")
            # Wait longer for rate limits
            await asyncio.sleep(10)
            raise
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise


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
            rate_limit_delay=0.5,
            max_retries=5  # More retries for rate limits
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
        
        success = await model.initialize()
        
        if success:
            return model
        else:
            logger.error(f"Failed to initialize {provider}")
            return None


# Test function
async def test_models():
    """Test all three models"""
    
    print("\n" + "="*60)
    print("TESTING MODEL IMPLEMENTATIONS")
    print("="*60)
    
    test_prompts = [
        "What is 2+2?",
        "Name three primary colors.",
        "Complete this sentence: The capital of France is"
    ]
    
    for provider in ['anthropic', 'google', 'openai']:
        print(f"\n{'='*40}")
        print(f"Testing: {provider.upper()}")
        print('='*40)
        
        model = await ModelFactory.create(provider)
        
        if model and model.is_initialized:
            for i, prompt in enumerate(test_prompts, 1):
                try:
                    print(f"\nTest {i}: {prompt}")
                    response, tokens = await model.generate(prompt)
                    print(f"Response: {response[:100]}...")
                    print(f"Tokens: {tokens}")
                except Exception as e:
                    print(f"✗ Failed: {e}")
            
            print(f"\n✓ {provider} testing complete")
            print(f"  Total requests: {model.total_requests}")
            print(f"  Total tokens: {model.total_tokens}")
        else:
            print(f"✗ {provider}: Could not initialize")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_models())
