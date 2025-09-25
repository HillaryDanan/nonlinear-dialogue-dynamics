"""
Comprehensive data collection for all three model providers
Following systematic experimental design principles (Kirk, 2013)
"""

import json
import time
import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import random
from dotenv import load_dotenv
import traceback

# Load environment variables
load_dotenv()

@dataclass
class Message:
    """Single message in conversation"""
    id: int
    role: str
    content: str
    reference_id: Optional[int] = None
    timestamp: datetime = None
    tokens: Optional[int] = None
    model_used: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class UnifiedModelInterface:
    """
    Unified interface for all three providers
    Ensures systematic comparison across models (Maxwell & Delaney, 2004)
    """
    
    # Model registry with September 2025 versions
    MODELS = {
        "openai": {
            "primary": "gpt-4o-mini",
            "fallback": "gpt-3.5-turbo",
            "provider": "OpenAI"
        },
        "anthropic": {
            "primary": "claude-3-5-haiku-20241022",
            "fallback": "claude-3-5-sonnet-20241022", 
            "provider": "Anthropic"
        },
        "google": {
            "primary": "gemini-2.5-flash",
            "fallback": "gemini-1.5-flash",
            "provider": "Google"
        }
    }
    
    def __init__(self, provider: str):
        """
        Initialize with specified provider
        provider: 'openai', 'anthropic', or 'google'
        """
        self.provider = provider.lower()
        self.model_name = None
        self.client = None
        self.initialized = False
        
        if self.provider not in self.MODELS:
            raise ValueError(f"Unknown provider: {provider}")
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate client"""
        
        try:
            if self.provider == "openai":
                self._init_openai()
            elif self.provider == "anthropic":
                self._init_anthropic()
            elif self.provider == "google":
                self._init_google()
                
            self.initialized = True
            print(f"✓ Initialized {self.provider} with {self.model_name}")
            
        except Exception as e:
            print(f"✗ Failed to initialize {self.provider}: {e}")
            self.initialized = False
    
    def _init_openai(self):
        """Initialize OpenAI client"""
        from openai import OpenAI
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        # Explicit api_key parameter to avoid the issue
        self.client = OpenAI(api_key=api_key)
        self.model_name = self.MODELS["openai"]["primary"]
        
        # Test the connection
        test = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
    
    def _init_anthropic(self):
        """Initialize Anthropic client"""
        import anthropic
        
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = self.MODELS["anthropic"]["primary"]
    
    def _init_google(self):
        """Initialize Google client"""
        import google.generativeai as genai
        
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")
        
        genai.configure(api_key=api_key)
        self.model_name = self.MODELS["google"]["primary"]
        self.client = genai.GenerativeModel(self.model_name)
    
    async def generate(self, 
                      prompt: str, 
                      history: List[Message] = None,
                      max_retries: int = 3) -> Tuple[str, int]:
        """
        Generate response with automatic retry and fallback
        Returns: (response_text, token_count)
        """
        
        if not self.initialized:
            return "Model not initialized", 0
        
        for attempt in range(max_retries):
            try:
                if self.provider == "openai":
                    return await self._generate_openai(prompt, history)
                elif self.provider == "anthropic":
                    return await self._generate_anthropic(prompt, history)
                elif self.provider == "google":
                    return await self._generate_google(prompt, history)
                    
            except Exception as e:
                print(f"  Attempt {attempt + 1} failed: {e}")
                
                if attempt == max_retries - 1:
                    # Try fallback model
                    if attempt == 0:  # Only try fallback once
                        print(f"  Trying fallback model...")
                        self.model_name = self.MODELS[self.provider]["fallback"]
                    else:
                        return f"Error after {max_retries} attempts", 0
                
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return "Failed to generate response", 0
    
    async def _generate_openai(self, prompt: str, history: List[Message]) -> Tuple[str, int]:
        """Generate with OpenAI"""
        messages = []
        
        if history:
            # Include recent context (last 10 messages)
            for msg in history[-10:]:
                role = "user" if msg.role == "user" else "assistant"
                messages.append({"role": role, "content": msg.content})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=150
        )
        
        return response.choices[0].message.content, response.usage.total_tokens
    
    async def _generate_anthropic(self, prompt: str, history: List[Message]) -> Tuple[str, int]:
        """Generate with Anthropic"""
        messages = []
        
        if history:
            for msg in history[-10:]:
                role = "user" if msg.role == "user" else "assistant"
                messages.append({"role": role, "content": msg.content})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.messages.create(
            model=self.model_name,
            messages=messages,
            max_tokens=150,
            temperature=0.7
        )
        
        text = response.content[0].text if hasattr(response.content[0], 'text') else str(response.content[0])
        tokens = 0
        
        if hasattr(response, 'usage'):
            tokens = getattr(response.usage, 'input_tokens', 0) + getattr(response.usage, 'output_tokens', 0)
        
        return text, tokens
    
    async def _generate_google(self, prompt: str, history: List[Message]) -> Tuple[str, int]:
        """Generate with Google"""
        
        # Build context
        context = ""
        if history:
            recent = history[-8:]
            context = "\n".join([f"{m.role}: {m.content}" for m in recent])
            full_prompt = f"Previous conversation:\n{context}\n\nUser: {prompt}"
        else:
            full_prompt = prompt
        
        response = self.client.generate_content(full_prompt)
        
        # Estimate tokens
        tokens = int((len(full_prompt.split()) + len(response.text.split())) * 1.3)
        
        return response.text, tokens

class ConversationManager:
    """
    Manages experimental conversations across all providers
    Implements counterbalanced design (Kirk, 2013)
    """
    
    def __init__(self, participant_id: str, condition: str, provider: str):
        self.participant_id = participant_id
        self.condition = condition
        self.provider = provider
        self.model = UnifiedModelInterface(provider)
        self.history: List[Message] = []
        self.current_id = 0
        
    def create_prompt(self, 
                     user_input: str, 
                     reference_id: Optional[int] = None) -> str:
        """
        Create condition-specific prompts
        Ensuring equal information content across conditions (Miller & Chapman, 2001)
        """
        
        if self.condition == "linear":
            # Standard sequential prompt
            return user_input
            
        elif self.condition == "referenced" and reference_id is not None:
            # Explicit reference to earlier message
            ref_msg = next((m for m in self.history if m.id == reference_id), None)
            if ref_msg:
                return f"""Previous statement (Message #{reference_id}): "{ref_msg.content}"

User is now responding to that specific statement with: "{user_input}"

Please provide a response that directly addresses the referenced message."""
            
        elif self.condition == "hybrid":
            # Mixed approach - 50% probability of reference
            if reference_id and random.random() > 0.5:
                ref_msg = next((m for m in self.history if m.id == reference_id), None)
                if ref_msg:
                    preview = ref_msg.content[:60] + "..." if len(ref_msg.content) > 60 else ref_msg.content
                    return f"Referring to earlier point about '{preview}':\n{user_input}"
                    
        return user_input
    
    async def send_message(self, 
                          user_input: str,
                          reference_id: Optional[int] = None) -> Message:
        """Send message and get response"""
        
        # Record user message
        user_msg = Message(
            id=self.current_id,
            role="user",
            content=user_input,
            reference_id=reference_id,
            model_used=self.provider
        )
        self.history.append(user_msg)
        self.current_id += 1
        
        # Generate condition-appropriate prompt
        prompt = self.create_prompt(user_input, reference_id)
        
        # Get model response
        start_time = time.time()
        response_text, tokens = await self.model.generate(prompt, self.history)
        latency = time.time() - start_time
        
        # Record assistant message
        assistant_msg = Message(
            id=self.current_id,
            role="assistant",
            content=response_text,
            tokens=tokens,
            model_used=f"{self.provider}:{self.model.model_name}"
        )
        
        # Add latency as metadata
        assistant_msg.latency = latency
        
        self.history.append(assistant_msg)
        self.current_id += 1
        
        return assistant_msg
    
    def save_conversation(self, path: Path) -> Path:
        """Save conversation with full experimental metadata"""
        
        path.mkdir(exist_ok=True, parents=True)
        
        # Calculate conversation metrics
        metrics = {
            "total_messages": len(self.history),
            "total_tokens": sum(m.tokens or 0 for m in self.history),
            "n_references": sum(1 for m in self.history if m.reference_id is not None),
            "avg_latency": np.mean([m.latency for m in self.history if hasattr(m, 'latency')]),
            "provider": self.provider,
            "model_used": self.model.model_name if self.model.initialized else "failed"
        }
        
        data = {
            "experiment_version": "1.0",
            "participant_id": self.participant_id,
            "condition": self.condition,
            "provider": self.provider,
            "timestamp": datetime.now().isoformat(),
            "messages": [asdict(msg) for msg in self.history],
            "metrics": metrics
        }
        
        filename = f"{self.participant_id}_{self.condition}_{self.provider}.json"
        filepath = path / filename
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return filepath

# Import numpy for metrics
try:
    import numpy as np
except ImportError:
    print("Warning: numpy not available for metrics calculation")
    np = None