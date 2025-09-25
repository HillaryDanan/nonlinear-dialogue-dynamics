"""
Data collection module with corrected models (September 2025)
"""

import json
import time
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import hashlib
import random
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Message:
    """Single message in conversation"""
    id: int
    role: str  # 'user' or 'assistant'
    content: str
    reference_id: Optional[int] = None
    timestamp: datetime = None
    tokens: Optional[int] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class ModelInterface:
    """Unified interface for different models - UPDATED FOR 2025"""
    
    def __init__(self, model_type: str):
        self.model_type = model_type
        self._initialize_client()
        
    def _initialize_client(self):
        if self.model_type in ["gpt-3.5", "gpt-4o-mini"]:
            import openai
            self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            # Use gpt-4o-mini by default (cheaper, better than 3.5)
            self.model_name = "gpt-4o-mini"
            
        elif self.model_type in ["claude", "claude-sonnet"]:
            import anthropic
            self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
            # Use current sonnet model
            self.model_name = "claude-3-5-sonnet-20241022"
            
        elif self.model_type == "claude-haiku":
            import anthropic
            self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
            # Cheaper, faster option
            self.model_name = "claude-3-5-haiku-20241022"
            
        elif self.model_type in ["gemini", "gemini-flash"]:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
            # Use flash by default (cheaper)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            
        elif self.model_type == "gemini-pro":
            import google.generativeai as genai
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
            self.model = genai.GenerativeModel('gemini-1.5-pro')
            
    async def generate(self, 
                      prompt: str, 
                      history: List[Message] = None) -> Tuple[str, int]:
        """Generate response and return (text, token_count)"""
        
        if self.model_type in ["gpt-3.5", "gpt-4o-mini"]:
            messages = self._format_messages_openai(prompt, history)
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=150
                )
                return response.choices[0].message.content, response.usage.total_tokens
            except Exception as e:
                print(f"OpenAI error: {e}")
                return f"Error: {e}", 0
            
        elif self.model_type in ["claude", "claude-sonnet", "claude-haiku"]:
            messages = self._format_messages_anthropic(prompt, history)
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=150,
                    temperature=0.7
                )
                # Handle both old and new response formats
                if hasattr(response, 'content') and response.content:
                    text = response.content[0].text if hasattr(response.content[0], 'text') else str(response.content[0])
                else:
                    text = str(response)
                    
                # Get token count if available
                tokens = 0
                if hasattr(response, 'usage'):
                    tokens = getattr(response.usage, 'input_tokens', 0) + getattr(response.usage, 'output_tokens', 0)
                    
                return text, tokens
            except Exception as e:
                print(f"Anthropic error: {e}")
                return f"Error: {e}", 0
            
        elif self.model_type in ["gemini", "gemini-flash", "gemini-pro"]:
            try:
                # Gemini doesn't use message history the same way
                full_prompt = prompt
                if history:
                    # Include some recent context
                    recent_context = "\n".join([f"{m.role}: {m.content}" for m in history[-4:]])
                    full_prompt = f"Previous context:\n{recent_context}\n\nUser: {prompt}"
                    
                response = self.model.generate_content(full_prompt)
                # Estimate tokens (Gemini doesn't provide counts easily)
                estimated_tokens = len(full_prompt.split()) * 1.3 + len(response.text.split()) * 1.3
                return response.text, int(estimated_tokens)
            except Exception as e:
                print(f"Gemini error: {e}")
                return f"Error: {e}", 0
    
    def _format_messages_openai(self, prompt: str, history: List[Message]) -> List[Dict]:
        """Format for OpenAI API"""
        messages = []
        if history:
            # Only include last 10 messages to avoid context length issues
            recent_history = history[-10:]
            for msg in recent_history:
                messages.append({
                    "role": msg.role if msg.role != "assistant" else "assistant",
                    "content": msg.content
                })
        messages.append({"role": "user", "content": prompt})
        return messages
        
    def _format_messages_anthropic(self, prompt: str, history: List[Message]) -> List[Dict]:
        """Format for Anthropic API"""
        messages = []
        if history:
            # Only include last 10 messages
            recent_history = history[-10:]
            for msg in recent_history:
                role = "user" if msg.role == "user" else "assistant"
                messages.append({"role": role, "content": msg.content})
        messages.append({"role": "user", "content": prompt})
        return messages

class ConversationManager:
    """Manages experimental conversations"""
    
    def __init__(self, participant_id: str, condition: str, model_type: str):
        self.participant_id = participant_id
        self.condition = condition
        self.model = ModelInterface(model_type)
        self.history: List[Message] = []
        self.current_id = 0
        
    def create_prompt(self, 
                     user_input: str, 
                     reference_id: Optional[int] = None) -> str:
        """Create prompt based on condition"""
        
        if self.condition == "linear":
            # Standard prompt
            return user_input
            
        elif self.condition == "referenced" and reference_id is not None:
            # Find referenced message
            ref_msg = next((m for m in self.history if m.id == reference_id), None)
            if ref_msg:
                return f"""Previous statement (Message #{reference_id}): "{ref_msg.content}"

User is now responding to that specific statement with: "{user_input}"

Please provide a response that directly addresses the referenced message."""
            
        elif self.condition == "hybrid":
            # Mix of both - randomly decide
            if reference_id and random.random() > 0.5:
                ref_msg = next((m for m in self.history if m.id == reference_id), None)
                if ref_msg:
                    return f"Referring to '{ref_msg.content[:50]}...': {user_input}"
            return user_input
            
        return user_input
    
    async def send_message(self, 
                          user_input: str,
                          reference_id: Optional[int] = None) -> Message:
        """Send message and get response"""
        
        # Create user message
        user_msg = Message(
            id=self.current_id,
            role="user",
            content=user_input,
            reference_id=reference_id
        )
        self.history.append(user_msg)
        self.current_id += 1
        
        # Generate prompt based on condition
        prompt = self.create_prompt(user_input, reference_id)
        
        # Get response with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response_text, tokens = await self.model.generate(prompt, self.history)
                break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    response_text = "Error: Could not generate response"
                    tokens = 0
                else:
                    await asyncio.sleep(2)  # Wait before retry
        
        # Create assistant message
        assistant_msg = Message(
            id=self.current_id,
            role="assistant",
            content=response_text,
            tokens=tokens
        )
        self.history.append(assistant_msg)
        self.current_id += 1
        
        return assistant_msg
    
    def save_conversation(self, path: Path):
        """Save conversation data"""
        data = {
            "participant_id": self.participant_id,
            "condition": self.condition,
            "model": self.model.model_type,
            "model_name": self.model.model_name if hasattr(self.model, 'model_name') else "unknown",
            "timestamp": datetime.now().isoformat(),
            "messages": [asdict(msg) for msg in self.history]
        }
        
        filename = f"{self.participant_id}_{self.condition}_{self.model.model_type}.json"
        with open(path / filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"Saved: {filename}")