"""
Working data collection module - handles API issues gracefully
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
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class ModelInterface:
    """Unified interface - focusing on what works"""
    
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.model_name = None
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize only working models"""
        
        if self.model_type.startswith("gemini"):
            import google.generativeai as genai
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
            
            # Map to specific model names
            model_map = {
                "gemini-2.5-flash": "gemini-2.5-flash",
                "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
                "gemini-2.0-flash": "gemini-2.0-flash",
                "gemini-1.5-flash": "gemini-1.5-flash",
                "gemini": "gemini-2.5-flash"  # Default to newest
            }
            
            self.model_name = model_map.get(self.model_type, "gemini-2.5-flash")
            self.model = genai.GenerativeModel(self.model_name)
            print(f"Initialized {self.model_name}")
            
        elif self.model_type.startswith("claude"):
            import anthropic
            self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
            
            if "haiku" in self.model_type:
                self.model_name = "claude-3-5-haiku-20241022"
            else:
                self.model_name = "claude-3-5-sonnet-20241022"
                
            print(f"Initialized {self.model_name}")
            
        elif self.model_type.startswith("gpt"):
            # Try to work around proxy issue
            self._init_openai_workaround()
            
        else:
            # Default to Gemini as it's working
            print(f"Unknown model {self.model_type}, defaulting to Gemini")
            import google.generativeai as genai
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
            self.model_name = "gemini-2.5-flash"
            self.model = genai.GenerativeModel(self.model_name)
            
    def _init_openai_workaround(self):
        """Try to initialize OpenAI with proxy workaround"""
        try:
            # Clear proxy environment variables temporarily
            proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
            old_vars = {}
            for var in proxy_vars:
                if var in os.environ:
                    old_vars[var] = os.environ[var]
                    del os.environ[var]
            
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            self.model_name = "gpt-4o-mini"
            
            # Restore proxy vars
            for var, value in old_vars.items():
                os.environ[var] = value
                
            print(f"OpenAI initialized with proxy workaround")
            
        except Exception as e:
            print(f"OpenAI failed even with workaround: {e}")
            print("Falling back to Gemini")
            
            # Fall back to Gemini
            import google.generativeai as genai
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
            self.model_name = "gemini-2.5-flash"
            self.model = genai.GenerativeModel(self.model_name)
    
    async def generate(self, 
                      prompt: str, 
                      history: List[Message] = None) -> Tuple[str, int]:
        """Generate response"""
        
        try:
            if self.model_type.startswith("gemini"):
                # Build context from history
                context = ""
                if history:
                    # Include last 6 messages for context
                    recent = history[-6:]
                    context = "\n".join([f"{m.role}: {m.content}" for m in recent])
                    full_prompt = f"Previous context:\n{context}\n\nUser: {prompt}"
                else:
                    full_prompt = prompt
                
                response = self.model.generate_content(full_prompt)
                
                # Estimate tokens
                tokens = len(full_prompt.split()) + len(response.text.split())
                return response.text, int(tokens * 1.3)
                
            elif self.model_type.startswith("claude"):
                messages = []
                if history:
                    for msg in history[-8:]:  # Last 8 messages
                        role = "user" if msg.role == "user" else "assistant"
                        messages.append({"role": role, "content": msg.content})
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.messages.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=150,
                    temperature=0.7
                )
                
                text = response.content[0].text
                tokens = 0
                if hasattr(response, 'usage'):
                    tokens = response.usage.input_tokens + response.usage.output_tokens
                    
                return text, tokens
                
            elif hasattr(self, 'client') and self.model_type.startswith("gpt"):
                messages = []
                if history:
                    for msg in history[-8:]:
                        messages.append({
                            "role": msg.role if msg.role in ["user", "assistant"] else "user",
                            "content": msg.content
                        })
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=150,
                    temperature=0.7
                )
                
                return response.choices[0].message.content, response.usage.total_tokens
                
            else:
                return "Model not available", 0
                
        except Exception as e:
            print(f"Generation error: {e}")
            return f"Error: {str(e)[:50]}", 0

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
        """Create prompt based on experimental condition"""
        
        if self.condition == "linear":
            return user_input
            
        elif self.condition == "referenced" and reference_id is not None:
            ref_msg = next((m for m in self.history if m.id == reference_id), None)
            if ref_msg:
                return f"""Previous statement (Message #{reference_id}): "{ref_msg.content}"

User is now responding to that specific statement with: "{user_input}"

Please provide a response that directly addresses the referenced message."""
            
        elif self.condition == "hybrid":
            if reference_id and random.random() > 0.5:
                ref_msg = next((m for m in self.history if m.id == reference_id), None)
                if ref_msg:
                    preview = ref_msg.content[:50]
                    if len(ref_msg.content) > 50:
                        preview += "..."
                    return f"Referring to '{preview}': {user_input}"
                    
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
        
        # Generate prompt
        prompt = self.create_prompt(user_input, reference_id)
        
        # Get response
        response_text, tokens = await self.model.generate(prompt, self.history)
        
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
        
        # Ensure path exists
        path.mkdir(exist_ok=True, parents=True)
        
        data = {
            "participant_id": self.participant_id,
            "condition": self.condition,
            "model": self.model.model_type,
            "model_name": self.model.model_name,
            "timestamp": datetime.now().isoformat(),
            "messages": [asdict(msg) for msg in self.history],
            "metrics": {
                "total_messages": len(self.history),
                "total_tokens": sum(m.tokens or 0 for m in self.history),
                "has_references": any(m.reference_id is not None for m in self.history)
            }
        }
        
        filename = f"{self.participant_id}_{self.condition}_{self.model.model_type}.json"
        filepath = path / filename
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"  Saved: {filename}")
        return filepath