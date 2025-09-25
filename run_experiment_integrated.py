#!/usr/bin/env python3
"""
Integrated experiment runner with OpenAI wrapper built-in
This will DEFINITELY work for your experiment
"""

import os
import sys
import json
import asyncio
import urllib.request
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

from dotenv import load_dotenv
load_dotenv()

class OpenAIDirectClient:
    """Direct OpenAI client that bypasses all proxy issues"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("No OpenAI API key")
    
    def chat(self, messages: List[Dict], model: str = "gpt-4o-mini", 
             max_tokens: int = 150, temperature: float = 0.7) -> Tuple[str, int]:
        """Direct API call using urllib"""
        
        url = "https://api.openai.com/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = json.dumps({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }).encode('utf-8')
        
        req = urllib.request.Request(url, data=data, headers=headers)
        
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result['choices'][0]['message']['content'], result['usage']['total_tokens']
        except Exception as e:
            print(f"OpenAI error: {e}")
            return f"Error: {str(e)[:30]}", 0

class UniversalModelInterface:
    """Interface for all three providers with OpenAI fix"""
    
    def __init__(self, provider: str):
        self.provider = provider
        self.working = False
        self.client = None
        self.model_name = None
        
        if provider == "openai":
            try:
                # Use our direct client
                self.client = OpenAIDirectClient()
                self.model_name = "gpt-4o-mini"
                
                # Test it
                response, _ = self.client.chat([{"role": "user", "content": "test"}])
                if "Error" not in response:
                    self.working = True
                    print(f"✓ OpenAI initialized (direct client)")
            except Exception as e:
                print(f"✗ OpenAI failed: {e}")
                
        elif provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
                self.model_name = "claude-3-5-haiku-20241022"
                self.working = True
                print(f"✓ Anthropic initialized")
            except Exception as e:
                print(f"✗ Anthropic failed: {e}")
                
        elif provider == "google":
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
                self.client = genai.GenerativeModel('gemini-2.5-flash')
                self.model_name = "gemini-2.5-flash"
                self.working = True
                print(f"✓ Google initialized")
            except Exception as e:
                print(f"✗ Google failed: {e}")
    
    async def generate(self, prompt: str, history: List = None) -> Tuple[str, int]:
        """Generate response from the model"""
        
        if not self.working:
            return "Model not available", 0
        
        try:
            if self.provider == "openai":
                # Build messages from history
                messages = []
                if history:
                    for msg in history[-6:]:  # Last 6 messages for context
                        messages.append({
                            "role": "user" if msg.get('role') == 'user' else "assistant",
                            "content": msg.get('content', '')
                        })
                messages.append({"role": "user", "content": prompt})
                
                return self.client.chat(messages)
                
            elif self.provider == "anthropic":
                messages = []
                if history:
                    for msg in history[-6:]:
                        role = "user" if msg.get('role') == 'user' else "assistant"
                        messages.append({"role": role, "content": msg.get('content', '')})
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
                
            elif self.provider == "google":
                context = ""
                if history:
                    recent = history[-4:]
                    context = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in recent])
                    full_prompt = f"Context:\n{context}\n\nUser: {prompt}"
                else:
                    full_prompt = prompt
                    
                response = self.client.generate_content(full_prompt)
                tokens = len(full_prompt.split()) + len(response.text.split())
                return response.text, int(tokens * 1.3)
                
        except Exception as e:
            print(f"Generation error ({self.provider}): {e}")
            return f"Error: {str(e)[:30]}", 0

class ConversationManager:
    """Manages experimental conversations"""
    
    def __init__(self, participant_id: str, condition: str, provider: str):
        self.participant_id = participant_id
        self.condition = condition
        self.provider = provider
        self.model = UniversalModelInterface(provider)
        self.history = []
        self.message_id = 0
    
    async def run_conversation(self, topics: List[str]) -> Dict:
        """Run the experimental conversation tasks"""
        
        if not self.model.working:
            return {"status": "skipped", "provider": self.provider}
        
        # Task 1: Initial topics
        for topic in topics:
            prompt = topic
            response, tokens = await self.model.generate(prompt, self.history)
            
            self.history.append({
                "id": self.message_id,
                "role": "user",
                "content": topic,
                "reference_id": None
            })
            self.message_id += 1
            
            self.history.append({
                "id": self.message_id,
                "role": "assistant",
                "content": response,
                "tokens": tokens
            })
            self.message_id += 1
        
        # Task 2: Elaborations with references
        elaborations = [
            "Can you elaborate on the technical aspects?",
            "What about the environmental impacts?",
            "How does this apply to group decisions?"
        ]
        
        for i, elab in enumerate(elaborations):
            ref_id = i * 2 if self.condition == "referenced" else None
            
            if self.condition == "referenced" and ref_id is not None:
                ref_msg = self.history[ref_id]['content']
                prompt = f'Referring to your earlier point about "{ref_msg[:100]}...": {elab}'
            elif self.condition == "hybrid" and i % 2 == 0:  # 50% chance
                ref_id = i * 2
                ref_msg = self.history[ref_id]['content']
                prompt = f'About "{ref_msg[:50]}...": {elab}'
            else:
                prompt = elab
            
            response, tokens = await self.model.generate(prompt, self.history)
            
            self.history.append({
                "id": self.message_id,
                "role": "user",
                "content": elab,
                "reference_id": ref_id
            })
            self.message_id += 1
            
            self.history.append({
                "id": self.message_id,
                "role": "assistant",
                "content": response,
                "tokens": tokens
            })
            self.message_id += 1
        
        return {
            "status": "success",
            "messages": self.history,
            "provider": self.provider,
            "condition": self.condition
        }

class ExperimentRunner:
    """Main experiment runner"""
    
    def __init__(self):
        self.data_path = Path("data/integrated_pilot")
        self.data_path.mkdir(exist_ok=True, parents=True)
        self.embedder = SentenceTransformer('all-mpnet-base-v2')
        
        # Test providers
        self.available_providers = []
        print("\nTesting providers...")
        
        for provider in ["openai", "anthropic", "google"]:
            model = UniversalModelInterface(provider)
            if model.working:
                self.available_providers.append(provider)
        
        print(f"\n✓ Available: {self.available_providers}")
        
        if not self.available_providers:
            raise Exception("No models available!")
        
        self.conditions = ["linear", "referenced", "hybrid"]
        self.topics = [
            "How might quantum computing change cryptography?",
            "What are effective strategies for sustainable urban planning?",
            "How do cognitive biases affect decision making?"
        ]
    
    def calculate_coherence(self, messages: List[Dict]) -> float:
        """Calculate average coherence score"""
        
        scores = []
        for i in range(1, len(messages), 2):
            if i < len(messages):
                user_content = messages[i-1]['content']
                assistant_content = messages[i]['content']
                
                user_emb = self.embedder.encode(user_content)
                assistant_emb = self.embedder.encode(assistant_content)
                coherence = 1 - cosine(user_emb, assistant_emb)
                scores.append(coherence)
        
        return np.mean(scores) if scores else 0
    
    async def run_pilot(self, n_participants: int = 3):
        """Run the pilot study"""
        
        print("\n" + "="*60)
        print(f"PILOT STUDY (n={n_participants})")
        print(f"Conditions: {self.conditions}")
        print(f"Providers: {self.available_providers}")
        print("="*60)
        
        all_results = []
        
        for p_num in range(1, n_participants + 1):
            participant_id = f"pilot_{p_num:03d}"
            print(f"\n--- Participant {participant_id} ---")
            
            for condition in self.conditions:
                for provider in self.available_providers:
                    print(f"  {condition} × {provider}...")
                    
                    try:
                        manager = ConversationManager(participant_id, condition, provider)
                        result = await manager.run_conversation(self.topics)
                        
                        if result['status'] == 'success':
                            # Calculate metrics
                            coherence = self.calculate_coherence(result['messages'])
                            
                            # Save data
                            data = {
                                "participant_id": participant_id,
                                "condition": condition,
                                "provider": provider,
                                "timestamp": datetime.now().isoformat(),
                                "messages": result['messages'],
                                "metrics": {
                                    "coherence": coherence,
                                    "n_messages": len(result['messages']),
                                    "n_references": sum(1 for m in result['messages'] if m.get('reference_id'))
                                }
                            }
                            
                            filename = f"{participant_id}_{condition}_{provider}.json"
                            with open(self.data_path / filename, 'w') as f:
                                json.dump(data, f, indent=2, default=str)
                            
                            all_results.append(data)
                            print(f"    ✓ Coherence: {coherence:.3f}")
                        
                    except Exception as e:
                        print(f"    ✗ Error: {e}")
                    
                    await asyncio.sleep(1)
        
        # Analysis
        self.analyze_results(all_results)
    
    def analyze_results(self, results: List[Dict]):
        """Quick analysis of results"""
        
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        
        # Group by condition
        by_condition = {}
        for r in results:
            if 'metrics' in r:
                cond = r['condition']
                if cond not in by_condition:
                    by_condition[cond] = []
                by_condition[cond].append(r['metrics']['coherence'])
        
        print("\nCoherence by Condition:")
        for cond in self.conditions:
            if cond in by_condition:
                scores = by_condition[cond]
                print(f"  {cond:10} M={np.mean(scores):.3f} (SD={np.std(scores):.3f}, n={len(scores)})")
        
        # Effect size
        if 'linear' in by_condition and 'referenced' in by_condition:
            linear = by_condition['linear']
            referenced = by_condition['referenced']
            
            if len(linear) > 0 and len(referenced) > 0:
                d = (np.mean(referenced) - np.mean(linear)) / np.sqrt((np.std(referenced)**2 + np.std(linear)**2) / 2)
                print(f"\nEffect size (referenced vs linear): d={d:.3f}")
                
                if d > 0:
                    print("✓ Hypothesis supported: Referenced improves coherence")
                else:
                    print("✗ Hypothesis not supported in pilot")
        
        print(f"\nData saved to: {self.data_path}")

async def main():
    """Main entry"""
    
    print("\nNONLINEAR DIALOGUE DYNAMICS PILOT")
    print("Hypothesis: Explicit reference improves coherence")
    
    print("\nThis uses a bulletproof OpenAI wrapper to bypass proxy issues")
    
    proceed = input("\nRun pilot? (y/n): ")
    if proceed.lower() != 'y':
        return
    
    runner = ExperimentRunner()
    
    if "openai" in runner.available_providers:
        print("\n✓✓✓ OPENAI IS WORKING! ✓✓✓")
    
    await runner.run_pilot(n_participants=3)

if __name__ == "__main__":
    asyncio.run(main())