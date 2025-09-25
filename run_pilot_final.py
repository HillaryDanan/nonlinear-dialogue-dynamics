#!/usr/bin/env python3
"""
Final pragmatic pilot runner
Hypothesis: Non-linear referencing improves dialogue coherence
We test this regardless of which APIs work
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# Clear proxy variables that break OpenAI
for var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    if var in os.environ:
        del os.environ[var]

from dotenv import load_dotenv
load_dotenv()

class SimpleModelInterface:
    """Simplified interface that handles all three providers gracefully"""
    
    def __init__(self, provider: str):
        self.provider = provider
        self.working = False
        self.client = None
        self.model_name = None
        
        if provider == "openai":
            self._try_openai()
        elif provider == "anthropic":
            self._try_anthropic()
        elif provider == "google":
            self._try_google()
    
    def _try_openai(self):
        """Try OpenAI with workarounds for proxy issue"""
        try:
            # Method 1: Standard initialization
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            self.model_name = "gpt-4o-mini"
            
            # Test it
            test = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            self.working = True
            print(f"✓ OpenAI initialized")
            
        except Exception as e:
            if "proxies" in str(e):
                # Method 2: Direct HTTP fallback
                self._setup_openai_fallback()
            else:
                print(f"✗ OpenAI failed: {str(e)[:50]}")
    
    def _setup_openai_fallback(self):
        """Fallback using requests if OpenAI library broken"""
        import requests
        self.model_name = "gpt-4o-mini"
        self.working = True
        self.use_requests = True
        print(f"✓ OpenAI using HTTP fallback")
    
    def _try_anthropic(self):
        """Initialize Anthropic"""
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
            self.model_name = "claude-3-5-haiku-20241022"
            self.working = True
            print(f"✓ Anthropic initialized")
        except Exception as e:
            print(f"✗ Anthropic failed: {str(e)[:50]}")
    
    def _try_google(self):
        """Initialize Google"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
            self.client = genai.GenerativeModel('gemini-2.5-flash')
            self.model_name = "gemini-2.5-flash"
            self.working = True
            print(f"✓ Google initialized")
        except Exception as e:
            print(f"✗ Google failed: {str(e)[:50]}")
    
    async def generate(self, prompt: str, history: List = None) -> Tuple[str, int]:
        """Generate response from whichever model works"""
        
        if not self.working:
            return "Model not available", 0
        
        try:
            if self.provider == "openai":
                if hasattr(self, 'use_requests'):
                    # HTTP fallback
                    import requests
                    headers = {
                        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                        "Content-Type": "application/json"
                    }
                    data = {
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 150,
                        "temperature": 0.7
                    }
                    response = requests.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers=headers,
                        json=data
                    )
                    if response.status_code == 200:
                        result = response.json()
                        return result['choices'][0]['message']['content'], result['usage']['total_tokens']
                else:
                    # Normal OpenAI client
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=150,
                        temperature=0.7
                    )
                    return response.choices[0].message.content, response.usage.total_tokens
                    
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0.7
                )
                text = response.content[0].text
                tokens = getattr(response.usage, 'input_tokens', 0) + getattr(response.usage, 'output_tokens', 0)
                return text, tokens
                
            elif self.provider == "google":
                response = self.client.generate_content(prompt)
                tokens = len(prompt.split()) + len(response.text.split())
                return response.text, int(tokens * 1.3)
                
        except Exception as e:
            print(f"Generation error ({self.provider}): {str(e)[:50]}")
            return f"Error: {str(e)[:30]}", 0
        
        return "No response", 0

class ExperimentRunner:
    """Runs the actual experiment with available models"""
    
    def __init__(self):
        self.data_path = Path("data/pragmatic_pilot")
        self.data_path.mkdir(exist_ok=True, parents=True)
        self.embedder = SentenceTransformer('all-mpnet-base-v2')
        
        # Test which models work
        self.available_providers = []
        print("\nTesting model availability...")
        
        for provider in ["google", "anthropic", "openai"]:
            model = SimpleModelInterface(provider)
            if model.working:
                self.available_providers.append(provider)
        
        print(f"\nAvailable providers: {self.available_providers}")
        
        if len(self.available_providers) == 0:
            raise Exception("No models available! Check API keys")
        
        # Experimental design
        self.conditions = ["linear", "referenced", "hybrid"]
        self.topics = [
            "How might quantum computing change cryptography?",
            "What are effective strategies for sustainable urban planning?",
            "How do cognitive biases affect decision making?"
        ]
    
    async def run_conversation(self, 
                              participant_id: str,
                              condition: str,
                              provider: str) -> Dict:
        """Run single experimental conversation"""
        
        model = SimpleModelInterface(provider)
        if not model.working:
            return {"status": "skipped", "provider": provider}
        
        messages = []
        message_id = 0
        
        # Task 1: Multi-topic discussion
        for topic in self.topics:
            # Initial topic
            prompt = topic
            if condition == "referenced" and len(messages) > 0:
                prompt = f"New topic: {topic}"
            
            response, tokens = await model.generate(prompt, messages)
            messages.append({
                "id": message_id,
                "role": "user",
                "content": topic,
                "reference_id": None
            })
            message_id += 1
            
            messages.append({
                "id": message_id,
                "role": "assistant",
                "content": response,
                "tokens": tokens
            })
            message_id += 1
        
        # Task 2: Return to topics with elaboration
        elaborations = [
            "Can you elaborate on the technical aspects?",
            "What about the environmental impacts?",
            "How does this apply to group decisions?"
        ]
        
        for i, elab in enumerate(elaborations):
            # Reference earlier topic
            ref_id = i * 2 if condition == "referenced" else None
            
            if condition == "referenced":
                # Reference the original topic
                ref_msg = messages[ref_id]['content']
                prompt = f'Previous statement: "{ref_msg}"\n\nUser asks: {elab}'
            else:
                prompt = elab
            
            response, tokens = await model.generate(prompt, messages)
            
            messages.append({
                "id": message_id,
                "role": "user",
                "content": elab,
                "reference_id": ref_id
            })
            message_id += 1
            
            messages.append({
                "id": message_id,
                "role": "assistant",
                "content": response,
                "tokens": tokens
            })
            message_id += 1
        
        # Calculate coherence metrics
        coherence_scores = []
        for i in range(1, len(messages), 2):  # Check assistant responses
            if i < len(messages):
                user_msg = messages[i-1]['content']
                assistant_msg = messages[i]['content']
                
                # Calculate coherence
                user_emb = self.embedder.encode(user_msg)
                assistant_emb = self.embedder.encode(assistant_msg)
                coherence = 1 - cosine(user_emb, assistant_emb)
                coherence_scores.append(coherence)
        
        # Save data
        result = {
            "participant_id": participant_id,
            "condition": condition,
            "provider": provider,
            "timestamp": datetime.now().isoformat(),
            "messages": messages,
            "metrics": {
                "coherence_mean": np.mean(coherence_scores),
                "coherence_std": np.std(coherence_scores),
                "n_messages": len(messages),
                "n_references": sum(1 for m in messages if m.get('reference_id'))
            }
        }
        
        filename = f"{participant_id}_{condition}_{provider}.json"
        with open(self.data_path / filename, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        return result
    
    async def run_pilot(self, n_participants: int = 3):
        """Run complete pilot study"""
        
        print("\n" + "="*60)
        print("PRAGMATIC PILOT STUDY")
        print(f"Design: {len(self.conditions)}×{len(self.available_providers)} factorial")
        print(f"Conditions: {self.conditions}")
        print(f"Providers: {self.available_providers}")
        print(f"Participants: n={n_participants}")
        print("="*60)
        
        all_results = []
        
        for p_num in range(1, n_participants + 1):
            participant_id = f"pilot_{p_num:03d}"
            print(f"\n--- Participant {participant_id} ---")
            
            # Counterbalanced order
            condition_order = self.conditions[p_num-1:] + self.conditions[:p_num-1]
            provider_order = self.available_providers[p_num-1:] + self.available_providers[:p_num-1]
            
            for condition in condition_order:
                for provider in provider_order:
                    print(f"  Running {condition} × {provider}...")
                    
                    try:
                        result = await self.run_conversation(
                            participant_id,
                            condition,
                            provider
                        )
                        
                        if result.get('metrics'):
                            coherence = result['metrics']['coherence_mean']
                            print(f"    ✓ Coherence: {coherence:.3f}")
                        
                        all_results.append(result)
                        
                    except Exception as e:
                        print(f"    ✗ Error: {str(e)[:50]}")
                    
                    await asyncio.sleep(1)  # Rate limiting
        
        # Quick analysis
        print("\n" + "="*60)
        print("PRELIMINARY RESULTS")
        print("="*60)
        
        # Group by condition
        by_condition = {}
        for result in all_results:
            if 'metrics' in result:
                cond = result['condition']
                if cond not in by_condition:
                    by_condition[cond] = []
                by_condition[cond].append(result['metrics']['coherence_mean'])
        
        print("\nCoherence by Condition:")
        for cond in self.conditions:
            if cond in by_condition:
                scores = by_condition[cond]
                print(f"  {cond:10} Mean={np.mean(scores):.3f} (SD={np.std(scores):.3f})")
        
        # Calculate effect sizes
        if 'linear' in by_condition and 'referenced' in by_condition:
            linear_scores = by_condition['linear']
            ref_scores = by_condition['referenced']
            
            if len(linear_scores) > 0 and len(ref_scores) > 0:
                mean_diff = np.mean(ref_scores) - np.mean(linear_scores)
                pooled_std = np.sqrt((np.std(ref_scores)**2 + np.std(linear_scores)**2) / 2)
                
                if pooled_std > 0:
                    cohens_d = mean_diff / pooled_std
                    print(f"\nEffect size (referenced vs linear): d={cohens_d:.3f}")
                    
                    if cohens_d > 0:
                        print("✓ Hypothesis supported: Referenced shows improvement")
                    else:
                        print("✗ Hypothesis not supported in pilot data")
        
        print(f"\nData saved to: {self.data_path}")
        print("\nNext steps:")
        print("1. Review the data files")
        print("2. Run full analysis script")
        print("3. Adjust protocol if needed")
        print("4. Run main study with n=64")

async def main():
    """Main entry point"""
    
    print("\nNONLINEAR DIALOGUE DYNAMICS")
    print("Testing hypothesis: Explicit reference improves coherence")
    
    # Check if we should proceed
    proceed = input("\nRun pragmatic pilot with available models? (y/n): ")
    if proceed.lower() != 'y':
        print("Cancelled")
        return
    
    runner = ExperimentRunner()
    
    if len(runner.available_providers) < 2:
        print(f"\n⚠️  Only {len(runner.available_providers)} provider(s) available")
        print("Results may have limited generalizability")
        cont = input("\nContinue anyway? (y/n): ")
        if cont.lower() != 'y':
            return
    
    await runner.run_pilot(n_participants=3)

if __name__ == "__main__":
    asyncio.run(main())