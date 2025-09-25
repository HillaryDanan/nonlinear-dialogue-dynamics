#!/usr/bin/env python3
"""
Fixed experiment runner with rate limit handling for OpenAI
"""

import os
import sys
import json
import asyncio
import urllib.request
import urllib.error
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

from dotenv import load_dotenv
load_dotenv()

class OpenAIRateLimitedClient:
    """OpenAI client with rate limit handling and exponential backoff"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("No OpenAI API key")
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum 1 second between requests
        self.retry_count = 0
        self.max_retries = 5
    
    def chat(self, messages: List[Dict], model: str = "gpt-4o-mini", 
             max_tokens: int = 150, temperature: float = 0.7) -> Tuple[str, int]:
        """API call with rate limit handling"""
        
        # Rate limiting - ensure minimum interval between requests
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        
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
        
        # Retry logic with exponential backoff
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req) as response:
                    self.last_request_time = time.time()
                    self.retry_count = 0  # Reset on success
                    
                    result = json.loads(response.read().decode('utf-8'))
                    return result['choices'][0]['message']['content'], result['usage']['total_tokens']
                    
            except urllib.error.HTTPError as e:
                if e.code == 429:  # Rate limit
                    # Parse retry-after header if present
                    retry_after = e.headers.get('Retry-After', None)
                    if retry_after:
                        wait_time = float(retry_after)
                    else:
                        # Exponential backoff: 2^attempt seconds
                        wait_time = (2 ** attempt) + (0.1 * (attempt + 1))
                    
                    print(f"    Rate limited. Waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    
                    # Increase minimum interval for future requests
                    self.min_request_interval = min(5.0, self.min_request_interval * 1.5)
                    
                elif e.code == 401:
                    print(f"    ✗ OpenAI authentication failed - check API key")
                    return "Auth error", 0
                    
                else:
                    print(f"    OpenAI HTTP error {e.code}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        return f"Error: HTTP {e.code}", 0
                        
            except Exception as e:
                print(f"    OpenAI error: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return f"Error: {str(e)[:30]}", 0
        
        return "Max retries exceeded", 0

class ModelInterface:
    """Unified interface with proper rate limiting"""
    
    def __init__(self, provider: str):
        self.provider = provider
        self.working = False
        self.client = None
        self.model_name = None
        
        if provider == "openai":
            try:
                self.client = OpenAIRateLimitedClient()
                # Test with small request
                print("  Testing OpenAI...")
                response, _ = self.client.chat([{"role": "user", "content": "Hi"}], max_tokens=5)
                
                if "Error" not in response and "Auth error" not in response:
                    self.working = True
                    self.model_name = "gpt-4o-mini"
                    print(f"  ✓ OpenAI working (with rate limiting)")
                else:
                    print(f"  ✗ OpenAI not working: {response}")
                    
            except Exception as e:
                print(f"  ✗ OpenAI failed: {e}")
                
        elif provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
                self.model_name = "claude-3-5-haiku-20241022"
                self.working = True
                print(f"  ✓ Anthropic initialized")
            except Exception as e:
                print(f"  ✗ Anthropic failed: {e}")
                
        elif provider == "google":
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
                self.client = genai.GenerativeModel('gemini-2.5-flash')
                self.model_name = "gemini-2.5-flash"
                self.working = True
                print(f"  ✓ Google initialized")
            except Exception as e:
                print(f"  ✗ Google failed: {e}")
    
    async def generate(self, prompt: str, history: List = None) -> Tuple[str, int]:
        """Generate response with provider-specific handling"""
        
        if not self.working:
            return "Model not available", 0
        
        try:
            if self.provider == "openai":
                messages = []
                if history:
                    # Only last 4 messages to reduce tokens
                    for msg in history[-4:]:
                        messages.append({
                            "role": "user" if msg.get('role') == 'user' else "assistant",
                            "content": msg.get('content', '')[:200]  # Truncate for rate limits
                        })
                messages.append({"role": "user", "content": prompt[:200]})  # Truncate
                
                return self.client.chat(messages, max_tokens=100)  # Reduced tokens
                
            elif self.provider == "anthropic":
                messages = []
                if history:
                    for msg in history[-4:]:
                        role = "user" if msg.get('role') == 'user' else "assistant"
                        messages.append({"role": role, "content": msg.get('content', '')[:200]})
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
                    
                # Small delay to be nice to Anthropic
                await asyncio.sleep(0.5)
                return text, tokens
                
            elif self.provider == "google":
                context = ""
                if history:
                    recent = history[-2:]  # Even less context
                    context = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')[:100]}" for m in recent])
                    full_prompt = f"Context:\n{context}\n\nUser: {prompt}"
                else:
                    full_prompt = prompt
                    
                response = self.client.generate_content(full_prompt)
                tokens = len(full_prompt.split()) + len(response.text.split())
                
                # Small delay for Google too
                await asyncio.sleep(0.3)
                return response.text, int(tokens * 1.3)
                
        except Exception as e:
            print(f"    Generation error ({self.provider}): {str(e)[:50]}")
            return f"Error: {str(e)[:30]}", 0

class QuickExperiment:
    """Streamlined experiment runner"""
    
    def __init__(self):
        self.data_path = Path("data/rate_limited_pilot")
        self.data_path.mkdir(exist_ok=True, parents=True)
        
        # Try to load sentence transformers, fall back to simple if not available
        try:
            self.embedder = SentenceTransformer('all-mpnet-base-v2')
            self.use_sbert = True
            print("Using SBERT for coherence calculation")
        except:
            self.use_sbert = False
            print("Using simple coherence calculation")
        
        # Test all providers
        self.providers = []
        print("\nTesting providers:")
        for p in ["openai", "anthropic", "google"]:
            model = ModelInterface(p)
            if model.working:
                self.providers.append(p)
        
        if not self.providers:
            print("\n⚠️  No providers available!")
            sys.exit(1)
            
        print(f"\n✓ Available providers: {self.providers}")
        
        # Simplified experimental design for speed
        self.conditions = ["linear", "referenced"]  # Skip hybrid for pilot
        self.topics = [
            "What is machine learning?",
            "Explain quantum computing",
            "How do biases affect decisions?"
        ]
    
    def calculate_coherence(self, text1: str, text2: str) -> float:
        """Calculate coherence score"""
        
        if self.use_sbert:
            emb1 = self.embedder.encode(text1)
            emb2 = self.embedder.encode(text2)
            return 1 - cosine(emb1, emb2)
        else:
            # Simple word overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 or not words2:
                return 0.0
                
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            return intersection / union if union > 0 else 0
    
    async def run_single_conversation(self, 
                                    participant_id: str,
                                    condition: str, 
                                    provider: str) -> Dict:
        """Run single conversation quickly"""
        
        print(f"\n  {condition} × {provider}")
        
        model = ModelInterface(provider)
        if not model.working:
            print(f"    Skipping - not available")
            return {"status": "skipped"}
        
        messages = []
        coherence_scores = []
        
        # Simplified task - just 3 topics with optional reference
        for i, topic in enumerate(self.topics):
            # Generate prompt
            if condition == "referenced" and i > 0:
                # Reference first topic
                prompt = f"Returning to the first topic: {topic}"
            else:
                prompt = topic
            
            # Get response
            response, tokens = await model.generate(prompt, messages)
            
            # Calculate coherence
            coherence = self.calculate_coherence(prompt, response)
            coherence_scores.append(coherence)
            
            # Store messages
            messages.append({"role": "user", "content": prompt})
            messages.append({"role": "assistant", "content": response})
            
            print(f"    Topic {i+1}: coherence={coherence:.3f}")
        
        # Save results
        result = {
            "participant_id": participant_id,
            "condition": condition,
            "provider": provider,
            "coherence_mean": np.mean(coherence_scores),
            "coherence_scores": coherence_scores,
            "timestamp": datetime.now().isoformat()
        }
        
        filename = f"{participant_id}_{condition}_{provider}.json"
        with open(self.data_path / filename, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        return result
    
    async def run_pilot(self, n_participants: int = 2):
        """Run quick pilot study"""
        
        print("\n" + "="*60)
        print(f"QUICK PILOT (n={n_participants})")
        print(f"Conditions: {self.conditions}")
        print(f"Providers: {self.providers}")
        print("="*60)
        
        all_results = []
        
        for p_num in range(1, n_participants + 1):
            participant_id = f"pilot_{p_num:03d}"
            print(f"\n--- Participant {participant_id} ---")
            
            for condition in self.conditions:
                for provider in self.providers:
                    try:
                        result = await self.run_single_conversation(
                            participant_id, condition, provider
                        )
                        
                        if result.get("status") != "skipped":
                            all_results.append(result)
                            
                    except KeyboardInterrupt:
                        print("\n⚠️  Interrupted by user")
                        self.analyze_results(all_results)
                        sys.exit(0)
                        
                    except Exception as e:
                        print(f"    Error: {e}")
        
        # Analysis
        self.analyze_results(all_results)
    
    def analyze_results(self, results: List[Dict]):
        """Quick analysis"""
        
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        
        if not results:
            print("No results to analyze")
            return
        
        # Group by condition
        by_condition = {}
        for r in results:
            if 'coherence_mean' in r:
                cond = r['condition']
                if cond not in by_condition:
                    by_condition[cond] = []
                by_condition[cond].append(r['coherence_mean'])
        
        # Print means
        print("\nCoherence by Condition:")
        for cond in self.conditions:
            if cond in by_condition:
                scores = by_condition[cond]
                mean = np.mean(scores)
                std = np.std(scores) if len(scores) > 1 else 0
                print(f"  {cond:10} Mean={mean:.3f} (SD={std:.3f}, n={len(scores)})")
        
        # Effect size if both conditions present
        if 'linear' in by_condition and 'referenced' in by_condition:
            linear = by_condition['linear']
            referenced = by_condition['referenced']
            
            if linear and referenced:
                mean_diff = np.mean(referenced) - np.mean(linear)
                
                # Cohen's d
                pooled_std = np.sqrt((np.var(referenced) + np.var(linear)) / 2)
                if pooled_std > 0:
                    d = mean_diff / pooled_std
                    print(f"\nEffect size (referenced vs linear): d={d:.3f}")
                    
                    if d > 0:
                        print("✓ Hypothesis supported: Referenced improves coherence")
                    else:
                        print("✗ Hypothesis not supported")
        
        print(f"\nData saved to: {self.data_path}")

async def main():
    """Main entry point"""
    
    print("\nNONLINEAR DIALOGUE DYNAMICS - RATE LIMITED VERSION")
    print("Handles OpenAI rate limits properly")
    
    # Check OpenAI specifically
    print("\nChecking OpenAI API key...")
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print(f"✓ API key found: {api_key[:7]}...{api_key[-4:]}")
    else:
        print("✗ No OpenAI API key found")
    
    proceed = input("\nRun quick pilot? (y/n): ")
    if proceed.lower() != 'y':
        return
    
    experiment = QuickExperiment()
    
    # Reduced n for quicker testing
    await experiment.run_pilot(n_participants=2)

if __name__ == "__main__":
    asyncio.run(main())