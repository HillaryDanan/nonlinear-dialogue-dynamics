#!/usr/bin/env python3
"""
COMPLETE NON-LINEAR DIALOGUE DYNAMICS EXPERIMENT - FIXED VERSION
================================================================
Now with working Google and OpenAI APIs

Using SBERT for proper semantic coherence measurement
n=50 prompts per condition for adequate power
"""

import os
import sys
import json
import time
import asyncio
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

# Load environment
from dotenv import load_dotenv
load_dotenv(override=True)

# Import the base experiment structure
sys.path.insert(0, '.')
from complete_nonlinear_experiment import (
    ReferenceType, 
    ExperimentConfig,
    PromptGenerator,
    CoherenceCalculator,
    DegradationAnalyzer,
    NonLinearDialogueExperiment
)

# ============================================================================
# FIXED MODEL INTERFACES
# ============================================================================

class FixedModelInterface:
    """
    Fixed model interface with working APIs
    """
    
    def __init__(self, provider: str):
        self.provider = provider
        self.working = False
        self.model_name = None
        self.total_tokens = 0
        self.client = None
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.5
        
        self._initialize()
    
    def _initialize(self):
        """Initialize with WORKING configurations"""
        
        if self.provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
                self.model_name = "claude-3-5-haiku-20241022"
                self.working = True
                print(f"  ✓ Anthropic initialized")
            except Exception as e:
                print(f"  ✗ Anthropic failed: {e}")
        
        elif self.provider == "google":
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
                # Use the WORKING model
                self.client = genai.GenerativeModel('gemini-2.0-flash')
                self.model_name = "gemini-2.0-flash"
                self.working = True
                print(f"  ✓ Google initialized (gemini-2.0-flash)")
            except Exception as e:
                print(f"  ✗ Google failed: {e}")
        
        elif self.provider == "openai":
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                self.model_name = "gpt-4o-mini"
                self.working = True
                print(f"  ✓ OpenAI initialized")
            except Exception as e:
                print(f"  ✗ OpenAI failed: {e}")
    
    async def generate(self, prompt: str, history: List[Dict] = None) -> Tuple[str, int]:
        """Generate with proper error handling"""
        
        if not self.working:
            return "Model not available", 0
        
        # Rate limiting
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - elapsed)
        
        try:
            if self.provider == "anthropic":
                return await self._generate_anthropic(prompt, history)
            elif self.provider == "google":
                return await self._generate_google(prompt, history)
            elif self.provider == "openai":
                return await self._generate_openai(prompt, history)
        except Exception as e:
            print(f"    Generation error: {str(e)[:50]}")
            return f"Error: {str(e)[:30]}", 0
    
    async def _generate_anthropic(self, prompt: str, history: List[Dict]) -> Tuple[str, int]:
        """Anthropic generation"""
        messages = []
        
        if history:
            for h in history[-6:]:
                if 'user_text' in h:
                    messages.append({"role": "user", "content": h['user_text'][:300]})
                if 'response' in h:
                    messages.append({"role": "assistant", "content": h['response'][:300]})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.messages.create(
            model=self.model_name,
            messages=messages,
            max_tokens=150,
            temperature=0.7
        )
        
        text = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens
        
        self.last_request_time = time.time()
        self.total_tokens += tokens
        
        return text, tokens
    
    async def _generate_google(self, prompt: str, history: List[Dict]) -> Tuple[str, int]:
        """Google generation - FIXED"""
        context = ""
        if history:
            for h in history[-4:]:
                if 'user_text' in h:
                    context += f"User: {h['user_text'][:200]}\n"
                if 'response' in h:
                    context += f"Assistant: {h['response'][:200]}\n"
        
        full_prompt = f"{context}\nUser: {prompt}" if context else prompt
        
        # Run sync function in executor
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            self.client.generate_content,
            full_prompt
        )
        
        text = response.text
        tokens = len(full_prompt.split()) + len(text.split())
        
        self.last_request_time = time.time()
        self.total_tokens += int(tokens * 1.3)
        
        return text, self.total_tokens
    
    async def _generate_openai(self, prompt: str, history: List[Dict]) -> Tuple[str, int]:
        """OpenAI generation - FIXED"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant participating in a research study."}
        ]
        
        if history:
            for h in history[-4:]:
                if 'user_text' in h:
                    messages.append({"role": "user", "content": h['user_text'][:300]})
                if 'response' in h:
                    messages.append({"role": "assistant", "content": h['response'][:300]})
        
        messages.append({"role": "user", "content": prompt})
        
        # Run sync client in executor for consistency
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=150,
                temperature=0.7
            )
        )
        
        text = response.choices[0].message.content
        tokens = response.usage.total_tokens
        
        self.last_request_time = time.time()
        self.total_tokens += tokens
        
        return text, tokens


# Override the ModelInterface in the original experiment
import complete_nonlinear_experiment
complete_nonlinear_experiment.ModelInterface = FixedModelInterface


async def main():
    """Run the complete experiment with all three models"""
    
    print("\n" + "="*70)
    print("NON-LINEAR DIALOGUE DYNAMICS EXPERIMENT - COMPLETE RUN")
    print("="*70)
    print("\nUsing SBERT for semantic coherence (proper measurement)")
    print("n=50 prompts per condition (adequate statistical power)")
    print("Testing Google and OpenAI to complete the study")
    
    # Check if we should skip Anthropic (already have data)
    existing_anthropic = list(Path("data/nonlinear_results").glob("anthropic_*.json"))
    
    if existing_anthropic:
        print(f"\n✓ Found existing Anthropic results: {existing_anthropic[0].name}")
        providers = ["google", "openai"]
        print("  Running only Google and OpenAI")
    else:
        providers = ["anthropic", "google", "openai"]
        print("  Running all three models")
    
    proceed = input("\nProceed with experiment? (y/n): ")
    if proceed.lower() != 'y':
        print("Experiment cancelled")
        return
    
    # Initialize experiment
    config = ExperimentConfig()
    experiment = NonLinearDialogueExperiment(config)
    
    print("\nInitializing models...")
    all_results = []
    
    # Load existing Anthropic results if available
    if existing_anthropic:
        with open(existing_anthropic[0], 'r') as f:
            anthropic_data = json.load(f)
            all_results.append(anthropic_data)
            print("  ✓ Loaded existing Anthropic results")
    
    # Run new experiments
    for provider in providers:
        try:
            print(f"\nStarting {provider}...")
            result = await experiment.run_model(provider)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"Error with {provider}: {e}")
    
    # Generate comprehensive report
    if all_results:
        report = experiment.generate_report(all_results)
        print("\n" + report)
        
        # Save combined report
        report_path = experiment.results_dir / f"complete_report_{datetime.now():%Y%m%d_%H%M}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n✓ Full report saved: {report_path}")
        
        # Quick statistical summary
        print("\n" + "="*70)
        print("STATISTICAL SUMMARY")
        print("="*70)
        
        for result in all_results:
            if result and 'comparisons' in result:
                provider = result['provider']
                print(f"\n{provider.upper()}:")
                
                for comp_name, comp_data in result['comparisons'].items():
                    condition = comp_name.split('_vs_')[0]
                    d = comp_data['cohens_d']
                    p = comp_data['p_value']
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                    
                    print(f"  {condition:15} : d={d:+.3f}, p={p:.4f} {sig}")
    else:
        print("\n✗ No results collected")


if __name__ == "__main__":
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    asyncio.run(main())
