#!/usr/bin/env python3
"""
COMPLETE NON-LINEAR DIALOGUE DYNAMICS EXPERIMENT
=================================================
A rigorous test of whether explicit conversational referencing improves coherence

THEORETICAL FOUNDATION:
- Clark & Brennan (1991): Grounding in communication requires explicit acknowledgment
- Miller (1956): The magical number 7±2 in information processing
- Cowan (2001): Working memory capacity limits (4±1 items)
- Baddeley (2000): Episodic buffer as interface between working and long-term memory
- Tulving (1983): Episodic memory retrieval mechanisms

HYPOTHESES:
H1: Explicit referencing improves coherence (expected d=0.27 based on HCI literature)
H2: Models show distinct degradation patterns with recursion depth
H3: Critical coherence breakdown occurs at depth 5-7 (Miller's limit)
H4: Contradiction handling varies by model architecture

Author: Hillary Danan
Date: September 2025
"""

import os
import sys
import json
import time
import asyncio
import hashlib
import urllib.request
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# EXPERIMENTAL DESIGN
# ============================================================================

class ReferenceType(Enum):
    """
    Reference types based on cognitive architecture
    Each tests different memory systems (Baddeley & Hitch, 1974)
    """
    LINEAR = "linear"  # No reference (control condition)
    IMMEDIATE = "immediate"  # N-1 (phonological loop, <2 sec)
    SHALLOW = "shallow"  # N-2 to N-3 (working memory, <30 sec)
    MEDIUM = "medium"  # N-4 to N-5 (episodic buffer)
    DEEP = "deep"  # N-6+ (long-term memory retrieval)
    CONTRADICTORY = "contradictory"  # Tests consistency maintenance
    BRANCHING = "branching"  # Multiple references (divided attention)

@dataclass
class ExperimentConfig:
    """
    Pre-registered experimental parameters
    Following Open Science Framework guidelines (Nosek et al., 2018)
    """
    # Core parameters
    n_prompts: int = 50  # Based on power analysis for d=0.27
    n_runs: int = 1  # Number of repetitions per condition
    
    # Conditions to test
    reference_types: List[ReferenceType] = field(
        default_factory=lambda: [
            ReferenceType.LINEAR,
            ReferenceType.IMMEDIATE,
            ReferenceType.SHALLOW,
            ReferenceType.DEEP,
            ReferenceType.CONTRADICTORY
        ]
    )
    
    # Recursion depths to test (spans Miller's range)
    recursion_depths: List[int] = field(
        default_factory=lambda: [1, 2, 3, 5, 7, 9, 11]
    )
    
    # Statistical parameters
    alpha: float = 0.01  # Bonferroni: 0.05/5 conditions
    power_target: float = 0.80
    expected_effect_size: float = 0.27
    
    # Operational parameters
    randomize_order: bool = False  # Keep consistent per your friend's suggestion
    save_intermediate: bool = True
    rate_limit_delay: float = 0.5  # Seconds between API calls

# ============================================================================
# PROMPT GENERATION (50 CAREFULLY DESIGNED PROMPTS)
# ============================================================================

class PromptGenerator:
    """
    Generates 50 prompts balanced across domains and cognitive demands
    Following Bloom's Taxonomy (Anderson & Krathwohl, 2001)
    """
    
    @staticmethod
    def generate_prompt_battery() -> List[Dict[str, Any]]:
        """
        10 domains × 5 cognitive levels = 50 prompts
        Ensures broad coverage and statistical power
        """
        
        domains = [
            # Hard sciences
            ("quantum mechanics", "physics"),
            ("molecular biology", "biology"),
            ("neural networks", "computer science"),
            
            # Complex systems
            ("climate dynamics", "earth science"),
            ("economic markets", "economics"),
            ("urban ecosystems", "urban planning"),
            
            # Abstract concepts
            ("consciousness", "philosophy"),
            ("emergence", "complexity science"),
            
            # Applied fields
            ("CRISPR technology", "biotechnology"),
            ("cryptographic protocols", "cryptography")
        ]
        
        # Cognitive levels (Bloom's revised taxonomy)
        question_templates = [
            # Level 1: Remember (retrieval)
            ("What are the fundamental principles of {}?", 1, "remember"),
            
            # Level 2: Understand (comprehension)
            ("How does {} relate to thermodynamics?", 2, "understand"),
            
            # Level 3: Apply (application)
            ("How would {} solve real-world problems?", 3, "apply"),
            
            # Level 4: Analyze (relationships)
            ("What contradictions exist in {} theory?", 4, "analyze"),
            
            # Level 5: Evaluate (judgment)
            ("What are the limitations and strengths of {}?", 5, "evaluate")
        ]
        
        prompts = []
        prompt_id = 0
        
        for domain, field in domains:
            for template, complexity, cognitive_level in question_templates:
                prompt_text = template.format(domain)
                
                prompts.append({
                    'id': prompt_id,
                    'text': prompt_text,
                    'domain': domain,
                    'field': field,
                    'complexity': complexity,
                    'cognitive_level': cognitive_level,
                    'tokens_estimate': len(prompt_text.split()) * 1.3
                })
                
                prompt_id += 1
        
        return prompts

# ============================================================================
# MODEL INTERFACES WITH RATE LIMITING
# ============================================================================

class ModelInterface:
    """
    Unified interface for all models with proper error handling
    """
    
    def __init__(self, provider: str):
        self.provider = provider
        self.working = False
        self.model_name = None
        self.total_tokens = 0
        self.total_cost = 0.0
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0
        
        self._initialize()
    
    def _initialize(self):
        """Initialize based on provider"""
        
        if self.provider == "anthropic":
            self._init_anthropic()
        elif self.provider == "google":
            self._init_google()
        elif self.provider == "openai":
            self._init_openai_direct()
        else:
            print(f"Unknown provider: {self.provider}")
    
    def _init_anthropic(self):
        """Initialize Anthropic Claude"""
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
            self.model_name = "claude-3-5-haiku-20241022"
            self.working = True
            print(f"  ✓ Anthropic initialized ({self.model_name})")
        except Exception as e:
            print(f"  ✗ Anthropic failed: {e}")
    
    def _init_google(self):
        """Initialize Google Gemini"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
            self.client = genai.GenerativeModel('gemini-1.5-flash')
            self.model_name = "gemini-1.5-flash"
            self.working = True
            print(f"  ✓ Google initialized ({self.model_name})")
        except Exception as e:
            print(f"  ✗ Google failed: {e}")
    
    def _init_openai_direct(self):
        """Initialize OpenAI with direct HTTP (bypasses proxy issues)"""
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            self.api_key = api_key
            self.model_name = "gpt-4o-mini"
            self.working = True
            self.use_direct = True
            print(f"  ✓ OpenAI initialized (direct HTTP)")
        else:
            print(f"  ✗ OpenAI: No API key")
    
    async def generate(self, prompt: str, history: List[Dict] = None) -> Tuple[str, int]:
        """
        Generate response with rate limiting
        Returns: (response_text, token_count)
        """
        
        if not self.working:
            return "Model not available", 0
        
        # Rate limiting
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - elapsed)
        
        try:
            if self.provider == "anthropic":
                response_text, tokens = await self._generate_anthropic(prompt, history)
            elif self.provider == "google":
                response_text, tokens = await self._generate_google(prompt, history)
            elif self.provider == "openai":
                response_text, tokens = await self._generate_openai(prompt, history)
            else:
                response_text, tokens = "Unknown provider", 0
            
            self.last_request_time = time.time()
            self.total_tokens += tokens
            
            return response_text, tokens
            
        except Exception as e:
            print(f"    Generation error: {str(e)[:50]}")
            return f"Error: {str(e)[:30]}", 0
    
    async def _generate_anthropic(self, prompt: str, history: List[Dict]) -> Tuple[str, int]:
        """Generate using Anthropic"""
        messages = []
        
        if history:
            # Include last 6 messages for context
            for h in history[-6:]:
                if 'user_text' in h:
                    messages.append({"role": "user", "content": h['user_text'][:200]})
                if 'response' in h:
                    messages.append({"role": "assistant", "content": h['response'][:200]})
        
        messages.append({"role": "user", "content": prompt[:300]})
        
        response = self.client.messages.create(
            model=self.model_name,
            messages=messages,
            max_tokens=150,
            temperature=0.7
        )
        
        text = response.content[0].text if hasattr(response.content[0], 'text') else str(response.content[0])
        tokens = 0
        
        if hasattr(response, 'usage'):
            tokens = response.usage.input_tokens + response.usage.output_tokens
        
        return text, tokens
    
    async def _generate_google(self, prompt: str, history: List[Dict]) -> Tuple[str, int]:
        """Generate using Google"""
        
        # Build context
        context = ""
        if history:
            for h in history[-4:]:
                if 'user_text' in h:
                    context += f"User: {h['user_text'][:100]}\n"
                if 'response' in h:
                    context += f"Assistant: {h['response'][:100]}\n"
        
        full_prompt = f"{context}\nUser: {prompt}" if context else prompt
        
        response = self.client.generate_content(full_prompt[:500])
        
        tokens = len(full_prompt.split()) + len(response.text.split())
        return response.text, int(tokens * 1.3)
    
    async def _generate_openai(self, prompt: str, history: List[Dict]) -> Tuple[str, int]:
        """Generate using OpenAI (direct HTTP)"""
        
        messages = []
        if history:
            for h in history[-4:]:
                if 'user_text' in h:
                    messages.append({"role": "user", "content": h['user_text'][:100]})
                if 'response' in h:
                    messages.append({"role": "assistant", "content": h['response'][:100]})
        
        messages.append({"role": "user", "content": prompt[:200]})
        
        # Direct HTTP request
        url = "https://api.openai.com/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = json.dumps({
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 150,
            "temperature": 0.7
        }).encode('utf-8')
        
        req = urllib.request.Request(url, data=data, headers=headers)
        
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result['choices'][0]['message']['content'], result['usage']['total_tokens']
        except urllib.error.HTTPError as e:
            if e.code == 429:  # Rate limit
                await asyncio.sleep(5)
                return "Rate limited", 0
            else:
                return f"HTTP {e.code}", 0

# ============================================================================
# COHERENCE CALCULATION
# ============================================================================

class CoherenceCalculator:
    """
    Calculate semantic coherence between texts
    Primary method: SBERT (Reimers & Gurevych, 2019)
    Fallback: Jaccard similarity
    """
    
    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer('all-mpnet-base-v2')
            self.method = "SBERT"
            print("  ✓ Using SBERT for coherence calculation")
        except ImportError:
            self.embedder = None
            self.method = "jaccard"
            print("  ⚠️ SBERT not available, using Jaccard similarity")
    
    def calculate(self, text1: str, text2: str) -> float:
        """
        Calculate coherence score between two texts
        Returns: float between 0 (no coherence) and 1 (perfect coherence)
        """
        
        if not text1 or not text2:
            return 0.0
        
        if self.embedder:
            # SBERT cosine similarity
            try:
                from scipy.spatial.distance import cosine
                emb1 = self.embedder.encode(text1)
                emb2 = self.embedder.encode(text2)
                similarity = 1 - cosine(emb1, emb2)
                return max(0.0, min(1.0, similarity))  # Clamp to [0,1]
            except Exception as e:
                print(f"    SBERT error: {e}, falling back to Jaccard")
                return self._jaccard_similarity(text1, text2)
        else:
            return self._jaccard_similarity(text1, text2)
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Fallback: Jaccard similarity"""
        
        # Tokenize
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Remove stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                    'to', 'for', 'of', 'with', 'by', 'is', 'was', 'are', 'were'}
        
        words1 = words1 - stopwords
        words2 = words2 - stopwords
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

# ============================================================================
# DEGRADATION ANALYSIS
# ============================================================================

class DegradationAnalyzer:
    """
    Analyze how coherence degrades with recursion depth
    Tests for different degradation patterns (linear, exponential, cliff)
    """
    
    @staticmethod
    def analyze_degradation(depths: List[int], coherences: List[float]) -> Dict:
        """
        Fit multiple degradation models and identify pattern
        """
        
        if len(depths) < 3 or len(coherences) < 3:
            return {"pattern": "insufficient_data"}
        
        results = {}
        
        # Convert to numpy arrays
        x = np.array(depths, dtype=float)
        y = np.array(coherences, dtype=float)
        
        # 1. Linear model: y = mx + b
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            results['linear'] = {
                'slope': slope,
                'intercept': intercept,
                'r2': r_value**2,
                'p_value': p_value
            }
        except:
            results['linear'] = None
        
        # 2. Exponential model: y = a * exp(-bx)
        try:
            def exp_func(x, a, b):
                return a * np.exp(-b * x)
            
            # Initial guess
            a0 = y[0] if len(y) > 0 else 1
            b0 = 0.1
            
            popt, pcov = curve_fit(exp_func, x, y, p0=[a0, b0], maxfev=5000)
            
            # Calculate R²
            y_pred = exp_func(x, *popt)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            results['exponential'] = {
                'a': popt[0],
                'b': popt[1],
                'r2': r2
            }
        except:
            results['exponential'] = None
        
        # 3. Find critical depth (where coherence < 0.5)
        critical_depth = None
        for d, c in zip(depths, coherences):
            if c < 0.5:
                critical_depth = d
                break
        
        results['critical_depth'] = critical_depth
        
        # 4. Classify pattern
        pattern = DegradationAnalyzer._classify_pattern(results, depths, coherences)
        results['pattern'] = pattern
        
        return results
    
    @staticmethod
    def _classify_pattern(results: Dict, depths: List, coherences: List) -> str:
        """
        Classify degradation pattern based on fit and characteristics
        """
        
        # Check for stable (minimal change)
        if np.std(coherences) < 0.05:
            return "stable"
        
        # Check for cliff (sudden drop)
        if len(coherences) > 1:
            diffs = np.diff(coherences)
            if np.min(diffs) < -0.3:  # 30% drop in single step
                return "cliff"
        
        # Compare model fits
        linear_r2 = results.get('linear', {}).get('r2', 0) if results.get('linear') else 0
        exp_r2 = results.get('exponential', {}).get('r2', 0) if results.get('exponential') else 0
        
        if exp_r2 > linear_r2 + 0.1:
            return "exponential"
        elif linear_r2 > 0.7:
            slope = results.get('linear', {}).get('slope', 0)
            if abs(slope) < 0.02:
                return "stable"
            else:
                return "gradual"
        else:
            return "irregular"

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

class NonLinearDialogueExperiment:
    """
    Main experimental protocol implementation
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.prompt_generator = PromptGenerator()
        self.coherence_calc = CoherenceCalculator()
        self.degradation_analyzer = DegradationAnalyzer()
        
        # Generate prompts
        self.prompts = self.prompt_generator.generate_prompt_battery()
        
        # Results storage
        self.results_dir = Path("data/nonlinear_results")
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Generate registration hash
        self.registration_hash = self._generate_registration()
        
        print(f"\n  Experiment ID: {self.registration_hash}")
        print(f"  Timestamp: {datetime.now().isoformat()}")
    
    def _generate_registration(self) -> str:
        """Generate unique experiment ID - FIXED"""
        # Just use string representation for hashing
        config_str = str(vars(self.config))
        timestamp = datetime.now().isoformat()
        combined = f"{config_str}{timestamp}"
        return hashlib.sha256(combined.encode()).hexdigest()[:12]
    
    def create_reference_prompt(self,
                               base_prompt: str,
                               history: List[Dict],
                               ref_type: ReferenceType,
                               turn_number: int) -> str:
        """
        Create prompt with specified reference type
        Implements both simple and complex recursion patterns
        """
        
        if ref_type == ReferenceType.LINEAR or not history:
            return base_prompt
        
        elif ref_type == ReferenceType.IMMEDIATE and len(history) >= 1:
            # Reference N-1 (immediate memory)
            ref = history[-1].get('response', '')[:50]
            return f"Following up on '{ref}...': {base_prompt}"
        
        elif ref_type == ReferenceType.SHALLOW and len(history) >= 3:
            # Reference N-2 to N-3 (working memory)
            ref = history[-3].get('response', '')[:50]
            return f"Returning to your point about '{ref}...': {base_prompt}"
        
        elif ref_type == ReferenceType.MEDIUM and len(history) >= 5:
            # Reference N-4 to N-5 (episodic buffer)
            ref = history[-5].get('response', '')[:50]
            return f"Earlier you mentioned '{ref}...'. Related to that: {base_prompt}"
        
        elif ref_type == ReferenceType.DEEP and len(history) >= 7:
            # Reference N-6+ (long-term memory)
            ref = history[-7].get('response', '')[:40]
            return f"Going back to the beginning about '{ref}...': {base_prompt}"
        
        elif ref_type == ReferenceType.CONTRADICTORY and len(history) >= 3:
            # Explicit contradiction
            ref = history[-3].get('response', '')[:50]
            return f"Actually, contrary to '{ref}...', {base_prompt}"
        
        elif ref_type == ReferenceType.BRANCHING and len(history) >= 5:
            # Multiple references
            refs = []
            if len(history) >= 1:
                refs.append(history[-1].get('response', '')[:20])
            if len(history) >= 3:
                refs.append(history[-3].get('response', '')[:20])
            if len(history) >= 5:
                refs.append(history[-5].get('response', '')[:20])
            
            if refs:
                return f"Connecting points about {', '.join(refs)}: {base_prompt}"
        
        return base_prompt
    
    async def run_condition(self,
                           model: ModelInterface,
                           ref_type: ReferenceType) -> Dict:
        """
        Run one experimental condition
        """
        
        print(f"\n  Condition: {ref_type.value}")
        print(f"  " + "="*40)
        
        history = []
        results = {
            'condition': ref_type.value,
            'coherence_scores': [],
            'response_times': [],
            'turn_numbers': [],
            'prompts_used': []
        }
        
        # Run through prompts
        for i, prompt_data in enumerate(self.prompts):
            
            # Create reference-modified prompt
            ref_prompt = self.create_reference_prompt(
                prompt_data['text'],
                history,
                ref_type,
                i
            )
            
            # Generate response
            start_time = time.time()
            
            try:
                response, tokens = await model.generate(ref_prompt, history)
                elapsed = time.time() - start_time
                
                # Calculate coherence
                coherence = self.coherence_calc.calculate(ref_prompt, response)
                
                # Store results
                results['coherence_scores'].append(coherence)
                results['response_times'].append(elapsed)
                results['turn_numbers'].append(i)
                results['prompts_used'].append(prompt_data['id'])
                
                # Update history
                history.append({
                    'turn': i,
                    'prompt_id': prompt_data['id'],
                    'user_text': prompt_data['text'],
                    'ref_prompt': ref_prompt,
                    'response': response,
                    'coherence': coherence,
                    'tokens': tokens
                })
                
                # Progress indicator
                if (i + 1) % 10 == 0:
                    recent_coherence = np.mean(results['coherence_scores'][-10:])
                    print(f"    Progress: {i+1}/{len(self.prompts)} | "
                          f"Recent coherence: {recent_coherence:.3f}")
                
            except KeyboardInterrupt:
                print("\n  ⚠️ Interrupted by user")
                break
            except Exception as e:
                print(f"    Error at prompt {i}: {e}")
                results['coherence_scores'].append(0)
                results['response_times'].append(0)
        
        # Analyze degradation
        if len(results['coherence_scores']) >= 7:
            # Sample at specific depths
            depth_coherences = {}
            for depth in self.config.recursion_depths:
                if depth < len(results['coherence_scores']):
                    depth_coherences[depth] = results['coherence_scores'][depth]
            
            if len(depth_coherences) >= 3:
                degradation = self.degradation_analyzer.analyze_degradation(
                    list(depth_coherences.keys()),
                    list(depth_coherences.values())
                )
                results['degradation_analysis'] = degradation
                
                print(f"\n    Degradation pattern: {degradation['pattern']}")
                if degradation['critical_depth']:
                    print(f"    Critical depth: {degradation['critical_depth']}")
        
        # Summary statistics
        if results['coherence_scores']:
            results['summary'] = {
                'mean': np.mean(results['coherence_scores']),
                'std': np.std(results['coherence_scores']),
                'median': np.median(results['coherence_scores']),
                'min': np.min(results['coherence_scores']),
                'max': np.max(results['coherence_scores']),
                'n': len(results['coherence_scores'])
            }
            
            print(f"\n    Mean coherence: {results['summary']['mean']:.3f}")
            print(f"    Std deviation: {results['summary']['std']:.3f}")
        
        return results
    
    async def run_model(self, provider: str) -> Dict:
        """
        Run complete experiment for one model
        """
        
        print(f"\n{'='*60}")
        print(f"Testing: {provider.upper()}")
        print(f"{'='*60}")
        
        # Initialize model
        model = ModelInterface(provider)
        
        if not model.working:
            print(f"  ✗ {provider} not available")
            return None
        
        model_results = {
            'provider': provider,
            'model_name': model.model_name,
            'timestamp': datetime.now().isoformat(),
            'conditions': {}
        }
        
        # Run each condition
        for ref_type in self.config.reference_types:
            try:
                condition_results = await self.run_condition(model, ref_type)
                model_results['conditions'][ref_type.value] = condition_results
                
                # Brief pause between conditions
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"  Error in condition {ref_type.value}: {e}")
        
        # Calculate comparisons
        model_results['comparisons'] = self._calculate_comparisons(model_results['conditions'])
        
        # Save results
        filename = f"{provider}_{self.registration_hash}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(model_results, f, indent=2, default=str)
        
        print(f"\n  ✓ Results saved: {filename}")
        
        return model_results
    
    def _calculate_comparisons(self, conditions: Dict) -> Dict:
        """
        Calculate effect sizes between conditions
        Following Lakens (2013) for effect size reporting
        """
        
        comparisons = {}
        
        # Get baseline (linear condition)
        baseline = conditions.get('linear', {}).get('coherence_scores', [])
        
        if not baseline:
            return comparisons
        
        baseline_mean = np.mean(baseline)
        baseline_std = np.std(baseline)
        
        # Compare each condition to baseline
        for condition_name, condition_data in conditions.items():
            if condition_name == 'linear':
                continue
            
            scores = condition_data.get('coherence_scores', [])
            
            if scores:
                # Calculate Cohen's d
                treatment_mean = np.mean(scores)
                treatment_std = np.std(scores)
                
                # Pooled standard deviation
                n1, n2 = len(baseline), len(scores)
                pooled_std = np.sqrt(((n1-1)*baseline_std**2 + (n2-1)*treatment_std**2) / (n1+n2-2))
                
                if pooled_std > 0:
                    d = (treatment_mean - baseline_mean) / pooled_std
                    
                    # 95% CI for effect size
                    se = pooled_std * np.sqrt(1/n1 + 1/n2)
                    ci_lower = d - 1.96 * (se/pooled_std)
                    ci_upper = d + 1.96 * (se/pooled_std)
                    
                    # Statistical test
                    t_stat, p_value = stats.ttest_ind(scores, baseline)
                    
                    comparisons[f"{condition_name}_vs_linear"] = {
                        'cohens_d': round(d, 3),
                        'ci_95': [round(ci_lower, 3), round(ci_upper, 3)],
                        'p_value': round(p_value, 4),
                        'significant': p_value < self.config.alpha,
                        'baseline_mean': round(baseline_mean, 3),
                        'treatment_mean': round(treatment_mean, 3),
                        'interpretation': self._interpret_effect(d)
                    }
        
        return comparisons
    
    def _interpret_effect(self, d: float) -> str:
        """Cohen's (1988) benchmarks"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def generate_report(self, all_results: List[Dict]) -> str:
        """
        Generate comprehensive scientific report
        """
        
        report = []
        report.append("="*70)
        report.append("NON-LINEAR DIALOGUE DYNAMICS: EXPERIMENTAL RESULTS")
        report.append("="*70)
        
        report.append(f"\nExperiment ID: {self.registration_hash}")
        report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append(f"Protocol: {len(self.prompts)} prompts × {len(self.config.reference_types)} conditions")
        report.append(f"Coherence method: {self.coherence_calc.method}")
        
        # Hypothesis testing
        report.append("\n" + "="*70)
        report.append("HYPOTHESIS TESTING")
        report.append("="*70)
        
        for result in all_results:
            if not result:
                continue
            
            provider = result['provider']
            report.append(f"\n{provider.upper()}")
            report.append("-"*40)
            
            comparisons = result.get('comparisons', {})
            
            # H1: Does referencing improve coherence?
            report.append("\nH1: Referencing improves coherence")
            
            for comp_name, comp_data in comparisons.items():
                condition = comp_name.split('_vs_')[0]
                d = comp_data['cohens_d']
                p = comp_data['p_value']
                sig = "✓" if comp_data['significant'] else "✗"
                
                direction = "improves" if d > 0 else "degrades"
                report.append(f"  {sig} {condition}: {direction} (d={d}, p={p})")
            
            # H2: Degradation patterns
            report.append("\nH2: Degradation patterns")
            
            for condition_name, condition_data in result['conditions'].items():
                if 'degradation_analysis' in condition_data:
                    pattern = condition_data['degradation_analysis']['pattern']
                    critical = condition_data['degradation_analysis'].get('critical_depth', 'None')
                    report.append(f"  {condition_name}: {pattern} (critical depth: {critical})")
        
        # Overall conclusions
        report.append("\n" + "="*70)
        report.append("CONCLUSIONS")
        report.append("="*70)
        
        # Aggregate effect sizes
        all_effects = []
        for result in all_results:
            if result:
                for comp in result.get('comparisons', {}).values():
                    all_effects.append(comp['cohens_d'])
        
        if all_effects:
            mean_effect = np.mean(all_effects)
            
            if mean_effect > 0.1:
                report.append(f"\n✓ HYPOTHESIS PARTIALLY SUPPORTED")
            elif mean_effect > -0.1:
                report.append(f"\n○ NO CLEAR EFFECT")
            else:
                report.append(f"\n✗ HYPOTHESIS NOT SUPPORTED")
            
            report.append(f"Mean effect size across all comparisons: d={mean_effect:.3f}")
            
            # Interpretation
            if mean_effect < 0:
                report.append("\nExplicit referencing appears to DEGRADE coherence.")
                report.append("This suggests models perform better with linear progression.")
            else:
                report.append("\nExplicit referencing shows minor improvement in coherence.")
                report.append("Effect is smaller than expected from theoretical predictions.")
        
        return "\n".join(report)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """
    Execute the complete experimental protocol
    """
    
    print("\n" + "="*70)
    print("NON-LINEAR DIALOGUE DYNAMICS EXPERIMENT")
    print("Testing explicit conversational referencing")
    print("="*70)
    
    print("\nTheoretical basis:")
    print("  - Clark & Brennan (1991): Grounding theory")
    print("  - Miller (1956): Cognitive capacity limits")
    print("  - Cowan (2001): Working memory constraints")
    
    print("\nExperimental design:")
    print("  - 50 prompts across 10 domains")
    print("  - 5 reference conditions")
    print("  - Testing depths 1-11")
    print("  - Expected runtime: ~20-30 minutes per model")
    
    proceed = input("\nProceed with experiment? (y/n): ")
    if proceed.lower() != 'y':
        print("Experiment cancelled")
        return
    
    # Initialize experiment
    config = ExperimentConfig()
    experiment = NonLinearDialogueExperiment(config)
    
    # Test available models
    print("\nInitializing models...")
    all_results = []
    
    for provider in ["anthropic", "google", "openai"]:
        try:
            result = await experiment.run_model(provider)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"Error with {provider}: {e}")
    
    # Generate report
    if all_results:
        report = experiment.generate_report(all_results)
        print("\n" + report)
        
        # Save report
        report_path = experiment.results_dir / f"report_{experiment.registration_hash}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n✓ Full report saved: {report_path}")
    else:
        print("\n✗ No results collected")

if __name__ == "__main__":
    # Set up environment
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Suppress warning
    
    # Run experiment
    asyncio.run(main())