"""
COMPLETE FOLLOW-UP STUDY: ALL THREE MODELS
===========================================
Finally, all this shit works!
"""

import os
import sys
import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path

# Add path for model interface
sys.path.insert(0, '../../src/core')

# Import working models
from dotenv import load_dotenv
load_dotenv(override=True)  # Reload with new key

import anthropic
import google.generativeai as genai
from openai import OpenAI


class CompleteFollowUpStudy:
    """The actual fucking study with all three models"""
    
    def __init__(self):
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize models
        self.anthropic_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.google_model = genai.GenerativeModel('gemini-2.0-flash')
        
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        print("✅ All three models initialized!")
    
    def run(self):
        """Run the complete study"""
        print("\n" + "="*70)
        print("FOLLOW-UP STUDY: IMPLICIT VS EXPLICIT REFERENCE MECHANISMS")
        print("Testing how reference style affects coherence")
        print("="*70)
        
        # Reference conditions based on your original findings
        conditions = {
            "linear": "{}",  # Baseline (your original finding: best)
            "implicit_only": "Building on our discussion, {}",
            "anaphoric": "Regarding that, {}",
            "nominal": "About the previous point, {}",
            "quotative": "As you said earlier, {}",
            "semantic": "In a related area, {}",
            "contradictory": "Actually, contrary to before, {}"
        }
        
        # Test prompts covering different complexity levels
        prompts = [
            # Factual
            "What is photosynthesis?",
            "Define machine learning",
            
            # Analytical  
            "How does consciousness emerge from neurons?",
            "What causes economic inequality?",
            
            # Creative
            "Design a sustainable city",
            "Create a new theory of time"
        ]
        
        all_results = []
        
        # Test each model
        models = [
            ('anthropic', self.generate_anthropic),
            ('google', self.generate_google),
            ('openai', self.generate_openai)
        ]
        
        for model_name, generate_func in models:
            print(f"\n{'='*50}")
            print(f"Testing {model_name.upper()}")
            print('='*50)
            
            model_results = []
            
            for cond_name, template in conditions.items():
                print(f"\n  {cond_name}:", end="")
                coherences = []
                
                for prompt in prompts:
                    # Apply reference template
                    modified_prompt = template.format(prompt)
                    
                    # Generate response
                    response, tokens = generate_func(modified_prompt)
                    
                    # Calculate coherence
                    coherence = self.calculate_coherence(modified_prompt, response)
                    coherences.append(coherence)
                    
                    print(".", end="", flush=True)
                    
                    # Rate limiting
                    time.sleep(0.5 if model_name == 'openai' else 0.3)
                
                mean_coh = np.mean(coherences)
                std_coh = np.std(coherences)
                
                print(f" {mean_coh:.3f} (±{std_coh:.3f})")
                
                model_results.append({
                    'condition': cond_name,
                    'mean': mean_coh,
                    'std': std_coh,
                    'n': len(coherences),
                    'raw': coherences
                })
            
            all_results.append({
                'model': model_name,
                'results': model_results
            })
        
        # Save complete results
        output = {
            'experiment': 'follow_up_reference_mechanisms',
            'timestamp': datetime.now().isoformat(),
            'hypothesis': 'Testing if implicit references perform better than explicit',
            'based_on': 'Initial finding that explicit references degrade coherence',
            'models': ['anthropic', 'google', 'openai'],
            'conditions': list(conditions.keys()),
            'data': all_results
        }
        
        filename = f"follow_up_complete_{datetime.now():%Y%m%d_%H%M}.json"
        filepath = self.results_dir / filename
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✅ Complete results saved to {filepath}")
        
        # Analysis
        self.analyze(all_results)
    
    def generate_anthropic(self, prompt: str) -> tuple:
        """Generate with Anthropic"""
        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-5-haiku-20241022",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.7
            )
            text = response.content[0].text
            tokens = response.usage.input_tokens + response.usage.output_tokens
            return text, tokens
        except Exception as e:
            print(f"Anthropic error: {e}")
            return "", 0
    
    def generate_google(self, prompt: str) -> tuple:
        """Generate with Google"""
        try:
            response = self.google_model.generate_content(prompt)
            text = response.text
            tokens = len(prompt.split()) + len(text.split())
            return text, tokens
        except Exception as e:
            print(f"Google error: {e}")
            return "", 0
    
    def generate_openai(self, prompt: str) -> tuple:
        """Generate with OpenAI"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.7
            )
            text = response.choices[0].message.content
            tokens = response.usage.total_tokens
            return text, tokens
        except Exception as e:
            print(f"OpenAI error: {e}")
            return "", 0
    
    def calculate_coherence(self, prompt: str, response: str) -> float:
        """Calculate semantic coherence"""
        if not response:
            return 0.0
        
        # Simple but effective: Jaccard similarity with cleaning
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        
        # Remove stopwords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'by', 'is', 'was', 'are', 'were', 'be',
            'that', 'this', 'it', 'as', 'you', 'said', 'mentioned', 'earlier',
            'about', 'regarding', 'point', 'previous', 'our', 'discussion'
        }
        
        prompt_words = prompt_words - stopwords
        response_words = response_words - stopwords
        
        if not prompt_words or not response_words:
            return 0.0
        
        # Jaccard similarity
        intersection = len(prompt_words & response_words)
        union = len(prompt_words | response_words)
        
        return intersection / union if union > 0 else 0.0
    
    def analyze(self, all_results):
        """Comprehensive analysis"""
        print("\n" + "="*70)
        print("ANALYSIS: REFERENCE MECHANISM EFFECTS")
        print("="*70)
        
        # 1. Overall effect by condition
        print("\n1. MEAN COHERENCE BY CONDITION (all models):")
        print("-"*50)
        
        condition_aggregates = {}
        for model_data in all_results:
            for result in model_data['results']:
                cond = result['condition']
                if cond not in condition_aggregates:
                    condition_aggregates[cond] = []
                condition_aggregates[cond].extend(result['raw'])
        
        baseline = np.mean(condition_aggregates.get('linear', [0]))
        
        for cond in ['linear', 'implicit_only', 'anaphoric', 'nominal', 
                     'quotative', 'semantic', 'contradictory']:
            if cond in condition_aggregates:
                values = condition_aggregates[cond]
                mean = np.mean(values)
                effect = (mean - baseline) / baseline * 100  # Percent change
                print(f"  {cond:15} : {mean:.3f}  ({effect:+.1f}% vs baseline)")
        
        # 2. Best condition per model
        print("\n2. OPTIMAL CONDITION PER MODEL:")
        print("-"*50)
        
        for model_data in all_results:
            model = model_data['model']
            best = max(model_data['results'], key=lambda x: x['mean'])
            worst = min(model_data['results'], key=lambda x: x['mean'])
            print(f"  {model:10} : Best={best['condition']} ({best['mean']:.3f}), "
                  f"Worst={worst['condition']} ({worst['mean']:.3f})")
        
        # 3. Consistency across models
        print("\n3. CROSS-MODEL AGREEMENT:")
        print("-"*50)
        
        rankings = {}
        for model_data in all_results:
            model = model_data['model']
            sorted_conditions = sorted(model_data['results'], 
                                     key=lambda x: x['mean'], 
                                     reverse=True)
            rankings[model] = [r['condition'] for r in sorted_conditions]
        
        # Check if top conditions agree
        top_conditions = [rankings[m][0] for m in rankings]
        if len(set(top_conditions)) == 1:
            print(f"  ✅ All models agree: {top_conditions[0]} is best")
        else:
            print(f"  ⚠️  Models disagree on best condition:")
            for model, ranking in rankings.items():
                print(f"      {model}: {ranking[0]}")
        
        # 4. Key finding
        print("\n4. KEY FINDING:")
        print("-"*50)
        
        if baseline > np.mean([v for k, v in condition_aggregates.items() if k != 'linear']):
            print("  ✅ REPLICATES ORIGINAL: Linear (no reference) performs best")
            print("  → Explicit referencing degrades coherence across all models")
        else:
            best_overall = max(condition_aggregates.items(), key=lambda x: np.mean(x[1]))
            print(f"  ⚠️  NEW FINDING: {best_overall[0]} outperforms linear")
            print("  → Some reference types may improve coherence")
        
        # 5. Statistical significance (simple t-test)
        print("\n5. STATISTICAL TESTS (vs linear baseline):")
        print("-"*50)
        
        from scipy import stats
        
        linear_scores = condition_aggregates.get('linear', [])
        if linear_scores:
            for cond in ['implicit_only', 'anaphoric', 'quotative', 'contradictory']:
                if cond in condition_aggregates:
                    t_stat, p_value = stats.ttest_ind(
                        linear_scores, 
                        condition_aggregates[cond]
                    )
                    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                    print(f"  {cond:15} : t={t_stat:.2f}, p={p_value:.4f} {sig}")


if __name__ == "__main__":
    print("\nStarting complete follow-up study with all three models...")
    study = CompleteFollowUpStudy()
    study.run()
