"""
FOLLOW-UP STUDY WITH ALL THREE MODELS
======================================
Using what actually works
"""

import os
import sys
import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path

# Add path for imports
sys.path.append('../../src/core')
from actually_working_models import WorkingModels


class FollowUpStudy:
    """Run the reference mechanism study"""
    
    def __init__(self):
        self.models = WorkingModels()
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
    
    def run_study(self):
        """Run simplified study"""
        print("\n" + "="*60)
        print("FOLLOW-UP: REFERENCE MECHANISMS")
        print("="*60)
        
        # Test conditions
        reference_types = {
            "none": "{}",  # No modification
            "implicit": "Building on our discussion, {}",
            "anaphoric": "Regarding that, {}",
            "explicit": "As you mentioned earlier, {}",
            "quoted": "You said 'important point'. Related to that, {}"
        }
        
        test_prompts = [
            "What is consciousness?",
            "Explain climate change",
            "How do neural networks work?",
            "What is quantum entanglement?",
            "Define emergence"
        ]
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'models': {
                'anthropic': self.models.status.anthropic,
                'google': self.models.status.google,
                'openai': self.models.status.openai
            },
            'data': []
        }
        
        # Test each model
        for model_name in ['anthropic', 'google', 'openai']:
            if not getattr(self.models.status, model_name):
                print(f"\n⚠️ Skipping {model_name} (not available)")
                continue
            
            print(f"\n Testing {model_name.upper()}...")
            
            for ref_name, ref_template in reference_types.items():
                print(f"  {ref_name}...", end="")
                coherences = []
                
                for prompt in test_prompts:
                    # Apply reference template
                    modified_prompt = ref_template.format(prompt)
                    
                    # Generate response
                    if model_name == 'anthropic':
                        response, tokens = self.models.generate_anthropic(modified_prompt, 100)
                    elif model_name == 'google':
                        response, tokens = self.models.generate_google(modified_prompt)
                    else:  # openai
                        response, tokens = self.models.generate_openai(modified_prompt, 100)
                    
                    # Calculate coherence
                    coherence = self._calc_coherence(modified_prompt, response)
                    coherences.append(coherence)
                    
                    # Rate limiting pause
                    time.sleep(0.3)
                
                mean_coherence = np.mean(coherences)
                results['data'].append({
                    'model': model_name,
                    'reference_type': ref_name,
                    'mean_coherence': float(mean_coherence),
                    'std_coherence': float(np.std(coherences)),
                    'n': len(coherences)
                })
                
                print(f" {mean_coherence:.3f}")
        
        # Save results
        filename = f"follow_up_{datetime.now():%Y%m%d_%H%M}.json"
        filepath = self.results_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to {filepath}")
        
        # Analysis
        self._analyze(results)
    
    def _calc_coherence(self, prompt: str, response: str) -> float:
        """Simple coherence metric"""
        if not response or "Error" in response:
            return 0.0
        
        # Tokenize and clean
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        
        # Remove stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                    'to', 'for', 'of', 'with', 'by', 'is', 'was', 'are', 'were',
                    'that', 'this', 'it', 'as', 'you', 'said', 'mentioned'}
        
        prompt_words -= stopwords
        response_words -= stopwords
        
        if not prompt_words:
            return 0.0
        
        # Jaccard similarity
        overlap = len(prompt_words & response_words)
        union = len(prompt_words | response_words)
        
        return overlap / union if union > 0 else 0.0
    
    def _analyze(self, results):
        """Analyze results"""
        print("\n" + "="*60)
        print("ANALYSIS")
        print("="*60)
        
        # By reference type across all models
        by_reference = {}
        for item in results['data']:
            ref_type = item['reference_type']
            if ref_type not in by_reference:
                by_reference[ref_type] = []
            by_reference[ref_type].append(item['mean_coherence'])
        
        print("\nMean coherence by reference type (all models):")
        for ref_type in ['none', 'implicit', 'anaphoric', 'explicit', 'quoted']:
            if ref_type in by_reference:
                values = by_reference[ref_type]
                print(f"  {ref_type:10} : {np.mean(values):.3f} (n={len(values)})")
        
        # Find best for each model
        print("\nBest reference type per model:")
        for model in ['anthropic', 'google', 'openai']:
            model_data = [d for d in results['data'] if d['model'] == model]
            if model_data:
                best = max(model_data, key=lambda x: x['mean_coherence'])
                print(f"  {model:10} : {best['reference_type']} ({best['mean_coherence']:.3f})")
        
        # Overall best
        if results['data']:
            overall_best = max(results['data'], key=lambda x: x['mean_coherence'])
            print(f"\n✅ BEST OVERALL: {overall_best['model']} with {overall_best['reference_type']} ({overall_best['mean_coherence']:.3f})")


if __name__ == "__main__":
    study = FollowUpStudy()
    study.run_study()
