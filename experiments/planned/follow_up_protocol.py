"""
Follow-Up Study: Implicit vs Explicit Reference Mechanisms
===========================================================
Building on falsified hypothesis from initial experiment

Research Questions:
1. Do implicit references perform better than explicit?
2. What is the optimal reference strategy for each model?
3. How do reference styles interact with task complexity?

Theoretical Framework:
- Jurafsky & Martin (2023): Computational discourse analysis
- Tenney et al. (2019): BERT attention patterns
- Rogers et al. (2020): Primer on neural language models
"""

import asyncio
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum

# Import our clean model interface
import sys
sys.path.append('src/core')
from model_interface import ModelFactory

# Import coherence calculator
sys.path.append('src/analysis')


@dataclass
class StudyConfig:
    """Pre-registered study parameters"""
    
    # Sample size from power analysis
    n_prompts_per_condition: int = 30  # Reduced from 50 for efficiency
    n_models: int = 3  # Anthropic, Google, OpenAI
    
    # Experimental conditions
    reference_styles: List[str] = field(default_factory=lambda: [
        "none",           # No reference (control)
        "implicit_only",  # "As discussed..."
        "anaphoric",      # "it", "that", "this"
        "nominal",        # "the previous point"
        "quotative",      # Direct quotes
        "semantic"        # Conceptual links
    ])
    
    # Task complexity levels
    complexity_levels: List[str] = field(default_factory=lambda: [
        "factual",        # Simple retrieval
        "analytical",     # Relationship analysis
        "creative"        # Generation tasks
    ])
    
    # Statistical parameters
    alpha: float = 0.008  # Bonferroni: 0.05/6
    expected_effect: float = 0.4  # Based on pilot


class ReferenceGenerator:
    """Generate different reference styles"""
    
    @staticmethod
    def create_reference(
        style: str,
        current_prompt: str,
        history: List[Dict],
        target_distance: int = 3
    ) -> str:
        """Create prompt with specified reference style"""
        
        if not history or len(history) < target_distance:
            return current_prompt
        
        # Get target response
        target = history[-target_distance].get('response', '')[:100]
        target_concept = history[-target_distance].get('main_concept', '')
        
        if style == "none":
            return current_prompt
            
        elif style == "implicit_only":
            return f"Building on our earlier discussion, {current_prompt}"
            
        elif style == "anaphoric":
            return f"Regarding that, {current_prompt}"
            
        elif style == "nominal":
            return f"About the previous point on {target_concept}, {current_prompt}"
            
        elif style == "quotative":
            snippet = target[:50] + "..."
            return f'You mentioned "{snippet}". {current_prompt}'
            
        elif style == "semantic":
            # Link concepts without explicit reference
            return f"In a related area, {current_prompt}"
        
        return current_prompt


class PromptBank:
    """Generate prompts by complexity level"""
    
    PROMPTS = {
        "factual": [
            "What is the capital of {}?",
            "When was {} invented?",
            "Who discovered {}?",
            "What is the chemical formula for {}?",
            "How many {} are there?",
        ],
        "analytical": [
            "What are the relationships between {} and {}?",
            "How does {} influence {}?",
            "Compare and contrast {} with {}.",
            "What patterns exist in {}?",
            "Analyze the structure of {}.",
        ],
        "creative": [
            "Design a new approach to {}.",
            "Create a metaphor for {}.",
            "Imagine a scenario where {}.",
            "Propose an innovation in {}.",
            "Generate a story about {}.",
        ]
    }
    
    TOPICS = [
        "quantum mechanics", "neural networks", "climate change",
        "protein folding", "dark matter", "consciousness",
        "evolution", "cryptography", "democracy", "creativity"
    ]
    
    @classmethod
    def generate_prompts(cls, complexity: str, n: int) -> List[str]:
        """Generate n prompts of specified complexity"""
        
        import random
        random.seed(42)  # Reproducibility
        
        templates = cls.PROMPTS[complexity]
        prompts = []
        
        for i in range(n):
            template = random.choice(templates)
            
            # Fill in topics
            n_blanks = template.count("{}")
            topics = random.sample(cls.TOPICS, n_blanks)
            
            prompt = template.format(*topics)
            prompts.append(prompt)
        
        return prompts


class FollowUpExperiment:
    """Main experimental protocol"""
    
    def __init__(self, config: StudyConfig):
        self.config = config
        self.results_dir = Path("experiments/ongoing/follow_up_results")
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
    async def run_condition(
        self,
        model,
        reference_style: str,
        complexity: str
    ) -> Dict[str, Any]:
        """Run one experimental condition"""
        
        print(f"    Condition: {reference_style} × {complexity}")
        
        # Generate prompts
        prompts = PromptBank.generate_prompts(
            complexity, 
            self.config.n_prompts_per_condition
        )
        
        history = []
        coherence_scores = []
        response_times = []
        
        for i, prompt in enumerate(prompts):
            # Apply reference style
            ref_prompt = ReferenceGenerator.create_reference(
                reference_style,
                prompt,
                history,
                target_distance=3
            )
            
            # Generate response
            start_time = datetime.now()
            response, tokens = await model.generate(ref_prompt, history)
            elapsed = (datetime.now() - start_time).total_seconds()
            
            # Calculate coherence (simplified for now)
            # In production, use SBERT
            coherence = self._calculate_coherence(ref_prompt, response)
            
            coherence_scores.append(coherence)
            response_times.append(elapsed)
            
            # Update history
            history.append({
                'prompt': prompt,
                'ref_prompt': ref_prompt,
                'response': response,
                'main_concept': prompt.split()[2] if len(prompt.split()) > 2 else ""
            })
            
            # Progress
            if (i + 1) % 10 == 0:
                mean_coherence = np.mean(coherence_scores[-10:])
                print(f"      Progress: {i+1}/{len(prompts)} | Coherence: {mean_coherence:.3f}")
        
        return {
            'reference_style': reference_style,
            'complexity': complexity,
            'coherence_scores': coherence_scores,
            'mean_coherence': np.mean(coherence_scores),
            'std_coherence': np.std(coherence_scores),
            'response_times': response_times
        }
    
    def _calculate_coherence(self, prompt: str, response: str) -> float:
        """Simplified coherence calculation"""
        # In production, use SBERT
        # For now, use word overlap
        
        if not prompt or not response:
            return 0.0
        
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        
        # Remove stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        prompt_words -= stopwords
        response_words -= stopwords
        
        if not prompt_words or not response_words:
            return 0.0
        
        overlap = len(prompt_words & response_words)
        total = len(prompt_words | response_words)
        
        return overlap / total if total > 0 else 0.0
    
    async def run_model(self, provider: str) -> Dict[str, Any]:
        """Run complete experiment for one model"""
        
        print(f"\n  Testing: {provider.upper()}")
        print("  " + "="*40)
        
        # Initialize model
        model = await ModelFactory.create(provider)
        
        if not model:
            print(f"  ✗ {provider} not available")
            return None
        
        results = {
            'provider': provider,
            'timestamp': datetime.now().isoformat(),
            'conditions': []
        }
        
        # Run all conditions
        for reference_style in self.config.reference_styles:
            for complexity in self.config.complexity_levels:
                try:
                    condition_results = await self.run_condition(
                        model,
                        reference_style,
                        complexity
                    )
                    results['conditions'].append(condition_results)
                    
                    # Brief pause
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    print(f"    Error: {e}")
        
        # Analyze results
        results['analysis'] = self._analyze_results(results['conditions'])
        
        # Save
        filename = f"{provider}_follow_up_{datetime.now():%Y%m%d_%H%M}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"  ✓ Results saved: {filename}")
        
        return results
    
    def _analyze_results(self, conditions: List[Dict]) -> Dict:
        """Analyze experimental results"""
        
        analysis = {
            'by_reference_style': {},
            'by_complexity': {},
            'interaction_effects': {}
        }
        
        # Group by reference style
        for style in self.config.reference_styles:
            style_data = [c for c in conditions if c['reference_style'] == style]
            if style_data:
                coherences = [c['mean_coherence'] for c in style_data]
                analysis['by_reference_style'][style] = {
                    'mean': np.mean(coherences),
                    'std': np.std(coherences)
                }
        
        # Group by complexity
        for complexity in self.config.complexity_levels:
            complexity_data = [c for c in conditions if c['complexity'] == complexity]
            if complexity_data:
                coherences = [c['mean_coherence'] for c in complexity_data]
                analysis['by_complexity'][complexity] = {
                    'mean': np.mean(coherences),
                    'std': np.std(coherences)
                }
        
        # Find optimal combination
        best_condition = max(conditions, key=lambda x: x['mean_coherence'])
        analysis['optimal'] = {
            'reference_style': best_condition['reference_style'],
            'complexity': best_condition['complexity'],
            'coherence': best_condition['mean_coherence']
        }
        
        return analysis
    
    async def run_all(self):
        """Run experiment on all models"""
        
        print("\n" + "="*60)
        print("FOLLOW-UP STUDY: REFERENCE MECHANISMS")
        print("="*60)
        
        print("\nExperimental Design:")
        print(f"  - {len(self.config.reference_styles)} reference styles")
        print(f"  - {len(self.config.complexity_levels)} complexity levels")
        print(f"  - {self.config.n_prompts_per_condition} prompts per condition")
        print(f"  - Testing 3 models (Anthropic, Google, OpenAI)")
        
        all_results = []
        
        for provider in ['anthropic', 'google', 'openai']:
            try:
                result = await self.run_model(provider)
                if result:
                    all_results.append(result)
            except Exception as e:
                print(f"Error with {provider}: {e}")
        
        # Generate report
        self._generate_report(all_results)
        
        return all_results
    
    def _generate_report(self, results: List[Dict]):
        """Generate summary report"""
        
        print("\n" + "="*60)
        print("SUMMARY REPORT")
        print("="*60)
        
        for model_results in results:
            if not model_results:
                continue
                
            provider = model_results['provider']
            analysis = model_results['analysis']
            
            print(f"\n{provider.upper()}")
            print("-"*40)
            
            # Best reference style
            best_style = max(
                analysis['by_reference_style'].items(),
                key=lambda x: x[1]['mean']
            )
            print(f"Optimal reference: {best_style[0]} (M={best_style[1]['mean']:.3f})")
            
            # Complexity effects
            print("\nComplexity effects:")
            for complexity, stats in analysis['by_complexity'].items():
                print(f"  {complexity}: M={stats['mean']:.3f}, SD={stats['std']:.3f}")
            
            # Overall optimal
            optimal = analysis['optimal']
            print(f"\nBest combination: {optimal['reference_style']} × {optimal['complexity']}")
            print(f"Coherence: {optimal['coherence']:.3f}")


async def main():
    """Run the follow-up study"""
    
    config = StudyConfig()
    experiment = FollowUpExperiment(config)
    
    await experiment.run_all()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("FOLLOW-UP STUDY PROTOCOL")
    print("Testing implicit vs explicit reference mechanisms")
    print("="*70)
    
    print("\nThis study will:")
    print("1. Test 6 reference styles across 3 complexity levels")
    print("2. Use all 3 models (Anthropic, Google, OpenAI)")
    print("3. Generate actionable recommendations")
    
    proceed = input("\nProceed with follow-up study? (y/n): ")
    if proceed.lower() == 'y':
        asyncio.run(main())
    else:
        print("Study cancelled")
