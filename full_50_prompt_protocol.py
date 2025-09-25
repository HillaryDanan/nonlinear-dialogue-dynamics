#!/usr/bin/env python3
"""
FULL 50-PROMPT PROTOCOL WITH RECURSION TESTING
Non-Linear Dialogue Dynamics Study

Theoretical Framework:
- Clark & Brennan (1991): Grounding in communication
- Miller (1956): Magic number 7±2 for cognitive limits  
- Cowan (2001): Working memory capacity constraints
- Baddeley (2000): Episodic buffer in working memory

Hypotheses:
H1: Explicit referencing improves coherence (d=0.27)
H2: Models show distinct degradation signatures with depth
H3: Critical depth exists at 5-7 (Miller's limit)
H4: Contradiction handling varies by architecture
"""

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import time

# For coherence calculation
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("⚠️ SBERT not available, using word overlap coherence")

class ReferenceType(Enum):
    """Types of referencing patterns based on cognitive theory"""
    LINEAR = "linear"  # No reference (control)
    SHALLOW = "shallow"  # N-1 (working memory)
    MEDIUM = "medium"  # N-3 (episodic buffer)
    DEEP = "deep"  # N-5, N-7 (long-term memory)
    CONTRADICTORY = "contradictory"  # Reference with contradiction
    BRANCHING = "branching"  # Multiple simultaneous references
    
@dataclass
class PromptBattery:
    """
    50 carefully designed prompts across domains
    Following your friend's suggestion for consistency across models
    """
    prompts: List[str] = field(default_factory=list)
    domains: List[str] = field(default_factory=list)
    complexity_levels: List[int] = field(default_factory=list)
    
    def generate_battery(self) -> List[Dict[str, any]]:
        """
        Generate 50 prompts with metadata
        Balanced across domains and complexity
        """
        
        # 10 domains × 5 question types = 50 prompts
        domains = [
            "quantum computing",
            "neural networks", 
            "climate systems",
            "evolutionary biology",
            "consciousness",
            "economic theory",
            "urban planning",
            "cryptography",
            "protein folding",
            "cosmology"
        ]
        
        # Question types testing different cognitive demands
        question_templates = [
            ("Explain the fundamental principles of {}", 1),  # Basic recall
            ("How does {} relate to information theory?", 2),  # Integration
            ("What contradictions exist in current {} understanding?", 3),  # Contradiction detection
            ("If {} assumptions were inverted, what would change?", 3),  # Counterfactual reasoning
            ("How might {} evolve over the next decade?", 2)  # Projection
        ]
        
        battery = []
        prompt_id = 0
        
        for domain in domains:
            for template, complexity in question_templates:
                prompt = template.format(domain)
                battery.append({
                    'id': prompt_id,
                    'prompt': prompt,
                    'domain': domain,
                    'complexity': complexity,
                    'hash': hashlib.md5(prompt.encode()).hexdigest()[:8]
                })
                prompt_id += 1
        
        return battery

@dataclass
class ExperimentConfig:
    """Pre-registered experimental parameters"""
    n_prompts: int = 50
    recursion_depths: List[int] = field(default_factory=lambda: [1, 3, 5, 7, 9, 11])
    reference_types: List[ReferenceType] = field(default_factory=lambda: list(ReferenceType))
    n_runs: int = 3  # Repeat for reliability
    randomize: bool = False  # Keep consistent per friend's suggestion
    save_intermediate: bool = True
    
    # Statistical parameters
    alpha: float = 0.05 / 6  # Bonferroni correction for 6 conditions
    target_power: float = 0.80
    expected_effect: float = 0.27

class CoherenceCalculator:
    """Calculate coherence with fallback methods"""
    
    def __init__(self):
        if SBERT_AVAILABLE:
            self.embedder = SentenceTransformer('all-mpnet-base-v2')
            self.method = "SBERT"
        else:
            self.embedder = None
            self.method = "word_overlap"
    
    def calculate(self, text1: str, text2: str) -> float:
        """
        Calculate semantic coherence
        SBERT if available, otherwise Jaccard similarity
        """
        
        if not text1 or not text2:
            return 0.0
        
        if self.embedder:
            # SBERT cosine similarity
            from scipy.spatial.distance import cosine
            emb1 = self.embedder.encode(text1)
            emb2 = self.embedder.encode(text2)
            return 1 - cosine(emb1, emb2)
        else:
            # Fallback: Jaccard similarity
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0

class DegradationAnalyzer:
    """
    Analyze how models degrade with recursion depth
    Tests your friend's hypothesis about model-specific signatures
    """
    
    @staticmethod
    def fit_degradation_curves(depths: List[int], coherences: List[float]) -> Dict:
        """
        Fit multiple degradation models
        Following psychometric curve fitting (Wichmann & Hill, 2001)
        """
        
        results = {}
        depths_arr = np.array(depths)
        coherences_arr = np.array(coherences)
        
        # 1. Linear degradation: c = a - b*d
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(depths_arr, coherences_arr)
            results['linear'] = {
                'slope': slope,
                'intercept': intercept,
                'r2': r_value**2,
                'p_value': p_value
            }
        except:
            results['linear'] = None
        
        # 2. Exponential decay: c = a * exp(-b*d)
        try:
            def exp_func(x, a, b):
                return a * np.exp(-b * x)
            
            popt, pcov = curve_fit(exp_func, depths_arr, coherences_arr, 
                                  p0=[1.0, 0.1], maxfev=5000)
            
            # Calculate R²
            residuals = coherences_arr - exp_func(depths_arr, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((coherences_arr - np.mean(coherences_arr))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            results['exponential'] = {
                'a': popt[0],
                'b': popt[1],
                'r2': r2
            }
        except:
            results['exponential'] = None
        
        # 3. Power law: c = a * d^(-b)
        try:
            def power_func(x, a, b):
                return a * np.power(x, -b)
            
            # Filter out zero depths for power law
            nonzero_mask = depths_arr > 0
            if np.any(nonzero_mask):
                popt, _ = curve_fit(power_func, depths_arr[nonzero_mask], 
                                   coherences_arr[nonzero_mask], p0=[1.0, 0.5])
                
                residuals = coherences_arr[nonzero_mask] - power_func(depths_arr[nonzero_mask], *popt)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((coherences_arr[nonzero_mask] - np.mean(coherences_arr[nonzero_mask]))**2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                results['power'] = {
                    'a': popt[0],
                    'b': popt[1],
                    'r2': r2
                }
        except:
            results['power'] = None
        
        # 4. Identify best fit
        best_model = None
        best_r2 = -1
        
        for model_name, model_results in results.items():
            if model_results and model_results.get('r2', -1) > best_r2:
                best_model = model_name
                best_r2 = model_results['r2']
        
        results['best_fit'] = best_model
        
        # 5. Find critical depth (where coherence < 0.5)
        critical_depth = None
        for d, c in zip(depths, coherences):
            if c < 0.5:
                critical_depth = d
                break
        
        results['critical_depth'] = critical_depth
        
        # 6. Classify degradation pattern
        if not results['linear']:
            pattern = 'undefined'
        elif results['linear']['slope'] > -0.02:
            pattern = 'stable'
        elif results['exponential'] and results['exponential']['r2'] > results['linear']['r2'] + 0.1:
            pattern = 'exponential_decay'
        elif critical_depth and critical_depth <= 5:
            pattern = 'early_cliff'
        elif critical_depth and critical_depth > 7:
            pattern = 'late_cliff'
        else:
            pattern = 'gradual_linear'
        
        results['pattern'] = pattern
        
        return results

class RecursionExperiment:
    """
    Main experimental protocol implementation
    Tests both coherence and recursion depth
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.prompt_battery = PromptBattery().generate_battery()
        self.coherence_calc = CoherenceCalculator()
        self.degradation_analyzer = DegradationAnalyzer()
        
        # Results storage
        self.results_path = Path("data/full_protocol_results")
        self.results_path.mkdir(exist_ok=True, parents=True)
        
        # Pre-registration for transparency
        self.registration_time = datetime.now()
        self.registration_hash = self._generate_registration_hash()
        
        print(f"Experiment registered: {self.registration_hash}")
        print(f"Protocol: {len(self.prompt_battery)} prompts × {len(self.config.reference_types)} conditions")
        print(f"Coherence method: {self.coherence_calc.method}")
    
    def _generate_registration_hash(self) -> str:
        """Generate hash of experimental parameters for pre-registration"""
        config_str = json.dumps(asdict(self.config), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]
    
    def create_reference_prompt(self,
                              prompt: str,
                              history: List[Dict],
                              ref_type: ReferenceType,
                              depth: int = None) -> str:
        """
        Create prompt with specified reference type
        Implements your friend's simple vs complex recursion
        """
        
        if ref_type == ReferenceType.LINEAR or not history:
            return prompt
        
        elif ref_type == ReferenceType.SHALLOW:
            # Reference N-1 (working memory)
            if len(history) >= 1:
                ref = history[-1]['content'][:50]
                return f"Building on '{ref}...': {prompt}"
            
        elif ref_type == ReferenceType.MEDIUM:
            # Reference N-3 (episodic buffer)
            if len(history) >= 3:
                ref = history[-3]['content'][:50]
                return f"Returning to your earlier point about '{ref}...': {prompt}"
            
        elif ref_type == ReferenceType.DEEP:
            # Reference N-5 and N-7 (long-term memory)
            refs = []
            if len(history) >= 5:
                refs.append(history[-5]['content'][:30])
            if len(history) >= 7:
                refs.append(history[-7]['content'][:30])
            
            if refs:
                return f"Connecting to earlier points [{', '.join(refs)}]: {prompt}"
            
        elif ref_type == ReferenceType.CONTRADICTORY:
            # Explicit contradiction
            if len(history) >= 3:
                ref = history[-3]['content'][:50]
                return f"Actually, contrary to the earlier claim about '{ref}': {prompt}"
            
        elif ref_type == ReferenceType.BRANCHING:
            # Multiple references (complex recursion)
            refs = []
            for idx in [1, 3, 5]:
                if len(history) >= idx:
                    refs.append(f"point {idx}: {history[-idx]['content'][:20]}")
            
            if refs:
                return f"Synthesizing {' AND '.join(refs)}: {prompt}"
        
        return prompt
    
    async def run_single_model(self, 
                              model_interface,
                              model_name: str) -> Dict:
        """
        Run complete protocol for one model
        """
        
        print(f"\n{'='*60}")
        print(f"Running protocol for: {model_name}")
        print(f"{'='*60}")
        
        results = {
            'model': model_name,
            'timestamp': datetime.now().isoformat(),
            'config': asdict(self.config),
            'conditions': {}
        }
        
        # Test each reference type
        for ref_type in self.config.reference_types:
            print(f"\nCondition: {ref_type.value}")
            
            condition_results = {
                'coherence_scores': [],
                'response_times': [],
                'depths': [],
                'messages': []
            }
            
            history = []
            
            # Run through prompt battery
            for i, prompt_data in enumerate(self.prompt_battery):
                
                # Create referenced version
                ref_prompt = self.create_reference_prompt(
                    prompt_data['prompt'],
                    history,
                    ref_type,
                    depth=i
                )
                
                # Get response
                start_time = time.time()
                
                try:
                    response, tokens = await model_interface.generate(ref_prompt, history)
                    elapsed = time.time() - start_time
                    
                    # Calculate coherence
                    coherence = self.coherence_calc.calculate(ref_prompt, response)
                    
                    # Store results
                    condition_results['coherence_scores'].append(coherence)
                    condition_results['response_times'].append(elapsed)
                    condition_results['depths'].append(i)
                    
                    # Update history
                    history.append({
                        'content': prompt_data['prompt'],
                        'response': response,
                        'coherence': coherence
                    })
                    
                    # Progress indicator
                    if (i + 1) % 10 == 0:
                        mean_coherence = np.mean(condition_results['coherence_scores'][-10:])
                        print(f"  Progress: {i+1}/{len(self.prompt_battery)} | "
                              f"Recent coherence: {mean_coherence:.3f}")
                    
                except Exception as e:
                    print(f"  Error at prompt {i}: {e}")
                    condition_results['coherence_scores'].append(0)
                    condition_results['response_times'].append(0)
                    
                # Rate limiting
                if model_name == "openai":
                    await asyncio.sleep(1.5)  # Respect rate limits
                else:
                    await asyncio.sleep(0.2)
            
            # Analyze degradation for this condition
            if len(condition_results['coherence_scores']) > 0:
                
                # Sample at specific depths for degradation analysis
                depth_coherences = {}
                for depth in self.config.recursion_depths:
                    if depth < len(condition_results['coherence_scores']):
                        depth_coherences[depth] = condition_results['coherence_scores'][depth]
                
                if len(depth_coherences) > 2:
                    degradation = self.degradation_analyzer.fit_degradation_curves(
                        list(depth_coherences.keys()),
                        list(depth_coherences.values())
                    )
                    condition_results['degradation_analysis'] = degradation
                    
                    print(f"\n  Degradation pattern: {degradation.get('pattern', 'unknown')}")
                    print(f"  Critical depth: {degradation.get('critical_depth', 'None')}")
                    print(f"  Best fit model: {degradation.get('best_fit', 'None')}")
            
            results['conditions'][ref_type.value] = condition_results
        
        # Calculate summary statistics
        results['summary'] = self._calculate_summary(results['conditions'])
        
        # Save results
        filename = f"{model_name}_{self.registration_hash}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.results_path / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✓ Results saved: {filename}")
        
        return results
    
    def _calculate_summary(self, conditions_data: Dict) -> Dict:
        """Calculate summary statistics across conditions"""
        
        summary = {
            'mean_coherence_by_condition': {},
            'effect_sizes': {},
            'degradation_patterns': {}
        }
        
        # Mean coherence per condition
        for condition, data in conditions_data.items():
            if data['coherence_scores']:
                summary['mean_coherence_by_condition'][condition] = {
                    'mean': np.mean(data['coherence_scores']),
                    'std': np.std(data['coherence_scores']),
                    'n': len(data['coherence_scores'])
                }
        
        # Calculate effect sizes (Cohen's d) between conditions
        baseline = conditions_data.get(ReferenceType.LINEAR.value, {}).get('coherence_scores', [])
        
        if baseline:
            for condition, data in conditions_data.items():
                if condition != ReferenceType.LINEAR.value and data['coherence_scores']:
                    # Cohen's d
                    treatment = data['coherence_scores']
                    
                    mean_diff = np.mean(treatment) - np.mean(baseline)
                    pooled_std = np.sqrt((np.var(treatment) + np.var(baseline)) / 2)
                    
                    if pooled_std > 0:
                        d = mean_diff / pooled_std
                        
                        # 95% CI for effect size
                        n1, n2 = len(treatment), len(baseline)
                        se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
                        ci_lower = d - 1.96 * se
                        ci_upper = d + 1.96 * se
                        
                        summary['effect_sizes'][f"{condition}_vs_linear"] = {
                            'd': d,
                            'ci_lower': ci_lower,
                            'ci_upper': ci_upper,
                            'interpretation': self._interpret_effect_size(d)
                        }
        
        # Degradation patterns
        for condition, data in conditions_data.items():
            if 'degradation_analysis' in data:
                summary['degradation_patterns'][condition] = data['degradation_analysis'].get('pattern', 'unknown')
        
        return summary
    
    def _interpret_effect_size(self, d: float) -> str:
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
        """Generate scientific report of findings"""
        
        report = []
        report.append("="*60)
        report.append("NON-LINEAR DIALOGUE DYNAMICS: EXPERIMENTAL RESULTS")
        report.append("="*60)
        report.append(f"\nRegistration: {self.registration_hash}")
        report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append(f"Protocol: {len(self.prompt_battery)} prompts × {len(self.config.reference_types)} conditions")
        
        # Aggregate results across models
        report.append("\n" + "="*60)
        report.append("HYPOTHESIS TESTING")
        report.append("="*60)
        
        for result in all_results:
            model = result['model']
            summary = result.get('summary', {})
            
            report.append(f"\n{model.upper()}")
            report.append("-" * 40)
            
            # H1: Referencing improves coherence
            linear_mean = summary['mean_coherence_by_condition'].get('linear', {}).get('mean', 0)
            
            for ref_type in ['shallow', 'medium', 'deep']:
                ref_mean = summary['mean_coherence_by_condition'].get(ref_type, {}).get('mean', 0)
                
                if linear_mean and ref_mean:
                    effect = summary['effect_sizes'].get(f'{ref_type}_vs_linear', {})
                    d = effect.get('d', 0)
                    interpretation = effect.get('interpretation', 'unknown')
                    
                    if d > 0:
                        report.append(f"  ✓ {ref_type}: Improved (d={d:.3f}, {interpretation})")
                    else:
                        report.append(f"  ✗ {ref_type}: Degraded (d={d:.3f}, {interpretation})")
            
            # H2: Degradation patterns
            patterns = summary.get('degradation_patterns', {})
            if patterns:
                report.append(f"\n  Degradation signatures:")
                for condition, pattern in patterns.items():
                    report.append(f"    {condition}: {pattern}")
        
        report.append("\n" + "="*60)
        report.append("CONCLUSIONS")
        report.append("="*60)
        
        # Overall findings
        all_effect_sizes = []
        for result in all_results:
            for effect_key, effect_data in result.get('summary', {}).get('effect_sizes', {}).items():
                all_effect_sizes.append(effect_data['d'])
        
        if all_effect_sizes:
            mean_effect = np.mean(all_effect_sizes)
            
            if mean_effect > 0:
                report.append(f"\n✓ HYPOTHESIS SUPPORTED")
                report.append(f"  Mean effect size: d={mean_effect:.3f}")
            else:
                report.append(f"\n✗ HYPOTHESIS NOT SUPPORTED")
                report.append(f"  Mean effect size: d={mean_effect:.3f}")
                report.append("  Referencing appears to degrade coherence")
        
        return "\n".join(report)

# Main execution
async def run_full_protocol():
    """Execute the complete experimental protocol"""
    
    print("\n" + "="*60)
    print("NON-LINEAR DIALOGUE DYNAMICS")
    print("Full 50-Prompt Protocol with Recursion Testing")
    print("="*60)
    
    # Initialize experiment
    config = ExperimentConfig()
    experiment = RecursionExperiment(config)
    
    print(f"\nExperimental design:")
    print(f"  Prompts: {config.n_prompts}")
    print(f"  Conditions: {len(config.reference_types)}")
    print(f"  Recursion depths: {config.recursion_depths}")
    print(f"  Expected runtime: ~15-30 minutes per model")
    
    proceed = input("\nProceed with full protocol? (y/n): ")
    if proceed.lower() != 'y':
        print("Cancelled")
        return
    
    # Import model interfaces
    from data_collector_all_models import UniversalModelInterface
    
    # Test available models
    all_results = []
    
    for provider in ["anthropic", "google"]:  # Skip OpenAI if broken
        print(f"\n{'='*60}")
        print(f"Testing {provider}...")
        
        try:
            model = UniversalModelInterface(provider)
            
            if model.working:
                result = await experiment.run_single_model(model, provider)
                all_results.append(result)
            else:
                print(f"✗ {provider} not available")
                
        except Exception as e:
            print(f"✗ {provider} error: {e}")
    
    # Generate report
    if all_results:
        report = experiment.generate_report(all_results)
        print("\n" + report)
        
        # Save report
        report_path = experiment.results_path / f"report_{experiment.registration_hash}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n✓ Report saved: {report_path}")
    else:
        print("\n✗ No results collected")

if __name__ == "__main__":
    asyncio.run(run_full_protocol())