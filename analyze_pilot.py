"""
Pilot data analysis - Calculate preliminary effect sizes
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from typing import Dict, List, Tuple

class PilotAnalyzer:
    """Analyze pilot study results"""
    
    def __init__(self, data_path: Path = Path("data/pilot")):
        self.data_path = data_path
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.results = []
        
    def load_conversations(self) -> List[Dict]:
        """Load all conversation files"""
        conversations = []
        
        for file in self.data_path.glob("*.json"):
            with open(file) as f:
                data = json.load(f)
                conversations.append(data)
                print(f"Loaded: {file.name}")
        
        return conversations
    
    def calculate_coherence(self, response: str, context: str) -> float:
        """Calculate semantic coherence between response and context"""
        resp_emb = self.model.encode(response)
        ctx_emb = self.model.encode(context)
        
        # Cosine similarity (0 to 1)
        similarity = 1 - cosine(resp_emb, ctx_emb)
        return similarity
    
    def analyze_conversation(self, conversation: Dict) -> Dict:
        """Analyze single conversation"""
        
        condition = conversation['condition']
        participant = conversation['participant_id']
        messages = conversation['messages']
        
        coherence_scores = []
        reference_effectiveness = []
        
        for i, msg in enumerate(messages):
            if msg['role'] == 'assistant' and i > 0:
                # Get the user message this responds to
                user_msg = messages[i-1]
                
                # Base coherence
                coherence = self.calculate_coherence(
                    msg['content'], 
                    user_msg['content']
                )
                
                # If there's a reference, check if it improved coherence
                if user_msg.get('reference_id') is not None:
                    # Find referenced message
                    ref_id = user_msg['reference_id']
                    ref_msg = next((m for m in messages if m['id'] == ref_id), None)
                    
                    if ref_msg:
                        # Coherence with referenced content
                        ref_coherence = self.calculate_coherence(
                            msg['content'],
                            ref_msg['content']
                        )
                        reference_effectiveness.append(ref_coherence)
                        
                        # Use combined coherence
                        coherence = max(coherence, ref_coherence)
                
                coherence_scores.append(coherence)
        
        return {
            'participant': participant,
            'condition': condition,
            'coherence_mean': np.mean(coherence_scores) if coherence_scores else 0,
            'coherence_std': np.std(coherence_scores) if coherence_scores else 0,
            'n_messages': len(messages),
            'n_references': sum(1 for m in messages if m.get('reference_id')),
            'reference_effectiveness': np.mean(reference_effectiveness) if reference_effectiveness else 0
        }
    
    def analyze_all(self):
        """Analyze all conversations"""
        
        conversations = self.load_conversations()
        
        if not conversations:
            print("No data found! Run pilot first.")
            return
        
        print(f"\nAnalyzing {len(conversations)} conversations...")
        
        for conv in conversations:
            result = self.analyze_conversation(conv)
            self.results.append(result)
        
        # Convert to DataFrame
        df = pd.DataFrame(self.results)
        
        # Summary by condition
        print("\n" + "="*50)
        print("COHERENCE BY CONDITION")
        print("="*50)
        
        summary = df.groupby('condition').agg({
            'coherence_mean': ['mean', 'std', 'count'],
            'n_references': 'sum'
        }).round(3)
        
        print(summary)
        
        # Calculate effect sizes
        print("\n" + "="*50)
        print("EFFECT SIZES (Cohen's d)")
        print("="*50)
        
        conditions = df['condition'].unique()
        
        for i, c1 in enumerate(conditions):
            for c2 in conditions[i+1:]:
                scores1 = df[df['condition'] == c1]['coherence_mean'].values
                scores2 = df[df['condition'] == c2]['coherence_mean'].values
                
                if len(scores1) > 0 and len(scores2) > 0:
                    # Cohen's d
                    mean_diff = scores1.mean() - scores2.mean()
                    pooled_std = np.sqrt((scores1.std()**2 + scores2.std()**2) / 2)
                    
                    if pooled_std > 0:
                        d = mean_diff / pooled_std
                        
                        # Interpretation
                        if abs(d) < 0.2:
                            interp = "negligible"
                        elif abs(d) < 0.5:
                            interp = "small"
                        elif abs(d) < 0.8:
                            interp = "medium"
                        else:
                            interp = "large"
                        
                        print(f"{c1} vs {c2}: d={d:.3f} ({interp})")
                    else:
                        print(f"{c1} vs {c2}: Cannot calculate (no variance)")
        
        # Check if referenced condition shows improvement
        print("\n" + "="*50)
        print("REFERENCE EFFECTIVENESS")
        print("="*50)
        
        ref_data = df[df['condition'] == 'referenced']
        if not ref_data.empty:
            print(f"Referenced condition mean coherence: {ref_data['coherence_mean'].mean():.3f}")
            print(f"Average reference effectiveness: {ref_data['reference_effectiveness'].mean():.3f}")
        
        # Power calculation for main study
        print("\n" + "="*50)
        print("POWER ANALYSIS FOR MAIN STUDY")
        print("="*50)
        
        # Get the largest effect size
        effect_sizes = []
        for i, c1 in enumerate(conditions):
            for c2 in conditions[i+1:]:
                scores1 = df[df['condition'] == c1]['coherence_mean'].values
                scores2 = df[df['condition'] == c2]['coherence_mean'].values
                
                if len(scores1) > 0 and len(scores2) > 0:
                    mean_diff = abs(scores1.mean() - scores2.mean())
                    pooled_std = np.sqrt((scores1.std()**2 + scores2.std()**2) / 2)
                    if pooled_std > 0:
                        effect_sizes.append(mean_diff / pooled_std)
        
        if effect_sizes:
            observed_d = np.mean(effect_sizes)
            
            from statsmodels.stats.power import ttest_power
            
            # Calculate required n for observed effect
            required_n = ttest_power(
                effect_size=observed_d,
                nobs=None,
                alpha=0.05/12,  # Bonferroni
                power=0.8,
                alternative='two-sided'
            )
            
            print(f"Observed mean effect size: d={observed_d:.3f}")
            print(f"Required n for 80% power: {int(np.ceil(required_n))}")
            
            # Reality check
            if observed_d < 0.2:
                print("\n⚠️  WARNING: Effect size is very small")
                print("Consider:")
                print("- Increasing prompt differentiation between conditions")
                print("- Refining task design to better elicit differences")
                print("- Checking if implementation is working as intended")
        
        return df

if __name__ == "__main__":
    analyzer = PilotAnalyzer()
    df = analyzer.analyze_all()
    
    if df is not None and not df.empty:
        # Save results
        output_path = Path("data/pilot_analysis.csv")
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        
        print("\n" + "="*50)
        print("RECOMMENDATIONS")
        print("="*50)
        
        mean_coherence = df['coherence_mean'].mean()
        
        if mean_coherence < 0.3:
            print("⚠️  Low overall coherence - check prompt quality")
        elif mean_coherence > 0.7:
            print("✓ Good coherence levels")
        
        ref_condition = df[df['condition'] == 'referenced']
        if not ref_condition.empty:
            if ref_condition['coherence_mean'].mean() > df['coherence_mean'].mean():
                print("✓ Referenced condition shows improvement!")
            else:
                print("⚠️  Referenced condition not showing expected improvement")
                print("   Consider strengthening the reference prompt structure")