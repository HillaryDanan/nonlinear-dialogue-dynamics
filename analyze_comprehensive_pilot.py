"""
Comprehensive analysis of 3×3 factorial pilot study
Following APA statistical reporting guidelines (7th edition)
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveAnalyzer:
    """
    Analyzes factorial design: Condition × Provider
    Tests main effects and interactions (Maxwell & Delaney, 2004)
    """
    
    def __init__(self, data_path: Path = Path("data/comprehensive_pilot")):
        self.data_path = data_path
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.conversations = []
        self.results = []
        
    def load_all_data(self) -> List[Dict]:
        """Load all conversation files"""
        
        files = list(self.data_path.glob("*.json"))
        conversations = []
        
        print(f"Found {len(files)} files")
        
        for file in files:
            if "summary" not in file.name:  # Skip summary files
                with open(file) as f:
                    data = json.load(f)
                    conversations.append(data)
                    
        print(f"Loaded {len(conversations)} conversations")
        return conversations
    
    def calculate_coherence(self, response: str, context: str) -> float:
        """
        Calculate semantic coherence using SBERT
        Following Reimers & Gurevych (2019)
        """
        resp_emb = self.model.encode(response)
        ctx_emb = self.model.encode(context)
        
        similarity = 1 - cosine(resp_emb, ctx_emb)
        return similarity
    
    def analyze_conversation(self, conversation: Dict) -> Dict:
        """Analyze single conversation with comprehensive metrics"""
        
        condition = conversation['condition']
        provider = conversation['provider']
        participant = conversation['participant_id']
        messages = conversation['messages']
        
        # Core metrics
        coherence_scores = []
        reference_effectiveness = []
        response_latencies = []
        
        # Analyze each assistant response
        for i, msg in enumerate(messages):
            if msg['role'] == 'assistant' and i > 0:
                user_msg = messages[i-1]
                
                # Calculate coherence
                coherence = self.calculate_coherence(
                    msg['content'],
                    user_msg['content']
                )
                
                # Check reference effectiveness
                if user_msg.get('reference_id') is not None:
                    ref_id = user_msg['reference_id']
                    ref_msg = next((m for m in messages if m['id'] == ref_id), None)
                    
                    if ref_msg:
                        ref_coherence = self.calculate_coherence(
                            msg['content'],
                            ref_msg['content']
                        )
                        reference_effectiveness.append(ref_coherence)
                        
                        # Use maximum coherence (best case)
                        coherence = max(coherence, ref_coherence)
                
                coherence_scores.append(coherence)
        
        # Calculate contradiction rate (simplified - looks for negations)
        contradictions = 0
        for i in range(1, len(messages)):
            if messages[i]['role'] == 'assistant':
                current = messages[i]['content'].lower()
                
                # Check against all previous assistant messages
                for j in range(i):
                    if messages[j]['role'] == 'assistant':
                        previous = messages[j]['content'].lower()
                        
                        # Simple contradiction detection
                        if any(neg in current for neg in ['no,', 'not', 'incorrect', 'actually']):
                            if any(word in previous for word in current.split()[:5]):
                                contradictions += 1
                                break
        
        contradiction_rate = contradictions / max(len([m for m in messages if m['role'] == 'assistant']), 1)
        
        # Topic drift - semantic distance from first to last response
        assistant_messages = [m['content'] for m in messages if m['role'] == 'assistant']
        if len(assistant_messages) > 1:
            first_emb = self.model.encode(assistant_messages[0])
            last_emb = self.model.encode(assistant_messages[-1])
            topic_drift = cosine(first_emb, last_emb)
        else:
            topic_drift = 0
        
        return {
            'participant': participant,
            'condition': condition,
            'provider': provider,
            'coherence_mean': np.mean(coherence_scores) if coherence_scores else 0,
            'coherence_std': np.std(coherence_scores) if coherence_scores else 0,
            'contradiction_rate': contradiction_rate,
            'topic_drift': topic_drift,
            'n_messages': len(messages),
            'n_references': sum(1 for m in messages if m.get('reference_id')),
            'reference_effectiveness': np.mean(reference_effectiveness) if reference_effectiveness else 0
        }
    
    def run_factorial_analysis(self, df: pd.DataFrame):
        """
        Run 3×3 factorial ANOVA
        Following Howell (2012) statistical methods
        """
        
        print("\n" + "="*60)
        print("FACTORIAL ANOVA: Condition × Provider")
        print("="*60)
        
        # Check if we have enough data
        if len(df) < 9:
            print("⚠️  Insufficient data for factorial analysis")
            print(f"   Need at least 9 observations, have {len(df)}")
            return
        
        try:
            import pingouin as pg
            
            # Two-way mixed ANOVA
            # Within: Condition (repeated measures)
            # Between: Provider (if testing across different groups)
            
            # For pilot, might be within-subjects for both
            aov = pg.rm_anova(
                data=df,
                dv='coherence_mean',
                within=['condition', 'provider'],
                subject='participant'
            )
            
            print("\nMain Effects and Interaction:")
            print(aov[['Source', 'F', 'p-unc', 'np2']])
            
            # Post-hoc comparisons for significant effects
            if aov['p-unc'].iloc[0] < 0.05:  # Condition effect
                print("\nPost-hoc: Condition differences")
                posthoc = pg.pairwise_tests(
                    data=df,
                    dv='coherence_mean',
                    within='condition',
                    subject='participant'
                )
                print(posthoc[['A', 'B', 'mean(A)', 'mean(B)', 'p-unc']])
            
            if aov['p-unc'].iloc[1] < 0.05:  # Provider effect
                print("\nPost-hoc: Provider differences")
                posthoc = pg.pairwise_tests(
                    data=df,
                    dv='coherence_mean',
                    within='provider',
                    subject='participant'
                )
                print(posthoc[['A', 'B', 'mean(A)', 'mean(B)', 'p-unc']])
                
        except ImportError:
            print("⚠️  pingouin not installed - using simplified analysis")
            
            # Fallback to simple comparisons
            print("\nCondition Means:")
            print(df.groupby('condition')['coherence_mean'].agg(['mean', 'std', 'count']))
            
            print("\nProvider Means:")
            print(df.groupby('provider')['coherence_mean'].agg(['mean', 'std', 'count']))
            
            print("\nCondition × Provider:")
            print(df.pivot_table(
                values='coherence_mean',
                index='condition',
                columns='provider',
                aggfunc='mean'
            ).round(3))
    
    def calculate_effect_sizes(self, df: pd.DataFrame):
        """
        Calculate comprehensive effect sizes
        Following Lakens (2013) for effect size reporting
        """
        
        print("\n" + "="*60)
        print("EFFECT SIZES (Cohen's d)")
        print("="*60)
        
        # Condition comparisons
        print("\nCondition Effects:")
        conditions = df['condition'].unique()
        
        for i, c1 in enumerate(conditions):
            for c2 in conditions[i+1:]:
                d1 = df[df['condition'] == c1]['coherence_mean'].values
                d2 = df[df['condition'] == c2]['coherence_mean'].values
                
                if len(d1) > 0 and len(d2) > 0:
                    # Cohen's d with pooled SD
                    mean_diff = d1.mean() - d2.mean()
                    pooled_std = np.sqrt((d1.std()**2 + d2.std()**2) / 2)
                    
                    if pooled_std > 0:
                        d = mean_diff / pooled_std
                        
                        # 95% CI for effect size
                        se = pooled_std * np.sqrt(1/len(d1) + 1/len(d2))
                        ci_lower = (mean_diff - 1.96*se) / pooled_std
                        ci_upper = (mean_diff + 1.96*se) / pooled_std
                        
                        print(f"  {c1} vs {c2}:")
                        print(f"    d = {d:.3f} [95% CI: {ci_lower:.3f}, {ci_upper:.3f}]")
                        print(f"    Means: {d1.mean():.3f} vs {d2.mean():.3f}")
        
        # Provider comparisons
        print("\nProvider Effects:")
        providers = df['provider'].unique()
        
        for i, p1 in enumerate(providers):
            for p2 in providers[i+1:]:
                d1 = df[df['provider'] == p1]['coherence_mean'].values
                d2 = df[df['provider'] == p2]['coherence_mean'].values
                
                if len(d1) > 0 and len(d2) > 0:
                    mean_diff = d1.mean() - d2.mean()
                    pooled_std = np.sqrt((d1.std()**2 + d2.std()**2) / 2)
                    
                    if pooled_std > 0:
                        d = mean_diff / pooled_std
                        print(f"  {p1} vs {p2}: d = {d:.3f}")
    
    def analyze_all(self):
        """Run complete analysis"""
        
        print("\n" + "="*60)
        print("COMPREHENSIVE PILOT ANALYSIS")
        print("="*60)
        
        # Load data
        conversations = self.load_all_data()
        
        if not conversations:
            print("No data found! Run pilot first.")
            return None
        
        # Analyze each conversation
        for conv in conversations:
            result = self.analyze_conversation(conv)
            self.results.append(result)
        
        # Create DataFrame
        df = pd.DataFrame(self.results)
        
        # Basic summary
        print("\n" + "="*60)
        print("DESCRIPTIVE STATISTICS")
        print("="*60)
        
        print("\nOverall Summary:")
        print(df[['coherence_mean', 'contradiction_rate', 'topic_drift']].describe().round(3))
        
        # Run factorial analysis
        self.run_factorial_analysis(df)
        
        # Calculate effect sizes
        self.calculate_effect_sizes(df)
        
        # Reference effectiveness
        print("\n" + "="*60)
        print("REFERENCE EFFECTIVENESS")
        print("="*60)
        
        ref_df = df[df['condition'] == 'referenced']
        if not ref_df.empty:
            print(f"Referenced condition coherence: {ref_df['coherence_mean'].mean():.3f}")
            print(f"Reference effectiveness: {ref_df['reference_effectiveness'].mean():.3f}")
            
            # Compare to linear baseline
            linear_df = df[df['condition'] == 'linear']
            if not linear_df.empty:
                improvement = (ref_df['coherence_mean'].mean() - linear_df['coherence_mean'].mean()) / linear_df['coherence_mean'].mean()
                print(f"Improvement over linear: {improvement*100:.1f}%")
        
        # Power analysis for main study
        self.calculate_power_requirements(df)
        
        return df
    
    def calculate_power_requirements(self, df: pd.DataFrame):
        """
        Calculate sample size for main study
        Based on observed pilot effect sizes
        """
        
        print("\n" + "="*60)
        print("POWER ANALYSIS FOR MAIN STUDY")
        print("="*60)
        
        # Get observed effect sizes
        conditions = df['condition'].unique()
        effect_sizes = []
        
        for i, c1 in enumerate(conditions):
            for c2 in conditions[i+1:]:
                d1 = df[df['condition'] == c1]['coherence_mean'].values
                d2 = df[df['condition'] == c2]['coherence_mean'].values
                
                if len(d1) > 1 and len(d2) > 1:
                    mean_diff = abs(d1.mean() - d2.mean())
                    pooled_std = np.sqrt((d1.std()**2 + d2.std()**2) / 2)
                    if pooled_std > 0:
                        effect_sizes.append(mean_diff / pooled_std)
        
        if effect_sizes:
            observed_d = np.median(effect_sizes)  # Use median for robustness
            
            try:
                from statsmodels.stats.power import ttest_power
                
                # Calculate required n for different power levels
                for power in [0.80, 0.90, 0.95]:
                    n = ttest_power(
                        effect_size=observed_d,
                        nobs=None,
                        alpha=0.05/12,  # Bonferroni correction
                        power=power,
                        alternative='two-sided'
                    )
                    print(f"For {power*100:.0f}% power: n = {int(np.ceil(n))}")
                
                print(f"\n(Based on median observed d = {observed_d:.3f})")
                
            except ImportError:
                print("Install statsmodels for power calculation")
        
        # Recommendations
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        mean_coherence = df['coherence_mean'].mean()
        
        print(f"Overall coherence: {mean_coherence:.3f}")
        
        if mean_coherence < 0.3:
            print("⚠️  Low coherence - consider improving prompts")
        elif mean_coherence > 0.6:
            print("✓ Good coherence levels")
        
        # Check if hypothesis supported
        ref_mean = df[df['condition'] == 'referenced']['coherence_mean'].mean()
        linear_mean = df[df['condition'] == 'linear']['coherence_mean'].mean()
        
        if ref_mean > linear_mean:
            print(f"✓ Hypothesis supported: Referenced ({ref_mean:.3f}) > Linear ({linear_mean:.3f})")
        else:
            print(f"✗ Hypothesis not supported: Referenced ({ref_mean:.3f}) ≤ Linear ({linear_mean:.3f})")
            print("  Consider: Strengthening reference prompts")

if __name__ == "__main__":
    analyzer = ComprehensiveAnalyzer()
    df = analyzer.analyze_all()
    
    if df is not None and not df.empty:
        # Save results
        output_path = Path("data/comprehensive_pilot_results.csv")
        df.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to: {output_path}")