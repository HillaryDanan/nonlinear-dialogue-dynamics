"""
Visualization of Non-Linear Dialogue Dynamics Results
======================================================
Publication-quality figures following Tufte (2001) principles
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import pandas as pd

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'


def load_anthropic_results():
    """Load validated Anthropic results"""
    
    # Find the results file
    results_dir = Path("data/nonlinear_results")
    anthropic_files = list(results_dir.glob("anthropic_*.json"))
    
    if not anthropic_files:
        print("No Anthropic results found!")
        return None
    
    with open(anthropic_files[0], 'r') as f:
        return json.load(f)


def create_coherence_comparison_plot(data):
    """Create main coherence comparison figure"""
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Extract conditions
    conditions = data['conditions']
    
    # 1. Box plot comparison
    ax = axes[0]
    condition_names = []
    coherence_data = []
    
    for condition_name, condition_data in conditions.items():
        if 'coherence_scores' in condition_data:
            condition_names.append(condition_name.title())
            coherence_data.append(condition_data['coherence_scores'])
    
    bp = ax.boxplot(coherence_data, labels=condition_names, patch_artist=True)
    
    # Color code by effect direction
    colors = ['lightblue', 'salmon', 'salmon', 'salmon', 'salmon']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Coherence Score')
    ax.set_title('A. Coherence by Reference Type')
    ax.axhline(y=0.72, color='blue', linestyle='--', alpha=0.5, label='Baseline')
    ax.set_ylim([0.5, 0.85])
    ax.legend()
    
    # 2. Effect size forest plot
    ax = axes[1]
    comparisons = data.get('comparisons', {})
    
    effect_sizes = []
    ci_lower = []
    ci_upper = []
    labels = []
    
    for comp_name, comp_data in comparisons.items():
        condition = comp_name.split('_vs_')[0]
        effect_sizes.append(comp_data['cohens_d'])
        ci_lower.append(comp_data['ci_95'][0])
        ci_upper.append(comp_data['ci_95'][1])
        labels.append(condition.title())
    
    y_pos = np.arange(len(labels))
    
    # Plot effect sizes
    ax.errorbar(effect_sizes, y_pos, 
                xerr=[np.array(effect_sizes) - np.array(ci_lower),
                      np.array(ci_upper) - np.array(effect_sizes)],
                fmt='o', capsize=5, capthick=2)
    
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.axvline(x=-0.2, color='gray', linestyle='--', alpha=0.3, label='Small')
    ax.axvline(x=-0.5, color='gray', linestyle='--', alpha=0.3, label='Medium')
    ax.axvline(x=-0.8, color='gray', linestyle='--', alpha=0.3, label='Large')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Cohen's d (vs Linear)")
    ax.set_title('B. Effect Sizes')
    ax.set_xlim([-1.2, 0.4])
    
    # Add significance markers
    for i, (comp_name, comp_data) in enumerate(comparisons.items()):
        if comp_data['significant']:
            ax.text(effect_sizes[i] + 0.05, i, '***' if comp_data['p_value'] < 0.001 else '*',
                   va='center', fontsize=12)
    
    # 3. Degradation patterns
    ax = axes[2]
    
    for condition_name, condition_data in conditions.items():
        if 'coherence_scores' in condition_data:
            scores = condition_data['coherence_scores'][:20]  # First 20 turns
            turns = range(1, len(scores) + 1)
            
            # Smooth with rolling mean
            window = 3
            if len(scores) >= window:
                scores_smooth = pd.Series(scores).rolling(window, center=True).mean()
                ax.plot(turns, scores_smooth, label=condition_name.title(), alpha=0.8)
    
    ax.set_xlabel('Turn Number')
    ax.set_ylabel('Coherence Score')
    ax.set_title('C. Degradation Over Time')
    ax.legend(loc='lower left', fontsize=8)
    ax.set_ylim([0.5, 0.85])
    
    plt.suptitle('Explicit Referencing Degrades LLM Coherence', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save
    output_dir = Path("results/figures")
    output_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_dir / "main_results.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "main_results.pdf", bbox_inches='tight')
    
    print(f"✓ Saved figures to {output_dir}")
    plt.show()


def create_statistical_summary_table(data):
    """Create LaTeX table for paper"""
    
    conditions = data['conditions']
    comparisons = data.get('comparisons', {})
    
    # Build table data
    table_data = []
    
    # Linear baseline
    linear = conditions.get('linear', {})
    if 'summary' in linear:
        table_data.append({
            'Condition': 'Linear (baseline)',
            'Mean': linear['summary']['mean'],
            'SD': linear['summary']['std'],
            'Cohen\'s d': '—',
            'p-value': '—',
            'N': linear['summary']['n']
        })
    
    # Other conditions
    for comp_name, comp_data in comparisons.items():
        condition_name = comp_name.split('_vs_')[0]
        condition = conditions.get(condition_name, {})
        
        if 'summary' in condition:
            table_data.append({
                'Condition': condition_name.title(),
                'Mean': condition['summary']['mean'],
                'SD': condition['summary']['std'],
                'Cohen\'s d': comp_data['cohens_d'],
                'p-value': comp_data['p_value'],
                'N': condition['summary']['n']
            })
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Format for LaTeX
    latex_table = df.to_latex(index=False, float_format="%.3f",
                              caption="Effect of Reference Type on Coherence",
                              label="tab:main_results")
    
    # Save
    output_dir = Path("results/tables")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    with open(output_dir / "main_results.tex", 'w') as f:
        f.write(latex_table)
    
    # Also save as CSV
    df.to_csv(output_dir / "main_results.csv", index=False)
    
    print(f"✓ Saved tables to {output_dir}")
    print("\nSummary Statistics:")
    print(df.to_string(index=False))
    
    return df


def main():
    """Generate all visualizations"""
    
    print("\n" + "="*60)
    print("VISUALIZING EXPERIMENTAL RESULTS")
    print("="*60)
    
    # Load data
    data = load_anthropic_results()
    
    if not data:
        print("No data to visualize!")
        return
    
    # Create visualizations
    create_coherence_comparison_plot(data)
    create_statistical_summary_table(data)
    
    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    main()
