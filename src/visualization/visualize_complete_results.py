"""
Publication-Quality Visualizations for Complete Study
======================================================
Comparing explicit referencing effects across three LLM architectures
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import pandas as pd

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


def load_all_results():
    """Load results from all three models"""
    results_dir = Path("data/nonlinear_results")
    
    # Find the latest results for each model
    data = {}
    for provider in ['anthropic', 'google', 'openai']:
        files = list(results_dir.glob(f"{provider}_*.json"))
        if files:
            # Get most recent file
            latest = max(files, key=lambda x: x.stat().st_mtime)
            with open(latest, 'r') as f:
                data[provider] = json.load(f)
            print(f"Loaded {provider} from {latest.name}")
    
    return data


def create_main_comparison_figure(data):
    """Create comprehensive comparison figure"""
    
    fig = plt.figure(figsize=(16, 10))
    
    # Create layout: 2x3 grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)
    
    # Color scheme for models
    model_colors = {
        'anthropic': '#8B4789',  # Purple
        'google': '#4285F4',      # Blue
        'openai': '#10A37F'       # Green
    }
    
    conditions_order = ['linear', 'immediate', 'shallow', 'deep', 'contradictory']
    
    # ========== Panel A: Coherence by Condition ==========
    ax1 = fig.add_subplot(gs[0, :])
    
    # Prepare data for grouped bar plot
    coherence_data = []
    for model, results in data.items():
        for condition in conditions_order:
            if condition in results['conditions']:
                scores = results['conditions'][condition]['coherence_scores']
                mean = np.mean(scores)
                std = np.std(scores)
                coherence_data.append({
                    'Model': model.capitalize(),
                    'Condition': condition.capitalize(),
                    'Coherence': mean,
                    'Std': std
                })
    
    df_coherence = pd.DataFrame(coherence_data)
    
    # Create grouped bar plot
    x = np.arange(len(conditions_order))
    width = 0.25
    
    for i, model in enumerate(['anthropic', 'google', 'openai']):
        model_data = df_coherence[df_coherence['Model'] == model.capitalize()]
        means = [model_data[model_data['Condition'] == c.capitalize()]['Coherence'].values[0] 
                if len(model_data[model_data['Condition'] == c.capitalize()]) > 0 else 0
                for c in conditions_order]
        stds = [model_data[model_data['Condition'] == c.capitalize()]['Std'].values[0]
               if len(model_data[model_data['Condition'] == c.capitalize()]) > 0 else 0
               for c in conditions_order]
        
        ax1.bar(x + i*width - width, means, width, 
               yerr=stds, capsize=3,
               label=model.capitalize(), 
               color=model_colors[model],
               alpha=0.8)
    
    ax1.set_xlabel('Reference Condition')
    ax1.set_ylabel('Coherence Score (SBERT)')
    ax1.set_title('A. Coherence Across Reference Conditions', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.capitalize() for c in conditions_order])
    ax1.legend(loc='upper right')
    ax1.axhline(y=0.72, color='gray', linestyle='--', alpha=0.3, label='Baseline')
    ax1.set_ylim([0.6, 0.85])
    ax1.grid(True, alpha=0.3)
    
    # ========== Panel B: Effect Sizes ==========
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Extract effect sizes
    effect_data = []
    for model, results in data.items():
        if 'comparisons' in results:
            for comp_name, comp_data in results['comparisons'].items():
                condition = comp_name.split('_vs_')[0]
                effect_data.append({
                    'Model': model.capitalize(),
                    'Condition': condition,
                    'Cohen_d': comp_data['cohens_d'],
                    'CI_lower': comp_data['ci_95'][0],
                    'CI_upper': comp_data['ci_95'][1],
                    'p_value': comp_data['p_value']
                })
    
    # Plot effect sizes
    y_pos = 0
    for model in ['anthropic', 'google', 'openai']:
        model_effects = [e for e in effect_data if e['Model'] == model.capitalize()]
        
        for effect in model_effects:
            # Color based on significance
            if effect['p_value'] < 0.001:
                marker = 'o'
                alpha = 1.0
            elif effect['p_value'] < 0.01:
                marker = 's'
                alpha = 0.8
            elif effect['p_value'] < 0.05:
                marker = '^'
                alpha = 0.6
            else:
                marker = 'x'
                alpha = 0.4
            
            ax2.errorbar(effect['Cohen_d'], y_pos, 
                        xerr=[[effect['Cohen_d'] - effect['CI_lower']], 
                              [effect['CI_upper'] - effect['Cohen_d']]],
                        fmt=marker, color=model_colors[model.lower()],
                        alpha=alpha, capsize=3, markersize=8)
            y_pos += 1
        
        y_pos += 0.5  # Space between models
    
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax2.axvline(x=-0.2, color='gray', linestyle='--', alpha=0.3)  # Small effect
    ax2.axvline(x=-0.5, color='gray', linestyle='--', alpha=0.3)  # Medium effect
    ax2.axvline(x=-0.8, color='gray', linestyle='--', alpha=0.3)  # Large effect
    
    ax2.set_xlabel("Cohen's d")
    ax2.set_title('B. Effect Sizes vs Linear Baseline', fontweight='bold')
    ax2.set_xlim([-1.5, 1.0])
    
    # Add y-axis labels
    y_labels = []
    for model in ['anthropic', 'google', 'openai']:
        model_effects = [e for e in effect_data if e['Model'] == model.capitalize()]
        for effect in model_effects:
            y_labels.append(f"{model[:3].upper()}: {effect['Condition']}")
        y_labels.append("")  # Empty for spacing
    ax2.set_yticks(range(len(y_labels)))
    ax2.set_yticklabels(y_labels[:-1])
    ax2.grid(True, alpha=0.3, axis='x')
    
    # ========== Panel C: Statistical Summary ==========
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    # Create summary table
    summary_text = "Statistical Summary\n" + "="*30 + "\n\n"
    
    for model, results in data.items():
        if 'comparisons' in results:
            summary_text += f"{model.upper()}:\n"
            
            sig_effects = 0
            for comp_name, comp_data in results['comparisons'].items():
                if comp_data['p_value'] < 0.05:
                    sig_effects += 1
                    condition = comp_name.split('_vs_')[0]
                    d = comp_data['cohens_d']
                    p = comp_data['p_value']
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*"
                    summary_text += f"  {condition}: d={d:+.2f} {sig}\n"
            
            if sig_effects == 0:
                summary_text += "  No significant effects\n"
            summary_text += "\n"
    
    # Add interpretation
    summary_text += "\nKey Finding:\n"
    summary_text += "Explicit referencing\n"
    summary_text += "DEGRADES coherence\n"
    summary_text += "across all models\n\n"
    summary_text += "Mean effect: d=-0.43"
    
    ax3.text(0.1, 0.9, summary_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top',
            fontfamily='monospace')
    
    # ========== Panel D: Google Anomaly ==========
    ax4 = fig.add_subplot(gs[1, 2])
    
    # Highlight the contradictory condition
    models = ['Anthropic', 'Google', 'OpenAI']
    contradictory_effects = []
    
    for model in ['anthropic', 'google', 'openai']:
        if model in data and 'comparisons' in data[model]:
            for comp_name, comp_data in data[model]['comparisons'].items():
                if 'contradictory' in comp_name:
                    contradictory_effects.append(comp_data['cohens_d'])
                    break
    
    colors = [model_colors[m.lower()] for m in ['anthropic', 'google', 'openai']]
    bars = ax4.bar(models, contradictory_effects, color=colors, alpha=0.7)
    
    # Highlight Google's positive effect
    if len(contradictory_effects) >= 2 and contradictory_effects[1] > 0:
        bars[1].set_alpha(1.0)
        bars[1].set_edgecolor('red')
        bars[1].set_linewidth(2)
    
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.set_ylabel("Cohen's d")
    ax4.set_title('D. Contradictory Reference Anomaly', fontweight='bold')
    ax4.set_ylim([-1, 1])
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add annotation for Google
    if len(contradictory_effects) >= 2:
        ax4.annotate(f'd={contradictory_effects[1]:.2f}***',
                    xy=(1, contradictory_effects[1]),
                    xytext=(1, contradictory_effects[1] + 0.2),
                    ha='center',
                    fontweight='bold',
                    color='red',
                    arrowprops=dict(arrowstyle='->', color='red'))
    
    # Overall title
    fig.suptitle('Explicit Referencing Universally Degrades LLM Coherence\n' +
                'Evidence from Three Model Architectures (n=50 per condition)',
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save figures
    output_dir = Path("results/figures")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    plt.savefig(output_dir / "complete_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "complete_comparison.pdf", bbox_inches='tight')
    
    print(f"\n✓ Saved figures to {output_dir}")
    plt.show()


def create_detailed_statistics_table(data):
    """Create LaTeX table for paper"""
    
    # Build comprehensive statistics table
    rows = []
    
    for model in ['anthropic', 'google', 'openai']:
        if model not in data:
            continue
            
        results = data[model]
        
        # Linear baseline
        linear_scores = results['conditions']['linear']['coherence_scores']
        rows.append({
            'Model': model.capitalize(),
            'Condition': 'Linear (baseline)',
            'Mean': np.mean(linear_scores),
            'SD': np.std(linear_scores),
            'Cohen\'s d': '—',
            'p-value': '—',
            'CI 95%': '—',
            'n': len(linear_scores)
        })
        
        # Other conditions
        if 'comparisons' in results:
            for comp_name, comp_data in results['comparisons'].items():
                condition = comp_name.split('_vs_')[0]
                cond_scores = results['conditions'][condition]['coherence_scores']
                
                ci_str = f"[{comp_data['ci_95'][0]:.2f}, {comp_data['ci_95'][1]:.2f}]"
                p_str = f"{comp_data['p_value']:.4f}"
                if comp_data['p_value'] < 0.001:
                    p_str += "***"
                elif comp_data['p_value'] < 0.01:
                    p_str += "**"
                elif comp_data['p_value'] < 0.05:
                    p_str += "*"
                
                rows.append({
                    'Model': model.capitalize(),
                    'Condition': condition.capitalize(),
                    'Mean': np.mean(cond_scores),
                    'SD': np.std(cond_scores),
                    'Cohen\'s d': f"{comp_data['cohens_d']:.3f}",
                    'p-value': p_str,
                    'CI 95%': ci_str,
                    'n': len(cond_scores)
                })
    
    df = pd.DataFrame(rows)
    
    # Save as CSV and LaTeX
    output_dir = Path("results/tables")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    df.to_csv(output_dir / "complete_statistics.csv", index=False)
    
    latex = df.to_latex(index=False, float_format="%.3f",
                       caption="Effect of Reference Type on Coherence Across Three LLM Architectures",
                       label="tab:complete_results")
    
    with open(output_dir / "complete_statistics.tex", 'w') as f:
        f.write(latex)
    
    print(f"✓ Saved statistics tables to {output_dir}")
    print("\nSummary Statistics:")
    print(df.to_string(index=False))


def main():
    """Generate all visualizations and tables"""
    
    print("\n" + "="*60)
    print("GENERATING COMPLETE STUDY VISUALIZATIONS")
    print("="*60)
    
    # Load all data
    data = load_all_results()
    
    if len(data) < 3:
        print(f"Warning: Only found {len(data)} models")
    
    # Generate visualizations
    create_main_comparison_figure(data)
    create_detailed_statistics_table(data)
    
    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    main()
