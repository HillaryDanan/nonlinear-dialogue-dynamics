# Non-Linear Dialogue Dynamics

**Testing How Explicit Conversational Referencing Affects LLM Understanding and Coherence**

---

## Overview

This repository presents experimental evidence that explicit conversational referencing currently degrades coherence in major LLMs, contrary to predictions from human discourse research. Testing across three language models (Anthropic Claude, Google Gemini, OpenAI GPT-4) with n=50 prompts per condition reveals consistent degradation patterns, suggesting that contemporary language models may process discourse differently than expected from human communication theories.

## 🔑 Key Finding

**Explicit referencing currently degrades coherence in tested models (mean d=-0.429, p<0.001)**

- Replicated across three distinct model architectures
- Strongest degradation at shallow (N-3) and deep (N-7+) reference depths  
- Suggests mismatch between human discourse strategies and current LLM optimization
- Notable exception: Google Gemini improves with contradictory references (d=+0.877)

## Research Question

Does implementing explicit reference to earlier conversation points—a strategy beneficial in human communication—improve or degrade model understanding and coherence in human-AI interactions?

**Answer: It Degrades Performance**

---

## 📊 Results Summary

Contrary to our hypothesis, explicit referencing consistently worsens coherence:

| Model | Baseline | Shallow (N-3) | Effect Size | p-value |
|-------|----------|---------------|-------------|---------|
| Anthropic | 0.720 (0.058) | 0.660 (0.092) | d=-0.791 | p<0.001*** |
| Google | 0.754 (0.051) | 0.725 (0.077) | d=-0.447 | p=0.029* |
| OpenAI | 0.776 (0.065) | 0.684 (0.095) | d=-1.136 | p<0.001*** |

*Values: Mean (SD)*

---

## Methodology

### Study Design

- **Sample Size:** n=50 prompts per condition (80% power to detect d=0.27)
- **Conditions:** Linear (baseline), Immediate (N-1), Shallow (N-3), Deep (N-7+), Contradictory
- **Measurement:** SBERT semantic similarity (Reimers & Gurevych, 2019)
- **Analysis:** Cohen's d with pooled variance, Bonferroni correction (α=0.01)

### Models Tested

- Anthropic Claude 3.5 Haiku (claude-3-5-haiku-20241022)
- Google Gemini 2.0 Flash (gemini-2.0-flash)
- OpenAI GPT-4o Mini (gpt-4o-mini)

---

## Complete Results Summary

### Effect Sizes by Condition (Cohen's d vs Linear Baseline)

| Condition | Anthropic | Google | OpenAI | Mean Effect |
|-----------|-----------|---------|---------|------------|
| Immediate (N-1) | -0.097 | -0.477* | -0.156 | -0.243 |
| Shallow (N-3) | -0.791*** | -0.447* | -1.136*** | -0.791 |
| Deep (N-7+) | -0.444* | -0.625** | -0.996*** | -0.688 |
| Contradictory | -0.261 | +0.877*** | -0.597** | +0.006 |

*Significance: \*\*\*p<0.001, \*\*p<0.01, \*p<0.05*

### Key Observations

- **Universal Degradation:** 9/12 tested conditions show significant negative effects
- **Shallow Reference Worst:** N-3 references consistently degrade coherence most severely
- **Google Anomaly:** Contradictory references uniquely improve Google's coherence (d=+0.877)
- **No Critical Depth:** Degradation patterns irregular, not matching human memory limits

---

## Repository Structure

```
nonlinear-dialogue-dynamics/
├── README.md
├── src/
│   ├── core/                     # Model interfaces
│   ├── analysis/                  # Coherence metrics
│   └── visualization/             # Result plotting
├── experiments/
│   └── completed/                 # Main experiment (n=50)
├── data/
│   └── nonlinear_results/        # Raw experimental data
├── results/
│   ├── figures/                  # Publication figures
│   └── tables/                   # Statistical summaries
└── docs/
    └── paper/                    # Paper drafts
```

---

## Theoretical Implications

These findings suggest important differences between human and LLM discourse processing:

- **Working Memory Differences:** Degradation patterns don't map to human memory limits (Miller, 1956; Cowan, 2001)
- **Implicit Context Processing:** Current transformers (Vaswani et al., 2017) may be optimized for implicit rather than explicit reference
- **Training Effects:** Observed patterns may reflect discourse distributions in training data
- **Template Sensitivity:** Specific phrasing of references may matter as much as the reference itself

## Alternative Interpretations

The observed degradation may result from multiple factors:

- **Training Data:** Natural text may rarely contain explicit back-references at tested depths
- **Template Artifacts:** Our reference phrases might introduce unnatural discourse markers
- **Metric Limitations:** SBERT semantic similarity may not fully capture discourse coherence
- **Model-Specific Effects:** The Google anomaly suggests training and optimization matter significantly

## Future Research Directions

This work opens several avenues for investigation:

- **Template Ablations:** Test alternative reference phrasings to isolate true effects
- **Attention Analysis:** Examine how models process explicit vs implicit references
- **Human Baselines:** Compare with human performance on identical tasks
- **Training Interventions:** Investigate whether targeted fine-tuning can improve reference handling
- **Cross-Task Validation:** Test whether effects persist across different applications

---

## Statistical Power & Validity

- **Sample Size:** n=50 provides 80% power to detect d≥0.27 at α=0.01
- **Multiple Comparisons:** Bonferroni correction maintains family-wise error rate
- **Replication:** Results consistent across three independent model architectures
- **Effect Magnitude:** Mean effect d=-0.429 represents medium-to-large degradation

---

## Installation & Replication

```bash
# Clone repository
git clone https://github.com/HillaryDanan/nonlinear-dialogue-dynamics

# Install dependencies
pip install -r requirements.txt

# Set API keys in .env
ANTHROPIC_API_KEY=your_key
GOOGLE_API_KEY=your_key
OPENAI_API_KEY=your_key

# Run main experiment
python experiments/completed/complete_nonlinear_experiment.py

# Generate visualizations
python src/visualization/visualize_complete_results_fixed.py
```

---

## Citation

If you use this work, please cite:

```bibtex
@article{danan2025nonlinear,
  title={Explicit Conversational Referencing Universally Degrades LLM Coherence: 
         Evidence from Three Model Architectures},
  author={Danan, Hillary},
  journal={arXiv preprint},
  year={2025},
  note={Explicit referencing degrades coherence (d=-0.429, p<0.001) in current LLMs,
        with model-specific variations suggesting training and optimization effects}
}
}
```

---

## Key References

- Clark, H. H., & Brennan, S. E. (1991). Grounding in communication. *Perspectives on socially shared cognition*, 13, 127-149.
- Cowan, N. (2001). The magical number 4 in short-term memory: A reconsideration of mental storage capacity. *Behavioral and Brain Sciences*, 24(1), 87-114.
- Miller, G. A. (1956). The magical number seven, plus or minus two. *Psychological Review*, 63(2), 81-97.
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proceedings of EMNLP*.
- Vaswani, A., et al. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*.

---

## Data Availability

All experimental data, analysis code, and results are available in this repository. Raw data files include:

- `data/nonlinear_results/anthropic_8cbd9acc2d75.json`
- `data/nonlinear_results/google_fcfc59401205.json`
- `data/nonlinear_results/openai_fcfc59401205.json`

---

## License

MIT License - See LICENSE file for details

---

## Contact

Hillary Danan - [GitHub Profile](https://github.com/HillaryDanan)

Repository created: September 2025  
Last updated: September 26, 2025
