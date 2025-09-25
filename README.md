# nonlinear-dialogue-dynamics

## Testing How Non-Linear Conversational Referencing Affects LLM Understanding and Coherence

### Overview

This repository investigates whether allowing explicit reference to earlier conversation points (non-linear dialogue) improves large language model (LLM) performance compared to strictly linear conversational progression. Current LLM interfaces enforce sequential turn-taking, contrasting with natural human communication patterns that frequently reference and revisit earlier discourse segments (Clark & Brennan, 1991).

### Research Question

Does implementing threaded reply capabilities—allowing specific responses to earlier conversation points—improve model understanding, reduce context confusion, and enhance reciprocal mirroring in human-AI interactions?

### Hypothesis

Allowing explicit reference to earlier conversation points will improve:
1. Topic coherence maintenance (d>0.4 expected effect size)
2. Reference resolution accuracy 
3. Reduction in contradictory responses (20-30% decrease expected)
4. User-reported conversation quality

**Note**: These are theoretical predictions requiring empirical validation through the experiments in this repository.

### Methodology

#### Study Design
Within-subjects comparative study across three conditions:
- **Linear Baseline**: Standard sequential conversation
- **Explicit Reference**: Ability to quote/reply to specific earlier messages
- **Hybrid**: Both linear and referenced responses intermixed

#### Metrics
- Coherence score: Cosine similarity between response and referenced content
- Contradiction rate: Frequency of conflicting statements  
- Topic drift: Semantic distance from initial subject
- Response latency: Time to first token

### Repository Structure

```
nonlinear-dialogue-dynamics/
├── README.md
├── theory/
│   └── theoretical_framework.md
├── experiments/
│   ├── pilot_study.py
│   ├── main_experiment.py
│   └── prompts.py
├── analysis/
│   ├── coherence_metrics.py
│   └── statistical_tests.py
├── data/
│   └── [experimental data files]
└── results/
    └── [analysis outputs]
```

### Theoretical Foundation

Human conversation rarely proceeds linearly. Research demonstrates:
- Speakers frequently reference earlier topics through anaphora and deixis (Clark & Schaefer, 1989)
- Working memory constraints necessitate explicit referencing when returning to topics (Cowan, 2001)
- Transformer attention already attempts to reference earlier segments (Vig, 2019)

Making these references explicit through UI affordances could improve attention weight allocation and provide clearer interpretability signals.

### Implementation

Using available APIs (GPT-3.5, Claude, Gemini) with prompt engineering to simulate threaded replies:

```python
# Example prompt structure for referenced conversation
REFERENCE_PROMPT = """
Previous statement (Message #{ref_id}): "{ref_content}"
User is now responding to that specific statement with: "{current_input}"
Provide a response that directly addresses the referenced message.
"""
```

### Current Status

**Working Theory**: This framework represents theoretical predictions requiring empirical validation. Initial pilot testing in progress.

### Requirements

- Python 3.8+
- API access to: OpenAI, Anthropic, and/or Google AI
- Dependencies: see requirements.txt

### Contributing

This is active research. Issues and pull requests welcome. Please maintain scientific rigor in all contributions.

### Citation

If you use this work, please cite:
```
@misc{danan2025nonlinear,
  author = {Danan, Hillary},
  title = {Non-Linear Dialogue Dynamics: Testing Threaded Conversation in LLMs},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/HillaryDanan/nonlinear-dialogue-dynamics}
}
```

### References

Clark, H. H., & Brennan, S. E. (1991). Grounding in communication. *Perspectives on socially shared cognition*, 13, 127-149.

Clark, H. H., & Schaefer, E. F. (1989). Contributing to discourse. *Cognitive Science*, 13(2), 259-294.

Cowan, N. (2001). The magical number 4 in short-term memory. *Behavioral and Brain Sciences*, 24(1), 87-114.

Vig, J. (2019). A multiscale visualization of attention in the transformer model. *ACL System Demonstrations*.

### License

MIT License - See LICENSE file for details

### Contact

Hillary Danan - [GitHub Profile](https://github.com/HillaryDanan)

---

*Repository created: September 2025*  
*Last updated: September 2025*
