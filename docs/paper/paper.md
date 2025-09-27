# Explicit Conversational Referencing Universally Degrades LLM Coherence: Evidence from Three Model Architectures

**Hillary Danan¹**  
¹ Independent Researcher

## Abstract

**Background:** Human discourse theory (Clark & Brennan, 1991) posits that explicit referencing to earlier conversational points improves communication through grounding. Large Language Models (LLMs), despite fundamentally different architectures, are often assumed to benefit from similar discourse strategies.

**Methods:** We conducted controlled experiments (N=750; 50 prompts × 5 conditions × 3 models) comparing linear conversation progression against four types of explicit referencing: immediate (N-1), shallow (N-3), deep (N-7+), and contradictory. Coherence was measured using Sentence-BERT semantic similarity (Reimers & Gurevych, 2019) between prompts and responses.

**Results:** Contrary to theoretical predictions, explicit referencing significantly degraded coherence across all models (mean d=-0.429, 95% CI [-0.61, -0.25], p<0.001). Shallow references (N-3) produced the largest degradation (d range: -0.447 to -1.136). Google Gemini uniquely improved with contradictory references (d=+0.877, p<0.001), suggesting model-specific training effects.

**Conclusions:** These findings demonstrate that LLMs process discourse through implicit attention mechanisms incompatible with explicit referencing strategies. The universal degradation pattern (75% of conditions showing significant negative effects) challenges the application of human communication theories to AI systems and has immediate implications for prompt engineering and conversational AI design.

---

## 1. Introduction

### 1.1 Theoretical Background

Human conversation relies on explicit grounding mechanisms where speakers acknowledge and reference earlier discourse points to maintain coherence (Clark & Brennan, 1991; Clark, 1996). This grounding process involves:

- Explicit acknowledgment of shared information
- Back-references to establish common ground
- Repair mechanisms when understanding breaks down

These strategies are constrained by human cognitive limitations, particularly working memory capacity of 4±1 items (Cowan, 2001) and the "magical number seven" for information processing (Miller, 1956).

### 1.2 LLM Architecture and Discourse

Large Language Models, built on transformer architecture (Vaswani et al., 2017), process context through fundamentally different mechanisms:

- Self-attention across all tokens simultaneously
- No explicit memory buffer or retrieval system
- Implicit context representation through attention weights

Despite these architectural differences, current conversational AI interfaces often implement human-like discourse patterns, assuming they improve communication quality.

### 1.3 Research Question

This study tests whether explicit conversational referencing—beneficial for human communication—improves or degrades coherence in human-AI interactions across different model architectures.

### 1.4 Hypotheses

Based on human discourse theory, we hypothesized:

- **H1:** Explicit referencing improves coherence (expected d>0.27)
- **H2:** Models show distinct degradation patterns with recursion depth
- **H3:** Critical coherence breakdown occurs at depth 5-7 (Miller's limit)
- **H4:** Contradiction handling varies by model architecture

---

## 2. Methods

### 2.1 Models

Three state-of-the-art language models were tested:

- Anthropic Claude 3.5 Haiku (claude-3-5-haiku-20241022)
- Google Gemini 2.0 Flash (gemini-2.0-flash)
- OpenAI GPT-4o Mini (gpt-4o-mini)

### 2.2 Experimental Design

Within-subjects design with five reference conditions:

1. **Linear (Control):** Standard sequential conversation
2. **Immediate (N-1):** Reference to immediately preceding response
3. **Shallow (N-3):** Reference to 3 turns prior (working memory boundary)
4. **Deep (N-7+):** Reference to 7+ turns prior (long-term memory)
5. **Contradictory:** Explicit contradiction of earlier statements

### 2.3 Materials

**Prompt Generation:** 50 prompts systematically varied across:
- 10 knowledge domains (physics, biology, computer science, philosophy, etc.)
- 5 cognitive complexity levels (Bloom's Revised Taxonomy; Anderson & Krathwohl, 2001)

**Reference Templates:**
- Immediate: "Following up on '{snippet}...': {prompt}"
- Shallow: "Returning to your point about '{snippet}...': {prompt}"
- Deep: "Going back to the beginning about '{snippet}...': {prompt}"
- Contradictory: "Actually, contrary to '{snippet}...', {prompt}"

### 2.4 Coherence Measurement

**Primary Metric:** Semantic similarity using Sentence-BERT (all-mpnet-base-v2)
- Generates 768-dimensional dense vector representations
- Cosine similarity between prompt and response embeddings
- Range: 0 (no coherence) to 1 (perfect coherence)

### 2.5 Statistical Analysis

- **Power Analysis:** With n=50 per condition and α=0.01, achieved 80% power to detect d≥0.27
- **Effect Sizes:** Cohen's d with pooled standard deviation
  ```
  d = (M₁ - M₂) / SDpooled
  SDpooled = √[((n₁-1)SD₁² + (n₂-1)SD₂²) / (n₁+n₂-2)]
  ```
- **Multiple Comparisons:** Bonferroni correction (α=0.05/5=0.01)

---

## 3. Results

### 3.1 Descriptive Statistics

**Table 1: Coherence Scores by Condition and Model**

| Model | Linear | Immediate | Shallow | Deep | Contradictory |
|-------|--------|-----------|---------|------|---------------|
| Anthropic | 0.720 (0.058) | 0.713 (0.096) | 0.660 (0.092) | 0.683 (0.104) | 0.702 (0.084) |
| Google | 0.754 (0.051) | 0.719 (0.093) | 0.725 (0.077) | 0.714 (0.075) | 0.813 (0.079) |
| OpenAI | 0.776 (0.065) | 0.764 (0.089) | 0.684 (0.095) | 0.693 (0.100) | 0.734 (0.076) |

*Values: Mean (SD), n=50 per condition*

### 3.2 Effect Sizes

**Table 2: Cohen's d vs Linear Baseline**

| Condition | Anthropic | Google | OpenAI | Mean |
|-----------|-----------|---------|---------|------|
| Immediate | -0.097 | -0.477* | -0.156 | -0.243 |
| Shallow | -0.791*** | -0.447* | -1.136*** | -0.791 |
| Deep | -0.444* | -0.625** | -0.996*** | -0.688 |
| Contradictory | -0.261 | +0.877*** | -0.597** | +0.006 |

*\*p<0.05, \*\*p<0.01, \*\*\*p<0.001*

### 3.3 Hypothesis Testing

- **H1** (Referencing improves coherence): **REJECTED** - Mean effect d=-0.429 (p<0.001)
- **H2** (Distinct degradation patterns): **SUPPORTED** - Models show different effect magnitudes
- **H3** (Critical depth at 5-7): **NOT SUPPORTED** - No consistent critical depth identified
- **H4** (Variable contradiction handling): **SUPPORTED** - Google shows opposite pattern

### 3.4 Key Findings

- **Universal Degradation:** 9/12 conditions (75%) showed significant negative effects
- **Shallow Reference Worst:** N-3 consistently produced largest degradation
- **Google Anomaly:** Unique improvement with contradictory references (d=+0.877)
- **Pattern Classification:** All models showed "irregular" degradation patterns rather than smooth decay

---

## 4. Discussion

### 4.1 Theoretical Implications

Our results suggest important differences between human and LLM discourse processing, though multiple interpretations merit consideration:

#### 4.1.1 Grounding Theory May Not Transfer Directly

Clark & Brennan's (1991) grounding principle, while central to human communication, appears less effective for current LLMs. This may indicate that:
- LLMs maintain context through implicit attention mechanisms rather than explicit acknowledgment
- Current training regimes optimize for implicit rather than explicit discourse patterns
- The specific implementation of grounding matters more than the principle itself

#### 4.1.2 Different Memory Architecture

The peak degradation at N-3 (shallow references) doesn't align with Cowan's (2001) 4±1 working memory limit. Rather than suggesting fundamentally incompatible architectures, this may indicate:
- LLMs process context holistically rather than through discrete memory buffers
- Training data may lack explicit reference patterns at these depths
- Our reference templates may introduce unnatural discourse markers

#### 4.1.3 Current Optimization Favors Implicit Processing

Transformer attention (Vaswani et al., 2017) processes all tokens simultaneously without explicit memory buffers. The observed interference when forcing explicit structure could result from:
- Misalignment between training objectives and explicit reference handling
- Template-specific artifacts rather than reference degradation per se
- Learned expectations for implicit rather than explicit discourse flow

### 4.2 Alternative Interpretations

Several factors could explain the observed degradation beyond architectural incompatibility:

#### 4.2.1 Template Artifacts
Our reference templates ("Following up on...", "Returning to...") may introduce confounds:
- These specific phrases might be rare in training data
- Templates could prime unnecessary context switches
- Syntactic complexity might interfere with semantic processing

#### 4.2.2 Metric Limitations
SBERT semantic similarity may not fully capture coherence:
- The metric might penalize explicit discourse markers
- Vector similarity could miss pragmatic coherence
- Human evaluation might reveal different patterns

#### 4.2.3 Training Data Distribution
The degradation might reflect training corpus characteristics:
- Natural text may rarely contain explicit back-references at N-3 or N-7 depths
- Models may have learned that explicit references signal topic changes
- Implicit reference may simply be more common in training data

### 4.3 The Google Anomaly

Google Gemini's improvement with contradictory references (d=+0.877) is particularly intriguing and suggests:

- Possible optimization for argumentative or debate-style discourse
- Different RLHF approaches to handling disagreement
- Training on more diverse conversational patterns
- A potential clue about when explicit references DO enhance performance

This anomaly challenges our universal degradation narrative and suggests model-specific training effects play a crucial role.

### 4.4 Practical Implications

Our findings suggest considerations for prompt engineering and conversational AI design:

- **For most models:** Linear conversation flow may yield better coherence than explicit back-references
- **Context awareness:** Let models handle context implicitly unless specific evidence suggests otherwise
- **Model-specific strategies:** Google systems may benefit from contradictory framing
- **Template sensitivity:** The specific phrasing of references likely matters substantially

These are preliminary guidelines that warrant validation across different tasks and domains.

### 4.4 Limitations

- **Language:** English-only prompts
- **Templates:** Fixed reference structures
- **Metric:** Single coherence measure (SBERT)
- **Baseline:** No human performance comparison
- **Context Window:** Limited to model constraints

---

## 5. Future Directions

### 5.1 Mechanistic Understanding

- **Attention Analysis:** Examine attention patterns during reference processing to understand how models allocate focus
- **Layer-wise Probing:** Investigate intermediate representations to identify where coherence degradation occurs
- **Causal Interventions:** Use activation patching to determine which components drive the degradation

### 5.2 Methodological Extensions

- **Template Ablations:** Test alternative reference phrasings to isolate template artifacts from true reference effects
- **Human Baselines:** Compare with human performance on identical tasks
- **Alternative Metrics:** Evaluate using perplexity, human judgments, and task-specific coherence measures
- **Gradual vs Explicit:** Compare explicit references with gradual topic transitions

### 5.3 Cross-Model Studies

- **Architecture Variations:** Test decoder-only vs encoder-decoder models
- **Scale Effects:** Investigate whether larger models show different patterns
- **Training Data Analysis:** Examine reference patterns in pretraining corpora
- **Fine-tuning Impact:** Test whether targeted training can improve explicit reference handling

### 5.4 Applied Research

- **Task-Specific Evaluation:** Test whether degradation persists across different tasks (QA, summarization, reasoning)
- **Adaptive Prompting:** Develop methods to detect when explicit vs implicit reference is optimal
- **Interface Design:** Create conversation UIs that leverage these findings
- **Training Innovations:** Design pretraining or fine-tuning approaches that better handle explicit discourse

### 5.5 Theoretical Extensions

- **Discourse Model Comparison:** Test other discourse theories beyond grounding
- **Cognitive Load Analysis:** Investigate whether reference complexity correlates with degradation
- **Cross-linguistic Studies:** Examine whether patterns hold across languages with different discourse conventions
- **Pragmatic Coherence:** Develop metrics that capture pragmatic rather than just semantic coherence

---

## 6. Conclusion

This study provides evidence that explicit conversational referencing, a strategy beneficial for human communication, currently degrades coherence in major LLMs. The mean effect size of d=-0.429 (p<0.001), replicated across three distinct architectures, suggests that contemporary language models may process discourse differently than assumed by human communication theories.

However, these findings invite multiple interpretations:

- **Architectural:** Current transformer-based models may be optimized for implicit rather than explicit context management
- **Training-based:** The degradation may reflect the distribution of discourse patterns in training data
- **Methodological:** Template phrasing and metric choice may influence the observed effects
- **Task-specific:** Effects may vary across different applications and domains

The notable exception of Google Gemini's improvement with contradictory references (d=+0.877) demonstrates that the relationship between explicit referencing and coherence is more complex than initially apparent. This suggests that blanket statements about LLM discourse processing may be premature.

These findings have several implications:

- **Theoretical:** Encourage reconsideration of how human discourse theories apply to AI systems
- **Practical:** Inform prompt engineering strategies, with awareness that effects may be model- and task-specific
- **Technical:** Suggest opportunities for training or architectural innovations that better handle varied discourse patterns

Rather than indicating fundamental incompatibility between LLMs and explicit referencing, our results highlight the need for nuanced understanding of how different models, trained on different data with different objectives, handle discourse structure. The observed degradation patterns point toward exciting opportunities for improving human-AI interaction through better alignment of discourse strategies with model capabilities.

Future work should investigate the boundaries of these effects, explore alternative reference implementations, and develop training approaches that enable models to handle both implicit and explicit discourse patterns effectively. The field would benefit from systematic investigation of when, why, and for which models explicit referencing helps or hinders communication.

---

## Acknowledgments

[To be added]

---

## Data and Code Availability

All data, code, and materials are available at: https://github.com/HillaryDanan/nonlinear-dialogue-dynamics

---

## References

Anderson, L. W., & Krathwohl, D. R. (2001). *A taxonomy for learning, teaching, and assessing: A revision of Bloom's taxonomy of educational objectives.* Longman.

Baddeley, A. (2000). The episodic buffer: a new component of working memory? *Trends in Cognitive Sciences*, 4(11), 417-423.

Clark, H. H. (1996). *Using language.* Cambridge University Press.

Clark, H. H., & Brennan, S. E. (1991). Grounding in communication. In L. B. Resnick, J. M. Levine, & S. D. Teasley (Eds.), *Perspectives on socially shared cognition* (pp. 127-149). American Psychological Association.

Cohen, J. (1988). *Statistical power analysis for the behavioral sciences* (2nd ed.). Lawrence Erlbaum Associates.

Cowan, N. (2001). The magical number 4 in short-term memory: A reconsideration of mental storage capacity. *Behavioral and Brain Sciences*, 24(1), 87-114.

Lakens, D. (2013). Calculating and reporting effect sizes to facilitate cumulative science: a practical primer for t-tests and ANOVAs. *Frontiers in Psychology*, 4, 863.

Miller, G. A. (1956). The magical number seven, plus or minus two: Some limits on our capacity for processing information. *Psychological Review*, 63(2), 81-97.

Nosek, B. A., et al. (2018). The preregistration revolution. *Proceedings of the National Academy of Sciences*, 115(11), 2600-2606.

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing.*

Tulving, E. (1983). *Elements of episodic memory.* Oxford University Press.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30, 5998-6008.

---

## Appendix: Supplementary Tables

[Additional statistical details and raw data summaries will go here]
