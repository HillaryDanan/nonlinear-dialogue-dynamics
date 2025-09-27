# Explicit Conversational Referencing Degrades LLM Coherence: Evidence Against Human-Analogous Discourse Processing

## Authors
Hillary Danan¹
¹ [Institution]

## Abstract (250 words)
**Background:** Current theories of human discourse processing (Clark & Brennan, 1991) suggest that explicit referencing to earlier conversational points improves coherence through grounding. Large Language Models (LLMs) operate through fundamentally different mechanisms (Vaswani et al., 2017), yet are often assumed to benefit from human-like discourse strategies.

**Methods:** We conducted a controlled experiment (N=50 prompts × 5 conditions) comparing linear conversation progression against four types of explicit referencing: immediate (N-1), shallow (N-3), deep (N-7+), and contradictory. Coherence was measured using semantic similarity (SBERT embeddings) between prompts and responses.

**Results:** Contrary to theoretical predictions, explicit referencing significantly degraded coherence. Shallow references (N-3) showed the largest negative effect (d=-0.791, 95% CI [-1.05, -0.53], p<0.001). All reference conditions performed worse than linear progression, with effect sizes ranging from d=-0.097 to d=-0.791.

**Conclusions:** These findings challenge the application of human discourse theories to LLMs. The degradation pattern, peaking at the working memory boundary (N-3), suggests LLMs lack analogous memory architecture. Transformers appear to handle discourse through implicit attention mechanisms that are disrupted by explicit referencing. This has important implications for prompt engineering and conversational AI design.

**Keywords:** large language models, discourse coherence, conversational referencing, transformer architecture, human-AI interaction

## 1. Introduction

### 1.1 Theoretical Background
- Human discourse processing (Clark & Brennan, 1991; Clark, 1996)
- Grounding theory and common ground
- Working memory constraints (Cowan, 2001; Miller, 1956)
- Episodic buffer model (Baddeley, 2000)

### 1.2 LLM Architecture
- Transformer attention mechanisms (Vaswani et al., 2017)
- Contextual embeddings (Devlin et al., 2019)
- Lack of explicit memory systems
- Implicit discourse tracking

### 1.3 Research Question
Do LLMs benefit from explicit conversational referencing strategies derived from human discourse theory?

### 1.4 Hypotheses
H1: Explicit referencing will improve coherence (FALSIFIED)
H2: Degradation patterns will mirror human memory constraints (PARTIALLY SUPPORTED)
H3: Critical breakdown at depth 5-7 (NOT OBSERVED)

## 2. Methods

### 2.1 Experimental Design
- Within-subjects design
- 5 reference conditions
- 50 prompts balanced across 10 domains
- Stratified by cognitive complexity (Bloom's taxonomy)

### 2.2 Reference Conditions
1. **Linear**: Standard sequential progression
2. **Immediate** (N-1): Phonological loop (<2 sec)
3. **Shallow** (N-3): Working memory boundary
4. **Deep** (N-7+): Long-term memory retrieval
5. **Contradictory**: Explicit contradiction

### 2.3 Coherence Measurement
- SBERT embeddings (Reimers & Gurevych, 2019)
- Cosine similarity metric
- Validated against human judgments (r=0.78)

### 2.4 Statistical Analysis
- Cohen's d with pooled variance
- Bonferroni correction (α=0.01)
- Degradation curve fitting (linear, exponential, cliff)

## 3. Results

### 3.1 Primary Findings
- Linear baseline: M=0.720, SD=0.058
- Shallow reference: M=0.660, SD=0.092
- Effect size: d=-0.791 (large negative effect)

### 3.2 Degradation Patterns
- Peak degradation at N-3 (working memory boundary)
- Irregular rather than smooth decay
- No critical depth identified

### 3.3 Model Comparisons
- Anthropic Claude: Valid results showing degradation
- Google Gemini: Technical failures (404 errors)
- OpenAI GPT-4: Baseline coherence near zero

## 4. Discussion

### 4.1 Theoretical Implications
- Challenges Clark & Brennan (1991) for AI systems
- LLMs lack human-like memory architecture
- Implicit > explicit discourse processing

### 4.2 Mechanistic Interpretation
- Attention mechanisms handle context implicitly
- Explicit references create competing representations
- Forced structure disrupts natural processing

### 4.3 Practical Implications
- Prompt engineering should avoid explicit back-references
- Linear progression optimal for coherence
- Rethink conversational AI design patterns

### 4.4 Limitations
- Single coherence metric
- Limited to one model (technical failures)
- English-only prompts
- No human baseline comparison

## 5. Future Directions

### 5.1 Proposed Studies
1. Implicit vs explicit reference mechanisms
2. Memory architecture mapping
3. Cross-linguistic validation
4. Human-AI comparative study

### 5.2 Methodological Improvements
- Multiple coherence metrics
- Longer conversation chains
- Real-world dialogue tasks

## 6. Conclusion
Explicit conversational referencing, a strategy beneficial for human discourse, significantly degrades LLM coherence. This negative result has important implications for understanding fundamental differences between human and artificial language processing.

## References
[Following APA 7th edition]

## Supplementary Materials
- Full experimental protocol
- Raw data and analysis code
- Additional statistical tests
- Reproducibility checklist
