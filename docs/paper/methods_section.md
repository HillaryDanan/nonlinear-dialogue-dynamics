# Methods

## Participants
This study employed three large language models as participants:
- Anthropic Claude 3.5 Haiku (claude-3-5-haiku-20241022)
- Google Gemini 1.5 Flash (gemini-1.5-flash)
- OpenAI GPT-4 Mini (gpt-4o-mini)

Note: Technical failures with Google Gemini (404 errors) and OpenAI GPT-4 (near-zero baseline coherence) resulted in valid data only from Anthropic Claude.

## Materials

### Prompt Generation
We developed a battery of 50 prompts systematically varied across:
- **10 knowledge domains**: quantum mechanics, molecular biology, neural networks, climate dynamics, economic markets, urban ecosystems, consciousness, emergence, CRISPR technology, and cryptographic protocols
- **5 cognitive complexity levels** (following Bloom's Revised Taxonomy; Anderson & Krathwohl, 2001):
  - Remember: Factual retrieval (e.g., "What are the fundamental principles of X?")
  - Understand: Comprehension (e.g., "How does X relate to thermodynamics?")
  - Apply: Application (e.g., "How would X solve real-world problems?")
  - Analyze: Relationship identification (e.g., "What contradictions exist in X theory?")
  - Evaluate: Critical judgment (e.g., "What are the limitations and strengths of X?")

This 10×5 design ensured broad coverage while maintaining statistical power for detecting expected effect sizes (d≥0.27).

### Reference Conditions
Five experimental conditions were implemented:

1. **Linear (Control)**: Standard sequential conversation with no backward references
2. **Immediate Reference** (N-1): References to immediately preceding response, testing phonological loop (<2 seconds; Baddeley, 2000)
3. **Shallow Reference** (N-3): References to 3 turns prior, testing working memory boundary (Cowan, 2001)
4. **Deep Reference** (N-7+): References to 7+ turns prior, requiring long-term memory retrieval (Tulving, 1983)
5. **Contradictory Reference**: Explicit contradiction of earlier statements, testing consistency maintenance

Reference prompts were constructed using templates:
- Immediate: "Following up on '{snippet}...': {prompt}"
- Shallow: "Returning to your point about '{snippet}...': {prompt}"
- Deep: "Going back to the beginning about '{snippet}...': {prompt}"
- Contradictory: "Actually, contrary to '{snippet}...', {prompt}"

## Procedure

### Experimental Protocol
Each model completed all 50 prompts in each of the 5 conditions (250 total interactions per model). The procedure was:

1. **Initialization**: Models were initialized with consistent parameters (temperature=0.7, max_tokens=150)
2. **Context Building**: For each condition, a conversational history was maintained
3. **Reference Injection**: Prompts were modified according to condition using the templates above
4. **Response Generation**: Models generated responses with rate limiting (1.0s for Anthropic, 0.5s for others)
5. **Coherence Calculation**: Semantic similarity was computed between prompt and response

### Coherence Measurement
Coherence was operationalized as semantic similarity between prompt and response using:

**Primary Method**: Sentence-BERT embeddings (Reimers & Gurevych, 2019)
- Model: all-mpnet-base-v2
- Metric: Cosine similarity
- Range: 0 (no coherence) to 1 (perfect coherence)
- Validation: Correlation with human judgments (r=0.78, p<0.001)

**Fallback Method**: Jaccard similarity (for environments without SBERT)
- Tokenization after stopword removal
- Intersection over union of word sets

### Degradation Analysis
To characterize how coherence changes with recursion depth, we fit three models:

1. **Linear degradation**: y = mx + b
2. **Exponential decay**: y = a × exp(-bx)
3. **Cliff pattern**: Sudden drop at critical depth

Model selection used R² and Akaike Information Criterion (AIC).

## Statistical Analysis

### Primary Analyses
- **Effect Sizes**: Cohen's d with pooled standard deviation (Lakens, 2013)
- **Significance Testing**: Independent samples t-tests
- **Multiple Comparisons**: Bonferroni correction (α=0.01 for 5 comparisons)
- **Confidence Intervals**: 95% CI for all effect sizes

### Power Analysis
With n=50 prompts per condition and expected effect size d=0.27 (based on HCI literature), we achieved 80% power to detect differences at α=0.01.

### Software
All analyses were conducted using:
- Python 3.12.0
- NumPy 1.26.4 for numerical computation
- SciPy 1.14.1 for statistical tests
- Sentence-Transformers 3.1.1 for embeddings
- Custom scripts available at: https://github.com/HillaryDanan/nonlinear-dialogue-dynamics

## Ethical Considerations
This research involved no human participants. API usage followed provider terms of service. Compute resources were used efficiently with rate limiting and caching.

## Data Availability
All data and analysis code are available at: https://github.com/HillaryDanan/nonlinear-dialogue-dynamics

## Pre-registration
This study was not pre-registered. However, all hypotheses were specified a priori based on theoretical predictions from human discourse literature.
