# Experimental Protocol

## Design
Within-subjects comparison across three conditions:
- Linear: Standard sequential conversation
- Referenced: Explicit reply to earlier messages  
- Hybrid: Mixed approach

## Sample Size
- Pilot: n=15
- Main: n=64
- Based on d=0.27, power=0.80, α=0.00417 (Bonferroni)

## Metrics
- Coherence score (cosine similarity)
- Contradiction rate
- Topic drift
- Response latency

## Analysis Plan
1. RM-ANOVA with Greenhouse-Geisser correction
2. ANCOVA with prompt length as covariate
3. Effect sizes with 95% CIs for all comparisons
