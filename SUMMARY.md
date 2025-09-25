# Non-Linear Dialogue Dynamics Study - Current Status

## Study Overview

**Core Research Question**: Does implementing threaded reply capabilities (explicit reference to earlier conversation points) improve LLM understanding and coherence compared to strictly linear conversational progression?

**Principal Investigator**: Hillary Danan  
**Date**: September 2025  
**Repository**: https://github.com/HillaryDanan/nonlinear-dialogue-dynamics

## Theoretical Framework

### Primary Literature
- **Clark & Brennan (1991)**: Grounding in communication - successful communication requires explicit acknowledgment and reference to shared content
- **Miller (1956)**: The magical number 7±2 - cognitive capacity limits
- **Cowan (2001)**: Working memory constraints (4±1 items)
- **Baddeley (2000)**: Episodic buffer model linking working and long-term memory

### Hypotheses
1. **H1**: Explicit referencing improves coherence (expected d=0.27 based on HCI meta-analysis)
2. **H2**: Models show distinct degradation signatures with recursion depth
3. **H3**: Critical coherence breakdown at depth 5-7 (Miller's limit)
4. **H4**: Contradiction handling varies by model architecture

## Current Implementation Status

### ✅ Completed
1. **Theoretical framework** documented
2. **50-prompt protocol** designed across 10 domains × 5 cognitive levels
3. **7 reference conditions** implemented:
   - Linear (control)
   - Immediate (N-1)
   - Shallow (N-2 to N-3)
   - Medium (N-4 to N-5)
   - Deep (N-6+)
   - Contradictory
   - Branching
4. **Coherence calculation** using SBERT embeddings
5. **Degradation analysis** with curve fitting (linear, exponential, cliff detection)
6. **Statistical framework** with Bonferroni correction, Cohen's d, 95% CIs

### ⚠️ Current Issues

#### OpenAI API
- **Problem**: HTTP 429 rate limiting, possible account limits
- **Status**: Falls back to direct HTTP requests, but still hitting limits
- **Workaround**: Continue with Anthropic + Google only

#### JSON Serialization
- **Problem**: ReferenceType enum not JSON serializable
- **Fix**: Replace line 572 in `complete_nonlinear_experiment.py`:
```python
# Instead of:
config_str = json.dumps(asdict(self.config), sort_keys=True)

# Use:
config_str = str(self.config)
```

### 📊 Pilot Results Summary

**Initial findings (n=6 prompts, underpowered)**:
```
Anthropic: Linear (0.793) > Referenced (0.726), d = -0.067
Google: Linear (0.793) > Referenced (0.727), d = -0.066
OpenAI: Broken (negative coherence scores)
```

**Interpretation**: Early results suggest referencing may DEGRADE coherence (opposite of hypothesis), but need full protocol for statistical validity.

## File Structure

```
nonlinear-dialogue-dynamics/
├── README.md                           # Project overview
├── theory/
│   └── theoretical_framework.md        # Detailed theory
├── experiments/
│   ├── complete_nonlinear_experiment.py # Main experiment (50 prompts)
│   ├── data_collector_all_models.py    # Model interfaces
│   └── run_experiment_fixed.py         # Rate-limited version
├── analysis/
│   ├── coherence_metrics.py           # SBERT coherence calculation
│   └── analyze_pilot.py               # Pilot analysis
├── data/
│   ├── pilot/                         # Initial pilot data
│   └── nonlinear_results/            # Full experiment results
└── SUMMARY.md                         # This file
```

## Next Steps

### Immediate (Fix & Run)
1. Fix JSON serialization error (see fix above)
2. Run full 50-prompt protocol
3. Expect ~30 minutes per model

### Analysis
1. Compare degradation patterns across models
2. Identify critical depths
3. Test all 4 hypotheses
4. Generate final report

### If We Hit Context Limits
**Key info for fresh Claude**:
- Study tests if referencing helps or hurts coherence
- 50 prompts × 5-7 conditions × 2-3 models
- Expecting small effect (d≈0.27)
- Pilot suggests referencing might HURT (opposite of hypothesis)
- OpenAI broken, use Anthropic + Google

## Git Commands

```bash
# Stage everything
git add .

# Commit with comprehensive message
git commit -m "Complete experimental protocol with 50-prompt battery

- Implemented 7 reference conditions testing different memory systems
- Added degradation analysis with curve fitting
- Statistical framework with Bonferroni correction
- Pilot results suggest referencing may degrade coherence
- OpenAI API issues, continuing with Anthropic + Google"

# Push to GitHub
git push origin main
```

## Quick Start for Fresh Session

```bash
# Clone repo
git clone https://github.com/HillaryDanan/nonlinear-dialogue-dynamics.git
cd nonlinear-dialogue-dynamics

# Install dependencies
python3 -m pip install numpy scipy sentence-transformers python-dotenv

# Fix the JSON error (line 572)
# Then run:
python3 complete_nonlinear_experiment.py
```

## Scientific Integrity Notes

- **Pre-registration**: Via git commits (transparent history)
- **Power analysis**: n=50 for 80% power to detect d=0.27
- **Multiple comparisons**: Bonferroni corrected α=0.01
- **Negative results valid**: If referencing hurts coherence, that's important!

## Contact

Hillary Danan - [GitHub](https://github.com/HillaryDanan)

---

*Last updated: September 2025*