# Study Limitations (Full Disclosure)

## What We CANNOT Measure
1. **Actual attention weights** - API access only, no model internals (Confirmed: OpenAI, Anthropic, Google)
2. **True causality** - Observational within-subjects design, not RCT
3. **Generalizability** - Limited to 3 models, may not extend to all LLMs

## Known Confounds
1. **Prompt length variation** - 7x token difference between conditions
2. **Order effects** - Despite counterbalancing, learning may occur
3. **Experimenter bias** - Single researcher coding responses

## Statistical Limitations
- Multiple comparisons problem (3 conditions × 4 metrics = 12 tests)
- Bonferroni correction will reduce power to ~0.65
- Effect sizes based on adjacent literature, not direct precedent
