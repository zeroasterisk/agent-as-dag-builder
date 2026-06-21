# Research Log: Hypothesis-Tree Learning Loop

**Date:** 2026-06-21
**Script:** `sandbox/17_hypothesis_tree.py`
**Results:** `sandbox/scores_hypothesis_tree.json`
**Model:** gemini-3.5-flash (all nodes + judge)
**Base:** `sandbox/11_multi_harness.py` (conservative learning loop, 87.7 best)

## Motivation

The previous integrated loop (87.7 best aggregate) showed two problems:
1. **Potential overfitting**: The same test set guides optimization AND evaluates progress. Score oscillations (85.9-87.7) may reflect overfitting to specific cases rather than genuine improvement.
2. **Single-path optimization**: One hypothesis per iteration limits exploration. A bad proposal wastes an entire iteration.

Arbor (arXiv:2606.11926) addresses both: held-out validation prevents overfitting, and branching hypotheses multiply exploration per iteration.

## Design

### A. Held-Out Validation (70/30 split)

Deterministic split with `random.seed(42)`:

| Harness | Train (70%) | Held-Out (30%) | Total |
|---------|-------------|----------------|-------|
| customer_support | 16 | 7 | 23 |
| it_helpdesk | 12 | 6 | 18 |
| sales_inquiry | 11 | 6 | 17 |

The learning loop ONLY sees training case results. Promotions require held-out improvement, preventing overfitting.

### B. Parallel Hypothesis Branching (3 per iteration)

Each iteration generates 3 variants for the weakest harness:

| Strategy | Focus | Example Addition |
|----------|-------|-----------------|
| templates | Issue-specific step-by-step templates | "For SCREEN FLICKERING: 1. Check refresh rate..." |
| knowledge | Domain knowledge (servers, URLs, IPs) | "VPN server: vpn.company.com, DNS: 10.0.0.53" |
| clarity | Instruction clarity and structure | "ALWAYS mention signal strength for Wi-Fi issues" |

All 3 evaluated on training cases. Winner validated on held-out. Promotion only if held-out improves.

### C. Insight Memory

Dataclass records for every hypothesis tested:

```python
@dataclass
class Insight:
    iteration: int
    harness: str
    strategy: str       # "templates", "knowledge", "clarity"
    description: str
    train_delta: float
    holdout_delta: float
    accepted: bool
```

Past insights are included in hypothesis generation prompts, allowing the optimizer to learn from its own history.

### D. Conservative Optimizer (inherited)

- Best-config tracking (optimize from best-ever)
- Stability zone (skip harnesses within 1.5 pts of best)
- One-at-a-time (only modify weakest harness)
- Rollback (restore all if aggregate drops >2 pts below best)

## Results

*(To be filled after run)*

### Score Progression

| Iter | Train Agg | Held-Out Agg | Hypotheses | Winner Strategy |
|------|-----------|--------------|------------|-----------------|
| | | | | |

### Train vs Held-Out Tracking

Key question: Do training and held-out scores move in the same direction?

- Direction agreement: __%
- Train range: __ pts
- Holdout range: __ pts
- Interpretation: __

### Strategy Win Distribution

| Strategy | Times Winner | Times Promoted | Avg Train Delta | Avg Holdout Delta |
|----------|-------------|----------------|-----------------|-------------------|
| templates | | | | |
| knowledge | | | | |
| clarity | | | | |

### Insight Memory Contents

*(Full insight dump after run)*

### Comparison with Previous Runs

| Metric | Conservative (11) | Integrated (11) | Hypothesis Tree (17) |
|--------|-------------------|-----------------|---------------------|
| Best aggregate | 84.8 | 87.7 | __ |
| Score range | 4.4 pts | 1.8 pts | __ |
| Rollbacks | 0 | 0 | __ |
| LLM calls | ~522 | ~1754 | ~870 est |

## Analysis

### Did Held-Out Validation Solve Oscillation?

*(To be determined: compare train vs holdout divergence. If they track closely, the 87.7 wasn't overfit. If they diverge, it was.)*

### Which Strategy Dominates?

*(To be determined: count wins per strategy across iterations.)*

### Is Insight Memory Useful?

*(To be determined: do later iterations make better proposals based on accumulated insights?)*

## LLM Call Budget

Per iteration (approximate):
- Base evaluation: (16+12+11) train + (7+6+6) holdout = 39+19 = 58 cases x 3 calls = 174
- Hypothesis generation: 3 strategies x 2 LLM calls = 6
- Hypothesis evaluation: 3 variants x ~13 train cases x 3 calls = ~117
- Winner validation: ~6 holdout cases x 3 calls = ~18
- **Total per iteration: ~315 LLM calls**
- **10 iterations: ~3,150 LLM calls** (much more than 1,754 for integrated loop)

Note: The actual number is higher because hypothesis evaluation runs only on the weakest harness, but the first evaluation covers all 3 harnesses.

## Conclusion

*(To be filled after run)*
