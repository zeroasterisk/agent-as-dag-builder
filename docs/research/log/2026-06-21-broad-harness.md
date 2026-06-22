# Broad 6-Harness Benchmark with Tuned Promotion Threshold

**Date:** 2026-06-21 (run completed 2026-06-22)
**Script:** `sandbox/18_broad_harness.py`
**Runtime:** ~14 hours
**Model:** gemini-3.5-flash (Vertex AI)

## Experiment Design

Extended Graph Gardener's hypothesis-tree learning loop with 3 new harness
domains and a tuned promotion threshold.

### New Domains
- **Healthcare Triage** (emergency, appointment, prescription, insurance, general-health)
- **E-commerce Support** (order-status, returns, payment, product-info, shipping)
- **Legal Intake** (personal-injury, family-law, business, criminal, real-estate)

### Key Changes from sandbox/17
- 6 harnesses (was 3), 30 cases each (was 18-23), 180 total
- Holdout split: 65/35 (was 70/30) -- more signal per holdout evaluation
- Holdout evaluated 2x and averaged -- reduces noise
- Promotion threshold: accept if holdout >= (best - 1.0 pt) -- lateral moves allowed
- 10 iterations, 3 hypothesis strategies (templates, knowledge, clarity)

## Results

### Main Results Table

```
Harness              | Cases | Baseline | Best   | Delta  | Promoted?
---------------------+-------+----------+--------+--------+------------------
customer_support     |   30  |   88.8   |  89.8  |  +1.0  | NO
it_helpdesk          |   30  |   84.2   |  88.7  |  +4.5  | NO (noise gain)
sales_inquiry        |   30  |   78.9   |  89.4  |  +10.5 | NO (noise gain)
healthcare_triage    |   30  |   89.7   |  90.5  |  +0.8  | NO
ecommerce_support    |   30  |   88.7   |  91.4  |  +2.7  | NO (noise gain)
legal_intake         |   30  |   75.4   |  83.3  |  +7.9  | YES (iter 1,3,4)

Aggregate holdout:  84.3 -> 87.7 (best at iter 8)
Final iter holdout: 86.3
Promotions: 3 / 9 hypothesis tests
```

Note: "Best" reflects the best holdout score seen at any iteration; some gains
are noise (LLM scoring variance) rather than config improvements. Legal intake
is the only domain where configs were actually promoted.

### Per-Iteration Aggregate Scores

```
Iter | Train |  Holdout | Hypotheses | Winner
-----+-------+----------+------------+----------------
   1 |  82.8 |     84.3 |          3 | clarity (legal)
   2 |  84.5 |     85.3 |          3 | clarity (rej)
   3 |  84.7 |     84.7 |          3 | templates (legal)
   4 |  84.5 |     85.5 |          3 | clarity (legal)
   5 |  85.6 |     87.2 |          3 | knowledge (rej)
   6 |  85.6 |     86.8 |          3 | clarity (rej)
   7 |  85.8 |     85.8 |          3 | templates (rej)
   8 |  85.5 |     87.7 |          3 | knowledge (rej)
   9 |  85.5 |     86.3 |          3 | clarity (rej)
  10 |  86.0 |     86.3 |          0 | (eval only)
```

### Promotion Details

| Iter | Harness       | Strategy  | Train Delta | Holdout Avg | Reason   |
|------|---------------|-----------|-------------|-------------|----------|
| 1    | legal_intake  | clarity   | +3.95       | 77.0        | NEW_BEST |
| 2    | legal_intake  | clarity   | +4.26       | 76.2        | REJECTED |
| 3    | legal_intake  | templates | +2.79       | 80.4        | NEW_BEST |
| 4    | legal_intake  | clarity   | +6.53       | 83.3        | NEW_BEST |
| 5    | sales_inquiry | knowledge | +2.37       | 88.0        | REJECTED |
| 6    | customer_supp | clarity   | +4.63       | 87.8        | REJECTED |
| 7    | sales_inquiry | templates | +1.63       | 82.0        | REJECTED |
| 8    | sales_inquiry | knowledge | -0.68       | 82.7        | REJECTED |
| 9    | customer_supp | clarity   | +2.47       | 87.0        | REJECTED |

### Legal Intake Improvement Trajectory
```
Iter 1: holdout 75.4 -> 77.0 (clarity promoted, +1.6)
Iter 2: holdout 79.0, hypothesis 76.2 (rejected)
Iter 3: holdout 76.8 -> 80.4 (templates promoted, +3.6)
Iter 4: holdout 77.2 -> 83.3 (clarity promoted, +6.1)
Iter 5+: holdout ~80-84 (no more legal_intake, now targeting others)
```

Total legal_intake improvement: 75.4 -> 83.3 holdout (+7.9 pts, from 3 promotions)

## Analysis

### 1. Tuned Promotion Threshold Performance

The 1.0 pt tolerance did not produce any LATERAL_MOVE promotions -- all 3
promotions were strict NEW_BEST improvements. The tolerance widened the gate
but the winning hypotheses either clearly improved (iter 1,3,4 for legal_intake)
or clearly regressed (everything else).

**Verdict:** The tolerance is reasonable but did not change behavior materially.
The binary outcome (clear improvement or regression) suggests that the 11-case
holdout evaluation, even with 2x averaging, has enough variance (~5-10 pts
between passes) that a 1.0 pt tolerance is too small relative to noise. A 2-3 pt
tolerance might be needed to actually enable lateral moves.

### 2. Domain Pattern Differences

**Original 3 domains (customer_support, it_helpdesk, sales_inquiry):**
- Baseline avg holdout: 83.9
- These were already well-tuned from prior experiments (sandbox/11-17)
- Hypotheses consistently improved training but failed holdout validation
- The existing configs are near-optimal for these categories

**New 3 domains (healthcare_triage, ecommerce_support, legal_intake):**
- Baseline avg holdout: 84.6
- Healthcare and ecommerce started strong (89.7 and 88.7 holdout) thanks to
  detailed YAML handler instructions with concrete domain knowledge baked in
- Legal intake started weakest (75.4) and showed the most improvement (+7.9)
- Legal intake's weakness was the business/real-estate classification boundary
  (case 16 "review a commercial lease" consistently misclassified)

**Key pattern:** Domains with already-strong configs (>88 holdout) resist
optimization because the hypotheses can't beat the noise floor. Legal intake
had clear room for improvement (weak handler instructions, ambiguous categories)
and absorbed 3 consecutive promotions.

### 3. Strategy Performance

```
Strategy   | Winner count | Promotions
-----------+--------------+-----------
clarity    | 5            | 2
templates  | 2            | 1
knowledge  | 2            | 0
```

**Clarity** was the most frequent winner and most promoted strategy. It works
by making handler instructions more explicit about what constitutes a good
response. This is especially effective for complex domains like legal intake
where the baseline instructions were vague.

**Templates** won twice, promoted once. Step-by-step templates helped legal
intake by providing concrete workflows for specific case types.

**Knowledge** won twice but was never promoted. It improves training scores
by injecting domain-specific details but these details often don't match
the holdout cases well enough (overfitting risk).

### 4. Overfitting Patterns

Nearly every hypothesis improved training scores but regressed on holdout:
- Iter 2: train +4.26, holdout -2.77
- Iter 6: train +4.63, holdout +2.86 (but below best of 88.8)
- Iter 8: train -0.68, holdout -3.09

The train/holdout gap grows as instructions become more specialized. This
validates the held-out validation gate as essential.

### 5. Category Accuracy

- Healthcare triage: 100% accuracy (emergency/appointment/prescription/insurance/general-health well-separated)
- IT helpdesk: 100% accuracy (password-reset/software-install/hardware/network well-separated)
- Ecommerce support: 100% accuracy (order-status/returns/payment/product-info/shipping well-separated)
- Customer support: 89.5% accuracy (billing/technical/general -- "upgrade plan" misclassified)
- Sales inquiry: 89.5% accuracy (pricing/features/demo/competitor -- "ROI" and "case study" misclassified)
- Legal intake: 94.7% accuracy (only "commercial lease review" misclassified as real-estate instead of business)

New domains achieved excellent classification accuracy thanks to detailed
classifier instructions with explicit boundary rules.

## Scores File

`sandbox/scores_broad.json` -- full iteration-by-iteration data with insights and promotions.

## Files Created

- `sandbox/18_broad_harness.py` -- experiment script
- `sandbox/healthcare_triage.yaml` -- healthcare triage DAG config
- `sandbox/ecommerce_support.yaml` -- e-commerce support DAG config
- `sandbox/legal_intake.yaml` -- legal intake DAG config
- `sandbox/legal_intake_broad_v2.yaml` -- promoted at iter 1 (clarity)
- `sandbox/legal_intake_broad_v4.yaml` -- promoted at iter 3 (templates)
- `sandbox/legal_intake_broad_v5.yaml` -- promoted at iter 4 (clarity)
- `sandbox/scores_broad.json` -- full score data
