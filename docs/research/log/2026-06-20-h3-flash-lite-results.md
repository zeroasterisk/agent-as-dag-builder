# H3 Hypothesis: Weaker Model = Larger Learning Deltas

**Date:** 2026-06-20
**Experiment:** `sandbox/12_flash_lite_experiment.py`
**Scores:** `sandbox/scores_flash_lite.json`

## Hypothesis

Using a less capable model (gemini-3.1-flash-lite) for DAG agents creates more room for the learning loop to improve. The ceiling is lower, so the delta from DAG optimization should be larger than with gemini-2.5-flash.

## Setup

| Component | Model |
|-----------|-------|
| DAG agents (classifier + handlers) | gemini-3.1-flash-lite |
| Judge/scoring | gemini-2.5-flash |
| Learning loop (proposals + YAML gen) | gemini-2.5-flash |

- 58 test cases across 3 harnesses (customer_support: 23, it_helpdesk: 18, sales_inquiry: 17)
- 5 iterations
- LLM delay: 0.3s between calls
- Regression tolerance: 2.0 points

## Results: Flash-Lite (gemini-3.1-flash-lite)

| Iter | Customer Support | IT Helpdesk | Sales Inquiry | AGGREGATE |
|------|-----------------|-------------|---------------|-----------|
| 1    | 87.5            | 82.2        | 84.5          | 85.0      |
| 2    | 88.1 (+0.6)     | 80.0 (-2.2) | 86.3 (+1.8)   | 85.1 (+0.1) |
| 3    | 85.3 (-2.8)     | 75.4 (-4.6) | 87.6 (+1.3)   | 82.9 (-2.2) |
| 4    | 87.1 (+1.8)     | 79.4 (+4.1) | 85.6 (-1.9)   | 84.3 (+1.4) |
| 5    | 87.8 (+0.7)     | 79.8 (+0.4) | 88.5 (+2.8)   | 85.5 (+1.2) |

**Total improvement (flash-lite):** +0.5 points over 5 iterations
**Regression gates triggered:** Iter 2 (it_helpdesk -2.2), Iter 3 (customer_support -2.8, it_helpdesk -4.6)

## Baseline: gemini-2.5-flash (first 5 iterations from earlier run)

| Iter | Customer Support | IT Helpdesk | Sales Inquiry | AGGREGATE |
|------|-----------------|-------------|---------------|-----------|
| 1    | 82.8            | 84.5        | 71.8          | 80.1      |
| 2    | 89.5 (+6.7)     | 80.4 (-4.1) | 86.5 (+14.6)  | 85.8 (+5.7) |
| 3    | 83.7 (-5.8)     | 78.4 (-1.9) | 84.4 (-2.1)   | 82.3 (-3.5) |
| 4    | 87.0 (+3.3)     | 80.8 (+2.3) | 84.2 (-0.2)   | 84.3 (+2.0) |
| 5    | 90.0 (+3.0)     | 81.8 (+1.0) | 82.3 (-1.9)   | 85.2 (+0.9) |

**Total improvement (2.5-flash):** +5.1 points over 5 iterations

## Side-by-Side Comparison

| Metric | Flash-Lite | 2.5-Flash | Delta |
|--------|-----------|-----------|-------|
| Baseline (iter 1) | 85.0 | 80.1 | +4.9 (flash-lite higher) |
| Final (iter 5) | 85.5 | 85.2 | +0.3 |
| Total improvement | +0.5 | +5.1 | -4.6 (2.5-flash improved more) |
| Peak score | 85.5 (iter 5) | 85.8 (iter 2) | -0.3 |
| Trough score | 82.9 (iter 3) | 80.1 (iter 1) | +2.8 |
| Category accuracy (iter 1) | 96.6% (2 misclass) | 100% | -3.4% |
| Category accuracy (iter 5) | 100% | 100% | 0% |

### Per-Harness Final Scores

| Harness | Flash-Lite (iter 5) | 2.5-Flash (iter 5) |
|---------|--------------------|--------------------|
| Customer Support | 87.8 | 90.0 |
| IT Helpdesk | 79.8 | 81.8 |
| Sales Inquiry | 88.5 | 82.3 |

## Hypothesis Verdict: NOT CONFIRMED

The hypothesis that a weaker model would show larger improvement deltas was **not supported** by the data. Key findings:

### 1. Flash-lite baseline was unexpectedly high (85.0 vs expected 65-75)

The flash-lite model scored surprisingly well on baseline. At 85.0, it was actually **higher** than the 2.5-flash baseline of 80.1. This is likely because:
- The customer_support_lite.yaml used the improved classifier instruction from `customer_support_adk.yaml` (with explicit category disambiguation), while the 2.5-flash baseline used a simpler instruction
- Flash-lite is a capable model for straightforward classification + response tasks

### 2. Less room to improve, not more

With a higher starting point (85.0), the flash-lite experiment had less headroom. Total improvement was only +0.5 vs +5.1 for 2.5-flash. The learning loop could not find much to optimize when the initial scores were already strong.

### 3. Higher variance / instability

Flash-lite showed more iteration-to-iteration variance:
- IT helpdesk swung from 82.2 to 75.4 to 79.4 across iterations
- The regression gate fired more frequently (2 of 4 inter-iteration transitions)
- This suggests the weaker model's output is less deterministic, making it harder for instruction improvements to have consistent effects

### 4. Classification was not the bottleneck

Flash-lite achieved 100% classification accuracy from iteration 2 onward (after fixing the "upgrade" -> billing misclassification in iter 1). The bottleneck was response quality, where flash-lite consistently scored lower on technical questions (75.6 quality for technical cases at iter 5 vs 97 for billing).

### 5. Quality gap is in nuanced response generation, not classification

The flash-lite model's weakness was not in routing (classification was near-perfect) but in generating detailed, specific troubleshooting responses. The learning loop improved instructions, but the model itself could not fully execute them. This suggests that DAG-level optimization has diminishing returns when the model capability is the bottleneck.

## Key Observations

1. **Classification is solved for both models.** Both flash-lite and 2.5-flash achieved near-perfect classification accuracy early on. The DAG structure (classifier -> handler) works well for routing.

2. **Response quality is model-bound.** Flash-lite's technical responses consistently scored 10-15 points lower than billing/general. The learning loop improved instructions, but the model could not produce the same depth of troubleshooting advice as 2.5-flash.

3. **Sales inquiry was the surprise.** Flash-lite actually outperformed 2.5-flash on sales inquiry by iter 5 (88.5 vs 82.3). This suggests that for more formulaic responses (pricing, features, demos), the learning loop can compensate for model weakness.

4. **Regression gates worked as designed.** They prevented bad iterations from being promoted, but the non-deterministic nature of flash-lite meant that the "same" config could score differently across iterations purely due to generation variance.

5. **The hypothesis needs reformulation.** A better test would use the **same** starting YAML for both models (we used slightly different configs) and measure improvement from a truly equal starting point.

## Files

- Experiment script: `sandbox/12_flash_lite_experiment.py`
- Flash-lite scores: `sandbox/scores_flash_lite.json`
- Baseline scores: `sandbox/scores.json`
- Flash-lite YAML configs: `sandbox/customer_support_lite.yaml`, `sandbox/it_helpdesk_lite.yaml`, `sandbox/sales_inquiry_lite.yaml`
- Evolved configs: `sandbox/customer_support_lite_v{2,5}.yaml`, `sandbox/it_helpdesk_lite_v{2,5}.yaml`, `sandbox/sales_inquiry_lite_v{2,5}.yaml`
