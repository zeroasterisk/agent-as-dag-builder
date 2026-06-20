# Research Log: Adversarial Verification A/B Test (Sprint B)

**Date:** 2026-06-20
**Hypothesis:** H2 -- Adding a "verifier" node that challenges handler output catches errors a single-pass pipeline misses.
**Status:** Partially confirmed -- consistent +1.8 improvement but below significance threshold

## Experiment Design

A/B test comparing two conditions across 41 test cases (23 customer support + 18 IT helpdesk):

- **Control:** Original DAG (classify -> handler -> done)
- **Treatment:** DAG with verifier nodes (classify -> handler -> verifier -> [revise if needed] -> done)

Verifier nodes receive the original query AND the handler's response, then output one of:
- APPROVED: pass through unchanged
- REVISE: send back to handler with specific feedback (max 2 loops)
- REJECT: flag as failed

Model: gemini-3.5-flash for all nodes. Judge: gemini-3.5-flash at temperature=0.

## Results

### A/B Scores

| Harness | Control | Treatment | Delta | Revisions |
|---------|---------|-----------|-------|-----------|
| Customer Support (23 cases) | 82.0 | 83.8 | +1.8 | 3/23 (13%) |
| IT Helpdesk (18 cases) | 76.3 | 78.1 | +1.8 | 9/18 (50%) |
| **Overall (41 cases)** | **79.5** | **81.3** | **+1.8** | **12/41 (29%)** |

### Category Breakdown

**Customer Support:**
| Category | Control | Treatment | Delta |
|----------|---------|-----------|-------|
| billing | 83.5 | 83.5 | +0.0 |
| technical | 78.2 | 79.9 | +1.6 |
| general | 84.7 | 88.7 | +4.0 |

**IT Helpdesk:**
| Category | Control | Treatment | Delta |
|----------|---------|-----------|-------|
| password-reset | 80.0 | 80.5 | +0.5 |
| software-install | 75.4 | 80.2 | +4.8 |
| hardware | 80.2 | 79.5 | -0.8 |
| network | 71.0 | 72.8 | +1.8 |

### Verifier Analysis

| Metric | Value |
|--------|-------|
| Approved first pass | 29/41 (71%) |
| Revisions triggered | 12/41 (29%) |
| Rejections | 0/41 (0%) |
| Revision improved score | 6/12 (50%) |
| Revision hurt score | 4/12 (33%) |
| Revision neutral | 2/12 (17%) |
| Avg score change per revision | -0.3 pts |

### Notable Individual Cases

**Biggest wins (treatment over control):**
- IT Case 10 (laptop flickering): 72 -> 97 (+25) -- no revision needed, just variance
- CS Case 20 (provide feedback): 75 -> 92 (+17) -- no revision needed
- CS Case 21 (small business services): 83 -> 98 (+15) -- no revision needed
- IT Case 9 (Slack update): 70 -> 84 (+14) -- no revision needed
- CS Case 9 (internet dropping): 80 -> 93 (+13) -- 2 revisions

**Biggest losses (treatment worse than control):**
- IT Case 11 (printer): 98 -> 78 (-20) -- verifier revised a near-perfect response
- CS Case 11 (invalid password): 80 -> 73 (-7) -- approved but scored lower
- IT Case 13 (keyboard characters): 76 -> 69 (-7) -- no revision, just variance

### Response Length

| Harness | Control | Treatment | Delta |
|---------|---------|-----------|-------|
| Customer Support | 870 chars | 976 chars | +12% |
| IT Helpdesk | 1716 chars | 1636 chars | -5% |

## Key Findings

### 1. The verifier helps, but modestly (+1.8 overall)

The +1.8 delta is consistent across both harnesses, suggesting a real but small effect. It falls below the 2-point threshold we'd want for confident confirmation.

### 2. Most improvement came from non-revision cases

Counter-intuitively, the biggest score improvements (IT Case 10: +25, CS Case 20: +17) happened on cases where the verifier APPROVED on the first pass. This suggests the improvement is partly due to run-to-run variance in LLM output, not the verification mechanism itself.

### 3. Revisions have mixed results

Of 12 cases where revisions were triggered:
- 6 improved (50%)
- 4 hurt (33%)
- 2 neutral (17%)

The avg per-revision score change was -0.3 pts -- revisions are roughly a coin flip. The worst case (IT Case 11, printer) saw a -20 drop when the verifier revised an already-excellent response.

### 4. IT helpdesk ceiling NOT broken

IT helpdesk moved from 76.3 to 78.1, remaining within the established 76-78 band. The verifier did not crack the ceiling. The bottleneck appears to be response quality within correctly-routed categories (especially network at 72.8), not errors the verifier can catch.

### 5. The verifier never rejects

Zero rejections across 41 cases. The verifier is too permissive -- it always approves or revises, never flags fundamental problems. This may indicate the handler outputs are "good enough" that a same-model verifier can't distinguish bad from mediocre.

### 6. Software-install showed the best category gain (+4.8)

The strongest category improvement was IT helpdesk software-install (75.4 -> 80.2, +4.8). This category had the most room for improvement and benefits from the verifier's checklist approach (mentioning approval process, licensing, etc.).

## IT Helpdesk Ceiling Analysis

The IT helpdesk has been stuck at 76-78 across all experiments. This run confirms:

- Classification is perfect (100% accuracy in both conditions)
- The bottleneck is handler response quality, not routing
- Network category is weakest (71.0 control, 72.8 treatment)
- Hardware is inconsistent (80.2 control, 79.5 treatment -- high variance)
- The verifier does not help because it cannot inject domain knowledge the handler lacks

To break the ceiling, the handler instructions themselves need enrichment (more specific troubleshooting steps, company-specific procedures, escalation paths). The verifier just applies the same model to review -- it cannot add knowledge the handler's instruction doesn't contain.

## Hypothesis H2 Verdict

**PARTIALLY CONFIRMED.** Adversarial verification produces a consistent but modest improvement (+1.8 pts). However:

1. The revision mechanism specifically is a net wash (-0.3 pts avg)
2. Most improvement comes from approved-first-pass cases (likely variance)
3. The verifier can hurt high-quality responses by over-revising (IT Case 11: -20)
4. Zero rejections means the verifier lacks discrimination power
5. It does NOT break the IT helpdesk ceiling

**Recommendation:** Verifier nodes add latency (extra LLM calls per case) for marginal quality gain. A better approach would be to enrich handler instructions directly, or use a stronger model as verifier while keeping the handler on flash. The same-model-as-verifier pattern lacks the "altitude" needed for meaningful quality control.

## Cost Analysis

- Elapsed time: 1368s (~23 minutes)
- Estimated LLM calls: ~250 (41 cases x 2 conditions x ~3 calls each)
- Control: 2 calls per case (classify + handler)
- Treatment: 3-5 calls per case (classify + handler + verifier + 0-2 revisions)
- Treatment is 50-150% more expensive per case than control

## Files Created

- `sandbox/14_adversarial_verification.py` -- Experiment runner
- `sandbox/customer_support_verified.yaml` -- Customer support DAG with verifier nodes
- `sandbox/it_helpdesk_verified.yaml` -- IT helpdesk DAG with verifier nodes
- `sandbox/scores_adversarial.json` -- Full per-case results
