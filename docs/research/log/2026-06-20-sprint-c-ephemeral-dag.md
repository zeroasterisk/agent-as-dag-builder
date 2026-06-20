# Sprint C: Ephemeral DAG Generation with Feedback Retry Loop

**Date:** 2026-06-20
**Prototype:** `sandbox/13_ephemeral_dag.py`
**Model:** gemini-3.5-flash (generation temp=0.3, expert/judge temp=0.0)

## Concept

Instead of pre-authored YAML DAGs, an LLM generates a complete DAG workflow
from a natural language task description. When execution produces poor results,
a simulated "domain expert" provides structured feedback, and the agent
regenerates the DAG incorporating that feedback (up to 3 retries per task).

## Architecture

```
Task description --> [DAG Generator] --> YAML DAG
                                            |
                                     [Execute vs test cases]
                                            |
                                     [Score each case 0-100]
                                            |
                                     [Domain Expert review]
                                       /          \
                                  Approved      Not approved
                                  (>=75)        (<75 + feedback)
                                    |               |
                                  Done        [Regenerate with feedback]
                                                    |
                                              (up to 3 retries)
```

## Novel Tasks Tested (10 domains)

| # | Domain | Categories Generated |
|---|--------|---------------------|
| 1 | Restaurant reservations | severe_allergy, large_party, dietary_preference, standard |
| 2 | Insurance claims | auto, home, health |
| 3 | Student enrollment | enrollment_request, drop_add_request, waitlist_management |
| 4 | Customer complaints | product_defect, shipping_issue, billing_dispute, service_quality |
| 5 | Job applications | chat_inquiry, resume_application |
| 6 | Travel booking | flights, hotels, car_rentals, packages, modifications |
| 7 | Medical appointment triage | emergency, urgent, routine |
| 8 | Real estate inquiries | buying, selling, renting, property_management |
| 9 | Event planning | corporate, wedding, birthday, conference |
| 10 | Financial advisory | investment, retirement, tax, debt, mortgage |

## Results (Full Run)

### Per-Task Results

```
=== Task 1: Restaurant reservations ===
  Attempt 1: Generated 4-node DAG -> Score: 68 -> "DAG YAML is incomplete, missing handle_standard node"
  Attempt 2: Generated 5-node DAG -> Score: 55 -> "Structurally incomplete, missing routing logic"
  Attempt 3: Generated 5-node DAG -> Score: 70 -> "Missing handler nodes for large_party"
  Attempt 4: Generated 5-node DAG -> Score: 68 -> "Missing node definitions for standard, large_party"
  Final: 68/100 (3 retries)

=== Task 2: Insurance claims ===
  Attempt 1: Generated 4-node DAG -> Score: 68 -> "Missing health claims handler node"
  Attempt 2: Generated 4-node DAG -> Score: 68 -> "YAML truncated, missing handle_health"
  Attempt 3: Generated 4-node DAG -> Score: 55 -> "Missing handle_health, routing transitions"
  Attempt 4: Generated 4-node DAG -> Score: 65 -> "Missing handle_health entirely"
  Final: 65/100 (3 retries)

=== Task 3: Student enrollment ===
  Attempt 1: Generated 4-node DAG -> Score: 42 -> "Fundamentally flawed, mutually exclusive classification"
  Attempt 2: Generated 5-node DAG -> Score: 65 -> "Classification-based routing fundamentally flawed"
  Attempt 3: Generated 6-node DAG -> Score: 65 -> "Missing waitlist management and drop/add nodes"
  Attempt 4: Generated 4-node DAG -> Score: 68 -> "Missing handle_waitlist node"
  Final: 68/100 (3 retries)

=== Task 4: Customer complaints ===
  Attempt 1: Generated 5-node DAG -> Score: 55 -> "YAML truncated, missing service quality handler"
  Attempt 2: Generated 5-node DAG -> Score: 72 -> "YAML truncated, missing billing/service nodes"
  Attempt 3: Generated 5-node DAG -> Score: 65 -> "Missing explicit routing logic"
  Attempt 4: Generated 5-node DAG -> Score: 68 -> "Missing handle_billing_dispute definitions"
  Final: 68/100 (3 retries)

=== Task 5: Job applications ===
  Attempt 1: Generated 4-node DAG -> Score: 55 -> "Missing handle_schedule_interview node"
  Attempt 2: Generated 4-node DAG -> Score: 58 -> "Bypasses screening, routes to match directly"
  Attempt 3: Generated 4-node DAG -> Score: 45 -> "Routes single queries to isolated steps"
  Attempt 4: Generated 5-node DAG -> Score: 65 -> "Missing scheduling interview category"
  Final: 65/100 (3 retries)

=== Task 6: Travel booking ===
  Attempt 1: Generated 5-node DAG -> Score: 72 -> "Incomplete car_rentals instruction"
  Attempt 2: Generated 6-node DAG -> Score: 70 -> "Truncated modifications handler"
  Attempt 3: Generated 6-node DAG -> Score: 78 -> APPROVED
  Final: 78/100 (2 retries) ** ONLY APPROVED TASK **

=== Task 7: Medical appointment triage ===
  Attempt 1: Generated 4-node DAG -> Score: 65 -> "Missing handle_routine node"
  Attempt 2: Generated 4-node DAG -> Score: 58 -> "Incomplete, only classification node"
  Attempt 3: Generated 4-node DAG -> Score: 68 -> "Missing urgent and routine handlers"
  Attempt 4: Generated 4-node DAG -> Score: 55 -> "Missing routine and urgent nodes"
  Final: 55/100 (3 retries)

=== Task 8: Real estate inquiries ===
  Attempt 1: Generated 5-node DAG -> Score: 65 -> "Missing renting and property mgmt handlers"
  Attempt 2: Generated 5-node DAG -> Score: 72 -> "Missing handle_property_management"
  Attempt 3: Generated 5-node DAG -> Score: 65 -> "Missing routing logic/edges"
  Attempt 4: Generated 5-node DAG -> Score: 58 -> "Missing routing transitions, property mgmt"
  Final: 58/100 (3 retries)

=== Task 9: Event planning ===
  Attempt 1: Generated 5-node DAG -> Score: 50 -> "Missing conference handler and routing"
  Attempt 2: Generated 5-node DAG -> Score: 58 -> "Truncated wedding handler, missing birthday"
  Attempt 3: Generated 5-node DAG -> Score: 50 -> "Truncated birthday instructions"
  Attempt 4: Generated 5-node DAG -> Score: 65 -> "Missing handle_birthday entirely"
  Final: 65/100 (3 retries)

=== Task 10: Financial advisory ===
  Attempt 1: Generated 5-node DAG -> Score: 62 -> "Missing handle_debt_management"
  Attempt 2: Generated 6-node DAG -> Score: 68 -> "Missing tax, debt, mortgage handlers"
  Attempt 3: (in progress when run killed)
  Final: 68/100 (2+ retries, incomplete)
```

### Summary Statistics

| Metric | Value |
|--------|-------|
| Tasks completed | 9 of 10 (task 10 partial) |
| Tasks approved (>=75) | 1/10 (Task 6: Travel booking) |
| Avg initial score | 60.2 |
| Avg final score | 65.6 |
| Avg improvement | +5.4 |
| Avg retries | 2.8 |
| Valid YAML rate | 100% (after prompt fix) |
| Total attempts | 37 |
| Approval rate | 2.7% (1/37 attempts) |

### Score Distribution (Final Scores)

```
90-100: 0
75-89:  1  #  (Travel booking: 78)
60-74:  7  ####### (Restaurant, Insurance, Enrollment, Complaints, Jobs, Event, Financial)
40-59:  2  ## (Medical triage: 55, Real estate: 58)
0-39:   0
```

## Key Findings

### Finding 1: YAML Truncation Corrupts Expert Review (Critical Bug)

The domain expert receives `dag_yaml[:2000]` for review. Generated DAGs with
4-5 nodes and detailed instructions routinely exceed 2000 characters. The
expert consistently sees truncated YAML and reports "incomplete structure" or
"missing handler nodes" even when the nodes exist and execute correctly.

**Evidence:** The expert's feedback almost always mentions "incomplete,"
"truncated," or "missing" nodes -- phrases that refer to the visible YAML
structure, not execution quality. Meanwhile, the per-case quality scores
(computed from actual execution) are reasonable (50-72 range).

**Fix:** Increase the truncation limit to 6000+ characters or pass the
complete YAML. This single fix would likely boost expert scores by 10-20
points on average.

### Finding 2: Feedback Loop Shows Marginal Improvement

Despite the expert providing specific feedback, average improvement across
retries was only +5.4 points (60.2 -> 65.6). This is partly due to Finding 1
(the expert keeps complaining about truncated YAML regardless of actual
improvements), but also reveals a fundamental challenge: the DAG generator
does not reliably incorporate feedback into regeneration.

**Pattern observed:**
- Task 3 (Student enrollment): Score went 42 -> 65 -> 65 -> 68 (+26 total)
- Task 4 (Customer complaints): Score went 55 -> 72 -> 65 -> 68 (non-monotonic)
- Task 7 (Medical triage): Score went 65 -> 58 -> 68 -> 55 (regressed)

The non-monotonic behavior suggests the generator sometimes "forgets" previous
improvements when incorporating new feedback.

### Finding 3: Task 6 (Travel Booking) Was the Only Approved Task

Travel booking succeeded because:
1. It has clear, mutually exclusive categories (flights, hotels, cars, packages)
2. The 6-node DAG (5 handlers + classifier) was small enough to pass truncation
3. The model naturally understands travel domain well
4. Adding "modifications" as a 5th category in attempt 3 addressed the expert's concern

### Finding 4: Sequential/Multi-Step Tasks Struggle with Classify-and-Route

Tasks 3 (Student enrollment) and 5 (Job applications) are inherently sequential
(verify prereqs THEN check capacity THEN confirm), but the classify-and-route
DAG pattern forces them into a single-step classification. The expert correctly
identified this: "fundamentally flawed because it uses mutually exclusive
classification."

**Implication:** The ephemeral DAG schema needs to support sequential pipeline
patterns, not just classify-and-route fan-out.

### Finding 5: Ephemeral-to-Durable Promotion Candidates

Based on structural signatures (node count, category count, edge pattern):

| Pattern | Tasks | Nodes | Categories |
|---------|-------|-------|-----------|
| 5-node, 4-category | Complaints, Event planning, Real estate, Financial | 5 | 4 |
| 4-node, 3-category | Insurance, Medical triage | 4 | 3 |

These similar structures suggest a reusable "4-category classifier" template
and a "3-category classifier" template could be promoted to durable configs.

## Recommendations for Next Sprint

1. **Fix YAML truncation bug** -- Pass complete YAML to expert (or at minimum
   increase to 6000 chars). This is the single highest-impact fix.

2. **Support sequential DAG patterns** -- The current schema only supports
   classify-and-route. Adding support for linear pipelines
   (`START -> step1 -> step2 -> step3`) would handle enrollment, job
   applications, and similar multi-step workflows.

3. **Improve feedback incorporation** -- The generator's prompt should include
   the previous YAML alongside the feedback so the model can make targeted
   edits rather than regenerating from scratch.

4. **Separate structural and execution scoring** -- The expert conflates YAML
   structure quality with execution quality. Split into two scores: structural
   validity (does the YAML have all needed nodes/edges?) and execution quality
   (do the responses match expected behavior?).

5. **Temperature tuning** -- DAG generation at temp=0.3 produces reasonable
   structures. Consider temp=0.1 for retry attempts to make more conservative
   edits rather than wholesale restructuring.

## Raw Data

Results JSON: `sandbox/scores_ephemeral.json` (saved on successful completion)
Script: `sandbox/13_ephemeral_dag.py`
Total LLM calls (estimated): ~400 (37 attempts x ~12 calls each)
Runtime: ~90 minutes (killed before completion of task 10)
