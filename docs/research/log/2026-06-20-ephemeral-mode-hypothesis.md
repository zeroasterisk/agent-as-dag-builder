# Research Log: Ephemeral DAG Mode + Adversarial Verification

**Date:** 2026-06-20
**Status:** Hypothesis — designing experiments

## Inspiration

Claude Code's Dynamic Workflows generate ephemeral execution harnesses per-task. Graph Gardener currently only supports durable (persisted) DAGs. The hybrid of both could be powerful.

## Hypotheses

### H1: Ephemeral DAG generation from task description
An LLM can generate a valid YAML DAG from a natural language task description, execute it, and produce a useful result — without any pre-existing DAG template.

**Experiment:** Give the system 10 novel task descriptions it has never seen. Measure: valid YAML rate, execution success rate, result quality.

### H2: Adversarial verification improves output quality
Adding a "verifier" node that challenges the output of other nodes will catch errors that a single-pass pipeline misses.

**Experiment:** Run the same test cases with and without a verifier node. Measure: quality score delta, false positive rate of the verifier.

### H3: Weaker models expose more learning opportunities
Using a less capable model (gemini-3.1-flash-lite) creates more room for the learning loop to improve — the ceiling is lower, so the delta from DAG optimization is larger.

**Experiment:** Run the same 3-harness benchmark with gemini-3.1-flash-lite instead of gemini-2.5-flash. Compare: baseline scores, improvement rate per iteration, and whether the learning loop compensates for model weakness.

### H4: Ephemeral-to-durable promotion
Patterns that repeat 3+ times as ephemeral DAGs should be automatically promoted to durable (persisted) DAGs, capturing learned optimizations.

**Experiment:** Generate 20 ephemeral DAGs from varied task descriptions. Cluster by similarity. Verify that repeated patterns get promoted and subsequent executions use the promoted (optimized) version.

## Model Choice

Switching to gemini-3.1-flash-lite for experiments:
- Faster inference, lower cost per experiment
- Less capable = more room for DAG optimization to matter
- If the learning loop can improve a weak model's output via better orchestration, that's a stronger signal than improving an already-capable model

## Planned Experiments (priority order)

1. **H3 first** — rerun 3-harness benchmark with flash-lite, 5 iterations. Quick validation.
2. **H2 next** — add verifier node to customer_support DAG, compare with/without.
3. **H1 then** — ephemeral DAG generation from 10 novel task descriptions.
4. **H4 last** — ephemeral-to-durable promotion (requires H1 working).
