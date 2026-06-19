# Graph Gardener -- Research Objectives

## Vision

Instead of agents operating purely through LLM reasoning (expensive, slow, non-deterministic), agents learn from interactions and encode their knowledge as executable workflow DAGs. These DAGs handle the majority of known tasks deterministically, while the live agent handles novel or exceptional cases -- and then updates the DAG with what it learned. The system continuously improves through a benchmark-driven learning loop with promotion gates to prevent regressions.

## Research Questions

### 1. Can AI agents reliably generate valid DAG configurations?

Can an LLM produce syntactically correct, structurally sound YAML DAG configs that pass schema validation and execute without errors?

**Validation:**
- Schema validation pass rate against the JSON Schema spec
- Structural correctness (acyclic, all nodes reachable, no orphans, valid edge targets)
- End-to-end execution success rate for generated configs

**Status:** Proven. Prototypes 03, 07, and 10 demonstrate LLM-generated YAML configs that parse, validate, and execute correctly. The learning loop in prototype 10 generates valid configs across multiple iterations.

### 2. Can a learning loop improve DAG performance over time?

Given benchmark results and an analysis step, can the system propose targeted improvements that measurably increase DAG quality?

**Validation:**
- Benchmark score improvement across iterations
- Category accuracy maintenance (no regressions in routing)
- Quality and helpfulness score trends

**Status:** Proven with caveats. The learning loop improved scores from v1 (92.3) to v2 (93.8), but v3 regressed to 90.5 due to over-fitting. The mechanism works but requires guardrails (see question 3).

### 3. What guardrails prevent over-fitting and regression?

How do we prevent the learning loop from degrading DAG quality through over-optimization?

**Validation:**
- Promotion gates that compare new version against previous version
- Canary testing with rollback triggers
- Holdout validation sets to detect over-fitting
- Regression detection before full promotion

**Status:** In progress. The need for promotion gates is proven by the v3 regression in prototype 10. The canary deployment pattern is designed in the BDD scenarios but not yet implemented as code.

### 4. Can YAML DAGs orchestrate heterogeneous agents via A2A and MCP?

Can a single YAML config define workflows that mix local LLM agents, remote A2A agents, and MCP tool-calling agents?

**Validation:**
- End-to-end workflow execution with A2A remote agent calls
- MCP tool discovery and invocation from YAML-defined nodes
- Mixed workflows (agent + a2a + mcp nodes in one DAG)

**Status:** Partially proven. Prototype 09 defines A2A and MCP node types in YAML and instantiates them correctly. The ADK integration code builds `RemoteA2aAgent` and `McpToolset` objects from config. Full end-to-end testing with live remote agents is not yet complete.

### 5. Does Temporal provide meaningful durability for agent workflows?

Does wrapping agent DAGs in Temporal workflows give us crash recovery, retry behavior, and observability that justifies the infrastructure?

**Validation:**
- Crash recovery: workflow survives worker restarts
- Retry behavior: failed activities are retried per policy
- Observability: each DAG node visible in Temporal UI
- Versioned deployment: new DAG versions do not disrupt in-flight workflows

**Status:** Proven. Prototype 08 demonstrated 3 workflows completing through a live Temporal server with full durable execution. Prototype 09 proved that the ADK Temporal plugin (`TemporalModel`) eliminates manual activity wrapping while preserving all durability guarantees.

## Current Status

| # | Objective | Status |
|---|---|---|
| 1 | LLM generates valid DAG configs | Proven |
| 2 | Learning loop improves DAGs | Proven (with caveats) |
| 3 | Guardrails prevent regression | In progress |
| 4 | A2A + MCP orchestration | Partially proven |
| 5 | Temporal durability | Proven |

## Validation Methodology

### Sandbox prototypes

Each research question is validated through focused sandbox prototypes (`sandbox/01` through `sandbox/10`). Prototypes are intentionally minimal -- they test one concept and produce a clear pass/fail signal.

### Benchmark harness

Prototype 10 establishes a repeatable benchmark with 23 test cases, LLM-as-judge scoring, and automated analysis. The benchmark measures three dimensions: category accuracy (deterministic, 40 points), quality (LLM-judged, 40 points), and helpfulness (LLM-judged, 20 points).

### Iteration and comparison

The learning loop runs multiple benchmark iterations, comparing scores across versions. This produces a concrete improvement trajectory and surfaces failure modes (like the v3 over-fitting regression) that inform guardrail design.

### BDD scenarios

The BDD scenarios in `design/bdd-scenarios.md` define acceptance criteria for the full system lifecycle: cold start, DAG execution, failure/recovery, mutation, canary deployment, and taxonomy tracking. These are the target criteria for a production implementation.
