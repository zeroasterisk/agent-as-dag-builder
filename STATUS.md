# Graph Gardener — Project Status

**Date:** 2026-07-01
**Phase:** Research (sandbox prototypes complete, production implementation not started)

---

## What's Working

### Proven Concepts

| Capability | Evidence | Prototype |
|---|---|---|
| **LLM generates valid DAG configs** | YAML configs parse, validate, and execute across 6 domains | 03, 07, 10 |
| **Temporal durable execution** | 3 workflows completed through live Temporal server with crash recovery | 05, 06, 08 |
| **ADK Temporal plugin** | `TemporalModel` eliminates manual activity wrapping; MCP tools via `TemporalMcpToolSet` | 09 |
| **Benchmark learning loop** | LLM-as-judge scoring across 180 test cases (6 domains × 30 cases) | 10, 11, 18 |
| **Held-out validation gates** | Prevented 6/7 overfitting promotions in HTR; 6/9 in broad benchmark | 17, 18 |
| **Hypothesis-tree refinement** | 3-strategy branching (templates, knowledge, clarity) with insight memory | 17 |
| **Cross-domain generalization** | Same learning loop works across customer support, IT helpdesk, sales, healthcare, e-commerce, legal | 18 |

### Key Metrics

- **Best holdout aggregate (3 harnesses):** 87.6/100 — near ceiling for current configs
- **Best holdout aggregate (6 harnesses):** 87.7/100
- **Legal intake improvement:** 75.4 → 83.3 holdout (+7.9 pts) from 3 promoted configs
- **Category classification accuracy:** 89-100% across all 6 domains
- **Promotion success rate:** 3/9 hypotheses promoted (broad), 1/7 (HTR) — held-out gate works

### DAG Config Spec (v0.1.0-draft)

The YAML config format is fully specified with:
- 5 node types: `function`, `agent`, `sub_dag`, `validator`, `router`
- First-class edges with feature flag conditions, data conditions, priority-based resolution
- JSON Schema for validation
- Agent tool commands for structured DAG mutation (`dag.add_node`, `dag.add_edge`, `dag.validate`, etc.)
- Full lifecycle: pattern recognition → proposal → validation → canary → promotion

### Sandbox Prototypes (01-18)

18 working prototypes covering the full stack:
- **01-04:** Config loading, conditional routing, ADK workflow from YAML
- **05-08:** Temporal integration, live durable execution
- **09:** ADK Temporal plugin with A2A + MCP node types
- **10-12:** Benchmark learning loop, multi-harness, flash-lite experiments
- **13-16:** Ephemeral DAG generation, adversarial verification, context strategies, IT helpdesk sprint
- **17:** Hypothesis-tree refinement with held-out validation
- **18:** 6-harness broad benchmark (180 test cases)

---

## What's Not Working / Known Issues

1. **18_broad_harness.py crash** — final summary print crashed on `NoneType` format (fixed: `winner_strategy` default changed from `None` to `"-"`)

2. **HTR research log incomplete** — `2026-06-21-hypothesis-tree.md` had placeholder sections never filled in after run (fixed: populated from `scores_hypothesis_tree.json`)

3. **Overfitting remains the primary risk** — hypotheses that improve training scores consistently regress on holdout. The held-out gate catches this, but the *optimizer itself* doesn't learn to avoid overfitting strategies. Insight memory helps but doesn't prevent the optimizer from re-targeting domains where no strategy works.

4. **A2A + MCP nodes are structurally defined but not end-to-end tested** — Prototype 09 instantiates `RemoteA2aAgent` and `McpToolset` from YAML config, but no live remote agent testing was done.

5. **No production code** — everything is sandbox prototypes. The DAG interpreter, Temporal bridge, promotion pipeline, and feature flag integration exist only as designs and BDD scenarios.

---

## What's Next

### Near-term (prototype refinements)

- **Adaptive strategy selection:** Use insight memory to skip domains/strategies with consistently negative holdout deltas instead of always targeting the lowest absolute score
- **Larger holdout sets:** 6-7 cases per holdout (30%) has high variance (~5-10 pts between evaluation passes). Increasing to 15-20 cases or using 3x averaging would reduce noise
- **A2A end-to-end test:** Stand up a remote agent and test the full A2A flow from YAML config through ADK

### Medium-term (production path)

- **DAG interpreter implementation:** ~2000 lines for the core interpreter (edge evaluation, node execution, state management)
- **JSON Schema validation tooling:** Automated pre-deployment validation (schema + graph analysis + simulation)
- **Promotion pipeline:** Canary deployment with feature flags, metric monitoring, rollback triggers (designed in BDD scenarios, Group 4)
- **Config persistence + versioning:** Git-backed DAG configs with CI/CD integration

### Long-term (system maturity)

- **Taxonomy tracker:** Automatic DAG creation suggestions based on interaction frequency and success rate
- **Pattern recognition agent:** Background analysis of trace stores to propose new DAG paths
- **Multi-model support:** Extend beyond Gemini to Claude, GPT-4, etc. (config already supports `model` field per node)

---

## Connection to A2A and Exgentic

### A2A (Agent-to-Agent Protocol)

Graph Gardener already defines `a2a` as a first-class node type in the DAG config spec. An A2A node wraps a `RemoteA2aAgent` pointed at an external agent's `/.well-known/agent.json`. This means:

- **DAG nodes can delegate to any A2A-compliant agent** — the DAG builder doesn't need to know the remote agent's implementation, just its agent card URL
- **DAG configs become orchestration contracts** — the YAML defines what gets delegated where, with edge conditions controlling routing
- **Exgentic agents can be A2A nodes** — if Exgentic benchmark agents expose A2A endpoints, they can be wired into Graph Gardener DAGs as remote nodes

### Exgentic / Benchmarking

The connection to Exgentic is through the **benchmark-driven learning loop**:

1. **Exgentic provides the test harness** — benchmark test cases with expected outcomes and quality criteria
2. **Graph Gardener provides the optimizer** — the hypothesis-tree loop that proposes, validates, and promotes DAG improvements
3. **A2A is the wire protocol** — Exgentic benchmark runners can invoke Graph Gardener DAGs via A2A, and Graph Gardener nodes can call Exgentic evaluation agents via A2A

Concrete integration path:
- Expose the Graph Gardener benchmark harness as an A2A agent (accepts test cases, returns scored results)
- Wire Exgentic's benchmark framework to submit cases via A2A
- Use Exgentic's scoring infrastructure in place of (or alongside) the current LLM-as-judge scorer
- Feed Exgentic benchmark results back into Graph Gardener's learning loop to drive DAG improvements

The 6-domain benchmark (180 test cases across customer support, IT helpdesk, sales, healthcare, e-commerce, and legal) is already structured as an Exgentic-compatible evaluation suite — it just needs the A2A wrapper to become accessible from external systems.

---

## Project Health

| Dimension | Assessment |
|---|---|
| **Design** | Strong — comprehensive spec with good survey work, explicit rationale for decisions |
| **Prototypes** | Thorough — 18 prototypes covering the full concept space |
| **Documentation** | Good — architecture, design decisions, research logs, BDD scenarios all exist |
| **Test coverage** | None — sandbox prototypes are scripts, not tested code |
| **Production readiness** | Not started — no package structure, no CI, no deployment artifacts |
| **Research completeness** | High — 5 research questions, 4 answered, 1 in progress |
