# Design Decisions

Key decisions made during the Graph Gardener research phase, with rationale and alternatives considered.

## 1. Why YAML over JSON or code for DAG config

**Decision:** Use YAML as the primary config format for workflow DAGs.

**Alternatives considered:**
- JSON (AWS Step Functions style)
- Python code (LangGraph, Temporal workflow-as-code)
- CUE / Jsonnet / Dhall (typed config languages)

**Rationale:**

YAML is the best format for LLM editability. LLMs generate syntactically correct YAML more reliably than JSON (no trailing comma issues, no bracket matching) and more safely than executable code (no injection risk). YAML diffs are clean and readable in git, making version control of workflow changes practical.

JSON was rejected primarily because it is verbose and bracket-heavy. LLMs make more syntax errors with JSON, and JSON diffs are noisy (brackets, commas). AWS Step Functions' ASL demonstrates the problem: renaming a state key breaks all references.

Code-first approaches (LangGraph, raw Temporal) were rejected because LLM-driven mutation of executable code is unsafe. A declarative config can be validated before execution; code cannot be safely validated without running it.

Typed config languages (CUE, Jsonnet, Dhall) solve a different problem -- they generate valid YAML/JSON from templates. CUE in particular has a high learning curve that makes it hostile to LLM generation. These could serve as a schema/validation layer on top of YAML, but the workflow definition itself should be plain YAML.

## 2. Why explicit edges (not implicit dependencies)

**Decision:** Model edges as first-class objects in the config, separate from nodes.

**Alternatives considered:**
- Implicit dependencies via `dependencies` lists on nodes (Argo Workflows style)
- Implicit ordering via sequential position (CNCF Serverless Workflow style)
- Routing states that mediate transitions (AWS Step Functions `Choice` style)

**Rationale:**

Feature flag conditions belong on edges, not on nodes. The same handler node should be reachable via multiple paths with different flag configurations. A v1 billing handler and a v2 billing handler can coexist in the same DAG, distinguished only by which edge is active.

Explicit edges are more diffable. Adding a feature-flag-gated alternative path means adding an edge to the config. With implicit dependencies, adding an alternative path requires modifying the target node's dependency list and adding a condition, which conflates routing logic with node definition.

The explicit edge model is borrowed from LangGraph, which has the cleanest graph API among the surveyed systems. LangGraph uses `add_edge(source, target)` and `add_conditional_edges(source, routing_function, mapping)` -- our YAML edges are the declarative equivalent.

## 3. Why Temporal (not ADK-only, not Dapr, not custom)

**Decision:** Use Temporal as the production runtime for durable DAG execution.

**Alternatives considered:**
- ADK-only (rely on ADK's built-in workflow execution)
- Dapr Workflows (sidecar-based durable execution)
- Custom runtime (build our own retry/recovery logic)
- Argo Workflows (Kubernetes-native)

**Rationale:**

Temporal is battle-tested for durable execution at scale. It provides exactly what agent workflows need: crash recovery via event replay, automatic retries with configurable backoff, workflow versioning for safe deployments, and rich observability.

ADK-only was rejected because ADK's workflow execution has no durability guarantees. If a worker crashes mid-workflow, the entire interaction is lost. ADK is excellent for defining agent behavior but does not solve the infrastructure problem.

Dapr Workflows was rejected because it is code-only with no declarative config option for workflow logic. YAML in Dapr is limited to operational configuration (concurrency limits), not workflow definitions.

A custom runtime was considered but would require reimplementing retry logic, crash recovery, event sourcing, and observability -- all of which Temporal provides out of the box. The estimated 2000 lines for a basic interpreter does not include production-grade durability.

Argo Workflows was rejected because it is Kubernetes-specific and container-centric. Wrapping LLM calls in containers adds unnecessary overhead for agent workflows.

## 4. Why ADK Temporal plugin (not raw temporalio)

**Decision:** Use the `temporalio[google-adk]` plugin with `TemporalModel` instead of manually wrapping each node in a Temporal activity.

**Alternatives considered:**
- Raw `@activity.defn` for each DAG node (prototypes 05-08)
- Custom activity wrapper that calls ADK agents

**Rationale:**

The ADK Temporal plugin eliminates boilerplate. With `TemporalModel`, every LLM call automatically executes as a durable Temporal activity. There is no need to define an `@activity.defn` for each node, serialize inputs/outputs, or manage activity registration. The plugin also handles MCP tool calls via `TemporalMcpToolSet`.

The raw approach (prototypes 05-08) worked but required writing and registering a separate activity for each node type. The plugin approach (prototype 09) uses standard ADK `Agent` objects with `TemporalModel` swapped in as the model, achieving the same durability with significantly less code.

Both approaches use the same Temporal infrastructure. The plugin is a convenience layer, not a different runtime.

## 5. Why LLM-as-judge for benchmark scoring

**Decision:** Use an LLM (separate from the DAG's own agents) to score response quality and helpfulness.

**Alternatives considered:**
- Deterministic scoring (keyword matching, schema validation only)
- Human evaluation
- Self-evaluation (DAG agents score their own output)

**Rationale:**

Customer service response quality is inherently subjective and context-dependent. A deterministic scorer cannot evaluate whether a response is "empathetic" or "actionable." Human evaluation does not scale and cannot be automated in a learning loop. Self-evaluation creates circular bias.

Using a separate, more capable model (gemini-2.5-flash) as a judge avoids self-evaluation bias. The judge evaluates against explicit quality criteria defined per test case, producing a structured score with reasoning. This is the same pattern used by academic LLM benchmarks (MT-Bench, Arena).

The scoring rubric splits into three dimensions: category accuracy (40 points, deterministic), quality (40 points, LLM-judged), and helpfulness (20 points, LLM-judged). The deterministic component ensures that routing correctness is never subject to judge variance.

## 6. Why the learning loop needs promotion gates

**Decision:** DAG improvements must pass validation gates before full promotion. Auto-promoting based on learning loop output alone is unsafe.

**Alternatives considered:**
- Auto-promote: if the learning loop proposes a change, apply it immediately
- Human-only review: require manual approval for all DAG changes
- Threshold-based: promote if aggregate score exceeds a fixed threshold

**Rationale:**

The v3 regression in prototype 10 proved this empirically. The learning loop improved scores from v1 (92.3) to v2 (93.8), but v3 regressed to 90.5 because the optimizer over-fit to the weakest cases from v2, making instructions too specific and reducing generality.

Auto-promotion is unsafe because any optimization step can introduce regressions. The learning loop generates improvements based on failure analysis, but it has no mechanism to verify that improvements to weak cases do not degrade strong cases.

A fixed threshold is insufficient because it does not detect relative regression. A config scoring 90.5 would pass an absolute threshold of 85, even though it is worse than the 93.8 it replaced.

The designed solution combines: canary testing (deploy to 5% of traffic), regression detection (compare against previous version, not just absolute thresholds), holdout validation (reserve test cases not used in analysis), and rollback triggers (automatic revert if metrics drop). This is described in the BDD scenarios (Group 4) and matches industry practice for progressive deployment.
