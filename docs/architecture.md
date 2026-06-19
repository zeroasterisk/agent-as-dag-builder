# Architecture

## Overview

Graph Gardener executes agent workflows defined as YAML DAG configurations. The system reads a declarative config, builds an execution graph, and runs it through either a direct interpreter (for development) or Temporal (for durable production execution).

```
YAML Config --> DAG Interpreter --> Temporal Workflow --> ADK Agents
                    |                     |                   |
              Load & validate     Durable execution     LLM calls via
              Build routing       Retry & recovery      TemporalModel
              Evaluate edges      Crash recovery        Tool invocation
```

## Components

### 1. YAML Config

Declarative workflow definition with explicit nodes and edges. Each config file defines a complete DAG: node definitions (type, model, instruction), edge definitions (source, target, conditions), and metadata (name, version, defaults).

**Location:** Any `.yaml` file following the schema in `design/dag-config-spec.md`.

### 2. DAG Interpreter

Reads a YAML config and creates an executable routing structure. The interpreter:
- Parses nodes into a lookup table by ID
- Groups edges into start edges, conditional edges, and unconditional edges
- Evaluates edge conditions against node outputs and feature flags at runtime
- Traverses the graph from start nodes to completion

### 3. Temporal Runtime

Provides durable execution for production workflows. Each DAG node executes as a Temporal activity, giving:
- **Crash recovery** -- if a worker dies, Temporal replays the event history and resumes from the last completed activity
- **Automatic retries** -- failed activities retry per configurable policy (max attempts, backoff)
- **Versioned deployment** -- new DAG versions do not disrupt in-flight workflows
- **Observability** -- each node appears in the Temporal UI as a separate activity

### 4. ADK Agent Nodes

LLM calls via Google ADK `Agent` objects. When running in Temporal mode, agents use `TemporalModel` which wraps LLM calls as durable activities. In direct mode, agents use plain model name strings.

Each agent node has:
- A model (e.g., `gemini-3.1-flash-lite`)
- An instruction (the system prompt)
- Optional tools (function refs, MCP toolsets)
- Optional output schema constraints

### 5. A2A Nodes

Remote agent invocation via the Agent-to-Agent (A2A) protocol. Configured in YAML with an `agent_card` URL pointing to the remote agent's `/.well-known/agent.json`. Instantiated as `RemoteA2aAgent` at runtime.

### 6. MCP Nodes

Tool discovery and invocation via the Model Context Protocol. Configured in YAML with a command and args for the MCP server process. The node is backed by an ADK `Agent` with an `McpToolset` (or `TemporalMcpToolSet` for durable execution).

### 7. Learning Loop

The feedback mechanism that improves DAGs over time:

```
Benchmark Run --> Score (LLM-as-Judge) --> Analyze Weaknesses
      ^                                         |
      |                                         v
      +--- Validate & Write <--- Generate Improved Config
```

- **TaxonomyTracker** records interaction patterns (category, success rate, latency)
- **Benchmark harness** runs test cases through the DAG and scores responses
- **Analysis step** identifies weak cases and proposes targeted improvements
- **Config generator** produces an updated YAML with improvements applied
- **Promotion gates** (designed, not yet implemented) prevent regressions from reaching production

## Node Types

| Type | YAML `type` | Backed By | Description |
|---|---|---|---|
| Agent | `agent` | `google.adk.agents.Agent` | LLM call with instruction and optional tools |
| A2A | `a2a` | `RemoteA2aAgent` | Delegates to an external agent via A2A protocol |
| MCP | `mcp` | `Agent` + `McpToolset` | Agent with tools discovered from an MCP server |
| Function | `function` | (spec only) | Deterministic function call (HTTP, gRPC, script) |
| Validator | `validator` | (spec only) | Quality/safety check with pass/fail + fallback |
| Router | `router` | (spec only) | Explicit conditional dispatch node |

The first three types are implemented in prototype 09. Function, validator, and router types are defined in the spec but not yet implemented as code.

## Edge Model

Edges are first-class objects with optional conditions:

```yaml
edges:
  # Unconditional edge
  - from: handle_billing
    to: verify_response

  # Data-conditional edge (evaluated against node output)
  - from: classify
    to: handle_billing
    condition: "billing"

  # Feature-flag-gated edge
  - from: classify
    to: handle_billing_v2
    condition:
      flag: "billing_v2_enabled"
      operator: is
      value: true
    when: "$.intent == 'billing'"
    priority: 10
```

**Edge resolution rules:**
1. All edges from a completed node are evaluated
2. `condition` (simple string) is matched against the node's text output
3. `when` expressions evaluate against structured output data
4. `condition.flag` queries the external feature flag service
5. `priority` breaks ties when multiple edges match (highest wins)
6. Equal-priority matches execute in parallel (fan-out)

## Execution Modes

### Direct mode (development/testing)

No Temporal server required. Agents use plain model name strings. Execution is sequential with no durability, retry, or crash recovery. Useful for rapid iteration and local testing.

```
python sandbox/09_adk_temporal.py --no-temporal
```

### Temporal mode (production)

Requires a running Temporal server. Agents use `TemporalModel` for durable LLM calls. Every node execution is a Temporal activity with configurable retry policies. Workflows survive worker crashes and can be observed in the Temporal UI.

```
temporal server start-dev
python sandbox/09_adk_temporal.py
```

Both modes use the same YAML config and the same routing logic. The only difference is the model wrapper passed to ADK agents.
