# Graph Gardener (Agent-as-DAG-Builder)

Agents that build and continuously refine their own workflow DAGs. Instead of relying on LLM reasoning for every interaction, Graph Gardener lets agents encode learned patterns as executable YAML-defined workflows. A benchmark-driven learning loop proposes improvements, validates them against regression, and promotes winning configurations through canary deployment.

## How it works

1. **Define** a workflow as a YAML DAG config with typed nodes (agent, A2A, MCP) and explicit edges with conditional routing and feature flags
2. **Execute** the DAG through a direct interpreter (dev) or Temporal (production, durable)
3. **Benchmark** responses with an LLM-as-judge scorer across a test suite
4. **Learn** by analyzing weak cases and generating improved DAG configs
5. **Promote** improvements through validation gates and canary deployment

## Quick start

```bash
# Clone and set up
git clone https://github.com/zeroasterisk/agent-as-dag-builder.git
cd agent-as-dag-builder

# Install dependencies
pip install google-adk google-genai temporalio pyyaml

# Set up credentials (Vertex AI)
export GOOGLE_GENAI_USE_VERTEXAI=1
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_CLOUD_LOCATION=global

# Run the benchmark learning loop (3 iterations)
python sandbox/10_benchmark_learning.py

# Run the ADK Temporal DAG (direct mode, no Temporal server needed)
python sandbox/09_adk_temporal.py --no-temporal

# Run with Temporal (requires: temporal server start-dev)
python sandbox/09_adk_temporal.py
```

## Architecture

```
YAML Config --> DAG Interpreter --> Temporal Workflow --> ADK Agents
```

- **YAML Config** -- Declarative node + edge definitions with conditional routing and feature flags
- **DAG Interpreter** -- Loads config, builds routing table, evaluates edge conditions at runtime
- **Temporal Runtime** -- Durable execution with crash recovery, retries, and observability
- **ADK Agents** -- LLM calls via TemporalModel (durable) or direct (dev mode)
- **Learning Loop** -- Benchmark, score, analyze, propose, validate, promote

Node types: `agent` (LLM), `a2a` (inter-agent protocol), `mcp` (tool discovery).

## Documentation

- **[Architecture](docs/architecture.md)** -- System components, node types, edge model, execution modes
- **[Design Decisions](docs/design-decisions.md)** -- Key choices with rationale (YAML, explicit edges, Temporal, promotion gates)
- **[Research Objectives](docs/research/objectives.md)** -- Research questions, validation methodology, current status
- **[Research Log](docs/research/log/)** -- Dated entries for each milestone:
  - [Initial design](docs/research/log/2026-06-08-initial-design.md) -- Spec created, 15+ systems surveyed
  - [Prototypes 01-04](docs/research/log/2026-06-08-prototypes-01-04.md) -- ADK workflow from config
  - [Temporal integration](docs/research/log/2026-06-08-temporal-integration.md) -- Durable execution proven
  - [ADK Temporal plugin](docs/research/log/2026-06-19-adk-temporal-plugin.md) -- TemporalModel + A2A + MCP
  - [Benchmark learning loop](docs/research/log/2026-06-19-benchmark-learning-loop.md) -- 3 iterations, regression analysis
- **[DAG Config Spec](design/dag-config-spec.md)** -- Full v0.1.0 specification
- **[BDD Scenarios](design/bdd-scenarios.md)** -- Behavioral scenarios for the full lifecycle

## Current status

Research phase. Key findings:

| Objective | Status |
|---|---|
| LLM generates valid DAG configs | Proven |
| Learning loop improves DAGs | Proven (with caveats) |
| Guardrails prevent regression | In progress |
| A2A + MCP orchestration | Partially proven |
| Temporal durability | Proven |

## Project structure

```
design/              Design specs and BDD scenarios
docs/                Public-facing documentation
  architecture.md    System architecture
  design-decisions.md Key decisions with rationale
  research/
    objectives.md    Research goals and validation
    log/             Dated research log entries
sandbox/             Prototypes 01-10
  *.py               Executable prototype scripts
  *.yaml             DAG config files
```

## License

Apache 2.0
