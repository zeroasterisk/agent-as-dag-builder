# About Graph Gardener

## Backstory

Alan Blount has been advocating for this design — agents that build, refine, and maintain their own workflow graphs — since 2023. The core insight: hand-authored agent orchestration doesn't scale. As agent ecosystems grow, the wiring between agents becomes the bottleneck. The graph itself should be a living artifact that agents can observe, propose changes to, validate, and evolve.

The idea sat as a design conviction through several generations of agent frameworks, waiting for the right building blocks to mature. In mid-2026, three things converged:

1. **Temporal's ADK integration** shipped, giving durable execution to ADK agents without custom infrastructure
2. **A2A and MCP protocols** matured enough that agents could discover and call each other at runtime
3. **ARD (Agentic Resource Discovery)** launched, making agent capabilities searchable across organizations

With these pieces in place, Alan started building Graph Gardener — turning a long-held design conviction into working code. The first prototype went from zero to a proven learning loop (YAML config → Temporal execution → benchmark → evolve DAG) in under two weeks.

## Why "Graph Gardener"

The name reflects the philosophy: the system doesn't just build graphs, it tends them. Like a gardener, it plants new nodes when patterns emerge, prunes paths that underperform, and cultivates the overall structure toward better outcomes — but always under the supervision of the humans who set the goals.

## Project

- **Author:** Alan Blount ([@zeroasterisk](https://github.com/zeroasterisk))
- **Repository:** [agent-as-dag-builder](https://github.com/zeroasterisk/agent-as-dag-builder)
- **Started:** June 2026 (concept since 2023)
- **Status:** Research prototype with proven learning loop
