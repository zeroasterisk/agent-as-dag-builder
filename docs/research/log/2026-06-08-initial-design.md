# 2026-06-08 -- Initial Design Spec

## What happened

Created the DAG configuration specification (v0.1.0-draft) after surveying 15+ workflow configuration systems. The spec defines a YAML-based DAG language for agents that build and refine their own workflows.

## Systems surveyed

Argo Workflows, AWS Step Functions (ASL), GitHub Actions, Tekton Pipelines, Temporal, Zigflow, Apache Airflow + DAG Factory, Dapr Workflows, Prefect, CNCF Serverless Workflow DSL 1.0, Google ADK Agent Config, CrewAI, LangGraph, and four config languages (CUE, Jsonnet, Dhall, Nickel).

Each system was evaluated against eight criteria: LLM editability, conditional support, feature flag support, diffability, pre-execution validation, node type flexibility, runtime availability, and maturity.

## Key finding

No existing system scores well across all criteria. In particular, no system combines native feature flag support on edges with agent-invocation nodes in a YAML config. All existing systems put conditions on nodes (Argo's `when`, Tekton's `when`) or use routing states (ASL's `Choice`, Serverless Workflow's `switch`). None are designed for LLM-driven mutation of the workflow definition itself.

## Decision: custom YAML format

We chose a custom YAML format synthesizing ideas from multiple systems:

| Borrowed From | What We Take |
|---|---|
| LangGraph | Explicit node + edge + conditional-edge graph model |
| CNCF Serverless Workflow | YAML fluency, task type system |
| Google ADK Agent Config | Agent-native node definitions |
| Argo Workflows | DAG task structure, parameter passing |
| AWS Step Functions | Error handling, retry policies |

## Design criteria

- **LLM-native**: Flat, regular YAML that LLMs can reliably generate and edit
- **Schema-validatable**: Full JSON Schema for pre-execution validation
- **Runtime-portable**: Config is separate from any specific execution engine
- **Explicit graph model**: Nodes AND edges are first-class, not implicit
- **Feature-flag-aware**: Edges carry optional conditions referencing external flags

## Artifacts produced

- `design/dag-config-spec.md` -- Full specification with survey, recommendation, node/edge definitions, JSON schema, agent-DAG interaction loop, and runtime architecture
- `design/bdd-scenarios.md` -- BDD scenarios covering cold start, DAG execution, failure/recovery, mutation, canary deployment, and taxonomy tracking
