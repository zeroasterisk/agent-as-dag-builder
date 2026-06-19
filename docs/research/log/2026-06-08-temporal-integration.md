# 2026-06-08 -- Prototypes 05-08: Temporal Integration

## Summary

Proved that YAML-defined DAGs can execute as durable Temporal workflows. Starting from raw Temporal activities wrapping LLM calls, we built up to a generic YAML-to-Temporal interpreter and ran live workflows through a Temporal dev server. Three workflows completed with full durable execution.

## Prototype 05: Raw Temporal integration

**File:** `sandbox/05_temporal_integration.py`

Wrapped individual DAG steps as Temporal activities: `classify_request`, `handle_billing`, `handle_technical`, `handle_general`, and `validate_response`. Each activity makes an LLM call via the Gemini API. A `CustomerSupportWorkflow` orchestrates the classify-route-handle-validate pipeline.

**Result:** The pattern works in both modes:
- With a running Temporal server: full durable execution with crash recovery
- Without Temporal: direct activity calls demonstrating the same pipeline

Each activity is independently retryable and observable. If a worker crashes mid-workflow, Temporal replays the event history and resumes from the last completed activity.

## Prototype 06: YAML-to-Temporal compiler

**File:** `sandbox/06_yaml_to_temporal.py`

Built a `DAGInterpreter` class that reads a YAML config and executes it step by step. Each node runs through a generic `run_llm_node` activity that takes node ID, instruction, model, and user input. Edge conditions are evaluated against node outputs, with feature flag support via a flags dict.

The interpreter implements topological traversal: find start edges, execute start nodes, evaluate outgoing edges using condition matching and feature flags, execute matching targets, repeat until no more nodes.

**Result:** Successfully ran the `customer_support.yaml` config through the interpreter with three test queries. Feature flag toggling worked -- enabling `billing_v2=True` caused the interpreter to select the v2 edge.

## Prototype 07: Agent learning loop

**File:** `sandbox/07_agent_learning_loop.py`

Implemented the core "agent-as-DAG-builder" concept: an agent that handles requests live, tracks patterns via a `TaxonomyTracker`, and proposes DAG configurations when a category reaches a frequency threshold.

The `TaxonomyTracker` records interaction category, success rate, and latency. When a category exceeds the threshold (configurable, default 3 for demo), the agent uses an LLM to generate a YAML DAG config based on example interactions from that category.

**Result:** After processing 9 simulated queries across billing, technical, and general categories, the tracker identified billing (4 occurrences) as a DAG candidate and generated a valid YAML config with validate-handle-verify nodes.

## Prototype 08: Live Temporal execution -- PROVEN WORKING

**File:** `sandbox/08_temporal_live.py`

The culminating proof: ran three workflows through a live Temporal dev server with full durable execution.

**Setup:** `temporal server start-dev` running locally.

**Results:**
- Connected to Temporal server
- Worker registered on `dag-builder` task queue
- Three workflows executed:
  - "I was charged twice" -> classified as billing, handled by billing specialist
  - "Internet keeps dropping" -> classified as technical, handled by tech support
  - "What are your hours?" -> classified as general, handled by general agent
- All three workflow statuses: `COMPLETED`
- Each step is a durable Activity with crash recovery and retry

This proves the full pipeline: YAML config defines the DAG, Temporal provides durable execution, each node is independently retryable and observable.

## Key takeaways

1. Temporal provides exactly the durability guarantees we need: crash recovery, automatic retries, workflow versioning
2. The YAML-to-Temporal bridge works -- a generic interpreter can execute any config-defined DAG
3. Feature flags can be injected at edge evaluation time without modifying the workflow structure
4. The `sandboxed=False` flag is needed for Temporal workflows that make async I/O calls (LLM calls)
5. Activity-level granularity gives good observability -- each DAG node appears as a separate activity in the Temporal UI
