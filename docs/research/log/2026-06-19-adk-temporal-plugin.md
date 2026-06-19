# 2026-06-19 -- Prototype 09: ADK Temporal Plugin

## Summary

Discovered that the Temporal SDK has an official ADK plugin (`temporalio[google-adk]`) that provides `TemporalModel` for durable LLM calls and `activity_tool` for durable tool execution. Rewrote the prototype to use this plugin instead of raw Temporal activities, and added A2A and MCP node types to the YAML config format. All 6 test scenarios passed in both direct and Temporal modes.

## Discovery

The `temporalio.contrib.google_adk_agents` module provides:

- **`TemporalModel`** -- Wraps any model name so that LLM calls execute as Temporal activities. This means every LLM call is automatically durable, retried, and observable without manual activity wrapping.
- **`GoogleAdkPlugin`** -- Temporal client plugin that registers the ADK integration.
- **`activity_tool`** -- Makes any ADK tool call execute as a durable Temporal activity.
- **`TemporalMcpToolSet` / `TemporalMcpToolSetProvider`** -- Wraps MCP tool calls in Temporal activities.

This eliminates the manual activity-per-node pattern from prototypes 05-08. Instead of wrapping each LLM call in a `@activity.defn`, we create standard ADK `Agent` objects with `TemporalModel` as their model, and the plugin handles durability automatically.

## Prototype 09: YAML-driven durable agent DAG

**File:** `sandbox/09_adk_temporal.py`

Rewrote the DAG executor to use:

1. **`Agent` with `TemporalModel`** for agent nodes -- each LLM call is a durable activity
2. **`RemoteA2aAgent`** for A2A nodes -- delegates to external agents via the Agent-to-Agent protocol
3. **`McpToolset` / `TemporalMcpToolSet`** for MCP nodes -- discovers and calls tools from MCP servers

The YAML config format was extended (`customer_support_adk.yaml`) with three node types:

```yaml
nodes:
  - id: classify
    type: agent           # ADK Agent with TemporalModel
    model: gemini-3.1-flash-lite
    instruction: "..."

  - id: escalate_a2a
    type: a2a             # RemoteA2aAgent via A2A protocol
    agent_card: "http://localhost:9100/.well-known/agent.json"

  - id: lookup_mcp
    type: mcp             # Agent with MCP toolset
    command: "npx"
    args: ["-y", "@anthropic/mcp-server-memory"]
```

## Execution modes

The prototype supports two modes:

- **Direct mode** (`--no-temporal`): Uses plain ADK agents with string model names. No durability. Good for development and testing.
- **Temporal mode** (default): Uses `TemporalModel` for all agent nodes. Every LLM call is a durable Temporal activity with retry policies.

Both modes use the same YAML config and the same routing logic. The only difference is whether `TemporalModel` or a plain string is passed as the model.

## Test results

All 6 test scenarios passed (3 queries x 2 modes):

| Query | Category | Direct | Temporal |
|---|---|---|---|
| "I was charged twice for my subscription" | billing | Pass | Pass |
| "My internet keeps disconnecting every hour" | technical | Pass | Pass |
| "What are your business hours?" | general | Pass | Pass |

In Temporal mode, all workflows completed with status `COMPLETED` and each agent node appeared as a separate activity in the Temporal UI.

## Key takeaways

1. The ADK Temporal plugin (`temporalio[google-adk]`) eliminates manual activity wrapping -- use `TemporalModel` and get durability for free
2. A2A and MCP node types can be defined in YAML and instantiated at runtime, enabling heterogeneous agent workflows
3. The same config works in both direct and Temporal modes, supporting a dev-to-production pipeline
4. Retry policies and timeouts are configured per-node via the `TemporalModel` activity config
