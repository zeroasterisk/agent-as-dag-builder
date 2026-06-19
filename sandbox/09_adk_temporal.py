"""Prototype 09: ADK Temporal integration -- YAML-driven durable agent DAG.

Migrates from raw temporalio + raw google.genai (prototypes 05-08) to:
  - google.adk.Agent          for each node
  - TemporalModel             for durable LLM calls
  - activity_tool              for durable tool execution
  - RemoteA2aAgent            for A2A inter-agent calls
  - TemporalMcpToolSet        for MCP tool calls
  - Temporal Workflow          as the single execution graph

Usage:
  # With Temporal dev server running:
  temporal server start-dev          # in another terminal
  python sandbox/09_adk_temporal.py

  # Without Temporal (falls back to direct ADK execution):
  python sandbox/09_adk_temporal.py --no-temporal

Environment:
  GOOGLE_GENAI_USE_VERTEXAI=1
  GOOGLE_CLOUD_PROJECT=alanblount-demo
  GOOGLE_CLOUD_LOCATION=global
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from contextlib import aclosing
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

import yaml

# -- Environment defaults (Vertex AI) ----------------------------------------
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "1")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "alanblount-demo")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")

# -- ADK imports --------------------------------------------------------------
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types

# -- Temporal + ADK Temporal plugin imports ------------------------------------
from temporalio import activity, workflow
from temporalio.client import Client as TemporalClient
from temporalio.common import RetryPolicy
from temporalio.worker import Worker

# The ADK Temporal integration lives in the Temporal SDK, not in google-adk.
from temporalio.contrib.google_adk_agents import (
    GoogleAdkPlugin,
    TemporalModel,
)
from temporalio.contrib.google_adk_agents.workflow import activity_tool

# -- Optional: A2A & MCP (imported lazily when the config uses them) ----------
try:
    from google.adk.agents.remote_a2a_agent import RemoteA2aAgent

    _HAS_A2A = True
except ImportError:
    _HAS_A2A = False

try:
    from google.adk.tools.mcp_tool import McpToolset
    from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
    from mcp import StdioServerParameters
    from temporalio.contrib.google_adk_agents import (
        TemporalMcpToolSet,
        TemporalMcpToolSetProvider,
    )

    _HAS_MCP = True
except ImportError:
    _HAS_MCP = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("dag_builder")


# =============================================================================
# 1. YAML config loader
# =============================================================================

DEFAULT_CONFIG = Path(__file__).parent / "customer_support_adk.yaml"


def load_dag_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    """Load and validate a DAG YAML config."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    config = raw["dag"]
    assert "nodes" in config, "YAML must define 'nodes'"
    assert "edges" in config, "YAML must define 'edges'"
    return config


# =============================================================================
# 2. Build ADK agents from YAML config
# =============================================================================


def build_adk_agents(
    config: dict,
    *,
    use_temporal_model: bool = True,
) -> dict[str, Agent | Any]:
    """Create an ADK Agent (or RemoteA2aAgent / MCP-tooled Agent) per YAML node.

    Returns a dict of {node_id: agent_or_wrapper}.
    """
    default_model = config.get("default_model", "gemini-3.1-flash-lite")
    agents: dict[str, Any] = {}
    mcp_providers: list = []  # collected for worker registration

    for node in config["nodes"]:
        node_id = node["id"]
        node_type = node.get("type", "agent")

        if node_type == "agent":
            model_name = node.get("model", default_model)
            if use_temporal_model:
                model = TemporalModel(
                    model_name,
                    activity_config={
                        "start_to_close_timeout": timedelta(seconds=60),
                        "retry_policy": RetryPolicy(maximum_attempts=3),
                        "summary": f"LLM call for node '{node_id}'",
                    },
                )
            else:
                model = model_name  # plain string -- ADK resolves it directly

            agents[node_id] = Agent(
                name=node_id,
                model=model,
                instruction=node["instruction"],
                description=node.get("description", f"Agent node: {node_id}"),
            )

        elif node_type == "a2a":
            if not _HAS_A2A:
                logger.warning(
                    "Node '%s' is type=a2a but google-adk[a2a] is not installed -- skipping",
                    node_id,
                )
                continue
            agent_card_url = node["agent_card"]
            agents[node_id] = RemoteA2aAgent(
                name=node_id,
                agent_card=agent_card_url,
                description=node.get("description", f"A2A remote agent: {node_id}"),
            )

        elif node_type == "mcp":
            if not _HAS_MCP:
                logger.warning(
                    "Node '%s' is type=mcp but MCP deps are not installed -- skipping",
                    node_id,
                )
                continue

            # Build MCP toolset factory from YAML config
            mcp_command = node["command"]
            mcp_args = node.get("args", [])
            mcp_name = f"mcp-{node_id}"

            def _make_factory(cmd: str, args: list):
                """Create a toolset factory closure."""

                def factory(_):
                    return McpToolset(
                        connection_params=StdioConnectionParams(
                            server_params=StdioServerParameters(
                                command=cmd,
                                args=args,
                            ),
                        ),
                    )

                return factory

            factory = _make_factory(mcp_command, mcp_args)

            if use_temporal_model:
                # Register provider for the Temporal worker
                provider = TemporalMcpToolSetProvider(mcp_name, factory)
                mcp_providers.append(provider)
                tools = [TemporalMcpToolSet(mcp_name, not_in_workflow_toolset=factory)]
            else:
                tools = [factory(None)]

            # Wrap MCP tools in an ADK Agent that can use them
            model_name = node.get("model", default_model)
            model = (
                TemporalModel(
                    model_name,
                    activity_config={
                        "start_to_close_timeout": timedelta(seconds=60),
                        "summary": f"LLM call for MCP node '{node_id}'",
                    },
                )
                if use_temporal_model
                else model_name
            )
            agents[node_id] = Agent(
                name=node_id,
                model=model,
                instruction=node.get(
                    "instruction",
                    "Use the available tools to help the user.",
                ),
                tools=tools,
                description=node.get("description", f"MCP-tooled agent: {node_id}"),
            )

        else:
            raise ValueError(f"Unknown node type '{node_type}' for node '{node_id}'")

    return agents, mcp_providers


# =============================================================================
# 3. Build the routing table from YAML edges
# =============================================================================


def build_routing(config: dict) -> dict:
    """Parse YAML edges into a routing structure.

    Returns:
        {
            "start_nodes": ["classify"],
            "conditional": {
                "classify": [
                    {"to": "handle_billing",   "condition": "billing"},
                    {"to": "handle_technical",  "condition": "technical"},
                    {"to": "handle_general",    "condition": "general"},
                ],
            },
            "unconditional": {
                "handle_billing": ["next_node"],
            },
        }
    """
    start_nodes = []
    conditional: dict[str, list] = {}
    unconditional: dict[str, list] = {}

    for edge in config["edges"]:
        src = edge["from"]
        dst = edge["to"]
        cond = edge.get("condition")

        if src == "START":
            start_nodes.append(dst)
        elif cond:
            conditional.setdefault(src, []).append({"to": dst, "condition": cond})
        else:
            unconditional.setdefault(src, []).append(dst)

    return {
        "start_nodes": start_nodes,
        "conditional": conditional,
        "unconditional": unconditional,
    }


# =============================================================================
# 4. Run a single ADK agent and collect its text output
# =============================================================================


async def run_agent_node(
    agent: Agent,
    user_message: str,
    app_name: str = "dag_builder",
) -> str:
    """Run an ADK agent with InMemoryRunner and return its final text."""
    runner = InMemoryRunner(agent=agent, app_name=app_name)
    session = await runner.session_service.create_session(
        user_id="user", app_name=app_name
    )
    result = ""
    async with aclosing(
        runner.run_async(
            user_id="user",
            session_id=session.id,
            new_message=genai_types.Content(
                role="user",
                parts=[genai_types.Part.from_text(text=user_message)],
            ),
        )
    ) as events:
        async for event in events:
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        result = part.text
    return result.strip()


# =============================================================================
# 5. Temporal Activity that wraps run_agent_node
# =============================================================================


@dataclass
class AgentNodeInput:
    """Serialisable input for the run_agent_activity."""

    node_id: str
    user_message: str


@dataclass
class AgentNodeOutput:
    """Serialisable output from the run_agent_activity."""

    node_id: str
    response: str


# We store built agents at module level so the activity can reference them.
# In production you'd use dependency injection or a registry.
_AGENTS: dict[str, Agent] = {}


@activity.defn
async def run_agent_activity(inp: AgentNodeInput) -> AgentNodeOutput:
    """Temporal Activity: run an ADK agent node."""
    agent = _AGENTS.get(inp.node_id)
    if agent is None:
        return AgentNodeOutput(node_id=inp.node_id, response=f"ERROR: unknown node '{inp.node_id}'")
    try:
        text = await run_agent_node(agent, inp.user_message)
        return AgentNodeOutput(node_id=inp.node_id, response=text)
    except Exception as e:
        logger.exception("Activity %s failed", inp.node_id)
        return AgentNodeOutput(node_id=inp.node_id, response=f"ERROR: {e}")


# =============================================================================
# 6. Temporal Workflow -- the durable DAG executor
# =============================================================================


@workflow.defn(sandboxed=False)
class DagWorkflow:
    """Temporal Workflow that executes a YAML-defined DAG of ADK agents.

    Each agent node runs as a Temporal Activity (durable, retried, observable).
    Conditional routing uses the output of the classifier node.
    """

    @workflow.run
    async def run(self, user_message: str) -> str:
        """Execute the DAG: classify -> route -> handle."""
        routing = _ROUTING  # set at module level before worker starts

        # -- Step 1: run start nodes (usually just 'classify') --
        classify_output = ""
        for node_id in routing["start_nodes"]:
            result = await workflow.execute_activity(
                run_agent_activity,
                AgentNodeInput(node_id=node_id, user_message=user_message),
                start_to_close_timeout=timedelta(seconds=60),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            classify_output = result.response
            workflow.logger.info("Node '%s' output: %s", node_id, classify_output[:80])

        # -- Step 2: conditional routing from classifier --
        last_output = classify_output
        visited: set[str] = set(routing["start_nodes"])
        to_visit: list[str] = []

        # Check conditional edges from each visited node
        for node_id in list(visited):
            if node_id in routing["conditional"]:
                for route in routing["conditional"][node_id]:
                    if route["condition"].lower() in last_output.lower():
                        to_visit.append(route["to"])
                        break  # first match wins

            if node_id in routing.get("unconditional", {}):
                to_visit.extend(routing["unconditional"][node_id])

        # -- Step 3: execute matched handler nodes --
        final_output = last_output
        for node_id in to_visit:
            if node_id in visited:
                continue
            visited.add(node_id)

            # Pass the original user message (not classification) to the handler
            result = await workflow.execute_activity(
                run_agent_activity,
                AgentNodeInput(node_id=node_id, user_message=user_message),
                start_to_close_timeout=timedelta(seconds=60),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            final_output = result.response
            workflow.logger.info("Node '%s' output: %s", node_id, final_output[:80])

            # Continue routing from handler (for chained edges)
            if node_id in routing["conditional"]:
                for route in routing["conditional"][node_id]:
                    if route["condition"].lower() in final_output.lower():
                        to_visit.append(route["to"])
                        break
            if node_id in routing.get("unconditional", {}):
                to_visit.extend(routing["unconditional"][node_id])

        return final_output


# Module-level routing table (set before workflow starts)
_ROUTING: dict = {}


# =============================================================================
# 7. Runner: Temporal mode vs direct mode
# =============================================================================


async def run_with_temporal(config: dict) -> None:
    """Run the DAG as a Temporal workflow with durable execution."""
    global _AGENTS, _ROUTING

    agents, mcp_providers = build_adk_agents(config, use_temporal_model=True)
    _AGENTS = agents
    _ROUTING = build_routing(config)

    plugins = [GoogleAdkPlugin(toolset_providers=mcp_providers)] if mcp_providers else [GoogleAdkPlugin()]
    client = await TemporalClient.connect("localhost:7233", plugins=plugins)
    print("[temporal] Connected to Temporal server")

    task_queue = f"dag-{config['name']}"

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[DagWorkflow],
        activities=[run_agent_activity],
    ):
        print(f"[temporal] Worker running on queue '{task_queue}'")

        queries = [
            ("I was charged twice for my subscription", "wf-billing"),
            ("My internet keeps disconnecting every hour", "wf-technical"),
            ("What are your business hours?", "wf-general"),
        ]

        print()
        for query, wf_id in queries:
            result = await client.execute_workflow(
                DagWorkflow.run,
                query,
                id=wf_id,
                task_queue=task_queue,
            )
            print(f"  [{wf_id}] {query}")
            print(f"    -> {result[:120]}")
            print()

        # Show workflow statuses
        print("=== WORKFLOW STATUS ===")
        for _, wf_id in queries:
            desc = await client.get_workflow_handle(wf_id).describe()
            print(f"  {wf_id}: {desc.status.name}")

    print("\nDone. All workflows completed with durable execution.")


async def run_direct(config: dict) -> None:
    """Run the DAG directly (no Temporal) -- uses ADK agents but no durability."""
    agents, _ = build_adk_agents(config, use_temporal_model=False)
    routing = build_routing(config)

    queries = [
        "I was charged twice for my subscription",
        "My internet keeps disconnecting every hour",
        "What are your business hours?",
    ]

    print("[direct] Running without Temporal (no durability)\n")

    for query in queries:
        print(f"Query: {query}")

        # Step 1: classify
        classify_output = ""
        for node_id in routing["start_nodes"]:
            agent = agents.get(node_id)
            if agent is None:
                print(f"  WARN: node '{node_id}' not found, skipping")
                continue
            classify_output = await run_agent_node(agent, query)
            print(f"  [{node_id}] -> {classify_output[:80]}")

        # Step 2: route
        to_visit = []
        for node_id in routing["start_nodes"]:
            if node_id in routing["conditional"]:
                for route in routing["conditional"][node_id]:
                    if route["condition"].lower() in classify_output.lower():
                        to_visit.append(route["to"])
                        break

        # Step 3: handle
        for node_id in to_visit:
            agent = agents.get(node_id)
            if agent is None:
                print(f"  WARN: node '{node_id}' not found, skipping")
                continue
            handler_output = await run_agent_node(agent, query)
            print(f"  [{node_id}] -> {handler_output[:120]}")

        print()

    print("Done. Direct execution complete (no durability).")


# =============================================================================
# 8. Main
# =============================================================================


async def main() -> None:
    parser = argparse.ArgumentParser(description="ADK Temporal DAG builder")
    parser.add_argument(
        "--no-temporal",
        action="store_true",
        help="Run without Temporal (direct ADK execution)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to YAML config",
    )
    args = parser.parse_args()

    config = load_dag_config(args.config)
    print(f"=== DAG: {config['name']} v{config['version']} ===")
    print(f"    Nodes: {[n['id'] for n in config['nodes']]}")
    print(f"    Edges: {len(config['edges'])}")
    print()

    if args.no_temporal:
        await run_direct(config)
    else:
        try:
            await run_with_temporal(config)
        except Exception as e:
            if "Connect" in str(e) or "refused" in str(e) or "ConnectionRefusedError" in str(type(e).__name__):
                print(f"[warn] No Temporal server at localhost:7233 ({e})")
                print("[warn] Falling back to direct execution...\n")
                await run_direct(config)
            else:
                raise


if __name__ == "__main__":
    asyncio.run(main())
