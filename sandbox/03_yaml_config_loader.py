"""Prototype 03: Load YAML config → build ADK Workflow → run.

This is the core proof: can we define a workflow in YAML and have it
execute as an ADK graph with conditional routing?
"""
import os
import asyncio
import yaml
import sys

os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "1")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "alanblount-demo")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")

from google.adk.agents import LlmAgent
from google.adk.workflow import Workflow
from google.adk.workflow._base_node import START
from google.adk.workflow._function_node import FunctionNode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


def load_dag_from_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)["dag"]


def build_workflow(config: dict, feature_flags: dict = None) -> Workflow:
    """Build an ADK Workflow from a YAML config dict."""
    if feature_flags is None:
        feature_flags = {}

    # Build nodes
    nodes = {}
    for node_cfg in config["nodes"]:
        nid = node_cfg["id"]
        if node_cfg["type"] == "agent":
            nodes[nid] = LlmAgent(
                name=nid,
                model=node_cfg["model"],
                instruction=node_cfg["instruction"],
            )
        elif node_cfg["type"] == "function":
            raise NotImplementedError("Function nodes not yet supported")

    # Build edges
    edges = []
    for edge in config["edges"]:
        src = START if edge["from"] == "START" else nodes[edge["from"]]

        if "condition" in edge:
            # Conditional edge — collect all conditions from this source
            # ADK uses routing maps: (source, {"value": target, ...})
            # We need to group all conditional edges from the same source
            pass  # Handled below
        else:
            edges.append((src, nodes[edge["to"]]))

    # Group conditional edges by source
    conditional_sources = {}
    for edge in config["edges"]:
        if "condition" in edge:
            src_id = edge["from"]
            if src_id not in conditional_sources:
                conditional_sources[src_id] = {}
            target = nodes[edge["to"]]

            # Check feature flags — skip edges whose flags are off
            flag_name = edge.get("flag")
            if flag_name and not feature_flags.get(flag_name, True):
                continue

            conditional_sources[src_id][edge["condition"]] = target

    for src_id, routing_map in conditional_sources.items():
        src = nodes[src_id]
        edges.append((src, routing_map))

    return Workflow(
        name=config["name"],
        nodes=list(nodes.values()),
        edges=edges,
    )


async def run_workflow(workflow: Workflow, message: str) -> str:
    """Run a workflow and return the final response text."""
    session_service = InMemorySessionService()
    runner = Runner(
        agent=workflow,
        app_name="dag_builder",
        session_service=session_service,
    )

    session = await session_service.create_session(
        app_name="dag_builder", user_id="user1"
    )

    result = ""
    async for event in runner.run_async(
        user_id="user1",
        session_id=session.id,
        new_message=types.Content(
            parts=[types.Part(text=message)], role="user"
        ),
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    result = part.text  # Keep last response
        if event.is_final_response():
            break

    return result


async def main():
    config_path = os.path.join(os.path.dirname(__file__), "customer_support.yaml")
    config = load_dag_from_yaml(config_path)

    print(f"Loaded config: {config['name']} v{config['version']}")
    print(f"Nodes: {[n['id'] for n in config['nodes']]}")
    print(f"Edges: {len(config['edges'])}")
    print()

    # Build the workflow
    workflow = build_workflow(config)
    print(f"Workflow built: {workflow.name}")
    print()

    # Test with different types of queries
    test_queries = [
        "I was charged twice for my subscription last month",
        "My internet keeps disconnecting every hour",
        "What are your business hours?",
    ]

    for query in test_queries:
        print(f"Query: {query}")
        result = await run_workflow(workflow, query)
        print(f"Response: {result[:150]}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
