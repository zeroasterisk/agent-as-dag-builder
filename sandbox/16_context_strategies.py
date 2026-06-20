"""Prototype 16: Context Strategy Comparison for Graph Gardener Domain Expert.

Tests 4 approaches to providing DAG context to the domain expert, replacing the
broken 2000-char truncation discovered in Sprint C.

Strategies:
  truncate   - Current broken approach (control): dag_yaml[:2000]
  minimal    - Topology-only view: node IDs, types, edge connections
  localize   - Full YAML for only the traversed branch (classify + matched handler)
  agent_driven - Two-step: expert says what it needs, localizer provides it

Design:
  - All strategies share the SAME generated DAGs and execution results
  - Only the expert feedback step differs between strategies
  - 10 novel tasks x 5 test cases each
  - Up to 3 retries per task when expert score < 75

Usage:
    python sandbox/16_context_strategies.py --strategy minimal
    python sandbox/16_context_strategies.py --strategy all        # run all 4, compare
    python sandbox/16_context_strategies.py --strategy all --tasks 3  # quick test

Environment:
    GOOGLE_GENAI_USE_VERTEXAI=1
    GOOGLE_CLOUD_PROJECT=alanblount-demo
    GOOGLE_CLOUD_LOCATION=global
"""

from __future__ import annotations

import abc
import argparse
import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# -- Environment defaults (Vertex AI) ----------------------------------------
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "1")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "alanblount-demo")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")

from google import genai

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("context_strategies")

SANDBOX_DIR = Path(__file__).parent
DAG_MODEL = "gemini-3.5-flash"
JUDGE_MODEL = "gemini-3.5-flash"
RESULTS_FILE = SANDBOX_DIR / "scores_context_strategies.json"

LLM_DELAY = 0.3
APPROVAL_THRESHOLD = 75


# =============================================================================
# Novel Tasks (same as 13_ephemeral_dag.py -- kept in sync)
# =============================================================================

NOVEL_TASKS = [
    {
        "id": 1,
        "name": "Restaurant reservations",
        "description": (
            "Handle restaurant reservation requests -- classify by party size "
            "and dietary restrictions, route to appropriate booking agent"
        ),
        "test_cases": [
            {"query": "I'd like to book a table for 2 tonight, no dietary restrictions",
             "expected_behavior": "Should route to small party booking, confirm availability"},
            {"query": "We need a table for 12 for a birthday dinner, 3 guests are vegan",
             "expected_behavior": "Should route to large party booking, note vegan dietary needs"},
            {"query": "Party of 4, one person has a severe nut allergy",
             "expected_behavior": "Should flag allergy restriction, route to allergy-aware booking"},
            {"query": "Can I reserve a private dining room for 8 people, all gluten-free?",
             "expected_behavior": "Should handle large party + dietary restriction, mention private room"},
            {"query": "Table for 1 tomorrow lunch, I'm vegetarian",
             "expected_behavior": "Should route to small party booking with vegetarian note"},
        ],
    },
    {
        "id": 2,
        "name": "Insurance claims",
        "description": (
            "Process insurance claims -- triage by type (auto, home, health), "
            "assess severity, route to adjuster"
        ),
        "test_cases": [
            {"query": "I was in a fender bender yesterday, minor scratches on bumper",
             "expected_behavior": "Should classify as auto, low severity, route to auto adjuster"},
            {"query": "My basement flooded during the storm, significant water damage",
             "expected_behavior": "Should classify as home, high severity, route to home claims"},
            {"query": "I need to file a claim for an emergency room visit last week",
             "expected_behavior": "Should classify as health, route to health claims processor"},
            {"query": "Someone broke into my house and stole electronics worth $5000",
             "expected_behavior": "Should classify as home, high severity, mention police report"},
            {"query": "I had a car accident on the highway, my car is totaled",
             "expected_behavior": "Should classify as auto, high severity, expedite claim"},
        ],
    },
    {
        "id": 3,
        "name": "Student enrollment",
        "description": (
            "Manage student enrollment -- verify prerequisites, check capacity, "
            "handle waitlist, confirm enrollment"
        ),
        "test_cases": [
            {"query": "I want to enroll in CS201 Data Structures, I've completed CS101",
             "expected_behavior": "Should verify CS101 prerequisite is met, check capacity"},
            {"query": "Can I join the Advanced Biology seminar? I haven't taken Bio 101 yet",
             "expected_behavior": "Should flag missing prerequisite, suggest completing Bio 101 first"},
            {"query": "The Machine Learning class is full, can I get on the waitlist?",
             "expected_behavior": "Should confirm class is full, add to waitlist, explain process"},
            {"query": "I need to enroll in 5 courses this semester for my graduation requirement",
             "expected_behavior": "Should check each course's prerequisites and capacity"},
            {"query": "I want to drop Calculus II and add Statistics instead",
             "expected_behavior": "Should handle drop/add, check Statistics prerequisites and capacity"},
        ],
    },
    {
        "id": 4,
        "name": "Customer complaints",
        "description": (
            "Route customer complaints to the right team -- product defect, "
            "shipping issue, billing dispute, service quality"
        ),
        "test_cases": [
            {"query": "The blender I bought broke after 2 uses, the blade came loose",
             "expected_behavior": "Should classify as product defect, offer replacement/refund"},
            {"query": "My package was supposed to arrive 3 days ago and tracking shows nothing",
             "expected_behavior": "Should classify as shipping issue, initiate investigation"},
            {"query": "I was charged $50 more than the advertised price",
             "expected_behavior": "Should classify as billing dispute, review charge details"},
            {"query": "The technician who came to fix my AC was rude and unprofessional",
             "expected_behavior": "Should classify as service quality, escalate to management"},
            {"query": "My order arrived but the box was completely crushed and items damaged",
             "expected_behavior": "Should classify as shipping issue, initiate damage claim"},
        ],
    },
    {
        "id": 5,
        "name": "Job applications",
        "description": (
            "Handle job applications -- screen resume, match to open positions, "
            "schedule interviews"
        ),
        "test_cases": [
            {"query": "I'm a software engineer with 5 years of Python experience looking for a role",
             "expected_behavior": "Should screen qualifications, match to relevant engineering positions"},
            {"query": "I just graduated with a marketing degree, looking for entry-level positions",
             "expected_behavior": "Should identify entry-level marketing roles, note fresh graduate"},
            {"query": "I have 10 years in project management and PMP certification",
             "expected_behavior": "Should match to senior PM roles, highlight PMP qualification"},
            {"query": "I'd like to schedule an interview for the Data Analyst position I applied to",
             "expected_behavior": "Should route to interview scheduling, check application status"},
            {"query": "I'm a nurse practitioner interested in your healthcare division openings",
             "expected_behavior": "Should match to healthcare roles, verify licensure requirements"},
        ],
    },
    {
        "id": 6,
        "name": "Travel booking",
        "description": (
            "Handle travel booking requests -- classify by type (flights, hotels, "
            "car rentals, packages), handle preferences and budget constraints"
        ),
        "test_cases": [
            {"query": "I need a round-trip flight from NYC to London next month, economy class",
             "expected_behavior": "Should route to flight booking, note economy preference"},
            {"query": "Find me a hotel in Paris near the Eiffel Tower for 3 nights under $200/night",
             "expected_behavior": "Should route to hotel booking with location and budget constraints"},
            {"query": "I need a rental car in Miami for a week, prefer an SUV",
             "expected_behavior": "Should route to car rental with vehicle preference"},
            {"query": "Can you put together a vacation package to Hawaii for 2 adults and 1 child?",
             "expected_behavior": "Should route to package booking, note family composition"},
            {"query": "I want to change my flight from Tuesday to Thursday, same route",
             "expected_behavior": "Should route to flight modification, check availability"},
        ],
    },
    {
        "id": 7,
        "name": "Medical appointment triage",
        "description": (
            "Triage medical appointment requests -- classify urgency (emergency, "
            "urgent, routine), route by department (primary care, specialist, "
            "mental health), and schedule appropriately"
        ),
        "test_cases": [
            {"query": "I've had a persistent cough for 3 weeks and mild fever",
             "expected_behavior": "Should classify as urgent, route to primary care"},
            {"query": "I need my annual physical exam scheduled",
             "expected_behavior": "Should classify as routine, schedule with primary care"},
            {"query": "I'm having chest pains and difficulty breathing right now",
             "expected_behavior": "Should classify as emergency, direct to ER immediately"},
            {"query": "I'd like to see a dermatologist about a suspicious mole",
             "expected_behavior": "Should classify as urgent, route to dermatology specialist"},
            {"query": "I've been feeling very anxious and having panic attacks lately",
             "expected_behavior": "Should classify as urgent, route to mental health"},
        ],
    },
    {
        "id": 8,
        "name": "Real estate inquiries",
        "description": (
            "Handle real estate inquiries -- classify by intent (buying, selling, "
            "renting, property management), route to appropriate agent, handle "
            "budget and location preferences"
        ),
        "test_cases": [
            {"query": "I'm looking to buy a 3-bedroom house in Austin under $400K",
             "expected_behavior": "Should classify as buying, note budget and location preferences"},
            {"query": "I want to list my condo for sale, it's a 2BR/2BA downtown",
             "expected_behavior": "Should classify as selling, gather property details"},
            {"query": "I need a 1-bedroom apartment to rent near the university",
             "expected_behavior": "Should classify as renting, note location preference"},
            {"query": "My tenant hasn't paid rent in 2 months, what are my options?",
             "expected_behavior": "Should classify as property management, explain landlord options"},
            {"query": "What's the current market value of homes in the Oak Park neighborhood?",
             "expected_behavior": "Should classify as buying/selling, provide market analysis routing"},
        ],
    },
    {
        "id": 9,
        "name": "Event planning",
        "description": (
            "Handle event planning requests -- classify by event type (corporate, "
            "wedding, birthday, conference), handle venue selection, catering, "
            "and logistics coordination"
        ),
        "test_cases": [
            {"query": "We need to plan a company retreat for 50 people in October",
             "expected_behavior": "Should classify as corporate, handle venue and logistics"},
            {"query": "I'm planning my wedding for 150 guests, need venue and catering",
             "expected_behavior": "Should classify as wedding, coordinate venue and catering"},
            {"query": "I want to throw a surprise birthday party for 20 people next Saturday",
             "expected_behavior": "Should classify as birthday, note surprise element and timeline"},
            {"query": "We're hosting a tech conference for 500 attendees, need AV equipment",
             "expected_behavior": "Should classify as conference, handle large-scale logistics"},
            {"query": "I need a caterer for a corporate lunch meeting for 15 people",
             "expected_behavior": "Should classify as corporate, route to catering coordination"},
        ],
    },
    {
        "id": 10,
        "name": "Financial advisory",
        "description": (
            "Handle financial advisory requests -- classify by service type "
            "(investment, retirement planning, tax consultation, debt management), "
            "assess client profile, route to appropriate advisor"
        ),
        "test_cases": [
            {"query": "I have $50K to invest, moderate risk tolerance, 10-year horizon",
             "expected_behavior": "Should classify as investment, note risk profile and timeline"},
            {"query": "I'm 55 and want to plan for retirement in 10 years",
             "expected_behavior": "Should classify as retirement planning, note age and timeline"},
            {"query": "I need help with my small business tax filing for this year",
             "expected_behavior": "Should classify as tax consultation, note business context"},
            {"query": "I have $30K in credit card debt and need a payoff strategy",
             "expected_behavior": "Should classify as debt management, assess debt situation"},
            {"query": "Should I refinance my mortgage? Current rate is 6.5% on a 30-year",
             "expected_behavior": "Should assess refinancing options, route to mortgage advisor"},
        ],
    },
]


# =============================================================================
# Context Strategy ABC and Implementations
# =============================================================================

class ContextStrategy(abc.ABC):
    """Base class for context strategies fed to the domain expert."""

    name: str = "base"

    @abc.abstractmethod
    async def prepare_context(
        self,
        client: genai.Client,
        dag_yaml: str,
        task_description: str,
        test_cases: list[dict],
        execution_results: list[dict],
    ) -> tuple[str, int]:
        """Prepare context string for the domain expert.

        Returns:
            (context_string, extra_llm_calls)
        """
        ...


class TruncateStrategy(ContextStrategy):
    """Control: the broken 2000-char truncation from Sprint C."""

    name = "truncate"

    async def prepare_context(
        self,
        client: genai.Client,
        dag_yaml: str,
        task_description: str,
        test_cases: list[dict],
        execution_results: list[dict],
    ) -> tuple[str, int]:
        context = f"GENERATED DAG (YAML):\n{dag_yaml[:2000]}"
        return context, 0


class MinimalStrategy(ContextStrategy):
    """Option A: topology-only view -- node IDs, types, edge connections."""

    name = "minimal"

    async def prepare_context(
        self,
        client: genai.Client,
        dag_yaml: str,
        task_description: str,
        test_cases: list[dict],
        execution_results: list[dict],
    ) -> tuple[str, int]:
        config = yaml.safe_load(dag_yaml)
        # Handle both top-level "dag:" wrapper and flat configs
        if isinstance(config, dict) and "dag" in config:
            config = config["dag"]

        nodes = []
        for n in config.get("nodes", []):
            nodes.append({
                "id": n.get("id", "?"),
                "type": n.get("type", "?"),
            })

        edges = []
        for e in config.get("edges", []):
            edge_info: dict[str, Any] = {
                "from": e.get("from", "?"),
                "to": e.get("to", "?"),
            }
            if e.get("condition"):
                edge_info["condition"] = e["condition"]
            edges.append(edge_info)

        context = (
            f"DAG Structure (topology only):\n"
            f"Nodes: {json.dumps(nodes)}\n"
            f"Edges: {json.dumps(edges)}\n\n"
            f"Node count: {len(nodes)}, Edge count: {len(edges)}"
        )
        return context, 0


class LocalizeStrategy(ContextStrategy):
    """Option B: full YAML for only the traversed branch per test case."""

    name = "localize"

    async def prepare_context(
        self,
        client: genai.Client,
        dag_yaml: str,
        task_description: str,
        test_cases: list[dict],
        execution_results: list[dict],
    ) -> tuple[str, int]:
        config = yaml.safe_load(dag_yaml)
        if isinstance(config, dict) and "dag" in config:
            config = config["dag"]

        nodes_by_id = {n["id"]: n for n in config.get("nodes", [])}
        edges = config.get("edges", [])

        # Collect all categories that were actually detected in execution
        traversed_categories = set()
        for result in execution_results:
            cat = result.get("category", "")
            if cat and cat != "unknown":
                traversed_categories.add(cat.lower())

        # Also collect all node IDs visited during execution
        traversed_node_ids = set()
        for result in execution_results:
            for nid in result.get("nodes_visited", []):
                traversed_node_ids.add(nid)

        # Always include the classifier node
        traversed_node_ids.add("classify")

        # Find matching nodes: the classify node + any handler whose ID or
        # edge condition matches a traversed category
        relevant_nodes = []
        for n in config.get("nodes", []):
            nid = n["id"]
            if nid in traversed_node_ids:
                relevant_nodes.append(n)
            else:
                # Check if any category is in this node's ID
                for cat in traversed_categories:
                    if cat in nid:
                        relevant_nodes.append(n)
                        break

        # Find matching edges
        relevant_node_ids = {n["id"] for n in relevant_nodes}
        relevant_edges = []
        for e in edges:
            if e.get("from") == "START":
                relevant_edges.append(e)
            elif e.get("from") in relevant_node_ids or e.get("to") in relevant_node_ids:
                relevant_edges.append(e)

        # Also include any nodes not yet found but pointed to by traversed edges
        for e in relevant_edges:
            for key in ("from", "to"):
                nid = e.get(key, "")
                if nid != "START" and nid not in relevant_node_ids and nid in nodes_by_id:
                    relevant_nodes.append(nodes_by_id[nid])
                    relevant_node_ids.add(nid)

        localized = {
            "traversed_nodes": relevant_nodes,
            "traversed_edges": relevant_edges,
            "categories_seen": sorted(traversed_categories),
        }
        context = (
            f"TRAVERSED BRANCH (full detail for visited nodes only):\n"
            f"{yaml.dump(localized, default_flow_style=False, width=120)}\n"
            f"Note: Showing {len(relevant_nodes)} of {len(config.get('nodes', []))} total nodes "
            f"(only those traversed during test execution)."
        )
        return context, 0


class AgentDrivenStrategy(ContextStrategy):
    """Option C: two-step -- expert says what it needs, localizer provides."""

    name = "agent_driven"

    async def prepare_context(
        self,
        client: genai.Client,
        dag_yaml: str,
        task_description: str,
        test_cases: list[dict],
        execution_results: list[dict],
    ) -> tuple[str, int]:
        # Summarize execution results for the needs-assessment prompt
        results_summary = ""
        categories_seen = set()
        for i, result in enumerate(execution_results, 1):
            cat = result.get("category", "unknown")
            categories_seen.add(cat)
            results_summary += (
                f"  Case {i}: query=\"{result['query'][:60]}\" "
                f"category={cat} score={result.get('quality_score', '?')}/100\n"
            )

        # Step 1: Ask the expert what context it needs
        need_prompt = f"""You are reviewing an AI workflow DAG execution. The task was:
{task_description}

Execution summary:
{results_summary}
Categories detected: {', '.join(sorted(categories_seen))}

What specific parts of the DAG configuration do you need to see to evaluate this?
List the specific node IDs or categories you want to inspect.
Reply with ONLY a JSON list of strings, e.g. ["classify", "handle_billing", "handle_shipping"]
No explanation, just the JSON list."""

        await asyncio.sleep(LLM_DELAY)
        resp = client.models.generate_content(
            model=JUDGE_MODEL,
            contents=[{"role": "user", "parts": [{"text": need_prompt}]}],
            config={"temperature": 0.0},
        )
        needs_text = resp.candidates[0].content.parts[0].text.strip()
        extra_calls = 1

        # Parse the requested node IDs
        requested_ids = set()
        try:
            # Try direct JSON parse
            if "```" in needs_text:
                match = re.search(r"```(?:json)?\s*(.*?)```", needs_text, re.DOTALL)
                if match:
                    needs_text = match.group(1).strip()
            parsed = json.loads(needs_text)
            if isinstance(parsed, list):
                requested_ids = {str(x).lower() for x in parsed}
        except (json.JSONDecodeError, ValueError):
            # Fall back: extract any quoted strings
            requested_ids = set(re.findall(r'"([^"]+)"', needs_text))

        # Always include classify
        requested_ids.add("classify")

        # Step 2: Extract matching nodes from the full config
        config = yaml.safe_load(dag_yaml)
        if isinstance(config, dict) and "dag" in config:
            config = config["dag"]

        matched_nodes = []
        for n in config.get("nodes", []):
            nid = n.get("id", "").lower()
            if nid in requested_ids:
                matched_nodes.append(n)
            else:
                # Fuzzy match: check if any requested ID is a substring
                for req in requested_ids:
                    if req in nid or nid in req:
                        matched_nodes.append(n)
                        break

        matched_node_ids = {n["id"] for n in matched_nodes}
        matched_edges = []
        for e in config.get("edges", []):
            if (e.get("from") in matched_node_ids or
                e.get("to") in matched_node_ids or
                e.get("from") == "START"):
                matched_edges.append(e)

        localized = {
            "requested_nodes": [n["id"] for n in matched_nodes],
            "nodes": matched_nodes,
            "edges": matched_edges,
        }
        context = (
            f"AGENT-REQUESTED CONTEXT (expert asked to see: {sorted(requested_ids)}):\n"
            f"{yaml.dump(localized, default_flow_style=False, width=120)}\n"
            f"Showing {len(matched_nodes)} of {len(config.get('nodes', []))} total nodes."
        )
        return context, extra_calls


# Strategy registry
STRATEGIES: dict[str, type[ContextStrategy]] = {
    "truncate": TruncateStrategy,
    "minimal": MinimalStrategy,
    "localize": LocalizeStrategy,
    "agent_driven": AgentDrivenStrategy,
}


# =============================================================================
# DAG Generator (from 13_ephemeral_dag.py)
# =============================================================================

async def generate_dag(
    client: genai.Client,
    task_description: str,
    feedback: str | None = None,
) -> tuple[str, dict]:
    """Generate a YAML DAG config from a task description."""
    prompt = f"""Generate a YAML DAG configuration for this task:
{task_description}

{"Previous attempt feedback: " + feedback if feedback else ""}

The YAML must follow this exact schema:
- Top-level key: dag
- dag.name: short snake_case name
- dag.version: "1.0.0"
- dag.description: one-line description of the workflow
- dag.nodes: list of agent nodes, each with:
    - id: unique snake_case identifier
    - type: "agent"
    - model: "gemini-3.5-flash"
    - instruction: detailed instruction string for the agent
- dag.edges: list of routing edges, each with:
    - from: source node id (or "START")
    - to: target node id
    - condition: (optional) string condition for conditional routing
- The FIRST edge must be from: START, to: a classifier node
- The classifier node should output ONLY one of the category names
- Each category should route to a dedicated handler node via conditional edges

Important rules:
- The first node MUST have id: "classify" (exactly)
- The classifier instruction MUST tell the model to output ONLY the category name
- Handler nodes should be named "handle_<category>" (e.g. handle_billing)
- Each handler node should have detailed, domain-specific instructions
- Keep it practical: 3-6 nodes total (1 classifier + 2-5 handlers)
- All nodes must use model: "gemini-3.5-flash"

Example structure (do NOT copy -- generate for the task above):
dag:
  name: example
  version: "1.0.0"
  description: "Example"
  nodes:
    - id: classify
      type: agent
      model: gemini-3.5-flash
      instruction: |
        Classify into one of: catA, catB. Reply with ONLY the category name.
    - id: handle_catA
      type: agent
      model: gemini-3.5-flash
      instruction: |
        Handle catA requests...
  edges:
    - from: START
      to: classify
    - from: classify
      to: handle_catA
      condition: "catA"

Output ONLY valid YAML, no markdown fences, no explanation.
Start with "dag:" on the first line."""

    await asyncio.sleep(LLM_DELAY)
    resp = client.models.generate_content(
        model=DAG_MODEL,
        contents=[{"role": "user", "parts": [{"text": prompt}]}],
        config={"temperature": 0.3},
    )
    raw_yaml = resp.candidates[0].content.parts[0].text.strip()

    # Strip markdown code blocks if present
    if "```" in raw_yaml:
        lines = raw_yaml.split("\n")
        in_block = False
        block_lines = []
        for line in lines:
            if line.strip().startswith("```"):
                if in_block:
                    break
                in_block = True
                continue
            if in_block:
                block_lines.append(line)
        if block_lines:
            raw_yaml = "\n".join(block_lines)

    parsed = yaml.safe_load(raw_yaml)
    if parsed is None:
        raise yaml.YAMLError("Parsed YAML is empty/None")

    config = parsed.get("dag", parsed) if isinstance(parsed, dict) else parsed
    validate_dag_config(config)

    return raw_yaml, config


def validate_dag_config(config: dict) -> None:
    """Validate a DAG config has all required fields."""
    if not isinstance(config, dict):
        raise ValueError(f"Config is not a dict: {type(config)}")
    if "nodes" not in config:
        raise ValueError("Config missing 'nodes'")
    if "edges" not in config:
        raise ValueError("Config missing 'edges'")
    if not isinstance(config["nodes"], list) or len(config["nodes"]) == 0:
        raise ValueError("'nodes' must be a non-empty list")
    if not isinstance(config["edges"], list) or len(config["edges"]) == 0:
        raise ValueError("'edges' must be a non-empty list")

    node_ids = set()
    has_classify = False
    for i, node in enumerate(config["nodes"]):
        if not isinstance(node, dict):
            raise ValueError(f"Node {i} is not a dict")
        for required_field in ("id", "type"):
            if required_field not in node:
                raise ValueError(f"Node {i} missing '{required_field}'")
        if node["type"] == "agent":
            for agent_field in ("model", "instruction"):
                if agent_field not in node:
                    raise ValueError(f"Agent node '{node.get('id', i)}' missing '{agent_field}'")
        if ("classify" in str(node.get("id", "")).lower() or
            "triage" in str(node.get("id", "")).lower() or
            "router" in str(node.get("id", "")).lower()):
            has_classify = True
        node_ids.add(node["id"])

    has_start = False
    start_target = None
    for i, edge in enumerate(config["edges"]):
        if not isinstance(edge, dict):
            raise ValueError(f"Edge {i} is not a dict")
        if "from" not in edge or "to" not in edge:
            raise ValueError(f"Edge {i} missing 'from' or 'to'")
        if edge["from"] == "START":
            has_start = True
            start_target = edge["to"]

    if not has_start:
        raise ValueError("No edge from START found")
    if not has_classify and start_target:
        has_classify = True
    if not has_classify:
        raise ValueError("No classifier node found")


# =============================================================================
# DAG Executor (from 13_ephemeral_dag.py)
# =============================================================================

def build_routing(config: dict) -> dict:
    """Parse YAML edges into a routing structure."""
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


def extract_categories_from_config(config: dict) -> list[str]:
    """Extract the list of valid categories from a DAG config's edges."""
    categories = []
    routing = build_routing(config)
    for node_id in routing["start_nodes"]:
        if node_id in routing["conditional"]:
            for route in routing["conditional"][node_id]:
                categories.append(route["condition"].lower())
    return categories


async def execute_dag(
    client: genai.Client,
    config: dict,
    query: str,
) -> dict:
    """Execute a single query through a DAG and return detailed results."""
    nodes_by_id = {n["id"]: n for n in config["nodes"]}
    routing = build_routing(config)
    default_model = config.get("default_model", DAG_MODEL)
    valid_categories = extract_categories_from_config(config)

    results: dict[str, Any] = {
        "category": "unknown",
        "response": "",
        "nodes_visited": [],
        "timings": {},
    }

    async def call_node(node_id: str, user_input: str) -> str:
        node = nodes_by_id.get(node_id)
        if node is None or node.get("type", "agent") != "agent":
            return ""

        model = node.get("model", default_model)
        instruction = node.get("instruction", "Help the user.")
        prompt = f"{instruction}\n\nUser request: {user_input}"

        t0 = time.time()
        try:
            response = client.models.generate_content(
                model=model,
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
            )
            text = ""
            for p in response.candidates[0].content.parts:
                if hasattr(p, "text") and p.text:
                    text += p.text
            elapsed = (time.time() - t0) * 1000
            results["timings"][node_id] = elapsed
            results["nodes_visited"].append(node_id)
            return text.strip()
        except Exception as e:
            logger.error("Node %s failed: %s", node_id, e)
            results["timings"][node_id] = (time.time() - t0) * 1000
            results["nodes_visited"].append(node_id)
            return f"ERROR: {e}"

    # Step 1: run start nodes (classifier)
    classify_output = ""
    for node_id in routing["start_nodes"]:
        classify_output = await call_node(node_id, query)

    # Extract category
    classify_lower = classify_output.lower().strip()
    detected_category = "unknown"
    for cat in valid_categories:
        if cat in classify_lower:
            detected_category = cat
            break
    results["category"] = detected_category

    # Step 2: route based on classification
    to_visit = []
    for node_id in routing["start_nodes"]:
        if node_id in routing["conditional"]:
            for route in routing["conditional"][node_id]:
                if route["condition"].lower() in classify_lower:
                    to_visit.append(route["to"])
                    break

    # Step 3: run handler nodes
    final_output = classify_output
    visited = set(routing["start_nodes"])
    for node_id in to_visit:
        if node_id in visited:
            continue
        visited.add(node_id)
        handler_output = await call_node(node_id, query)
        if handler_output:
            final_output = handler_output

        # Continue routing
        if node_id in routing["conditional"]:
            for route in routing["conditional"][node_id]:
                if route["condition"].lower() in handler_output.lower():
                    to_visit.append(route["to"])
                    break
        if node_id in routing.get("unconditional", {}):
            to_visit.extend(routing["unconditional"][node_id])

    results["response"] = final_output
    return results


# =============================================================================
# LLM-as-Judge Scoring (per test case)
# =============================================================================

async def score_test_case(
    client: genai.Client,
    query: str,
    expected_behavior: str,
    dag_response: str,
    task_name: str,
) -> int:
    """Score a single test case response (0-100)."""
    prompt = f"""You are a strict scoring judge for a {task_name} system.
Evaluate how well the response matches the expected behavior.

INPUT:
- Query: "{query}"
- Expected behavior: {expected_behavior}

RESPONSE:
{dag_response[:1000]}

Score this response from 0-100:
- 90-100: Fully meets expected behavior with good detail
- 70-89: Mostly meets expected behavior, minor gaps
- 50-69: Partially meets expected behavior
- 25-49: Barely addresses expected behavior
- 0-24: Fails to meet expected behavior

Output ONLY a JSON object: {{"score": <int>, "reasoning": "<one sentence>"}}"""

    await asyncio.sleep(LLM_DELAY)
    try:
        resp = client.models.generate_content(
            model=JUDGE_MODEL,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            config={"temperature": 0.0},
        )
        judge_text = resp.candidates[0].content.parts[0].text.strip()

        if "```" in judge_text:
            match = re.search(r"```(?:json)?\s*(.*?)```", judge_text, re.DOTALL)
            if match:
                judge_text = match.group(1).strip()
        json_match = re.search(r'\{[^{}]*\}', judge_text)
        if json_match:
            judge_text = json_match.group(0)

        scores = json.loads(judge_text)
        return max(0, min(100, int(scores.get("score", 50))))
    except Exception as e:
        logger.warning("Scoring failed: %s", e)
        return 50


# =============================================================================
# Domain Expert with Pluggable Context Strategy
# =============================================================================

async def domain_expert_feedback(
    client: genai.Client,
    strategy: ContextStrategy,
    task_description: str,
    dag_yaml: str,
    test_cases: list[dict],
    execution_results: list[dict],
) -> tuple[dict, int]:
    """Get domain expert feedback using the specified context strategy.

    Returns: (feedback_dict, extra_llm_calls)
    """
    # Prepare context via strategy
    context, extra_calls = await strategy.prepare_context(
        client, dag_yaml, task_description, test_cases, execution_results,
    )

    # Format execution results for review
    results_text = ""
    for i, result in enumerate(execution_results, 1):
        results_text += f"""
Test case {i}: "{result['query']}"
  Expected: {result['expected_behavior']}
  Category detected: {result.get('category', 'unknown')}
  Response summary: {result.get('response', '')[:300]}
  Quality score: {result.get('quality_score', 'N/A')}/100
"""

    prompt = f"""You are a domain expert reviewing an AI workflow system.

TASK DESCRIPTION:
{task_description}

{context}

EXECUTION RESULTS:
{results_text}

Evaluate the overall quality of this DAG and its execution results.

Consider:
1. Does the DAG structure match the task requirements?
2. Are test cases being classified/routed correctly?
3. Are the handler responses appropriate and helpful?
4. Are there missing categories or edge cases?
5. Are the instructions specific enough for the domain?

Provide your evaluation as EXACTLY one JSON object:
{{
    "approved": <true if score >= {APPROVAL_THRESHOLD}, false otherwise>,
    "score": <integer 0-100>,
    "feedback": "<2-3 sentences of specific, actionable improvement guidance>"
}}

SCORING RUBRIC:
- 90-100: Excellent - all cases handled well, good structure
- 75-89: Good - most cases handled, minor gaps
- 60-74: Fair - several issues need fixing
- 40-59: Poor - major structural or routing problems
- 0-39: Failing - fundamental issues

Output ONLY the JSON, no markdown, no explanation."""

    await asyncio.sleep(LLM_DELAY)
    resp = client.models.generate_content(
        model=JUDGE_MODEL,
        contents=[{"role": "user", "parts": [{"text": prompt}]}],
        config={"temperature": 0.0},
    )
    judge_text = resp.candidates[0].content.parts[0].text.strip()

    # Parse JSON
    if "```" in judge_text:
        match = re.search(r"```(?:json)?\s*(.*?)```", judge_text, re.DOTALL)
        if match:
            judge_text = match.group(1).strip()
    json_match = re.search(r'\{[^{}]*\}', judge_text)
    if json_match:
        judge_text = json_match.group(0)

    try:
        result = json.loads(judge_text)
        score = max(0, min(100, int(result.get("score", 50))))
        return {
            "approved": score >= APPROVAL_THRESHOLD,
            "score": score,
            "feedback": str(result.get("feedback", "No specific feedback")),
        }, extra_calls
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to parse expert feedback: %s", e)
        return {
            "approved": False,
            "score": 50,
            "feedback": f"Evaluation parsing failed: {e}. Retry with clearer DAG structure.",
        }, extra_calls


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class AttemptResult:
    attempt: int
    dag_yaml: str
    config: dict
    node_count: int
    execution_results: list[dict]
    expert_feedback: dict
    score: int
    yaml_valid: bool
    extra_llm_calls: int = 0


@dataclass
class TaskResult:
    task_id: int
    task_name: str
    task_description: str
    strategy_name: str
    attempts: list[AttemptResult] = field(default_factory=list)

    @property
    def initial_score(self) -> int:
        return self.attempts[0].score if self.attempts else 0

    @property
    def final_score(self) -> int:
        return self.attempts[-1].score if self.attempts else 0

    @property
    def retries_needed(self) -> int:
        return len(self.attempts) - 1

    @property
    def total_llm_calls(self) -> int:
        """Estimate total LLM calls for this task across all attempts."""
        total = 0
        for a in self.attempts:
            if a.yaml_valid:
                # 1 (generate) + 5*(1 exec + 1 score) + 1 (expert) = 12
                total += 12
                total += a.extra_llm_calls
            else:
                total += 1  # just the failed generation
        return total


@dataclass
class StrategyResults:
    strategy_name: str
    tasks: list[TaskResult] = field(default_factory=list)

    @property
    def avg_initial(self) -> float:
        scores = [t.initial_score for t in self.tasks]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def avg_final(self) -> float:
        scores = [t.final_score for t in self.tasks]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def improvement(self) -> float:
        return self.avg_final - self.avg_initial

    @property
    def total_llm_calls(self) -> int:
        return sum(t.total_llm_calls for t in self.tasks)

    @property
    def expert_accuracy(self) -> float:
        """Fraction of expert judgments where the expert's feedback was
        actionable (i.e., did NOT mention truncation/incomplete YAML).

        We detect 'corruption' by checking if the expert's feedback mentions
        truncation artifacts: 'incomplete', 'truncated', 'missing node',
        'missing handler' when the node actually exists in the full config.
        """
        total = 0
        accurate = 0
        for task in self.tasks:
            for attempt in task.attempts:
                if not attempt.yaml_valid:
                    continue
                total += 1
                feedback_lower = attempt.expert_feedback.get("feedback", "").lower()
                # Check for truncation-artifact keywords
                has_artifact = any(
                    kw in feedback_lower
                    for kw in [
                        "truncated", "incomplete yaml", "yaml is incomplete",
                        "missing node", "yaml truncated",
                    ]
                )
                if not has_artifact:
                    accurate += 1
        return accurate / total if total else 0.0

    @property
    def tasks_approved(self) -> int:
        return sum(1 for t in self.tasks if t.final_score >= APPROVAL_THRESHOLD)


# =============================================================================
# Shared DAG Cache: generate once, evaluate with each strategy
# =============================================================================

@dataclass
class CachedDagRun:
    """Stores the strategy-independent work for one task attempt."""
    dag_yaml: str
    config: dict
    node_count: int
    yaml_valid: bool
    execution_results: list[dict]  # includes quality_score per case
    feedback_text: str | None  # feedback from previous attempt (for regeneration)


async def generate_and_execute_task(
    client: genai.Client,
    task: dict,
    max_retries: int,
    prev_feedbacks: dict[str, str | None] | None = None,
) -> list[CachedDagRun]:
    """Generate a DAG and execute it against test cases.

    This is strategy-independent -- produces the same DAG and execution
    results that all strategies will then evaluate.

    We run a single attempt here (no feedback loop -- the feedback loop
    is strategy-dependent). Returns a single CachedDagRun.
    """
    # Use first (None) feedback for initial generation
    feedback = None

    runs: list[CachedDagRun] = []

    for attempt_num in range(max_retries + 1):
        yaml_valid = True
        try:
            raw_yaml, config = await generate_dag(
                client, task["description"], feedback=feedback,
            )
            node_count = len(config.get("nodes", []))
            categories = extract_categories_from_config(config)
            print(f"    Attempt {attempt_num + 1}: Generated {node_count}-node DAG "
                  f"({', '.join(categories)})", end="")
        except (yaml.YAMLError, ValueError) as e:
            yaml_valid = False
            print(f"    Attempt {attempt_num + 1}: YAML generation failed: {e}")
            runs.append(CachedDagRun(
                dag_yaml="(invalid)",
                config={},
                node_count=0,
                yaml_valid=False,
                execution_results=[],
                feedback_text=str(e),
            ))
            feedback = (
                f"The previous YAML was invalid: {e}. "
                "Generate clean, valid YAML with proper indentation. "
                "Ensure all nodes have id, type, model, and instruction fields."
            )
            continue

        # Execute DAG against test cases
        execution_results = []
        for tc in task["test_cases"]:
            await asyncio.sleep(LLM_DELAY)
            dag_result = await execute_dag(client, config, tc["query"])

            await asyncio.sleep(LLM_DELAY)
            quality_score = await score_test_case(
                client, tc["query"], tc["expected_behavior"],
                dag_result["response"], task["name"],
            )

            execution_results.append({
                "query": tc["query"],
                "expected_behavior": tc["expected_behavior"],
                "category": dag_result["category"],
                "response": dag_result["response"],
                "quality_score": quality_score,
                "nodes_visited": dag_result["nodes_visited"],
            })

        avg_score = sum(r["quality_score"] for r in execution_results) / len(execution_results)
        print(f" -> avg case score: {avg_score:.0f}")

        runs.append(CachedDagRun(
            dag_yaml=raw_yaml,
            config=config,
            node_count=node_count,
            yaml_valid=True,
            execution_results=execution_results,
            feedback_text=None,
        ))

        # For initial generation, only produce one valid run
        # The feedback loop will be driven per-strategy
        break

    return runs


async def run_strategy_on_cached(
    client: genai.Client,
    strategy: ContextStrategy,
    task: dict,
    initial_run: CachedDagRun,
    max_retries: int,
) -> TaskResult:
    """Run the feedback loop for one strategy on a task.

    The initial DAG and execution are already cached. Only the expert
    feedback (and possible regeneration with that feedback) differs.
    """
    task_result = TaskResult(
        task_id=task["id"],
        task_name=task["name"],
        task_description=task["description"],
        strategy_name=strategy.name,
    )

    # Use the cached initial run
    current_run = initial_run

    for attempt_num in range(max_retries + 1):
        if not current_run.yaml_valid:
            task_result.attempts.append(AttemptResult(
                attempt=attempt_num + 1,
                dag_yaml=current_run.dag_yaml,
                config={},
                node_count=0,
                execution_results=[],
                expert_feedback={"approved": False, "score": 0,
                                 "feedback": current_run.feedback_text or "YAML invalid"},
                score=0,
                yaml_valid=False,
            ))
            break

        # Get expert feedback using this strategy
        expert, extra_calls = await domain_expert_feedback(
            client, strategy, task["description"],
            current_run.dag_yaml, task["test_cases"],
            current_run.execution_results,
        )

        attempt = AttemptResult(
            attempt=attempt_num + 1,
            dag_yaml=current_run.dag_yaml,
            config=current_run.config,
            node_count=current_run.node_count,
            execution_results=current_run.execution_results,
            expert_feedback=expert,
            score=expert["score"],
            yaml_valid=True,
            extra_llm_calls=extra_calls,
        )
        task_result.attempts.append(attempt)

        status = "Approved" if expert["approved"] else expert["feedback"][:60]
        print(f"      [{strategy.name:13s}] Attempt {attempt_num + 1}: "
              f"Score {expert['score']} -> {status}")

        if expert["approved"]:
            break

        # Not approved -- regenerate with feedback (only if we have retries left)
        if attempt_num < max_retries:
            feedback = expert["feedback"]
            try:
                raw_yaml, config = await generate_dag(
                    client, task["description"], feedback=feedback,
                )
                node_count = len(config.get("nodes", []))

                # Re-execute
                execution_results = []
                for tc in task["test_cases"]:
                    await asyncio.sleep(LLM_DELAY)
                    dag_result = await execute_dag(client, config, tc["query"])
                    await asyncio.sleep(LLM_DELAY)
                    quality_score = await score_test_case(
                        client, tc["query"], tc["expected_behavior"],
                        dag_result["response"], task["name"],
                    )
                    execution_results.append({
                        "query": tc["query"],
                        "expected_behavior": tc["expected_behavior"],
                        "category": dag_result["category"],
                        "response": dag_result["response"],
                        "quality_score": quality_score,
                        "nodes_visited": dag_result["nodes_visited"],
                    })

                current_run = CachedDagRun(
                    dag_yaml=raw_yaml,
                    config=config,
                    node_count=node_count,
                    yaml_valid=True,
                    execution_results=execution_results,
                    feedback_text=feedback,
                )
            except (yaml.YAMLError, ValueError) as e:
                current_run = CachedDagRun(
                    dag_yaml="(invalid)",
                    config={},
                    node_count=0,
                    yaml_valid=False,
                    execution_results=[],
                    feedback_text=str(e),
                )

    return task_result


# =============================================================================
# Main Runner
# =============================================================================

async def run_experiment(
    strategy_names: list[str],
    num_tasks: int = 10,
    max_retries: int = 3,
) -> dict:
    """Run the context strategy comparison experiment."""
    client = genai.Client(
        vertexai=True,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT", "alanblount-demo"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
    )

    tasks_to_run = NOVEL_TASKS[:num_tasks]
    strategies = [STRATEGIES[name]() for name in strategy_names]

    print("=" * 78)
    print("CONTEXT STRATEGY COMPARISON EXPERIMENT")
    print("=" * 78)
    print(f"Tasks: {len(tasks_to_run)}")
    print(f"Test cases per task: 5")
    print(f"Max retries: {max_retries}")
    print(f"Strategies: {', '.join(s.name for s in strategies)}")
    print(f"DAG model: {DAG_MODEL} (temp=0.3)")
    print(f"Expert/Judge model: {JUDGE_MODEL} (temp=0.0)")
    print(f"Approval threshold: {APPROVAL_THRESHOLD}/100")
    print(f"LLM delay: {LLM_DELAY}s")
    print()

    t_start = time.time()
    all_strategy_results: dict[str, StrategyResults] = {}

    for s in strategies:
        all_strategy_results[s.name] = StrategyResults(strategy_name=s.name)

    for task in tasks_to_run:
        print(f"\n{'='*60}")
        print(f"  Task {task['id']}: {task['name']}")
        print(f"{'='*60}")

        # Step 1: Generate initial DAG (shared across all strategies)
        print(f"  [shared] Generating initial DAG and executing test cases...")
        initial_runs = await generate_and_execute_task(
            client, task, max_retries=0,
        )

        if not initial_runs:
            print(f"  [shared] Failed to generate any valid DAG. Skipping task.")
            for s in strategies:
                all_strategy_results[s.name].tasks.append(TaskResult(
                    task_id=task["id"],
                    task_name=task["name"],
                    task_description=task["description"],
                    strategy_name=s.name,
                ))
            continue

        initial_run = initial_runs[0]

        # Step 2: Evaluate with each strategy (using the same initial DAG)
        for s in strategies:
            print(f"  [{s.name}] Running feedback loop...")
            task_result = await run_strategy_on_cached(
                client, s, task, initial_run, max_retries=max_retries,
            )
            all_strategy_results[s.name].tasks.append(task_result)

    elapsed = time.time() - t_start

    # =========================================================================
    # Results Comparison
    # =========================================================================
    print(f"\n\n{'='*78}")
    print(f"=== CONTEXT STRATEGY COMPARISON ({len(tasks_to_run)} tasks, 5 cases each) ===")
    print(f"{'='*78}")

    header = (
        f"{'Strategy':<17s} | {'Avg Initial':>11s} | {'Avg Final':>9s} | "
        f"{'Improvement':>11s} | {'Expert Accuracy':>15s} | {'LLM Calls':>9s} | "
        f"{'Approved':>8s}"
    )
    print(header)
    print("-" * len(header))

    best_strategy = None
    best_improvement = -999.0

    for sname, sr in all_strategy_results.items():
        imp = sr.improvement
        acc = sr.expert_accuracy
        print(
            f"{sname:<17s} | {sr.avg_initial:11.1f} | {sr.avg_final:9.1f} | "
            f"{imp:+11.1f} | {acc:14.0%} | {sr.total_llm_calls:9d} | "
            f"{sr.tasks_approved:>3d}/{len(sr.tasks)}"
        )
        if imp > best_improvement:
            best_improvement = imp
            best_strategy = sname

    print()
    if best_strategy:
        sr = all_strategy_results[best_strategy]
        print(f"Winner: {best_strategy} -- best improvement ({sr.improvement:+.1f}) "
              f"with {sr.total_llm_calls} LLM calls")
    print(f"Elapsed time: {elapsed:.0f}s")

    # =========================================================================
    # Per-task breakdown
    # =========================================================================
    print(f"\n{'='*78}")
    print("=== PER-TASK FINAL SCORES ===")
    print(f"{'='*78}")

    header2 = f"{'Task':<25s}"
    for sname in all_strategy_results:
        header2 += f" | {sname:>13s}"
    print(header2)
    print("-" * len(header2))

    for i, task in enumerate(tasks_to_run):
        row = f"{task['name']:<25s}"
        for sname, sr in all_strategy_results.items():
            if i < len(sr.tasks) and sr.tasks[i].attempts:
                final = sr.tasks[i].final_score
                row += f" | {final:>13d}"
            else:
                row += f" | {'N/A':>13s}"
        print(row)

    # =========================================================================
    # Expert accuracy detail
    # =========================================================================
    print(f"\n{'='*78}")
    print("=== EXPERT ACCURACY DETAIL ===")
    print(f"{'='*78}")
    print("(Checks whether expert feedback avoids truncation-artifact complaints)")
    print()

    for sname, sr in all_strategy_results.items():
        total_feedbacks = 0
        artifact_count = 0
        for task in sr.tasks:
            for attempt in task.attempts:
                if not attempt.yaml_valid:
                    continue
                total_feedbacks += 1
                fb = attempt.expert_feedback.get("feedback", "").lower()
                if any(kw in fb for kw in [
                    "truncated", "incomplete yaml", "yaml is incomplete",
                    "missing node", "yaml truncated",
                ]):
                    artifact_count += 1
        clean = total_feedbacks - artifact_count
        print(f"  {sname:<17s}: {clean}/{total_feedbacks} clean feedbacks "
              f"({artifact_count} had truncation artifacts)")

    # =========================================================================
    # Save results
    # =========================================================================
    results_data: dict[str, Any] = {
        "experiment": "context_strategy_comparison",
        "model": DAG_MODEL,
        "judge_model": JUDGE_MODEL,
        "tasks": len(tasks_to_run),
        "max_retries": max_retries,
        "elapsed_seconds": round(elapsed),
        "winner": best_strategy,
        "strategies": {},
    }

    for sname, sr in all_strategy_results.items():
        strat_data: dict[str, Any] = {
            "avg_initial": round(sr.avg_initial, 1),
            "avg_final": round(sr.avg_final, 1),
            "improvement": round(sr.improvement, 1),
            "expert_accuracy": round(sr.expert_accuracy, 3),
            "total_llm_calls": sr.total_llm_calls,
            "tasks_approved": sr.tasks_approved,
            "per_task": [],
        }
        for tr in sr.tasks:
            task_data = {
                "task_id": tr.task_id,
                "task_name": tr.task_name,
                "initial_score": tr.initial_score,
                "final_score": tr.final_score,
                "retries": tr.retries_needed,
                "llm_calls": tr.total_llm_calls,
                "attempts": [
                    {
                        "attempt": a.attempt,
                        "score": a.score,
                        "node_count": a.node_count,
                        "yaml_valid": a.yaml_valid,
                        "feedback": a.expert_feedback.get("feedback", "")[:200],
                        "extra_llm_calls": a.extra_llm_calls,
                    }
                    for a in tr.attempts
                ],
            }
            strat_data["per_task"].append(task_data)
        results_data["strategies"][sname] = strat_data

    with open(RESULTS_FILE, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE.name}")

    return results_data


# =============================================================================
# CLI
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Context Strategy Comparison for Graph Gardener Domain Expert"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="all",
        choices=["all", "truncate", "minimal", "localize", "agent_driven"],
        help="Which strategy to run (default: all)",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=10,
        help="Number of tasks to run (default: 10)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per task (default: 3)",
    )
    args = parser.parse_args()

    if args.strategy == "all":
        strategy_names = list(STRATEGIES.keys())
    else:
        strategy_names = [args.strategy]

    await run_experiment(
        strategy_names=strategy_names,
        num_tasks=args.tasks,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    asyncio.run(main())
