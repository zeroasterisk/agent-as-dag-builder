"""Prototype 13: Ephemeral DAG Generation with Feedback Retry Loop.

Instead of pre-authored YAML DAGs, an LLM generates a YAML DAG from a natural
language task description. When execution produces poor results, a simulated
"domain expert" provides structured feedback, and the agent regenerates the DAG
incorporating that feedback (up to 3 retries).

Key components:
  Part 1 - DAG Generator: LLM generates valid YAML DAG from task description
  Part 2 - Domain Expert Simulator: LLM evaluates execution and gives feedback
  Part 3 - Feedback Retry Loop: generate -> execute -> judge -> retry if needed
  Part 4 - 10 Novel Tasks with synthetic test cases
  Part 5 - Ephemeral-to-Durable Promotion: detect similar DAG structures

Usage:
    python sandbox/13_ephemeral_dag.py                  # Run all 10 tasks
    python sandbox/13_ephemeral_dag.py --tasks 3        # Run first 3 tasks
    python sandbox/13_ephemeral_dag.py --max-retries 2  # Limit retries

Environment:
    GOOGLE_GENAI_USE_VERTEXAI=1
    GOOGLE_CLOUD_PROJECT=alanblount-demo
    GOOGLE_CLOUD_LOCATION=global
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import time
from collections import Counter
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
logger = logging.getLogger("ephemeral_dag")

SANDBOX_DIR = Path(__file__).parent
DAG_MODEL = "gemini-3.5-flash"        # Generation + execution
JUDGE_MODEL = "gemini-3.5-flash"      # Scoring / domain expert
RESULTS_FILE = SANDBOX_DIR / "scores_ephemeral.json"

# Rate-limit delay between LLM calls (seconds)
LLM_DELAY = 0.3

# Scoring threshold for domain expert approval
APPROVAL_THRESHOLD = 75


# =============================================================================
# Part 4: Novel Tasks and Synthetic Test Cases (defined first for reference)
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
# Part 1: DAG Generator
# =============================================================================


async def generate_dag(
    client: genai.Client,
    task_description: str,
    feedback: str | None = None,
) -> tuple[str, dict]:
    """Generate a YAML DAG config from a task description.

    If feedback is provided, incorporate it into the regenerated DAG.

    Returns:
        (raw_yaml_text, parsed_config_dict)
    Raises on invalid YAML.
    """
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

    # Parse and validate
    parsed = yaml.safe_load(raw_yaml)
    if parsed is None:
        raise yaml.YAMLError("Parsed YAML is empty/None")

    config = parsed.get("dag", parsed) if isinstance(parsed, dict) else parsed
    validate_dag_config(config)

    return raw_yaml, config


def validate_dag_config(config: dict) -> None:
    """Validate a DAG config has all required fields.

    Raises ValueError if invalid.
    """
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

    # Validate each node
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
        if "classify" in str(node.get("id", "")).lower() or \
           "triage" in str(node.get("id", "")).lower() or \
           "router" in str(node.get("id", "")).lower():
            has_classify = True
        node_ids.add(node["id"])

    # Validate edges
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

    # Also accept the START target as a classifier (common LLM naming)
    if not has_classify and start_target:
        has_classify = True  # first node in DAG acts as classifier

    if not has_classify:
        raise ValueError("No classifier node found")


# =============================================================================
# Part 2: Domain Expert Simulator
# =============================================================================


async def domain_expert_feedback(
    client: genai.Client,
    task_description: str,
    dag_yaml: str,
    execution_results: list[dict],
) -> dict:
    """Simulate a domain expert reviewing the DAG execution.

    Returns: {
        "approved": bool,
        "feedback": str,    # specific guidance for improvement
        "score": int,       # 0-100
    }
    """
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

GENERATED DAG (YAML):
{dag_yaml[:2000]}

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

    # Parse JSON from response
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
        }
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to parse expert feedback: %s", e)
        return {
            "approved": False,
            "score": 50,
            "feedback": f"Evaluation parsing failed: {e}. Retry with clearer DAG structure.",
        }


# =============================================================================
# DAG Executor (adapted from earlier prototypes)
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

    results = {
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

    # Extract category from classifier output
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

        # Continue routing from handler
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

        # Parse JSON
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
        return 50  # default middle score on failure


# =============================================================================
# Part 3: Feedback Retry Loop
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


@dataclass
class TaskResult:
    task_id: int
    task_name: str
    task_description: str
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
    def yaml_valid(self) -> bool:
        return all(a.yaml_valid for a in self.attempts)

    @property
    def final_node_count(self) -> int:
        return self.attempts[-1].node_count if self.attempts else 0

    @property
    def final_node_ids(self) -> list[str]:
        if not self.attempts:
            return []
        return [n["id"] for n in self.attempts[-1].config.get("nodes", [])]

    @property
    def final_categories(self) -> list[str]:
        if not self.attempts:
            return []
        return extract_categories_from_config(self.attempts[-1].config)


async def run_with_feedback(
    client: genai.Client,
    task: dict,
    max_retries: int = 3,
) -> TaskResult:
    """Generate DAG, execute, get expert feedback, retry if needed."""
    task_result = TaskResult(
        task_id=task["id"],
        task_name=task["name"],
        task_description=task["description"],
    )

    prev_feedback = None
    yaml_generation_failures = 0

    for attempt_num in range(max_retries + 1):
        print(f"  Attempt {attempt_num + 1}:", end=" ")

        # Step 1: Generate DAG
        yaml_valid = True
        try:
            raw_yaml, config = await generate_dag(
                client, task["description"], feedback=prev_feedback,
            )
            node_count = len(config.get("nodes", []))
            categories = extract_categories_from_config(config)
            print(f"Generated {node_count}-node DAG ({', '.join(categories)})", end="")
        except (yaml.YAMLError, ValueError) as e:
            yaml_valid = False
            yaml_generation_failures += 1
            print(f"YAML generation failed: {e}")
            prev_feedback = (
                f"The previous YAML was invalid: {e}. "
                "Generate clean, valid YAML with proper indentation. "
                "Ensure all nodes have id, type, model, and instruction fields."
            )
            # Record the failed attempt
            task_result.attempts.append(AttemptResult(
                attempt=attempt_num + 1,
                dag_yaml="(invalid)",
                config={},
                node_count=0,
                execution_results=[],
                expert_feedback={"approved": False, "score": 0, "feedback": str(e)},
                score=0,
                yaml_valid=False,
            ))
            continue

        # Step 2: Execute DAG against test cases
        execution_results = []
        total_score = 0
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
            total_score += quality_score

        avg_case_score = total_score / len(task["test_cases"]) if task["test_cases"] else 0

        # Step 3: Get domain expert feedback
        await asyncio.sleep(LLM_DELAY)
        expert = await domain_expert_feedback(
            client, task["description"], raw_yaml, execution_results,
        )

        print(f" -> Score: {expert['score']}", end="")

        # Record the attempt
        attempt = AttemptResult(
            attempt=attempt_num + 1,
            dag_yaml=raw_yaml,
            config=config,
            node_count=node_count,
            execution_results=execution_results,
            expert_feedback=expert,
            score=expert["score"],
            yaml_valid=yaml_valid,
        )
        task_result.attempts.append(attempt)

        if expert["approved"]:
            print(f' -> Expert: "Approved"')
            break
        else:
            feedback_short = expert["feedback"][:80]
            print(f' -> Expert: "{feedback_short}"')
            prev_feedback = expert["feedback"]

            if attempt_num < max_retries:
                print(f"    Retrying with feedback...")
            else:
                print(f"    Max retries reached.")

    return task_result


# =============================================================================
# Part 5: Ephemeral-to-Durable Promotion
# =============================================================================


def compute_dag_signature(config: dict) -> str:
    """Compute a structural signature of a DAG for similarity comparison.

    The signature captures:
    - Number of nodes
    - Node types (sorted)
    - Edge structure (from -> to patterns, ignoring conditions)
    """
    if not config or "nodes" not in config:
        return "empty"

    nodes = config.get("nodes", [])
    edges = config.get("edges", [])

    # Node count and presence of classifier
    n_nodes = len(nodes)
    has_classify = any("classify" in str(n.get("id", "")).lower() for n in nodes)

    # Edge pattern: count conditional vs unconditional
    n_conditional = sum(1 for e in edges if e.get("condition"))
    n_unconditional = sum(1 for e in edges if not e.get("condition") and e.get("from") != "START")

    # Number of categories (conditional edges from classifier)
    categories = extract_categories_from_config(config)
    n_categories = len(categories)

    return f"nodes={n_nodes},classify={has_classify},categories={n_categories},cond={n_conditional},uncond={n_unconditional}"


def find_durable_candidates(task_results: list[TaskResult]) -> list[dict]:
    """Find tasks that generated similar DAG structures (promotion candidates).

    Two tasks are considered similar if they have the same structural signature
    (same node count, same number of categories, same edge pattern).
    """
    # Group by signature
    sig_groups: dict[str, list[TaskResult]] = {}
    for tr in task_results:
        if not tr.attempts or not tr.attempts[-1].config:
            continue
        sig = compute_dag_signature(tr.attempts[-1].config)
        sig_groups.setdefault(sig, []).append(tr)

    candidates = []
    for sig, group in sig_groups.items():
        if len(group) >= 2:
            candidates.append({
                "signature": sig,
                "tasks": [{"id": tr.task_id, "name": tr.task_name, "score": tr.final_score} for tr in group],
                "template_node_count": group[0].final_node_count,
                "template_categories": len(group[0].final_categories),
            })

    return candidates


# =============================================================================
# Main Runner
# =============================================================================


async def run_all_tasks(
    num_tasks: int = 10,
    max_retries: int = 3,
) -> None:
    """Run all novel tasks through the ephemeral DAG generation pipeline."""
    client = genai.Client(
        vertexai=True,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT", "alanblount-demo"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
    )

    tasks_to_run = NOVEL_TASKS[:num_tasks]
    all_results: list[TaskResult] = []

    print("=" * 70)
    print("Ephemeral DAG Generation with Feedback Retry Loop")
    print("=" * 70)
    print(f"Tasks: {len(tasks_to_run)}")
    print(f"Test cases per task: 5")
    print(f"Max retries: {max_retries}")
    print(f"DAG model: {DAG_MODEL} (temperature=0.3)")
    print(f"Expert model: {JUDGE_MODEL} (temperature=0.0)")
    print(f"Approval threshold: {APPROVAL_THRESHOLD}/100")
    print(f"LLM delay: {LLM_DELAY}s")
    est_calls = len(tasks_to_run) * (max_retries + 1) * (1 + 5 * 2 + 1)  # gen + 5*(exec+judge) + expert
    print(f"Estimated max LLM calls: ~{est_calls}")
    print()

    t_start = time.time()

    for task in tasks_to_run:
        print(f"=== Task {task['id']}: {task['name']} ===")
        result = await run_with_feedback(client, task, max_retries=max_retries)
        all_results.append(result)
        print()

    elapsed = time.time() - t_start

    # =========================================================================
    # Per-task results
    # =========================================================================
    print("=" * 70)
    print("=== DETAILED RESULTS ===")
    print("=" * 70)

    total_yaml_valid = 0
    total_yaml_attempts = 0

    for tr in all_results:
        print(f"\n=== Task {tr.task_id}: {tr.task_name} ===")
        for a in tr.attempts:
            total_yaml_attempts += 1
            if a.yaml_valid:
                total_yaml_valid += 1
            expert_msg = a.expert_feedback.get("feedback", "")[:80]
            status = "Approved" if a.expert_feedback.get("approved") else expert_msg
            print(f"  Attempt {a.attempt}: Generated {a.node_count}-node DAG -> "
                  f"Score: {a.score} -> Expert: \"{status}\"")
        print(f"  Final: {tr.final_score}/100 ({tr.retries_needed} retries)")

    # =========================================================================
    # Summary statistics
    # =========================================================================
    initial_scores = [tr.initial_score for tr in all_results]
    final_scores = [tr.final_score for tr in all_results]
    retries = [tr.retries_needed for tr in all_results]

    avg_initial = sum(initial_scores) / len(initial_scores) if initial_scores else 0
    avg_final = sum(final_scores) / len(final_scores) if final_scores else 0
    avg_retries = sum(retries) / len(retries) if retries else 0
    yaml_valid_rate = (total_yaml_valid / total_yaml_attempts * 100) if total_yaml_attempts else 0

    tasks_approved = sum(1 for tr in all_results if tr.final_score >= APPROVAL_THRESHOLD)
    tasks_improved = sum(1 for tr in all_results if tr.final_score > tr.initial_score)

    # =========================================================================
    # Ephemeral-to-durable promotion
    # =========================================================================
    durable_candidates = find_durable_candidates(all_results)

    # =========================================================================
    # Print summary
    # =========================================================================
    print(f"\n{'='*70}")
    print("=== SUMMARY ===")
    print(f"{'='*70}")
    print(f"Tasks: {len(all_results)}")
    print(f"Avg initial score: {avg_initial:.1f}")
    print(f"Avg final score: {avg_final:.1f}")
    print(f"Avg improvement: {avg_final - avg_initial:+.1f}")
    print(f"Avg retries: {avg_retries:.1f}")
    print(f"Valid YAML rate: {yaml_valid_rate:.0f}% ({total_yaml_valid}/{total_yaml_attempts})")
    print(f"Tasks approved (>={APPROVAL_THRESHOLD}): {tasks_approved}/{len(all_results)}")
    print(f"Tasks improved by feedback: {tasks_improved}/{len(all_results)}")
    print(f"Elapsed time: {elapsed:.0f}s")

    if durable_candidates:
        print(f"\nEphemeral-to-durable candidates: {len(durable_candidates)} (similar structures found)")
        for cand in durable_candidates:
            task_names = [t["name"] for t in cand["tasks"]]
            avg_score = sum(t["score"] for t in cand["tasks"]) / len(cand["tasks"])
            print(f"  Pattern: {cand['signature']}")
            print(f"    Tasks: {', '.join(task_names)}")
            print(f"    Avg score: {avg_score:.0f}")
            print(f"    Nodes: {cand['template_node_count']}, Categories: {cand['template_categories']}")
    else:
        print(f"\nEphemeral-to-durable candidates: 0 (no similar structures found)")

    # =========================================================================
    # Score distribution
    # =========================================================================
    print(f"\nScore distribution (final):")
    buckets = {"90-100": 0, "75-89": 0, "60-74": 0, "40-59": 0, "0-39": 0}
    for s in final_scores:
        if s >= 90:
            buckets["90-100"] += 1
        elif s >= 75:
            buckets["75-89"] += 1
        elif s >= 60:
            buckets["60-74"] += 1
        elif s >= 40:
            buckets["40-59"] += 1
        else:
            buckets["0-39"] += 1
    for bucket, count in buckets.items():
        bar = "#" * count
        print(f"  {bucket}: {count} {bar}")

    # =========================================================================
    # Save results to JSON
    # =========================================================================
    results_data = {
        "summary": {
            "tasks": len(all_results),
            "avg_initial_score": round(avg_initial, 1),
            "avg_final_score": round(avg_final, 1),
            "avg_improvement": round(avg_final - avg_initial, 1),
            "avg_retries": round(avg_retries, 1),
            "yaml_valid_rate": round(yaml_valid_rate, 1),
            "tasks_approved": tasks_approved,
            "tasks_improved": tasks_improved,
            "elapsed_seconds": round(elapsed, 0),
        },
        "tasks": [],
        "durable_candidates": durable_candidates,
    }

    for tr in all_results:
        task_data = {
            "id": tr.task_id,
            "name": tr.task_name,
            "initial_score": tr.initial_score,
            "final_score": tr.final_score,
            "retries": tr.retries_needed,
            "final_node_count": tr.final_node_count,
            "final_categories": tr.final_categories,
            "attempts": [],
        }
        for a in tr.attempts:
            task_data["attempts"].append({
                "attempt": a.attempt,
                "node_count": a.node_count,
                "score": a.score,
                "yaml_valid": a.yaml_valid,
                "expert_feedback": a.expert_feedback.get("feedback", "")[:200],
            })
        results_data["tasks"].append(task_data)

    with open(RESULTS_FILE, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE.name}")

    return results_data


async def main():
    parser = argparse.ArgumentParser(
        description="Ephemeral DAG Generation with Feedback Retry Loop"
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

    await run_all_tasks(num_tasks=args.tasks, max_retries=args.max_retries)


if __name__ == "__main__":
    asyncio.run(main())
