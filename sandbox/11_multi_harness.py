"""Prototype 11: Multi-Harness Benchmark + Learning Loop with Conservative Optimization.

Runs multiple domain harnesses (customer support, IT helpdesk, sales inquiry)
through a benchmark, scores responses with LLM-as-judge, proposes per-harness
improvements, and uses conservative optimization to prevent score degradation.

Conservative optimization strategy:
  A. Best-config tracking -- always optimize from best-ever, not latest
  B. Stability zone -- don't modify harnesses within 1.5 pts of their best
  C. One-at-a-time -- only modify the single weakest harness per iteration
  D. Rollback -- if aggregate drops >2 pts below best, rollback all to best

Integrated improvements (from Sprints B/C/D):
  - Template injection: APPEND issue-specific templates to handlers, not rewrite
  - Domain knowledge: inject realistic knowledge bases (server names, DNS IPs)
  - Agent-driven context: expert asks what it needs to see for improvement
  - No verifier nodes: same-model verification is a coin flip (Sprint B)

Usage:
    python sandbox/11_multi_harness.py                    # Run 3 iterations
    python sandbox/11_multi_harness.py --iterations 5     # Run 5 iterations
    python sandbox/11_multi_harness.py --reset            # Reset all versioned YAMLs

Environment:
    GOOGLE_GENAI_USE_VERTEXAI=1
    GOOGLE_CLOUD_PROJECT=alanblount-demo
    GOOGLE_CLOUD_LOCATION=global
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import os
import re
import shutil
import sys
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
logger = logging.getLogger("multi_harness")

SANDBOX_DIR = Path(__file__).parent
JUDGE_MODEL = "gemini-3.5-flash"
SCORES_FILE = SANDBOX_DIR / "scores.json"

# Rate-limit delay between LLM calls (seconds)
LLM_DELAY = 0.3

# Conservative optimization thresholds
STABILITY_THRESHOLD = 1.5   # Don't modify a harness within this many pts of its best
ROLLBACK_THRESHOLD = 2.0    # Rollback all configs if aggregate drops more than this below best


# =============================================================================
# 1. Test Cases for All Harnesses
# =============================================================================

CUSTOMER_SUPPORT_CASES = [
    # --- Billing (8 cases) ---
    {"id": 1, "query": "I was charged twice for my subscription", "expected_category": "billing",
     "quality_criteria": "Should acknowledge the double charge, mention investigation or refund process"},
    {"id": 2, "query": "Can I get a refund for last month?", "expected_category": "billing",
     "quality_criteria": "Should explain the refund policy or process for requesting a refund"},
    {"id": 3, "query": "Why is my bill higher than usual this month?", "expected_category": "billing",
     "quality_criteria": "Should suggest common reasons (plan change, overage) and offer to review the account"},
    {"id": 4, "query": "I need to update my credit card on file", "expected_category": "billing",
     "quality_criteria": "Should provide steps to update payment method or direct to account settings"},
    {"id": 5, "query": "How do I cancel my subscription?", "expected_category": "billing",
     "quality_criteria": "Should explain cancellation process; may mention retention offers"},
    {"id": 6, "query": "I see an unauthorized charge from your company", "expected_category": "billing",
     "quality_criteria": "Should take the concern seriously, suggest investigation, mention fraud protection"},
    {"id": 7, "query": "When is my next billing date?", "expected_category": "billing",
     "quality_criteria": "Should explain how to find billing date or offer to look it up"},
    {"id": 8, "query": "Do you offer any discounts for annual plans?", "expected_category": "billing",
     "quality_criteria": "Should mention available discount options or direct to pricing page"},
    # --- Technical (8 cases) ---
    {"id": 9, "query": "My internet keeps dropping every 30 minutes", "expected_category": "technical",
     "quality_criteria": "Should ask about router/modem or suggest troubleshooting steps like restarting equipment"},
    {"id": 10, "query": "The app crashes when I try to open settings", "expected_category": "technical",
     "quality_criteria": "Should suggest clearing cache, reinstalling, or checking for updates"},
    {"id": 11, "query": "I can't log into my account, it says invalid password", "expected_category": "technical",
     "quality_criteria": "Should suggest password reset process and check for account lock"},
    {"id": 12, "query": "My download speeds are extremely slow, only getting 2 Mbps", "expected_category": "technical",
     "quality_criteria": "Should ask about connection type, suggest speed test, check for interference"},
    {"id": 13, "query": "The website keeps showing a 404 error on the dashboard page", "expected_category": "technical",
     "quality_criteria": "Should suggest clearing browser cache, trying different browser, or report known issue"},
    {"id": 14, "query": "My smart TV can't connect to your streaming service anymore", "expected_category": "technical",
     "quality_criteria": "Should suggest checking TV firmware, reinstalling app, verifying network connection"},
    {"id": 15, "query": "Email notifications are not working, I'm not receiving any alerts", "expected_category": "technical",
     "quality_criteria": "Should suggest checking notification settings, spam folder, and email verification"},
    {"id": 16, "query": "Two-factor authentication is not sending the verification code", "expected_category": "technical",
     "quality_criteria": "Should suggest checking phone number, trying alternative methods, checking SMS blockers"},
    # --- General (7 cases) ---
    {"id": 17, "query": "What are your business hours?", "expected_category": "general",
     "quality_criteria": "Should provide specific hours or direct to a page with hours information"},
    {"id": 18, "query": "How do I contact customer support by phone?", "expected_category": "general",
     "quality_criteria": "Should provide a phone number or explain how to find contact information"},
    {"id": 19, "query": "Do you have a referral program?", "expected_category": "general",
     "quality_criteria": "Should explain whether a referral program exists and how to participate"},
    {"id": 20, "query": "I'd like to provide feedback about your service", "expected_category": "general",
     "quality_criteria": "Should welcome feedback and explain how to submit it (survey, email, form)"},
    {"id": 21, "query": "What services do you offer for small businesses?", "expected_category": "general",
     "quality_criteria": "Should describe business offerings or direct to business solutions page"},
    {"id": 22, "query": "Is there a mobile app available?", "expected_category": "general",
     "quality_criteria": "Should confirm app availability and mention platforms (iOS/Android) or download links"},
    {"id": 23, "query": "I want to upgrade my current plan", "expected_category": "general",
     "quality_criteria": "Should explain upgrade options and how to change plans"},
]

IT_HELPDESK_CASES = [
    # --- Password Reset (4 cases) ---
    {"id": 1, "query": "I forgot my password and can't log in", "expected_category": "password-reset",
     "quality_criteria": "Should guide through password reset process, mention self-service portal"},
    {"id": 2, "query": "My account got locked after too many login attempts", "expected_category": "password-reset",
     "quality_criteria": "Should explain account unlock process and how to prevent future lockouts"},
    {"id": 3, "query": "I need to change my password, it's been 90 days", "expected_category": "password-reset",
     "quality_criteria": "Should provide steps for password change and mention password policy requirements"},
    {"id": 4, "query": "My SSO login isn't working with the company portal", "expected_category": "password-reset",
     "quality_criteria": "Should troubleshoot SSO issues, suggest clearing cookies or checking IdP status"},
    # --- Software Install (5 cases) ---
    {"id": 5, "query": "I need Adobe Photoshop installed on my workstation", "expected_category": "software-install",
     "quality_criteria": "Should explain software request process, mention approval workflow and licensing"},
    {"id": 6, "query": "How do I install the company VPN client?", "expected_category": "software-install",
     "quality_criteria": "Should provide VPN client download location and installation steps"},
    {"id": 7, "query": "Microsoft Office keeps asking me to activate my license", "expected_category": "software-install",
     "quality_criteria": "Should troubleshoot license activation, suggest signing in with corporate account"},
    {"id": 8, "query": "I need Python and VS Code set up for development", "expected_category": "software-install",
     "quality_criteria": "Should explain developer tool provisioning process or self-service install steps"},
    {"id": 9, "query": "Slack is not updating to the latest version", "expected_category": "software-install",
     "quality_criteria": "Should suggest manual update steps, check for admin restrictions on updates"},
    # --- Hardware (4 cases) ---
    {"id": 10, "query": "My laptop screen is flickering constantly", "expected_category": "hardware",
     "quality_criteria": "Should suggest display driver update, external monitor test, and hardware repair if needed"},
    {"id": 11, "query": "The printer on the 3rd floor isn't working", "expected_category": "hardware",
     "quality_criteria": "Should suggest basic troubleshooting (power cycle, paper jam) and offer to dispatch support"},
    {"id": 12, "query": "My laptop won't turn on at all", "expected_category": "hardware",
     "quality_criteria": "Should suggest checking power adapter, battery reset, and offer replacement if needed"},
    {"id": 13, "query": "My keyboard is typing the wrong characters", "expected_category": "hardware",
     "quality_criteria": "Should suggest checking language/layout settings, trying external keyboard, driver update"},
    # --- Network (5 cases) ---
    {"id": 14, "query": "I can't connect to the VPN from home", "expected_category": "network",
     "quality_criteria": "Should troubleshoot VPN connection, check credentials, firewall, and ISP blocking"},
    {"id": 15, "query": "The office Wi-Fi keeps disconnecting", "expected_category": "network",
     "quality_criteria": "Should suggest forgetting and reconnecting, checking signal strength, trying other band"},
    {"id": 16, "query": "I can't access the internal wiki from my desk", "expected_category": "network",
     "quality_criteria": "Should check if on corporate network, DNS resolution, and proxy settings"},
    {"id": 17, "query": "Video calls keep freezing and dropping", "expected_category": "network",
     "quality_criteria": "Should suggest bandwidth check, wired connection, closing other apps, QoS settings"},
    {"id": 18, "query": "I'm getting a DNS resolution error for company websites", "expected_category": "network",
     "quality_criteria": "Should suggest flushing DNS cache, checking DNS settings, trying alternate DNS"},
]

SALES_INQUIRY_CASES = [
    # --- Pricing (4 cases) ---
    {"id": 1, "query": "How much does the enterprise plan cost?", "expected_category": "pricing",
     "quality_criteria": "Should mention enterprise pricing tiers or offer to schedule a pricing call"},
    {"id": 2, "query": "Do you offer a free trial?", "expected_category": "pricing",
     "quality_criteria": "Should explain free trial availability, duration, and what's included"},
    {"id": 3, "query": "What's included in the basic vs premium plan?", "expected_category": "pricing",
     "quality_criteria": "Should compare plan features and pricing differences clearly"},
    {"id": 4, "query": "Can we get volume pricing for 500 users?", "expected_category": "pricing",
     "quality_criteria": "Should mention volume discount availability and suggest contacting sales"},
    # --- Features (5 cases) ---
    {"id": 5, "query": "Do you support single sign-on (SSO)?", "expected_category": "features",
     "quality_criteria": "Should confirm SSO support, mention supported protocols (SAML, OIDC)"},
    {"id": 6, "query": "Can your platform integrate with Salesforce?", "expected_category": "features",
     "quality_criteria": "Should describe integration capabilities, mention API or native integrations"},
    {"id": 7, "query": "Is there an API we can use for automation?", "expected_category": "features",
     "quality_criteria": "Should confirm API availability, mention documentation, rate limits, and SDKs"},
    {"id": 8, "query": "What security certifications do you have?", "expected_category": "features",
     "quality_criteria": "Should mention relevant certifications (SOC 2, ISO 27001, etc.)"},
    {"id": 9, "query": "Does the platform support multi-language content?", "expected_category": "features",
     "quality_criteria": "Should explain language support, localization features, and available languages"},
    # --- Demo Request (4 cases) ---
    {"id": 10, "query": "Can I get a live demo of the platform?", "expected_category": "demo-request",
     "quality_criteria": "Should offer to schedule a demo, ask about use case and team size"},
    {"id": 11, "query": "I'd like to see how the analytics dashboard works", "expected_category": "demo-request",
     "quality_criteria": "Should offer demo of analytics features, mention available demo formats"},
    {"id": 12, "query": "Our team wants to evaluate your product for Q3", "expected_category": "demo-request",
     "quality_criteria": "Should acknowledge timeline, offer pilot program or trial, schedule walkthrough"},
    {"id": 13, "query": "Can you walk us through the onboarding process?", "expected_category": "demo-request",
     "quality_criteria": "Should describe onboarding, offer a walkthrough session, mention support resources"},
    # --- Competitor Comparison (4 cases) ---
    {"id": 14, "query": "How do you compare to Competitor X?", "expected_category": "competitor-comparison",
     "quality_criteria": "Should highlight key differentiators professionally, offer comparison materials"},
    {"id": 15, "query": "Why should we switch from our current vendor?", "expected_category": "competitor-comparison",
     "quality_criteria": "Should focus on unique value props, migration support, and ROI benefits"},
    {"id": 16, "query": "What makes you better than the open-source alternatives?", "expected_category": "competitor-comparison",
     "quality_criteria": "Should compare managed service benefits vs open-source (support, reliability, features)"},
    {"id": 17, "query": "We're evaluating three vendors including you, what stands out?", "expected_category": "competitor-comparison",
     "quality_criteria": "Should summarize top differentiators concisely, offer detailed comparison"},
]


# =============================================================================
# 2. Harness Configuration
# =============================================================================

HARNESSES = [
    {
        "name": "customer_support",
        "yaml": "customer_support_adk.yaml",
        "cases": CUSTOMER_SUPPORT_CASES,
    },
    {
        "name": "it_helpdesk",
        "yaml": "it_helpdesk.yaml",
        "cases": IT_HELPDESK_CASES,
    },
    {
        "name": "sales_inquiry",
        "yaml": "sales_inquiry.yaml",
        "cases": SALES_INQUIRY_CASES,
    },
]


# =============================================================================
# 3. DAG Executor (direct mode, no Temporal)
# =============================================================================


def load_dag_config(path: Path | str) -> dict:
    """Load and validate a DAG YAML config."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    config = raw["dag"]
    assert "nodes" in config, "YAML must define 'nodes'"
    assert "edges" in config, "YAML must define 'edges'"
    return config


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


async def run_dag_query(
    client: genai.Client,
    config: dict,
    query: str,
) -> dict:
    """Execute a single query through the DAG and return detailed results."""
    nodes_by_id = {n["id"]: n for n in config["nodes"]}
    routing = build_routing(config)
    default_model = config.get("default_model", "gemini-3.5-flash")
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
# 4. LLM-as-Judge Scoring
# =============================================================================


async def _single_judge_call(
    client: genai.Client,
    judge_prompt: str,
) -> dict:
    """Make a single LLM judge call and return parsed scores dict.

    Returns {"quality_score": int, "helpfulness_score": int, "reasoning": str}
    or raises on failure.
    """
    resp = client.models.generate_content(
        model=JUDGE_MODEL,
        contents=[{"role": "user", "parts": [{"text": judge_prompt}]}],
        config={"temperature": 0.0},
    )
    judge_text = resp.candidates[0].content.parts[0].text.strip()

    # Parse JSON from response (handle markdown code blocks)
    if "```" in judge_text:
        match = re.search(r"```(?:json)?\s*(.*?)```", judge_text, re.DOTALL)
        if match:
            judge_text = match.group(1).strip()

    # Also handle bare JSON with leading/trailing junk
    json_match = re.search(r'\{[^{}]*\}', judge_text)
    if json_match:
        judge_text = json_match.group(0)

    scores = json.loads(judge_text)
    return scores


async def score_response(
    client: genai.Client,
    query: str,
    expected_category: str,
    actual_category: str,
    response: str,
    quality_criteria: str,
    harness_name: str,
) -> dict:
    """Score a response using LLM-as-judge with variance reduction.

    Scoring breakdown (100 points total):
    - Category correctness: 50 points (deterministic, no variance)
    - Quality: 30 points (LLM-scored, temperature=0, averaged over 2 calls)
    - Helpfulness: 20 points (LLM-scored, temperature=0, averaged over 2 calls)

    Returns dict with: total_score, category_score, quality_score, helpfulness_score, reasoning
    """
    # Category correctness: deterministic (50 points) -- high weight = low variance
    category_score = 50 if actual_category == expected_category else 0

    # Quality + helpfulness scored by LLM (50 points total)
    # Using structured prompt, temperature=0, and 2x averaging
    judge_prompt = f"""You are a strict scoring judge for a {harness_name.replace('_', ' ')} system.
Evaluate the response below and output EXACTLY one JSON object.

INPUT:
- Customer query: "{query}"
- Expected category: {expected_category}
- Actual category: {actual_category}
- Quality criteria: {quality_criteria}

RESPONSE:
{response[:2000]}

SCORING RUBRIC:
quality_score (integer 0-30):
  0-7   = completely misses criteria
  8-15  = partially addresses criteria
  16-22 = mostly meets criteria
  23-30 = fully meets criteria with good detail

helpfulness_score (integer 0-20):
  0-5   = unhelpful or confusing
  6-10  = somewhat helpful
  11-15 = helpful and clear
  16-20 = excellent, concise, clear, actionable

OUTPUT FORMAT (no markdown, no explanation, ONLY this JSON):
{{"quality_score": <int>, "helpfulness_score": <int>, "reasoning": "<one sentence>"}}"""

    quality_scores = []
    helpfulness_scores = []
    reasonings = []

    # Run judge 2x and average to reduce variance by ~sqrt(2)
    for attempt in range(2):
        try:
            await asyncio.sleep(LLM_DELAY)
            scores = await _single_judge_call(client, judge_prompt)
            q = max(0, min(30, int(scores.get("quality_score", 15))))
            h = max(0, min(20, int(scores.get("helpfulness_score", 10))))
            quality_scores.append(q)
            helpfulness_scores.append(h)
            reasonings.append(scores.get("reasoning", ""))
        except Exception as e:
            logger.warning("Judge scoring attempt %d failed: %s", attempt + 1, e)
            # Use middle-of-road defaults on failure
            quality_scores.append(15)
            helpfulness_scores.append(10)
            reasonings.append(f"Scoring error: {e}")

    # Average the scores (rounds down)
    quality_score = round(sum(quality_scores) / len(quality_scores))
    helpfulness_score = round(sum(helpfulness_scores) / len(helpfulness_scores))
    reasoning = reasonings[0] if reasonings else ""

    total = category_score + quality_score + helpfulness_score
    return {
        "total_score": total,
        "category_score": category_score,
        "quality_score": quality_score,
        "helpfulness_score": helpfulness_score,
        "reasoning": reasoning,
    }


# =============================================================================
# 5. Data Structures
# =============================================================================


@dataclass
class InteractionRecord:
    query: str
    case_id: int
    expected_category: str
    actual_category: str
    response: str
    score: dict
    quality_criteria: str


@dataclass
class HarnessResult:
    harness_name: str
    iteration: int
    config_path: str
    records: list[InteractionRecord] = field(default_factory=list)

    @property
    def aggregate_score(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.score["total_score"] for r in self.records) / len(self.records)

    @property
    def category_accuracy(self) -> float:
        if not self.records:
            return 0.0
        correct = sum(1 for r in self.records if r.actual_category == r.expected_category)
        return correct / len(self.records)

    @property
    def avg_quality(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.score["quality_score"] for r in self.records) / len(self.records)

    @property
    def avg_helpfulness(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.score["helpfulness_score"] for r in self.records) / len(self.records)

    def category_breakdown(self) -> dict:
        breakdown = {}
        for r in self.records:
            cat = r.expected_category
            if cat not in breakdown:
                breakdown[cat] = {"scores": [], "correct": 0, "total": 0}
            breakdown[cat]["scores"].append(r.score["total_score"])
            breakdown[cat]["total"] += 1
            if r.actual_category == r.expected_category:
                breakdown[cat]["correct"] += 1
        for cat in breakdown:
            scores = breakdown[cat]["scores"]
            breakdown[cat]["avg_score"] = sum(scores) / len(scores)
            breakdown[cat]["accuracy"] = breakdown[cat]["correct"] / breakdown[cat]["total"]
        return breakdown

    def weakest_cases(self, n: int = 5) -> list[InteractionRecord]:
        return sorted(self.records, key=lambda r: r.score["total_score"])[:n]


@dataclass
class IterationResult:
    iteration: int
    harness_results: dict[str, HarnessResult] = field(default_factory=dict)

    @property
    def aggregate_score(self) -> float:
        all_scores = []
        for hr in self.harness_results.values():
            for r in hr.records:
                all_scores.append(r.score["total_score"])
        if not all_scores:
            return 0.0
        return sum(all_scores) / len(all_scores)

    @property
    def total_cases(self) -> int:
        return sum(len(hr.records) for hr in self.harness_results.values())


# =============================================================================
# 6. Learning Loop: Per-Harness Analysis & Proposals
# =============================================================================


def _build_agent_driven_context(
    harness_result: HarnessResult,
    current_config: dict,
    harness_name: str,
) -> str:
    """Build agent-driven context for the improvement prompt.

    Instead of truncating the full YAML or dumping everything, we extract
    only the relevant handler nodes for the weakest category, plus the
    classifier. This is the 'localize' pattern from Sprint C that gives
    the optimizer the right context without noise.
    """
    breakdown = harness_result.category_breakdown()
    weak_cases = harness_result.weakest_cases(5)

    # Find the weakest category
    weakest_cat = min(breakdown.items(), key=lambda x: x[1]["avg_score"])
    weakest_cat_name = weakest_cat[0]

    # Extract relevant nodes: classifier + weakest handler
    nodes_by_id = {n["id"]: n for n in current_config.get("nodes", [])}
    relevant_nodes = []
    for n in current_config.get("nodes", []):
        nid = n["id"]
        if "classify" in nid:
            relevant_nodes.append(n)
        elif weakest_cat_name.replace("-", "_") in nid or weakest_cat_name.replace("-", "") in nid:
            relevant_nodes.append(n)

    # Also include cases from the weakest category for full context
    weak_category_cases = [r for r in harness_result.records
                          if r.expected_category == weakest_cat_name]

    context = f"WEAKEST CATEGORY: {weakest_cat_name}\n"
    context += f"Category score: {weakest_cat[1]['avg_score']:.1f}/100\n"
    context += f"Category accuracy: {weakest_cat[1]['accuracy']:.0%}\n\n"

    context += "RELEVANT NODES (classifier + weakest handler):\n"
    context += yaml.dump({"nodes": relevant_nodes}, default_flow_style=False, width=120)
    context += "\n"

    context += f"WEAK CASES IN {weakest_cat_name}:\n"
    for r in weak_category_cases:
        context += f"  Case {r.case_id}: \"{r.query}\"\n"
        context += f"    Expected: {r.expected_category}, Got: {r.actual_category}\n"
        context += f"    Score: {r.score['total_score']}, Quality: {r.score['quality_score']}, "
        context += f"Helpfulness: {r.score['helpfulness_score']}\n"
        context += f"    Judge: {r.score.get('reasoning', '')[:150]}\n"
        context += f"    Criteria: {r.quality_criteria}\n"
        context += f"    Response preview: {r.response[:200]}\n\n"

    return context


async def analyze_and_propose_for_harness(
    client: genai.Client,
    harness_result: HarnessResult,
    current_config: dict,
    harness_name: str,
) -> list[str]:
    """Analyze benchmark results for one harness and propose template additions.

    Uses agent-driven context (Sprint C) to focus on the weakest category,
    and proposes issue-specific templates + domain knowledge (Sprint D)
    rather than full instruction rewrites.
    """
    breakdown = harness_result.category_breakdown()
    weak_cases = harness_result.weakest_cases(5)

    valid_categories = extract_categories_from_config(current_config)

    # Use agent-driven context: focused on the weakest category
    focused_context = _build_agent_driven_context(
        harness_result, current_config, harness_name,
    )

    analysis = f"""Benchmark Results for {harness_name} (Iteration {harness_result.iteration}):
- Aggregate Score: {harness_result.aggregate_score:.1f}/100
- Category Accuracy: {harness_result.category_accuracy:.0%}
- Quality Score: {harness_result.avg_quality:.1f}/30
- Helpfulness Score: {harness_result.avg_helpfulness:.1f}/20

Valid Categories: {', '.join(valid_categories)}

Category Breakdown:
"""
    for cat, info in breakdown.items():
        analysis += f"  {cat}: avg={info['avg_score']:.1f}, accuracy={info['accuracy']:.0%} ({info['total']} cases)\n"

    analysis += f"\n{focused_context}"

    prompt = f"""You are an AI workflow optimizer specializing in ADDITIVE instruction improvement.
Analyze these {harness_name.replace('_', ' ')} DAG benchmark results.

{analysis}

IMPORTANT RULES:
1. You must NOT rewrite handler instructions from scratch
2. You must ONLY propose ADDITIONS to append to the existing handler instructions
3. Focus on the weakest category shown above

Propose 2-3 specific TEMPLATE ADDITIONS to append to the weakest handler. Each addition should be one of:

TYPE A - Issue-specific template:
  A step-by-step template for a specific sub-issue that weak cases reveal.
  Example: "For SCREEN FLICKERING: 1. Check refresh rate... 2. Update driver..."

TYPE B - Domain knowledge injection:
  Concrete, realistic details like server names, URLs, IP addresses, specific
  commands, tool names, or policy details that make responses more actionable.
  Example: "KNOWLEDGE BASE: VPN server: vpn.company.com, DNS: 10.0.0.53..."

TYPE C - Quality criteria hint:
  A brief instruction that directly addresses what the judge is looking for
  in the weak cases (based on the quality criteria and judge reasoning).
  Example: "ALWAYS mention signal strength diagnosis and band switching for Wi-Fi issues."

For each proposal, specify:
- TYPE (A, B, or C)
- The exact text to APPEND to the handler instruction
- Which handler node it applies to
- Why (referencing specific weak cases)

Reply with a numbered list. Be specific and concrete."""

    await asyncio.sleep(LLM_DELAY)
    resp = client.models.generate_content(
        model=JUDGE_MODEL,
        contents=[{"role": "user", "parts": [{"text": prompt}]}],
    )
    proposals_text = resp.candidates[0].content.parts[0].text.strip()

    # Split into individual proposals
    proposals = []
    current = ""
    for line in proposals_text.split("\n"):
        if re.match(r"^\d+[\.\)]\s", line.strip()) and current:
            proposals.append(current.strip())
            current = line
        else:
            current += "\n" + line
    if current.strip():
        proposals.append(current.strip())

    return proposals


async def generate_improved_dag(
    client: genai.Client,
    current_config: dict,
    proposals: list[str],
    iteration: int,
    harness_name: str,
) -> dict:
    """Generate an improved DAG config by APPENDING templates to handler instructions.

    Key difference from the original: we do NOT ask the LLM to rewrite the full
    config. Instead, we ask it to extract the text to append, then we do the
    appending ourselves programmatically. This preserves the base instruction
    and only adds specificity (Sprint D finding: additive > rewrite).

    Fallback: if extraction fails, we fall back to programmatic append of the
    raw proposals text onto the weakest handler.
    """
    valid_categories = extract_categories_from_config(current_config)

    # Identify the weakest handler from the proposals (they reference it)
    proposals_text = "\n".join(proposals)

    # Step 1: Ask the LLM to extract structured additions from the proposals
    extract_prompt = f"""You are a precise text extractor. Given the improvement proposals below,
extract the text that should be APPENDED to handler node instructions.

PROPOSALS:
{proposals_text}

CURRENT HANDLER NODE IDs: {', '.join(n['id'] for n in current_config['nodes'] if n['id'] != 'classify')}

For each proposal, output a JSON object with:
- "node_id": which handler node to append to (must be one of the node IDs above)
- "append_text": the exact text to append to that node's instruction

Output a JSON array of these objects. No markdown, no explanation.
Example: [{{"node_id": "handle_network", "append_text": "\\nFor VPN issues:\\n1. Check credentials..."}}]"""

    await asyncio.sleep(LLM_DELAY)
    try:
        resp = client.models.generate_content(
            model=JUDGE_MODEL,
            contents=[{"role": "user", "parts": [{"text": extract_prompt}]}],
            config={"temperature": 0.0},
        )
        extract_text = resp.candidates[0].content.parts[0].text.strip()

        # Parse JSON
        if "```" in extract_text:
            match = re.search(r"```(?:json)?\s*(.*?)```", extract_text, re.DOTALL)
            if match:
                extract_text = match.group(1).strip()

        # Try to find JSON array
        json_match = re.search(r'\[.*\]', extract_text, re.DOTALL)
        if json_match:
            extract_text = json_match.group(0)

        additions = json.loads(extract_text)
        if not isinstance(additions, list):
            additions = [additions]

    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Failed to extract structured additions: %s. Using fallback.", e)
        # Fallback: find the weakest handler and append raw proposal text
        handler_ids = [n["id"] for n in current_config["nodes"] if n["id"] != "classify"]
        # Try to identify the target handler from proposals text
        target_handler = handler_ids[0]  # default to first
        for hid in handler_ids:
            if hid in proposals_text.lower() or hid.replace("handle_", "") in proposals_text.lower():
                target_handler = hid
                break

        # Clean the proposals into a reasonable append block
        additions = [{
            "node_id": target_handler,
            "append_text": "\n\nADDITIONAL GUIDANCE (from optimization):\n" + proposals_text[:1500],
        }]

    # Step 2: Apply additions to a deep copy of the current config
    new_config = copy.deepcopy(current_config)
    new_config["version"] = f"2.0.{iteration}"

    nodes_by_id = {n["id"]: n for n in new_config["nodes"]}
    applied_count = 0

    for addition in additions:
        node_id = addition.get("node_id", "")
        append_text = addition.get("append_text", "")

        if not node_id or not append_text:
            continue

        # Find the node (try exact match, then fuzzy)
        target_node = nodes_by_id.get(node_id)
        if target_node is None:
            # Fuzzy match
            for nid, node in nodes_by_id.items():
                if node_id in nid or nid in node_id:
                    target_node = node
                    break

        if target_node is None or target_node["id"] == "classify":
            continue  # Don't modify the classifier

        # Append the text to the instruction, preserving the base
        current_instruction = target_node.get("instruction", "")
        # Prevent instruction from growing unboundedly
        if len(current_instruction) + len(append_text) > 4000:
            append_text = append_text[:4000 - len(current_instruction)]

        target_node["instruction"] = current_instruction.rstrip() + "\n\n" + append_text.strip() + "\n"
        applied_count += 1

    if applied_count == 0:
        logger.warning("No additions were applied to %s config. Returning original.", harness_name)
        new_config["version"] = f"2.0.{iteration}"

    # Validate the result
    _validate_dag_config(new_config, harness_name)
    return new_config


def _validate_dag_config(config: dict, harness_name: str) -> None:
    """Validate a DAG config has all required fields.

    Raises ValueError if invalid.
    """
    if not isinstance(config, dict):
        raise ValueError(f"[{harness_name}] Config is not a dict: {type(config)}")

    if "nodes" not in config:
        raise ValueError(f"[{harness_name}] Config missing 'nodes'")
    if "edges" not in config:
        raise ValueError(f"[{harness_name}] Config missing 'edges'")

    if not isinstance(config["nodes"], list) or len(config["nodes"]) == 0:
        raise ValueError(f"[{harness_name}] 'nodes' must be a non-empty list")
    if not isinstance(config["edges"], list) or len(config["edges"]) == 0:
        raise ValueError(f"[{harness_name}] 'edges' must be a non-empty list")

    # Validate each node
    node_ids = set()
    has_classify = False
    for i, node in enumerate(config["nodes"]):
        if not isinstance(node, dict):
            raise ValueError(f"[{harness_name}] Node {i} is not a dict")
        for required_field in ("id", "type"):
            if required_field not in node:
                raise ValueError(f"[{harness_name}] Node {i} missing '{required_field}'")
        if node["type"] == "agent":
            for agent_field in ("model", "instruction"):
                if agent_field not in node:
                    raise ValueError(f"[{harness_name}] Agent node '{node['id']}' missing '{agent_field}'")
            # Check instruction is not markdown-contaminated
            instr = str(node.get("instruction", ""))
            if instr.startswith("```") or "```yaml" in instr or "```json" in instr:
                raise ValueError(f"[{harness_name}] Node '{node['id']}' instruction contains markdown code blocks")
        if "classify" in str(node.get("id", "")):
            has_classify = True
        node_ids.add(node["id"])

    if not has_classify:
        raise ValueError(f"[{harness_name}] No classify node found")

    # Validate edges
    for i, edge in enumerate(config["edges"]):
        if not isinstance(edge, dict):
            raise ValueError(f"[{harness_name}] Edge {i} is not a dict")
        if "from" not in edge:
            raise ValueError(f"[{harness_name}] Edge {i} missing 'from'")
        if "to" not in edge:
            raise ValueError(f"[{harness_name}] Edge {i} missing 'to'")

    # Check START edge exists
    has_start = any(e.get("from") == "START" for e in config["edges"])
    if not has_start:
        raise ValueError(f"[{harness_name}] No edge from START found")


# =============================================================================
# 7. Regression Gate
# =============================================================================

REGRESSION_TOLERANCE = 2.0  # Max allowed score drop per harness


def check_regression_gate(
    current: IterationResult,
    previous: IterationResult | None,
) -> tuple[bool, list[str]]:
    """Check if the current iteration passes the regression gate.

    Returns (passed, reasons).
    A harness passes if:
      - Its score improved or stayed stable (within REGRESSION_TOLERANCE)
    The aggregate passes if ALL harnesses pass.
    """
    if previous is None:
        return True, ["First iteration - no regression to check"]

    reasons = []
    all_pass = True

    for name, hr in current.harness_results.items():
        prev_hr = previous.harness_results.get(name)
        if prev_hr is None:
            reasons.append(f"  [{name}] NEW harness, no baseline")
            continue

        delta = hr.aggregate_score - prev_hr.aggregate_score
        if delta >= 0:
            reasons.append(f"  [{name}] IMPROVED: {prev_hr.aggregate_score:.1f} -> {hr.aggregate_score:.1f} (+{delta:.1f})")
        elif abs(delta) <= REGRESSION_TOLERANCE:
            reasons.append(f"  [{name}] STABLE: {prev_hr.aggregate_score:.1f} -> {hr.aggregate_score:.1f} ({delta:.1f}, within tolerance)")
        else:
            reasons.append(f"  [{name}] REGRESSED: {prev_hr.aggregate_score:.1f} -> {hr.aggregate_score:.1f} ({delta:.1f}, exceeds -{REGRESSION_TOLERANCE})")
            all_pass = False

    agg_delta = current.aggregate_score - previous.aggregate_score
    if agg_delta >= 0:
        reasons.append(f"  [aggregate] {previous.aggregate_score:.1f} -> {current.aggregate_score:.1f} (+{agg_delta:.1f})")
    else:
        reasons.append(f"  [aggregate] {previous.aggregate_score:.1f} -> {current.aggregate_score:.1f} ({agg_delta:.1f})")

    return all_pass, reasons


# =============================================================================
# 8. Score Tracking
# =============================================================================


def save_scores(all_iterations: list[IterationResult]) -> None:
    """Save score history to scores.json."""
    data = {"iterations": []}
    for it_result in all_iterations:
        entry = {
            "iteration": it_result.iteration,
            "aggregate_score": round(it_result.aggregate_score, 2),
            "total_cases": it_result.total_cases,
            "harnesses": {},
        }
        for name, hr in it_result.harness_results.items():
            entry["harnesses"][name] = {
                "score": round(hr.aggregate_score, 2),
                "cases": len(hr.records),
                "category_accuracy": round(hr.category_accuracy, 4),
                "avg_quality": round(hr.avg_quality, 2),
                "avg_helpfulness": round(hr.avg_helpfulness, 2),
                "category_breakdown": {
                    cat: {
                        "avg_score": round(info["avg_score"], 2),
                        "accuracy": round(info["accuracy"], 4),
                        "count": info["total"],
                    }
                    for cat, info in hr.category_breakdown().items()
                },
            }
        data["iterations"].append(entry)

    with open(SCORES_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nScores saved to {SCORES_FILE.name}")


# =============================================================================
# 9. Main Benchmark Runner
# =============================================================================


async def run_harness_benchmark(
    client: genai.Client,
    config: dict,
    config_path: str,
    cases: list[dict],
    harness_name: str,
    iteration: int,
) -> HarnessResult:
    """Run all benchmark cases for one harness and score them."""
    result = HarnessResult(
        harness_name=harness_name,
        iteration=iteration,
        config_path=config_path,
    )

    for i, case in enumerate(cases):
        # Run query through DAG
        dag_result = await run_dag_query(client, config, case["query"])

        await asyncio.sleep(LLM_DELAY)

        # Score the response
        score = await score_response(
            client,
            query=case["query"],
            expected_category=case["expected_category"],
            actual_category=dag_result["category"],
            response=dag_result["response"],
            quality_criteria=case["quality_criteria"],
            harness_name=harness_name,
        )

        record = InteractionRecord(
            query=case["query"],
            case_id=case["id"],
            expected_category=case["expected_category"],
            actual_category=dag_result["category"],
            response=dag_result["response"],
            score=score,
            quality_criteria=case["quality_criteria"],
        )
        result.records.append(record)

        cat_ok = "correct" if dag_result["category"] == case["expected_category"] else f"WRONG:{dag_result['category']}"
        print(f"    Case {case['id']:2d}: \"{case['query'][:45]}...\" -> {dag_result['category']} ({cat_ok}) -> {score['total_score']}")

        await asyncio.sleep(LLM_DELAY)

    return result


def _find_latest_config(harness: dict) -> Path:
    """Find the latest versioned config for a harness, or the base config."""
    base_name = Path(harness["yaml"]).stem
    # Find all versioned configs
    versioned = sorted(SANDBOX_DIR.glob(f"{base_name}_v*.yaml"))
    if versioned:
        return versioned[-1]
    return SANDBOX_DIR / harness["yaml"]


def _load_saved_scores() -> list[dict]:
    """Load previously saved iteration scores from scores.json."""
    if SCORES_FILE.exists():
        with open(SCORES_FILE) as f:
            data = json.load(f)
        return data.get("iterations", [])
    return []


async def run_multi_harness_loop(
    iterations: int = 3,
    resume: bool = False,
    start_iter: int | None = None,
) -> None:
    """Run the full multi-harness benchmark + conservative learning loop.

    Conservative optimization strategy:
      A. Best-config tracking -- always optimize from best-ever config, not latest
      B. Stability zone -- skip harnesses within STABILITY_THRESHOLD of their best
      C. One-at-a-time -- only modify the single weakest harness per iteration
      D. Rollback -- if aggregate drops >ROLLBACK_THRESHOLD below best, restore all
    """
    client = genai.Client(
        vertexai=True,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT", "alanblount-demo"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
    )

    all_iterations: list[IterationResult] = []
    promotion_log: list[dict] = []  # Track promotions/rejections/rollbacks per iteration

    # -- Best-config tracking (A) --
    # Stores the actual YAML content (not filename) of the best-ever config per harness
    best_config_content: dict[str, str] = {}   # harness_name -> YAML string
    best_harness_score: dict[str, float] = {}  # harness_name -> best score
    best_aggregate_score: float = 0.0

    # Track current config path per harness
    current_configs: dict[str, Path] = {}
    for h in HARNESSES:
        base_path = SANDBOX_DIR / h["yaml"]
        current_configs[h["name"]] = base_path
        # Initialize best configs from base YAML
        with open(base_path) as f:
            best_config_content[h["name"]] = f.read()
        best_harness_score[h["name"]] = 0.0

    for iteration in range(1, iterations + 1):
        print(f"\n{'='*70}")
        print(f"=== ITERATION {iteration} ===")
        print(f"{'='*70}")

        it_result = IterationResult(iteration=iteration)

        for harness in HARNESSES:
            hname = harness["name"]
            config_path = current_configs[hname]
            config = load_dag_config(config_path)
            dag_version = config.get("version", "?")

            print(f"\n  --- {hname} ({config_path.name}, v{dag_version}) ---")
            print(f"  Running {len(harness['cases'])} test cases...")

            hr = await run_harness_benchmark(
                client, config, str(config_path),
                harness["cases"], hname, iteration,
            )
            it_result.harness_results[hname] = hr

        # Print iteration summary
        all_iterations.append(it_result)
        prev_result = all_iterations[-2] if len(all_iterations) > 1 else None

        print(f"\n{'='*70}")
        print(f"=== ITERATION {iteration} RESULTS ===")
        print(f"{'='*70}")

        for hname, hr in it_result.harness_results.items():
            delta_str = ""
            if prev_result and hname in prev_result.harness_results:
                delta = hr.aggregate_score - prev_result.harness_results[hname].aggregate_score
                delta_str = f" ({'+' if delta >= 0 else ''}{delta:.1f})"
            best_str = f" [best={best_harness_score[hname]:.1f}]"
            print(f"  {hname:20s}: {hr.aggregate_score:.1f}/100 ({len(hr.records)} cases){delta_str}{best_str}")

        agg_delta_str = ""
        if prev_result:
            agg_delta = it_result.aggregate_score - prev_result.aggregate_score
            agg_delta_str = f" ({'+' if agg_delta >= 0 else ''}{agg_delta:.1f})"
        print(f"  {'Aggregate':20s}: {it_result.aggregate_score:.1f}/100{agg_delta_str} [best={best_aggregate_score:.1f}]")

        # -- (A) Update best-config tracking --
        new_bests = []
        for hname, hr in it_result.harness_results.items():
            if hr.aggregate_score > best_harness_score[hname]:
                old_best = best_harness_score[hname]
                best_harness_score[hname] = hr.aggregate_score
                # Save the YAML content that produced this score
                config_path = current_configs[hname]
                with open(config_path) as f:
                    best_config_content[hname] = f.read()
                new_bests.append(f"  [{hname}] NEW BEST: {old_best:.1f} -> {hr.aggregate_score:.1f} (config: {config_path.name})")

        if it_result.aggregate_score > best_aggregate_score:
            best_aggregate_score = it_result.aggregate_score

        if new_bests:
            print(f"\nBest-config updates:")
            for msg in new_bests:
                print(msg)
        else:
            print(f"\nNo new best scores this iteration.")

        # -- (D) Rollback check --
        rollback_triggered = False
        if iteration > 1 and it_result.aggregate_score < (best_aggregate_score - ROLLBACK_THRESHOLD):
            rollback_triggered = True
            print(f"\n*** ROLLBACK TRIGGERED: score {it_result.aggregate_score:.1f} below best {best_aggregate_score:.1f} "
                  f"(threshold: -{ROLLBACK_THRESHOLD}) ***")
            # Restore all configs to their best-ever versions
            for harness in HARNESSES:
                hname = harness["name"]
                base_name = Path(harness["yaml"]).stem
                rollback_path = SANDBOX_DIR / f"{base_name}_v{iteration + 1}.yaml"
                with open(rollback_path, "w") as f:
                    f.write(best_config_content[hname])
                current_configs[hname] = rollback_path
                print(f"  [{hname}] Rolled back to best config -> {rollback_path.name}")

            promotion_log.append({
                "iteration": iteration,
                "gate_passed": False,
                "promoted": False,
                "action": "ROLLBACK",
            })
        else:
            promotion_log.append({
                "iteration": iteration,
                "gate_passed": True,
                "promoted": iteration > 1 and len(new_bests) > 0,
                "action": "OK" if not new_bests else "PROMOTED",
            })

        # -- Learning loop (skip after last iteration and after rollback) --
        if iteration < iterations and not rollback_triggered:
            # -- (C) One-at-a-time: find the single weakest harness --
            harness_gaps: list[tuple[str, float, float]] = []  # (name, score, gap_from_best)
            for harness in HARNESSES:
                hname = harness["name"]
                score = it_result.harness_results[hname].aggregate_score
                gap = best_harness_score[hname] - score
                harness_gaps.append((hname, score, gap))

            harness_gaps.sort(key=lambda x: -x[2])  # largest gap first
            weakest_name, weakest_score, weakest_gap = harness_gaps[0]

            print(f"\nHarness gaps from best:")
            for hname, score, gap in harness_gaps:
                within_stability = gap <= STABILITY_THRESHOLD
                status = "STABLE (skip)" if within_stability else "CANDIDATE"
                print(f"  {hname:20s}: score={score:.1f}, best={best_harness_score[hname]:.1f}, gap={gap:.1f} -> {status}")

            # -- (B) Stability zone check --
            if weakest_gap <= STABILITY_THRESHOLD:
                print(f"\nAll harnesses within stability zone ({STABILITY_THRESHOLD} pts). No modifications.")
                promotion_log[-1]["action"] = "STABLE"
            else:
                print(f"\nModifying ONLY weakest harness: {weakest_name} (gap={weakest_gap:.1f})")

                # Start from the BEST config, not the latest (A)
                # Write best config to a temp location, load it
                best_yaml = best_config_content[weakest_name]
                best_parsed = yaml.safe_load(best_yaml)
                best_config = best_parsed.get("dag", best_parsed)

                harness_obj = next(h for h in HARNESSES if h["name"] == weakest_name)
                hr = it_result.harness_results[weakest_name]

                proposals = await analyze_and_propose_for_harness(
                    client, hr, best_config, weakest_name,
                )
                for i, prop in enumerate(proposals, 1):
                    first_line = prop.split("\n")[0].strip()
                    print(f"  [{weakest_name}] Proposal {i}: {first_line[:70]}")

                next_version = iteration + 1
                try:
                    new_config = await generate_improved_dag(
                        client, best_config, proposals, next_version, weakest_name,
                    )
                except (yaml.YAMLError, ValueError) as e:
                    print(f"  [{weakest_name}] YAML VALIDATION FAILED: {e}")
                    print(f"  [{weakest_name}] Keeping best config (not promoting invalid YAML)")
                except Exception as e:
                    print(f"  [{weakest_name}] WARNING: Failed to generate improved config: {e}")
                    print(f"  [{weakest_name}] Keeping best config")
                else:
                    # Write the new config for the weakest harness only
                    base_name = Path(harness_obj["yaml"]).stem
                    new_config_path = SANDBOX_DIR / f"{base_name}_v{next_version}.yaml"
                    with open(new_config_path, "w") as f:
                        yaml.dump({"dag": new_config}, f, default_flow_style=False, sort_keys=False)
                    print(f"  [{weakest_name}] Wrote {new_config_path.name}")
                    current_configs[weakest_name] = new_config_path

                # Ensure other harnesses stay on their best configs (unchanged)
                for harness in HARNESSES:
                    hname = harness["name"]
                    if hname != weakest_name:
                        print(f"  [{hname}] Unchanged (not the weakest)")

        # Save scores after each iteration (crash-safe)
        save_scores(all_iterations)

    # =================================================================
    # Final Summary Table
    # =================================================================
    print(f"\n{'='*70}")
    print("=== GRAPH GARDENER: {}-ITERATION CONSERVATIVE HILL CLIMB ===".format(iterations))
    print(f"{'='*70}\n")

    harness_names = [h["name"] for h in HARNESSES]
    # Header
    hdr = "Iter |"
    for hname in harness_names:
        hdr += f" {hname:^17s}|"
    hdr += f" {'AGGREGATE':^11s}| Action"
    print(hdr)
    print("-----|" + "-" * 17 + "|" + "-" * 17 + "|" + "-" * 17 + "|" + "-" * 11 + "|-----------")

    for it in all_iterations:
        row = f"  {it.iteration:2d} |"
        for hname in harness_names:
            hr = it.harness_results.get(hname)
            if hr:
                row += f"      {hr.aggregate_score:5.1f}      |"
            else:
                row += f"       N/A       |"
        row += f"   {it.aggregate_score:5.1f}   |"
        # Action status
        plog = next((p for p in promotion_log if p["iteration"] == it.iteration), None)
        if plog is None or it.iteration == 1:
            row += " baseline"
        else:
            row += f" {plog.get('action', '?'):9s}"
        print(row)

    # Best scores
    print(f"\nBest scores achieved:")
    for hname in harness_names:
        print(f"  {hname:20s}: {best_harness_score[hname]:.1f}")
    print(f"  {'Best aggregate':20s}: {best_aggregate_score:.1f}")

    # Total improvement
    if len(all_iterations) >= 2:
        total_improvement = all_iterations[-1].aggregate_score - all_iterations[0].aggregate_score
        sign = "+" if total_improvement >= 0 else ""
        print(f"\nTotal improvement (first->last): {sign}{total_improvement:.1f} points over {iterations} iterations")
        best_improvement = best_aggregate_score - all_iterations[0].aggregate_score
        sign2 = "+" if best_improvement >= 0 else ""
        print(f"Best improvement (first->best):  {sign2}{best_improvement:.1f} points")
    else:
        print(f"\nOnly 1 iteration run.")

    promotions = sum(1 for p in promotion_log if p.get("action") == "PROMOTED")
    rollbacks = sum(1 for p in promotion_log if p.get("action") == "ROLLBACK")
    stables = sum(1 for p in promotion_log if p.get("action") == "STABLE")
    print(f"\nPromotions: {promotions}")
    print(f"Rollbacks:  {rollbacks}")
    print(f"Stability skips: {stables}")

    print(f"\nTotal test cases per iteration: {all_iterations[0].total_cases if all_iterations else 0}")
    total_judge_calls = sum(it.total_cases for it in all_iterations) * 3  # 1 dag + 2 judge per case
    learning_calls = sum(1 for p in promotion_log if p.get("action") not in ("ROLLBACK", "STABLE", None)) * 2
    print(f"Total LLM calls: ~{total_judge_calls + learning_calls} (dag + 2x judge + learning loop)")

    # Weakest cases from final iteration
    print(f"\nWeakest cases in final iteration:")
    final = all_iterations[-1]
    for hname, hr in final.harness_results.items():
        for r in hr.weakest_cases(2):
            print(f"  [{hname}] Case {r.case_id}: \"{r.query[:45]}\" score={r.score['total_score']} ({r.score.get('reasoning', '')[:50]})")


# =============================================================================
# 10. Reset & CLI
# =============================================================================


def reset_all():
    """Remove all versioned DAG configs for all harnesses."""
    patterns = [
        "customer_support_adk_v*.yaml",
        "it_helpdesk_v*.yaml",
        "sales_inquiry_v*.yaml",
    ]
    removed = []
    for pattern in patterns:
        for f in SANDBOX_DIR.glob(pattern):
            f.unlink()
            removed.append(f.name)

    if SCORES_FILE.exists():
        SCORES_FILE.unlink()
        removed.append(SCORES_FILE.name)

    if removed:
        print(f"Reset: removed {', '.join(removed)}")
    else:
        print("Reset: no versioned configs or scores found, nothing to remove.")
    print("Original DAGs unchanged:")
    for h in HARNESSES:
        print(f"  {h['yaml']}")


async def main():
    parser = argparse.ArgumentParser(
        description="Multi-Harness Benchmark + Learning Loop with Regression Gates"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of benchmark iterations (default: 3)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Remove all versioned DAG configs and scores, then exit",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from saved scores.json (skip completed iterations)",
    )
    parser.add_argument(
        "--start-iter",
        type=int,
        default=None,
        help="Start from this iteration number (1-based). Uses latest versioned configs.",
    )
    args = parser.parse_args()

    if args.reset:
        reset_all()
        return

    total_cases = sum(len(h["cases"]) for h in HARNESSES)
    print("=" * 70)
    print("Multi-Harness Benchmark + Conservative Learning Loop")
    print("=" * 70)
    print(f"Harnesses: {len(HARNESSES)}")
    for h in HARNESSES:
        print(f"  {h['name']:20s}: {len(h['cases'])} cases ({h['yaml']})")
    print(f"Total cases per iteration: {total_cases}")
    print(f"Iterations: {args.iterations}")
    print(f"Judge model: {JUDGE_MODEL}")
    print(f"Strategy: conservative (best-config tracking, stability zone, one-at-a-time, rollback)")
    print(f"Stability threshold: {STABILITY_THRESHOLD} points")
    print(f"Rollback threshold: {ROLLBACK_THRESHOLD} points")
    print(f"LLM delay: {LLM_DELAY}s")
    print(f"Estimated LLM calls: ~{total_cases * 3 * args.iterations + 2 * (args.iterations - 1)} (dag + 2x judge + learning)")

    await run_multi_harness_loop(
        args.iterations,
        resume=args.resume,
        start_iter=args.start_iter,
    )


if __name__ == "__main__":
    asyncio.run(main())
