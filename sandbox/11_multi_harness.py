"""Prototype 11: Multi-Harness Benchmark + Learning Loop with Regression Gates.

Runs multiple domain harnesses (customer support, IT helpdesk, sales inquiry)
through a benchmark, scores responses with LLM-as-judge, proposes per-harness
improvements, and enforces regression gates before promoting.

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
JUDGE_MODEL = "gemini-2.5-flash"
SCORES_FILE = SANDBOX_DIR / "scores.json"

# Rate-limit delay between LLM calls (seconds)
LLM_DELAY = 0.5


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
    default_model = config.get("default_model", "gemini-2.5-flash")
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


async def score_response(
    client: genai.Client,
    query: str,
    expected_category: str,
    actual_category: str,
    response: str,
    quality_criteria: str,
    harness_name: str,
) -> dict:
    """Score a response using LLM-as-judge.

    Returns dict with: total_score, category_score, quality_score, helpfulness_score, reasoning
    """
    # Category correctness: deterministic (40 points)
    category_score = 40 if actual_category == expected_category else 0

    # Quality + helpfulness scored by LLM (60 points total)
    judge_prompt = f"""You are evaluating a {harness_name.replace('_', ' ')} response. Score it on two dimensions.

CUSTOMER QUERY: {query}
EXPECTED CATEGORY: {expected_category}
ACTUAL CATEGORY: {actual_category}

RESPONSE TO EVALUATE:
{response[:1000]}

QUALITY CRITERIA: {quality_criteria}

Score these two dimensions:
1. QUALITY (0-40): Does the response meet the quality criteria? Is it accurate and relevant?
   - 0-10: Completely misses the criteria
   - 11-20: Partially addresses criteria
   - 21-30: Mostly meets criteria
   - 31-40: Fully meets criteria with good detail

2. HELPFULNESS (0-20): Is the response concise, clear, and actionable?
   - 0-5: Unhelpful or confusing
   - 6-10: Somewhat helpful
   - 11-15: Helpful and clear
   - 16-20: Excellent - concise, clear, and actionable

Reply with ONLY a JSON object (no markdown, no code blocks):
{{"quality_score": <0-40>, "helpfulness_score": <0-20>, "reasoning": "<brief explanation>"}}"""

    try:
        resp = client.models.generate_content(
            model=JUDGE_MODEL,
            contents=[{"role": "user", "parts": [{"text": judge_prompt}]}],
        )
        judge_text = resp.candidates[0].content.parts[0].text.strip()

        # Parse JSON from response (handle markdown code blocks)
        if "```" in judge_text:
            match = re.search(r"```(?:json)?\s*(.*?)```", judge_text, re.DOTALL)
            if match:
                judge_text = match.group(1).strip()

        scores = json.loads(judge_text)
        quality_score = max(0, min(40, int(scores.get("quality_score", 20))))
        helpfulness_score = max(0, min(20, int(scores.get("helpfulness_score", 10))))
        reasoning = scores.get("reasoning", "")

    except Exception as e:
        logger.warning("Judge scoring failed: %s", e)
        quality_score = 20  # default middle score
        helpfulness_score = 10
        reasoning = f"Scoring error: {e}"

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


async def analyze_and_propose_for_harness(
    client: genai.Client,
    harness_result: HarnessResult,
    current_config: dict,
    harness_name: str,
) -> list[str]:
    """Analyze benchmark results for one harness and propose improvements."""
    breakdown = harness_result.category_breakdown()
    weak_cases = harness_result.weakest_cases(5)

    valid_categories = extract_categories_from_config(current_config)

    analysis = f"""Benchmark Results for {harness_name} (Iteration {harness_result.iteration}):
- Aggregate Score: {harness_result.aggregate_score:.1f}/100
- Category Accuracy: {harness_result.category_accuracy:.0%}
- Quality Score: {harness_result.avg_quality:.1f}/40
- Helpfulness Score: {harness_result.avg_helpfulness:.1f}/20

Valid Categories: {', '.join(valid_categories)}

Category Breakdown:
"""
    for cat, info in breakdown.items():
        analysis += f"  {cat}: avg={info['avg_score']:.1f}, accuracy={info['accuracy']:.0%} ({info['total']} cases)\n"

    analysis += "\nWeakest Cases:\n"
    for r in weak_cases:
        analysis += f"  Case {r.case_id}: \"{r.query[:50]}\" score={r.score['total_score']}, "
        analysis += f"expected={r.expected_category}, got={r.actual_category}\n"
        analysis += f"    Judge: {r.score.get('reasoning', '')[:100]}\n"

    prompt = f"""You are an AI workflow optimizer. Analyze these {harness_name.replace('_', ' ')} DAG benchmark results
and propose specific, actionable improvements to the DAG configuration.

{analysis}

Current DAG config (YAML nodes):
{yaml.dump({'nodes': current_config['nodes']}, default_flow_style=False)}

Propose 2-3 specific improvements. Focus on:
1. Improving the classifier instruction to reduce misclassifications
2. Making handler instructions more specific to address quality gaps
3. Adding detail to handler instructions for poorly-scoring categories

For each proposal, explain:
- What to change
- Why (based on the data)
- Expected impact

Reply with a numbered list of proposals. Be specific about instruction text changes."""

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
    """Generate an improved DAG YAML config based on proposals."""
    current_yaml = yaml.dump({"dag": current_config}, default_flow_style=False)
    valid_categories = extract_categories_from_config(current_config)

    prompt = f"""You are a workflow configuration generator. Apply the following improvement
proposals to the {harness_name.replace('_', ' ')} DAG configuration.

CURRENT CONFIG:
```yaml
{current_yaml}
```

IMPROVEMENT PROPOSALS:
{chr(10).join(proposals)}

Generate the complete updated YAML config with these changes applied.
Rules:
- Keep all existing node IDs unless renaming is necessary
- Update instructions to be more specific and detailed based on proposals
- Keep the same edge structure unless adding new nodes/routes
- Keep model as gemini-2.5-flash for all nodes
- Update the version to "2.0.{iteration}"
- Keep nodes of type 'agent' only
- IMPORTANT: Keep the classify node's instruction format so it outputs
  ONLY one of: {', '.join(valid_categories)}
- All edges from classify must use conditions: {', '.join(valid_categories)}

Reply with ONLY the YAML (no markdown code blocks, no explanation).
Start with "dag:" on the first line."""

    await asyncio.sleep(LLM_DELAY)
    resp = client.models.generate_content(
        model=JUDGE_MODEL,
        contents=[{"role": "user", "parts": [{"text": prompt}]}],
    )
    yaml_text = resp.candidates[0].content.parts[0].text.strip()

    # Strip markdown code blocks if present
    if "```" in yaml_text:
        lines = yaml_text.split("\n")
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
            yaml_text = "\n".join(block_lines)

    try:
        parsed = yaml.safe_load(yaml_text)
        if "dag" in parsed:
            return parsed["dag"]
        return parsed
    except yaml.YAMLError as e:
        logger.error("Failed to parse generated YAML for %s: %s", harness_name, e)
        logger.error("Raw YAML:\n%s", yaml_text[:500])
        raise


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


async def run_multi_harness_loop(iterations: int = 3) -> None:
    """Run the full multi-harness benchmark + learning loop."""
    client = genai.Client(
        vertexai=True,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT", "alanblount-demo"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
    )

    all_iterations: list[IterationResult] = []
    # Track current config path per harness
    current_configs: dict[str, Path] = {}
    for h in HARNESSES:
        current_configs[h["name"]] = SANDBOX_DIR / h["yaml"]

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
            print(f"  {hname:20s}: {hr.aggregate_score:.1f}/100 ({len(hr.records)} cases){delta_str}")

        agg_delta_str = ""
        if prev_result:
            agg_delta = it_result.aggregate_score - prev_result.aggregate_score
            agg_delta_str = f" ({'+' if agg_delta >= 0 else ''}{agg_delta:.1f})"
        print(f"  {'Aggregate':20s}: {it_result.aggregate_score:.1f}/100{agg_delta_str}")

        # Regression gate
        gate_passed, gate_reasons = check_regression_gate(it_result, prev_result)
        print(f"\nRegression gate: {'PASS' if gate_passed else 'FAIL'}")
        for reason in gate_reasons:
            print(reason)

        # Learning loop (skip after last iteration)
        if iteration < iterations:
            print(f"\nLearning loop proposals:")

            for harness in HARNESSES:
                hname = harness["name"]
                config_path = current_configs[hname]
                config = load_dag_config(config_path)
                hr = it_result.harness_results[hname]

                proposals = await analyze_and_propose_for_harness(
                    client, hr, config, hname,
                )
                for i, prop in enumerate(proposals, 1):
                    first_line = prop.split("\n")[0].strip()
                    print(f"  [{hname}] Proposal {i}: {first_line[:70]}")

                # Only generate improved DAG if gate passed or this is first iteration
                if gate_passed or prev_result is None:
                    next_version = iteration + 1
                    try:
                        new_config = await generate_improved_dag(
                            client, config, proposals, next_version, hname,
                        )
                    except Exception as e:
                        print(f"  [{hname}] WARNING: Failed to generate improved config: {e}")
                        continue

                    # Validate
                    if "nodes" not in new_config or "edges" not in new_config:
                        print(f"  [{hname}] WARNING: Generated config missing nodes/edges, keeping current")
                        continue

                    has_classify = any("classify" in n["id"] for n in new_config["nodes"])
                    if not has_classify:
                        print(f"  [{hname}] WARNING: Generated config missing classify node, keeping current")
                        continue

                    # Write the new config
                    base_name = Path(harness["yaml"]).stem
                    new_config_path = SANDBOX_DIR / f"{base_name}_v{next_version}.yaml"
                    with open(new_config_path, "w") as f:
                        yaml.dump({"dag": new_config}, f, default_flow_style=False, sort_keys=False)
                    print(f"  [{hname}] Wrote {new_config_path.name}")
                    current_configs[hname] = new_config_path
                else:
                    print(f"  [{hname}] Regression gate FAILED - keeping current config (no promotion)")

        # Save scores after each iteration
        save_scores(all_iterations)

    # =================================================================
    # Final Summary Table
    # =================================================================
    print(f"\n{'='*70}")
    print("=== FINAL SUMMARY ===")
    print(f"{'='*70}")

    # Header
    header = f"  {'Harness':20s}"
    for it in all_iterations:
        header += f"  {'Iter ' + str(it.iteration):>10s}"
    header += f"  {'Delta':>8s}"
    print(header)
    print("  " + "-" * (20 + 12 * len(all_iterations) + 10))

    # Per-harness rows
    for harness in HARNESSES:
        hname = harness["name"]
        row = f"  {hname:20s}"
        scores_list = []
        for it in all_iterations:
            hr = it.harness_results.get(hname)
            if hr:
                score = hr.aggregate_score
                scores_list.append(score)
                row += f"  {score:>9.1f}"
            else:
                row += f"  {'N/A':>9s}"
        if len(scores_list) >= 2:
            delta = scores_list[-1] - scores_list[0]
            row += f"  {'+' if delta >= 0 else ''}{delta:>6.1f}"
        else:
            row += f"  {'N/A':>8s}"
        print(row)

    # Aggregate row
    agg_row = f"  {'AGGREGATE':20s}"
    agg_scores = []
    for it in all_iterations:
        agg = it.aggregate_score
        agg_scores.append(agg)
        agg_row += f"  {agg:>9.1f}"
    if len(agg_scores) >= 2:
        delta = agg_scores[-1] - agg_scores[0]
        agg_row += f"  {'+' if delta >= 0 else ''}{delta:>6.1f}"
    print("  " + "-" * (20 + 12 * len(all_iterations) + 10))
    print(agg_row)

    print(f"\nTotal test cases per iteration: {all_iterations[0].total_cases if all_iterations else 0}")
    print(f"Total LLM calls: ~{all_iterations[0].total_cases * 2 * len(all_iterations) if all_iterations else 0} (dag+judge) + learning loop calls")

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
    args = parser.parse_args()

    if args.reset:
        reset_all()
        return

    total_cases = sum(len(h["cases"]) for h in HARNESSES)
    print("=" * 70)
    print("Multi-Harness Benchmark + Learning Loop")
    print("=" * 70)
    print(f"Harnesses: {len(HARNESSES)}")
    for h in HARNESSES:
        print(f"  {h['name']:20s}: {len(h['cases'])} cases ({h['yaml']})")
    print(f"Total cases per iteration: {total_cases}")
    print(f"Iterations: {args.iterations}")
    print(f"Judge model: {JUDGE_MODEL}")
    print(f"Regression tolerance: {REGRESSION_TOLERANCE} points")
    print(f"Estimated LLM calls: ~{total_cases * 2 * args.iterations + len(HARNESSES) * 2 * (args.iterations - 1)}")

    await run_multi_harness_loop(args.iterations)


if __name__ == "__main__":
    asyncio.run(main())
