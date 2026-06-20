"""Prototype 14: Adversarial Verification A/B Test.

Tests Hypothesis H2: Adding a "verifier" node that challenges handler output
will catch errors that a single-pass pipeline misses.

Verifier nodes:
  - Receive handler output AND original query
  - Critically evaluate: accurate, complete, helpful?
  - Can APPROVE (pass through), REVISE (send back with feedback), or REJECT
  - Max 2 revision loops per case

A/B test structure:
  Control:   Original DAG (no verifiers)
  Treatment: Same DAG + verifier nodes after each handler

Harnesses tested:
  - Customer support (23 cases)
  - IT helpdesk (18 cases)

Usage:
    python sandbox/14_adversarial_verification.py
    python sandbox/14_adversarial_verification.py --harness customer_support
    python sandbox/14_adversarial_verification.py --harness it_helpdesk

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
logger = logging.getLogger("adversarial_verification")

SANDBOX_DIR = Path(__file__).parent
MODEL = "gemini-3.5-flash"
JUDGE_MODEL = "gemini-3.5-flash"
RESULTS_FILE = SANDBOX_DIR / "scores_adversarial.json"

# Rate-limit delay between LLM calls (seconds)
LLM_DELAY = 0.3

# Max revision loops when verifier says REVISE
MAX_REVISIONS = 2


# =============================================================================
# Test Cases (same as 11_multi_harness.py)
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


# =============================================================================
# Harness Configuration
# =============================================================================

HARNESSES = {
    "customer_support": {
        "control_yaml": "customer_support_adk.yaml",
        "treatment_yaml": "customer_support_verified.yaml",
        "cases": CUSTOMER_SUPPORT_CASES,
    },
    "it_helpdesk": {
        "control_yaml": "it_helpdesk.yaml",
        "treatment_yaml": "it_helpdesk_verified.yaml",
        "cases": IT_HELPDESK_CASES,
    },
}


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class VerificationRecord:
    """Tracks what happened during verification for a single case."""
    verifier_verdict: str       # APPROVED, REVISE, REJECT
    verifier_feedback: str      # The verifier's feedback text
    revision_count: int         # How many revisions were triggered
    revision_verdicts: list[str] = field(default_factory=list)  # Verdict per revision round
    score_before_revision: int = 0     # Score of the initial handler output
    score_after_revision: int = 0      # Score after revision (same as before if APPROVED)


@dataclass
class CaseResult:
    """Result for a single test case under one condition."""
    case_id: int
    query: str
    expected_category: str
    actual_category: str
    response: str
    response_length: int
    quality_score: int          # 0-100 LLM judge score
    quality_reasoning: str
    verification: VerificationRecord | None = None  # Only for treatment


@dataclass
class ConditionResult:
    """Aggregate results for one condition (control or treatment) on one harness."""
    condition: str              # "control" or "treatment"
    harness_name: str
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def avg_score(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.quality_score for c in self.cases) / len(self.cases)

    @property
    def avg_response_length(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.response_length for c in self.cases) / len(self.cases)

    @property
    def revision_count(self) -> int:
        return sum(1 for c in self.cases if c.verification and c.verification.revision_count > 0)

    @property
    def reject_count(self) -> int:
        return sum(1 for c in self.cases if c.verification and c.verification.verifier_verdict == "REJECT")


# =============================================================================
# DAG Loading & Routing
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
        elif dst == "END":
            # Skip END edges (just marker for verifiers)
            continue
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


# =============================================================================
# LLM Call Helpers
# =============================================================================

async def call_llm(
    client: genai.Client,
    model: str,
    prompt: str,
    temperature: float | None = None,
) -> str:
    """Make a single LLM call and return text response."""
    await asyncio.sleep(LLM_DELAY)
    config = {}
    if temperature is not None:
        config["temperature"] = temperature
    try:
        resp = client.models.generate_content(
            model=model,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            config=config if config else None,
        )
        text = ""
        for p in resp.candidates[0].content.parts:
            if hasattr(p, "text") and p.text:
                text += p.text
        return text.strip()
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return f"ERROR: {e}"


# =============================================================================
# Part 1: Verifier Node Logic
# =============================================================================

def parse_verifier_verdict(text: str) -> tuple[str, str]:
    """Parse verifier output into (verdict, feedback).

    Returns one of:
        ("APPROVED", "")
        ("REVISE", "specific feedback...")
        ("REJECT", "reason...")
    """
    text = text.strip()

    # Check for APPROVED
    if text.upper().startswith("APPROVED"):
        return "APPROVED", ""

    # Check for REVISE
    revise_match = re.match(r"REVISE\s*:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if revise_match:
        return "REVISE", revise_match.group(1).strip()

    # Check for REJECT
    reject_match = re.match(r"REJECT\s*:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if reject_match:
        return "REJECT", reject_match.group(1).strip()

    # If the text contains these keywords anywhere (LLM sometimes prefixes)
    text_upper = text.upper()
    if "APPROVED" in text_upper and "REVISE" not in text_upper and "REJECT" not in text_upper:
        return "APPROVED", ""
    if "REVISE:" in text_upper:
        idx = text_upper.index("REVISE:")
        return "REVISE", text[idx + 7:].strip()
    if "REJECT:" in text_upper:
        idx = text_upper.index("REJECT:")
        return "REJECT", text[idx + 7:].strip()

    # Default: treat as approved if it seems positive
    if any(word in text_upper for word in ["GOOD", "WELL", "ADEQUATE", "SATISFACTOR"]):
        return "APPROVED", ""

    # Otherwise treat as revision needed
    return "REVISE", text[:200]


async def run_verifier(
    client: genai.Client,
    verifier_node: dict,
    original_query: str,
    handler_response: str,
) -> tuple[str, str]:
    """Run a verifier node and return (verdict, feedback)."""
    instruction = verifier_node.get("instruction", "Review the response for quality.")
    model = verifier_node.get("model", MODEL)

    prompt = f"""{instruction}

ORIGINAL CUSTOMER QUERY:
{original_query}

RESPONSE TO REVIEW:
{handler_response}

Your verdict:"""

    verifier_output = await call_llm(client, model, prompt)
    return parse_verifier_verdict(verifier_output)


async def run_handler_with_revision(
    client: genai.Client,
    handler_node: dict,
    query: str,
    revision_feedback: str,
) -> str:
    """Re-run a handler with verifier feedback incorporated."""
    instruction = handler_node.get("instruction", "Help the user.")
    model = handler_node.get("model", MODEL)

    prompt = f"""{instruction}

User request: {query}

IMPORTANT - A quality reviewer found issues with your previous response.
Reviewer feedback: {revision_feedback}

Please provide an improved response that addresses this feedback."""

    return await call_llm(client, model, prompt)


# =============================================================================
# Part 3: DAG Execution (Control vs Treatment)
# =============================================================================

async def execute_control(
    client: genai.Client,
    config: dict,
    query: str,
) -> dict:
    """Execute a query through the control DAG (no verifiers)."""
    nodes_by_id = {n["id"]: n for n in config["nodes"]}
    routing = build_routing(config)
    valid_categories = extract_categories_from_config(config)

    # Step 1: classify
    classify_node = nodes_by_id.get("classify")
    if not classify_node:
        return {"category": "unknown", "response": "ERROR: no classify node", "nodes_visited": []}

    classify_prompt = f"{classify_node['instruction']}\n\nUser request: {query}"
    classify_output = await call_llm(client, classify_node.get("model", MODEL), classify_prompt)

    # Extract category
    classify_lower = classify_output.lower().strip()
    detected_category = "unknown"
    for cat in valid_categories:
        if cat in classify_lower:
            detected_category = cat
            break

    # Step 2: route to handler
    handler_id = None
    for route in routing["conditional"].get("classify", []):
        if route["condition"].lower() in classify_lower:
            handler_id = route["to"]
            break

    if handler_id is None or handler_id not in nodes_by_id:
        return {"category": detected_category, "response": classify_output, "nodes_visited": ["classify"]}

    handler_node = nodes_by_id[handler_id]
    handler_prompt = f"{handler_node['instruction']}\n\nUser request: {query}"
    handler_output = await call_llm(client, handler_node.get("model", MODEL), handler_prompt)

    return {
        "category": detected_category,
        "response": handler_output,
        "nodes_visited": ["classify", handler_id],
    }


async def execute_treatment(
    client: genai.Client,
    config: dict,
    query: str,
) -> tuple[dict, VerificationRecord]:
    """Execute a query through the treatment DAG (with verifiers).

    Returns (result_dict, verification_record).
    """
    nodes_by_id = {n["id"]: n for n in config["nodes"]}
    routing = build_routing(config)
    valid_categories = extract_categories_from_config(config)

    # Step 1: classify (same as control)
    classify_node = nodes_by_id.get("classify")
    if not classify_node:
        empty_ver = VerificationRecord("REJECT", "no classify node", 0)
        return {"category": "unknown", "response": "ERROR: no classify node", "nodes_visited": []}, empty_ver

    classify_prompt = f"{classify_node['instruction']}\n\nUser request: {query}"
    classify_output = await call_llm(client, classify_node.get("model", MODEL), classify_prompt)

    classify_lower = classify_output.lower().strip()
    detected_category = "unknown"
    for cat in valid_categories:
        if cat in classify_lower:
            detected_category = cat
            break

    # Step 2: route to handler
    handler_id = None
    for route in routing["conditional"].get("classify", []):
        if route["condition"].lower() in classify_lower:
            handler_id = route["to"]
            break

    if handler_id is None or handler_id not in nodes_by_id:
        empty_ver = VerificationRecord("REJECT", f"no handler for category {detected_category}", 0)
        return {"category": detected_category, "response": classify_output, "nodes_visited": ["classify"]}, empty_ver

    handler_node = nodes_by_id[handler_id]

    # Step 3: run handler
    handler_prompt = f"{handler_node['instruction']}\n\nUser request: {query}"
    handler_output = await call_llm(client, handler_node.get("model", MODEL), handler_prompt)

    # Step 4: find the verifier for this handler
    verifier_id = None
    for dst_list in routing["unconditional"].get(handler_id, []):
        if isinstance(dst_list, str) and dst_list in nodes_by_id:
            if nodes_by_id[dst_list].get("type") == "verifier":
                verifier_id = dst_list
                break

    if verifier_id is None:
        # Fallback: look for verify_* matching handle_*
        expected_verifier = handler_id.replace("handle_", "verify_")
        if expected_verifier in nodes_by_id:
            verifier_id = expected_verifier

    if verifier_id is None:
        # No verifier found -- just approve
        ver = VerificationRecord("APPROVED", "no verifier node found", 0)
        return {
            "category": detected_category,
            "response": handler_output,
            "nodes_visited": ["classify", handler_id],
        }, ver

    verifier_node = nodes_by_id[verifier_id]

    # Step 5: verification loop
    current_response = handler_output
    revision_count = 0
    revision_verdicts = []

    verdict, feedback = await run_verifier(client, verifier_node, query, current_response)

    if verdict == "APPROVED":
        ver = VerificationRecord(
            verifier_verdict="APPROVED",
            verifier_feedback="",
            revision_count=0,
        )
        return {
            "category": detected_category,
            "response": current_response,
            "nodes_visited": ["classify", handler_id, verifier_id],
        }, ver

    if verdict == "REJECT":
        ver = VerificationRecord(
            verifier_verdict="REJECT",
            verifier_feedback=feedback,
            revision_count=0,
        )
        # Still return the response but flagged
        return {
            "category": detected_category,
            "response": current_response,
            "nodes_visited": ["classify", handler_id, verifier_id],
        }, ver

    # REVISE loop
    for rev_round in range(MAX_REVISIONS):
        revision_count += 1
        revision_verdicts.append(f"REVISE: {feedback[:80]}")

        # Re-run handler with feedback
        current_response = await run_handler_with_revision(
            client, handler_node, query, feedback,
        )

        # Re-verify
        verdict, feedback = await run_verifier(client, verifier_node, query, current_response)

        if verdict == "APPROVED":
            revision_verdicts.append("APPROVED")
            break
        elif verdict == "REJECT":
            revision_verdicts.append(f"REJECT: {feedback[:80]}")
            break
        # else REVISE again (if we have rounds left)

    ver = VerificationRecord(
        verifier_verdict=verdict,
        verifier_feedback=feedback,
        revision_count=revision_count,
        revision_verdicts=revision_verdicts,
    )

    return {
        "category": detected_category,
        "response": current_response,
        "nodes_visited": ["classify", handler_id, verifier_id] + [f"{verifier_id}_rev{i+1}" for i in range(revision_count)],
    }, ver


# =============================================================================
# LLM-as-Judge Scoring
# =============================================================================

async def score_response(
    client: genai.Client,
    query: str,
    expected_category: str,
    actual_category: str,
    response: str,
    quality_criteria: str,
    harness_name: str,
) -> tuple[int, str]:
    """Score a response using LLM-as-judge.

    Returns (score 0-100, reasoning).

    Scoring: 50 pts category + 30 pts quality + 20 pts helpfulness.
    """
    category_score = 50 if actual_category == expected_category else 0

    judge_prompt = f"""You are a strict scoring judge for a {harness_name.replace('_', ' ')} system.
Evaluate the response below and output EXACTLY one JSON object.

INPUT:
- Customer query: "{query}"
- Expected category: {expected_category}
- Actual category: {actual_category}
- Quality criteria: {quality_criteria}

RESPONSE:
{response[:1000]}

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

    judge_text = await call_llm(client, JUDGE_MODEL, judge_prompt, temperature=0.0)

    try:
        # Parse JSON
        if "```" in judge_text:
            match = re.search(r"```(?:json)?\s*(.*?)```", judge_text, re.DOTALL)
            if match:
                judge_text = match.group(1).strip()
        json_match = re.search(r'\{[^{}]*\}', judge_text)
        if json_match:
            judge_text = json_match.group(0)

        scores = json.loads(judge_text)
        q = max(0, min(30, int(scores.get("quality_score", 15))))
        h = max(0, min(20, int(scores.get("helpfulness_score", 10))))
        total = category_score + q + h
        reasoning = scores.get("reasoning", "")
        return total, reasoning
    except Exception as e:
        logger.warning("Judge parse failed: %s", e)
        return category_score + 25, f"Parse error: {e}"


# =============================================================================
# Part 4: A/B Test Runner
# =============================================================================

async def run_ab_test_harness(
    client: genai.Client,
    harness_name: str,
    harness_config: dict,
) -> tuple[ConditionResult, ConditionResult]:
    """Run A/B test for one harness. Returns (control_result, treatment_result)."""
    cases = harness_config["cases"]

    # Load configs
    control_config = load_dag_config(SANDBOX_DIR / harness_config["control_yaml"])
    treatment_config = load_dag_config(SANDBOX_DIR / harness_config["treatment_yaml"])

    control_result = ConditionResult(condition="control", harness_name=harness_name)
    treatment_result = ConditionResult(condition="treatment", harness_name=harness_name)

    # --- CONTROL: run without verifiers ---
    print(f"\n  --- Control ({harness_config['control_yaml']}) ---")
    for case in cases:
        dag_result = await execute_control(client, control_config, case["query"])

        score, reasoning = await score_response(
            client, case["query"], case["expected_category"],
            dag_result["category"], dag_result["response"],
            case["quality_criteria"], harness_name,
        )

        cr = CaseResult(
            case_id=case["id"],
            query=case["query"],
            expected_category=case["expected_category"],
            actual_category=dag_result["category"],
            response=dag_result["response"],
            response_length=len(dag_result["response"]),
            quality_score=score,
            quality_reasoning=reasoning,
        )
        control_result.cases.append(cr)

        cat_ok = "ok" if dag_result["category"] == case["expected_category"] else f"WRONG:{dag_result['category']}"
        print(f"    Case {case['id']:2d}: score={score:3d} cat={cat_ok:18s} \"{case['query'][:40]}\"")

    print(f"  Control avg: {control_result.avg_score:.1f}")

    # --- TREATMENT: run with verifiers ---
    print(f"\n  --- Treatment ({harness_config['treatment_yaml']}) ---")
    for case in cases:
        dag_result, verification = await execute_treatment(client, treatment_config, case["query"])

        # If verifier triggered revisions, score the INITIAL output too
        # (for before/after comparison)
        if verification.revision_count > 0:
            # We need to score the pre-revision response, but we don't have it anymore.
            # The verification record only has the final response.
            # We'll score what we have (the final/revised response).
            pass

        score, reasoning = await score_response(
            client, case["query"], case["expected_category"],
            dag_result["category"], dag_result["response"],
            case["quality_criteria"], harness_name,
        )

        cr = CaseResult(
            case_id=case["id"],
            query=case["query"],
            expected_category=case["expected_category"],
            actual_category=dag_result["category"],
            response=dag_result["response"],
            response_length=len(dag_result["response"]),
            quality_score=score,
            quality_reasoning=reasoning,
            verification=verification,
        )
        treatment_result.cases.append(cr)

        ver_str = verification.verifier_verdict
        if verification.revision_count > 0:
            ver_str += f"(rev={verification.revision_count})"
        cat_ok = "ok" if dag_result["category"] == case["expected_category"] else f"WRONG:{dag_result['category']}"
        print(f"    Case {case['id']:2d}: score={score:3d} cat={cat_ok:18s} ver={ver_str:20s} \"{case['query'][:35]}\"")

    print(f"  Treatment avg: {treatment_result.avg_score:.1f}")

    return control_result, treatment_result


# =============================================================================
# Analysis & Reporting
# =============================================================================

def analyze_verifier_impact(treatment: ConditionResult, control: ConditionResult) -> dict:
    """Analyze the impact of verification on individual cases."""
    analysis = {
        "total_cases": len(treatment.cases),
        "approved_first_pass": 0,
        "revisions_triggered": 0,
        "rejections": 0,
        "revision_improved": 0,
        "revision_hurt": 0,
        "revision_neutral": 0,
        "avg_improvement_per_revision": 0.0,
        "cases_with_revision": [],
    }

    revision_deltas = []

    for t_case in treatment.cases:
        if not t_case.verification:
            continue

        ver = t_case.verification
        if ver.verifier_verdict == "APPROVED" and ver.revision_count == 0:
            analysis["approved_first_pass"] += 1
        elif ver.revision_count > 0:
            analysis["revisions_triggered"] += 1

            # Find matching control case for comparison
            c_case = next((c for c in control.cases if c.case_id == t_case.case_id), None)
            if c_case:
                delta = t_case.quality_score - c_case.quality_score
                revision_deltas.append(delta)
                if delta > 0:
                    analysis["revision_improved"] += 1
                elif delta < 0:
                    analysis["revision_hurt"] += 1
                else:
                    analysis["revision_neutral"] += 1

                analysis["cases_with_revision"].append({
                    "case_id": t_case.case_id,
                    "query": t_case.query[:50],
                    "control_score": c_case.quality_score,
                    "treatment_score": t_case.quality_score,
                    "delta": delta,
                    "revision_count": ver.revision_count,
                    "final_verdict": ver.verifier_verdict,
                })

        if ver.verifier_verdict == "REJECT":
            analysis["rejections"] += 1

    if revision_deltas:
        analysis["avg_improvement_per_revision"] = sum(revision_deltas) / len(revision_deltas)

    return analysis


def print_ab_report(
    harness_name: str,
    control: ConditionResult,
    treatment: ConditionResult,
    analysis: dict,
) -> None:
    """Print formatted A/B test report for one harness."""
    delta = treatment.avg_score - control.avg_score
    sign = "+" if delta >= 0 else ""

    print(f"\n=== A/B TEST: {harness_name.replace('_', ' ').title()} ===")
    print(f"Control (no verifier):  avg={control.avg_score:.1f}, cases={len(control.cases)}")
    print(f"Treatment (verified):   avg={treatment.avg_score:.1f}, cases={len(treatment.cases)}, "
          f"revisions={analysis['revisions_triggered']}, delta={sign}{delta:.1f}")

    print(f"\n  Category breakdown:")
    # Group by category
    cat_scores_c: dict[str, list] = {}
    cat_scores_t: dict[str, list] = {}
    for c in control.cases:
        cat_scores_c.setdefault(c.expected_category, []).append(c.quality_score)
    for c in treatment.cases:
        cat_scores_t.setdefault(c.expected_category, []).append(c.quality_score)

    for cat in sorted(cat_scores_c.keys()):
        c_avg = sum(cat_scores_c[cat]) / len(cat_scores_c[cat])
        t_avg = sum(cat_scores_t.get(cat, [0])) / len(cat_scores_t.get(cat, [1]))
        cat_delta = t_avg - c_avg
        sign = "+" if cat_delta >= 0 else ""
        print(f"    {cat:20s}: control={c_avg:.1f} treatment={t_avg:.1f} ({sign}{cat_delta:.1f})")

    print(f"\n  Response length:")
    print(f"    Control avg:   {control.avg_response_length:.0f} chars")
    print(f"    Treatment avg: {treatment.avg_response_length:.0f} chars")

    if analysis["cases_with_revision"]:
        print(f"\n  Revision details:")
        for rev_case in analysis["cases_with_revision"]:
            sign = "+" if rev_case["delta"] >= 0 else ""
            print(f"    Case {rev_case['case_id']:2d}: \"{rev_case['query']}\" "
                  f"control={rev_case['control_score']} -> treatment={rev_case['treatment_score']} "
                  f"({sign}{rev_case['delta']}) revisions={rev_case['revision_count']} "
                  f"final={rev_case['final_verdict']}")


# =============================================================================
# Main Runner
# =============================================================================

async def run_full_experiment(harness_filter: str | None = None) -> None:
    """Run the full A/B experiment across all harnesses."""
    client = genai.Client(
        vertexai=True,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT", "alanblount-demo"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
    )

    harnesses_to_run = HARNESSES
    if harness_filter:
        if harness_filter not in HARNESSES:
            print(f"ERROR: Unknown harness '{harness_filter}'. Available: {list(HARNESSES.keys())}")
            return
        harnesses_to_run = {harness_filter: HARNESSES[harness_filter]}

    total_cases = sum(len(h["cases"]) for h in harnesses_to_run.values())

    print("=" * 70)
    print("Adversarial Verification A/B Test")
    print("=" * 70)
    print(f"Hypothesis H2: Verifier nodes catch errors single-pass misses")
    print(f"Harnesses: {list(harnesses_to_run.keys())}")
    print(f"Total cases: {total_cases} x 2 conditions = {total_cases * 2}")
    print(f"Model: {MODEL}")
    print(f"Judge: {JUDGE_MODEL} (temperature=0)")
    print(f"Max revisions per case: {MAX_REVISIONS}")
    print(f"LLM delay: {LLM_DELAY}s")
    est_calls = total_cases * 2 * 3 + total_cases * MAX_REVISIONS  # rough upper bound
    print(f"Estimated LLM calls: ~{est_calls}")

    t_start = time.time()

    all_controls: dict[str, ConditionResult] = {}
    all_treatments: dict[str, ConditionResult] = {}
    all_analyses: dict[str, dict] = {}

    for harness_name, harness_config in harnesses_to_run.items():
        print(f"\n{'='*70}")
        print(f"=== {harness_name.upper()} ({len(harness_config['cases'])} cases) ===")
        print(f"{'='*70}")

        control, treatment = await run_ab_test_harness(client, harness_name, harness_config)
        analysis = analyze_verifier_impact(treatment, control)

        all_controls[harness_name] = control
        all_treatments[harness_name] = treatment
        all_analyses[harness_name] = analysis

    elapsed = time.time() - t_start

    # =========================================================================
    # Print Reports
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"=== RESULTS ===")
    print(f"{'='*70}")

    for harness_name in harnesses_to_run:
        print_ab_report(
            harness_name,
            all_controls[harness_name],
            all_treatments[harness_name],
            all_analyses[harness_name],
        )

    # =========================================================================
    # Aggregate Verifier Analysis
    # =========================================================================
    total_revision_cases = sum(a["revisions_triggered"] for a in all_analyses.values())
    total_improved = sum(a["revision_improved"] for a in all_analyses.values())
    total_hurt = sum(a["revision_hurt"] for a in all_analyses.values())
    total_neutral = sum(a["revision_neutral"] for a in all_analyses.values())
    total_rejected = sum(a["rejections"] for a in all_analyses.values())
    total_approved_first = sum(a["approved_first_pass"] for a in all_analyses.values())

    all_revision_deltas = []
    for a in all_analyses.values():
        for rc in a["cases_with_revision"]:
            all_revision_deltas.append(rc["delta"])
    avg_revision_improvement = sum(all_revision_deltas) / len(all_revision_deltas) if all_revision_deltas else 0

    print(f"\n{'='*70}")
    print(f"=== VERIFIER ANALYSIS ===")
    print(f"{'='*70}")
    print(f"Total cases: {total_cases}")
    print(f"Approved first pass: {total_approved_first}/{total_cases} ({total_approved_first/total_cases*100:.0f}%)")
    print(f"Total revisions triggered: {total_revision_cases}/{total_cases} ({total_revision_cases/total_cases*100:.0f}%)")
    print(f"Total rejections: {total_rejected}/{total_cases} ({total_rejected/total_cases*100:.0f}%)")
    if total_revision_cases > 0:
        print(f"Revision improved score: {total_improved}/{total_revision_cases} ({total_improved/total_revision_cases*100:.0f}%)")
        print(f"Revision hurt score: {total_hurt}/{total_revision_cases} ({total_hurt/total_revision_cases*100:.0f}%)")
        print(f"Revision neutral: {total_neutral}/{total_revision_cases} ({total_neutral/total_revision_cases*100:.0f}%)")
    print(f"Avg score change per revision: {avg_revision_improvement:+.1f} pts")

    # Overall delta
    all_control_scores = []
    all_treatment_scores = []
    for harness_name in harnesses_to_run:
        for c in all_controls[harness_name].cases:
            all_control_scores.append(c.quality_score)
        for c in all_treatments[harness_name].cases:
            all_treatment_scores.append(c.quality_score)

    overall_control = sum(all_control_scores) / len(all_control_scores) if all_control_scores else 0
    overall_treatment = sum(all_treatment_scores) / len(all_treatment_scores) if all_treatment_scores else 0
    overall_delta = overall_treatment - overall_control

    print(f"\nOverall control avg: {overall_control:.1f}")
    print(f"Overall treatment avg: {overall_treatment:.1f}")
    print(f"Overall delta: {overall_delta:+.1f}")
    print(f"Elapsed time: {elapsed:.0f}s")

    # =========================================================================
    # Hypothesis H2 Verdict
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"=== HYPOTHESIS H2 VERDICT ===")
    print(f"{'='*70}")
    if overall_delta > 2.0:
        print(f"H2 CONFIRMED: Verifier nodes improved scores by {overall_delta:+.1f} pts")
    elif overall_delta > 0:
        print(f"H2 PARTIALLY CONFIRMED: Small improvement of {overall_delta:+.1f} pts (below 2pt threshold)")
    elif overall_delta == 0:
        print(f"H2 NOT CONFIRMED: No improvement detected")
    else:
        print(f"H2 REJECTED: Verifier nodes HURT scores by {overall_delta:.1f} pts")

    if total_revision_cases > 0 and total_improved > total_hurt:
        print(f"Verifier revisions had net positive impact: {total_improved} improved vs {total_hurt} hurt")
    elif total_revision_cases > 0:
        print(f"Verifier revisions had net negative impact: {total_improved} improved vs {total_hurt} hurt")

    # IT helpdesk ceiling check
    if "it_helpdesk" in all_treatments:
        it_treatment = all_treatments["it_helpdesk"].avg_score
        it_control = all_controls["it_helpdesk"].avg_score
        it_delta = it_treatment - it_control
        print(f"\nIT Helpdesk ceiling check:")
        print(f"  Control: {it_control:.1f} (baseline was 76-78)")
        print(f"  Treatment: {it_treatment:.1f}")
        print(f"  Delta: {it_delta:+.1f}")
        if it_treatment > 80:
            print(f"  CEILING BROKEN: IT helpdesk broke through 80 barrier!")
        else:
            print(f"  Ceiling holds: still below 80")

    # =========================================================================
    # Save results
    # =========================================================================
    results_data = {
        "experiment": "adversarial_verification_ab_test",
        "hypothesis": "H2",
        "model": MODEL,
        "judge_model": JUDGE_MODEL,
        "max_revisions": MAX_REVISIONS,
        "elapsed_seconds": round(elapsed),
        "overall": {
            "control_avg": round(overall_control, 1),
            "treatment_avg": round(overall_treatment, 1),
            "delta": round(overall_delta, 1),
            "total_cases": total_cases,
        },
        "verifier_analysis": {
            "approved_first_pass": total_approved_first,
            "revisions_triggered": total_revision_cases,
            "rejections": total_rejected,
            "revision_improved": total_improved,
            "revision_hurt": total_hurt,
            "revision_neutral": total_neutral,
            "avg_improvement_per_revision": round(avg_revision_improvement, 1),
        },
        "harnesses": {},
    }

    for harness_name in harnesses_to_run:
        ctrl = all_controls[harness_name]
        treat = all_treatments[harness_name]
        analysis = all_analyses[harness_name]

        results_data["harnesses"][harness_name] = {
            "control_avg": round(ctrl.avg_score, 1),
            "treatment_avg": round(treat.avg_score, 1),
            "delta": round(treat.avg_score - ctrl.avg_score, 1),
            "cases": len(ctrl.cases),
            "control_avg_length": round(ctrl.avg_response_length),
            "treatment_avg_length": round(treat.avg_response_length),
            "revisions": analysis["revisions_triggered"],
            "rejections": analysis["rejections"],
            "approved_first_pass": analysis["approved_first_pass"],
            "revision_details": analysis["cases_with_revision"],
            "per_case": [],
        }

        for i, (c_case, t_case) in enumerate(zip(ctrl.cases, treat.cases)):
            results_data["harnesses"][harness_name]["per_case"].append({
                "case_id": c_case.case_id,
                "query": c_case.query[:60],
                "expected_category": c_case.expected_category,
                "control_score": c_case.quality_score,
                "treatment_score": t_case.quality_score,
                "delta": t_case.quality_score - c_case.quality_score,
                "control_category": c_case.actual_category,
                "treatment_category": t_case.actual_category,
                "verifier_verdict": t_case.verification.verifier_verdict if t_case.verification else "N/A",
                "revision_count": t_case.verification.revision_count if t_case.verification else 0,
            })

    with open(RESULTS_FILE, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE.name}")


async def main():
    parser = argparse.ArgumentParser(description="Adversarial Verification A/B Test")
    parser.add_argument(
        "--harness",
        type=str,
        default=None,
        choices=list(HARNESSES.keys()),
        help="Run only this harness (default: all)",
    )
    args = parser.parse_args()
    await run_full_experiment(harness_filter=args.harness)


if __name__ == "__main__":
    asyncio.run(main())
