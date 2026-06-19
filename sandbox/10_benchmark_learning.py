"""Prototype 10: Benchmark + Learning Loop for Customer Support DAG.

Runs the customer support DAG through a benchmark of test cases,
scores responses with an LLM-as-judge, then uses the TaxonomyTracker
pattern to propose DAG improvements. Repeats for multiple iterations,
tracking score improvement.

Usage:
    python sandbox/10_benchmark_learning.py                  # Run 3 iterations
    python sandbox/10_benchmark_learning.py --iterations 5   # Run 5 iterations
    python sandbox/10_benchmark_learning.py --reset           # Reset to v1 DAG

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
logger = logging.getLogger("benchmark")

SANDBOX_DIR = Path(__file__).parent
ORIGINAL_DAG = SANDBOX_DIR / "customer_support_adk.yaml"
JUDGE_MODEL = "gemini-2.5-flash"
# The DAG's own nodes use whatever model the YAML specifies


# =============================================================================
# 1. Benchmark Test Cases
# =============================================================================

BENCHMARK_CASES = [
    # --- Billing (8 cases) ---
    {
        "id": 1,
        "query": "I was charged twice for my subscription",
        "expected_category": "billing",
        "quality_criteria": "Should acknowledge the double charge, mention investigation or refund process",
    },
    {
        "id": 2,
        "query": "Can I get a refund for last month?",
        "expected_category": "billing",
        "quality_criteria": "Should explain the refund policy or process for requesting a refund",
    },
    {
        "id": 3,
        "query": "Why is my bill higher than usual this month?",
        "expected_category": "billing",
        "quality_criteria": "Should suggest common reasons (plan change, overage) and offer to review the account",
    },
    {
        "id": 4,
        "query": "I need to update my credit card on file",
        "expected_category": "billing",
        "quality_criteria": "Should provide steps to update payment method or direct to account settings",
    },
    {
        "id": 5,
        "query": "How do I cancel my subscription?",
        "expected_category": "billing",
        "quality_criteria": "Should explain cancellation process; may mention retention offers",
    },
    {
        "id": 6,
        "query": "I see an unauthorized charge from your company",
        "expected_category": "billing",
        "quality_criteria": "Should take the concern seriously, suggest investigation, mention fraud protection",
    },
    {
        "id": 7,
        "query": "When is my next billing date?",
        "expected_category": "billing",
        "quality_criteria": "Should explain how to find billing date or offer to look it up",
    },
    {
        "id": 8,
        "query": "Do you offer any discounts for annual plans?",
        "expected_category": "billing",
        "quality_criteria": "Should mention available discount options or direct to pricing page",
    },
    # --- Technical (8 cases) ---
    {
        "id": 9,
        "query": "My internet keeps dropping every 30 minutes",
        "expected_category": "technical",
        "quality_criteria": "Should ask about router/modem or suggest troubleshooting steps like restarting equipment",
    },
    {
        "id": 10,
        "query": "The app crashes when I try to open settings",
        "expected_category": "technical",
        "quality_criteria": "Should suggest clearing cache, reinstalling, or checking for updates",
    },
    {
        "id": 11,
        "query": "I can't log into my account, it says invalid password",
        "expected_category": "technical",
        "quality_criteria": "Should suggest password reset process and check for account lock",
    },
    {
        "id": 12,
        "query": "My download speeds are extremely slow, only getting 2 Mbps",
        "expected_category": "technical",
        "quality_criteria": "Should ask about connection type, suggest speed test, check for interference",
    },
    {
        "id": 13,
        "query": "The website keeps showing a 404 error on the dashboard page",
        "expected_category": "technical",
        "quality_criteria": "Should suggest clearing browser cache, trying different browser, or report known issue",
    },
    {
        "id": 14,
        "query": "My smart TV can't connect to your streaming service anymore",
        "expected_category": "technical",
        "quality_criteria": "Should suggest checking TV firmware, reinstalling app, verifying network connection",
    },
    {
        "id": 15,
        "query": "Email notifications are not working, I'm not receiving any alerts",
        "expected_category": "technical",
        "quality_criteria": "Should suggest checking notification settings, spam folder, and email verification",
    },
    {
        "id": 16,
        "query": "Two-factor authentication is not sending the verification code",
        "expected_category": "technical",
        "quality_criteria": "Should suggest checking phone number, trying alternative methods, checking SMS blockers",
    },
    # --- General (7 cases) ---
    {
        "id": 17,
        "query": "What are your business hours?",
        "expected_category": "general",
        "quality_criteria": "Should provide specific hours or direct to a page with hours information",
    },
    {
        "id": 18,
        "query": "How do I contact customer support by phone?",
        "expected_category": "general",
        "quality_criteria": "Should provide a phone number or explain how to find contact information",
    },
    {
        "id": 19,
        "query": "Do you have a referral program?",
        "expected_category": "general",
        "quality_criteria": "Should explain whether a referral program exists and how to participate",
    },
    {
        "id": 20,
        "query": "I'd like to provide feedback about your service",
        "expected_category": "general",
        "quality_criteria": "Should welcome feedback and explain how to submit it (survey, email, form)",
    },
    {
        "id": 21,
        "query": "What services do you offer for small businesses?",
        "expected_category": "general",
        "quality_criteria": "Should describe business offerings or direct to business solutions page",
    },
    {
        "id": 22,
        "query": "Is there a mobile app available?",
        "expected_category": "general",
        "quality_criteria": "Should confirm app availability and mention platforms (iOS/Android) or download links",
    },
    {
        "id": 23,
        "query": "I want to upgrade my current plan",
        "expected_category": "general",
        "quality_criteria": "Should explain upgrade options and how to change plans",
    },
]


# =============================================================================
# 2. DAG Executor (direct mode, no Temporal)
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


async def run_dag_query(
    client: genai.Client,
    config: dict,
    query: str,
) -> dict:
    """Execute a single query through the DAG and return detailed results.

    Returns dict with keys: category, response, nodes_visited, timings
    """
    nodes_by_id = {n["id"]: n for n in config["nodes"]}
    routing = build_routing(config)
    default_model = config.get("default_model", "gemini-2.5-flash")

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
    for cat in ["billing", "technical", "general"]:
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
# 3. LLM-as-Judge Scoring
# =============================================================================


async def score_response(
    client: genai.Client,
    query: str,
    expected_category: str,
    actual_category: str,
    response: str,
    quality_criteria: str,
) -> dict:
    """Score a response using LLM-as-judge.

    Returns dict with: total_score, category_score, quality_score, helpfulness_score, reasoning
    """
    # Category correctness: deterministic (40 points)
    category_score = 40 if actual_category == expected_category else 0

    # Quality + helpfulness scored by LLM (60 points total)
    judge_prompt = f"""You are evaluating a customer support response. Score it on two dimensions.

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
# 4. Interaction Records & Taxonomy Tracker (from prototype 07)
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
class BenchmarkResult:
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


# =============================================================================
# 5. Learning Loop: Analyze Results & Propose DAG Improvements
# =============================================================================


async def analyze_and_propose(
    client: genai.Client,
    result: BenchmarkResult,
    current_config: dict,
) -> list[str]:
    """Analyze benchmark results and propose DAG improvements.

    Returns a list of improvement proposals as text.
    """
    # Build analysis summary
    breakdown = result.category_breakdown()
    weak_cases = result.weakest_cases(5)

    analysis = f"""Benchmark Results (Iteration {result.iteration}):
- Aggregate Score: {result.aggregate_score:.1f}/100
- Category Accuracy: {result.category_accuracy:.0%}
- Quality Score: {result.avg_quality:.1f}/40
- Helpfulness Score: {result.avg_helpfulness:.1f}/20

Category Breakdown:
"""
    for cat, info in breakdown.items():
        analysis += f"  {cat}: avg={info['avg_score']:.1f}, accuracy={info['accuracy']:.0%} ({info['total']} cases)\n"

    analysis += "\nWeakest Cases:\n"
    for r in weak_cases:
        analysis += f"  Case {r.case_id}: \"{r.query[:50]}\" score={r.score['total_score']}, "
        analysis += f"expected={r.expected_category}, got={r.actual_category}\n"
        analysis += f"    Judge: {r.score.get('reasoning', '')[:100]}\n"

    prompt = f"""You are an AI workflow optimizer. Analyze these customer support DAG benchmark results
and propose specific, actionable improvements to the DAG configuration.

{analysis}

Current DAG config (YAML nodes):
{yaml.dump({'nodes': current_config['nodes']}, default_flow_style=False)}

Propose 2-4 specific improvements. Focus on:
1. Improving the classifier instruction to reduce misclassifications
2. Making handler instructions more specific to address quality gaps
3. Adding sub-handlers for complex cases if a category scores poorly
4. Improving response quality criteria in handler instructions

For each proposal, explain:
- What to change
- Why (based on the data)
- Expected impact

Reply with a numbered list of proposals. Be specific about instruction text changes."""

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
) -> dict:
    """Generate an improved DAG YAML config based on proposals.

    Returns the new config dict.
    """
    current_yaml = yaml.dump({"dag": current_config}, default_flow_style=False)

    prompt = f"""You are a workflow configuration generator. Apply the following improvement
proposals to the customer support DAG configuration.

CURRENT CONFIG:
```yaml
{current_yaml}
```

IMPROVEMENT PROPOSALS:
{chr(10).join(proposals)}

Generate the complete updated YAML config with these changes applied.
Rules:
- Keep all existing node IDs unless renaming is necessary
- You may add new nodes (sub-handlers, validators, etc.)
- Update instructions to be more specific and detailed based on proposals
- Keep the same edge structure unless adding new nodes/routes
- Keep model as gemini-2.5-flash for all nodes
- Update the version to "2.0.{iteration}"
- Keep nodes of type 'agent' only (no a2a or mcp nodes)
- IMPORTANT: Keep the classify node's instruction format so it outputs
  ONLY one of: billing, technical, general
- All edges from classify must use conditions: billing, technical, general

Reply with ONLY the YAML (no markdown code blocks, no explanation).
Start with "dag:" on the first line."""

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
        logger.error("Failed to parse generated YAML: %s", e)
        logger.error("Raw YAML:\n%s", yaml_text[:500])
        raise


# =============================================================================
# 6. Main Benchmark Runner
# =============================================================================


async def run_benchmark_iteration(
    client: genai.Client,
    config: dict,
    config_path: str,
    iteration: int,
) -> BenchmarkResult:
    """Run all benchmark cases through a DAG config and score them."""
    result = BenchmarkResult(iteration=iteration, config_path=config_path)
    total = len(BENCHMARK_CASES)

    for i, case in enumerate(BENCHMARK_CASES):
        # Run query through DAG
        dag_result = await run_dag_query(client, config, case["query"])

        # Score the response
        score = await score_response(
            client,
            query=case["query"],
            expected_category=case["expected_category"],
            actual_category=dag_result["category"],
            response=dag_result["response"],
            quality_criteria=case["quality_criteria"],
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

        # Category match indicator
        cat_ok = "correct" if dag_result["category"] == case["expected_category"] else f"WRONG:{dag_result['category']}"
        print(f"  Case {case['id']:2d}: \"{case['query'][:40]}...\" -> {dag_result['category']} ({cat_ok}) -> Score: {score['total_score']}")

    return result


async def run_learning_loop(iterations: int = 3) -> None:
    """Run the full benchmark + learning loop."""
    client = genai.Client(
        vertexai=True,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT", "alanblount-demo"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
    )

    all_results: list[BenchmarkResult] = []
    current_config_path = ORIGINAL_DAG

    for iteration in range(1, iterations + 1):
        config = load_dag_config(current_config_path)
        dag_name = config.get("name", "customer_support_adk")
        dag_version = config.get("version", "?")

        print(f"\n{'='*60}")
        print(f"=== ITERATION {iteration} ({Path(current_config_path).name}, v{dag_version}) ===")
        print(f"{'='*60}")
        print(f"Running {len(BENCHMARK_CASES)} test cases...\n")

        # Run benchmark
        result = await run_benchmark_iteration(client, config, str(current_config_path), iteration)
        all_results.append(result)

        # Print iteration summary
        improvement = ""
        if len(all_results) > 1:
            delta = result.aggregate_score - all_results[-2].aggregate_score
            improvement = f" ({'+' if delta >= 0 else ''}{delta:.1f})"
        print(f"\n  Aggregate Score: {result.aggregate_score:.1f}/100{improvement}")
        print(f"  Category Accuracy: {result.category_accuracy:.0%}")
        print(f"  Quality: {result.avg_quality:.1f}/40  |  Helpfulness: {result.avg_helpfulness:.1f}/20")

        # Category breakdown
        breakdown = result.category_breakdown()
        for cat, info in sorted(breakdown.items()):
            print(f"    {cat:12s}: avg={info['avg_score']:.1f}, accuracy={info['accuracy']:.0%}")

        # Learning loop (skip after last iteration)
        if iteration < iterations:
            print(f"\nLearning loop analyzing results...")

            proposals = await analyze_and_propose(client, result, config)
            for i, prop in enumerate(proposals, 1):
                # Print first line of each proposal
                first_line = prop.split("\n")[0].strip()
                print(f"  Proposal {i}: {first_line[:80]}")

            # Generate improved DAG
            next_version = iteration + 1
            new_config = await generate_improved_dag(client, config, proposals, next_version)

            # Validate the new config has required structure
            if "nodes" not in new_config or "edges" not in new_config:
                print("  WARNING: Generated config missing nodes/edges, keeping current config")
                continue

            # Ensure we have valid routing (classify -> handlers)
            node_ids = {n["id"] for n in new_config["nodes"]}
            has_classify = any("classify" in n["id"] for n in new_config["nodes"])
            if not has_classify:
                print("  WARNING: Generated config missing classify node, keeping current config")
                continue

            # Write the new config
            new_config_path = SANDBOX_DIR / f"customer_support_adk_v{next_version}.yaml"
            with open(new_config_path, "w") as f:
                yaml.dump({"dag": new_config}, f, default_flow_style=False, sort_keys=False)
            print(f"\n  Wrote {new_config_path.name}")

            current_config_path = new_config_path

    # =================================================================
    # Final Summary
    # =================================================================
    print(f"\n{'='*60}")
    print("=== SUMMARY ===")
    print(f"{'='*60}")

    scores_line = " -> ".join(
        f"v{i+1}: {r.aggregate_score:.1f}" for i, r in enumerate(all_results)
    )
    print(f"Scores: {scores_line}")

    if len(all_results) > 1:
        total_improvement = all_results[-1].aggregate_score - all_results[0].aggregate_score
        print(f"Total improvement: {'+' if total_improvement >= 0 else ''}{total_improvement:.1f} points over {len(all_results)} iterations")

    print(f"\nCategory accuracy trend:")
    for i, r in enumerate(all_results):
        print(f"  v{i+1}: {r.category_accuracy:.0%}")

    print(f"\nWeakest cases in final iteration:")
    final = all_results[-1]
    for r in final.weakest_cases(3):
        print(f"  Case {r.case_id}: \"{r.query[:50]}\" score={r.score['total_score']} ({r.score.get('reasoning', '')[:60]})")


def reset_dag():
    """Remove all versioned DAG configs, keeping only the original."""
    removed = []
    for f in SANDBOX_DIR.glob("customer_support_adk_v*.yaml"):
        f.unlink()
        removed.append(f.name)
    if removed:
        print(f"Reset: removed {', '.join(removed)}")
    else:
        print("Reset: no versioned configs found, nothing to remove.")
    print(f"Original DAG: {ORIGINAL_DAG.name} (unchanged)")


# =============================================================================
# 7. CLI Entry Point
# =============================================================================


async def main():
    parser = argparse.ArgumentParser(
        description="Benchmark + Learning Loop for Customer Support DAG"
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
        help="Remove all versioned DAG configs and exit",
    )
    args = parser.parse_args()

    if args.reset:
        reset_dag()
        return

    print("=" * 60)
    print("Customer Support DAG - Benchmark + Learning Loop")
    print("=" * 60)
    print(f"Test cases: {len(BENCHMARK_CASES)}")
    print(f"Iterations: {args.iterations}")
    print(f"Judge model: {JUDGE_MODEL}")
    print(f"Original DAG: {ORIGINAL_DAG.name}")

    await run_learning_loop(args.iterations)


if __name__ == "__main__":
    asyncio.run(main())
