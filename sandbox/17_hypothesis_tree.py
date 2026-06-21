"""Prototype 17: Hypothesis-Tree Learning Loop with Held-Out Validation.

Arbor-inspired improvements to Graph Gardener's conservative learning loop:

  A. Held-out validation (70/30 split with seed=42):
     Training cases guide the optimizer, held-out cases decide promotions.
     Prevents overfitting to the test set.

  B. Parallel hypothesis branching (3 variants per iteration):
     - templates: Issue-specific templates (Sprint D approach)
     - knowledge: Domain knowledge injection (server names, DNS, URLs)
     - clarity: Instruction clarity and conciseness refinement
     All variants evaluated on training cases; winner validated on held-out.

  C. Insight memory:
     Growing list of what worked and what didn't, fed back into
     hypothesis generation so the optimizer accumulates wisdom.

  D. Full run: 10 iterations with all 3 improvements active.

Conservative optimizer still applies (stability zones, one-at-a-time, rollback).

Usage:
    python sandbox/17_hypothesis_tree.py                    # Run 10 iterations
    python sandbox/17_hypothesis_tree.py --iterations 5     # Run 5 iterations
    python sandbox/17_hypothesis_tree.py --reset            # Reset versioned YAMLs

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
import random
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

# -- Imports from the base multi-harness system --------------------------------
# The filename starts with a number, so we use importlib to load it.
import importlib.util as _ilu
import sys as _sys
_spec = _ilu.spec_from_file_location(
    "multi_harness",
    str(Path(__file__).parent / "11_multi_harness.py"),
)
_mh = _ilu.module_from_spec(_spec)
_sys.modules["multi_harness"] = _mh  # register so @dataclass can resolve module
_spec.loader.exec_module(_mh)

CUSTOMER_SUPPORT_CASES = _mh.CUSTOMER_SUPPORT_CASES
IT_HELPDESK_CASES = _mh.IT_HELPDESK_CASES
SALES_INQUIRY_CASES = _mh.SALES_INQUIRY_CASES
BASE_HARNESSES = _mh.HARNESSES
SANDBOX_DIR = _mh.SANDBOX_DIR
JUDGE_MODEL = _mh.JUDGE_MODEL
LLM_DELAY = _mh.LLM_DELAY
STABILITY_THRESHOLD = _mh.STABILITY_THRESHOLD
ROLLBACK_THRESHOLD = _mh.ROLLBACK_THRESHOLD
InteractionRecord = _mh.InteractionRecord
HarnessResult = _mh.HarnessResult
IterationResult = _mh.IterationResult
load_dag_config = _mh.load_dag_config
build_routing = _mh.build_routing
extract_categories_from_config = _mh.extract_categories_from_config
run_dag_query = _mh.run_dag_query
score_response = _mh.score_response
_single_judge_call = _mh._single_judge_call
_validate_dag_config = _mh._validate_dag_config
_build_agent_driven_context = _mh._build_agent_driven_context

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("hypothesis_tree")

SCORES_FILE = SANDBOX_DIR / "scores_hypothesis_tree.json"

# Hypothesis strategies
STRATEGIES = ["templates", "knowledge", "clarity"]


# =============================================================================
# 1. Held-Out Validation: 70/30 Split
# =============================================================================


def split_cases(cases: list[dict], seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Split cases into 70% training and 30% held-out with deterministic seed.

    Returns (train_cases, holdout_cases).
    """
    rng = random.Random(seed)
    shuffled = list(cases)
    rng.shuffle(shuffled)
    split_point = int(len(shuffled) * 0.7)
    train_cases = shuffled[:split_point]
    holdout_cases = shuffled[split_point:]
    return train_cases, holdout_cases


# Build harness definitions with train/holdout splits
HARNESSES = []
for h in BASE_HARNESSES:
    train, holdout = split_cases(h["cases"])
    HARNESSES.append({
        "name": h["name"],
        "yaml": h["yaml"],
        "cases": h["cases"],         # full set (for reference)
        "train_cases": train,         # 70% -- shown to optimizer
        "holdout_cases": holdout,     # 30% -- used for promotion decisions
    })


# =============================================================================
# 2. Insight Memory
# =============================================================================


@dataclass
class Insight:
    """Record of what worked or didn't in a hypothesis test."""
    iteration: int
    harness: str
    strategy: str       # "templates", "knowledge", "clarity"
    description: str    # What was tried
    train_delta: float  # Score change on training set
    holdout_delta: float  # Score change on held-out set
    accepted: bool      # Whether this hypothesis was promoted

    def __str__(self) -> str:
        status = "ACCEPTED" if self.accepted else "REJECTED"
        return (
            f"Iter {self.iteration}: \"{self.strategy}\" strategy on {self.harness}: "
            f"train {self.train_delta:+.1f}, holdout {self.holdout_delta:+.1f} ({status})"
        )


# Global insight memory
insight_memory: list[Insight] = []


def get_relevant_insights(harness_name: str, max_insights: int = 10) -> str:
    """Format relevant past insights for inclusion in hypothesis generation prompt."""
    relevant = [i for i in insight_memory if i.harness == harness_name]
    # Also include cross-harness insights that are general patterns
    general = [i for i in insight_memory if i.harness != harness_name and abs(i.train_delta) > 3.0]

    all_insights = relevant + general[-3:]  # Up to 3 cross-harness insights
    all_insights = all_insights[-max_insights:]  # Limit to most recent

    if not all_insights:
        return "No past insights available yet."

    lines = ["Past insights for this harness:"]
    for insight in all_insights:
        prefix = "" if insight.harness == harness_name else f"[from {insight.harness}] "
        lines.append(f"  - {prefix}{insight}")
    return "\n".join(lines)


# =============================================================================
# 3. Parallel Hypothesis Branching
# =============================================================================


STRATEGY_PROMPTS = {
    "templates": """You are an AI workflow optimizer specializing in ISSUE-SPECIFIC TEMPLATES.
Your strategy: Add step-by-step templates for specific sub-issues that the weak cases reveal.

For each weak case pattern, create a concrete template like:
  "For SCREEN FLICKERING: 1. Check refresh rate settings 2. Update display driver
   3. Test with external monitor 4. If persists, submit hardware ticket"

Focus on ACTIONABLE STEPS for specific issue types seen in the weak cases.
Each template should be a numbered, specific troubleshooting or response flow.""",

    "knowledge": """You are an AI workflow optimizer specializing in DOMAIN KNOWLEDGE INJECTION.
Your strategy: Add concrete, realistic details that make responses more actionable.

Inject knowledge like:
  - Server names: "VPN server: vpn.company.com, Backup: vpn2.company.com"
  - DNS IPs: "Primary DNS: 10.0.0.53, Secondary: 10.0.0.54"
  - URLs: "Self-service portal: https://ithelp.company.com/password-reset"
  - Tool names: "Use 'ipconfig /flushdns' on Windows, 'sudo dscacheutil -flushcache' on Mac"
  - Policy details: "Password must be 12+ chars with uppercase, number, and symbol"

Focus on CONCRETE DETAILS that the handler currently lacks.""",

    "clarity": """You are an AI workflow optimizer specializing in INSTRUCTION CLARITY.
Your strategy: Make handler instructions clearer and more focused to improve response quality.

Improvements include:
  - Remove ambiguous or redundant phrases
  - Add explicit quality expectations: "ALWAYS mention X when the user asks about Y"
  - Add explicit structure requirements: "Structure your response as: 1. Acknowledge 2. Diagnose 3. Steps 4. Escalation path"
  - Add explicit scope: "You handle X, Y, Z issues. For anything else, direct to general support."
  - Add response format hints: "Keep responses under 200 words. Use numbered steps."

Focus on making the instruction UNAMBIGUOUS about what a good response looks like.""",
}


async def generate_hypothesis_variant(
    client: genai.Client,
    current_config: dict,
    strategy: str,
    harness_result: HarnessResult,
    harness_name: str,
    insights_text: str,
    iteration: int,
) -> dict:
    """Generate one hypothesis variant using a specific strategy.

    Returns a modified copy of current_config.
    """
    strategy_prompt = STRATEGY_PROMPTS[strategy]
    valid_categories = extract_categories_from_config(current_config)

    # Build focused context (reuse the agent-driven context from base)
    focused_context = _build_agent_driven_context(
        harness_result, current_config, harness_name,
    )

    breakdown = harness_result.category_breakdown()

    analysis = f"""Benchmark Results for {harness_name} (Iteration {iteration}, TRAINING SET ONLY):
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

    prompt = f"""{strategy_prompt}

{analysis}

{insights_text}

IMPORTANT RULES:
1. You must NOT rewrite handler instructions from scratch
2. You must ONLY propose ADDITIONS to append to the existing handler instructions
3. Focus on the weakest category shown above
4. Use your specific strategy ({strategy}) to guide what kind of additions you propose

Propose 2-3 specific ADDITIONS to append to the weakest handler.
For each proposal, specify:
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

    # Now apply proposals to a copy of the config
    new_config = await _apply_proposals_to_config(
        client, current_config, proposals, iteration, harness_name, strategy,
    )
    return new_config


async def _apply_proposals_to_config(
    client: genai.Client,
    current_config: dict,
    proposals: list[str],
    iteration: int,
    harness_name: str,
    strategy: str,
) -> dict:
    """Apply proposals to a deep copy of the config (same logic as generate_improved_dag)."""
    proposals_text = "\n".join(proposals)

    # Ask the LLM to extract structured additions
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

        if "```" in extract_text:
            match = re.search(r"```(?:json)?\s*(.*?)```", extract_text, re.DOTALL)
            if match:
                extract_text = match.group(1).strip()

        json_match = re.search(r'\[.*\]', extract_text, re.DOTALL)
        if json_match:
            extract_text = json_match.group(0)

        additions = json.loads(extract_text)
        if not isinstance(additions, list):
            additions = [additions]

    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Failed to extract structured additions for %s/%s: %s. Using fallback.", harness_name, strategy, e)
        handler_ids = [n["id"] for n in current_config["nodes"] if n["id"] != "classify"]
        target_handler = handler_ids[0]
        for hid in handler_ids:
            if hid in proposals_text.lower() or hid.replace("handle_", "") in proposals_text.lower():
                target_handler = hid
                break
        additions = [{
            "node_id": target_handler,
            "append_text": f"\n\nADDITIONAL GUIDANCE ({strategy} strategy, iteration {iteration}):\n" + proposals_text[:1500],
        }]

    # Apply additions to a deep copy
    new_config = copy.deepcopy(current_config)
    new_config["version"] = f"3.0.{iteration}-{strategy}"

    nodes_by_id = {n["id"]: n for n in new_config["nodes"]}
    applied_count = 0

    for addition in additions:
        node_id = addition.get("node_id", "")
        append_text = addition.get("append_text", "")

        if not node_id or not append_text:
            continue

        target_node = nodes_by_id.get(node_id)
        if target_node is None:
            for nid, node in nodes_by_id.items():
                if node_id in nid or nid in node_id:
                    target_node = node
                    break

        if target_node is None or target_node["id"] == "classify":
            continue

        current_instruction = target_node.get("instruction", "")
        if len(current_instruction) + len(append_text) > 4000:
            append_text = append_text[:4000 - len(current_instruction)]

        target_node["instruction"] = current_instruction.rstrip() + "\n\n" + append_text.strip() + "\n"
        applied_count += 1

    if applied_count == 0:
        logger.warning("No additions applied for %s/%s. Returning copy of original.", harness_name, strategy)
        new_config["version"] = f"3.0.{iteration}-{strategy}"

    _validate_dag_config(new_config, harness_name)
    return new_config


async def generate_hypotheses(
    client: genai.Client,
    config: dict,
    harness_result: HarnessResult,
    harness_name: str,
    iteration: int,
) -> list[tuple[str, dict]]:
    """Generate 3 parallel hypothesis configs, one per strategy.

    Returns list of (strategy_name, config_dict) tuples.
    """
    insights_text = get_relevant_insights(harness_name)

    variants = []
    for strategy in STRATEGIES:
        try:
            variant_config = await generate_hypothesis_variant(
                client, config, strategy,
                harness_result, harness_name, insights_text, iteration,
            )
            variants.append((strategy, variant_config))
        except Exception as e:
            logger.warning("Failed to generate %s variant for %s: %s", strategy, harness_name, e)
            print(f"    WARNING: {strategy} variant generation failed: {e}")

    return variants


# =============================================================================
# 4. Benchmark Runner (supports train/holdout split)
# =============================================================================


async def run_harness_benchmark(
    client: genai.Client,
    config: dict,
    config_path: str,
    cases: list[dict],
    harness_name: str,
    iteration: int,
    label: str = "",
) -> HarnessResult:
    """Run benchmark cases for one harness and score them.

    label: optional prefix for logging (e.g., "train", "holdout", "hyp-A")
    """
    result = HarnessResult(
        harness_name=harness_name,
        iteration=iteration,
        config_path=config_path,
    )

    prefix = f"[{label}] " if label else ""

    for i, case in enumerate(cases):
        dag_result = await run_dag_query(client, config, case["query"])
        await asyncio.sleep(LLM_DELAY)

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
        print(f"    {prefix}Case {case['id']:2d}: \"{case['query'][:40]}...\" -> {dag_result['category']} ({cat_ok}) -> {score['total_score']}")

        await asyncio.sleep(LLM_DELAY)

    return result


# =============================================================================
# 5. Score Tracking
# =============================================================================


def save_scores(
    all_iterations: list[dict],
    insights: list[Insight],
) -> None:
    """Save score history and insights to scores_hypothesis_tree.json."""
    data = {
        "iterations": all_iterations,
        "insights": [
            {
                "iteration": i.iteration,
                "harness": i.harness,
                "strategy": i.strategy,
                "description": i.description,
                "train_delta": round(i.train_delta, 2),
                "holdout_delta": round(i.holdout_delta, 2),
                "accepted": i.accepted,
            }
            for i in insights
        ],
    }
    with open(SCORES_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nScores saved to {SCORES_FILE.name}")


# =============================================================================
# 6. Main Loop: Hypothesis-Tree Learning
# =============================================================================


async def run_hypothesis_tree_loop(iterations: int = 10) -> None:
    """Run the full hypothesis-tree learning loop.

    Each iteration:
      1. Evaluate all harnesses on training set
      2. Evaluate all harnesses on held-out set
      3. Find weakest harness (training scores)
      4. Generate 3 hypothesis variants for weakest
      5. Evaluate all 3 on training cases of weakest harness
      6. Pick winner (best training score)
      7. Validate winner on held-out cases
      8. Promote only if held-out improves
      9. Record insight
    """
    client = genai.Client(
        vertexai=True,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT", "alanblount-demo"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
    )

    global insight_memory
    insight_memory = []

    # Iteration tracking
    all_iteration_data: list[dict] = []
    promotion_log: list[dict] = []

    # Best-config tracking
    best_config_content: dict[str, str] = {}   # harness_name -> YAML string
    best_harness_train: dict[str, float] = {}  # harness_name -> best training score
    best_harness_holdout: dict[str, float] = {}  # harness_name -> best held-out score
    best_aggregate_train: float = 0.0
    best_aggregate_holdout: float = 0.0

    # Current config paths
    current_configs: dict[str, Path] = {}
    for h in HARNESSES:
        base_path = SANDBOX_DIR / h["yaml"]
        current_configs[h["name"]] = base_path
        with open(base_path) as f:
            best_config_content[h["name"]] = f.read()
        best_harness_train[h["name"]] = 0.0
        best_harness_holdout[h["name"]] = 0.0

    # Summary table header
    summary_rows: list[dict] = []

    for iteration in range(1, iterations + 1):
        print(f"\n{'='*70}")
        print(f"=== ITERATION {iteration} ===")
        print(f"{'='*70}")

        iter_data = {
            "iteration": iteration,
            "harnesses": {},
            "hypotheses_tested": 0,
            "winner_strategy": None,
        }

        # ---- Step 1: Evaluate all harnesses on TRAINING set ----
        train_results: dict[str, HarnessResult] = {}
        for harness in HARNESSES:
            hname = harness["name"]
            config_path = current_configs[hname]
            config = load_dag_config(config_path)
            dag_version = config.get("version", "?")

            print(f"\n  --- {hname} TRAINING ({config_path.name}, v{dag_version}) ---")
            print(f"  Running {len(harness['train_cases'])} training cases...")

            hr = await run_harness_benchmark(
                client, config, str(config_path),
                harness["train_cases"], hname, iteration, label="train",
            )
            train_results[hname] = hr

        # ---- Step 2: Evaluate all harnesses on HELD-OUT set ----
        holdout_results: dict[str, HarnessResult] = {}
        for harness in HARNESSES:
            hname = harness["name"]
            config_path = current_configs[hname]
            config = load_dag_config(config_path)

            print(f"\n  --- {hname} HELD-OUT ({config_path.name}) ---")
            print(f"  Running {len(harness['holdout_cases'])} held-out cases...")

            hr = await run_harness_benchmark(
                client, config, str(config_path),
                harness["holdout_cases"], hname, iteration, label="holdout",
            )
            holdout_results[hname] = hr

        # Compute aggregate scores
        train_agg = sum(
            hr.aggregate_score for hr in train_results.values()
        ) / len(train_results) if train_results else 0.0

        holdout_agg = sum(
            hr.aggregate_score for hr in holdout_results.values()
        ) / len(holdout_results) if holdout_results else 0.0

        # Print summary
        print(f"\n=== ITERATION {iteration} SCORES ===")
        print(f"Training scores: ", end="")
        train_parts = []
        for hname in [h["name"] for h in HARNESSES]:
            train_parts.append(f"{hname.split('_')[0].upper()[:2]}:{train_results[hname].aggregate_score:.1f}")
        print(f"{', '.join(train_parts)} (aggregate: {train_agg:.1f})")

        print(f"Held-out scores: ", end="")
        holdout_parts = []
        for hname in [h["name"] for h in HARNESSES]:
            holdout_parts.append(f"{hname.split('_')[0].upper()[:2]}:{holdout_results[hname].aggregate_score:.1f}")
        print(f"{', '.join(holdout_parts)} (aggregate: {holdout_agg:.1f})")

        # Store per-harness data
        for harness in HARNESSES:
            hname = harness["name"]
            iter_data["harnesses"][hname] = {
                "train_score": round(train_results[hname].aggregate_score, 2),
                "holdout_score": round(holdout_results[hname].aggregate_score, 2),
                "train_cases": len(harness["train_cases"]),
                "holdout_cases": len(harness["holdout_cases"]),
            }

        # ---- Update best scores ----
        for hname in [h["name"] for h in HARNESSES]:
            ts = train_results[hname].aggregate_score
            hs = holdout_results[hname].aggregate_score
            if ts > best_harness_train[hname]:
                best_harness_train[hname] = ts
            if hs > best_harness_holdout[hname]:
                best_harness_holdout[hname] = hs
                config_path = current_configs[hname]
                with open(config_path) as f:
                    best_config_content[hname] = f.read()

        if train_agg > best_aggregate_train:
            best_aggregate_train = train_agg
        if holdout_agg > best_aggregate_holdout:
            best_aggregate_holdout = holdout_agg

        # ---- Step 3: Rollback check ----
        rollback_triggered = False
        if iteration > 1 and holdout_agg < (best_aggregate_holdout - ROLLBACK_THRESHOLD):
            rollback_triggered = True
            print(f"\n*** ROLLBACK TRIGGERED: held-out {holdout_agg:.1f} below best {best_aggregate_holdout:.1f} "
                  f"(threshold: -{ROLLBACK_THRESHOLD}) ***")
            for harness in HARNESSES:
                hname = harness["name"]
                base_name = Path(harness["yaml"]).stem
                rollback_path = SANDBOX_DIR / f"{base_name}_ht_v{iteration + 1}.yaml"
                with open(rollback_path, "w") as f_out:
                    f_out.write(best_config_content[hname])
                current_configs[hname] = rollback_path
                print(f"  [{hname}] Rolled back to best config -> {rollback_path.name}")

        # ---- Step 4: Hypothesis generation (skip after last iter and rollback) ----
        winner_strategy = None
        hypotheses_tested = 0

        if iteration < iterations and not rollback_triggered:
            # Find weakest harness by training score gap
            harness_gaps = []
            for harness in HARNESSES:
                hname = harness["name"]
                score = train_results[hname].aggregate_score
                gap = best_harness_train[hname] - score
                harness_gaps.append((hname, score, gap))

            harness_gaps.sort(key=lambda x: -x[2])
            weakest_name, weakest_score, weakest_gap = harness_gaps[0]

            # If all are at their best (gap=0), pick lowest absolute score
            if weakest_gap <= 0:
                harness_gaps.sort(key=lambda x: x[1])  # lowest score first
                weakest_name, weakest_score, weakest_gap = harness_gaps[0]
                weakest_gap = 0.0  # treat as 0 gap since it is at its best

            print(f"\nWeakest: {weakest_name} (train: {weakest_score:.1f}, gap: {weakest_gap:.1f} from best)")

            # Stability zone check
            if weakest_gap <= STABILITY_THRESHOLD and weakest_gap > 0:
                print(f"All harnesses within stability zone ({STABILITY_THRESHOLD} pts). No modifications.")
                iter_data["winner_strategy"] = "STABLE"
            else:
                print(f"Generating 3 hypotheses for {weakest_name}...")

                # Load best config for this harness
                best_yaml = best_config_content[weakest_name]
                best_parsed = yaml.safe_load(best_yaml)
                best_config = best_parsed.get("dag", best_parsed)

                harness_obj = next(h for h in HARNESSES if h["name"] == weakest_name)
                train_hr = train_results[weakest_name]

                # Generate 3 hypothesis variants
                variants = await generate_hypotheses(
                    client, best_config, train_hr, weakest_name, iteration,
                )
                hypotheses_tested = len(variants)

                if not variants:
                    print("  No valid hypothesis variants generated. Skipping.")
                else:
                    # Evaluate each variant on the TRAINING cases of the weakest harness
                    variant_train_scores: list[tuple[str, float, dict]] = []

                    for strategy, variant_config in variants:
                        print(f"\n  Evaluating Hypothesis ({strategy}) on training set...")
                        variant_hr = await run_harness_benchmark(
                            client, variant_config, f"hypothesis-{strategy}",
                            harness_obj["train_cases"], weakest_name, iteration,
                            label=f"hyp-{strategy[0].upper()}",
                        )
                        variant_train_scores.append((strategy, variant_hr.aggregate_score, variant_config))
                        print(f"  Hypothesis ({strategy}): training score = {variant_hr.aggregate_score:.1f}")

                    # Pick winner: highest training score
                    variant_train_scores.sort(key=lambda x: -x[1])
                    winner_strategy, winner_train_score, winner_config = variant_train_scores[0]

                    # Validate winner on HELD-OUT cases
                    print(f"\n  Winner: {winner_strategy} (train={winner_train_score:.1f})")
                    print(f"  Validating on held-out set...")
                    holdout_hr = await run_harness_benchmark(
                        client, winner_config, f"winner-{winner_strategy}",
                        harness_obj["holdout_cases"], weakest_name, iteration,
                        label="holdout-val",
                    )
                    winner_holdout_score = holdout_hr.aggregate_score
                    current_holdout_score = holdout_results[weakest_name].aggregate_score

                    holdout_delta = winner_holdout_score - current_holdout_score
                    train_delta = winner_train_score - train_results[weakest_name].aggregate_score

                    # Print all hypothesis results
                    print(f"\n  Hypothesis results for {weakest_name}:")
                    for strategy, train_score, _ in variant_train_scores:
                        marker = " <- WINNER" if strategy == winner_strategy else ""
                        print(f"    Hypothesis ({strategy}): training={train_score:.1f}{marker}")
                    print(f"  Winner held-out: {winner_holdout_score:.1f} (current: {current_holdout_score:.1f}, delta: {holdout_delta:+.1f})")

                    # Promote only if held-out improves
                    accepted = holdout_delta > 0
                    if accepted:
                        base_name = Path(harness_obj["yaml"]).stem
                        new_config_path = SANDBOX_DIR / f"{base_name}_ht_v{iteration + 1}.yaml"
                        with open(new_config_path, "w") as f_out:
                            yaml.dump({"dag": winner_config}, f_out, default_flow_style=False, sort_keys=False)
                        current_configs[weakest_name] = new_config_path
                        print(f"  PROMOTING Hypothesis ({winner_strategy}). Wrote {new_config_path.name}")
                        print(f"  Insight: \"{winner_strategy} strategy {holdout_delta:+.1f}pts on {weakest_name}\"")
                    else:
                        print(f"  REJECTING winner: held-out did not improve ({holdout_delta:+.1f})")
                        print(f"  Insight: \"{winner_strategy} strategy overfit: train {train_delta:+.1f} but holdout {holdout_delta:+.1f}\"")

                    # Record insights for ALL variants (not just winner)
                    for strategy, train_score, _ in variant_train_scores:
                        is_winner = strategy == winner_strategy
                        s_train_delta = train_score - train_results[weakest_name].aggregate_score

                        insight = Insight(
                            iteration=iteration,
                            harness=weakest_name,
                            strategy=strategy,
                            description=f"{strategy} strategy on {weakest_name}",
                            train_delta=s_train_delta,
                            holdout_delta=holdout_delta if is_winner else 0.0,  # Only winner has holdout data
                            accepted=accepted and is_winner,
                        )
                        insight_memory.append(insight)

                    iter_data["winner_strategy"] = winner_strategy if accepted else f"{winner_strategy}(rejected)"
                    iter_data["hypotheses_tested"] = hypotheses_tested

                    # Store hypothesis scores
                    iter_data["hypothesis_details"] = {
                        "target_harness": weakest_name,
                        "variants": [
                            {"strategy": s, "train_score": round(ts, 2)}
                            for s, ts, _ in variant_train_scores
                        ],
                        "winner": winner_strategy,
                        "winner_holdout": round(winner_holdout_score, 2),
                        "current_holdout": round(current_holdout_score, 2),
                        "holdout_delta": round(holdout_delta, 2),
                        "accepted": accepted,
                    }

        # Build summary row
        summary_row = {
            "iteration": iteration,
            "train_agg": round(train_agg, 1),
            "holdout_agg": round(holdout_agg, 1),
            "hypotheses_tested": hypotheses_tested,
            "winner_strategy": iter_data.get("winner_strategy", "-"),
        }
        summary_rows.append(summary_row)

        iter_data["train_aggregate"] = round(train_agg, 2)
        iter_data["holdout_aggregate"] = round(holdout_agg, 2)
        all_iteration_data.append(iter_data)

        # Save after each iteration (crash-safe)
        save_scores(all_iteration_data, insight_memory)

    # =================================================================
    # Final Summary
    # =================================================================
    print(f"\n{'='*70}")
    print(f"=== HYPOTHESIS TREE: {iterations}-ITERATION SUMMARY ===")
    print(f"{'='*70}\n")

    # Summary table
    print(f"{'Iter':>4s} | {'Train Agg':>9s} | {'Held-Out Agg':>12s} | {'Hypotheses':>10s} | {'Winner Strategy':>15s}")
    print("-" * 62)
    for row in summary_rows:
        print(f"  {row['iteration']:2d} | {row['train_agg']:9.1f} | {row['holdout_agg']:12.1f} | "
              f"{row['hypotheses_tested']:10d} | {row['winner_strategy']:>15s}")

    # Per-harness detail table
    print(f"\nPer-harness scores (train / held-out):")
    hnames = [h["name"] for h in HARNESSES]
    hdr = f"{'Iter':>4s} |"
    for hname in hnames:
        hdr += f" {hname:^25s} |"
    print(hdr)
    print("-" * (7 + 28 * len(hnames)))
    for iter_data in all_iteration_data:
        row = f"  {iter_data['iteration']:2d} |"
        for hname in hnames:
            h = iter_data["harnesses"].get(hname, {})
            ts = h.get("train_score", 0)
            hs = h.get("holdout_score", 0)
            row += f" {ts:6.1f} / {hs:6.1f}           |"
        print(row)

    # Best scores
    print(f"\nBest scores achieved:")
    for hname in hnames:
        print(f"  {hname:20s}: train={best_harness_train[hname]:.1f}, holdout={best_harness_holdout[hname]:.1f}")
    print(f"  {'Best train agg':20s}: {best_aggregate_train:.1f}")
    print(f"  {'Best holdout agg':20s}: {best_aggregate_holdout:.1f}")

    # Strategy wins
    print(f"\nStrategy win counts:")
    strategy_wins = {}
    strategy_accepts = {}
    for insight in insight_memory:
        if insight.strategy not in strategy_wins:
            strategy_wins[insight.strategy] = 0
            strategy_accepts[insight.strategy] = 0
        # Count as "win" if this was the highest training score variant
        # (We recorded all variants; the winner is the one that was also tested on holdout)
        if insight.holdout_delta != 0.0 or insight.accepted:
            strategy_wins[insight.strategy] += 1
            if insight.accepted:
                strategy_accepts[insight.strategy] += 1

    for strategy in STRATEGIES:
        wins = strategy_wins.get(strategy, 0)
        accepts = strategy_accepts.get(strategy, 0)
        print(f"  {strategy:12s}: {wins} times as winner, {accepts} promoted")

    # Insight memory dump
    print(f"\nInsight Memory ({len(insight_memory)} entries):")
    for insight in insight_memory:
        print(f"  {insight}")

    # Train vs holdout tracking
    if len(summary_rows) >= 2:
        train_scores = [r["train_agg"] for r in summary_rows]
        holdout_scores = [r["holdout_agg"] for r in summary_rows]
        train_range = max(train_scores) - min(train_scores)
        holdout_range = max(holdout_scores) - min(holdout_scores)
        correlation_check = sum(
            1 for i in range(1, len(summary_rows))
            if (train_scores[i] - train_scores[i-1]) * (holdout_scores[i] - holdout_scores[i-1]) >= 0
        )
        tracking_pct = correlation_check / (len(summary_rows) - 1) * 100

        print(f"\nTrain vs Held-Out Tracking:")
        print(f"  Train range:   {min(train_scores):.1f} - {max(train_scores):.1f} ({train_range:.1f} pts)")
        print(f"  Holdout range: {min(holdout_scores):.1f} - {max(holdout_scores):.1f} ({holdout_range:.1f} pts)")
        print(f"  Direction agreement: {tracking_pct:.0f}% ({correlation_check}/{len(summary_rows)-1} iterations)")

    # Comparison with previous
    print(f"\nComparison with previous integrated loop (87.7 best):")
    print(f"  Previous best aggregate: 87.7")
    print(f"  This run best holdout:   {best_aggregate_holdout:.1f}")
    diff = best_aggregate_holdout - 87.7
    sign = "+" if diff >= 0 else ""
    print(f"  Delta: {sign}{diff:.1f}")

    # Total LLM calls estimate
    total_train_cases = sum(len(h["train_cases"]) for h in HARNESSES)
    total_holdout_cases = sum(len(h["holdout_cases"]) for h in HARNESSES)
    # Per iteration: train eval (3 harnesses) + holdout eval (3 harnesses) = (train_cases + holdout_cases) * 3 calls each
    # Plus hypotheses: 3 strategies * (2 LLM gen calls + train_eval) + 1 holdout_eval
    base_eval_calls = iterations * (total_train_cases + total_holdout_cases) * 3  # dag + 2x judge
    hypothesis_gen_calls = (iterations - 1) * 3 * 2  # 3 strategies * 2 LLM calls per strategy
    # Hypothesis eval calls: 3 variants * weakest_train_cases * 3 + winner_holdout_cases * 3
    # Approximate: weakest harness has ~5 train cases (70% of ~18 or ~13 of ~23)
    avg_train = total_train_cases // len(HARNESSES)
    avg_holdout = total_holdout_cases // len(HARNESSES)
    hypothesis_eval_calls = (iterations - 1) * (3 * avg_train * 3 + avg_holdout * 3)
    total_calls = base_eval_calls + hypothesis_gen_calls + hypothesis_eval_calls
    print(f"\nEstimated total LLM calls: ~{total_calls}")


# =============================================================================
# 7. Reset & CLI
# =============================================================================


def reset_all():
    """Remove all hypothesis-tree versioned configs and scores."""
    patterns = [
        "customer_support_adk_ht_v*.yaml",
        "it_helpdesk_ht_v*.yaml",
        "sales_inquiry_ht_v*.yaml",
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
        print("Reset: no hypothesis-tree configs or scores found.")
    print("Original DAGs unchanged:")
    for h in HARNESSES:
        print(f"  {h['yaml']}")


async def main():
    parser = argparse.ArgumentParser(
        description="Hypothesis-Tree Learning Loop with Held-Out Validation"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of benchmark iterations (default: 10)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Remove all hypothesis-tree configs and scores, then exit",
    )
    args = parser.parse_args()

    if args.reset:
        reset_all()
        return

    total_train = sum(len(h["train_cases"]) for h in HARNESSES)
    total_holdout = sum(len(h["holdout_cases"]) for h in HARNESSES)

    print("=" * 70)
    print("Hypothesis-Tree Learning Loop (Arbor-Inspired)")
    print("=" * 70)
    print(f"Harnesses: {len(HARNESSES)}")
    for h in HARNESSES:
        print(f"  {h['name']:20s}: {len(h['train_cases'])} train + {len(h['holdout_cases'])} holdout = {len(h['cases'])} total ({h['yaml']})")
    print(f"Total per iteration: {total_train} train + {total_holdout} holdout = {total_train + total_holdout}")
    print(f"Iterations: {args.iterations}")
    print(f"Judge model: {JUDGE_MODEL}")
    print(f"Strategies: {', '.join(STRATEGIES)}")
    print(f"Hypothesis variants per iteration: 3")
    print(f"Stability threshold: {STABILITY_THRESHOLD} pts")
    print(f"Rollback threshold: {ROLLBACK_THRESHOLD} pts")
    print(f"LLM delay: {LLM_DELAY}s")
    print(f"Split: 70/30 with seed=42")

    await run_hypothesis_tree_loop(args.iterations)


if __name__ == "__main__":
    asyncio.run(main())
