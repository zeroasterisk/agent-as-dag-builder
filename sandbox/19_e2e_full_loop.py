"""Prototype 19: E2E Full Loop -- score -> propose -> validate -> canary -> promote

Proves the core DAG Builder thesis end-to-end on ONE domain (customer_support)
using real LLM calls, real scoring, and real validation gates. No mocks.

===========================================================================
WHAT THIS SCRIPT DOES (5-phase pipeline)
===========================================================================

Phase 1 -- SCORE
    Load the customer_support DAG config (customer_support_adk.yaml).
    Run it against 10 training test cases.
    Score each response with LLM-as-judge (category correctness + quality
    + helpfulness, 100-point scale, 2x averaged at temperature=0).
    Output: baseline scores, weak-case analysis, category breakdown.

Phase 2 -- PROPOSE
    Feed the scoring results (weak cases, category gaps, handler instructions)
    to the LLM optimizer. It proposes concrete, additive mutations to the DAG
    config -- appending issue-specific templates, domain knowledge, or quality
    hints to the weakest handler's instruction.
    The mutation is applied programmatically (deep copy + append), NOT by
    asking the LLM to regenerate the full YAML.
    Output: a mutated DAG config dict (in memory).

Phase 3 -- VALIDATE
    Run the mutated DAG against a held-out test set (8 cases, disjoint from
    the training set). Compare aggregate score against the baseline run of
    the same held-out cases through the ORIGINAL config.
    Gate: proceed only if the mutated config scores within VALIDATION_TOLERANCE
    (1.0 pts) of the original on held-out cases. This tolerance accounts for
    inherent LLM-as-judge scoring variance (~1-2 pts even with temp=0 + 2x avg).
    Output: pass / fail + score delta.

Phase 4 -- CANARY
    If validation passed, run the mutated DAG against 5 additional "canary"
    cases (also disjoint). These are the final safety net -- cases the
    optimizer has never seen.
    Gate: canary aggregate must be >= 50 (above floor) AND must not regress
    more than 5 points vs the original config on the same canary cases.
    Output: pass / fail + canary scores.

Phase 5 -- PROMOTE
    If canary passed, write the mutated config as the promoted version
    (customer_support_adk_e2e_promoted.yaml). Log what changed and whether
    the loop produced a measurably better config.

===========================================================================
ENVIRONMENT
===========================================================================

    GOOGLE_GENAI_USE_VERTEXAI=1
    GOOGLE_CLOUD_PROJECT=alanblount-demo
    GOOGLE_CLOUD_LOCATION=global

===========================================================================
USAGE
===========================================================================

    python sandbox/19_e2e_full_loop.py
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# -- Environment defaults (Vertex AI) ----------------------------------------
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "1")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "alanblount-demo")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")

from google import genai

# -- Import shared infrastructure from 11_multi_harness.py --------------------
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "multi_harness",
    str(Path(__file__).parent / "11_multi_harness.py"),
)
_mh = _ilu.module_from_spec(_spec)
sys.modules["multi_harness"] = _mh
_spec.loader.exec_module(_mh)

# Shared functions
load_dag_config = _mh.load_dag_config
build_routing = _mh.build_routing
extract_categories_from_config = _mh.extract_categories_from_config
run_dag_query = _mh.run_dag_query
score_response = _mh.score_response
_validate_dag_config = _mh._validate_dag_config
_build_agent_driven_context = _mh._build_agent_driven_context

# Shared data structures
InteractionRecord = _mh.InteractionRecord
HarnessResult = _mh.HarnessResult

# Shared constants
SANDBOX_DIR = _mh.SANDBOX_DIR
JUDGE_MODEL = _mh.JUDGE_MODEL
LLM_DELAY = _mh.LLM_DELAY

# Customer support test cases (all 23)
ALL_CASES = _mh.CUSTOMER_SUPPORT_CASES

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("e2e_full_loop")

# -- Paths --------------------------------------------------------------------
ORIGINAL_DAG = SANDBOX_DIR / "customer_support_adk.yaml"
PROMOTED_DAG = SANDBOX_DIR / "customer_support_adk_e2e_promoted.yaml"
RESULTS_FILE = SANDBOX_DIR / "scores_e2e_full_loop.json"

HARNESS_NAME = "customer_support"

# Validation gate tolerance: allow this much regression before failing.
# LLM-as-judge scoring has inherent variance (~1-2 pts) even with temperature=0
# and 2x averaging. Prototype 18 uses PROMOTION_TOLERANCE = 1.0 for the same reason.
VALIDATION_TOLERANCE = 1.0  # points


# =============================================================================
# 1. Test Case Splits (deterministic, disjoint)
# =============================================================================

# 10 training cases (used in SCORE phase -- optimizer sees these)
TRAIN_CASES = ALL_CASES[:10]  # cases 1-10

# 8 held-out cases (used in VALIDATE phase -- optimizer never sees these)
HOLDOUT_CASES = ALL_CASES[10:18]  # cases 11-18

# 5 canary cases (used in CANARY phase -- final safety net)
CANARY_CASES = ALL_CASES[18:23]  # cases 19-23

assert len(TRAIN_CASES) == 10, f"Expected 10 training cases, got {len(TRAIN_CASES)}"
assert len(HOLDOUT_CASES) == 8, f"Expected 8 holdout cases, got {len(HOLDOUT_CASES)}"
assert len(CANARY_CASES) == 5, f"Expected 5 canary cases, got {len(CANARY_CASES)}"
assert len(TRAIN_CASES) + len(HOLDOUT_CASES) + len(CANARY_CASES) == len(ALL_CASES)


# =============================================================================
# 2. Benchmark Runner (reusable across phases)
# =============================================================================


async def run_benchmark(
    client: genai.Client,
    config: dict,
    cases: list[dict],
    label: str,
) -> HarnessResult:
    """Run benchmark cases through a DAG config and score them.

    Returns a HarnessResult with all interaction records.
    """
    result = HarnessResult(
        harness_name=HARNESS_NAME,
        iteration=0,
        config_path=label,
    )

    for case in cases:
        dag_result = await run_dag_query(client, config, case["query"])
        await asyncio.sleep(LLM_DELAY)

        score = await score_response(
            client,
            query=case["query"],
            expected_category=case["expected_category"],
            actual_category=dag_result["category"],
            response=dag_result["response"],
            quality_criteria=case["quality_criteria"],
            harness_name=HARNESS_NAME,
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

        cat_ok = (
            "correct"
            if dag_result["category"] == case["expected_category"]
            else f"WRONG:{dag_result['category']}"
        )
        print(
            f"    [{label}] Case {case['id']:2d}: "
            f'"{case["query"][:45]}..." '
            f"-> {dag_result['category']} ({cat_ok}) "
            f"-> {score['total_score']}"
        )

        await asyncio.sleep(LLM_DELAY)

    return result


def print_result_summary(result: HarnessResult, label: str) -> None:
    """Print a summary of benchmark results."""
    print(f"\n  {label} Summary:")
    print(f"    Aggregate Score: {result.aggregate_score:.1f}/100")
    print(f"    Category Accuracy: {result.category_accuracy:.0%}")
    print(f"    Quality: {result.avg_quality:.1f}/30")
    print(f"    Helpfulness: {result.avg_helpfulness:.1f}/20")

    breakdown = result.category_breakdown()
    for cat, info in sorted(breakdown.items()):
        print(
            f"    {cat:12s}: avg={info['avg_score']:.1f}, "
            f"accuracy={info['accuracy']:.0%} ({info['total']} cases)"
        )


# =============================================================================
# 3. PROPOSE Phase -- Analyze + Generate Mutation
# =============================================================================


async def propose_mutation(
    client: genai.Client,
    config: dict,
    train_result: HarnessResult,
) -> dict:
    """Analyze training scores and propose a DAG config mutation.

    Returns a new config dict with additive instruction improvements.
    """
    # Build focused context for the optimizer
    focused_context = _build_agent_driven_context(
        train_result, config, HARNESS_NAME
    )

    breakdown = train_result.category_breakdown()
    valid_categories = extract_categories_from_config(config)

    analysis = f"""E2E Full Loop -- SCORE Phase Results (customer_support):
- Aggregate Score: {train_result.aggregate_score:.1f}/100
- Category Accuracy: {train_result.category_accuracy:.0%}
- Quality Score: {train_result.avg_quality:.1f}/30
- Helpfulness Score: {train_result.avg_helpfulness:.1f}/20

Valid Categories: {', '.join(valid_categories)}

Category Breakdown:
"""
    for cat, info in breakdown.items():
        analysis += (
            f"  {cat}: avg={info['avg_score']:.1f}, "
            f"accuracy={info['accuracy']:.0%} ({info['total']} cases)\n"
        )

    analysis += f"\n{focused_context}"

    # Step 1: Generate proposals
    prompt = f"""You are an AI workflow optimizer. Analyze these customer support DAG benchmark
results and propose specific, additive improvements.

{analysis}

IMPORTANT RULES:
1. You must NOT rewrite handler instructions from scratch.
2. You must ONLY propose ADDITIONS to append to the existing handler instructions.
3. Focus on the weakest category shown above.
4. Each addition should be one of:
   - Issue-specific template: step-by-step response flow for a specific sub-issue
   - Domain knowledge injection: concrete details (URLs, phone numbers, specific policies)
   - Quality criteria hint: instruction that addresses what the judge is looking for

Propose 2-3 specific ADDITIONS to append to the weakest handler.
For each proposal, specify:
- The exact text to APPEND to the handler instruction
- Which handler node it applies to (e.g., handle_billing, handle_technical, handle_general)
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

    print(f"\n  Proposals generated: {len(proposals)}")
    for i, prop in enumerate(proposals, 1):
        first_line = prop.split("\n")[0].strip()
        print(f"    Proposal {i}: {first_line[:80]}")

    # Step 2: Extract structured additions from proposals
    new_config = await _apply_proposals(
        client, config, proposals
    )
    return new_config


async def _apply_proposals(
    client: genai.Client,
    config: dict,
    proposals: list[str],
) -> dict:
    """Apply proposals to a deep copy of the config using structured extraction."""
    proposals_text = "\n".join(proposals)

    extract_prompt = f"""You are a precise text extractor. Given the improvement proposals below,
extract the text that should be APPENDED to handler node instructions.

PROPOSALS:
{proposals_text}

CURRENT HANDLER NODE IDs: {', '.join(n['id'] for n in config['nodes'] if n['id'] != 'classify')}

For each proposal, output a JSON object with:
- "node_id": which handler node to append to (must be one of the node IDs above)
- "append_text": the exact text to append to that node's instruction

Output a JSON array of these objects. No markdown, no explanation.
Example: [{{"node_id": "handle_billing", "append_text": "\\nFor refund requests:\\n1. Verify the charge..."}}]"""

    await asyncio.sleep(LLM_DELAY)
    try:
        resp = client.models.generate_content(
            model=JUDGE_MODEL,
            contents=[{"role": "user", "parts": [{"text": extract_prompt}]}],
            config={"temperature": 0.0},
        )
        extract_text = resp.candidates[0].content.parts[0].text.strip()

        # Parse JSON -- handle markdown code blocks
        if "```" in extract_text:
            match = re.search(r"```(?:json)?\s*(.*?)```", extract_text, re.DOTALL)
            if match:
                extract_text = match.group(1).strip()

        json_match = re.search(r"\[.*\]", extract_text, re.DOTALL)
        if json_match:
            extract_text = json_match.group(0)

        additions = json.loads(extract_text)
        if not isinstance(additions, list):
            additions = [additions]

    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Failed to extract structured additions: %s. Using fallback.", e)
        handler_ids = [
            n["id"] for n in config["nodes"] if n["id"] != "classify"
        ]
        target_handler = handler_ids[0]
        for hid in handler_ids:
            if (
                hid in proposals_text.lower()
                or hid.replace("handle_", "") in proposals_text.lower()
            ):
                target_handler = hid
                break
        additions = [
            {
                "node_id": target_handler,
                "append_text": (
                    "\n\nADDITIONAL GUIDANCE (from E2E optimization):\n"
                    + proposals_text[:1500]
                ),
            }
        ]

    # Apply additions to a deep copy
    new_config = copy.deepcopy(config)
    new_config["version"] = "e2e-proposed"

    nodes_by_id = {n["id"]: n for n in new_config["nodes"]}
    applied_count = 0

    for addition in additions:
        node_id = addition.get("node_id", "")
        append_text = addition.get("append_text", "")

        if not node_id or not append_text:
            continue

        target_node = nodes_by_id.get(node_id)
        if target_node is None:
            # Fuzzy match
            for nid, node in nodes_by_id.items():
                if node_id in nid or nid in node_id:
                    target_node = node
                    break

        if target_node is None or target_node["id"] == "classify":
            continue

        current_instruction = target_node.get("instruction", "")
        if len(current_instruction) + len(append_text) > 4000:
            append_text = append_text[: 4000 - len(current_instruction)]

        target_node["instruction"] = (
            current_instruction.rstrip() + "\n\n" + append_text.strip() + "\n"
        )
        applied_count += 1
        print(f"    Applied addition to {target_node['id']} (+{len(append_text)} chars)")

    if applied_count == 0:
        logger.warning("No additions applied. Returning copy of original config.")
        new_config["version"] = "e2e-proposed-noop"

    _validate_dag_config(new_config, HARNESS_NAME)
    return new_config


# =============================================================================
# 4. Main E2E Loop
# =============================================================================


async def run_e2e_full_loop() -> dict:
    """Run the complete E2E loop: score -> propose -> validate -> canary -> promote.

    Returns a results dict with all phase data.
    """
    client = genai.Client(
        vertexai=True,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT", "alanblount-demo"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
    )

    results = {
        "domain": "customer_support",
        "original_config": str(ORIGINAL_DAG.name),
        "phases": {},
        "outcome": "INCOMPLETE",
    }

    # Load the original DAG config
    original_config = load_dag_config(ORIGINAL_DAG)
    dag_version = original_config.get("version", "?")

    print(f"\nLoaded config: {ORIGINAL_DAG.name} v{dag_version}")
    print(f"Nodes: {[n['id'] for n in original_config['nodes'] if n.get('type') == 'agent']}")
    print(f"Model: {original_config.get('default_model', 'N/A')}")
    print(f"\nTest case splits:")
    print(f"  Training (SCORE):    {len(TRAIN_CASES)} cases (ids {TRAIN_CASES[0]['id']}-{TRAIN_CASES[-1]['id']})")
    print(f"  Held-out (VALIDATE): {len(HOLDOUT_CASES)} cases (ids {HOLDOUT_CASES[0]['id']}-{HOLDOUT_CASES[-1]['id']})")
    print(f"  Canary (CANARY):     {len(CANARY_CASES)} cases (ids {CANARY_CASES[0]['id']}-{CANARY_CASES[-1]['id']})")
    print(f"  Total:               {len(ALL_CASES)} cases")

    t_start = time.time()

    # =========================================================================
    # PHASE 1: SCORE
    # =========================================================================
    print(f"\n{'='*70}")
    print("PHASE 1: SCORE -- Baseline evaluation on training set")
    print(f"{'='*70}")
    print(f"  Running {len(TRAIN_CASES)} training cases through original DAG...\n")

    train_baseline = await run_benchmark(
        client, original_config, TRAIN_CASES, "SCORE"
    )
    print_result_summary(train_baseline, "SCORE (Training Baseline)")

    results["phases"]["score"] = {
        "cases": len(TRAIN_CASES),
        "aggregate_score": round(train_baseline.aggregate_score, 2),
        "category_accuracy": round(train_baseline.category_accuracy, 4),
        "avg_quality": round(train_baseline.avg_quality, 2),
        "avg_helpfulness": round(train_baseline.avg_helpfulness, 2),
        "category_breakdown": {
            cat: {
                "avg_score": round(info["avg_score"], 2),
                "accuracy": round(info["accuracy"], 4),
                "count": info["total"],
            }
            for cat, info in train_baseline.category_breakdown().items()
        },
        "weak_cases": [
            {
                "case_id": r.case_id,
                "query": r.query[:60],
                "score": r.score["total_score"],
                "expected": r.expected_category,
                "actual": r.actual_category,
            }
            for r in train_baseline.weakest_cases(3)
        ],
    }

    # =========================================================================
    # PHASE 2: PROPOSE
    # =========================================================================
    print(f"\n{'='*70}")
    print("PHASE 2: PROPOSE -- Generate DAG mutation from training analysis")
    print(f"{'='*70}")

    proposed_config = await propose_mutation(
        client, original_config, train_baseline
    )

    # Show what changed
    print(f"\n  Proposed config version: {proposed_config.get('version', '?')}")
    for node in proposed_config["nodes"]:
        if node.get("type") == "agent" and node["id"] != "classify":
            orig_node = next(
                (n for n in original_config["nodes"] if n["id"] == node["id"]),
                None,
            )
            if orig_node:
                orig_len = len(orig_node.get("instruction", ""))
                new_len = len(node.get("instruction", ""))
                delta = new_len - orig_len
                if delta > 0:
                    print(f"    {node['id']}: instruction grew by {delta} chars")

    results["phases"]["propose"] = {
        "version": proposed_config.get("version", "?"),
        "nodes_modified": [
            n["id"]
            for n in proposed_config["nodes"]
            if n.get("type") == "agent"
            and n["id"] != "classify"
            and len(n.get("instruction", ""))
            > len(
                next(
                    (
                        o.get("instruction", "")
                        for o in original_config["nodes"]
                        if o["id"] == n["id"]
                    ),
                    "",
                )
            )
        ],
    }

    # =========================================================================
    # PHASE 3: VALIDATE -- Held-out gate
    # =========================================================================
    print(f"\n{'='*70}")
    print("PHASE 3: VALIDATE -- Held-out evaluation (original vs proposed)")
    print(f"{'='*70}")

    # Run original config on holdout
    print(f"\n  Running {len(HOLDOUT_CASES)} held-out cases through ORIGINAL config...\n")
    holdout_original = await run_benchmark(
        client, original_config, HOLDOUT_CASES, "VALIDATE-orig"
    )
    print_result_summary(holdout_original, "VALIDATE (Original on Held-Out)")

    # Run proposed config on holdout
    print(f"\n  Running {len(HOLDOUT_CASES)} held-out cases through PROPOSED config...\n")
    holdout_proposed = await run_benchmark(
        client, proposed_config, HOLDOUT_CASES, "VALIDATE-prop"
    )
    print_result_summary(holdout_proposed, "VALIDATE (Proposed on Held-Out)")

    # Validation gate (with noise tolerance -- LLM judge has ~1-2 pt variance)
    holdout_delta = holdout_proposed.aggregate_score - holdout_original.aggregate_score
    validation_passed = holdout_delta >= -VALIDATION_TOLERANCE

    print(f"\n  VALIDATION GATE (tolerance={VALIDATION_TOLERANCE} pts for LLM judge noise):")
    print(f"    Original held-out score:  {holdout_original.aggregate_score:.1f}")
    print(f"    Proposed held-out score:  {holdout_proposed.aggregate_score:.1f}")
    print(f"    Delta:                    {holdout_delta:+.1f}")
    print(f"    Threshold:                >= {-VALIDATION_TOLERANCE:+.1f}")
    print(f"    Gate:                     {'PASSED' if validation_passed else 'FAILED'}")

    results["phases"]["validate"] = {
        "holdout_cases": len(HOLDOUT_CASES),
        "original_score": round(holdout_original.aggregate_score, 2),
        "proposed_score": round(holdout_proposed.aggregate_score, 2),
        "delta": round(holdout_delta, 2),
        "passed": validation_passed,
    }

    if not validation_passed:
        print(f"\n  Validation FAILED -- proposed config regressed on held-out set.")
        print(f"  Loop terminates. Original config preserved.")
        results["outcome"] = "VALIDATION_FAILED"
        results["elapsed_seconds"] = round(time.time() - t_start, 1)
        _save_results(results)
        return results

    # =========================================================================
    # PHASE 4: CANARY -- Final safety net
    # =========================================================================
    print(f"\n{'='*70}")
    print("PHASE 4: CANARY -- Final safety net (5 unseen cases)")
    print(f"{'='*70}")

    # Run original config on canary cases
    print(f"\n  Running {len(CANARY_CASES)} canary cases through ORIGINAL config...\n")
    canary_original = await run_benchmark(
        client, original_config, CANARY_CASES, "CANARY-orig"
    )
    print_result_summary(canary_original, "CANARY (Original)")

    # Run proposed config on canary cases
    print(f"\n  Running {len(CANARY_CASES)} canary cases through PROPOSED config...\n")
    canary_proposed = await run_benchmark(
        client, proposed_config, CANARY_CASES, "CANARY-prop"
    )
    print_result_summary(canary_proposed, "CANARY (Proposed)")

    # Canary gate: must be above floor AND not regress more than 5 points
    canary_delta = canary_proposed.aggregate_score - canary_original.aggregate_score
    canary_above_floor = canary_proposed.aggregate_score >= 50
    canary_no_regression = canary_delta >= -5.0
    canary_passed = canary_above_floor and canary_no_regression

    print(f"\n  CANARY GATE:")
    print(f"    Original canary score: {canary_original.aggregate_score:.1f}")
    print(f"    Proposed canary score: {canary_proposed.aggregate_score:.1f}")
    print(f"    Delta:                 {canary_delta:+.1f}")
    print(f"    Above floor (>=50):    {'YES' if canary_above_floor else 'NO'}")
    print(f"    No regression (>=-5):  {'YES' if canary_no_regression else 'NO'}")
    print(f"    Gate:                  {'PASSED' if canary_passed else 'FAILED'}")

    results["phases"]["canary"] = {
        "canary_cases": len(CANARY_CASES),
        "original_score": round(canary_original.aggregate_score, 2),
        "proposed_score": round(canary_proposed.aggregate_score, 2),
        "delta": round(canary_delta, 2),
        "above_floor": canary_above_floor,
        "no_regression": canary_no_regression,
        "passed": canary_passed,
    }

    if not canary_passed:
        print(f"\n  Canary FAILED -- proposed config didn't pass safety net.")
        print(f"  Loop terminates. Original config preserved.")
        results["outcome"] = "CANARY_FAILED"
        results["elapsed_seconds"] = round(time.time() - t_start, 1)
        _save_results(results)
        return results

    # =========================================================================
    # PHASE 5: PROMOTE -- Write the new config
    # =========================================================================
    print(f"\n{'='*70}")
    print("PHASE 5: PROMOTE -- Writing promoted config")
    print(f"{'='*70}")

    proposed_config["version"] = "e2e-promoted"
    with open(PROMOTED_DAG, "w") as f:
        yaml.dump(
            {"dag": proposed_config}, f, default_flow_style=False, sort_keys=False
        )

    print(f"\n  Promoted config written to: {PROMOTED_DAG.name}")

    results["phases"]["promote"] = {
        "promoted_path": str(PROMOTED_DAG.name),
        "version": proposed_config.get("version", "?"),
    }
    results["outcome"] = "PROMOTED"
    results["elapsed_seconds"] = round(time.time() - t_start, 1)

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print(f"\n{'='*70}")
    print("E2E FULL LOOP -- FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  Domain:            {HARNESS_NAME}")
    print(f"  Original config:   {ORIGINAL_DAG.name}")
    print(f"  Promoted config:   {PROMOTED_DAG.name}")
    print(f"  Outcome:           {results['outcome']}")
    print(f"  Elapsed:           {results['elapsed_seconds']:.0f}s")
    print()
    print(f"  Phase 1 (SCORE):    {train_baseline.aggregate_score:.1f}/100 on {len(TRAIN_CASES)} training cases")
    print(f"  Phase 2 (PROPOSE):  {len(results['phases']['propose'].get('nodes_modified', []))} nodes modified")
    print(f"  Phase 3 (VALIDATE): {holdout_original.aggregate_score:.1f} -> {holdout_proposed.aggregate_score:.1f} ({holdout_delta:+.1f}) -- PASSED")
    print(f"  Phase 4 (CANARY):   {canary_original.aggregate_score:.1f} -> {canary_proposed.aggregate_score:.1f} ({canary_delta:+.1f}) -- PASSED")
    print(f"  Phase 5 (PROMOTE):  Written to {PROMOTED_DAG.name}")
    print()
    print(f"  Did the loop produce a better config?")
    # Overall improvement: average across all test sets
    total_orig = (
        train_baseline.aggregate_score * len(TRAIN_CASES)
        + holdout_original.aggregate_score * len(HOLDOUT_CASES)
        + canary_original.aggregate_score * len(CANARY_CASES)
    ) / len(ALL_CASES)
    total_proposed_estimate = (
        train_baseline.aggregate_score * len(TRAIN_CASES)  # training: same (optimizer saw these)
        + holdout_proposed.aggregate_score * len(HOLDOUT_CASES)
        + canary_proposed.aggregate_score * len(CANARY_CASES)
    ) / len(ALL_CASES)
    overall_delta = total_proposed_estimate - total_orig
    print(f"    Weighted average (orig):     {total_orig:.1f}")
    print(f"    Weighted average (proposed): {total_proposed_estimate:.1f}")
    print(f"    Overall delta:               {overall_delta:+.1f}")
    if overall_delta > 0:
        print(f"    Answer: YES -- the loop improved the config by {overall_delta:.1f} points.")
    elif overall_delta == 0:
        print(f"    Answer: NEUTRAL -- no measurable change.")
    else:
        print(f"    Answer: MIXED -- held-out/canary passed but weighted average is {overall_delta:+.1f}.")

    _save_results(results)
    return results


def _save_results(results: dict) -> None:
    """Save results to JSON file."""
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {RESULTS_FILE.name}")


# =============================================================================
# 5. CLI Entry Point
# =============================================================================


async def main():
    print("=" * 70)
    print("E2E Full Loop: score -> propose -> validate -> canary -> promote")
    print("=" * 70)
    print(f"Domain:       {HARNESS_NAME}")
    print(f"Config:       {ORIGINAL_DAG.name}")
    print(f"Judge model:  {JUDGE_MODEL}")
    print(f"Total cases:  {len(ALL_CASES)} (10 train + 8 holdout + 5 canary)")
    print(f"LLM delay:    {LLM_DELAY}s")
    print()
    print("Phases:")
    print("  1. SCORE    -- Evaluate baseline on 10 training cases")
    print("  2. PROPOSE  -- Generate DAG mutation from training analysis")
    print("  3. VALIDATE -- Compare original vs proposed on 8 held-out cases")
    print("  4. CANARY   -- Final safety net on 5 unseen cases")
    print("  5. PROMOTE  -- Write promoted config if gates pass")

    results = await run_e2e_full_loop()

    if results["outcome"] == "PROMOTED":
        print(f"\nSUCCESS: Full loop completed -- config promoted!")
    elif results["outcome"] == "VALIDATION_FAILED":
        print(f"\nSTOPPED: Validation gate failed -- proposed config regressed.")
    elif results["outcome"] == "CANARY_FAILED":
        print(f"\nSTOPPED: Canary gate failed -- proposed config unsafe.")
    else:
        print(f"\nINCOMPLETE: Loop did not finish (outcome={results['outcome']}).")


if __name__ == "__main__":
    asyncio.run(main())
