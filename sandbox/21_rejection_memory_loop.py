"""Prototype 21: E2E Full Loop + Rejection Memory (Procedural Graphs idea #3).

Extends sandbox/19_e2e_full_loop.py with ONE additional mechanism from
arXiv:2609.09153 "Procedural Graphs": when a proposed mutation fails the
VALIDATE or CANARY gate, persist it to a rejection-memory file instead of
discarding it silently. Future PROPOSE calls are shown a summary of past
rejections ("don't repeat these") so the optimizer doesn't keep re-proposing
mutations that already failed validation for a given node/category.

This isolates the PG claim: does conditioning the optimizer on its own
rejection history reduce repeated failed optimization attempts (the same
overfitting failure mode documented in design-decisions.md #6 / the v3
regression in prototype 10)?

Usage:
    python sandbox/21_rejection_memory_loop.py --iterations 5
    python sandbox/21_rejection_memory_loop.py --iterations 5 --reset-memory

Environment:
    GOOGLE_GENAI_USE_VERTEXAI=1
    GOOGLE_CLOUD_PROJECT=alanblount-sandbox
    GOOGLE_CLOUD_LOCATION=global
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "1")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "alanblount-sandbox")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")

from google import genai

SANDBOX_DIR = Path(__file__).parent

# -- Import shared infrastructure from 19_e2e_full_loop.py --------------------
import importlib.util as _ilu

_spec19 = _ilu.spec_from_file_location("e2e_full_loop", str(SANDBOX_DIR / "19_e2e_full_loop.py"))
_e2e = _ilu.module_from_spec(_spec19)
sys.modules["e2e_full_loop"] = _e2e
_spec19.loader.exec_module(_e2e)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("rejection_memory_loop")

REJECTION_MEMORY_FILE = SANDBOX_DIR / "rejection_memory.json"
RESULTS_FILE = SANDBOX_DIR / "scores_rejection_memory.json"


def load_rejection_memory() -> list[dict]:
    if REJECTION_MEMORY_FILE.exists():
        with open(REJECTION_MEMORY_FILE) as f:
            return json.load(f)
    return []


def save_rejection_memory(memory: list[dict]) -> None:
    with open(REJECTION_MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)


def format_rejection_memory_for_prompt(memory: list[dict], max_entries: int = 8) -> str:
    """Summarize past rejections into a short block for the PROPOSE prompt.

    PG (Sec 3, offline evolution) keeps rejected edits so the refiner does not
    re-propose the same bad mutation. We mirror this at the instruction-append
    granularity: node_id + a short reason + a snippet of the rejected text.
    """
    if not memory:
        return ""
    recent = memory[-max_entries:]
    lines = ["PREVIOUSLY REJECTED MUTATIONS (do not repeat these strategies -- they failed validation):"]
    for i, entry in enumerate(recent, 1):
        snippet = entry["append_text"][:150].replace("\n", " ")
        lines.append(
            f"  {i}. node={entry['node_id']} reason={entry['reason']} "
            f"(delta={entry.get('delta', '?'):+.1f} pts) text=\"{snippet}...\""
        )
    return "\n".join(lines)


async def propose_mutation_with_memory(
    client: genai.Client,
    config: dict,
    train_result,
    rejection_memory: list[dict],
) -> dict:
    """Same as e2e_full_loop.propose_mutation but injects rejection memory
    into the optimizer prompt. We monkey-patch the analysis by calling the
    original building blocks directly rather than duplicating them."""
    focused_context = _e2e._build_agent_driven_context(train_result, config, _e2e.HARNESS_NAME)
    breakdown = train_result.category_breakdown()
    valid_categories = _e2e.extract_categories_from_config(config)

    rejection_block = format_rejection_memory_for_prompt(rejection_memory)

    analysis = f"""E2E Full Loop -- SCORE Phase Results (customer_support):
- Aggregate Score: {train_result.aggregate_score:.1f}/100
- Category Accuracy: {train_result.category_accuracy:.0%}
- Quality Score: {train_result.avg_quality:.1f}/30
- Helpfulness Score: {train_result.avg_helpfulness:.1f}/20

Valid Categories: {', '.join(valid_categories)}

Category Breakdown:
"""
    for cat, info in breakdown.items():
        analysis += f"  {cat}: avg={info['avg_score']:.1f}, accuracy={info['accuracy']:.0%} ({info['total']} cases)\n"

    analysis += f"\n{focused_context}"
    if rejection_block:
        analysis += f"\n\n{rejection_block}\n"

    prompt = f"""You are an AI workflow optimizer. Analyze these customer support DAG benchmark
results and propose specific, additive improvements.

{analysis}

IMPORTANT RULES:
1. You must NOT rewrite handler instructions from scratch.
2. You must ONLY propose ADDITIONS to append to the existing handler instructions.
3. Focus on the weakest category shown above.
4. If a "PREVIOUSLY REJECTED MUTATIONS" list is shown above, propose something
   MEANINGFULLY DIFFERENT in substance from those -- do not just reword a
   rejected strategy.
5. Each addition should be one of:
   - Issue-specific template: step-by-step response flow for a specific sub-issue
   - Domain knowledge injection: concrete details (URLs, phone numbers, specific policies)
   - Quality criteria hint: instruction that addresses what the judge is looking for

Propose 2-3 specific ADDITIONS to append to the weakest handler.
For each proposal, specify:
- The exact text to APPEND to the handler instruction
- Which handler node it applies to (e.g., handle_billing, handle_technical, handle_general)
- Why (referencing specific weak cases)

Reply with a numbered list. Be specific and concrete."""

    await asyncio.sleep(_e2e.LLM_DELAY)
    resp = client.models.generate_content(
        model=_e2e.JUDGE_MODEL,
        contents=[{"role": "user", "parts": [{"text": prompt}]}],
    )
    proposals_text = resp.candidates[0].content.parts[0].text.strip()

    proposals = []
    current = ""
    import re
    for line in proposals_text.split("\n"):
        if re.match(r"^\d+[\.\)]\s", line.strip()) and current:
            proposals.append(current.strip())
            current = line
        else:
            current += "\n" + line
    if current.strip():
        proposals.append(current.strip())

    print(f"\n  Proposals generated: {len(proposals)}  (rejection memory: {len(rejection_memory)} entries)")
    for i, prop in enumerate(proposals, 1):
        print(f"    Proposal {i}: {prop.split(chr(10))[0].strip()[:80]}")

    new_config = await _e2e._apply_proposals(client, config, proposals)
    return new_config


def _diff_additions(original: dict, mutated: dict) -> list[dict]:
    """Extract what was actually appended, per node, for rejection logging."""
    orig_by_id = {n["id"]: n for n in original["nodes"]}
    diffs = []
    for n in mutated["nodes"]:
        if n.get("type") != "agent" or n["id"] == "classify":
            continue
        orig_instr = orig_by_id.get(n["id"], {}).get("instruction", "")
        new_instr = n.get("instruction", "")
        if len(new_instr) > len(orig_instr):
            diffs.append({"node_id": n["id"], "append_text": new_instr[len(orig_instr):].strip()})
    return diffs


async def run_iteration(
    client: genai.Client,
    original_config: dict,
    rejection_memory: list[dict],
    iteration: int,
) -> dict:
    """One SCORE -> PROPOSE(+memory) -> VALIDATE -> CANARY cycle. On failure,
    append the rejected mutation(s) to rejection_memory (mutated in place)."""
    print(f"\n{'#'*70}\n# ITERATION {iteration}\n{'#'*70}")

    train_baseline = await _e2e.run_benchmark(client, original_config, _e2e.TRAIN_CASES, "SCORE")
    _e2e.print_result_summary(train_baseline, "SCORE (Training Baseline)")

    proposed_config = await propose_mutation_with_memory(
        client, original_config, train_baseline, rejection_memory
    )

    holdout_original = await _e2e.run_benchmark(client, original_config, _e2e.HOLDOUT_CASES, "VALIDATE-orig")
    holdout_proposed = await _e2e.run_benchmark(client, proposed_config, _e2e.HOLDOUT_CASES, "VALIDATE-prop")
    holdout_delta = holdout_proposed.aggregate_score - holdout_original.aggregate_score
    validation_passed = holdout_delta >= -_e2e.VALIDATION_TOLERANCE

    print(f"\n  VALIDATE: {holdout_original.aggregate_score:.1f} -> {holdout_proposed.aggregate_score:.1f} "
          f"({holdout_delta:+.1f}) -- {'PASSED' if validation_passed else 'FAILED'}")

    outcome = {"iteration": iteration, "holdout_delta": round(holdout_delta, 2)}

    if not validation_passed:
        for diff in _diff_additions(original_config, proposed_config):
            rejection_memory.append({
                **diff,
                "reason": "validation_failed",
                "delta": round(holdout_delta, 2),
                "iteration": iteration,
            })
        save_rejection_memory(rejection_memory)
        outcome["outcome"] = "VALIDATION_FAILED"
        outcome["promoted"] = False
        return outcome

    canary_original = await _e2e.run_benchmark(client, original_config, _e2e.CANARY_CASES, "CANARY-orig")
    canary_proposed = await _e2e.run_benchmark(client, proposed_config, _e2e.CANARY_CASES, "CANARY-prop")
    canary_delta = canary_proposed.aggregate_score - canary_original.aggregate_score
    canary_passed = canary_proposed.aggregate_score >= 50 and canary_delta >= -5.0

    print(f"  CANARY:   {canary_original.aggregate_score:.1f} -> {canary_proposed.aggregate_score:.1f} "
          f"({canary_delta:+.1f}) -- {'PASSED' if canary_passed else 'FAILED'}")

    outcome["canary_delta"] = round(canary_delta, 2)

    if not canary_passed:
        for diff in _diff_additions(original_config, proposed_config):
            rejection_memory.append({
                **diff,
                "reason": "canary_failed",
                "delta": round(canary_delta, 2),
                "iteration": iteration,
            })
        save_rejection_memory(rejection_memory)
        outcome["outcome"] = "CANARY_FAILED"
        outcome["promoted"] = False
        return outcome

    outcome["outcome"] = "PROMOTED"
    outcome["promoted"] = True
    outcome["new_config"] = proposed_config
    return outcome


async def main_async(args):
    client = genai.Client(
        vertexai=True,
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location=os.environ["GOOGLE_CLOUD_LOCATION"],
    )

    if args.reset_memory and REJECTION_MEMORY_FILE.exists():
        REJECTION_MEMORY_FILE.unlink()

    rejection_memory = load_rejection_memory()
    config = _e2e.load_dag_config(_e2e.ORIGINAL_DAG)

    history = []
    promotions = 0
    for it in range(1, args.iterations + 1):
        outcome = await run_iteration(client, config, rejection_memory, it)
        history.append({k: v for k, v in outcome.items() if k != "new_config"})
        if outcome.get("promoted"):
            config = outcome["new_config"]
            promotions += 1

    print(f"\n{'='*70}\nRUN COMPLETE: {promotions}/{args.iterations} iterations promoted, "
          f"{len(rejection_memory)} rejection-memory entries accumulated\n{'='*70}")

    with open(RESULTS_FILE, "w") as f:
        json.dump({"history": history, "rejection_memory_size": len(rejection_memory), "promotions": promotions}, f, indent=2)
    print(f"Results: {RESULTS_FILE}")
    print(f"Rejection memory: {REJECTION_MEMORY_FILE} ({len(rejection_memory)} entries)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--reset-memory", action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(args))
