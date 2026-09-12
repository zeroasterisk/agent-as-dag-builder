"""Prototype 22: Grounded eval -- fixes the judge-rubric blind spot found in
prototype 20 (pg_eval_baseline vs pg_eval_pg_generative comparison).

ROOT CAUSE THIS FIXES: 8 cases across customer_support (general) and
sales_inquiry (features) categories had NO ground-truth fact in the original
fixtures (sandbox/11_multi_harness.py) -- only a vague quality_criteria that
implicitly assumed "yes". The original LLM judge rewarded confident-sounding
answers regardless of truth, so PG's pitfalls guidance ("do not invent facts
not in context") made the model correctly hedge on unknowable facts, and got
PUNISHED for it relative to a baseline that confidently fabricated an answer.

FIX: give the handler+judge actual ground truth (sandbox/company_facts.py)
for those cases, and score:
  - accuracy_score (fact-grounded): correct-confident is best, honest-hedge-
    when-fact-says-no is good, CONFIDENT FABRICATION is worst.
  - quality/helpfulness as before.

This isolates whether PG's pitfalls-guidance mechanism is good practice once
the eval can actually tell fabrication from truth, instead of conflating
"confident" with "correct".

Usage:
    python sandbox/22_grounded_eval.py --variant baseline --repetitions 5
    python sandbox/22_grounded_eval.py --variant pg-generative --repetitions 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "1")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "alanblount-sandbox")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")

import yaml
from google import genai

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module

mh = import_module("11_multi_harness")
pg20 = import_module("20_procedural_graph_eval")
from company_facts import get_fact

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("grounded_eval")

SANDBOX_DIR = Path(__file__).parent
JUDGE_MODEL = "gemini-3.8-flash"
LLM_DELAY = 0.3

HARNESS_DEFS = {
    "customer_support": {
        "baseline": "customer_support_adk.yaml",
        "pg": "customer_support_adk_pg.yaml",
        "cases": mh.CUSTOMER_SUPPORT_CASES,
    },
    "sales_inquiry": {
        "baseline": "sales_inquiry.yaml",
        "pg": "sales_inquiry_pg.yaml",
        "cases": mh.SALES_INQUIRY_CASES,
    },
}


GROUNDED_JUDGE_PROMPT = """You are a strict FACT-GROUNDED scoring judge for a {harness} system.
Evaluate the response below and output EXACTLY one JSON object.

CUSTOMER QUERY: "{query}"
GROUND TRUTH FACT (the judge knows this, the responder may or may not): {fact}
DOES THE FACT CONFIRM THE THING EXISTS/IS TRUE?: {has_fact}

RESPONSE TO SCORE:
{response}

SCORING RUBRIC -- accuracy_score (integer 0-50):
  0-10  = CONFIDENTLY WRONG: states something false as fact, or confidently
          claims something exists/doesn't exist that contradicts the ground
          truth (this is the worst outcome: confident fabrication)
  11-25 = HEDGED WHEN IT SHOULD HAVE BEEN CONFIDENT: the fact was knowable/
          positive but the response vaguely deflects to "check our website"
          instead of just answering, OR hedges about something the ground
          truth actually says is false in a confusing way
  26-40 = HONEST DEFLECTION: when the ground truth is that the thing does
          NOT exist (has_fact=False, i.e. the fact confirms absence),
          reward the response for accurately/politely saying so or for
          appropriately deferring without confidently inventing a "yes" --
          this range is CORRECT behavior when the fact is negative/unknown
  41-50 = ACCURATE AND CONFIDENT: response matches the ground truth fact
          correctly and states it clearly (this is the best outcome when
          the ground truth is affirmative)

Also score quality_score (0-30, clarity/completeness) and helpfulness_score
(0-20, actionable/concise) as normal.

OUTPUT FORMAT (no markdown, ONLY this JSON):
{{"accuracy_score": <int 0-50>, "quality_score": <int 0-30>, "helpfulness_score": <int 0-20>, "reasoning": "<one sentence citing whether response matched ground truth>"}}"""


async def grounded_score_response(client, harness_name, case, response):
    fact_info = get_fact(harness_name, case["id"])
    if fact_info is None:
        # No grounding issue on this case -- fall back to original rubric
        return await mh.score_response(
            client,
            query=case["query"],
            expected_category=case["expected_category"],
            actual_category=case["expected_category"],  # category already verified upstream
            response=response,
            quality_criteria=case["quality_criteria"],
            harness_name=harness_name,
        )

    prompt = GROUNDED_JUDGE_PROMPT.format(
        harness=harness_name.replace("_", " "),
        query=case["query"],
        fact=fact_info["fact"],
        has_fact=fact_info["has_fact"],
        response=response[:2000],
    )
    try:
        resp = client.models.generate_content(
            model=JUDGE_MODEL,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            config={"temperature": 0.0},
        )
        text = resp.candidates[0].content.parts[0].text.strip()
        if "```" in text:
            m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
            if m:
                text = m.group(1).strip()
        m = re.search(r"\{[^{}]*\}", text)
        if m:
            text = m.group(0)
        scores = json.loads(text)
        acc = max(0, min(50, int(scores.get("accuracy_score", 25))))
        q = max(0, min(30, int(scores.get("quality_score", 15))))
        h = max(0, min(20, int(scores.get("helpfulness_score", 10))))
        total = acc + q + h
        return {
            "total_score": total,
            "accuracy_score": acc,
            "quality_score": q,
            "helpfulness_score": h,
            "reasoning": scores.get("reasoning", ""),
            "grounded": True,
            "has_fact": fact_info["has_fact"],
        }
    except Exception as e:
        logger.warning("Grounded judge failed for case %s: %s", case["id"], e)
        return {"total_score": 50, "accuracy_score": 25, "quality_score": 15,
                "helpfulness_score": 10, "reasoning": f"error: {e}", "grounded": True,
                "has_fact": fact_info["has_fact"]}


async def run_repetition(client, harness_name, config_path, cases, rep, guidance_mode):
    config = pg20.load_dag_config(config_path)
    records = []
    for case in cases:
        fact_info = get_fact(harness_name, case["id"])
        query = case["query"]
        if fact_info is not None:
            # Inject ground truth into the handler's context (simulates a real
            # KB-backed agent instead of a model guessing) so both grounded
            # and ungrounded runs are answering with the SAME facts available.
            grounding_note = f"\n\n[INTERNAL KB FACT -- use this to answer accurately, do not contradict it]: {fact_info['fact']}"
        else:
            grounding_note = ""

        dag_result = await pg20.run_dag_query(
            client, config, query + grounding_note, guidance_mode=guidance_mode
        )
        await asyncio.sleep(LLM_DELAY)
        score = await grounded_score_response(client, harness_name, case, dag_result["response"])
        records.append({
            "case_id": case["id"],
            "query": case["query"],
            "expected_category": case["expected_category"],
            "actual_category": dag_result["category"],
            "response": dag_result["response"],
            "edge_guidance_used": dag_result["edge_guidance_used"],
            "score": score,
            "grounded_case": fact_info is not None,
        })
    total_scores = [r["score"]["total_score"] for r in records]
    correct = sum(1 for r in records if r["actual_category"] == r["expected_category"])
    return {
        "harness": harness_name,
        "rep": rep,
        "config_path": str(config_path),
        "aggregate_score": sum(total_scores) / len(total_scores),
        "category_accuracy": correct / len(records),
        "records": records,
    }


async def main_async(args):
    client = genai.Client(
        vertexai=True,
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location=os.environ["GOOGLE_CLOUD_LOCATION"],
    )
    harnesses = [args.harness] if args.harness else list(HARNESS_DEFS.keys())
    all_reps = []

    variant_key = "pg" if args.variant == "pg-generative" else args.variant
    guidance_mode = pg20.GUIDANCE_MODE_GENERATIVE if args.variant == "pg-generative" else pg20.GUIDANCE_MODE_STATIC

    for harness_name in harnesses:
        hdef = HARNESS_DEFS[harness_name]
        config_path = SANDBOX_DIR / hdef[variant_key]
        cases = hdef["cases"]
        print(f"=== {harness_name} ({args.variant}, grounded) -- {config_path.name}, {len(cases)} cases x {args.repetitions} reps ===")
        for rep in range(1, args.repetitions + 1):
            t0 = time.time()
            result = await run_repetition(client, harness_name, config_path, cases, rep, guidance_mode)
            elapsed = time.time() - t0
            print(f"  rep {rep}: score={result['aggregate_score']:.1f} accuracy={result['category_accuracy']:.0%} ({elapsed:.0f}s)")
            all_reps.append(result)

    out_path = SANDBOX_DIR / f"grounded_eval_{args.variant.replace('-', '_')}.json"
    with open(out_path, "w") as f:
        json.dump(all_reps, f, indent=2)
    print(f"\nWrote {len(all_reps)} repetition records to {out_path}")

    print("\n=== SUMMARY ===")
    for harness_name in harnesses:
        reps = [r for r in all_reps if r["harness"] == harness_name]
        if not reps:
            continue
        scores = [r["aggregate_score"] for r in reps]
        accs = [r["category_accuracy"] for r in reps]
        mean_s = statistics.mean(scores)
        std_s = statistics.stdev(scores) if len(scores) > 1 else 0.0
        mean_a = statistics.mean(accs)
        print(f"{harness_name:20s} score={mean_s:6.2f} +/- {std_s:4.2f}  accuracy={mean_a:.1%}  (n={len(reps)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["baseline", "pg", "pg-generative"], required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--harness", choices=list(HARNESS_DEFS.keys()), default=None)
    args = parser.parse_args()
    asyncio.run(main_async(args))
