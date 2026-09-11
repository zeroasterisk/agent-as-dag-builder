"""Prototype 20: Procedural-Graphs-inspired eval harness.

Implements a controlled A/B between:
  - BASELINE: existing GG DAGs (monolithic node instructions only)
  - PG variant: same DAGs, but conditional edges carry `guidance` +
    `pitfalls` attributes (per arXiv:2609.09153 "Procedural Graphs") which
    are localized (only the traversed edge, not the whole graph) and
    injected into the handler prompt alongside the node's base instruction.

This isolates ONE testable claim from the paper: does attaching small,
edge-scoped situational guidance (vs. folding everything into static node
instructions) measurably change response quality/helpfulness on the same
judge rubric used by sandbox/11_multi_harness.py?

Runs N repetitions per config and reports mean/stdev per harness + overall,
plus per-case trajectories (prompt inputs, model outputs, judge output) so
results can be diffed qualitatively, not just by score.

Usage:
    python sandbox/20_procedural_graph_eval.py --variant baseline --repetitions 5
    python sandbox/20_procedural_graph_eval.py --variant pg --repetitions 5
    python sandbox/20_procedural_graph_eval.py --variant baseline --repetitions 5 --harness customer_support

Environment:
    GOOGLE_GENAI_USE_VERTEXAI=1
    GOOGLE_CLOUD_PROJECT=alanblount-sandbox
    GOOGLE_CLOUD_LOCATION=global
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path

import yaml

os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "1")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "alanblount-sandbox")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")

from google import genai

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module

mh = import_module("11_multi_harness")

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("pg_eval")

SANDBOX_DIR = Path(__file__).parent
LLM_DELAY = 0.3

# Cheap model used ONLY to synthesize per-query situational guidance from the
# edge's raw guidance/pitfalls fields (PG Sec 2.2: "guidance LLM" -- a small
# model that turns the localized subgraph into step-specific text). Using
# flash-lite here is itself the "well-defined task -> smaller model" pattern:
# distilling 2 short paragraphs into 2-3 sentences needs no frontier model.
GUIDANCE_MODEL = "gemini-3.5-flash-lite"

GUIDANCE_MODE_STATIC = "static"       # v1 (prototype 20 original): inject raw text as-is
GUIDANCE_MODE_GENERATIVE = "generative"  # v2 (PG-faithful): synthesize per-query guidance

HARNESS_DEFS = {
    "customer_support": {
        "baseline": "customer_support_adk.yaml",
        "pg": "customer_support_adk_pg.yaml",
        "cases": mh.CUSTOMER_SUPPORT_CASES,
    },
    "it_helpdesk": {
        "baseline": "it_helpdesk.yaml",
        "pg": "it_helpdesk_pg.yaml",
        "cases": mh.IT_HELPDESK_CASES,
    },
    "sales_inquiry": {
        "baseline": "sales_inquiry.yaml",
        "pg": "sales_inquiry_pg.yaml",
        "cases": mh.SALES_INQUIRY_CASES,
    },
}


def load_dag_config(path: Path) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    config = raw["dag"]
    assert "nodes" in config, "YAML must define 'nodes'"
    assert "edges" in config, "YAML must define 'edges'"
    return config


def build_routing(config: dict) -> dict:
    start_nodes = []
    conditional: dict[str, list] = {}
    unconditional: dict[str, list] = {}
    for edge in config["edges"]:
        src, dst, cond = edge["from"], edge["to"], edge.get("condition")
        if src == "START":
            start_nodes.append(dst)
        elif cond:
            conditional.setdefault(src, []).append(edge)
        else:
            unconditional.setdefault(src, []).append(edge)
    return {"start_nodes": start_nodes, "conditional": conditional, "unconditional": unconditional}


def extract_categories(config: dict) -> list[str]:
    routing = build_routing(config)
    cats = []
    for node_id in routing["start_nodes"]:
        for route in routing["conditional"].get(node_id, []):
            cats.append(route["condition"].lower())
    return cats


async def synthesize_guidance(
    client: genai.Client, query: str, guidance: str, pitfalls: str
) -> str:
    """PG-faithful 'guidance LLM' (Sec 2.2 of arXiv:2609.09153): given the
    localized edge's raw guidance/pitfalls plus the ACTUAL query, a small
    model distills step-specific situational guidance for THIS query --
    rather than injecting the same static block for every case in the
    category. This is what prototype-20-v1 (GUIDANCE_MODE_STATIC) got wrong
    and what caused the -0.99..-4.40 pt regression across all 3 harnesses:
    static per-category text can steer the model toward generic advice
    (e.g. "restart the device") that doesn't fit the specific complaint
    (e.g. "check your spam folder"), matching the paper's own ablation
    finding that raw/full-graph injection underperforms generative,
    localized guidance (Table 3).
    """
    prompt = f"""You are a routing assistant. A user query has been classified onto an edge
with the following general guidance and pitfalls for that category:

GENERAL GUIDANCE: {guidance}
GENERAL PITFALLS TO AVOID: {pitfalls}

USER'S ACTUAL QUERY: "{query}"

In 2-3 sentences, write SITUATIONAL guidance specific to this exact query --
pick out which parts of the general guidance actually apply here, and note
anything the general guidance misses for this specific case. Be concrete
and query-specific, not a restatement of the general guidance verbatim."""

    try:
        resp = client.models.generate_content(
            model=GUIDANCE_MODEL,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            config={"temperature": 0.0},
        )
        text = ""
        for p in resp.candidates[0].content.parts:
            if hasattr(p, "text") and p.text:
                text += p.text
        return text.strip() or guidance
    except Exception as e:
        logger.warning("Guidance synthesis failed (%s), falling back to raw guidance", e)
        return guidance


async def run_dag_query(
    client: genai.Client,
    config: dict,
    query: str,
    guidance_mode: str = GUIDANCE_MODE_STATIC,
) -> dict:
    """Execute one query through the DAG. If the traversed edge carries
    `guidance`/`pitfalls`, inject them into the handler's prompt.

    guidance_mode="static": inject the edge's raw guidance/pitfalls verbatim
        (prototype-20-v1 behavior -- kept for comparison, this is the mode
        that regressed vs baseline).
    guidance_mode="generative": synthesize per-query situational guidance via
        a small model (GUIDANCE_MODEL) before injecting (PG-faithful, Sec 2.2).

    Baseline YAMLs have no guidance/pitfalls fields, so both modes are a
    no-op there -- same code path for both variants.
    """
    nodes_by_id = {n["id"]: n for n in config["nodes"]}
    routing = build_routing(config)
    default_model = config.get("default_model", "gemini-3.8-flash")
    valid_categories = extract_categories(config)

    results = {
        "category": "unknown",
        "response": "",
        "nodes_visited": [],
        "timings": {},
        "prompts": {},
        "edge_guidance_used": None,
    }

    async def call_node(node_id: str, user_input: str, extra_guidance: str = "") -> str:
        node = nodes_by_id.get(node_id)
        if node is None or node.get("type", "agent") != "agent":
            return ""
        model = node.get("model", default_model)
        instruction = node.get("instruction", "Help the user.")
        prompt = f"{instruction}\n\nUser request: {user_input}"
        if extra_guidance:
            prompt += f"\n\n{extra_guidance}"
        results["prompts"][node_id] = prompt
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
            results["timings"][node_id] = (time.time() - t0) * 1000
            results["nodes_visited"].append(node_id)
            return text.strip()
        except Exception as e:
            logger.error("Node %s failed: %s", node_id, e)
            results["timings"][node_id] = (time.time() - t0) * 1000
            results["nodes_visited"].append(node_id)
            return f"ERROR: {e}"

    classify_output = ""
    for node_id in routing["start_nodes"]:
        classify_output = await call_node(node_id, query)

    classify_lower = classify_output.lower().strip()
    detected_category = "unknown"
    for cat in valid_categories:
        if cat in classify_lower:
            detected_category = cat
            break
    results["category"] = detected_category

    # Find matching edge (carries condition + optional guidance/pitfalls)
    to_visit: list[tuple[str, str]] = []  # (node_id, extra_guidance)
    for node_id in routing["start_nodes"]:
        for route in routing["conditional"].get(node_id, []):
            if route["condition"].lower() in classify_lower:
                guidance = route.get("guidance", "")
                pitfalls = route.get("pitfalls", "")
                extra = ""
                if guidance or pitfalls:
                    if guidance_mode == GUIDANCE_MODE_GENERATIVE:
                        situational = await synthesize_guidance(client, query, guidance, pitfalls)
                        extra = f"SITUATIONAL GUIDANCE FOR THIS CASE: {situational}"
                    else:
                        if guidance:
                            extra += f"SITUATIONAL GUIDANCE FOR THIS CASE: {guidance}"
                        if pitfalls:
                            extra += f"\nAVOID THIS PITFALL: {pitfalls}"
                if extra:
                    results["edge_guidance_used"] = extra
                to_visit.append((route["to"], extra))
                break

    final_output = classify_output
    visited = set(routing["start_nodes"])
    queue = list(to_visit)
    while queue:
        node_id, extra = queue.pop(0)
        if node_id in visited:
            continue
        visited.add(node_id)
        handler_output = await call_node(node_id, query, extra_guidance=extra)
        if handler_output:
            final_output = handler_output
        for route in routing["conditional"].get(node_id, []):
            if route["condition"].lower() in handler_output.lower():
                queue.append((route["to"], route.get("guidance", "")))
                break
        for route in routing["unconditional"].get(node_id, []):
            queue.append((route["to"], ""))

    results["response"] = final_output
    return results


async def run_repetition(
    client: genai.Client,
    harness_name: str,
    config_path: Path,
    cases: list[dict],
    rep: int,
    guidance_mode: str = GUIDANCE_MODE_STATIC,
) -> dict:
    config = load_dag_config(config_path)
    records = []
    for case in cases:
        dag_result = await run_dag_query(client, config, case["query"], guidance_mode=guidance_mode)
        await asyncio.sleep(LLM_DELAY)
        score = await mh.score_response(
            client,
            query=case["query"],
            expected_category=case["expected_category"],
            actual_category=dag_result["category"],
            response=dag_result["response"],
            quality_criteria=case["quality_criteria"],
            harness_name=harness_name,
        )
        records.append({
            "case_id": case["id"],
            "query": case["query"],
            "expected_category": case["expected_category"],
            "actual_category": dag_result["category"],
            "response": dag_result["response"],
            "edge_guidance_used": dag_result["edge_guidance_used"],
            "score": score,
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
    guidance_mode = GUIDANCE_MODE_GENERATIVE if args.variant == "pg-generative" else GUIDANCE_MODE_STATIC

    for harness_name in harnesses:
        hdef = HARNESS_DEFS[harness_name]
        config_path = SANDBOX_DIR / hdef[variant_key]
        if not config_path.exists():
            print(f"SKIP {harness_name}: {config_path} not found (run edge-attribution step first)")
            continue
        cases = hdef["cases"]
        print(f"=== {harness_name} ({args.variant}) -- {config_path.name}, {len(cases)} cases x {args.repetitions} reps, guidance_mode={guidance_mode} ===")
        for rep in range(1, args.repetitions + 1):
            t0 = time.time()
            result = await run_repetition(client, harness_name, config_path, cases, rep, guidance_mode=guidance_mode)
            elapsed = time.time() - t0
            print(f"  rep {rep}: score={result['aggregate_score']:.1f} accuracy={result['category_accuracy']:.0%} ({elapsed:.0f}s)")
            all_reps.append(result)

    out_path = SANDBOX_DIR / f"pg_eval_{args.variant.replace('-', '_')}.json"
    with open(out_path, "w") as f:
        json.dump(all_reps, f, indent=2)
    print(f"\nWrote {len(all_reps)} repetition records to {out_path}")

    # Summary stats
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
