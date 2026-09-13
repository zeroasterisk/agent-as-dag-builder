# Intent — Multi-Turn Eval Harness for Graph Gardener (via Harbor)

## Why this branch exists

Handoff from `experiment/procedural-graphs-20260911` (Procedural Graphs paper
evaluation, arXiv:2609.09153). That work concluded the paper's core mechanism
(attributed edges + generative guidance) is a wash-to-marginal improvement on
GG's current eval harness — but surfaced a **structural limitation in the
harness itself**, independent of the paper: it can only score a single turn.

Specifically: when PG's guidance correctly made a technical-support handler
ask a clarifying question ("which device/OS?") instead of dumping every
possible troubleshooting step, the eval scored it *worse* — because the
judge only ever sees turn 1, and a good clarifying question necessarily
looks incomplete compared to a presumptive checklist. There is no way,
today, for GG's eval to reward "asked the right follow-up question" over
"gave a complete-sounding but presumptive answer."

This is not fixable by rubric-tweaking (already tried and validated that
angle on a *different* bug — the fabrication-vs-honesty rubric gap — in the
prior branch). It requires an eval harness that can run an actual back-and-
forth conversation and score the trajectory, not just one shot.

## What we're building

A multi-turn evaluation capability for GG, built on **Harbor**
(`harbor-framework/harbor`, Apache-2.0, ~5.2k★, actively maintained by the
Terminal-Bench team — verified live via GitHub API on 2026-09-12, not
assumed from memory).

Concretely:
1. Wrap GG's DAG runner as a Harbor `BaseAgent`.
2. Build a `tau3-bench`-style task: GG's DAG vs. a simulated user (Harbor
   RFC 0002 — "simulated users") that can answer follow-up questions like
   "what device/OS are you on?" instead of the conversation ending at turn 1.
3. Write one new custom rewardkit judge criterion: does the agent ask a
   well-targeted clarifying question when the request is ambiguous, vs.
   giving a premature complete answer? (No such criterion exists off the
   shelf in Harbor — this is the one real net-new build; everything else
   — simulated user, trajectory logging, multi-turn orchestration — is
   already provided by Harbor.)
4. Re-run baseline vs. PG-generative-guidance (artifacts already committed
   on the prior branch) under this new harness on the 3 ambiguous technical
   cases that exposed the gap (customer_support cases 14/15/16) and see
   whether PG's edge-guidance advantage actually shows up once multi-turn
   scoring is possible.

## Non-goals for this branch

- Not re-litigating the Procedural Graphs paper findings — those are closed
  out on `experiment/procedural-graphs-20260911` (7 commits, results in
  `sandbox/pg_eval_*.json` and `sandbox/grounded_eval_*.json`).
- Not migrating GG's whole eval suite to Harbor wholesale on day one — start
  with one harness (customer_support, the one with the known gap) and prove
  it out before deciding whether to migrate `it_helpdesk`/`sales_inquiry`.
- Not building the "reward clarifying questions" judge criterion as a
  generic GG feature yet — first get one working multi-turn eval end to
  end, then decide if it's worth generalizing.

## Handoff notes

- This branch was prepared in one Discord thread; development is expected
  to continue in a **different** thread/session with no memory of the prior
  conversation. `research.md` in this same directory has the full Harbor
  findings (independently verified, not just a subagent's self-report).
  `plan.md` has the concrete step-by-step build plan.
- Prior branch (`experiment/procedural-graphs-20260911`) has the full PG
  paper eval code + data if you need to cross-reference the failing cases
  or the eval harness patterns (`sandbox/11_multi_harness.py`,
  `sandbox/20_procedural_graph_eval.py`, `sandbox/22_grounded_eval.py`).
