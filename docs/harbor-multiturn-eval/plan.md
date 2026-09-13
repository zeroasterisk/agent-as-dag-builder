# Plan — Multi-Turn Eval Harness Build (Harbor)

Status: not started. This is a handoff plan for a fresh session/thread.

## Phase 0 — De-risk the one structural unknown ✅ DONE (2026-09-13)

**Result: PASSED.** Full findings in
`docs/harbor-multiturn-eval/phase0-spike-results.md`. Summary: `harbor
init` + `harbor run --agent oracle` on a hello-world task scored
**reward 1.0/1.0** end to end (image build → oracle solution → pytest
verifier → reward collected to host). No fallback to Inspect AI needed —
Harbor's container model works fine here.

Two sandbox-specific (not Harbor-specific, not GG-specific) issues were
found and fixed, and **must be carried forward into every subsequent
phase**:

1. `~/.cache` isn't writable by the sandbox user → run harbor with
   `HOME` redirected to a scratch dir containing `.cache/harbor` AND a
   copy of `~/.docker/cli-plugins/docker-compose` (redirecting `HOME`
   hides the real compose plugin otherwise).
2. **Task directories and `harbor run` must live under
   `/shared/workspace/`, not `/home/node/work/`.** This sandbox's Docker
   access goes through a separate DinD daemon with its own filesystem;
   bind mounts (which Harbor's verifier relies on to collect
   `reward.txt`) silently no-op for any path outside `/shared/workspace/`
   — including paths that are otherwise real-disk-backed. Using the wrong
   directory produces a misleading `RewardFileNotFoundError` even when
   the task/solution/verifier all ran correctly inside the container.

Still open for Phase 1 (not yet spiked): whether GG's DAG runner
(currently `sandbox/` scripts calling Vertex AI directly) needs any
changes to run headless inside a Docker container with Vertex AI
credentials mounted (ADC file or env-based auth). Do this as the first
step of Phase 1, using `/shared/workspace/` per the constraint above.

## Phase 1 — Wrap GG as a Harbor BaseAgent (single-turn parity)

Goal: prove the plumbing works before adding conversation complexity.

1. Write a `BaseAgent` subclass that takes an instruction, runs it through
   GG's existing DAG executor (reuse `run_dag_query` from
   `sandbox/20_procedural_graph_eval.py` on the
   `experiment/procedural-graphs-20260911` branch — it already supports
   both static and generative guidance modes), and returns the final
   response text.
2. Author a `task.toml` + `instruction.md` for ONE known case from the
   existing eval suite (suggest: case 15, "email notifications not
   working" — one of the three cases that showed the clarify-vs-checklist
   tension in the PG experiment).
3. Score it with a simple rewardkit criterion first (even just "did it
   mention checking spam folder") to confirm the reward-writing path
   works before building the harder judge.
4. Run `harbor run` end to end, confirm ATIF trajectory is logged and
   readable.

**Exit criteria:** one Harbor `Job` completes, produces a reward, and the
ATIF trajectory file is inspectable and matches what actually happened.

## Phase 2 — Add the simulated user (real multi-turn)

**Note:** RFC 0002 is status "Draft," and a patch
(`0002-simulated-users-patch.md`) supersedes its interaction mechanism —
build against the patch (`acpx` CLI + `--user-agent` flag), not the base
RFC's `chat`/`--user` (see research.md §3 for the full correction).

1. Study the `tau3-bench` adapter's actual task structure
   (`adapters/tau3-bench/` in the Harbor repo) as the concrete template —
   don't design a simulated-user mechanism from scratch, reuse theirs.
2. Read RFC 0002 (`rfcs/0002-simulated-users.md`) **and its patch**
   (`rfcs/0002-simulated-users-patch.md`) in full for the intended
   authoring pattern — the patch is the current mechanism, the base RFC
   alone is outdated on the transport layer.
3. Build a simulated-user persona for GG's ambiguous technical-support
   scenario: a "customer" who, when asked "what device/OS are you using?",
   answers with a specific realistic detail (e.g. "iPhone, iOS 17") rather
   than ending the conversation.
4. Wire this into a multi-step `task.toml` for case 15 (and optionally 14,
   16) using `[[steps]]` + `--resume-trajectory` so the DAG's conversation
   state carries across turns.

**Exit criteria:** a full simulated back-and-forth runs end to end — GG's
DAG asks a clarifying question, the simulated user answers, GG's DAG gives
a final targeted response — and the whole exchange is captured in one ATIF
trajectory.

## Phase 3 — Write the clarifying-question judge criterion

This is the one genuinely new piece (confirmed no off-the-shelf equivalent
exists in Harbor — see research.md §3).

1. Design the rubric: given an ATIF trajectory, does the agent (a) ask a
   clarifying question when the initial request was ambiguous (device/OS/
   error message not specified), AND (b) use the user's answer correctly
   in its final response? Score both halves — asking isn't enough if the
   final answer ignores what was asked.
2. Implement as a rewardkit `@criterion`-decorated LLM-judge, TOML-configured
   per Harbor's judge-criteria pattern.
3. Validate the criterion itself: hand-construct 2-3 synthetic trajectories
   (one that asks-then-uses-the-answer-well, one that asks-then-ignores-the-
   answer, one that never asks and just presumes) and confirm the judge
   scores them in the expected relative order BEFORE trusting it on real
   GG runs. (This mirrors the "smoke test the judge on adversarial pairs"
   pattern that caught the fabrication-rewarding bug in the prior branch's
   `22_grounded_eval.py` — do this again here, it's cheap and catches
   rubric bugs early.)

**Exit criteria:** the judge criterion reliably orders the 3 synthetic
trajectories correctly, unprompted, before being run on anything real.

## Phase 4 — Re-run baseline vs. PG-generative under the new harness

1. Pull `customer_support_adk.yaml` (baseline) and
   `customer_support_adk_pg.yaml` (PG generative guidance) from
   `experiment/procedural-graphs-20260911` — both already exist, no need
   to re-author.
2. Run both through the new multi-turn Harbor harness on cases 14/15/16
   (5 repetitions each, consistent with the prior branch's methodology).
3. Compare: does PG's edge-guidance ("ask clarifying questions about
   device/OS/error messages if not already provided") produce a
   measurably better multi-turn trajectory score than baseline, now that
   asking a good question can actually be rewarded instead of penalized?

**Exit criteria:** a real before/after number, with the same rigor as the
prior branch (repetitions, stdev, trajectory-level diffing on outliers —
don't stop at the aggregate if it looks surprising, dig into 2-3 actual
transcripts like the prior branch did for the referral-program and
mobile-app cases).

## Phase 5 — Decide on generalization

Only after Phase 4 has a real result:
- If PG's advantage shows up clearly: consider whether to extend this
  multi-turn harness to `it_helpdesk` and `sales_inquiry` too, and whether
  to formally adopt Harbor as GG's eval backbone going forward (would mean
  migrating `11_multi_harness.py`'s other cases).
- If it doesn't show up: this is still a valid, useful result — it would
  mean the clarify-vs-checklist tension wasn't actually PG-specific value,
  and the multi-turn capability is still worth keeping for future harness
  work regardless.

## Explicitly out of scope for this plan

- Migrating GG's full eval suite to Harbor on day one (see intent.md
  non-goals).
- Building the clarifying-question judge as a generalized GG feature
  before it's proven once, narrowly.
- Re-opening the Procedural Graphs paper analysis — that's closed on the
  prior branch.

## Reference material already committed (prior branch)

On `experiment/procedural-graphs-20260911` (not this branch — check it out
or cherry-pick specific files if needed):
- `sandbox/11_multi_harness.py` — base eval harness, fixture cases (now
  fact-grounded for cases 19/22/9, see commit "fix: eliminate
  fabrication-rewarding ambiguity...")
- `sandbox/20_procedural_graph_eval.py` — baseline vs static vs generative
  guidance A/B harness, `run_dag_query` is directly reusable
- `sandbox/22_grounded_eval.py` — fact-grounded judge pattern (reference
  for how to smoke-test a judge on adversarial pairs before trusting it —
  same technique recommended in Phase 3 above)
- `sandbox/company_facts.py` — ground-truth facts used to fix the
  fabrication-rewarding rubric bug
- `sandbox/customer_support_adk.yaml` / `_pg.yaml` — the two DAG configs
  to compare
- `sandbox/pg_eval_*.json`, `sandbox/grounded_eval_*.json` — all prior
  eval run data, including the exact per-case trajectories for cases
  14/15/16 that motivated this whole branch
