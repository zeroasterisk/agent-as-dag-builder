# Research — Harbor Evaluation Framework

Compiled 2026-09-12. Findings below combine a subagent research pass with
independent verification via direct GitHub API calls (not taking the
subagent's self-report at face value — repo existence, star count, license,
and specific file paths were re-checked live).

## 1. Identity (verified)

- **Repo**: `github.com/harbor-framework/harbor`
- **License**: Apache-2.0 (confirmed via GitHub API)
- **Stars**: 5,167 / **Forks**: 1,776 / **Open issues**: 863 (as of 2026-09-12)
- **Description** (from GitHub API): "Framework for evaluating and improving agents"
- **Activity**: pushed to same-day as research (2026-09-12T19:14:27Z) — actively maintained, releases roughly biweekly (v0.18→v0.23 Jul–Sep 2026)
- **Homepage**: harborframework.com
- **Install**: `pip install harbor` / `uv tool install harbor`
- **Lineage**: built by the Terminal-Bench team (`terminal-bench-1`/`terminal-bench-2`, 2,576★) as the generalized successor harness; Harbor is now the official Terminal-Bench-2.0 runner. Also owns `harbor-cookbook`, `harbor-datasets`.
- **Not** an AI21 project — no such thing found. Name collision risk noted and ruled out.
- Repo top-level structure (verified via API): `adapters/`, `apps/`, `docs/`, `docs-mintlify/`, `examples/`, `packages/`, `rfcs/`, `scripts/`, `skills/`, `src/`, `tests/`, plus `registry.json`, `CITATION.cff`, `AGENTS.md`/`CLAUDE.md` (agent-facing contributor docs).

## 2. Core architecture

- **Task** = a directory: `task.toml` (config) + `instruction.md` (the prompt) + `environment/` (Dockerfile defining the sandbox) + `solution/` (reference solution) + `tests/` (verifier).
- **Environment** = the sandboxed execution context. Default: Docker. Also supports cloud sandboxes behind a common `BaseEnvironment` interface: Daytona, Modal, E2B, Runloop, LangSmith, Blaxel, Novita, EC2, Beam, Vercel Sandbox, GKE/TPU.
- **Agent** = `BaseAgent` (external driver, controls the env via exec calls) or `BaseInstalledAgent` (agent binary lives inside the container, runs headless). ~15 pre-integrated agents ship out of the box (Claude Code, Codex, Gemini CLI, OpenHands, etc.).
- **Scoring**: the verifier writes `reward.txt` / `reward.json` into `/logs/verifier/` inside the container. A separate, standalone-usable package `rewardkit` provides:
  - Programmatic criteria (file exists, command exit code, JSON diff, HTTP response, image diff, etc. — 20+ built-ins)
  - LLM-judge criteria (TOML-defined rubrics)
  - Agent-judge criteria
  - **Trajectory-level checks** (a built-in category, not something you build from scratch — e.g. `trajectory_turn_count`)
- **Trial** = one agent attempt at one task → produces one reward.
- **Job** = a batch of trials across a matrix of {tasks × agents × models}, launched via `harbor run -c job.yaml`.

## 3. Multi-turn support — confirmed native, two mechanisms

**(a) Multi-step tasks.** `task.toml` can define `[[steps]]`: an ordered list
of steps sharing one environment. Each step has its own instruction/tests/
setup. A `--resume-trajectory` flag controls whether the agent's own
conversation state carries over between steps (continuous conversation) or
resets (fresh context per step). Per-step `min_reward` can gate early
stopping. Trial-level reward aggregates across steps via `mean` or `final`
strategy.

**(b) ATIF — Agent Trajectory Interchange Format** (RFC 0001, status
**Active**, v1.8 — full text independently re-read by me on 2026-09-13,
not just the subagent's paraphrase).
A standardized JSON schema explicitly designed, per its own stated purpose,
to unify "single-turn tasks and multi-turn conversational interactions."
Key fields (confirmed by direct read): each step has a `source`
(system/user/agent), `tool_calls`, `observation`, per-step `metrics`
(token counts, cost, `logprobs`, token IDs — **note: no per-step `reward`
field exists in ATIF**; reward/scoring lives in rewardkit and the
trial/job layer, not in the trajectory schema itself), `is_copied_context`
(context-compression tracking), and `subagent_trajectories` (hierarchical/
multi-agent delegation — directly relevant to GG's DAG-routed handler
structure, and resolved via a `trajectory_id`, not `session_id`, per v1.7).

**(c) Conversational-agent adapters that already exist** (verified present
via API listing of `adapters/`): `tau3-bench`, `textarena`, `locomo`,
`gaia`, `gaia2`.
- `tau3-bench`: agent vs. simulated user, "half-duplex conversation,"
  binary + pass^k reward computed by the official tau2 evaluator over the
  full recorded conversation. **This is structurally identical to GG's
  problem** — a customer-service-style agent that needs to hold a real
  back-and-forth, not answer in one shot. (README independently read:
  MIT-licensed, built on `sierra-research/tau2-bench`, 375 tasks across
  4 domains, ships a Dockerized runtime + MCP sidecar.)
- RFC 0002 (`rfcs/0002-simulated-users.md`) defines "simulated users" as a
  Harbor concept — **both RFC 0002 and its patch fully re-read by me,
  not relayed from the subagent.** Two corrections to flag:
  1. **RFC 0002's status is "Draft"**, not finalized/merged — treat the
     mechanism below as a proposal under review, not a settled API.
  2. **A patch (`0002-simulated-users-patch.md`) supersedes the base
     RFC's interaction mechanism.** Base RFC 0002 proposes a bespoke
     `chat "<message>"` CLI wrapper for the user agent to talk to the
     target agent. The patch replaces this with driving the target
     through the real **acpx** CLI (an existing third-party ACP client,
     https://acpx.sh) instead of a hand-rolled wrapper — this was explicit
     PR review feedback (forcing communication through a custom `chat`
     tool "risks unnatural model behavior"). The patch also renames the
     `--user` flag to `--user-agent` (clearer, avoids collision with an
     existing `harbor job share --user` flag). **Anyone implementing
     Phase 2 should build against the patch's acpx-based mechanism and
     `--user-agent` flag, not the base RFC's `chat`/`--user`.**
  Mechanism (patched version): Harbor generates a per-trial
  `.acpxrc.json` pinning the target agent + policy (`approve-all`
  permissions, `quiet` output format), starts `acpx sessions ensure`,
  then runs the user agent with `acpx prompt "<msg>"` as its only new
  tool — the user agent's own agentic loop drives the whole conversation
  turn-by-turn; Harbor does not orchestrate turns itself. This is the
  mechanism needed to make a fake customer answer "which device are you
  on?" instead of the conversation ending at turn 1.

**Gap, confirmed real (not hand-waved):** there is no built-in "reward
asking a good clarifying question" scoring primitive. `trajectory_turn_count`
exists as a rewardkit criterion (counts turns) but doesn't judge quality of
a turn. This must be authored as a custom LLM-judge criterion in rewardkit
that reads the ATIF trajectory turn-by-turn and scores whether a clarifying
question was warranted and well-targeted. This is the one non-trivial,
genuinely new piece of work — everything else needed (simulated user,
multi-turn orchestration, trajectory logging) is off the shelf.

## 4. Authoring model

- Config-first: TOML for tasks (`task.toml`) and judge criteria. Scaffolding
  via `harbor init --task`.
- Python SDK for anything custom: `@criterion` decorator for judge logic,
  `BaseAgent`/`BaseEnvironment` for custom agents/environments.
- No heavyweight plugin registry — adapters (tau3-bench, gaia2, etc.) are
  just task-generator scripts living under `adapters/`, not a formal plugin
  API. Low ceremony to add a new one.
- Ships a Claude/Codex-agent skill (`npx skills add harbor-framework/harbor
  --skill create-task`) that interactively authors a task for you — worth
  trying before hand-writing `task.toml` from scratch.

## 5. Comparison to alternatives

| Framework | Multi-turn model | Fit for GG |
|---|---|---|
| **Harbor** | Container/trial-based; multi-turn via multi-step tasks + resume-trajectory, or agent's own loop against a simulated-user sidecar (tau3-bench pattern) | Strong — direct precedent (tau3-bench) for GG's exact shape, but container-first adds infra weight |
| **Inspect AI** (aisi.org.uk) | In-process: `Solver`/`Agent` operates on a persistent `AgentState.messages` list inside a `generate_loop`; built-in ReAct/Deep Agent, multi-agent composition, checkpointing, mid-run intervention | Architecturally the most natively "multi-turn conversational" of the group — lower infra overhead, turn-by-turn scoring more idiomatic. Worth a second look if Harbor's container requirement proves too heavy. |
| **AgentBench** | Single benchmark suite, not a general harness | Not a fit as infra — Harbor already has an AgentBench-style adapter if specific tasks are wanted |
| **OpenAI Evals** | Single-shot prompt → grade, no native trajectory/agent model | Weakest fit — exactly the limitation we're trying to escape |

**Bottom line on alternatives:** if Harbor's Docker-centric model turns out
to be a poor match for how GG's DAG runner is deployed, Inspect AI is the
credible fallback — it's less infrastructure but has no ready-made
tau3-bench-equivalent adapter to crib from, so more of the simulated-user
mechanism would need to be built from scratch there.

## 6. Integration effort estimate (for GG specifically)

- **Low-medium (days, not weeks)** for a first working single-turn parity
  eval: wrap GG's DAG runner as a `BaseAgent` — receives an instruction,
  runs the DAG, returns output. This alone doesn't unlock multi-turn, but
  validates the container/wiring works before adding conversation.
- **Medium effort** for the actual multi-turn clarifying-question goal:
  1. Author a task modeled on `tau3-bench` (simulated user via MCP sidecar
     or scripted turn injection, per RFC 0002).
  2. Wire GG's `customer_support_adk.yaml` DAG (and its `_pg.yaml` variant
     from the prior branch) as the agent under test.
  3. Write the one custom rewardkit judge criterion for "asked a
     well-targeted clarifying question vs. gave a premature complete
     answer" — no off-the-shelf equivalent exists, this is the core
     net-new build.
- **Good structural fit**: GG's node-classification routing maps cleanly
  onto Harbor's per-step `source: agent` steps with `tool_calls`/
  `observation`; ATIF's `subagent_trajectories` field is a good match if we
  want full trajectory fidelity across GG's DAG-routed handlers (classify →
  handle_technical, etc.) rather than collapsing the DAG traversal into one
  opaque "agent turn."
- **Main risk/cost**: Harbor assumes Docker/container-first execution. If
  GG's DAG runner (currently a set of Python scripts under `sandbox/`,
  called directly against Vertex AI) isn't already easy to containerize,
  that's the actual setup tax — not the eval logic itself. Worth spiking
  this specifically before committing further, since it's the one
  structural unknown.

## Open questions for whoever picks this up

1. Does GG's DAG runner need any changes to run headless inside a Docker
   container (env vars, credential mounting for Vertex AI auth, etc.), or
   does it already work that way in `sandbox/19_e2e_full_loop.py`'s
   execution model?
2. Is `tau3-bench`'s simulated-user mechanism (RFC 0002) sufficient as-is,
   or does GG's specific ambiguous-technical-support scenario (cases 14/15/16
   from the prior branch's customer_support harness) need a custom
   simulated-user persona rather than reusing tau3-bench's out of the box?
3. Should the new clarifying-question judge criterion be scoped narrowly
   (just these 3 technical cases) first, or designed from day one to
   generalize across `it_helpdesk`/`sales_inquiry` too?
