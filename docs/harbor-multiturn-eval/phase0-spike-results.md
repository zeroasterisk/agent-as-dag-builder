# Phase 0 Spike Results — Harbor Docker/Container Execution Model

Status: **PASSED — Harbor works in this environment**, with one required
environment-specific workaround documented below. De-risking complete;
safe to proceed to Phase 1.

## What was tested

1. `uv tool install harbor` — clean install, v0.23 (matches research.md's
   "v0.18→v0.23 Jul–Sep 2026" observation, no drift).
2. `harbor init --task alan/hello-world` — scaffolds `task.toml`,
   `instruction.md`, `environment/Dockerfile`, `solution/solve.sh`,
   `tests/test.sh` + `tests/test_outputs.py` (pytest variant), exactly as
   research.md §2 described.
3. Wrote a trivial task (write "hello harbor" to a file, verify it),
   ran `harbor run --agent oracle --path ./hello-world`.
4. **Result: reward 1.0/1.0.** Full pipeline confirmed working — Docker
   image build, oracle solution execution, pytest-based verifier run,
   reward file collection back to the host job directory.

## Two real bugs found and root-caused (not GG-specific, environment-specific)

### Bug 1 — `~/.cache` not writable by the sandbox user (blocks every `harbor` invocation)

`harbor`'s notification/state cache path is hardcoded under `~/.cache/harbor`
(no env override — checked `harbor/constants.py`, no `XDG_CACHE_HOME`
support). In this sandbox, `~/.cache` is owned by `root`, not the `node`
user harbor runs as, so **every** `harbor run` fails immediately on a
permissions error before touching Docker at all.

**Workaround:** run harbor with `HOME` redirected to a writable scratch
dir that has `.cache/harbor` pre-created, e.g.:
```
mkdir -p /tmp/harbor-home/.cache/harbor
HOME=/tmp/harbor-home harbor run ...
```

**Side-effect to watch:** redirecting `HOME` also hides
`~/.docker/cli-plugins/docker-compose` from Docker's plugin discovery,
which makes `docker compose` silently fall back to a broken built-in that
doesn't understand `--project-name` (looks like a Docker/Compose version
incompatibility but isn't one — confirmed the real
`docker-compose` v2 plugin is fully compatible with Harbor's compose
invocations when discoverable). Fix: copy or symlink the cli-plugin into
the redirected `HOME` too:
```
mkdir -p /tmp/harbor-home/.docker/cli-plugins
cp ~/.docker/cli-plugins/docker-compose /tmp/harbor-home/.docker/cli-plugins/
```

### Bug 2 (the real structural finding) — Docker bind mounts don't work from this sandbox's default working directories

This sandbox's Docker access goes through a **separate Docker-in-Docker
daemon** (`DOCKER_HOST=tcp://sandbox-dind:2375`), not a local socket. The
DinD daemon has its own root filesystem, entirely independent of this
container's filesystem. `docker run -v <path>:/mnt` bind mounts are
resolved against *the daemon's* filesystem — so any path that only exists
inside our own container (which is most of the filesystem, including
`/home/node/work` and even the real-disk-backed `/home/node/agent`) is
invisible to the daemon and mounts as empty.

Confirmed via direct test: `docker run -v /home/node/agent/X:/mnt alpine
cat /mnt/...` fails even though `/home/node/agent` is genuinely a real
disk-backed mount (`/dev/sda1`) in *this* container — because DinD doesn't
share that mount namespace either.

This silently broke Harbor's verifier: the oracle solution ran
successfully, `test.sh` ran successfully and wrote `reward.txt` — but
**inside the container**, at a path Harbor's bind mount was supposed to
sync back to the host job directory. Since the bind mount was a no-op,
Harbor's host-side check for `reward.txt` found nothing and raised
`RewardFileNotFoundError`, even though the task itself passed. This is a
misleading failure mode: it looks like the task/verifier config is broken
when the actual root cause is environment path mapping.

**The fix — use `/shared/workspace` instead of `/home/node/work`.**
`/shared/workspace` (also real-disk-backed, `/dev/sda1`) *is* visible to
the DinD daemon:
```
mkdir -p /shared/workspace/bindtest && echo test > /shared/workspace/bindtest/marker.txt
docker run --rm -v /shared/workspace/bindtest:/mnt alpine cat /mnt/marker.txt
# -> "test" (works)
```
vs. the same test under `/home/node/work` or `/home/node/agent` returns
"No such file or directory" every time.

**Action for all future Harbor work:** any Harbor task directory MUST live
under `/shared/workspace/` (not `/home/node/work/` or elsewhere) for the
verifier/reward bind mount to function. Confirmed end to end: task moved
to `/shared/workspace/harbor-spike/hello-world`, identical `harbor run`
invocation, reward correctly collected as `1.0`.

## Bottom line for Phase 1+

- Harbor's CLI, Docker image build, agent execution, and verifier/reward
  pipeline all work correctly in this sandbox.
- No changes needed to Harbor itself or to GG's DAG runner architecture to
  make this work — this was purely a sandbox path-mapping issue.
- **Constraint for all subsequent phases:** run Harbor task directories
  and `harbor run` invocations from under `/shared/workspace/`, with
  `HOME` pointed at a scratch dir containing `.cache/harbor` and a copy of
  `~/.docker/cli-plugins/docker-compose`.
- The Docker/container assumption in plan.md's Phase 0 is de-risked. No
  need to fall back to Inspect AI (research.md §5) on infra-weight
  grounds — the actual weight was a fixable sandbox quirk, not Harbor
  being a poor fit.
