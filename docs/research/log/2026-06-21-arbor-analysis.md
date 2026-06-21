# Research Log: Arbor Framework Analysis — Ramifications for Graph Gardener

**Date:** 2026-06-21
**Paper:** "Toward Generalist Autonomous Research via Hypothesis-Tree Refinement" (arXiv:2606.11926)
**Authors:** Renmin University of China + Microsoft Research
**Code:** https://github.com/RUC-NLPIR/Arbor

## Key Concepts

### Hypothesis-Tree Refinement (HTR)
Instead of linear optimization (v1→v2→v3), Arbor organizes work as a branching tree where:
- Each branch is a hypothesis (proposed change)
- Branches are tested in isolated environments (git worktrees)
- Failed branches are pruned, successful ones harvested
- Insights propagate back so later hypotheses start smarter

### Coordinator/Executor Separation
- **Coordinator** (long-lived): strategic direction, hypothesis generation, evidence evaluation
- **Executors** (short-lived): isolated implementation + testing + reporting
- Coordinator never touches code directly

### Results
- 2.5× average performance gain vs Claude Code and Codex on same compute budget
- BrowseComp: 45.3% → 67.7% (competitors stalled at 50-53%)
- MLE-Bench Lite: 86.36% Any Medal (strongest in comparison)

## Mapping to Graph Gardener

| Arbor Concept | GG Equivalent | Gap |
|---------------|---------------|-----|
| Hypothesis tree | Linear iteration (v1→v2→v3) | Need branching |
| Isolated worktrees | In-place config modification | Need parallel configs |
| Coordinator | Learning loop | Need separation from execution |
| Executor | DAG execution | Already separated |
| Pruning | Conservative optimizer | Already have |
| Insight propagation | Additive templates | Have basic version |
| Held-out validation | Same test set for both | Critical gap |

## Action Items

1. **Held-out validation**: Split test cases 70/30 — optimize on 70%, validate on 30%. This is likely why scores oscillate.
2. **Parallel branching**: Try 2-3 template variants per iteration instead of 1.
3. **Insight memory**: Track WHAT types of additions helped (not just best score) and carry forward.
4. **Consider Arbor integration**: Use Arbor as the optimization engine with GG's YAML DAGs as the target codebase.
