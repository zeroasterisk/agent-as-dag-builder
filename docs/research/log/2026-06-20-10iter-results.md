# Research Log: 10-Iteration Hill Climb Results

**Date:** 2026-06-20
**Status:** Complete — key findings documented

## Results

| Iter | CS | IT | Sales | Aggregate | Promoted? |
|------|----|----|-------|-----------|-----------|
| 1 | 83 | 84 | 72 | 80.1 | - |
| 2 | 90 | 80 | 86 | 85.8 | YES |
| 3 | 84 | 78 | 84 | 82.3 | NO |
| 4 | 87 | 81 | 84 | 84.3 | YES |
| 5 | 90 | 82 | 82 | 85.2 | YES |
| 6* | 85 | 74 | 85 | 81.7 | NO |
| 7 | 80 | 67 | 80 | 75.8 | NO |
| 8 | 80 | 67 | 81 | 76.6 | YES |
| 9 | 81 | 68 | 78 | 76.0 | NO |
| 10 | 81 | 67 | 80 | 76.6 | YES |

*Model changed from gemini-2.5-flash to gemini-3.5-flash between iterations 5-6

## Key Findings

### 1. Hill climbing works for 2-5 iterations, then plateaus or degrades
- Iterations 1-5: +5.1 points (80.1 → 85.2) — genuine improvement
- Iterations 6-10: -8.6 points from peak — degradation from over-modification

### 2. The learning loop over-modifies
- Modifies ALL harnesses every iteration, even well-performing ones
- No "best config" tracking — always modifies the latest, not the best
- Cascading changes across harnesses cause instability

### 3. Mid-run model change caused a discontinuity
- Switching from 2.5-flash to 3.5-flash between iters 5-6 dropped IT helpdesk from ~82 to ~67
- Different model + different judge = non-comparable scores
- Lesson: model must be constant within an experiment run

### 4. Regression gate works but isn't sufficient
- Blocked 4/9 promotions correctly
- But 2-point tolerance is too loose given 5-10 point scoring variance
- Gate should compare against best-ever, not previous iteration

## Fixes Needed for Next Run

1. **Best-config tracking**: always compare against and revert to the best-ever config
2. **Stability zone**: don't modify harnesses within 2 points of their best
3. **One-at-a-time**: only modify the weakest harness per iteration
4. **Constant model**: lock model for the entire experiment
5. **Tighter gate**: 1-point tolerance, compare against best-ever not previous

## Scoring Improvements Applied

- Category correctness: 50 points (up from 40)
- LLM quality: 30 points (down from 40)
- Helpfulness: 20 points (unchanged)
- Temperature=0 for judge calls
- 2x scoring with averaging
- YAML validation with rejection on invalid output
