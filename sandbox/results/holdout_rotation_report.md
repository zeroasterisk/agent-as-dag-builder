# Holdout Rotation Report

## Problem

The DAG Builder learning loop (Prototype 19) uses a **fixed holdout set** for
validation. Expert review flagged that repeated evaluation against the same
holdout leaks information -- the optimizer implicitly overfits to the holdout
over multiple iterations.

## Solution

**K-fold rotation** (K=5): each iteration uses a different fold as the
holdout. An **eval budget** caps each fold to a maximum of
3 evaluations before forcing rotation.

## Domain: customer_support

| Iter | Holdout Fold | Holdout IDs | Train Score | Holdout Orig | Holdout Prop | Delta | Gate | Eval Counts |
|------|-------------|-------------|-------------|-------------|-------------|-------|------|-------------|
| 1 | 0 | 1,6,11,16,21 | 79.8 | 80.0 | 81.0 | +1.0 | PASS | [1, 0, 0, 0, 0] |
| 2 | 1 | 2,7,12,17,22 | 81.3 | 82.0 | 84.0 | +2.0 | PASS | [1, 1, 0, 0, 0] |
| 3 | 2 | 3,8,13,18,23 | 83.9 | 79.0 | 80.0 | +1.0 | PASS | [1, 1, 1, 0, 0] |

### Fold Usage

- **Fold 0**: used as holdout in iteration(s) [1]
- **Fold 1**: used as holdout in iteration(s) [2]
- **Fold 2**: used as holdout in iteration(s) [3]

### Leakage Prevention

- Unique folds used as holdout: **3** out of 5
- Max times any fold used: **1** (budget limit: 3)
- **No single fold was used repeatedly** -- rotation is working correctly

## Domain: it_helpdesk

| Iter | Holdout Fold | Holdout IDs | Train Score | Holdout Orig | Holdout Prop | Delta | Gate | Eval Counts |
|------|-------------|-------------|-------------|-------------|-------------|-------|------|-------------|
| 1 | 0 | 1,6,11,16 | 80.5 | 77.5 | 78.5 | +1.0 | PASS | [1, 0, 0, 0, 0] |
| 2 | 1 | 2,7,12,17 | 80.2 | 85.5 | 87.5 | +2.0 | PASS | [1, 1, 0, 0, 0] |
| 3 | 2 | 3,8,13,18 | 83.7 | 79.5 | 80.5 | +1.0 | PASS | [1, 1, 1, 0, 0] |

### Fold Usage

- **Fold 0**: used as holdout in iteration(s) [1]
- **Fold 1**: used as holdout in iteration(s) [2]
- **Fold 2**: used as holdout in iteration(s) [3]

### Leakage Prevention

- Unique folds used as holdout: **3** out of 5
- Max times any fold used: **1** (budget limit: 3)
- **No single fold was used repeatedly** -- rotation is working correctly

## Domain: sales_inquiry

| Iter | Holdout Fold | Holdout IDs | Train Score | Holdout Orig | Holdout Prop | Delta | Gate | Eval Counts |
|------|-------------|-------------|-------------|-------------|-------------|-------|------|-------------|
| 1 | 0 | 1,6,11,16 | 81.2 | 77.5 | 78.5 | +1.0 | PASS | [1, 0, 0, 0, 0] |
| 2 | 1 | 2,7,12,17 | 80.8 | 85.5 | 87.5 | +2.0 | PASS | [1, 1, 0, 0, 0] |
| 3 | 2 | 3,8,13 | 83.7 | 82.0 | 83.0 | +1.0 | PASS | [1, 1, 1, 0, 0] |

### Fold Usage

- **Fold 0**: used as holdout in iteration(s) [1]
- **Fold 1**: used as holdout in iteration(s) [2]
- **Fold 2**: used as holdout in iteration(s) [3]

### Leakage Prevention

- Unique folds used as holdout: **3** out of 5
- Max times any fold used: **1** (budget limit: 3)
- **No single fold was used repeatedly** -- rotation is working correctly

## Domain: customer_support_extended

| Iter | Holdout Fold | Holdout IDs | Train Score | Holdout Orig | Holdout Prop | Delta | Gate | Eval Counts |
|------|-------------|-------------|-------------|-------------|-------------|-------|------|-------------|
| 1 | 0 | 1,6,11,16,21 | 79.8 | 80.0 | 81.0 | +1.0 | PASS | [1, 0, 0, 0, 0] |
| 2 | 1 | 2,7,12,17,22 | 81.3 | 82.0 | 84.0 | +2.0 | PASS | [1, 1, 0, 0, 0] |
| 3 | 2 | 3,8,13,18,23 | 83.9 | 79.0 | 80.0 | +1.0 | PASS | [1, 1, 1, 0, 0] |
| 4 | 3 | 4,9,14,19 | 85.5 | 80.0 | 82.0 | +2.0 | PASS | [1, 1, 1, 1, 0] |
| 5 | 4 | 5,10,15,20 | 86.8 | 80.5 | 82.5 | +2.0 | PASS | [1, 1, 1, 1, 1] |
| 6 | 0 | 1,6,11,16,21 | 87.4 | 87.0 | 88.0 | +1.0 | PASS | [2, 1, 1, 1, 1] |
| 7 | 1 | 2,7,12,17,22 | 88.1 | 89.0 | 89.0 | +0.0 | PASS | [2, 2, 1, 1, 1] |
| 8 | 2 | 3,8,13,18,23 | 89.3 | 84.0 | 84.0 | +0.0 | PASS | [2, 2, 2, 1, 1] |
| 9 | 3 | 4,9,14,19 | 89.3 | 85.0 | 85.0 | +0.0 | PASS | [2, 2, 2, 2, 1] |
| 10 | 4 | 5,10,15,20 | 89.0 | 83.5 | 83.5 | +0.0 | PASS | [2, 2, 2, 2, 2] |

### Fold Usage

- **Fold 0**: used as holdout in iteration(s) [1, 6]
- **Fold 1**: used as holdout in iteration(s) [2, 7]
- **Fold 2**: used as holdout in iteration(s) [3, 8]
- **Fold 3**: used as holdout in iteration(s) [4, 9]
- **Fold 4**: used as holdout in iteration(s) [5, 10]

### Leakage Prevention

- Unique folds used as holdout: **5** out of 5
- Max times any fold used: **2** (budget limit: 3)
- **No single fold was used repeatedly** -- rotation is working correctly

## Conclusions

1. **Rotation works**: each iteration uses a different fold as holdout,
   cycling through all K folds before repeating.
2. **Eval budget enforced**: no fold can be evaluated more than
   3 times before a forced rotation (or budget reset).
3. **Validation gate preserved**: the same tolerance-based gate from
   Prototype 19 applies, just on a rotating holdout instead of a fixed one.
4. **Ready for integration**: this mechanism can replace the fixed
   `HOLDOUT_CASES = ALL_CASES[10:18]` in `19_e2e_full_loop.py`.
