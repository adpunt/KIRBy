# V10 — Permutation null distribution for ΔC

**Type:** Validation diagnostic (significance test)
**Domain:** Any competition where candidates are ranked by a continuous metric and small differences are claimed
**Cite:** Standard nonparametric inference; Good, *Permutation Tests* (Springer)

## What it is
For two candidates A and B with predictions p_A, p_B and a metric M (e.g. C-index):
1. Compute observed ΔC = M(p_A, y) − M(p_B, y).
2. For r = 1..R (typically 1000): randomly swap each patient's prediction between A and B with prob 0.5, recompute ΔC_r.
3. Empirical p-value = fraction of |ΔC_r| ≥ |ΔC_obs|.

Equivalently, for a single candidate vs anchor: shuffle prediction labels within the candidate, recompute ΔC against anchor.

## Why it helps
We routinely treat ΔC of 0.005 as "evidence of improvement". The permutation null tells you what ΔC values arise *by chance* under the same data and same candidate complexity. Often the null spans ±0.015 — meaning a 0.005 lift is well within noise.

## When to use
- Comparing fusion candidates whose OOF C differs by < 0.01
- Evaluating "best of top-3" claims
- Building publication-grade significance evidence

## When not to use
- Large ΔC (> 0.05) — clearly significant, don't waste compute
- Adversarial / non-exchangeable settings

## Implementation
```python
def permutation_p(p_A, p_B, y_struct, R=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(p_A)
    obs = c_index(p_A, y_struct) - c_index(p_B, y_struct)
    null = []
    for _ in range(R):
        mask = rng.integers(0, 2, n).astype(bool)
        a = np.where(mask, p_B, p_A)
        b = np.where(mask, p_A, p_B)
        null.append(c_index(a, y_struct) - c_index(b, y_struct))
    return obs, np.mean(np.abs(null) >= abs(obs))
```

## Decision rule
- p < 0.05: ΔC is unlikely under the null; treat as real
- p > 0.20: indistinguishable from noise; don't pick the OOF winner
- 0.05 ≤ p ≤ 0.20: ambiguous; use multiple methodological families (V01, V02, V08) before deciding

## Trade-offs
- R=1000 ⇒ 1000 metric recomputes; trivially parallel
- Tests the *paired* difference (correct for OOF on same patients)
- Doesn't address selection bias from picking the *max-of-many* candidates — pair with V08

## Empirical results (fill in as tested)
| Competition | Date | Primary metric | LB before → after | Notes |
|---|---|---|---|---|

---
[← back to index](../README.md)
