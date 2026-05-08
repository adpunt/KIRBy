# V08 — Held-out fusion-test gate

**Type:** Validation protocol (selection-bias control)
**Domain:** Any competition that uses ensembling/fusion of OOF predictions

## What it is
Carve out 5–10% of training data **before any fusion logic touches it**. All candidate fusion variants must beat the anchor on this set, not on training OOF, to qualify for submission. The hold-out is never used to *fit* any fusion weight — only to *evaluate* candidates.

## Why it helps
Greedy max-OOF selection over a large candidate library is mathematically guaranteed to overfit. Caruana 2004 showed selection from 100+ models inflates apparent performance by 0.01–0.03 even when each candidate is honest. A held-out set the candidates have never seen breaks the selection-OOF correlation.

## Why bootstrap-CI on training OOF is *not* a substitute
Bootstrap CI measures *stability* of the OOF metric. It does not measure *transfer* to unseen data. A variant that's stably overfit will have a tight, high CI on training OOF and still lose on a held-out gate. We learned this the hard way (ANNITIA cherry_pick: 99% CI but LB regression of 0.034).

## When to use
- Always, when fusing 5+ candidates
- Any time a meta-learner or selection process touches OOF predictions

## When not to use
- True hold-out leaderboard sets are already provided
- Tiny training data (<300 patients) — hold-out cost too high; use V09 nested CV instead

## Implementation
```python
# Carve hold-out ONCE, persist random seed
rng = np.random.default_rng(2026)
holdout_idx = rng.choice(n_train, size=int(0.1 * n_train), replace=False)
fit_idx = np.setdiff1d(np.arange(n_train), holdout_idx)

# Run K-fold CV / fusion fitting ONLY on fit_idx
# Evaluate each fusion candidate on holdout_idx
# Anchor = current LB-best recipe
# Gate: candidate.holdout_C > anchor.holdout_C - eps
```

## Gate logic
- ε = 1× SE of bootstrap CI on hold-out, NOT 0
- If candidate is within ε of anchor, prefer the more diverse one (broader base coverage)
- If no candidate beats anchor on hold-out → don't submit anything new

## Trade-offs
- Costs ~10% of training signal — acceptable
- Hold-out itself has variance; for n < 1000, pair with V01 repeated K-fold *within fit_idx*
- Strict adherence requires discipline: it's tempting to "just check" hold-out C while iterating on fusion design — that contaminates it

## Empirical results (fill in as tested)
| Competition | Date | Primary metric | LB before → after | Notes |
|---|---|---|---|---|
| ANNITIA MASLD | 2026-05-08 | Combined LB (0.7C_hep + 0.3C_death) | TBD | Built but not enforced for cherry_pick — caused 0.034 regression |

---
[← back to index](../README.md)
