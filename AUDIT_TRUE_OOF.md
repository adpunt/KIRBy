# AUDIT: True per-variant training-side OOF predictions

**Date:** 2026-05-08
**Script:** `tests/experiments/annitia/reverse_fusion.py`
**Outputs:**
- 84 `.npy` files at `tests/results/annitia/full/proxy_train_oof_<variant>_<endpoint>.npy`
- Differentiated ranking at `tests/results/annitia/full/per_variant_true_oof_C.csv`

## TL;DR

The bug: `oof_to_leaderboard.py::variant_proxy_oof` collapsed 14 of the
top-15 leaderboard variants onto the SAME training-side OOF (rank-mean
of base OOFs). All meta-aware / NNLS / isotonic / weighted-bucket
machinery is invisible on the training side, so bootstrap CIs and
holdout fusion-tests rank every variant identically.

The fix: re-apply each variant's actual fusion logic to the training-
side OOF arrays, producing a per-variant true training-side prediction.

Results:
- 42 variants produced (16 base × 2 endpoints + 26 meta × 2 endpoints
  with NNLS reconstruction).
- The previously-collapsed top-15 now span training-side
  weighted-LB ∈ [0.918, 0.931] (a real spread, not a single number).
- Spearman ρ(true-OOF LB-proxy, real LB) = **−0.20** on the 9 LB-mapped
  variants. This is *not* a clean positive correlation, which itself is
  a useful finding (see below).

## Method

Two paths.

### Path A — direct replay (16 variants)

`refuse_variants.py` and `refuse_meta_aware.py` define functions that
already compute the variant's training-side OOF prediction internally
as part of their honest-OOF C scoring step. We import those functions
and harvest the `'oof'` array directly:

```
variant_mean, variant_median, variant_trimmed, variant_residual_greedy,
variant_uncertainty_route, variant_anomaly_blend, variant_disagreement_route,
variant_cluster_route, variant_specialist_override,
variant_soft_blend_uncertainty
```

For the meta-aware family (`anomaly_router_meta`, `disagreement_router_meta`,
`cluster_router_meta`, `anomaly_blend_meta`, `weighted_anomaly_per_patient`),
the bucket-routing arithmetic is replayed inline using the already-
computed base-variant OOFs (which themselves came from path A above).

`coxnet_stack_with_uncertainty` was deferred (its nested-CV refit is
heavier and the script timeboxes; can be added with a flag).

### Path B — NNLS reconstruction (26 meta variants)

For variants that don't have a direct training-OOF replay path (the
`meta_avg_*` family, `FINAL`, `best_combined`, `BEST_BET`,
`ultra_meta`, `isotonic_avg`, `nnls_stack`, `bootstrap_dropout`,
`holdout_pick`, `diverse`, `HIER`, `honest`):

1. Read the variant's `sub_master_<v>.csv` val_test predictions.
2. Stack the val-test rank vectors of the 10 base variants into a
   matrix `V_base ∈ R^{n_val × 10}`.
3. Solve NNLS: `target_val_ranks ≈ V_base @ w`, normalise `w` to sum 1.
4. Apply the same `w` to the base-variant TRAINING-side rank matrix
   `T_base ∈ R^{n_train × 10}` to obtain the training-side reconstruction
   `T_base @ w`.

This is a data-driven model-of-the-fusion. It does NOT recover the
exact fusion arithmetic; it recovers the implied linear combination
that best reproduces the val predictions. Reconstruction quality
(Spearman ρ between recovered and original val predictions) is logged
per variant — most are above 0.97, some are 1.00 (the variant IS one
of our base variants), and a couple are 0.91-0.93 (NNLS is incapable
of capturing the true non-linearity).

## Headline numbers

Top 15 variants by true training-side weighted LB-proxy:

| Variant | hep_C | death_C | LB-proxy |
|---|---|---|---|
| anomaly_blend_meta | 0.9139 | 0.9697 | **0.9306** |
| cluster_router_meta | 0.9116 | 0.9698 | 0.9290 |
| disagreement_router_meta | 0.9094 | 0.9710 | 0.9279 |
| meta_avg | 0.9064 | 0.9734 | 0.9265 |
| anomaly_router_meta | 0.9094 | 0.9662 | 0.9264 |
| meta_avg_aware_weighted_no_nnls_v3 | 0.9063 | 0.9722 | 0.9261 |
| meta_avg_weighted | 0.9055 | 0.9740 | 0.9261 |
| meta_avg_aware_weighted_v3 | 0.9060 | 0.9725 | 0.9260 |
| meta_avg_top5 | 0.9053 | 0.9741 | 0.9259 |
| meta_avg_grand_all_v3 | 0.9047 | 0.9730 | 0.9252 |
| diverse | 0.9038 | 0.9730 | 0.9246 |
| rank_median | 0.9046 | 0.9707 | 0.9244 |
| weighted_anomaly_per_patient | 0.9036 | 0.9722 | 0.9242 |
| rank_mean | 0.9029 | 0.9730 | 0.9239 |
| rank_trimmed | 0.9029 | 0.9730 | 0.9239 |

The full table is at `tests/results/annitia/full/per_variant_true_oof_C.csv`.

## Leaderboard cross-check

Comparing the new training-side LB-proxy against the actual public
leaderboard scores (only 9 variants are in `leaderboard_history.json`):

| Variant | TRUE_LB_proxy | REAL_LB | gap |
|---|---|---|---|
| meta_avg | 0.9265 | 0.9269 | −0.0004 |
| meta_avg_weighted | 0.9261 | 0.9284 | −0.0024 |
| meta_avg_top5 | 0.9259 | 0.9278 | −0.0019 |
| rank_median | 0.9244 | 0.9112 | +0.0132 |
| FINAL | 0.9212 | 0.9268 | −0.0056 |
| meta_avg_all | 0.9200 | 0.9294 | −0.0094 |
| meta_avg_grand_all | 0.9199 | 0.9294 | −0.0094 |
| meta_avg_aware_weighted | 0.9195 | 0.9302 | −0.0107 |
| best_combined | 0.9162 | 0.9123 | +0.0039 |

**Spearman ρ(TRUE_OOF_LB, REAL_LB) = −0.20.**

Compare to the OLD proxy (which collapses everything onto one number,
so ρ is effectively undefined / 0 within numerical noise).

## What the −0.20 ρ tells us

A *negative* correlation between training-side OOF and held-out LB is a
striking finding. Three interpretations:

1. **NNLS reconstruction noise.** The reconstruction is data-driven on
   val space. For variants whose val rank pattern is noisy or close to
   another variant's, NNLS picks a different mixture than the
   underlying fusion did. The training-side OOF then reflects a
   *different* recipe than the leaderboard recipe. This noise alone
   could swamp the small signal in a 9-point sample.

2. **Anti-overfitting.** Variants like `meta_avg_aware_weighted` and
   `meta_avg_grand_all` use heavier regularisation / shrinkage that
   *lowers* training-side OOF C but *raises* held-out generalisation.
   Their val_test predictions push toward consensus — which is what
   wins the LB. The OOF would naturally be lower for these.

3. **Leaderboard label noise.** With 9 points and observed gaps of
   ±0.013, any one variant's rank can flip easily. This is not a
   reliable signal.

The correct read: these training-side OOFs are now **DIFFERENTIATED**
(no longer collapsed). Variant ranking on training-side OOF is now a
meaningful object — but it is NOT a reliable leaderboard predictor.
The next tooling layer (bootstrap CIs and holdout fusion-tests) can
now operate on these differentiated arrays and the within-variant CI
will collapse against the across-variant spread.

## Files written

```
tests/results/annitia/full/proxy_train_oof_<variant>_<endpoint>.npy
    -- 84 files (42 variants × 2 endpoints).
tests/results/annitia/full/per_variant_true_oof_C.csv
    -- 42 rows, sorted by weighted_LB descending.
```

## Caveats and follow-ups

- **NNLS-on-val is approximate.** For 6/26 meta variants the
  reconstruction Spearman ρ is below 0.95. These training-side OOFs
  should be flagged as approximate. A future version could fit
  isotonic-on-rank or LightGBM-as-meta to capture non-linearities.

- **`coxnet_stack_with_uncertainty` was deferred.** Its nested-CV
  refit is the only meta-aware variant that doesn't replay cleanly.
  Add a `--include-coxnet` flag if needed.

- **The script does NOT modify any existing `refuse_*.py`.** It
  imports the variant functions and harvests their internal OOF
  arrays. Existing val_test outputs are untouched.

- **Both endpoints use the same anomaly_train and disagreement
  signals.** This matches the original implementations.

- **The `valid_mask` differs per endpoint** (hep n=1106 valid, death
  n=966 valid). C-index uses the per-endpoint mask consistently.

## Bottom line

Local validation can now distinguish between variants that previously
shared a single proxy OOF. The refuse_bootstrap, refuse_holdout, and
oof_to_leaderboard scripts can now consume `proxy_train_oof_*.npy` per
variant and produce non-degenerate CIs / rankings. The variant_proxy_oof
helper in `oof_to_leaderboard.py` should be replaced with a lookup of
these `.npy` files (with a fallback to the consensus rank-mean for
unrecognised variants).
