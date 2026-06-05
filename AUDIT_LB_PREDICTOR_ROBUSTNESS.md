# AUDIT — LB Predictor Robustness (Jackknife Anchor Removal)

**Date:** 2026-05-08
**Script:** `/tmp/annitia_runs/lb_jackknife.py` (imports `tests/experiments/annitia/lb_reverse_engineer.py` —
no re-implementation; same `fit_bayesian` / `predict_bayesian` codepath).
**Inputs:** 10 usable anchors (4 historical CSVs already lost: `results_r2`, `results_stack`,
`master_2way`, `annitia_submission`). 88 `sub_master_*` candidates.
**Procedure:** for each of the 10 anchors, refit the Bayesian predictor on the remaining 9, score
all 88 candidates, record top-15 ranking. Compare against the full-10 ranking via Kendall τ on the
union of the two top-15 lists, plus top-10 overlap.

Outputs:
- `tests/results/annitia/full/lb_predictor_jackknife.csv`
- `tests/results/annitia/full/lb_predictor_top10_consensus.csv`

---

## 1. Full-anchor top-15 (reference)

| Rank | Variant | pred |
|---|---|---|
| 1 | `sub_master_blend_top1` | 0.92796 |
| 2 | `sub_master_blend_top2` | 0.92793 |
| 3 | `sub_master_blend_top5` | 0.92792 |
| 4 | `sub_master_blend_top4` | 0.92787 |
| 5 | `sub_master_blend_top3` | 0.92781 |
| 6 | `sub_master_proven_meta_weighted` | 0.92765 |
| 7 | `sub_master_proven_meta` | 0.92765 |
| 8 | `sub_master_FINAL` | 0.92756 |
| 9 | `sub_master_proven_median` | 0.92731 |
| 10 | `sub_master_ultra_meta_weighted` | 0.92674 |
| 11 | `sub_master_ultra_meta` | 0.92666 |
| 12 | `sub_master_BEST_BET` | 0.92650 |
| 13 | `sub_master_meta_avg_grand_all` | 0.92626 |
| 14 | `sub_master_meta_avg_aware_weighted` | 0.92619 |
| 15 | `sub_master_meta_avg_all` | 0.92388 |

The top-9 are tightly packed within Δpred ≈ 0.00065, well inside the LOO MAE (0.015) of the
predictor — so any apparent ordering inside that band is below the predictor's own noise floor.

## 2. Jackknife stability

Sorted by τ ascending (most disruptive removal first):

| Anchor removed | Anchor LB | Kendall τ | Top-10 overlap |
|---|---:|---:|---:|
| `sub_master_best_combined` | 0.9123 | **0.676** | 9 |
| `sub_master_meta_avg` | 0.9269 | 0.829 | 10 |
| `submission` | 0.7181 | 0.833 | 10 |
| `sub_master_meta_avg_weighted` | 0.9285 | 0.900 | 10 |
| `sub_master_meta_avg_top5` | 0.9278 | 0.900 | 10 |
| `sub_master_meta_avg_aware_weighted` | 0.9302 | 0.924 | 10 |
| `sub_master_meta_avg_all` | 0.9294 | 0.950 | 10 |
| `sub_master_meta_avg_grand_all` | 0.9294 | 0.981 | 10 |
| `sub_master_FINAL` | 0.9268 | 1.000 | 10 |
| `sub_master_rank_median` | 0.9112 | 1.000 | 10 |

Mean τ = 0.899, median τ = 0.912. Removing any single anchor leaves the top-10 set unchanged in
9/10 runs. The lone exception drops one variant from the top-10 — see below.

## 3. Most-influential anchor

**`sub_master_best_combined` (LB 0.91225)** is the most-influential anchor. Removing it:
- Drops τ to **0.676** (the only run below 0.83).
- Causes 1 swap in/out of the top-10 (overlap = 9, not 10).
- Bumps `sub_master_ultra_meta` into the top-10 in place of one of the rank-9/10 picks (the
  consensus table shows `ultra_meta` appears in top-10 in only 1 of 10 jackknife runs and
  `ultra_meta_weighted` in 9 of 10 — both correspond to the `best_combined`-removed run).

Why? `best_combined` is a *low-end* anchor (LB 0.912, well below the leading cluster). It is the
predictor's main constraint pinning down what "below 0.92" looks like in the latent rank space.
Without it, the Bayesian posterior over `t` shifts, and the differential ordering inside the
narrow 0.926–0.928 band rearranges. This is consistent with the LOO MAE being driven by exactly
this anchor (cf. `AUDIT_REVERSE_ENGINEER.md`). The `submission` outlier (LB 0.7181) is *not*
the most influential — its removal only changes τ to 0.833. The Gaussian relaxation absorbs the
outlier into the prior tail, where it carries little leverage on the top-of-the-board ordering.

## 4. Robust top picks (consensus)

Variants appearing in the top-10 across all 10 jackknife runs:

| Variant | n_appearances |
|---|---:|
| `sub_master_blend_top1` | 10 |
| `sub_master_blend_top2` | 10 |
| `sub_master_blend_top3` | 10 |
| `sub_master_blend_top4` | 10 |
| `sub_master_blend_top5` | 10 |
| `sub_master_proven_meta_weighted` | 10 |
| `sub_master_proven_meta` | 10 |
| `sub_master_FINAL` | 10 |
| `sub_master_proven_median` | 10 |
| `sub_master_ultra_meta_weighted` | 9 |
| `sub_master_ultra_meta` | 1 |

Nine variants are unanimous top-10 picks. `ultra_meta_weighted` is robust at 9/10. `ultra_meta`
appears only in the `best_combined`-removed run.

## 5. Honest verdict

**The predictor's top-K *set* is robust. The within-set *ordering* is not.**

- Top-10 set: 9 of 10 candidates appear in 100% of leave-one-out runs. The 10th slot oscillates
  between two near-twin variants (`ultra_meta` vs `ultra_meta_weighted`) that are themselves
  almost identical predictions. As a "which submissions are worth burning a query on" filter,
  the predictor is reliable.
- Top-K ordering: the leading 9 candidates are bunched within Δpred ≈ 0.00065, an order of
  magnitude **below** the predictor's own LOO MAE (0.015). Calling the rank-1 candidate "the
  best" is not defensible from this analysis — it is one of nine effectively tied picks. Mean
  Kendall τ of 0.90 across runs reflects real reordering inside that band when anchors are
  jittered.
- One anchor (`sub_master_best_combined`) carries materially more weight than the other nine.
  If it turns out to be a noisy LB measurement, the entire top-K is biased in a correlated way
  that this jackknife cannot detect (we only test removal, not perturbation).
- The `submission` LB-0.7181 outlier has surprisingly little leverage (τ = 0.833 on removal).
  The Gaussian-relaxation prior is doing its job of damping out-of-cluster anchors.

**Practical conclusion:**
- TRUST the top-10 set as a candidate pool for further evaluation (e.g. burn LB queries on these,
  not on candidates outside the top-15).
- DO NOT trust the rank-1 pick as uniquely best. Treat the 9-variant unanimous-top-10 as a tied
  shortlist; use an orthogonal signal (OOF, cross-reference, calibration) to break ties — the
  predictor cannot.
- If forced to pick one submission to burn the deadline query on, `sub_master_blend_top1` is the
  rank-1 in the full fit AND in 9/10 jackknife runs, but the gain over `blend_top2..5`,
  `proven_meta`, `proven_meta_weighted`, `FINAL`, `proven_median` is below the noise floor.
