# AUDIT — Aggressive mechanism-divergent candidates

Date: 2026-05-08

## Why this exists

The Bayesian leaderboard predictor (`tests/results/annitia/full/lb_predictions.csv`)
was fit on 10 anchor submissions all scoring between 0.89 and 0.93. Its prior
mean dominates the posterior for every variant outside the anchor cluster, so
it predicts ~0.927 ± 0.002 for **all** new candidates. That's a structural
limitation: it cannot tell us whether a mechanism-different candidate will
score 0.95 or 0.85 — it just regresses to the anchor mode.

The public LB top is **0.979**. So mechanism-different candidates plausibly
score 0.95+. We have been picking by `predicted_LB_bayesian`, which biases
toward anchor-imitators. This audit picks by **mechanism divergence** instead,
using `tests/results/annitia/full/variant_clusters.csv` as the diversity map.

## High-uniqueness building blocks (from variant_clusters.csv)

| Variant | uniqueness | Notes |
|---|---|---|
| `coxnet_stack_with_uncertainty` | 0.711 | Far outlier — most-different fusion |
| `LEARNED_STACK` | 0.231 | hep cluster 1 |
| `round3` | 0.189 | Scored 0.7181 (broken) — excluded |
| `subgroup_routed` | 0.148 | death cluster 13 |
| `disagreement_route` | 0.124 | hep cluster 9 |
| `nnls_stack` | 0.124 | death cluster 7 |
| `uncertainty_route` | 0.116 | hep cluster 37 |
| `new_cohort_split` | 0.086 | hep cluster 18 |
| `subgroup_blend` | 0.076 | hep cluster 30 |

## Trust map (from `oof_diagnostics.md`)

**Hepatic** (47 events, all base models flag LOW because lower-CI floor < 0.85):
top-5 by pooled C — synaug-none-rsf (0.868), tsfm-rsf (0.842), tsfm-gbsa
(0.842), catboost-D2H (0.838), tabpfn25-disc-K10 (0.838).

**Death** (76 events): tabpfn-landmark (0.933 HIGH), catboost-H2D (0.901),
dsm-K4 visit_density (0.890), tabpfn25-disc-K20 (0.888 HIGH), dsm-K4 meld_na
(0.877).

## The 5 candidates

| File | Mechanism | OOF hep | OOF death | OOF weighted |
|---|---|---|---|---|
| `sub_master_aggressive_xstack.csv` | 50/50 rank-blend anchor + coxnet_stack_with_uncertainty (uniqueness 0.71) | 0.854 | 0.950 | **0.883** |
| `sub_master_aggressive_routed_blend.csv` | Anchor for trust 90%; for top-10% disagreement (std across anchor/router/disagreement_route) swap in `cluster_router_meta` ⊕ `disagreement_route` average | 0.868 | 0.938 | **0.889** |
| `sub_master_aggressive_extreme.csv` | Tied-band reshape of anchor: top 5% → 0.99, bottom 50% → 0.25, middle 45% kept | 0.854 | 0.911 | **0.871** |
| `sub_master_aggressive_specialist_only.csv` | Drop low-trust models entirely. Rank-mean of 5 trusted hep + 5 trusted death | 0.869 | 0.941 | **0.890** |
| `sub_master_aggressive_topk_only.csv` | K=3 each endpoint, rank-mean — most aggressive cut | 0.867 | 0.938 | **0.888** |

All pass `pre_submit_check.py` (cols, NaN, row count, monotone agreement with
anchor between 0.4 and 0.999, no constants, no inf).

## What each candidate tests

**xstack** — Does the most-unique variant
(`coxnet_stack_with_uncertainty`, uniqueness 0.711) carry information that the
anchor smooths away? Its raw OOF is unclear (no train-side OOF saved for
this stack), but on val it sits at the structural extreme. If it adds signal,
50/50 rank-blend should pull LB above 0.93. If it's noise, LB drops noticeably.

**routed_blend** — Bigger swap than `sub_master_anomaly_blend_meta` (which
only swaps high-anomaly patients with a small weight). Here we replace the
anchor entirely on the top-10% disagreement subset using two of the most
mechanism-different variants we have. If patient routing helps at all, this is
the cleanest test.

**extreme** — Trustii's per-patient C-index reduces to **rank** comparisons
within event-pairs, so a strict monotone reshape would be a no-op. We instead
introduce ties: top 5% all collapse to 0.99, bottom 50% all collapse to 0.25,
middle keeps its rank. The ties remove discriminative information **inside**
those bands, which is a real test of whether the model's mid-rank ordering
matters or only the overall risk classification matters. OOF drops modestly
(0.871 vs anchor's ≈0.890), confirming the test is non-trivial.

**specialist_only** — Drops every "LOW" / unreliable model from the OOF
diagnostics; uses **only** the top-5 most trusted per endpoint. If the anchor
is being dragged down by noise from low-trust ensembles, this should beat it.
Its OOF (0.890) is on par with the anchor proxy, suggesting trust filtering is
neutral on training but may help on test where low-trust models could overfit.

**topk_only** — K=3 most aggressive. If the LB top (0.979) is being achieved
by very small ensembles of very strong specialists, this is the closest we get
without retraining. Slightly under specialist_only on OOF (0.888) — risk of
under-diversification.

## OOF caveat

The OOF "C" is computed against training labels using the 5 trusted models'
ranks as a stand-in for each variant's training-side fusion logic. For
**specialist_only** and **topk_only** this is exact (the val and OOF use the
same set of base models). For **xstack** and **routed_blend** the OOF is a
proxy: we approximate the anchor's training-side fusion as the trusted
rank-mean, and the coxnet/router contributions via the consensus / median of
trusted models. The OOF gap (`oof_lb - lb`) is consistently ~0 to -0.04 for
the proven recipes (`oof_to_leaderboard.md`), so an OOF of 0.88 corresponds
roughly to LB ≈ 0.92-0.93. **None of the OOFs above 0.89 should be read as
"projected LB"** — they're just sanity checks that the candidate isn't
catastrophically broken.

## Selection guidance

Submit in this order if you have 5 attempts left:

1. **`sub_master_aggressive_specialist_only.csv`** — highest OOF, lowest
   risk, mechanism-clean (drop low-trust models entirely, cleanest hypothesis).
2. **`sub_master_aggressive_routed_blend.csv`** — most aggressive routing
   yet attempted, OOF only 0.001 below specialist_only.
3. **`sub_master_aggressive_xstack.csv`** — wildcard. If coxnet adds signal,
   biggest upside. Lower OOF (0.883) means highest variance.
4. **`sub_master_aggressive_topk_only.csv`** — sanity check on whether
   minimal ensembles beat broader ones.
5. **`sub_master_aggressive_extreme.csv`** — only submit if you have a free
   slot. Tests a structural hypothesis (do mid-rank ties hurt LB?) but OOF is
   visibly worse, so it's the most likely to lose.

## What this audit does NOT do

- Does not run a leave-one-anchor-out re-fit of the LB predictor — the prior
  is too strong, the answer would be the same.
- Does not retrain any base models. We're recombining existing OOFs/val preds.
- Does not resolve the structural ambiguity that the LB top (0.979) might be
  achieved by features we lack entirely (e.g., visit-time features we
  excluded). If that's the case, none of these candidates will reach 0.95.

## Files

- Builder: `tests/experiments/annitia/build_aggressive_candidates.py`
- Submission CSVs: `tests/results/annitia/full/sub_master_aggressive_*.csv`
- Summary: `tests/results/annitia/full/aggressive_candidates_summary.csv`
- Trust map: `tests/results/annitia/full/oof_diagnostics.md`
- Diversity map: `tests/results/annitia/full/variant_clusters.csv`
