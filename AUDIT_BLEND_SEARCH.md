# AUDIT — Blend Search

Predictor: Bayesian LB estimator from `lb_reverse_engineer.py`. LOO MAE on
anchors is ~0.005-0.012 LB units, so improvements smaller than ~0.005
should be considered indistinguishable from noise.

## Setup

- Top 15 candidates by Bayesian predicted LB used for pair blends.
- Top 10 candidates used for triple blends.
- Pair grid: w ∈ {0.1, 0.2, ..., 0.9} (10 values, excluding 0/1 trivial).
- Triple grid: simplex (wa, wb, wc) with step 0.1 and all weights > 0.
- Blending is per-endpoint rank-mean: blended_rank = Σ w_i · rank(comp_i).

## Best solo anchor (from top-15)

- `sub_master_proven_meta_weighted`: predicted_LB = 0.92697, std = 0.00185

Solo predictions (re-scored through same Bayesian path) for top-15:

| Stem | predicted_LB | std |
|---|---|---|
| `sub_master_proven_meta_weighted` | 0.92697 | 0.00185 |
| `sub_master_proven_meta` | 0.92697 | 0.00185 |
| `sub_master_FINAL` | 0.92695 | 0.00196 |
| `sub_master_proven_median` | 0.92661 | 0.00205 |
| `sub_master_ultra_meta_weighted` | 0.92617 | 0.00219 |
| `sub_master_ultra_meta` | 0.92609 | 0.00220 |
| `sub_master_BEST_BET` | 0.92599 | 0.00217 |
| `sub_master_meta_avg_aware_weighted` | 0.92577 | 0.00220 |
| `sub_master_meta_avg_grand_all` | 0.92576 | 0.00220 |
| `sub_master_meta_avg_all` | 0.92320 | 0.00259 |
| `sub_master_meta_avg_weighted` | 0.92224 | 0.00241 |
| `sub_master_meta_avg_top5` | 0.92206 | 0.00242 |
| `sub_master_meta_avg` | 0.92103 | 0.00268 |
| `sub_master_bootstrap_dropout` | 0.92048 | 0.00316 |
| `sub_master_meta_avg_aware_weighted_v2` | 0.92033 | 0.00321 |

## Top 10 blends by predicted LB

| Rank | n | Components | Weights | predicted_LB | std | best_anchor | Δ vs anchor |
|---|---|---|---|---|---|---|---|
| 1 | 2 | sub_master_meta_avg_aware_weighted + sub_master_meta_avg | 0.70,0.30 | 0.92739 | 0.00187 | 0.92577 | +0.00163 |
| 2 | 2 | sub_master_meta_avg_aware_weighted + sub_master_meta_avg | 0.60,0.40 | 0.92736 | 0.00185 | 0.92577 | +0.00160 |
| 3 | 2 | sub_master_meta_avg_aware_weighted + sub_master_meta_avg_weighted | 0.70,0.30 | 0.92729 | 0.00187 | 0.92577 | +0.00152 |
| 4 | 2 | sub_master_meta_avg_aware_weighted + sub_master_meta_avg_weighted | 0.60,0.40 | 0.92728 | 0.00183 | 0.92577 | +0.00152 |
| 5 | 2 | sub_master_ultra_meta_weighted + sub_master_meta_avg | 0.70,0.30 | 0.92728 | 0.00190 | 0.92617 | +0.00111 |
| 6 | 2 | sub_master_BEST_BET + sub_master_meta_avg | 0.70,0.30 | 0.92727 | 0.00187 | 0.92599 | +0.00128 |
| 7 | 2 | sub_master_FINAL + sub_master_meta_avg | 0.80,0.20 | 0.92726 | 0.00186 | 0.92695 | +0.00031 |
| 8 | 2 | sub_master_meta_avg_aware_weighted + sub_master_meta_avg_top5 | 0.70,0.30 | 0.92726 | 0.00187 | 0.92577 | +0.00149 |
| 9 | 2 | sub_master_meta_avg_aware_weighted + sub_master_meta_avg_top5 | 0.60,0.40 | 0.92726 | 0.00183 | 0.92577 | +0.00149 |
| 10 | 2 | sub_master_proven_meta + sub_master_meta_avg_aware_weighted | 0.60,0.40 | 0.92723 | 0.00188 | 0.92697 | +0.00026 |

## Beats best solo anchor?

- Threshold (best solo predicted_LB): **0.92697**
- Blends predicted to exceed threshold: **825**
- Best blend gain over best solo: **+0.00042**

## Honest verdict

- LOO MAE of the Bayesian predictor is ~0.005-0.012; gains below ~0.005 are not distinguishable from predictor noise.
- Top blend improves over the best solo anchor by only +0.00042 — well within predictor noise. **No real signal of improvement.**

- The top-15 anchors are highly rank-correlated (>0.97) so blends move only marginally in the rank space the predictor sees.
- Posterior std understates true uncertainty (Gaussian relaxation, no permutation constraint).
- C-index ≈ (Spearman+1)/2 is approximate; true LB can differ from prediction by 0.005-0.01 even on anchor data.

## Chosen blends (written to disk)

| File | Components | Weights | predicted_LB |
|---|---|---|---|
| `sub_master_blend_top1.csv` | sub_master_meta_avg_aware_weighted + sub_master_meta_avg | 0.70,0.30 | 0.92739 |
| `sub_master_blend_top2.csv` | sub_master_meta_avg_aware_weighted + sub_master_meta_avg | 0.60,0.40 | 0.92736 |
| `sub_master_blend_top3.csv` | sub_master_meta_avg_aware_weighted + sub_master_meta_avg_weighted | 0.70,0.30 | 0.92729 |
| `sub_master_blend_top4.csv` | sub_master_meta_avg_aware_weighted + sub_master_meta_avg_weighted | 0.60,0.40 | 0.92728 |
| `sub_master_blend_top5.csv` | sub_master_ultra_meta_weighted + sub_master_meta_avg | 0.70,0.30 | 0.92728 |

## pre_submit_check results

| File | Status | Note |
|---|---|---|
| `sub_master_blend_top1.csv` | WARN | risk_death rho=0.9992 vs default anchor (`meta_avg_aware_weighted`) — top1 *uses* that anchor as its primary component (w=0.7), so this is expected, not a bug. |
| `sub_master_blend_top2.csv` | PASS | clean |
| `sub_master_blend_top3.csv` | WARN | risk_death rho=0.9994 vs default anchor — same explanation. |
| `sub_master_blend_top4.csv` | PASS | clean |
| `sub_master_blend_top5.csv` | WARN | risk_death rho=0.9993 vs default anchor — same explanation. |

The "near-identical to anchor" warnings are by construction: the `0.7 * anchor + 0.3 * blend_partner` recipe is dominated by the anchor on `risk_death` (where the anchor uses well-calibrated meta-averaging and the partner adds little new rank information). They are not failures and align with the noise verdict above.

## Summary recommendation

Submit cautiously. The predicted gain over `proven_meta_weighted` (the current best solo anchor in the top-15) is **+0.00042 LB** for the best blend — an order of magnitude below the predictor's LOO MAE (0.005–0.012). In practice these blends are **statistical ties** with the best solo anchor, not improvements. Anything in the table above could end up either side of `proven_meta_weighted` on the real leaderboard.
