# AUDIT — Leaderboard Reverse Engineering

Anchors used: **10 of 14** (4 older CSVs lost: annitia_submission, master_2way, results_r2, results_stack — only `submission.csv`, `sub_master_*` available).

## Method recap

- **simple** — Nadaraya-Watson kernel regression over anchors using ρ_hep and ρ_death (Spearman) as similarity kernels (raised to the 4th power); a small linear adjustment on top fitted by leave-one-anchor-out.
- **bayesian** — Gaussian relaxation. Latent ranks t_hep, t_death (each N=423) with iso-Gaussian prior; LB observation is a noisy linear equation in t. Closed-form posterior via dual representation.
- **simplification used:** C-index ≈ (Spearman+1)/2. This ignores censoring structure in val_test.

## LOO cross-validation

Predicting each anchor's LB from the other anchors:

| Method | MAE | RMSE | Pearson r | Spearman ρ |
|---|---|---|---|---|
| Approach 1 (pairwise) | 0.0460 | 0.0720 | -0.584 | -0.745 |
| Approach 2 (Bayesian) | 0.0147 | 0.0257 | 0.978 | 0.709 |

LOO **excluding** the broken `submission` (which is a one-off outlier;
real candidates live in the LB ∈ [0.89, 0.93] cluster):

| Method | MAE | RMSE | Pearson r | Spearman ρ |
|---|---|---|---|---|
| Approach 1 (pairwise) | 0.0063 | 0.0077 | -0.908 | -0.850 |
| Approach 2 (Bayesian) | 0.0066 | 0.0081 | 0.961 | 0.517 |

Per-anchor LOO predictions:

| Stem | LB_true | pred_simple | pred_bayes | err_simple | err_bayes |
|---|---|---|---|---|---|
| `sub_master_meta_avg_aware_weighted` | 0.9302 | 0.8864 | 0.9252 | -0.0438 | -0.0050 |
| `sub_master_meta_avg_all` | 0.9294 | 0.8901 | 0.9219 | -0.0393 | -0.0075 |
| `sub_master_meta_avg_grand_all` | 0.9294 | 0.8878 | 0.9255 | -0.0416 | -0.0038 |
| `sub_master_meta_avg_weighted` | 0.9284 | 0.9061 | 0.9214 | -0.0224 | -0.0070 |
| `sub_master_meta_avg_top5` | 0.9278 | 0.9061 | 0.9213 | -0.0217 | -0.0065 |
| `sub_master_meta_avg` | 0.9269 | 0.9141 | 0.9199 | -0.0128 | -0.0070 |
| `sub_master_FINAL` | 0.9268 | 0.8897 | 0.9277 | -0.0371 | +0.0009 |
| `sub_master_best_combined` | 0.9123 | 0.8950 | 0.9025 | -0.0173 | -0.0097 |
| `sub_master_rank_median` | 0.9112 | 0.9266 | 0.8877 | +0.0154 | -0.0235 |
| `submission` | 0.7181 | 0.9269 | 0.7937 | +0.2088 | +0.0756 |

## Top-10 candidates by predicted LB (Bayesian)

| Variant | pred_simple | pred_bayes | std | LB_actual (if anchor) |
|---|---|---|---|---|
| `sub_master_blend_top1` | 0.8975 | 0.9280 | 0.0378 |  |
| `sub_master_blend_top2` | 0.9002 | 0.9279 | 0.0381 |  |
| `sub_master_blend_top5` | 0.8964 | 0.9279 | 0.0378 |  |
| `sub_master_blend_top4` | 0.8975 | 0.9279 | 0.0380 |  |
| `sub_master_blend_top3` | 0.8955 | 0.9278 | 0.0377 |  |
| `sub_master_proven_meta_weighted` | 0.8991 | 0.9277 | 0.0382 |  |
| `sub_master_proven_meta` | 0.8991 | 0.9276 | 0.0382 |  |
| `sub_master_FINAL` | 0.8932 | 0.9276 | 0.0375 | 0.9268 |
| `sub_master_proven_median` | 0.8942 | 0.9273 | 0.0376 |  |
| `sub_master_ultra_meta_weighted` | 0.8888 | 0.9267 | 0.0370 |  |

## Confidence / uncertainty

- Non-anchor candidates: 79
- Median posterior std (bayes): 0.0061
- Median anchor-disagreement std (simple): 0.0404
- Predicted LB range across non-anchor candidates: [0.6996, 0.9280]

## Honest verdict

- Existing OOF→LB has |Spearman ρ| = 0.309 (and the sign is wrong: OOF *anti-correlates* with LB).
- Reverse-engineer **simple** LOO Spearman = -0.745.
- Reverse-engineer **bayes** LOO Spearman = 0.709.

**Verdict:** the **Bayesian** predictor beats the OOF correlate (and points the right way).
The **simple** predictor is *worse* than OOF — its Nadaraya-Watson form collapses to a
constant when all candidates are nearly equidistant from all anchors, which is exactly
the regime here (most sub_master_* CSVs are >0.97 Spearman with each anchor).

### Caveats (read these before trusting the rankings)
- Only 10 of 14 anchors usable (4 lost). The broken `submission` (LB 0.7181) is the only outlier — without it the LB range is 0.89–0.93, very narrow, so LOO is dominated by within-cluster noise.
- C-index ≠ (Spearman+1)/2. Censoring on val_test will distort the linearisation; bias can be O(0.01) in LB units.
- All sub_master_* candidates likely share OOF inputs (rank-correlation ~1 across them on training). The predictor leans heavily on **how anchors differ**, which is also a low-rank signal.
- Posterior std underestimates true uncertainty — the Gaussian relaxation does not enforce that t is a permutation.
- LOO with n=10 anchors is high-variance. Treat MAE numbers ±50%.
