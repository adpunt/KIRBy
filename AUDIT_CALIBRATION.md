# Calibration metrics (Brier / IBS / slope) — ANNITIA OOF

_Generated 2026-05-08. 29 variants (14 historic + 15 new candidates)._

## Method

OOFs are RISK SCORES, not survival probabilities — Brier needs S(t).
For each variant we:

1. Construct a training-side proxy OOF via `variant_proxy_oof` (rank-mean,
   rank-median, or trimmed-mean of base OOFs depending on the variant stem).
2. Fit `sksurv.linear_model.CoxPHSurvivalAnalysis(alpha=1e-3)` with the
   standardized risk as the single feature on the full training set.
3. Compute baseline cumulative hazard via Breslow (vectorized — sksurv's
   `predict_survival_function` was too slow to call per-grid-point).
4. Brier at 1, 3, 5, 10 years (only horizons inside [t_min+eps, t_max-eps]).
5. IBS via `sksurv.metrics.integrated_brier_score` on a 50-pt grid.

_Note:_ `np.trapezoid` was missing under numpy 1.26 — backfilled to `np.trapz`
so sksurv's IBS works.

## Interpretation

- **Lower Brier / IBS = better calibrated**. Brier in [0, 0.25].
- Slope here is the Cox coefficient on standardized risk (in-sample).
  Higher slope ⇒ stronger signal in the score, NOT directly the textbook
  external calibration slope. For external calibration slope you would need
  to refit Cox on a held-out set, which we do not have for the test split.
- **Concordance is rank-based — calibration does NOT affect it.**
  These metrics are INFORMATIONAL for the qualitative 30% of the Trustii
  score (and to flag synthetic-aug calibration drift), not for picking
  candidates by predicted LB.

## Headline numbers (3 distinct calibration profiles)

The proxy collapses most variants into one of three aggregator classes:

| Aggregator | IBS hep | IBS death | IBS weighted | Slope hep | Slope death |
|---|---|---|---|---|---|
| rank-mean (default for `meta_avg`, `failure_*`, `noise_*`, `best_combined`, etc.) | 0.0626 | 0.0752 | 0.0664 | 1.378 | 1.751 |
| trimmed-mean (`rank_trimmed`) | 0.0633 | 0.0752 | 0.0671 | 1.361 | 1.736 |
| rank-median (`rank_median`) | 0.0673 | 0.0771 | 0.0702 | 1.346 | 1.558 |

**Rank-mean wins on calibration too** — it has both the lowest IBS and the
highest slopes (strongest in-sample fit). This matches the LB ranking
where `rank_median` (LB 0.9112) trailed every meta_avg / FINAL variant
(LB 0.9268-0.9302).

Death-endpoint slope (1.55-1.75) is markedly higher than hepatic (1.34-1.38),
indicating the consensus signal is sharper for death prediction — consistent
with the OOF concordances (death OOF C ≈ 0.94 vs hepatic ≈ 0.82).

## Brier scores at fixed horizons (rank-mean proxy)

| Horizon | Brier hep | Brier death |
|---|---|---|
| 3y | 0.0144 | 0.0054 |
| 5y | 0.0200 | 0.0220 |
| 10y | 0.0394 | 0.0503 |

_Brier at 1y is NaN (1y < t_min for both endpoints — no events that early)._

These are very low Brier values — partly because event rates are low at
early horizons (most patients survive), and partly because the calibration
is in-sample. **Treat as floor estimates.**

## Full per-variant table (lowest weighted IBS first)

| Variant | LB | IBS hep | IBS death | IBS weighted | Slope hep | Slope death |
|---|---|---|---|---|---|---|
| `sub_master_FINAL` | 0.9268 | 0.0626 | 0.0752 | 0.0664 | 1.378 | 1.751 |
| `sub_master_failure_xref_aware` | — | 0.0626 | 0.0752 | 0.0664 | 1.378 | 1.751 |
| `sub_master_failure_robust` | — | 0.0626 | 0.0752 | 0.0664 | 1.378 | 1.751 |
| `sub_master_failure_hard_specialist` | — | 0.0626 | 0.0752 | 0.0664 | 1.378 | 1.751 |
| `sub_master_failure_fp_suppress` | — | 0.0626 | 0.0752 | 0.0664 | 1.378 | 1.751 |
| `sub_master_failure_fn_recovery` | — | 0.0626 | 0.0752 | 0.0664 | 1.378 | 1.751 |
| `sub_master_best_combined_v2` | — | 0.0626 | 0.0752 | 0.0664 | 1.378 | 1.751 |
| `sub_master_rank_mean` | — | 0.0626 | 0.0752 | 0.0664 | 1.378 | 1.751 |
| `sub_master_meta_avg_new_variants_only` | — | 0.0626 | 0.0752 | 0.0664 | 1.378 | 1.751 |
| `sub_master_meta_avg_grand_all_v2` | — | 0.0626 | 0.0752 | 0.0664 | 1.378 | 1.751 |
| `sub_master_meta_avg_aware_weighted_v2` | — | 0.0626 | 0.0752 | 0.0664 | 1.378 | 1.751 |
| `sub_master_meta_avg_aware_weighted_no_nnls` | — | 0.0626 | 0.0752 | 0.0664 | 1.378 | 1.751 |
| `sub_master_noise_censoraware` | — | 0.0626 | 0.0752 | 0.0664 | 1.378 | 1.751 |
| `sub_master_meta_avg_aware_top5` | — | 0.0626 | 0.0752 | 0.0664 | 1.378 | 1.751 |
| `master_2way` | 0.9119 | 0.0626 | 0.0752 | 0.0664 | 1.378 | 1.751 |
| `results_stack` | 0.9120 | 0.0626 | 0.0752 | 0.0664 | 1.378 | 1.751 |
| `results_r2` | 0.9099 | 0.0626 | 0.0752 | 0.0664 | 1.378 | 1.751 |
| `submission` | 0.7181 | 0.0626 | 0.0752 | 0.0664 | 1.378 | 1.751 |
| `sub_master_best_combined` | 0.9123 | 0.0626 | 0.0752 | 0.0664 | 1.378 | 1.751 |
| `sub_master_meta_avg` | 0.9269 | 0.0626 | 0.0752 | 0.0664 | 1.378 | 1.751 |

## Calibration verdict (per variant class)

### GOOD calibration (rank-mean class, lowest IBS, highest slopes)

All of these submissions share the same training-side proxy and therefore
identical Brier/IBS under this audit. They are tied for first on calibration:

- `sub_master_FINAL` (LB 0.9268)
- `sub_master_meta_avg_aware_weighted` (LB 0.9302) — current best public LB
- `sub_master_meta_avg_grand_all` (LB 0.9294)
- `sub_master_meta_avg_all` (LB 0.9294)
- `sub_master_meta_avg_weighted` (LB 0.9285)
- `sub_master_meta_avg_top5` (LB 0.9278)
- `sub_master_meta_avg` (LB 0.9269)
- `sub_master_best_combined` (LB 0.9123)
- 12 newer rank-mean candidates not yet uploaded (see CSV)

### MEDIUM calibration

- `sub_master_rank_trimmed` (IBS 0.0671) — trimmed-mean discards 20% of
  the per-patient model spread; minor calibration cost.

### WORST calibration in this set

- `sub_master_rank_median` (LB 0.9112, IBS 0.0702) — median has the
  highest IBS and lowest slopes. Consistent with its lower LB score.

## Synthetic-aug calibration drift check

`sub_master_noise_*` variants share the rank-mean profile (IBS 0.0664,
slopes 1.38/1.75) — **no detectable calibration drift** from the noise-aware
aggregators relative to the vanilla `meta_avg`. This is informational only
because the proxy approximates these variants by the same rank-mean.

## Caveats / what this analysis CANNOT tell you

- **In-sample Cox fit** ⇒ absolute IBS is optimistic. Treat as a lower
  bound on test-time Brier. A hold-out re-fit would be more honest, but
  we do not have OOF predictions for the val/test split.
- **Proxy collapses sub-variants** of the same aggregator class. To
  differentiate `meta_avg_top5` from `meta_avg_all` we would need each
  variant to save its own training-side OOF, not just the val/test
  prediction CSV. The proxy is faithful to `rank_median`, `rank_trimmed`,
  and rank-mean families but cannot resolve weighting nuances.
- **Slope is in-sample Cox coef on standardized risk**, not the external
  calibration slope from a held-out re-fit. Use as a relative diagnostic.
- **Brier @ 1y is NaN** because no events fall before 1y in either
  endpoint (t_min ≈ 1.0 yr post-baseline). This is dataset-driven, not
  a bug.

## Bottom line

The current best LB submission (`sub_master_meta_avg_aware_weighted`,
LB 0.9302) is in the **best-calibrated** class under this audit. There is
no calibration-driven reason to switch the final submission. The 30%
qualitative score should benefit from the low IBS (0.066 weighted) and
the strong death-endpoint slope (1.75).