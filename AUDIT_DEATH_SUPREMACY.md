# AUDIT: Death-supremacy submission variants (2026-05-08)

## Premise

Top LB is 0.979. With death events 76 / 966 (vs hep 47/1106), death is statistically
easier; current best ensemble OOF C_death ~0.974 per existing variants. Hep is the
hard ceiling. If we hold hep at the anchor and squeeze death from ~0.974 to ~0.99, the
weighted LB lift is +0.005. Worth a calibrated swing.

All three variants reuse the anchor (`sub_master_meta_avg_aware_weighted.csv`) for the
hep column and replace the death column. Anchor hep ~ 0.95 LB.

## Inputs

- OOF predictions: `tests/results/annitia/branches/oof_death_<model>.npy` (n=1253; valid 966; events 76)
- Val (LB) predictions: `tests/results/annitia/branches/val_death_<model>.npy` (n=423)
- Trust labels from `oof_diagnostics.md`: HIGH=2, MEDIUM=7, LOW=5, SUSPECT=0
- Per-horizon C from `per_horizon_C.csv`

## Top-5 high-trust death pool

| # | Model | OOF C | 95% CI | flag |
|---|---|---|---|---|
| 1 | tabpfn-landmark__snapshot+clinical_pro+fpca+gp_smooth | 0.933 | [0.906, 0.956] | HIGH |
| 2 | catboost-H2D__snapshot+clinical_pro+fpca+gp_smooth | 0.901 | [0.864, 0.933] | MEDIUM |
| 3 | dsm-K4__snapshot+clinical_pro+fpca+visit_density+clinical_slope | 0.890 | [0.846, 0.928] | MEDIUM |
| 4 | tabpfn25-disc-K20__snapshot+clinical_pro+fpca+gp_smooth | 0.888 | [0.857, 0.923] | HIGH |
| 5 | dsm-K4__snapshot+clinical_pro+meld_na+visit_density | 0.877 | [0.832, 0.913] | MEDIUM |

## Variant 1 — `sub_master_death_supremacy`

Strategy: nan-mean of per-row [0,1] ranks across the 5 high-trust models.

| Metric | Value |
|---|---|
| OOF C_death (ensemble) | **0.9398** |
| OOF C_hep (reference: synaug-none-rsf-500) | 0.8681 |
| Naive weighted (0.7 * 0.868 + 0.3 * 0.940) | 0.890 |
| ρ vs anchor death | 0.811 |
| ρ vs FINAL death | 0.815 |
| Range | [0.028, 0.993] |
| Unique values | 384 |
| pre_submit_check | PASS (only hep=anchor WARN) |

**Verdict:** Best ensemble C of the three variants. Beats the single best (tabpfn-landmark
@ 0.933) by +0.006. Modest, within bootstrap noise (CI half-width ≈ 0.04 at 76 events),
but the rank-mean is more robust to single-fold luck. Worth submitting.

## Variant 2 — `sub_master_death_conformal`

Strategy: each model's val score is mapped to its empirical OOF CDF position, then averaged.

| Metric | Value |
|---|---|
| OOF C_death (self-conformal = OOF rank avg) | **0.9398** (tied with V1) |
| ρ vs anchor death | 0.810 |
| ρ vs FINAL death | 0.814 |
| Range | [0.107, 0.983] |
| Unique values | 397 |
| pre_submit_check | PASS |

**Verdict:** OOF-on-OOF is rank-identical to V1, so OOF C matches. The val-side
calibration differs (val rho 0.810 vs 0.811) because the empirical CDF stretches
extreme test scores differently from a pure rank transform. Diversity vs V1 is small
(both share the same 5 base models and rank backbone). **Skip if budget is tight** —
file the difference under "submission variant".

## Variant 3 — `sub_master_death_per_horizon`

Strategy: pick the best death model for each horizon (3y, 5y, 10y) per
`per_horizon_C.csv`, then rank-mean. Used: tsfm-gbsa (3y, C@3y=0.970), tabpfn-landmark
(5y/10y, C@5y=0.953, C@10y=0.955), catboost-H2D (mid as diversity filler).

| Metric | Value |
|---|---|
| OOF C_death (ensemble) | **0.9372** |
| ρ vs anchor death | 0.759 |
| ρ vs FINAL death | 0.763 |
| Range | [0.020, 0.976] |
| Unique values | 367 |
| pre_submit_check | PASS |

**Verdict:** Slightly worse OOF C than V1/V2 (-0.003) but the most diverse vs anchor
(rho 0.76 vs 0.81). The horizon-best signal mostly collapses back to tabpfn-landmark
(it wins 5y AND 10y), so the only true horizon-specialist is tsfm-gbsa@3y, whose
overall pooled C is 0.840 — it drags the ensemble. **Submit only if anchor is being
beaten by a model with very different per-horizon profile**; otherwise V1 is better.

## Net assessment

The top-5 trust pool ensemble (V1) lifts overall OOF C_death from 0.933 (best single)
to 0.940 — a +0.007 bump. Multiplied by the 0.3 LB weight, that's +0.002 LB if it
generalises. Within the 76-event noise floor (CI ±0.04), V1 vs V2 is a coin flip. V3
is statistically worse and only useful for diversity routing.

**Recommended submission order:**

1. `sub_master_death_supremacy` (highest OOF C, smallest deviation from anchor on the
    rank metric — safest swing).
2. `sub_master_death_per_horizon` (max diversity vs anchor — useful blend ingredient
    even if it doesn't win on its own).
3. `sub_master_death_conformal` (only if remaining budget allows; rank-equivalent to V1
    by construction so unlikely to materially differ on LB).

## Caveats

- Hep OOF C reference (0.868) is the single best HEP base model. The anchor hep column
  is a composite ensemble whose true OOF C is **higher** — actual LB hep contribution
  is closer to 0.95 (per memory note). Adjust the 0.7-weighted estimate upward
  accordingly: realistic LB ≈ 0.7 × 0.95 + 0.3 × 0.940 = 0.947.
- All three variants share the anchor hep column → they are differentiated only by the
  death column. If the anchor hep column happens to underperform on the LB private
  split, all three will track the anchor's regression.
- Bootstrap CI on death C with 76 events ≈ ±0.04. The +0.007 bump from V1 is within
  noise; treat the LB lift as expected-value rather than guaranteed.

## Files written

- `tests/results/annitia/full/sub_master_death_supremacy.csv`
- `tests/results/annitia/full/sub_master_death_conformal.csv`
- `tests/results/annitia/full/sub_master_death_per_horizon.csv`
- `/tmp/annitia_runs/death_supremacy_summary.json` (numeric summary)
- `/tmp/annitia_runs/death_supremacy.py` (build script, NOT under `tests/experiments/annitia/` since this is a one-off ad-hoc experiment)
