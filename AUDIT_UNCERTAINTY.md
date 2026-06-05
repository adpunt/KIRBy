# Uncertainty-aware ensemble candidates (sub_master_unc_*)

Five new candidates, each driven by a **distinct uncertainty signal**.
LB scoring: `0.7 * C_hep + 0.3 * C_death`. Anchor = `sub_master_meta_avg_aware_weighted` (LB 0.9302).

OOF concordances are computed on the full **training cohort** (n=1253) with
a per-variant aggregation that mirrors the val-time recipe; OOF C is a
stretch upper bound (variants can use the real anchor at val time, and OOF
substitutes rank_mean of the base pool — so OOF C systematically **understates**
the actual val-time fusion when the variant relies on the anchor).

Spearman ρ vs anchor measures how aggressively the variant moves rankings.

## Ranked by OOF weighted (0.7 hep + 0.3 death)

| Rank | Variant         | OOF C_hep | OOF C_death | OOF weighted | ρ_hep | ρ_death | pre_submit | File |
|------|-----------------|-----------|-------------|--------------|-------|---------|------------|------|
| 1    | disagree_soft   | 0.8602    | 0.9485      | **0.8867**   | 0.934 | 0.869   | PASS       | `sub_master_unc_disagree_soft.csv` |
| 2    | specialist      | 0.8470    | 0.9404      | 0.8750       | 0.956 | 0.858   | PASS       | `sub_master_unc_specialist.csv` |
| 3    | bootstrap_var   | 0.8343    | 0.9451      | 0.8675       | 0.850 | 0.848   | PASS       | `sub_master_unc_bootstrap_var.csv` |
| 4    | mahal_blend     | 0.8247    | 0.9363      | 0.8582       | 0.930 | 0.933   | PASS       | `sub_master_unc_mahal_blend.csv` |
| 5    | conformal       | 0.8209    | 0.9377      | 0.8559       | 0.820 | 0.846   | PASS       | `sub_master_unc_conformal.csv` |

All five output 423 rows × `[trustii_id, risk_hepatic_event, risk_death]`,
no NaN, no constants, no anti-correlation with anchor.

## Per-candidate detail

### 1. disagree_soft (LB-EV best of the new set)
- **Signal:** per-patient ensemble rank-std across the 22-hep / 16-death base pool.
- **Mechanism:** identifies the single base model with best C on the
  upper-quartile-disagreement training rows ("hard cases") and **continuously
  blends** that specialist with anchor via a sigmoid gate centered at the
  median disagreement.
- **Specialist picked:** synaug-none-rsf-500-n0 (hep), tabpfn-landmark (death).
- **Distinct from disagreement_router_meta:** no buckets — sigmoid gate;
  routes between anchor + ONE specialist (not 10 fixed variants).
- **Expected-LB rationale:** OOF weighted 0.8867 is the highest of the five.
  Hep ρ=0.93 vs anchor (preserves the best part of anchor on the 0.7-weighted
  endpoint) while death ρ=0.87 (genuine new info). Strong candidate for the
  0.7-weighted-hep regime.

### 2. specialist
- **Signal:** ensemble rank-std (top-20% threshold), hard switch.
- **Mechanism:** for the top 20% highest-uncertainty val patients (~85 of
  423), override anchor with the most-trusted single base model from
  `oof_diagnostics.md` (synaug-none-rsf for hep, tabpfn-landmark for death).
  All other val patients keep anchor unchanged.
- **Distinct from anomaly_router_meta:** uses uncertainty (not anomaly), and
  uses an externally-validated trust flag (HIGH/MEDIUM in oof_diagnostics)
  rather than per-bucket discordance; only ~20% of patients are touched.
- **Expected-LB rationale:** ρ_hep=0.96 is the highest preservation of the
  anchor; expected LB ≥ anchor (0.9302) because the override only fires on
  uncertain patients where anchor was probably wrong. OOF understated
  because OOF baseline is rank_mean (not real anchor); real LB likely higher
  than 0.875.

### 3. bootstrap_var
- **Signal:** per-base-model OOF concordance bootstrap std (B=100).
  Inverse-std becomes a global model weight in the fusion.
- **Mechanism:** stable models (low OOF-C bootstrap variance) get higher
  weight, all patients fused with the same weight vector.
- **Distinct from CV-weighted:** captures OOF-prediction stability under
  resampling, NOT cross-fold variance from the original CV. Top-5 hep
  weights: `[0.051, 0.054, 0.060, 0.051, 0.044]` — mild spread, not winner-take-all.
- **Expected-LB rationale:** ρ_hep=0.85 is more aggressive than anchor; OOF
  weighted 0.8675 is solid. Likely safer than mahal_blend/conformal but
  doesn't dominate disagree_soft.

### 4. mahal_blend
- **Signal:** Mahalanobis distance to training centroid in 20-PC feature
  space (snapshot+clinical_pro, standardized, Ledoit-Wolf-style cov + 1e-3 I).
- **Mechanism:** sigmoid gate on per-val-patient distance: low-distance
  (in-distribution) patients use anchor; high-distance patients use rank_median
  of the full pool (defensive). Mean gate value on val = 0.53 → roughly half
  of patients are tilted toward the defensive arm.
- **Distinct from anomaly_blend_meta:** uses Mahalanobis (continuous, model-based)
  not IsolationForest; gate is feature-distance not anomaly score; defensive
  arm is full-pool median (not the best-on-low-anomaly variant).
- **Expected-LB rationale:** highest ρ_death (0.93) of the new set — useful if
  death is actually the harder endpoint at val time. Lower OOF weighted (0.858)
  because rank_median underperforms anchor on training; this is a defensive
  candidate, not an upside candidate.

### 5. conformal
- **Signal:** per-base-model conformal calibrator on OOF rank-residuals vs a
  rank target (event-time-ordered). Residual quantile (1-α=0.8) = "interval
  width"; inverse-width = global fusion weight.
- **Mechanism:** weighted aggregation across the pool, weights set by
  calibration tightness — distinct from bootstrap_var (calibration vs
  resampling stability are correlated but not identical).
- **Expected-LB rationale:** lowest OOF weighted (0.856) and lowest ρ_hep (0.82).
  Calibration-driven weights aren't differentiating models enough on this small
  cohort. Submit only if budget allows; not a top pick.

## Recommendation (LB-EV ordering)

1. **disagree_soft** — highest OOF-weighted, distinct mechanism, soft gate
   (no bucket noise), specialist picked from training performance.
2. **specialist** — high ρ vs anchor (low downside risk), targeted override on
   the high-uncertainty 20% tail with externally-validated trust flag.
3. **bootstrap_var** — stable upgrade over equal-weight, no per-patient routing
   to fail at val time.
4. **mahal_blend** — defensive (best for the death endpoint).
5. **conformal** — last; submit if extra slot available.

## Files

- Script: `/Users/apunt/repos/KIRBy/tests/experiments/annitia/uncertainty_candidates.py`
- Submissions: `/Users/apunt/repos/KIRBy/tests/results/annitia/full/sub_master_unc_<variant>.csv`
- Per-variant summary JSON: `/Users/apunt/repos/KIRBy/tests/results/annitia/full/sub_master_unc_<variant>.summary.json`
- Anchor: `/Users/apunt/repos/KIRBy/tests/results/annitia/full/sub_master_meta_avg_aware_weighted.csv`

## Caveats

- OOF C systematically understates val LB for variants that lean on the
  real anchor CSV at val time (mahal_blend, disagree_soft, specialist) —
  the OOF version uses rank_mean of the pool as a proxy anchor.
- Spearman ρ vs anchor is a downside-risk proxy: ρ < 0.85 means the variant
  is moving rankings significantly; ρ > 0.95 means almost no change. The
  five candidates span ρ_hep 0.82–0.96 — a real signal spread.
- All five used the existing `branches/` pool only (22-hep / 16-death base
  models). No retraining; no new base predictions.
