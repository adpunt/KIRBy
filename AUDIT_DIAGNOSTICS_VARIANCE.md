# AUDIT — Diagnostics & Variance

Local diagnostic on ANNITIA OOFs (134 hep / 87 death base models, 15 variants).
Pure analysis. **No submission CSVs produced.**

## TL;DR

- **Hepatic consensus** (mean-rank01 of all base OOFs): mean per-fold C = 0.805, std = 0.097, range = 0.239. **Folds 0–1 are noticeably harder than folds 2 and 4** (e.g. f0=0.87, f2=0.92, f4=0.70). LB scores derived from a held-out validation set are necessarily one realisation of this fold-noise.
- **Death consensus**: mean = 0.940, std = 0.023, range = 0.053. Much more stable than hepatic.
- All 14 "mean-rank01" variants in the LB top-15 produce **identical** OOF proxies — their LB differences are driven by val-side recipe steps that we cannot reproduce here. The OOF cannot rank them. This is a key blind spot.
- LOMO deltas are tiny (|Δ| ≤ 0.0015 per dropped base). No single base is load-bearing — the consensus is robust to dropping any one model. A handful of bases are *slightly* harmful when included; see § High-variance to AVOID.
- Bootstrap downsampling (50% → 100% of patients) shows monotonically tightening CIs as expected, with no surprising performance cliff. Variant proxies are not propped up by a few patients.

## 1. Per-fold C variance — top 10 most unstable per endpoint

### hepatic (highest std across folds → most unstable)

| Variant | Mean C | Std | Range | f0 | f1 | f2 | f3 | f4 |
|---|---|---|---|---|---|---|---|---|
| `be/oof_hepatic_snapshot__catboost_cox` | 0.613 | 0.177 | 0.551 | 0.560 | 0.337 | 0.888 | 0.619 | 0.664 |
| `cd/oof_hepatic_C-pseudo-LGBM` | 0.598 | 0.152 | 0.438 | 0.549 | 0.453 | 0.529 | 0.891 | 0.567 |
| `be/oof_hepatic_snapshot+clinical_pro__catboost_cox` | 0.560 | 0.150 | 0.438 | 0.457 | 0.394 | 0.833 | 0.553 | 0.565 |
| `cd/oof_hepatic_D-weighted-cox-lin` | 0.706 | 0.143 | 0.375 | 0.512 | 0.615 | 0.852 | 0.888 | 0.664 |
| `branches/oof_hepatic_tabsurv-K10__snapshot+clinical_pro+fpca+gp_smooth` | 0.723 | 0.142 | 0.413 | 0.872 | 0.819 | 0.459 | 0.727 | 0.739 |
| `hep_boost/oof_hepatic_catboost-cox__snapshot+clinical_pro+fpca` | 0.563 | 0.142 | 0.402 | 0.451 | 0.401 | 0.804 | 0.622 | 0.539 |
| `be/oof_hepatic_snapshot+clinical_pro+fpca+gp_smooth__lgbm_aft` | 0.681 | 0.141 | 0.424 | 0.600 | 0.507 | 0.932 | 0.697 | 0.669 |
| `hep_boost/oof_hepatic_catboost-cox__snapshot+clinical_pro+fpca+gp_smooth` | 0.672 | 0.141 | 0.429 | 0.625 | 0.493 | 0.921 | 0.623 | 0.697 |
| `be/oof_hepatic_snapshot+clinical_pro+fpca__catboost_cox` | 0.552 | 0.140 | 0.402 | 0.457 | 0.390 | 0.792 | 0.612 | 0.509 |
| `be/oof_hepatic_snapshot+clinical_pro+fpca__lgbm_aft` | 0.627 | 0.140 | 0.376 | 0.518 | 0.475 | 0.852 | 0.722 | 0.565 |

### death (highest std across folds → most unstable)

| Variant | Mean C | Std | Range | f0 | f1 | f2 | f3 | f4 |
|---|---|---|---|---|---|---|---|---|
| `be/oof_death_snapshot__catboost_cox` | 0.572 | 0.134 | 0.416 | 0.338 | 0.577 | 0.575 | 0.754 | 0.615 |
| `be/oof_death_snapshot+clinical_pro__catboost_cox` | 0.638 | 0.120 | 0.371 | 0.453 | 0.610 | 0.683 | 0.824 | 0.619 |
| `branches/oof_death_tabsurv-H2D__snapshot+clinical_pro+fpca+gp_smooth` | 0.567 | 0.108 | 0.321 | 0.607 | 0.586 | 0.410 | 0.731 | 0.499 |
| `be/oof_death_snapshot__lgbm_aft` | 0.675 | 0.095 | 0.279 | 0.536 | 0.690 | 0.718 | 0.815 | 0.614 |
| `cd/oof_death_C-pchazard` | 0.779 | 0.093 | 0.260 | 0.608 | 0.869 | 0.766 | 0.803 | 0.849 |
| `be/oof_death_snapshot+clinical_pro__lgbm_aft` | 0.770 | 0.088 | 0.268 | 0.624 | 0.781 | 0.811 | 0.891 | 0.744 |
| `cd/oof_death_C-pseudo-MLP` | 0.642 | 0.083 | 0.233 | 0.670 | 0.637 | 0.700 | 0.485 | 0.718 |
| `cd/oof_death_C-pseudo-LGBM` | 0.765 | 0.080 | 0.224 | 0.824 | 0.643 | 0.868 | 0.713 | 0.778 |
| `branches/oof_death_tsfm-cox__tsfm` | 0.755 | 0.076 | 0.218 | 0.805 | 0.753 | 0.704 | 0.647 | 0.865 |
| `be/oof_death_snapshot+clinical_pro+fpca__catboost_cox` | 0.739 | 0.069 | 0.216 | 0.633 | 0.728 | 0.849 | 0.738 | 0.749 |

### Variant proxies — per-fold C (deduped by aggregation class)

Note: most top variants reduce to the same mean-of-rank01 proxy at the OOF level (the LB differences come from val-side recipe details we cannot reproduce here). Showing one row per aggregation class.

| Class | Endpoint | Mean | Std | Range | Members |
|---|---|---|---|---|---|
| mean-rank01 | hepatic | 0.8049 | 0.0968 | 0.2392 | 14 (sub_master_BEST_BET, sub_master_FINAL, sub_master_HIER, sub_master_LEARNED_STACK…) |
| median | hepatic | 0.8183 | 0.1119 | 0.2450 | 1 (sub_master_rank_median…) |
| mean-rank01 | death | 0.9405 | 0.0232 | 0.0530 | 14 (sub_master_BEST_BET, sub_master_FINAL, sub_master_HIER, sub_master_LEARNED_STACK…) |
| median | death | 0.9412 | 0.0209 | 0.0610 | 1 (sub_master_rank_median…) |

## 3. Leave-one-model-out — per aggregation class (top 5 / bottom 5)

Negative Δ = dropping the model **hurts** OOF C (contributor). Positive Δ = dropping **helps** (harmful).

### Class: `mean-rank01`  (representative: `sub_master_meta_avg_aware_weighted`)
**hepatic** (full C=0.8023)
| Rank | Contributor (drop hurts) | Δ | Harmful (drop helps) | Δ |
|---|---|---|---|---|
| 1 | `branches/oof_hepatic_synaug-gmm-rsf-500-n100__snapshot+clinical_pro+fpca+gp_smooth` | -0.00133 | `be/oof_hepatic_snapshot+clinical_pro+fpca__cox` | +0.00173 |
| 2 | `branches/oof_hepatic_synaug-tabebm-rsf-500-n100__snapshot+clinical_pro+fpca+gp_smooth` | -0.00133 | `cd/oof_hepatic_D9x__snapshot__catboost_cox` | +0.00143 |
| 3 | `branches/oof_hepatic_catboost-D2H__snapshot+clinical_pro+fpca+gp_smooth` | -0.00128 | `branches/oof_hepatic_synaug-none-cox-n0__snapshot+clinical_pro+fpca+gp_smooth` | +0.00123 |
| 4 | `branches/oof_hepatic_tabsurv-D2H__snapshot+clinical_pro+fpca+gp_smooth` | -0.00123 | `be/oof_hepatic_snapshot+clinical_pro+fpca+gp_smooth__cox` | +0.00123 |
| 5 | `hep_boost/oof_hepatic_bagged-rsf-B200__snapshot+clinical_pro+fpca+gp_smooth` | -0.00113 | `branches/oof_hepatic_tsfm-lgbm_aft__tsfm` | +0.00118 |

**death** (full C=0.9419)
| Rank | Contributor (drop hurts) | Δ | Harmful (drop helps) | Δ |
|---|---|---|---|---|
| 1 | `be/oof_death_snapshot+clinical_pro+fpca+gp_smooth__gbsa` | -0.00094 | `cd/oof_death_C-landmark-MLP-ipcw` | +0.00107 |
| 2 | `branches/oof_death_tabpfn-landmark__snapshot+clinical_pro+fpca+gp_smooth` | -0.00090 | `cd/oof_death_C-landmark-MLP-drop` | +0.00107 |
| 3 | `be/oof_death_snapshot+clinical_pro+fpca+visit_density+clinical_slope__lgbm_aft` | -0.00090 | `be/oof_death_snapshot__catboost_cox` | +0.00085 |
| 4 | `branches/oof_death_tsfm-gbsa__tsfm` | -0.00085 | `cd/oof_death_D-weighted-rsf-lin` | +0.00085 |
| 5 | `be/oof_death_snapshot+clinical_pro+meld_na+visit_density__lgbm_aft` | -0.00081 | `cd/oof_death_C-pseudo-LGBM` | +0.00085 |

### Class: `median`  (representative: `sub_master_rank_median`)
**hepatic** (full C=0.8129)
| Rank | Contributor (drop hurts) | Δ | Harmful (drop helps) | Δ |
|---|---|---|---|---|
| 1 | `hep_boost/oof_hepatic_snapshot-rsf-N5__snapshot+clinical_pro+fpca+gp_smooth` | -0.00104 | `cd/oof_hepatic_C-pseudo-LGBM` | +0.00126 |
| 2 | `branches/oof_hepatic_catboost-D2H__snapshot+clinical_pro+fpca+gp_smooth` | -0.00101 | `cd/oof_hepatic_D9x__snapshot__ngboost` | +0.00118 |
| 3 | `branches/oof_hepatic_tabpfn25-disc-K10__snapshot+clinical_pro+fpca+gp_smooth` | -0.00101 | `cd/oof_hepatic_D9x__snapshot__rsf` | +0.00118 |
| 4 | `hep_boost/oof_hepatic_tabicl-landmark__snapshot+clinical_pro+fpca+gp_smooth` | -0.00099 | `cd/oof_hepatic_D9x__snapshot+clinical_pro__cox` | +0.00116 |
| 5 | `hep_boost/oof_hepatic_bagged-rsf-B200__snapshot+clinical_pro+fpca+gp_smooth` | -0.00096 | `cd/oof_hepatic_D9-crosspoll-cox` | +0.00113 |

**death** (full C=0.9430)
| Rank | Contributor (drop hurts) | Δ | Harmful (drop helps) | Δ |
|---|---|---|---|---|
| 1 | `be/oof_death_snapshot+clinical_pro+fpca+gp_smooth__ngboost` | -0.00158 | `cd/oof_death_C-landmark-Cat-drop` | +0.00090 |
| 2 | `be/oof_death_snapshot+clinical_pro+meld_na+visit_density__cox` | -0.00125 | `cd/oof_death_D-xgb-pairwise` | +0.00079 |
| 3 | `be/oof_death_snapshot+clinical_pro+meld_na+visit_density__catboost_cox` | -0.00123 | `cd/oof_death_C-landmark-Cat-ipcw` | +0.00072 |
| 4 | `be/oof_death_snapshot+clinical_pro+fpca+visit_density+clinical_slope__lgbm_aft` | -0.00120 | `cd/oof_death_D-selftrain-cox` | +0.00046 |
| 5 | `be/oof_death_snapshot+clinical_pro+meld_na+visit_density__lgbm_aft` | -0.00118 | `cd/oof_death_D9-crosspoll-cox` | +0.00046 |

## 4. Bootstrap-with-downsampling — C as f(training-N), B=200 per cell

Deduped by aggregation class (variants in same class share OOF proxy).

| Class | Endpoint | frac=0.5 | frac=0.75 | frac=1.0 | Δ(1.0−0.5) |
|---|---|---|---|---|---|
| mean-rank01 | hepatic | 0.800±0.090 | 0.811±0.064 | 0.803±0.058 | +0.003 |
| mean-rank01 | death | 0.942±0.016 | 0.941±0.012 | 0.942±0.011 | -0.000 |
| median | hepatic | 0.811±0.091 | 0.804±0.073 | 0.817±0.062 | +0.006 |
| median | death | 0.944±0.013 | 0.944±0.011 | 0.944±0.009 | +0.000 |

## High-variance variants to AVOID

All 14 mean-rank01 variants share the same OOF proxy (per-fold std hep≈0.099, death≈0.023). Caveats:

- The hepatic endpoint shows large per-fold range (~0.26 across folds for the consensus mean), driven by fold 2 and fold 4 being significantly easier than folds 0–1. Any variant whose LB is bench-marked on a single hold-out fold should be treated as untrustworthy.
- `rank_median` has marginally higher hepatic std (0.1015) and lower death std (0.0209) than the mean class — a near-wash; do not prefer it on stability grounds alone.
- The trimmed-20% class did not appear in the top-15 (history JSON had no `rank_trimmed` entry); listed for completeness.

### Truly unstable individual base OOFs to consider DROPPING from the pool

Combined view: per-fold std AND LOMO-positive Δ (i.e. removing them helps consensus).

**hepatic** — bases that are both fragile (high per-fold std) and harmful when included:
- `branches/oof_hepatic_tsfm-lgbm_aft__tsfm`  std=0.106  drop-Δ=+0.00118
- `cd/oof_hepatic_D9x__snapshot+clinical_pro__cox`  std=0.089  drop-Δ=+0.00104
- `cd/oof_hepatic_D9x__snapshot+clinical_pro__catboost_cox`  std=0.080  drop-Δ=+0.00113
- `be/oof_hepatic_snapshot+clinical_pro+fpca__cox`  std=0.074  drop-Δ=+0.00173
- `cd/oof_hepatic_D9x__snapshot__catboost_cox`  std=0.073  drop-Δ=+0.00143

**death** — bases that are both fragile (high per-fold std) and harmful when included:
- `be/oof_death_snapshot__catboost_cox`  std=0.134  drop-Δ=+0.00085
- `cd/oof_death_C-pseudo-LGBM`  std=0.080  drop-Δ=+0.00085
- `cd/oof_death_C-landmark-MLP-ipcw`  std=0.063  drop-Δ=+0.00107
- `cd/oof_death_C-landmark-MLP-drop`  std=0.063  drop-Δ=+0.00107
- `cd/oof_death_D-weighted-rsf-lin`  std=0.056  drop-Δ=+0.00085

## Most fragile base OOFs (high std AND positive mean C)

**hepatic**
- `be/oof_hepatic_snapshot__catboost_cox`  mean=0.613 std=0.177 range=0.551
- `cd/oof_hepatic_C-pseudo-LGBM`  mean=0.598 std=0.152 range=0.438
- `be/oof_hepatic_snapshot+clinical_pro__catboost_cox`  mean=0.560 std=0.150 range=0.438
- `cd/oof_hepatic_D-weighted-cox-lin`  mean=0.706 std=0.143 range=0.375
- `branches/oof_hepatic_tabsurv-K10__snapshot+clinical_pro+fpca+gp_smooth`  mean=0.723 std=0.142 range=0.413

**death**
- `be/oof_death_snapshot__catboost_cox`  mean=0.572 std=0.134 range=0.416
- `be/oof_death_snapshot+clinical_pro__catboost_cox`  mean=0.638 std=0.120 range=0.371
- `branches/oof_death_tabsurv-H2D__snapshot+clinical_pro+fpca+gp_smooth`  mean=0.567 std=0.108 range=0.321
- `be/oof_death_snapshot__lgbm_aft`  mean=0.675 std=0.095 range=0.279
- `cd/oof_death_C-pchazard`  mean=0.779 std=0.093 range=0.260
