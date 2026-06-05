# ANNITIA sensitivity audit

Top 15 variants × 5 stress tests. Generated 2026-05-08.

**Distinct proxy fingerprints:** 2 of 15 variants. Most "meta_avg*" recipes collapse to the same rank-mean proxy in `variant_proxy_oof`, so their sensitivity numbers are identical by construction — interpret with care.

## Robustness ranking (ascending avg LB drop = most robust first)

| Rank | Variant | Public LB | Base OOF LB | Δperm20 | Δcensor20 | Δdur_noise | Δtop5_del | avg drop |
|---|---|---|---|---|---|---|---|---|
| 1 | `sub_master_meta_avg_aware_weighted` | 0.9302 | 0.8569 | +0.0638 | +0.0053 | -0.0005 | +0.0370 | **+0.0264** |
| 2 | `sub_master_meta_avg_all` | 0.9294 | 0.8569 | +0.0638 | +0.0053 | -0.0005 | +0.0370 | **+0.0264** |
| 3 | `sub_master_meta_avg_grand_all` | 0.9294 | 0.8569 | +0.0638 | +0.0053 | -0.0005 | +0.0370 | **+0.0264** |
| 4 | `sub_master_meta_avg_weighted` | 0.9284 | 0.8569 | +0.0638 | +0.0053 | -0.0005 | +0.0370 | **+0.0264** |
| 5 | `sub_master_meta_avg_top5` | 0.9278 | 0.8569 | +0.0638 | +0.0053 | -0.0005 | +0.0370 | **+0.0264** |
| 6 | `sub_master_meta_avg` | 0.9269 | 0.8569 | +0.0638 | +0.0053 | -0.0005 | +0.0370 | **+0.0264** |
| 7 | `sub_master_FINAL` | 0.9268 | 0.8569 | +0.0638 | +0.0053 | -0.0005 | +0.0370 | **+0.0264** |
| 8 | `sub_master_best_combined` | 0.9123 | 0.8569 | +0.0638 | +0.0053 | -0.0005 | +0.0370 | **+0.0264** |
| 9 | `results_stack` | 0.9120 | 0.8569 | +0.0638 | +0.0053 | -0.0005 | +0.0370 | **+0.0264** |
| 10 | `master_2way` | 0.9119 | 0.8569 | +0.0638 | +0.0053 | -0.0005 | +0.0370 | **+0.0264** |
| 11 | `results_r2` | 0.9099 | 0.8569 | +0.0638 | +0.0053 | -0.0005 | +0.0370 | **+0.0264** |
| 12 | `annitia_submission` | 0.8898 | 0.8569 | +0.0638 | +0.0053 | -0.0005 | +0.0370 | **+0.0264** |
| 13 | `sub_master_rank_median` | 0.9112 | 0.8575 | +0.0667 | +0.0058 | -0.0002 | +0.0376 | **+0.0275** |

## Fragile variants (DO NOT submit — single-fold luck risk)

Variants in the top tercile of avg-drop. These are the most sensitive to label / censoring / time / patient perturbations:

- `sub_master_meta_avg_aware_weighted` (avg drop **+0.0264**, public LB 0.9302)
- `sub_master_meta_avg_all` (avg drop **+0.0264**, public LB 0.9294)
- `sub_master_meta_avg_grand_all` (avg drop **+0.0264**, public LB 0.9294)
- `sub_master_meta_avg_weighted` (avg drop **+0.0264**, public LB 0.9284)
- `sub_master_meta_avg_top5` (avg drop **+0.0264**, public LB 0.9278)
- `sub_master_meta_avg` (avg drop **+0.0264**, public LB 0.9269)
- `sub_master_FINAL` (avg drop **+0.0264**, public LB 0.9268)
- `sub_master_best_combined` (avg drop **+0.0264**, public LB 0.9123)
- `results_stack` (avg drop **+0.0264**, public LB 0.9120)
- `master_2way` (avg drop **+0.0264**, public LB 0.9119)
- `results_r2` (avg drop **+0.0264**, public LB 0.9099)
- `annitia_submission` (avg drop **+0.0264**, public LB 0.8898)
- `sub_master_rank_median` (avg drop **+0.0275**, public LB 0.9112)

## Most "load-bearing" training patients

Patients whose removal MOST drops C across variants. These are the ones any val_test bug-spotting effort should examine first.

### Hepatic endpoint

| Rank | Patient idx | Patient ID | # variants in top-5 |
|---|---|---|---|
| 1 | 627 | `TLCQFMUIW7J3` | 13/13 |
| 2 | 650 | `52ALUNQR8O1K` | 13/13 |
| 3 | 178 | `45BG34IQ7ZJ3` | 13/13 |
| 4 | 591 | `BKTMAV6763V1` | 13/13 |
| 5 | 77 | `POYIPK0QTL3Q` | 13/13 |

### Death endpoint

| Rank | Patient idx | Patient ID | # variants in top-5 |
|---|---|---|---|
| 1 | 129 | `ZPKY32ORWIXZ` | 13/13 |
| 2 | 728 | `11ZSB8LV6TX0` | 13/13 |
| 3 | 293 | `RW6TM2O04K0L` | 13/13 |
| 4 | 454 | `GUO8I5JWK6Z0` | 13/13 |
| 5 | 851 | `4EPXR9OGJEGX` | 12/13 |
| 6 | 503 | `M5HGBO06FHRX` | 1/13 |

## Stress-test summary tables

### A. Label permutation (mean over 5 seeds)

Weighted LB at each permutation fraction, averaged across the 5 perturbation seeds. Variants degrade roughly linearly with permutation fraction; faster degradation = more reliance on noisy labels (could be either bad signal or genuine signal).

| Variant | f=0 | f=0.05 | f=0.10 | f=0.20 |
|---|---|---|---|---|
| `sub_master_meta_avg_aware_weighted` | 0.8569 | 0.8354 | 0.8166 | 0.7932 |
| `sub_master_meta_avg_all` | 0.8569 | 0.8354 | 0.8166 | 0.7932 |
| `sub_master_meta_avg_grand_all` | 0.8569 | 0.8354 | 0.8166 | 0.7932 |
| `sub_master_meta_avg_weighted` | 0.8569 | 0.8354 | 0.8166 | 0.7932 |
| `sub_master_meta_avg_top5` | 0.8569 | 0.8354 | 0.8166 | 0.7932 |
| `sub_master_meta_avg` | 0.8569 | 0.8354 | 0.8166 | 0.7932 |
| `sub_master_FINAL` | 0.8569 | 0.8354 | 0.8166 | 0.7932 |
| `sub_master_best_combined` | 0.8569 | 0.8354 | 0.8166 | 0.7932 |
| `results_stack` | 0.8569 | 0.8354 | 0.8166 | 0.7932 |
| `master_2way` | 0.8569 | 0.8354 | 0.8166 | 0.7932 |
| `sub_master_rank_median` | 0.8575 | 0.8359 | 0.8178 | 0.7909 |
| `results_r2` | 0.8569 | 0.8354 | 0.8166 | 0.7932 |
| `annitia_submission` | 0.8569 | 0.8354 | 0.8166 | 0.7932 |

### C. Pair concentration (top 1% of pairs share of total C)

| Variant | hep total pairs | hep top1% conc. | death total pairs | death top1% conc. |
|---|---|---|---|---|
| `sub_master_meta_avg_aware_weighted` | 17158 | 0.012 | 19312 | 0.011 |
| `sub_master_meta_avg_all` | 17158 | 0.012 | 19312 | 0.011 |
| `sub_master_meta_avg_grand_all` | 17158 | 0.012 | 19312 | 0.011 |
| `sub_master_meta_avg_weighted` | 17158 | 0.012 | 19312 | 0.011 |
| `sub_master_meta_avg_top5` | 17158 | 0.012 | 19312 | 0.011 |
| `sub_master_meta_avg` | 17158 | 0.012 | 19312 | 0.011 |
| `sub_master_FINAL` | 17158 | 0.012 | 19312 | 0.011 |
| `sub_master_best_combined` | 17158 | 0.012 | 19312 | 0.011 |
| `results_stack` | 17158 | 0.012 | 19312 | 0.011 |
| `master_2way` | 17158 | 0.012 | 19312 | 0.011 |
| `sub_master_rank_median` | 17158 | 0.012 | 19312 | 0.011 |
| `results_r2` | 17158 | 0.012 | 19312 | 0.011 |
| `annitia_submission` | 17158 | 0.012 | 19312 | 0.011 |

## Caveats

- `variant_proxy_oof` collapses meta-aware variants to a rank-mean fallback. The 15 stems map to only 3 recipes (`rank_mean`, `rank_median`, `rank_trimmed`). Differential rankings across variants reflect that small recipe set, not the actual submission diversity. Treat `meta_avg*` rows as a single point estimate.
- All concordances are computed on training-side OOF, not the public leaderboard. Use as a robustness *signal*, not as an LB predictor.
- Adversarial-deletion top-5 sets are computed independently per endpoint — overlap across variants is the diagnostic signal, not absolute drop magnitude.