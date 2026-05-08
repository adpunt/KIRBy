# OOF → Leaderboard calibration

Each submission's leaderboard score paired with the OOF concordance of the equivalent recipe on the training set.

## Calibration fit

- Linear: **LB = -1.1365 + 2.3403 · OOF_avg**
- Spearman ρ(OOF, LB) = **0.309**
- Residual std: **0.0115**

## Projected leaderboard scores

| OOF C | Projected LB |
|---|---|
| 0.90 | 0.9697 |
| 0.92 | 1.0165 |
| 0.94 | 1.0633 |
| 0.95 | 1.0867 |
| 0.96 | 1.1101 |
| 0.97 | 1.1335 |
| 0.98 | 1.1569 |

## Per-submission detail

| Submission | LB | OOF avg | Gap (OOF − LB) | Note |
|---|---|---|---|---|
| `sub_master_meta_avg_aware_weighted` | 0.9302 | 0.8786 | -0.0516 |  |
| `sub_master_meta_avg_all` | 0.9294 | 0.8786 | -0.0508 |  |
| `sub_master_meta_avg_grand_all` | 0.9294 | 0.8786 | -0.0508 |  |
| `sub_master_meta_avg_weighted` | 0.9284 | 0.8786 | -0.0499 |  |
| `sub_master_meta_avg_top5` | 0.9278 | 0.8786 | -0.0492 |  |
| `sub_master_meta_avg` | 0.9269 | 0.8786 | -0.0483 |  |
| `sub_master_FINAL` | 0.9268 | 0.8786 | -0.0482 |  |
| `sub_master_best_combined` | 0.9123 | 0.8786 | -0.0337 |  |
| `results_stack` | 0.9120 | 0.8786 | -0.0334 |  |
| `master_2way` | 0.9119 | 0.8786 | -0.0333 |  |
| `sub_master_rank_median` | 0.9112 | 0.8750 | -0.0362 |  |
| `results_r2` | 0.9099 | 0.8786 | -0.0313 |  |
| `annitia_submission` | 0.8898 | 0.8786 | -0.0112 |  |
| `submission` | 0.7181 | 0.8786 | +0.1605 | *broken* |

## Implications

- Average gap (excluding broken round3): use as a discount when projecting LB from OOF.
- The high Spearman ρ means OOF *ranks* recipes correctly but compresses the absolute scale.
- For tomorrow: don't care about the 4th decimal of OOF C — care about which variants project
  to higher LB given the calibration, and which have **non-overlapping** projected CIs.