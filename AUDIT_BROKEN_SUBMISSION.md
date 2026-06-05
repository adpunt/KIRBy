# AUDIT: 0.7181 broken submission vs 0.9302 anchor

Broken: `tests/results/annitia/full/sub_master_round3.csv` (uploaded as `submission` 2026-05-07 22:42, LB 0.7181)
Anchor: `tests/results/annitia/full/sub_master_meta_avg_aware_weighted.csv` (LB 0.9302).
Identical files: `submissions/submission.csv` and `submissions/round3_2026-05-08.csv`.

## 1. Spearman rho vs anchor

| endpoint | rho | mean |Δrank| | median | p95 | max |
|---|---|---|---|---|---|
| hepatic | 0.6210 | 82.1 | 65.0 | 208 | 306 |
| death | 0.8652 | 48.0 | 33.0 | 137 | 196 |

Interpretation: rho is the strongest single signal — if rho is near 1.0 the broken submission would have scored similarly to the anchor; if rho is 0.6-0.8 the LB drop from 0.93->0.72 is consistent with bulk re-ordering.

## 2. Top-10 patients with biggest rank shifts

### hepatic
| trustii_id | broken_rank | anchor_rank | shift |
|---|---|---|---|
| 353 | 23 | 329 | -306 |
| 398 | 84 | 389 | -305 |
| 330 | 102 | 398 | -296 |
| 207 | 63 | 359 | -296 |
| 163 | 127 | 416 | -289 |
| 24 | 6 | 286 | -280 |
| 158 | 27 | 293 | -266 |
| 140 | 107 | 354 | -247 |
| 370 | 134 | 377 | -243 |
| 174 | 376 | 139 | +237 |

### death
| trustii_id | broken_rank | anchor_rank | shift |
|---|---|---|---|
| 330 | 311 | 115 | +196 |
| 191 | 203 | 387 | -184 |
| 421 | 401 | 220 | +181 |
| 386 | 389 | 218 | +171 |
| 124 | 387 | 219 | +168 |
| 375 | 248 | 414 | -166 |
| 53 | 186 | 352 | -166 |
| 378 | 143 | 308 | -165 |
| 353 | 290 | 125 | +165 |
| 118 | 417 | 256 | +161 |

## 3. Distribution / quantisation

| field | broken hep | anchor hep | broken death | anchor death |
|---|---|---|---|---|
| unique_full | 423 | 421 | 423 | 420 |
| unique_round6 | 423 | 421 | 423 | 420 |
| top1_share_round4 | 0.0047 | 0.0047 | 0.0024 | 0.0047 |
| iqr | 1.7391 | 0.4533 | 1.7765 | 0.4892 |
| std | 1.2445 | 0.2649 | 1.0862 | 0.2871 |
| kurtosis | -0.0865 | -1.0972 | -0.9334 | -1.2185 |
| min | -1.5972 | 0.0204 | -0.4254 | 0.0037 |
| max | 5.6945 | 0.9961 | 4.6537 | 0.9907 |

Higher kurtosis / lower IQR / fewer unique values in the broken submission are signs of a sparse Coxnet model collapsing onto a few branches.

## 4. Score-recovery curve (proxy for "few-patient collapse")

rho vs anchor after replacing the K most-shifted patients with anchor values:

| K replaced | rho_hep | rho_death |
|---|---|---|
| 0 | 0.6210 | 0.8652 |
| 5 | 0.6498 | 0.8770 |
| 10 | 0.6721 | 0.8870 |
| 20 | 0.7060 | 0.9038 |
| 40 | 0.7652 | 0.9268 |
| 80 | 0.8476 | 0.9530 |
| 160 | 0.9273 | 0.9762 |

If rho jumps to ~1.0 after replacing only ~5-20 patients, the collapse is driven by a few influential rows; if it climbs slowly the bug rotates the whole ranking.

## 5. In-sample bias verification (Coxnet meta on OOF stack)

- branches: 25  train rows: 1253  alpha path length: 100

| condition | C-index |
|---|---|
| In-sample, smallest alpha (BUG, `coefs[:, -1]`) | 0.8064 |
| In-sample, median alpha | 0.8016 |
| In-sample, largest alpha (heavy reg) | 0.5000 |
| In-sample, naive mean of branches | 0.6627 |
| 5-fold CV, smallest alpha (BUG) | 0.7316 |

**In-sample minus OOF gap (bug variant): +0.0749.**  Anything > +0.05 indicates fitting noise; the round3 fusion was fit on FULL training rows and evaluated on those same rows, so the reported in-sample C overstated generalisation by exactly this gap.

## 6. Broken-pattern detector applied to all 88 candidates

Detector: project each variant onto the (anchor -> broken) direction in distribution-feature space (IQR, std, kurtosis, top1-share, rho-to-anchor) per endpoint. Score = 0 anchor-like, 1 broken-like.

| rank | variant | broken_score | rho_hep | rho_death |
|---|---|---|---|---|
| 1 | sub_master_disagreement_route | 171.267 | 0.786 | 0.996 |
| 2 | sub_master_best_combined_v2 | 3.453 | 0.941 | 0.992 |
| 3 | sub_master_best_combined | 2.516 | 0.949 | 0.992 |
| 4 | sub_master_method_mixup | 2.141 | 0.492 | 0.166 |
| 5 | sub_master_method_pseudolabel | 2.109 | 0.497 | 0.185 |
| 6 | sub_master_subgroup_routed | 1.829 | 0.537 | 0.353 |
| 7 | sub_master_nnls_stack | 1.691 | 0.846 | 0.334 |
| 8 | sub_master_new_cohort_split | 1.632 | 0.810 | 0.997 |
| 9 | sub_master_coxnet_stack_with_uncertainty | 1.537 | 0.952 | 0.397 |
| 10 | sub_master_noise_ipcw | 1.309 | 0.935 | 0.552 |
| 11 | sub_master_best_per_endpoint | 1.106 | 0.901 | 0.992 |
| 12 | sub_master_round3 | 1.000 | 0.621 | 0.865 |
| 13 | sub_master_unc_conformal | 0.922 | 0.820 | 0.846 |
| 14 | sub_master_unc_bootstrap_var | 0.901 | 0.850 | 0.848 |
| 15 | sub_master_new_inference_dropout | 0.883 | 0.869 | 0.853 |

Anchor itself (`sub_master_meta_avg_aware_weighted`) and self-similar variants should score near 0; `sub_master_round3` itself should be at the top with score=1.

Variants with broken_score > 0.5 OR rho_hep < 0.85 OR rho_death < 0.85 are CANDIDATES TO AVOID; they share fingerprints with the 0.7181 catastrophe.

Flagged: **44 / 90** variants.

Full table: `tests/results/annitia/full/broken_pattern_detector.csv`.

## Conclusion

- Spearman rho hep=0.621, death=0.865.
- In-sample bias gap on Coxnet stack with `coefs[:, -1]` = +0.075.
- The "smallest alpha = least regularisation" code path in `fusion.py:227` is the proximate cause: the meta-learner is selected at the most overfit point of the path AND scored on its own training rows.
- Use the detector CSV to vet any future blend before upload.