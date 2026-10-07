# Precompute decisions (single record)

Source: slim copies of `result_db` pulled 2026-10-03 to /Volumes/seagate/slim_precompute (41 datasets). Scores are from the whole-dataset search (leaky, accepted). Augmentation = none only.

## Step 1: two downstream models per dataset

One row is one dataset. Each cell lists the three models with the best single-configuration score in that search, in the dataset's own metric. `general` and `general_distance` are the two searches stored per dataset. DECISION is blank until Adelaide fills it.

| dataset | metric | general: top 3 | general_distance: top 3 | same top 2? | DECISION |
|---|---|---|---|---|---|
| abagym_Ang2 | spearman | gp 0.5348, svm 0.5311, dnn 0.5199 | gp 0.5348, svm 0.5311, dnn 0.5199 | yes | **gp+svm** (2026-10-03) |
| abagym_EGFR | spearman | svm 0.3871, ridge 0.3831, knn 0.3709 | svm 0.3853, dnn 0.3817, gp 0.3586 | NO | **svm+dnn** (2026-10-03, Adelaide) |
| abagym_VEGF | spearman | gp 0.6763, knn 0.6324, svm 0.6147 | knn 0.6295, gp 0.6227, svm 0.6147 | yes (knn excluded) | **gp+svm** (2026-10-03, knn not allowed) |
| abcov_kd | spearman | xgboost 0.4990, lightgbm 0.4985, knn 0.4865 | dnn 0.4876, xgboost 0.4838, rf 0.4824 | NO | **lightgbm+xgboost** (2026-10-03, Adelaide) |
| adme_hlm | rmse | svm 0.4209, dnn 0.4309, lightgbm 0.4459 | svm 0.4209, dnn 0.4370, lightgbm 0.4461 | yes | **svm+dnn** (2026-10-03) |
| adme_mdr1 | rmse | svm 0.4306, dnn 0.4345, lightgbm 0.4559 | svm 0.4346, dnn 0.4490, lightgbm 0.4579 | yes | **svm+dnn** (2026-10-03) |
| adme_rlm | rmse | svm 0.4769, dnn 0.4912, lightgbm 0.5099 | svm 0.4811, dnn 0.5137, lightgbm 0.5146 | yes | **svm+dnn** (2026-10-03) |
| bh_doyle_plates | rmse | rf 14.9891, dnn 15.3272, lightgbm 15.3327 | rf 14.9474, dnn 15.3272, lightgbm 15.3327 | yes | **rf+lightgbm** (2026-10-07, Adelaide; was rf+dnn 2026-10-03. dnn 15.3272 and lightgbm 15.3327 are 0.0055 rmse apart against a fold spread of 2.579; lightgbm 1.7 s per configuration to dnn's 3.8) |
| bindingdb_kd | ci | rf 0.7565, dnn 0.7406, xgboost 0.7362 | no rows | NO | **rf+dnn** (2026-10-03, one search only) |
| buchwald_hartwig | rmse | rf 13.5888, lightgbm 14.0573, dnn 14.2599 | rf 13.5888, knn 13.7893, lightgbm 14.0573 | yes (knn excluded) | **rf+lightgbm** (2026-10-03, knn not allowed) |
| caco2 | rmse | dnn 0.2696, svm 0.2715, lightgbm 0.2856 | svm 0.2724, dnn 0.2772, lightgbm 0.2856 | yes | **dnn+svm** (2026-10-03) |
| cdr_flexibility | auc | xgboost 0.8936, lightgbm 0.8909, rf 0.8855 | xgboost 0.8966, lightgbm 0.8909, rf 0.8855 | yes | **xgboost+lightgbm** (2026-10-03) |
| ddg_combined | spearman | svm 0.4355, dnn 0.4096, xgboost 0.3741 | svm 0.4494, dnn 0.4096, xgboost 0.3618 | yes | **svm+dnn** (2026-10-03) |
| flab_binding_adams | auc | lightgbm 0.6316, xgboost 0.6264, dnn 0.6248 | lightgbm 0.6316, dnn 0.6275, xgboost 0.6265 | NO | **lightgbm+xgboost** (2026-10-03, Adelaide) |
| flab_binding_koenig | spearman | rf 0.5168, svm 0.5118, lightgbm 0.4999 | rf 0.5168, svm 0.5118, lightgbm 0.4999 | yes | **rf+svm** (2026-10-03) |
| flab_expression_adams | spearman | dnn 0.4281, svm 0.3881, lightgbm 0.3046 | dnn 0.4278, svm 0.3947, lightgbm 0.3046 | yes | **dnn+svm** (2026-10-03) |
| flab_expression_koenig | spearman | svm 0.7716, dnn 0.7639, rf 0.7527 | svm 0.7713, dnn 0.7553, rf 0.7527 | yes | **svm+dnn** (2026-10-03) |
| flab_thermo_tresanco | spearman | dnn 0.7096, ridge 0.6809, svm 0.6774 | dnn 0.7096, svm 0.6774, rf 0.6654 | NO | **dnn+svm** (2026-10-03, Adelaide) |
| gb_human_enhancers_cohn | mcc | dnn 0.5147, lightgbm 0.5012, rf 0.4887 | dnn 0.5118, lightgbm 0.5012, rf 0.4887 | yes | **dnn+lightgbm** (2026-10-03) |
| ginkgo_acsins | spearman | lightgbm 0.5812, gp 0.5774, dnn 0.5752 | gp 0.5774, dnn 0.5752, lightgbm 0.5691 | NO | **lightgbm+gp** (2026-10-03, Adelaide) |
| ginkgo_colloidal | spearman | dnn 0.4196, svm 0.4140, gp 0.3827 | dnn 0.4081, svm 0.4010, gp 0.3815 | yes | **dnn+svm** (2026-10-03) |
| ginkgo_hic | spearman | gp 0.5854, dnn 0.5759, ridge 0.5431 | gp 0.5818, dnn 0.5580, svm 0.5289 | yes | **gp+dnn** (2026-10-03) |
| ginkgo_pr_cho | spearman | gp 0.5471, dnn 0.5307, svm 0.5161 | dnn 0.5359, gp 0.5177, svm 0.5161 | yes | **gp+dnn** (2026-10-03) |
| ginkgo_titer | spearman | dnn 0.4369, xgboost 0.4311, rf 0.4286 | dnn 0.4396, rf 0.4286, gp 0.4231 | NO | **dnn+rf** (2026-10-03, Adelaide) |
| herg | auc | lightgbm 0.8682, xgboost 0.8662, svm 0.8661 | lightgbm 0.8687, xgboost 0.8640, svm 0.8622 | yes | **lightgbm+xgboost** (2026-10-03) |
| herg_ki | rmse | svm 0.5472, dnn 0.5537, lightgbm 0.5707 | svm 0.5473, dnn 0.5643, lightgbm 0.5707 | yes | **svm+dnn** (2026-10-03) |
| humanness_therapeutic | auc | xgboost 0.9721, rf 0.9714, svm 0.9706 | rf 0.9709, lightgbm 0.9697, xgboost 0.9692 | NO | **xgboost+rf** (2026-10-03, Adelaide) |
| kiba | ci | dnn 0.7266, rf 0.7215, lightgbm 0.7116 | no rows | NO | **dnn+rf** (2026-10-03, one search only) |
| lhasa_herg | auc | lightgbm 0.8701, xgboost 0.8674, svm 0.8666 | no rows | NO | **lightgbm+xgboost** (2026-10-03, one search only) |
| logd | rmse | dnn 0.4049, svm 0.4221, ridge 0.4426 | dnn 0.4180, svm 0.4345, lightgbm 0.4937 | yes | **dnn+svm** (2026-10-03) |
| nt_H3K27me3 | mcc | dnn 0.5687, lightgbm 0.5521, rf 0.5402 | dnn 0.5624, lightgbm 0.5513, rf 0.5402 | yes | **dnn+lightgbm** (2026-10-03) |
| nt_H3K4me3 | mcc | dnn 0.6723, lightgbm 0.6692, ridge 0.6631 | dnn 0.6688, lightgbm 0.6646, rf 0.6575 | yes | **dnn+lightgbm** (2026-10-03) |
| nt_H3K9ac | mcc | dnn 0.5297, lightgbm 0.5180, rf 0.4973 | dnn 0.5235, lightgbm 0.5159, rf 0.4982 | yes | **dnn+lightgbm** (2026-10-03) |
| nt_H3K9me3 | mcc | dnn 0.3680, lightgbm 0.3443, rf 0.3214 | dnn 0.3636, lightgbm 0.3412, rf 0.3214 | yes | **dnn+lightgbm** (2026-10-03) |
| nt_promoter_tata | mcc | lightgbm 0.7173, svm 0.7104, dnn 0.6995 | lightgbm 0.7129, svm 0.7037, dnn 0.6925 | yes | **lightgbm+svm** (2026-10-03) |
| qm9 | rmse | dnn 0.3184, lightgbm 0.3449, svm 0.3473 | dnn 0.3221, lightgbm 0.3454, svm 0.3473 | yes | **dnn+lightgbm** (2026-10-03) |
| suzuki | rmse | rf 18.9658, lightgbm 19.0343, xgboost 19.4361 | rf 18.9658, lightgbm 18.9836, xgboost 19.4671 | yes | **rf+lightgbm** (2026-10-03) |
| tcga_brca_pam50 | f1_weighted | dnn 0.9120, svm 0.9041, xgboost 0.9010 | dnn 0.9120, svm 0.9041, lightgbm 0.9001 | yes | **dnn+svm** (2026-10-03) |
| tcga_luad_vs_lusc | auc | lightgbm 0.9999, xgboost 0.9999, rf 0.9998 | lightgbm 0.9996, xgboost 0.9995, rf 0.9994 | yes | **lightgbm+xgboost** (2026-10-03) |
| tz_her2_binary | auc | xgboost 0.8599, knn 0.8536, lightgbm 0.8455 | xgboost 0.8599, svm 0.8554, knn 0.8536 | NO | **xgboost+lightgbm** (2026-10-03, Adelaide) |
| tz_her2_multiclass | f1_weighted | knn 0.7065, svm 0.6988, xgboost 0.6852 | knn 0.7065, svm 0.6988, xgboost 0.6810 | yes (knn excluded) | **svm+xgboost** (2026-10-03, knn not allowed) |

Nothing stored: chembl_qsar.


Rules from Adelaide 2026-10-03: knn is not allowed as a downstream model. gp is allowed except on datasets with massive sample sizes (none of the gp picks so far is large: 197-672 rows per BILL_OF_MATERIALS.md, a dated source). Ridge: not yet answered.
knn: not one of the two models for any dataset. knn precompute is run anyway for datasets in experiments that use it (similarity/noise); that is separate from the 2-model rule (Adelaide 2026-10-03).

## Step 2: budgets

One row is one dataset, using only that dataset's two chosen models (both as downstream model and as internal scorer), augmentation none, both stored searches pooled. Each cell is the best score reached at that budget, in the dataset's own metric (* = best budget). Fold spread = the fold-to-fold standard deviation of the single best configuration. Proposed lowest budget = the lowest budget whose best score is within one fold spread of the overall best. Nothing above it is proposed for cutting, because no budget is worse than the best by more than the spread at the top end. DECISION is blank until Adelaide fills it.

| dataset | metric | models | 10 | 25 | 50 | 100 | 250 | 500 | 1000 | 2000 | best budget | fold spread | proposed lowest budget | DECISION |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abagym_Ang2 | spearman | gp+svm | 0.389 | 0.502 | 0.531* | 0.481 | 0.489 | 0.495 | 0.464 | 0.452 | 50 | 0.153 | none | |
| abagym_EGFR | spearman | svm+dnn | 0.235 | 0.317 | 0.382 | 0.367 | 0.384 | 0.385 | 0.387* | 0.375 | 1000 | 0.078 | 25 | |
| abagym_VEGF | spearman | gp+svm | 0.676* | 0.637 | 0.606 | 0.600 | 0.650 | 0.666 | 0.612 | 0.591 | 10 | 0.089 | none | |
| abcov_kd | spearman | lightgbm+xgboost | 0.392 | 0.479 | 0.499* | 0.428 | 0.435 | 0.448 | 0.428 | 0.471 | 50 | 0.143 | none | |
| adme_hlm | rmse | svm+dnn | 0.503 | 0.480 | 0.459 | 0.444 | 0.427 | 0.422* | 0.422 | 0.422 | 500 | 0.005 | 500 | |
| adme_mdr1 | rmse | svm+dnn | 0.521 | 0.490 | 0.470 | 0.453 | 0.439 | 0.435 | 0.431* | 0.432 | 1000 | 0.016 | 250 | |
| adme_rlm | rmse | svm+dnn | 0.577 | 0.551 | 0.526 | 0.510 | 0.489 | 0.481 | 0.478 | 0.477* | 2000 | 0.014 | 250 | |
| bh_doyle_plates | rmse | rf+dnn | 16.984 | 15.737 | 15.796 | 15.327* | 15.369 | 15.597 | 15.662 | 15.711 | 100 | 2.579 | none | |
| bindingdb_kd | ci | rf+dnn | 0.716 | 0.737 | 0.742 | 0.747 | 0.749 | 0.751 | 0.751 | 0.756* | 2000 | 0.014 | 100 | |
| buchwald_hartwig | rmse | rf+lightgbm | 15.902 | 15.004 | 14.259 | 14.227 | 13.837 | 13.589* | 13.704 | 13.686 | 500 | 2.931 | none | |
| caco2 | rmse | dnn+svm | 0.335 | 0.300 | 0.287 | 0.283 | 0.276 | 0.273 | 0.272 | 0.270* | 2000 | 0.027 | 50 | |
| cdr_flexibility | auc | xgboost+lightgbm | 0.885 | 0.887 | 0.887* | 0.884 | 0.882 | 0.877 | 0.879 | 0.875 | 50 | 0.039 | none | |
| ddg_combined | spearman | svm+dnn | 0.264 | 0.285 | 0.305 | 0.398 | 0.436* | 0.368 | 0.363 | 0.397 | 250 | 0.104 | 100 | |
| flab_binding_adams | auc | lightgbm+xgboost | 0.619 | 0.624 | 0.627 | 0.632* | 0.628 | 0.628 | 0.626 | 0.626 | 100 | 0.039 | none | |
| flab_binding_koenig | spearman | rf+svm | 0.445 | 0.480 | 0.502 | 0.509 | 0.517* | 0.511 | 0.514 | 0.509 | 250 | 0.045 | 25 | |
| flab_expression_adams | spearman | dnn+svm | 0.386 | 0.401 | 0.401 | 0.416 | 0.412 | 0.424* | 0.416 | 0.421 | 500 | 0.025 | 25 | |
| flab_expression_koenig | spearman | svm+dnn | 0.661 | 0.707 | 0.739 | 0.761 | 0.767 | 0.766 | 0.770 | 0.771* | 2000 | 0.018 | 100 | |
| flab_thermo_tresanco | spearman | dnn+svm | 0.623 | 0.663 | 0.645 | 0.670 | 0.676 | 0.688* | 0.682 | 0.680 | 500 | 0.068 | none | |
| gb_human_enhancers_cohn | mcc | dnn+lightgbm | 0.441 | 0.474 | 0.481 | 0.497 | 0.501 | 0.510 | 0.515* | 0.509 | 1000 | 0.081 | none | |
| ginkgo_acsins | spearman | lightgbm+gp | 0.515 | 0.551 | 0.567 | 0.563 | 0.569 | 0.575* | 0.573 | 0.573 | 500 | 0.114 | none | |
| ginkgo_colloidal | spearman | dnn+svm | 0.277 | 0.339 | 0.330 | 0.369 | 0.392 | 0.408 | 0.414* | 0.407 | 1000 | 0.102 | 25 | |
| ginkgo_hic | spearman | gp+dnn | 0.375 | 0.468 | 0.507 | 0.542 | 0.545 | 0.571 | 0.582* | 0.580 | 1000 | 0.044 | 100 | |
| ginkgo_pr_cho | spearman | gp+dnn | 0.467 | 0.489 | 0.547* | 0.506 | 0.521 | 0.525 | 0.522 | 0.531 | 50 | 0.057 | 50 | |
| ginkgo_titer | spearman | dnn+rf | 0.346 | 0.367 | 0.399 | 0.406 | 0.429 | 0.420 | 0.437* | 0.436 | 1000 | 0.052 | 50 | |
| herg | auc | lightgbm+xgboost | 0.811 | 0.844 | 0.855 | 0.862 | 0.868* | 0.868 | 0.867 | 0.866 | 250 | 0.008 | 100 | |
| herg_ki | rmse | svm+dnn | 0.672 | 0.608 | 0.582 | 0.563 | 0.551 | 0.548 | 0.547* | 0.551 | 1000 | 0.044 | 50 | |
| humanness_therapeutic | auc | xgboost+rf | 0.966 | 0.967 | 0.968 | 0.970 | 0.968 | 0.972* | 0.969 | 0.969 | 500 | 0.000 | 500 (spread stored as 0: unreliable) | |
| kiba | ci | dnn+rf | 0.691 | 0.691 | 0.710 | 0.713 | 0.721 | 0.720 | 0.721 | 0.727* | 2000 | 0.035 | 25 | |
| lhasa_herg | auc | lightgbm+xgboost | 0.820 | 0.846 | 0.857 | 0.863 | 0.868 | 0.868 | 0.868 | 0.870* | 2000 | 0.012 | 100 | |
| logd | rmse | dnn+svm | 0.688 | 0.562 | 0.508 | 0.475 | 0.440 | 0.424 | 0.412 | 0.407* | 2000 | 0.014 | 1000 | |
| nt_H3K27me3 | mcc | dnn+lightgbm | 0.513 | 0.532 | 0.542 | 0.549 | 0.557 | 0.562 | 0.564 | 0.569* | 2000 | 0.021 | 100 | |
| nt_H3K4me3 | mcc | dnn+lightgbm | 0.620 | 0.642 | 0.654 | 0.659 | 0.665 | 0.667 | 0.672* | 0.671 | 1000 | 0.018 | 100 | |
| nt_H3K9ac | mcc | dnn+lightgbm | 0.448 | 0.468 | 0.484 | 0.505 | 0.521 | 0.523 | 0.527 | 0.530* | 2000 | 0.031 | 100 | |
| nt_H3K9me3 | mcc | dnn+lightgbm | 0.243 | 0.291 | 0.319 | 0.345 | 0.354 | 0.364 | 0.368* | 0.365 | 1000 | 0.027 | 100 | |
| nt_promoter_tata | mcc | lightgbm+svm | 0.615 | 0.636 | 0.663 | 0.693 | 0.700 | 0.704 | 0.712 | 0.717* | 2000 | 0.031 | 100 | |
| qm9 | rmse | dnn+lightgbm | 0.570 | 0.448 | 0.406 | 0.365 | 0.335 | 0.322 | 0.318* | 0.324 | 1000 | 0.021 | 250 | |
| suzuki | rmse | rf+lightgbm | 21.255 | 19.298 | 18.966* | 19.151 | 19.287 | 19.264 | 19.264 | 19.264 | 50 | 0.000 | 50 (spread stored as 0: unreliable) | |
| tcga_brca_pam50 | f1_weighted | dnn+svm | 0.844 | 0.901 | 0.899 | 0.902 | 0.903 | 0.909* | 0.906 | 0.903 | 500 | 0.061 | 25 | |
| tcga_luad_vs_lusc | auc | lightgbm+xgboost | 0.998 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | 1.000 | 1.000* | 2000 | 0.000 | 2000 (spread stored as 0: unreliable) | |
| tz_her2_binary | auc | xgboost+lightgbm | 0.784 | 0.815 | 0.849 | 0.837 | 0.816 | 0.860* | 0.848 | 0.852 | 500 | 0.056 | 25 | |
| tz_her2_multiclass | f1_weighted | svm+xgboost | 0.585 | 0.627 | 0.659 | 0.670 | 0.695* | 0.687 | 0.689 | 0.687 | 250 | 0.080 | 25 | |

## Step 2 (rule: lowest budget within one fold spread of the best; highest = the best budget itself)

Replaces the earlier Step 2 table's proposal. Per-budget scores are in the table above. One row is one dataset, two chosen models only. Lowest = lowest budget whose best score is within one fold spread of the overall best. Highest = the budget with the best score (the peak dictates). Cut budgets = every budget below lowest or above highest. DECISION blank until Adelaide fills it.

| dataset | metric | best budget | fold spread | lowest | highest | budgets cut | evidence for the upper cut | DECISION |
|---|---|---|---|---|---|---|---|---|
| abagym_Ang2 | spearman | 50 | 0.153 | 10 | 50 | 100/250/500/1000/2000 | upper cut rests on the peak only (no budget above it is worse than the best by more than the spread) | |
| abagym_EGFR | spearman | 1000 | 0.078 | 25 | 1000 | 10/2000 | upper cut rests on the peak only (no budget above it is worse than the best by more than the spread) | |
| abagym_VEGF | spearman | 10 | 0.089 | 10 | 10 | 25/50/100/250/500/1000/2000 | upper cut rests on the peak only (no budget above it is worse than the best by more than the spread) | |
| abcov_kd | spearman | 50 | 0.143 | 10 | 50 | 100/250/500/1000/2000 | upper cut rests on the peak only (no budget above it is worse than the best by more than the spread) | |
| adme_hlm | rmse | 500 | 0.005 | 500 | 500 | 10/25/50/100/250/1000/2000 | upper cut rests on the peak only (no budget above it is worse than the best by more than the spread) | |
| adme_mdr1 | rmse | 1000 | 0.016 | 250 | 1000 | 10/25/50/100/2000 | upper cut rests on the peak only (no budget above it is worse than the best by more than the spread) | |
| adme_rlm | rmse | 2000 | 0.014 | 250 | 2000 | 10/25/50/100 | no upper cut | |
| bh_doyle_plates | rmse | 100 | 2.579 | 10 | 100 | 250/500/1000/2000 | upper cut rests on the peak only (no budget above it is worse than the best by more than the spread) | |
| bindingdb_kd | ci | 2000 | 0.014 | 100 | 2000 | 10/25/50 | no upper cut | |
| buchwald_hartwig | rmse | 500 | 2.931 | 10 | 500 | 1000/2000 | upper cut rests on the peak only (no budget above it is worse than the best by more than the spread) | |
| caco2 | rmse | 2000 | 0.027 | 50 | 2000 | 10/25 | no upper cut | |
| cdr_flexibility | auc | 50 | 0.039 | 10 | 50 | 100/250/500/1000/2000 | upper cut rests on the peak only (no budget above it is worse than the best by more than the spread) | |
| ddg_combined | spearman | 250 | 0.104 | 100 | 250 | 10/25/50/500/1000/2000 | upper cut rests on the peak only (no budget above it is worse than the best by more than the spread) | |
| flab_binding_adams | auc | 100 | 0.039 | 10 | 100 | 250/500/1000/2000 | upper cut rests on the peak only (no budget above it is worse than the best by more than the spread) | |
| flab_binding_koenig | spearman | 250 | 0.045 | 25 | 250 | 10/500/1000/2000 | upper cut rests on the peak only (no budget above it is worse than the best by more than the spread) | |
| flab_expression_adams | spearman | 500 | 0.025 | 25 | 500 | 10/1000/2000 | upper cut rests on the peak only (no budget above it is worse than the best by more than the spread) | |
| flab_expression_koenig | spearman | 2000 | 0.018 | 100 | 2000 | 10/25/50 | no upper cut | |
| flab_thermo_tresanco | spearman | 500 | 0.068 | 10 | 500 | 1000/2000 | upper cut rests on the peak only (no budget above it is worse than the best by more than the spread) | |
| gb_human_enhancers_cohn | mcc | 1000 | 0.081 | 10 | 1000 | 2000 | upper cut rests on the peak only (no budget above it is worse than the best by more than the spread) | |
| ginkgo_acsins | spearman | 500 | 0.114 | 10 | 500 | 1000/2000 | upper cut rests on the peak only (no budget above it is worse than the best by more than the spread) | |
| ginkgo_colloidal | spearman | 1000 | 0.102 | 25 | 1000 | 10/2000 | upper cut rests on the peak only (no budget above it is worse than the best by more than the spread) | |
| ginkgo_hic | spearman | 1000 | 0.044 | 100 | 1000 | 10/25/50/2000 | upper cut rests on the peak only (no budget above it is worse than the best by more than the spread) | |
| ginkgo_pr_cho | spearman | 50 | 0.057 | 50 | 50 | 10/25/100/250/500/1000/2000 | upper cut rests on the peak only (no budget above it is worse than the best by more than the spread) | |
| ginkgo_titer | spearman | 1000 | 0.052 | 50 | 1000 | 10/25/2000 | upper cut rests on the peak only (no budget above it is worse than the best by more than the spread) | |
| herg | auc | 250 | 0.008 | 100 | 250 | 10/25/50/500/1000/2000 | upper cut rests on the peak only (no budget above it is worse than the best by more than the spread) | |
| herg_ki | rmse | 1000 | 0.044 | 50 | 1000 | 10/25/2000 | upper cut rests on the peak only (no budget above it is worse than the best by more than the spread) | |
| humanness_therapeutic | auc | 500 | 0.000 | hold | hold | - | spread stored as 0: no range proposed | |
| kiba | ci | 2000 | 0.035 | 25 | 2000 | 10 | no upper cut | |
| lhasa_herg | auc | 2000 | 0.012 | 100 | 2000 | 10/25/50 | no upper cut | |
| logd | rmse | 2000 | 0.014 | 1000 | 2000 | 10/25/50/100/250/500 | no upper cut | |
| nt_H3K27me3 | mcc | 2000 | 0.021 | 100 | 2000 | 10/25/50 | no upper cut | |
| nt_H3K4me3 | mcc | 1000 | 0.018 | 100 | 1000 | 10/25/50/2000 | upper cut rests on the peak only (no budget above it is worse than the best by more than the spread) | |
| nt_H3K9ac | mcc | 2000 | 0.031 | 100 | 2000 | 10/25/50 | no upper cut | |
| nt_H3K9me3 | mcc | 1000 | 0.027 | 100 | 1000 | 10/25/50/2000 | upper cut rests on the peak only (no budget above it is worse than the best by more than the spread) | |
| nt_promoter_tata | mcc | 2000 | 0.031 | 100 | 2000 | 10/25/50 | no upper cut | |
| qm9 | rmse | 1000 | 0.021 | 250 | 1000 | 10/25/50/100/2000 | upper cut rests on the peak only (no budget above it is worse than the best by more than the spread) | |
| suzuki | rmse | 50 | 0.000 | hold | hold | - | spread stored as 0: no range proposed | |
| tcga_brca_pam50 | f1_weighted | 500 | 0.061 | 25 | 500 | 10/1000/2000 | upper cut rests on the peak only (no budget above it is worse than the best by more than the spread) | |
| tcga_luad_vs_lusc | auc | 2000 | 0.000 | hold | hold | - | spread stored as 0: no range proposed | |
| tz_her2_binary | auc | 500 | 0.056 | 25 | 500 | 10/1000/2000 | upper cut rests on the peak only (no budget above it is worse than the best by more than the spread) | |
| tz_her2_multiclass | f1_weighted | 250 | 0.080 | 25 | 250 | 10/500/1000/2000 | upper cut rests on the peak only (no budget above it is worse than the best by more than the spread) | |

## Step 2 (v2): budgets from all stored configurations

One row is one dataset, using only its two chosen models (downstream and scorer), augmentation none, both stored searches pooled. Each cell is the average score of the 10 best configurations at that budget, in the dataset's own metric. Kept budgets = every budget whose cell is within 1% of the best cell. Lowest and highest are the ends of that set. DECISION blank until Adelaide fills it.

| dataset | metric | 10 | 25 | 50 | 100 | 250 | 500 | 1000 | 2000 | kept budgets | lowest | highest | DECISION |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abagym_Ang2 | spearman | 0.362 | 0.491* | 0.486 | 0.478 | 0.459 | 0.468 | 0.450 | 0.444 | 25/50 | 25 | 50 | |
| abagym_EGFR | spearman | 0.227 | 0.290 | 0.327 | 0.351 | 0.366 | 0.374* | 0.361 | 0.357 | 500 | 500 | 500 | |
| abagym_VEGF | spearman | 0.625* | 0.610 | 0.582 | 0.590 | 0.616 | 0.609 | 0.591 | 0.556 | 10 | 10 | 10 | |
| abcov_kd | spearman | 0.372 | 0.396 | 0.420 | 0.407 | 0.424 | 0.428 | 0.417 | 0.432* | 2000 | 2000 | 2000 | |
| adme_hlm | rmse | 0.507 | 0.481 | 0.461 | 0.445 | 0.427 | 0.422* | 0.423 | 0.423 | 500/1000/2000 | 500 | 2000 | |
| adme_mdr1 | rmse | 0.523 | 0.492 | 0.471 | 0.454 | 0.441 | 0.435 | 0.432* | 0.433 | 500/1000/2000 | 500 | 2000 | |
| adme_rlm | rmse | 0.582 | 0.553 | 0.529 | 0.511 | 0.490 | 0.482 | 0.479 | 0.478* | 500/1000/2000 | 500 | 2000 | |
| bh_doyle_plates | rmse | 17.274 | 15.879 | 15.900 | 15.537 | 15.439* | 15.641 | 15.804 | 15.799 | 100/250 | 100 | 250 | |
| bindingdb_kd | ci | 0.710 | 0.732 | 0.740 | 0.744 | 0.748 | 0.749 | 0.750 | 0.752* | 250/500/1000/2000 | 250 | 2000 | |
| buchwald_hartwig | rmse | 16.171 | 15.193 | 14.512 | 14.287 | 13.918 | 13.701* | 13.768 | 13.747 | 500/1000/2000 | 500 | 2000 | |
| caco2 | rmse | 0.336 | 0.303 | 0.289 | 0.284 | 0.277 | 0.274 | 0.273 | 0.272* | 500/1000/2000 | 500 | 2000 | |
| cdr_flexibility | auc | 0.883 | 0.884* | 0.881 | 0.880 | 0.877 | 0.875 | 0.874 | 0.873 | 10/25/50/100/250 | 10 | 250 | |
| ddg_combined | spearman | 0.248 | 0.277 | 0.302 | 0.369 | 0.385* | 0.359 | 0.336 | 0.342 | 250 | 250 | 250 | |
| flab_binding_adams | auc | 0.617 | 0.623 | 0.625 | 0.628* | 0.627 | 0.624 | 0.625 | 0.623 | 25/50/100/250/500/1000/2000 | 25 | 2000 | |
| flab_binding_koenig | spearman | 0.431 | 0.475 | 0.498 | 0.503 | 0.511* | 0.509 | 0.507 | 0.506 | 250/500/1000 | 250 | 1000 | |
| flab_expression_adams | spearman | 0.385 | 0.394 | 0.394 | 0.407 | 0.411 | 0.421* | 0.411 | 0.414 | 500 | 500 | 500 | |
| flab_expression_koenig | spearman | 0.655 | 0.705 | 0.738 | 0.760 | 0.766 | 0.766 | 0.769 | 0.770* | 250/500/1000/2000 | 250 | 2000 | |
| flab_thermo_tresanco | spearman | 0.612 | 0.652 | 0.643 | 0.664 | 0.671 | 0.681* | 0.674 | 0.668 | 500 | 500 | 500 | |
| gb_human_enhancers_cohn | mcc | 0.441 | 0.472 | 0.480 | 0.491 | 0.499 | 0.507* | 0.505 | 0.503 | 500/1000/2000 | 500 | 2000 | |
| ginkgo_acsins | spearman | 0.500 | 0.531 | 0.547 | 0.552 | 0.559* | 0.556 | 0.557 | 0.556 | 250/500/1000/2000 | 250 | 2000 | |
| ginkgo_colloidal | spearman | 0.256 | 0.302 | 0.318 | 0.343 | 0.382 | 0.395 | 0.398 | 0.400* | 1000/2000 | 1000 | 2000 | |
| ginkgo_hic | spearman | 0.359 | 0.430 | 0.500 | 0.528 | 0.544 | 0.565 | 0.573 | 0.574* | 1000/2000 | 1000 | 2000 | |
| ginkgo_pr_cho | spearman | 0.456 | 0.463 | 0.478 | 0.492 | 0.507 | 0.513 | 0.511 | 0.514* | 500/1000/2000 | 500 | 2000 | |
| ginkgo_titer | spearman | 0.317 | 0.356 | 0.375 | 0.395 | 0.418* | 0.414 | 0.413 | 0.410 | 250 | 250 | 250 | |
| herg | auc | 0.805 | 0.841 | 0.852 | 0.859 | 0.865 | 0.865* | 0.865 | 0.864 | 100/250/500/1000/2000 | 100 | 2000 | |
| herg_ki | rmse | 0.675 | 0.614 | 0.585 | 0.566 | 0.553 | 0.550* | 0.550 | 0.553 | 250/500/1000/2000 | 250 | 2000 | |
| humanness_therapeutic | auc | 0.958 | 0.965 | 0.968 | 0.968 | 0.968 | 0.969* | 0.967 | 0.968 | 25/50/100/250/500/1000/2000 | 25 | 2000 | |
| kiba | ci | 0.691 | 0.691 | 0.704 | 0.709 | 0.716 | 0.715 | 0.716 | 0.721* | 250/500/1000/2000 | 250 | 2000 | |
| lhasa_herg | auc | 0.813 | 0.843 | 0.854 | 0.861 | 0.866 | 0.866 | 0.868 | 0.868* | 100/250/500/1000/2000 | 100 | 2000 | |
| logd | rmse | 0.714 | 0.574 | 0.514 | 0.478 | 0.442 | 0.427 | 0.422 | 0.422* | 1000/2000 | 1000 | 2000 | |
| nt_H3K27me3 | mcc | 0.513 | 0.531 | 0.538 | 0.547 | 0.555 | 0.559* | 0.558 | 0.557 | 250/500/1000/2000 | 250 | 2000 | |
| nt_H3K4me3 | mcc | 0.620 | 0.641 | 0.652 | 0.657 | 0.665 | 0.666 | 0.666 | 0.667* | 250/500/1000/2000 | 250 | 2000 | |
| nt_H3K9ac | mcc | 0.446 | 0.466 | 0.483 | 0.504 | 0.518 | 0.520 | 0.522 | 0.524* | 500/1000/2000 | 500 | 2000 | |
| nt_H3K9me3 | mcc | 0.242 | 0.290 | 0.316 | 0.343 | 0.351 | 0.357 | 0.360* | 0.360 | 500/1000/2000 | 500 | 2000 | |
| nt_promoter_tata | mcc | 0.610 | 0.634 | 0.661 | 0.684 | 0.698 | 0.703 | 0.706* | 0.706 | 500/1000/2000 | 500 | 2000 | |
| qm9 | rmse | 0.595 | 0.453 | 0.409 | 0.370 | 0.338 | 0.325* | 0.325 | 0.330 | 500/1000 | 500 | 1000 | |
| suzuki | rmse | 21.378 | 19.424 | 19.045* | 19.218 | 19.343 | 19.309 | 19.334 | 19.347 | 50/100 | 50 | 100 | |
| tcga_brca_pam50 | f1_weighted | 0.839 | 0.891 | 0.898 | 0.899 | 0.902 | 0.906* | 0.903 | 0.899 | 50/100/250/500/1000/2000 | 50 | 2000 | |
| tcga_luad_vs_lusc | auc | 0.997 | 0.998 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999* | 10/25/50/100/250/500/1000/2000 | 10 | 2000 | |
| tz_her2_binary | auc | 0.772 | 0.802 | 0.823* | 0.806 | 0.808 | 0.818 | 0.808 | 0.804 | 50/500 | 50 | 500 | |
| tz_her2_multiclass | f1_weighted | 0.580 | 0.624 | 0.653 | 0.666 | 0.689* | 0.685 | 0.678 | 0.676 | 250/500 | 250 | 500 | |

## Step 2 (v3, current proposal): budget range

Rule: band 3% for datasets under about 1,000 rows (list from BILL_OF_MATERIALS.md, dated), 1% otherwise (the three reaction datasets and datasets with no row count are treated as 1%, unverified). Band = budgets whose average of the 10 best configurations is within the band of the best budget's average. Range = lowest to highest of that band, widened to at least 3 consecutive budgets by adding the neighbouring budget with the better average. DECISION blank until Adelaide fills it.

| dataset | band | lowest | highest | budgets in range | budgets cut | note | DECISION |
|---|---|---|---|---|---|---|---|
| abagym_Ang2 | 3% | 25 | 100 | 3 | 10/250/500/1000/2000 |  | **25-100** (2026-10-03, Adelaide) |
| abagym_EGFR | 3% | 250 | 1000 | 3 | 10/25/50/100/2000 | (band alone: 250-500) | **250-2000** (2026-10-03, revised after removing the scorer limit; was 250-1000) |
| abagym_VEGF | 3% | 10 | 500 | 6 | 1000/2000 |  | **10-500** (2026-10-03, Adelaide) |
| abcov_kd | 3% | 50 | 2000 | 6 | 10/25 |  | **50-2000** (2026-10-03, Adelaide) |
| adme_hlm | 1% | 500 | 2000 | 3 | 10/25/50/100/250 |  | **500-2000** (2026-10-03, Adelaide) |
| adme_mdr1 | 1% | 500 | 2000 | 3 | 10/25/50/100/250 |  | **500-2000** (2026-10-03, Adelaide) |
| adme_rlm | 1% | 500 | 2000 | 3 | 10/25/50/100/250 |  | **500-2000** (2026-10-03, Adelaide) |
| bh_doyle_plates | 1% | 100 | 500 | 3 | 10/25/50/1000/2000 | (band alone: 100-250) | **100-500** (2026-10-03, Adelaide) |
| bindingdb_kd | 1% | 250 | 2000 | 4 | 10/25/50/100 |  | **250-2000** (2026-10-03, Adelaide) |
| buchwald_hartwig | 1% | 500 | 2000 | 3 | 10/25/50/100/250 |  | **500-2000** (2026-10-03, Adelaide) |
| caco2 | 1% | 500 | 2000 | 3 | 10/25/50/100/250 |  | **500-2000** (2026-10-03, Adelaide) |
| cdr_flexibility | 1% | 10 | 250 | 5 | 500/1000/2000 |  | **10-100** (2026-10-03, revised after removing the scorer limit; was 10-250) |
| ddg_combined | 3% | 100 | 500 | 3 | 10/25/50/1000/2000 | (band alone: 250-250) | **100-500** (2026-10-03, Adelaide) |
| flab_binding_adams | 1% | 25 | 2000 | 7 | 10 |  | **25-2000** (2026-10-03, Adelaide) |
| flab_binding_koenig | 1% | 250 | 1000 | 3 | 10/25/50/100/2000 |  | **250-1000** (2026-10-03, Adelaide) |
| flab_expression_adams | 1% | 250 | 1000 | 3 | 10/25/50/100/2000 | (band alone: 500-500) | **500-2000** (2026-10-03, revised after removing the scorer limit; was 250-1000) |
| flab_expression_koenig | 1% | 250 | 2000 | 4 | 10/25/50/100 |  | **250-2000** (2026-10-03, Adelaide) |
| flab_thermo_tresanco | 3% | 100 | 2000 | 5 | 10/25/50 |  | **50-2000** (2026-10-03, revised after removing the scorer limit; was 100-2000) |
| gb_human_enhancers_cohn | 1% | 500 | 2000 | 3 | 10/25/50/100/250 |  | **500-2000** (2026-10-03, Adelaide) |
| ginkgo_acsins | 3% | 50 | 2000 | 6 | 10/25 |  | **50-2000** (2026-10-03, Adelaide) |
| ginkgo_colloidal | 3% | 500 | 2000 | 3 | 10/25/50/100/250 |  | **500-2000** (2026-10-03, Adelaide) |
| ginkgo_hic | 3% | 500 | 2000 | 3 | 10/25/50/100/250 |  | **250-2000** (2026-10-03, revised after removing the scorer limit; was 500-2000) |
| ginkgo_pr_cho | 3% | 250 | 2000 | 4 | 10/25/50/100 |  | **250-2000** (2026-10-03, Adelaide) |
| ginkgo_titer | 3% | 250 | 2000 | 4 | 10/25/50/100 |  | **250-2000** (2026-10-03, Adelaide) |
| herg | 1% | 100 | 2000 | 5 | 10/25/50 |  | **100-2000** (2026-10-03, Adelaide) |
| herg_ki | 1% | 250 | 2000 | 4 | 10/25/50/100 |  | **250-2000** (2026-10-03, Adelaide) |
| humanness_therapeutic | 3% | 10 | 2000 | 8 | none |  | **10-2000** (2026-10-03, Adelaide) |
| kiba | 1% | 250 | 2000 | 4 | 10/25/50/100 |  | **250-2000** (2026-10-03, Adelaide) |
| lhasa_herg | 1% | 100 | 2000 | 5 | 10/25/50 |  | **100-2000** (2026-10-03, Adelaide) |
| logd | 1% | 500 | 2000 | 3 | 10/25/50/100/250 | (band alone: 1000-2000) | **500-2000** (2026-10-03, Adelaide) |
| nt_H3K27me3 | 1% | 250 | 2000 | 4 | 10/25/50/100 |  | **250-2000** (2026-10-03, Adelaide) |
| nt_H3K4me3 | 1% | 250 | 2000 | 4 | 10/25/50/100 |  | **250-2000** (2026-10-03, Adelaide) |
| nt_H3K9ac | 1% | 500 | 2000 | 3 | 10/25/50/100/250 |  | **250-2000** (2026-10-03, revised after removing the scorer limit; was 500-2000) |
| nt_H3K9me3 | 1% | 500 | 2000 | 3 | 10/25/50/100/250 |  | **500-2000** (2026-10-03, Adelaide) |
| nt_promoter_tata | 1% | 500 | 2000 | 3 | 10/25/50/100/250 |  | **500-2000** (2026-10-03, Adelaide) |
| qm9 | 1% | 500 | 2000 | 3 | 10/25/50/100/250 | (band alone: 500-1000) | **500-2000** (2026-10-03, Adelaide) |
| suzuki | 1% | 50 | 250 | 3 | 10/25/500/1000/2000 | (band alone: 50-100) | **50-250** (2026-10-03, Adelaide) |
| tcga_brca_pam50 | 3% | 25 | 2000 | 7 | 10 |  | **25-2000** (2026-10-03, Adelaide) |
| tcga_luad_vs_lusc | 3% | 10 | 2000 | 8 | none |  | **10-2000** (2026-10-03, Adelaide) |
| tz_her2_binary | 1% | 50 | 500 | 4 | 10/25/1000/2000 |  | **100-500** (2026-10-03, revised after removing the scorer limit; was 50-500) |
| tz_her2_multiclass | 1% | 250 | 1000 | 3 | 10/25/50/100/2000 | (band alone: 250-500) | **250-1000** (2026-10-03, Adelaide) |

## Step 3: importance scoring

One row is one dataset. Only configurations that survived the decided pairs and budget ranges. Each cell is the average score of the 10 best configurations using that importance method, in the dataset's own metric (rmse lower is better). Keep = within the same band as budgets (3% under about 1,000 rows, 1% otherwise) of the best method. DECISION blank until Adelaide fills it.

| dataset | metric | deeplift | lightgbm_gain | treeshap | keep | cut | DECISION |
|---|---|---|---|---|---|---|---|
| abagym_Ang2 | spearman | 0.3704 | 0.4409 | 0.5030* | treeshap | deeplift/lightgbm_gain | |
| abagym_EGFR | spearman | 0.3500 | 0.3806* | 0.3461 | lightgbm_gain | treeshap/deeplift | |
| abagym_VEGF | spearman | 0.6099 | 0.5969 | 0.6463* | treeshap | deeplift/lightgbm_gain | |
| abcov_kd | spearman | 0.4201 | 0.4345 | 0.4436* | lightgbm_gain/treeshap | deeplift | |
| adme_hlm | rmse | 0.4247 | 0.4223* | 0.4225 | deeplift/lightgbm_gain/treeshap | none | |
| adme_mdr1 | rmse | 0.4348 | 0.4319* | 0.4342 | deeplift/lightgbm_gain/treeshap | none | |
| adme_rlm | rmse | 0.4796 | 0.4801 | 0.4781* | deeplift/lightgbm_gain/treeshap | none | |
| bh_doyle_plates | rmse | 15.4162* | 15.7082 | 15.6630 | deeplift | lightgbm_gain/treeshap | |
| bindingdb_kd | ci | 0.7488 | 0.7504 | 0.7520* | deeplift/lightgbm_gain/treeshap | none | |
| buchwald_hartwig | rmse | 13.7112 | 14.1942 | 13.6945* | deeplift/treeshap | lightgbm_gain | |
| caco2 | rmse | 0.2739 | 0.2729 | 0.2727* | deeplift/lightgbm_gain/treeshap | none | |
| cdr_flexibility | auc | 0.8763 | 0.8730 | 0.8854* | treeshap | lightgbm_gain/deeplift | |
| ddg_combined | spearman | 0.2284 | 0.3501 | 0.3952* | treeshap | deeplift/lightgbm_gain | |
| flab_binding_adams | auc | 0.6188 | 0.6256 | 0.6288* | lightgbm_gain/treeshap | deeplift | |
| flab_binding_koenig | spearman | 0.4958 | 0.5081 | 0.5122* | lightgbm_gain/treeshap | deeplift | |
| flab_expression_adams | spearman | - | 0.4125 | 0.4204* | treeshap | lightgbm_gain | |
| flab_expression_koenig | spearman | 0.7628 | 0.7692 | 0.7700* | deeplift/lightgbm_gain/treeshap | none | |
| flab_thermo_tresanco | spearman | 0.6516 | 0.6775 | 0.6802* | lightgbm_gain/treeshap | deeplift | |
| gb_human_enhancers_cohn | mcc | - | 0.5098* | - | lightgbm_gain | none | |
| ginkgo_acsins | spearman | 0.5471 | 0.5664* | 0.5607 | lightgbm_gain/treeshap | deeplift | |
| ginkgo_colloidal | spearman | 0.4052* | 0.3886 | 0.3829 | deeplift | lightgbm_gain/treeshap | |
| ginkgo_hic | spearman | 0.5712 | 0.5748* | 0.5605 | deeplift/lightgbm_gain/treeshap | none | |
| ginkgo_pr_cho | spearman | 0.5144 | 0.5006 | 0.5191* | deeplift/treeshap | lightgbm_gain | |
| ginkgo_titer | spearman | 0.4188 | 0.3921 | 0.4236* | treeshap/deeplift | lightgbm_gain | |
| herg | auc | - | 0.8671* | - | lightgbm_gain | none | |
| herg_ki | rmse | 0.5551 | 0.5486* | 0.5521 | lightgbm_gain/treeshap | deeplift | |
| humanness_therapeutic | auc | 0.9685 | 0.9683 | 0.9688* | deeplift/lightgbm_gain/treeshap | none | |
| kiba | ci | 0.7161 | 0.7208* | 0.7187 | deeplift/lightgbm_gain/treeshap | none | |
| lhasa_herg | auc | 0.8684* | 0.8676 | 0.8665 | deeplift/lightgbm_gain/treeshap | none | |
| logd | rmse | 0.4354 | 0.4164* | 0.4456 | lightgbm_gain | deeplift/treeshap | |
| nt_H3K27me3 | mcc | - | 0.5616* | - | lightgbm_gain | none | |
| nt_H3K4me3 | mcc | - | 0.6688* | - | lightgbm_gain | none | |
| nt_H3K9ac | mcc | - | 0.5258* | - | lightgbm_gain | none | |
| nt_H3K9me3 | mcc | - | 0.3655* | - | lightgbm_gain | none | |
| nt_promoter_tata | mcc | - | 0.7102* | - | lightgbm_gain | none | |
| qm9 | rmse | - | 0.3228* | - | lightgbm_gain | none | |
| suzuki | rmse | 19.2234 | 19.0567* | 19.3050 | deeplift/lightgbm_gain | treeshap | |
| tcga_brca_pam50 | f1_weighted | 0.9048 | 0.9056* | 0.9043 | deeplift/lightgbm_gain/treeshap | none | |
| tcga_luad_vs_lusc | auc | 0.9996* | 0.9992 | 0.9988 | deeplift/lightgbm_gain/treeshap | none | |
| tz_her2_binary | auc | - | 0.8363* | - | lightgbm_gain | none | |
| tz_her2_multiclass | f1_weighted | - | 0.6917* | - | lightgbm_gain | none | |

## Step 3 (2026-10-07): one importance method everywhere

Adelaide 2026-10-07: "if one of the importance methods is a standout across the board, let's just keep using only that". On the 3 October record (full.txt, `general`), keeping only lightgbm_gain puts no dataset with a measured fold spread more than its spread behind the best configuration; the two flagged, humanness_therapeutic (0.0030 auc) and tcga_luad_vs_lusc (0.0005 auc), have a recorded spread of 0.0000. Keeping only treeshap would cost logd 0.0224 rmse and suzuki 0.1722 rmse beyond their spreads; only deeplift would cost ddg_combined, logd and suzuki. By "best most often" the count is treeshap 12, lightgbm_gain 10, deeplift 6 of 28 datasets. DECISION: **lightgbm_gain only, every dataset, both searches** (2026-10-07, Adelaide). Searches already stored keep their other methods' configurations; experiments read lightgbm_gain's.

## Step 3 (v2, cautious): importance scoring

One row is one dataset, surviving configurations only (chosen pair, recorded budget range). Cells: average score of the 10 best configurations using that method (dataset's own metric, * best), then in brackets how many of the 30 best configurations overall use that method (even split would be 10 each). A method is cut only if BOTH hold: its average is worse than the best method's by more than the band (3% under about 1,000 rows, 1% otherwise) AND it has 0 or 1 of the 30 best configurations. Two methods are cut only if both meet that test. DECISION blank until Adelaide fills it.

| dataset | deeplift | lightgbm_gain | treeshap | cut | DECISION |
|---|---|---|---|---|---|
| abagym_Ang2 | 0.370 (0) | 0.441 (0) | 0.503* (30) | deeplift/lightgbm_gain | **cut deeplift** (2026-10-03, revised after removing the scorer limit; was cut deeplift/lightgbm_gain) |
| abagym_EGFR | 0.350 (2) | 0.381* (28) | 0.346 (0) | treeshap | **cut deeplift** (2026-10-03, revised after removing the scorer limit; was cut treeshap) |
| abagym_VEGF | 0.610 (9) | 0.597 (4) | 0.646* (17) | none | **keep all stored methods** (2026-10-03, Adelaide) |
| abcov_kd | 0.420 (4) | 0.434 (8) | 0.444* (18) | none | **keep all stored methods** (2026-10-03, Adelaide) |
| adme_hlm | 0.425 (0) | 0.422* (18) | 0.423 (12) | none | **keep all stored methods** (2026-10-03, Adelaide) |
| adme_mdr1 | 0.435 (4) | 0.432* (19) | 0.434 (7) | none | **keep all stored methods** (2026-10-03, Adelaide) |
| adme_rlm | 0.480 (9) | 0.480 (7) | 0.478* (14) | none | **keep all stored methods** (2026-10-03, Adelaide) |
| bh_doyle_plates | 15.416* (22) | 15.708 (2) | 15.663 (6) | none | **cut treeshap** (2026-10-03, revised after removing the scorer limit; was keep all stored methods) |
| bindingdb_kd | 0.749 (4) | 0.750 (10) | 0.752* (16) | none | **keep all stored methods** (2026-10-03, Adelaide) |
| buchwald_hartwig | 13.711 (13) | 14.194 (0) | 13.694* (17) | lightgbm_gain | **cut lightgbm_gain** (2026-10-03, Adelaide) |
| caco2 | 0.274 (4) | 0.273 (17) | 0.273* (9) | none | **keep all stored methods** (2026-10-03, Adelaide) |
| cdr_flexibility | 0.876 (2) | 0.873 (0) | 0.885* (28) | lightgbm_gain | **cut lightgbm_gain** (2026-10-03, Adelaide) |
| ddg_combined | 0.228 (0) | 0.350 (2) | 0.395* (28) | deeplift | **cut deeplift** (2026-10-03, Adelaide) |
| flab_binding_adams | 0.619 (0) | 0.626 (3) | 0.629* (27) | deeplift | **cut deeplift** (2026-10-03, Adelaide) |
| flab_binding_koenig | 0.496 (0) | 0.508 (7) | 0.512* (23) | deeplift | **cut deeplift** (2026-10-03, Adelaide) |
| flab_expression_adams | - | 0.412 (6) | 0.420* (24) | none | **keep all stored methods** (2026-10-03, Adelaide) |
| flab_expression_koenig | 0.763 (0) | 0.769 (17) | 0.770* (13) | none | **keep all stored methods** (2026-10-03, Adelaide) |
| flab_thermo_tresanco | 0.652 (0) | 0.678 (14) | 0.680* (16) | deeplift | **cut deeplift** (2026-10-03, Adelaide) |
| gb_human_enhancers_cohn | only lightgbm_gain stored | | | none | **lightgbm_gain only** (fast tier as run; do not extend to other methods) (2026-10-03, Adelaide) |
| ginkgo_acsins | 0.547 (1) | 0.566* (20) | 0.561 (9) | deeplift | **keep all stored methods** (2026-10-03, revised after removing the scorer limit; was cut deeplift) |
| ginkgo_colloidal | 0.405* (28) | 0.389 (0) | 0.383 (2) | lightgbm_gain | **keep all stored methods** (2026-10-03, revised after removing the scorer limit; was cut lightgbm_gain) |
| ginkgo_hic | 0.571 (7) | 0.575* (21) | 0.560 (2) | none | **keep all stored methods** (2026-10-03, Adelaide) |
| ginkgo_pr_cho | 0.514 (13) | 0.501 (0) | 0.519* (17) | lightgbm_gain | **cut lightgbm_gain** (2026-10-03, Adelaide) |
| ginkgo_titer | 0.419 (10) | 0.392 (0) | 0.424* (20) | lightgbm_gain | **cut lightgbm_gain** (2026-10-03, Adelaide) |
| herg | only lightgbm_gain stored | | | none | **lightgbm_gain only** (fast tier as run; do not extend to other methods) (2026-10-03, Adelaide) |
| herg_ki | 0.555 (0) | 0.549* (26) | 0.552 (4) | deeplift | **cut deeplift** (2026-10-03, Adelaide) |
| humanness_therapeutic | 0.969 (7) | 0.968 (12) | 0.969* (11) | none | **keep all stored methods** (2026-10-03, Adelaide) |
| kiba | 0.716 (5) | 0.721* (12) | 0.719 (13) | none | **keep all stored methods** (2026-10-03, Adelaide) |
| lhasa_herg | 0.868* (14) | 0.868 (10) | 0.866 (6) | none | **keep all stored methods** (2026-10-03, Adelaide) |
| logd | 0.435 (2) | 0.416* (28) | 0.446 (0) | treeshap | **cut deeplift/treeshap** (2026-10-03, revised after removing the scorer limit; was cut treeshap) |
| nt_H3K27me3 | only lightgbm_gain stored | | | none | **lightgbm_gain only** (fast tier as run; do not extend to other methods) (2026-10-03, Adelaide) |
| nt_H3K4me3 | only lightgbm_gain stored | | | none | **lightgbm_gain only** (fast tier as run; do not extend to other methods) (2026-10-03, Adelaide) |
| nt_H3K9ac | only lightgbm_gain stored | | | none | **lightgbm_gain only** (fast tier as run; do not extend to other methods) (2026-10-03, Adelaide) |
| nt_H3K9me3 | only lightgbm_gain stored | | | none | **lightgbm_gain only** (fast tier as run; do not extend to other methods) (2026-10-03, Adelaide) |
| nt_promoter_tata | only lightgbm_gain stored | | | none | **lightgbm_gain only** (fast tier as run; do not extend to other methods) (2026-10-03, Adelaide) |
| qm9 | only lightgbm_gain stored | | | none | **lightgbm_gain only** (fast tier as run; do not extend to other methods) (2026-10-03, Adelaide) |
| suzuki | 19.223 (12) | 19.057* (14) | 19.305 (4) | none | **keep all stored methods** (2026-10-03, Adelaide) |
| tcga_brca_pam50 | 0.905 (10) | 0.906* (11) | 0.904 (9) | none | **keep all stored methods** (2026-10-03, Adelaide) |
| tcga_luad_vs_lusc | 1.000* (15) | 0.999 (13) | 0.999 (2) | none | **keep all stored methods** (2026-10-03, Adelaide) |
| tz_her2_binary | only lightgbm_gain stored | | | none | **lightgbm_gain only** (fast tier as run; do not extend to other methods) (2026-10-03, Adelaide) |
| tz_her2_multiclass | only lightgbm_gain stored | | | none | **lightgbm_gain only** (fast tier as run; do not extend to other methods) (2026-10-03, Adelaide) |


## Fast-tier datasets: the stored grid is the cut

The ten datasets below were searched on the fast tier. Its grid is the decided search space for them: do not run the methods it left out. Allocation methods, representation counts and importance methods actually stored (augmentation none, `general`):

| dataset | allocation methods | representation counts | importance methods |
|---|---|---|---|
| gb_human_enhancers_cohn | performance_weighted, power_2, rank_exp_d0.3 | all, 3 | lightgbm_gain |
| herg | performance_weighted, power_2, rank_exp_d0.3 | all, 3 | lightgbm_gain |
| nt_H3K27me3 | performance_weighted, power_2, rank_exp_d0.3 | all, 3 | lightgbm_gain |
| nt_H3K4me3 | performance_weighted, power_2, rank_exp_d0.3 | all, 3 | lightgbm_gain |
| nt_H3K9ac | performance_weighted, power_2, rank_exp_d0.3 | all, 3 | lightgbm_gain |
| nt_H3K9me3 | performance_weighted, power_2, rank_exp_d0.3 | all, 3 | lightgbm_gain |
| nt_promoter_tata | performance_weighted, power_2, rank_exp_d0.3 | all, 3 | lightgbm_gain |
| qm9 | performance_weighted, power_2, rank_exp_d0.3 | all, 3 | lightgbm_gain |
| tz_her2_binary | performance_weighted, power_2, rank_exp_d0.3 | all, 3 | lightgbm_gain |
| tz_her2_multiclass | performance_weighted, power_2, rank_exp_d0.3 | all, 3 | lightgbm_gain |

## Step 4: allocation methods

One row is one dataset, surviving configurations only (chosen pair, recorded budget range, importance methods not cut). Cells: average score of the 10 best configurations using that allocation method (dataset's own metric, * best), then in brackets how many of the 30 best configurations overall use it (an even split of five would be 6 each). Cut only if BOTH hold: its average is worse than the best method's by more than the band (3% under about 1,000 rows, 1% otherwise) AND it has 0 or 1 of the 30 best. Fast-tier datasets keep their stored three methods and are not widened. DECISION blank until Adelaide fills it.

| dataset | softmax_T1 | power_2 | performance_weighted | rank_exp_d0.3 | wglobal_power2 | cut | DECISION |
|---|---|---|---|---|---|---|---|
| abagym_Ang2 | 0.467 (6) | 0.469 (6) | 0.462 (6) | 0.491* (8) | 0.463 (4) | none | **keep all stored methods** (2026-10-03, Adelaide) |
| abagym_EGFR | 0.358 (6) | 0.364 (3) | 0.361 (5) | 0.370* (10) | 0.359 (6) | none | **keep all stored methods** (2026-10-03, Adelaide) |
| abagym_VEGF | 0.600 (3) | 0.600 (4) | 0.595 (4) | 0.601 (4) | 0.641* (15) | none | **keep all stored methods** (2026-10-03, Adelaide) |
| abcov_kd | 0.426 (9) | 0.421 (4) | 0.428 (7) | 0.444* (9) | 0.406 (1) | wglobal_power2 | **keep all stored methods** (2026-10-03, revised after removing the scorer limit; was cut wglobal_power2) |
| adme_hlm | 0.423 (9) | 0.423* (5) | 0.423 (9) | 0.423 (4) | 0.424 (3) | none | **keep all stored methods** (2026-10-03, Adelaide) |
| adme_mdr1 | 0.434 (5) | 0.433* (10) | 0.434 (7) | 0.435 (5) | 0.435 (3) | none | **keep all stored methods** (2026-10-03, Adelaide) |
| adme_rlm | 0.479* (10) | 0.480 (7) | 0.479 (8) | 0.480 (4) | 0.484 (1) | wglobal_power2 | **keep all stored methods** (2026-10-03, revised after removing the scorer limit; was cut wglobal_power2) |
| bh_doyle_plates | 15.569* (4) | 15.615 (8) | 15.635 (4) | 15.625 (6) | 15.578 (8) | none | **keep all stored methods** (2026-10-03, Adelaide) |
| bindingdb_kd | 0.746 (0) | 0.749 (7) | 0.748 (4) | 0.751 (11) | 0.752* (8) | none | **keep all stored methods** (2026-10-03, Adelaide) |
| buchwald_hartwig | 13.758* (8) | 13.827 (4) | 13.811 (5) | 13.817 (6) | 13.797 (7) | none | **keep all stored methods** (2026-10-03, Adelaide) |
| caco2 | 0.273* (10) | 0.273 (8) | 0.273 (9) | 0.275 (3) | 0.276 (0) | wglobal_power2 | **cut wglobal_power2** (2026-10-03, Adelaide) |
| cdr_flexibility | 0.882 (6) | 0.882* (8) | 0.882 (6) | 0.881 (5) | 0.881 (5) | none | **keep all stored methods** (2026-10-03, Adelaide) |
| ddg_combined | 0.382* (16) | 0.284 (0) | 0.290 (0) | 0.374 (14) | 0.312 (0) | performance_weighted/power_2/wglobal_power2 | **cut power_2/wglobal_power2** (2026-10-03, revised after removing the scorer limit; was cut performance_weighted/power_2/wglobal_power2) |
| flab_binding_adams | 0.627 (7) | 0.626 (7) | 0.628* (9) | 0.625 (5) | 0.624 (2) | none | **keep all stored methods** (2026-10-03, Adelaide) |
| flab_binding_koenig | 0.508 (6) | 0.508 (6) | 0.510* (7) | 0.507 (6) | 0.507 (5) | none | **keep all stored methods** (2026-10-03, Adelaide) |
| flab_expression_adams | 0.412 (5) | 0.415* (8) | 0.410 (6) | 0.411 (6) | 0.413 (5) | none | **keep all stored methods** (2026-10-03, Adelaide) |
| flab_expression_koenig | 0.769 (8) | 0.769 (8) | 0.769* (10) | 0.766 (4) | - | none | **keep all stored methods** (2026-10-03, Adelaide) |
| flab_thermo_tresanco | 0.663 (1) | 0.666 (2) | 0.665 (2) | 0.681* (12) | 0.676 (13) | none | **keep all stored methods** (2026-10-03, Adelaide) |
| gb_human_enhancers_cohn (fast tier, as run) | - | 0.504 (9) | 0.506 (9) | 0.506* (12) | - | none | **stored three methods only** (fast tier as run; do not extend) (2026-10-03, Adelaide) |
| ginkgo_acsins | 0.555 (5) | 0.559 (8) | 0.558 (7) | 0.562* (9) | 0.545 (1) | wglobal_power2 | **cut wglobal_power2** (2026-10-03, Adelaide) |
| ginkgo_colloidal | 0.396 (8) | 0.394 (6) | 0.398* (8) | 0.389 (5) | 0.385 (3) | none | **keep all stored methods** (2026-10-03, Adelaide) |
| ginkgo_hic | 0.569 (7) | 0.571* (8) | 0.571 (8) | 0.569 (6) | 0.561 (1) | none | **keep all stored methods** (2026-10-03, Adelaide) |
| ginkgo_pr_cho | 0.513* (8) | 0.501 (2) | 0.506 (3) | 0.510 (6) | 0.513 (11) | none | **keep all stored methods** (2026-10-03, Adelaide) |
| ginkgo_titer | 0.410 (4) | 0.411 (6) | 0.416 (8) | 0.408 (3) | 0.416* (9) | none | **keep all stored methods** (2026-10-03, Adelaide) |
| herg (fast tier, as run) | - | 0.865 (10) | 0.865 (9) | 0.866* (11) | - | none | **stored three methods only** (fast tier as run; do not extend) (2026-10-03, Adelaide) |
| herg_ki | 0.549* (10) | 0.551 (9) | 0.551 (9) | 0.553 (2) | 0.557 (0) | wglobal_power2 | **cut wglobal_power2** (2026-10-03, Adelaide) |
| humanness_therapeutic | 0.969* (18) | 0.968 (3) | 0.968 (4) | 0.967 (3) | 0.967 (2) | none | **keep all stored methods** (2026-10-03, Adelaide) |
| kiba | 0.715 (5) | 0.720* (16) | 0.716 (4) | 0.717 (5) | - | none | **keep all stored methods** (2026-10-03, Adelaide) |
| lhasa_herg | 0.867 (6) | 0.867 (4) | 0.867 (4) | 0.868* (10) | 0.867 (6) | none | **keep all stored methods** (2026-10-03, Adelaide) |
| logd | - | 0.424* (10) | 0.424 (12) | 0.424 (8) | - | none | **keep all stored methods** (2026-10-03, Adelaide) |
| nt_H3K27me3 (fast tier, as run) | - | 0.558 (6) | 0.557 (5) | 0.560* (19) | - | none | **stored three methods only** (fast tier as run; do not extend) (2026-10-03, Adelaide) |
| nt_H3K4me3 (fast tier, as run) | - | 0.666 (8) | 0.666 (11) | 0.667* (11) | - | none | **stored three methods only** (fast tier as run; do not extend) (2026-10-03, Adelaide) |
| nt_H3K9ac (fast tier, as run) | - | 0.521 (9) | 0.521 (10) | 0.524* (11) | - | none | **stored three methods only** (fast tier as run; do not extend) (2026-10-03, Adelaide) |
| nt_H3K9me3 (fast tier, as run) | - | 0.359* (10) | 0.359 (9) | 0.359 (11) | - | none | **stored three methods only** (fast tier as run; do not extend) (2026-10-03, Adelaide) |
| nt_promoter_tata (fast tier, as run) | - | 0.706 (9) | 0.707* (11) | 0.703 (10) | - | none | **stored three methods only** (fast tier as run; do not extend) (2026-10-03, Adelaide) |
| qm9 (fast tier, as run) | - | 0.327 (9) | 0.330 (6) | 0.324* (15) | - | none | **stored three methods only** (fast tier as run; do not extend) (2026-10-03, Adelaide) |
| suzuki | 19.089* (16) | 21.006 (0) | 21.006 (0) | 19.154 (14) | 20.987 (0) | performance_weighted/power_2/wglobal_power2 | **HELD**: power_2, performance_weighted, wglobal_power2 show 21.006 / 21.006 / 20.99 rmse; check why before cutting (2026-10-03, Adelaide) |
| tcga_brca_pam50 | 0.903 (5) | 0.904 (7) | 0.903 (5) | 0.903 (3) | 0.904* (10) | none | **keep all stored methods** (2026-10-03, Adelaide) |
| tcga_luad_vs_lusc | 1.000* (30) | 0.997 (0) | 0.997 (0) | 0.997 (0) | 0.997 (0) | none | **keep all stored methods** (2026-10-03, Adelaide) |
| tz_her2_binary (fast tier, as run) | - | 0.820 (15) | 0.813 (7) | 0.823* (8) | - | none | **stored three methods only** (fast tier as run; do not extend) (2026-10-03, Adelaide) |
| tz_her2_multiclass (fast tier, as run) | - | 0.685 (10) | 0.685 (10) | 0.686* (10) | - | none | **stored three methods only** (fast tier as run; do not extend) (2026-10-03, Adelaide) |


## Step 5: internal evaluator (the model that scores a configuration during the search)

One row is one dataset, surviving configurations only (chosen pair, budget range, importance cuts, allocation cuts; suzuki's held allocation methods left in). Each evaluator cell: average score of the 10 best configurations scored by that evaluator (dataset's own metric, * best), then in brackets how many of the 30 best configurations use it. Cut only if BOTH hold: its average is worse than the best evaluator's by more than the band (3% under about 1,000 rows, 1% otherwise) AND it has 0 or 1 of the 30 best. DECISION blank until Adelaide fills it.

| dataset | evaluator 1 | evaluator 2 | cut | DECISION |
|---|---|---|---|---|
| abagym_Ang2 | gp: 0.379 (0) | svm: 0.503* (30) | gp | |
| abagym_EGFR | svm: 0.381* (22) | dnn: 0.363 (8) | none | |
| abagym_VEGF | gp: 0.647* (29) | svm: 0.595 (1) | svm | |
| abcov_kd | lightgbm: 0.444* (23) | xgboost: 0.433 (7) | none | |
| adme_hlm | svm: 0.422* (26) | dnn: 0.423 (4) | none | |
| adme_mdr1 | svm: 0.433 (11) | dnn: 0.432* (19) | none | |
| adme_rlm | svm: 0.478* (19) | dnn: 0.479 (11) | none | |
| bh_doyle_plates | rf: 15.532 (10) | dnn: 15.439* (20) | none | |
| bindingdb_kd | rf: 0.752* (23) | dnn: 0.750 (7) | none | |
| buchwald_hartwig | rf: 13.671* (19) | lightgbm: 13.737 (11) | none | |
| caco2 | dnn: 0.272* (27) | svm: 0.275 (3) | none | |
| cdr_flexibility | xgboost: 0.879 (3) | lightgbm: 0.885* (27) | none | |
| ddg_combined | svm: 0.348 (4) | dnn: 0.395* (26) | none | |
| flab_binding_adams | lightgbm: 0.629* (27) | xgboost: 0.625 (3) | none | |
| flab_binding_koenig | rf: 0.511* (18) | svm: 0.510 (12) | none | |
| flab_expression_adams | dnn: 0.420* (23) | svm: 0.413 (7) | none | |
| flab_expression_koenig | svm: 0.769 (16) | dnn: 0.770* (14) | none | |
| flab_thermo_tresanco | dnn: 0.683* (30) | svm: 0.662 (0) | svm | |
| gb_human_enhancers_cohn | dnn: 0.507 (15) | lightgbm: 0.509* (15) | none | |
| ginkgo_acsins | lightgbm: 0.563 (13) | gp: 0.565* (17) | none | |
| ginkgo_colloidal | dnn: 0.384 (2) | svm: 0.406* (28) | none | |
| ginkgo_hic | gp: 0.563 (4) | dnn: 0.576* (26) | none | |
| ginkgo_pr_cho | gp: 0.518* (13) | dnn: 0.516 (17) | none | |
| ginkgo_titer | dnn: 0.414 (12) | rf: 0.426* (18) | none | |
| herg | lightgbm: 0.867* (29) | xgboost: 0.862 (1) | none | |
| herg_ki | svm: 0.549* (14) | dnn: 0.550 (16) | none | |
| humanness_therapeutic | xgboost: 0.968 (4) | rf: 0.969* (26) | none | |
| kiba | dnn: 0.720* (17) | rf: 0.720 (13) | none | |
| lhasa_herg | lightgbm: 0.869* (28) | xgboost: 0.866 (2) | none | |
| logd | dnn: 0.416* (27) | svm: 0.431 (3) | none | |
| nt_H3K27me3 | dnn: 0.559 (17) | lightgbm: 0.561* (13) | none | |
| nt_H3K4me3 | dnn: 0.668* (17) | lightgbm: 0.667 (13) | none | |
| nt_H3K9ac | dnn: 0.523 (14) | lightgbm: 0.524* (16) | none | |
| nt_H3K9me3 | dnn: 0.363* (13) | lightgbm: 0.361 (17) | none | |
| nt_promoter_tata | lightgbm: 0.709* (15) | svm: 0.706 (15) | none | |
| qm9 | dnn: 0.326 (9) | lightgbm: 0.324* (21) | none | |
| suzuki | rf: 19.186 (10) | lightgbm: 19.066* (20) | none | |
| tcga_brca_pam50 | dnn: 0.906* (22) | svm: 0.904 (8) | none | |
| tcga_luad_vs_lusc | lightgbm: 0.999* (15) | xgboost: 0.999 (15) | none | |
| tz_her2_binary | xgboost: 0.824 (10) | lightgbm: 0.827* (20) | none | |
| tz_her2_multiclass | svm: 0.692* (27) | xgboost: 0.676 (3) | none | |

## Step 5 (v2): internal evaluator

Replaces Step 5 above, which wrongly limited the scorer to the two chosen models. One row is one dataset, in the `general` search, downstream model = the chosen pair, surviving configurations only (budget range, importance cuts, allocation cuts as recorded). rf, lightgbm and dnn were run for every configuration. svm, xgboost, gp and knn appear only where that model was itself the downstream model (matched evaluator), so they have fewer rows. Cell: average score of the 10 best configurations scored by that evaluator (dataset's own metric, * best), then in brackets how many of the 30 best use it. Cut only if its average is worse than the best's by more than the band (3% under about 1,000 rows, 1% otherwise) AND it has 0 or 1 of the 30 best. Evaluators with under 10 configurations are not judged. DECISION blank until Adelaide fills it.

| dataset | rf | lightgbm | dnn | svm | xgboost | gp | knn | cut | DECISION |
|---|---|---|---|---|---|---|---|---|---|
| abagym_Ang2 | 0.462 (5) | 0.444 (3) | 0.452 (4) | 0.492* (18) | - | 0.379 (0) | - | gp | |
| abagym_EGFR | 0.340 (1) | 0.366 (11) | 0.358 (5) | 0.375* (13) | - | - | - | rf | |
| abagym_VEGF | 0.580 (1) | 0.586 (1) | 0.580 (0) | 0.587 (0) | - | 0.647* (28) | - | rf/lightgbm/dnn/svm | |
| abcov_kd | 0.469* (13) | 0.440 (3) | 0.460 (11) | - | 0.433 (3) | - | - | none | |
| adme_hlm | 0.422 (9) | 0.422* (11) | 0.423 (1) | 0.422 (9) | - | - | - | none | |
| adme_mdr1 | 0.433 (6) | 0.433 (5) | 0.432* (12) | 0.433 (7) | - | - | - | none | |
| adme_rlm | 0.478* (10) | 0.478 (9) | 0.479 (2) | 0.478 (9) | - | - | - | none | |
| bh_doyle_plates | 15.632 (5) | 15.296* (13) | 15.520 (12) | - | - | - | - | none | |
| bindingdb_kd | 0.752* (15) | 0.751 (10) | 0.750 (5) | - | - | - | - | none | |
| buchwald_hartwig | 13.713* (18) | 13.737 (12) | 14.367 (0) | - | - | - | - | dnn | |
| caco2 | 0.273 (6) | 0.272 (14) | 0.272* (10) | - | - | - | - | none | |
| cdr_flexibility | 0.889* (19) | 0.884 (3) | 0.888 (7) | - | 0.879 (1) | - | - | xgboost | |
| ddg_combined | 0.379* (13) | 0.320 (3) | 0.378 (12) | 0.309 (2) | - | - | - | none | |
| flab_binding_adams | 0.625 (3) | 0.628* (18) | 0.626 (6) | - | 0.625 (3) | - | - | none | |
| flab_binding_koenig | 0.510 (8) | 0.510* (12) | 0.510 (8) | 0.506 (2) | - | - | - | none | |
| flab_expression_adams | 0.423* (14) | 0.420 (12) | 0.417 (4) | 0.323 (0) | - | - | - | svm | |
| flab_expression_koenig | 0.770* (10) | 0.770 (8) | 0.770 (6) | 0.769 (6) | - | - | - | none | |
| flab_thermo_tresanco | 0.676 (7) | 0.681* (9) | 0.680 (14) | 0.658 (0) | - | - | - | svm | |
| gb_human_enhancers_cohn | 0.507* (10) | 0.507 (9) | 0.506 (11) | - | - | - | - | none | |
| ginkgo_acsins | 0.561 (7) | 0.563 (6) | 0.564 (8) | - | - | 0.565* (9) | - | none | |
| ginkgo_colloidal | 0.401 (9) | 0.402 (8) | 0.384 (1) | 0.405* (12) | - | - | - | dnn | |
| ginkgo_hic | 0.576 (5) | 0.580* (18) | 0.576 (6) | - | - | 0.563 (1) | - | none | |
| ginkgo_pr_cho | 0.523* (14) | 0.513 (4) | 0.515 (5) | - | - | 0.518 (7) | - | none | |
| ginkgo_titer | 0.424* (17) | 0.396 (1) | 0.412 (12) | - | - | - | - | lightgbm | |
| herg | 0.866 (11) | 0.867* (15) | 0.864 (3) | - | 0.862 (1) | - | - | none | |
| herg_ki | 0.549 (8) | 0.549* (10) | 0.550 (4) | 0.550 (8) | - | - | - | none | |
| humanness_therapeutic | 0.969 (4) | 0.969 (5) | 0.970* (18) | - | 0.968 (3) | - | - | none | |
| kiba | 0.720 (8) | 0.720 (11) | 0.720* (11) | - | - | - | - | none | |
| lhasa_herg | 0.869* (18) | 0.869 (12) | 0.865 (0) | - | 0.866 (0) | - | - | none | |
| logd | 0.414* (12) | 0.415 (10) | 0.416 (8) | 0.432 (0) | - | - | - | svm | |
| nt_H3K27me3 | 0.560 (13) | 0.560* (8) | 0.558 (9) | - | - | - | - | none | |
| nt_H3K4me3 | 0.668* (11) | 0.667 (8) | 0.668 (11) | - | - | - | - | none | |
| nt_H3K9ac | 0.521 (8) | 0.524* (12) | 0.523 (10) | - | - | - | - | none | |
| nt_H3K9me3 | 0.361 (10) | 0.360 (11) | 0.362* (9) | - | - | - | - | none | |
| nt_promoter_tata | 0.709 (7) | 0.709 (9) | 0.710* (10) | 0.702 (4) | - | - | - | none | |
| qm9 | 0.327 (7) | 0.324* (16) | 0.328 (7) | - | - | - | - | none | |
| suzuki | 19.273 (6) | 19.163* (11) | 19.173 (13) | - | - | - | - | none | |
| tcga_brca_pam50 | 0.898 (0) | 0.906* (9) | 0.905 (19) | 0.900 (2) | - | - | - | none | |
| tcga_luad_vs_lusc | 1.000* (16) | 0.999 (8) | 0.998 (0) | - | 0.999 (6) | - | - | none | |
| tz_her2_binary | 0.827* (12) | 0.818 (4) | 0.823 (7) | - | 0.824 (7) | - | - | none | |
| tz_her2_multiclass | 0.687 (9) | 0.689* (7) | 0.683 (4) | 0.688 (9) | 0.676 (1) | - | - | xgboost | |

## Step 5 (v3): internal evaluator, with match as its own evaluator

Supersedes Step 5 and Step 5 (v2) above. Evaluators are rf, lightgbm, dnn and match (match = the downstream model's own type). A row counts for match whenever its evaluator equals the downstream model; if that model is rf, lightgbm or dnn, the same row also counts for that named evaluator. One row is one dataset, `general` search, downstream model = chosen pair, surviving configurations (budget range, importance cuts, allocation cuts as recorded). Cell: average score of the 10 best configurations (dataset's own metric, * best), then how many of the 30 best use that evaluator. Cut only if its average is worse than the best's by more than the band (3% under about 1,000 rows, 1% otherwise) AND it has 0 or 1 of the 30 best. DECISION blank until Adelaide fills it.

| dataset | pair | rf | lightgbm | dnn | match | cut | DECISION |
|---|---|---|---|---|---|---|---|
| abagym_Ang2 | gp+svm | 0.462 (5) | 0.444 (3) | 0.452 (4) | 0.492* (18) | none | **keep all four evaluators** (2026-10-03, Adelaide) |
| abagym_EGFR | svm+dnn | 0.340 (1) | 0.366 (11) | 0.358 (5) | 0.375* (13) | rf | **cut rf** (2026-10-03, Adelaide) |
| abagym_VEGF | gp+svm | 0.580 (1) | 0.586 (1) | 0.580 (0) | 0.647* (28) | rf/lightgbm/dnn | **cut rf/lightgbm/dnn** (2026-10-03, Adelaide) |
| abcov_kd | lightgbm+xgboost | 0.469* (13) | 0.440 (3) | 0.460 (11) | 0.450 (6) | none | **keep all four evaluators** (2026-10-03, Adelaide) |
| adme_hlm | svm+dnn | 0.422 (9) | 0.422* (11) | 0.423 (1) | 0.422 (9) | none | **keep all four evaluators** (2026-10-03, Adelaide) |
| adme_mdr1 | svm+dnn | 0.433 (6) | 0.433 (5) | 0.432* (12) | 0.433 (7) | none | **keep all four evaluators** (2026-10-03, Adelaide) |
| adme_rlm | svm+dnn | 0.478* (10) | 0.478 (9) | 0.479 (2) | 0.478 (9) | none | **keep all four evaluators** (2026-10-03, Adelaide) |
| bh_doyle_plates | rf+dnn | 15.632 (5) | 15.296* (13) | 15.520 (12) | 15.705 (3) | none | **keep all four evaluators** (2026-10-03, Adelaide) |
| bindingdb_kd | rf+dnn | 0.752* (15) | 0.751 (10) | 0.750 (5) | 0.752* (15) | none | **keep all four evaluators** (2026-10-03, Adelaide) |
| buchwald_hartwig | rf+lightgbm | 13.713* (18) | 13.737 (12) | 14.367 (0) | 13.713* (18) | dnn | **cut dnn** (2026-10-03, Adelaide) |
| caco2 | dnn+svm | 0.273 (6) | 0.272 (14) | 0.272* (10) | 0.274 (3) | none | **keep all four evaluators** (2026-10-03, Adelaide) |
| cdr_flexibility | xgboost+lightgbm | 0.889* (19) | 0.884 (3) | 0.888 (7) | 0.883 (4) | none | **keep all four evaluators** (2026-10-03, Adelaide) |
| ddg_combined | svm+dnn | 0.379* (13) | 0.320 (3) | 0.378 (12) | 0.352 (3) | none | **keep all four evaluators** (2026-10-03, Adelaide) |
| flab_binding_adams | lightgbm+xgboost | 0.625 (3) | 0.628* (18) | 0.626 (6) | 0.628* (20) | none | **keep all four evaluators** (2026-10-03, Adelaide) |
| flab_binding_koenig | rf+svm | 0.510 (8) | 0.510* (12) | 0.510 (8) | 0.510 (10) | none | **keep all four evaluators** (2026-10-03, Adelaide) |
| flab_expression_adams | dnn+svm | 0.423* (14) | 0.420 (12) | 0.417 (4) | 0.417 (4) | none | **keep all four evaluators** (2026-10-03, Adelaide) |
| flab_expression_koenig | svm+dnn | 0.770* (10) | 0.770 (8) | 0.770 (6) | 0.769 (6) | none | **keep all four evaluators** (2026-10-03, Adelaide) |
| flab_thermo_tresanco | dnn+svm | 0.676 (7) | 0.681* (9) | 0.680 (14) | 0.680 (10) | none | **keep all four evaluators** (2026-10-03, Adelaide) |
| gb_human_enhancers_cohn | dnn+lightgbm | 0.507* (10) | 0.507 (9) | 0.506 (11) | 0.506 (9) | none | **keep all four evaluators** (2026-10-03, Adelaide) |
| ginkgo_acsins | lightgbm+gp | 0.561 (7) | 0.563 (6) | 0.564 (8) | 0.570* (15) | none | **keep all four evaluators** (2026-10-03, Adelaide) |
| ginkgo_colloidal | dnn+svm | 0.401 (9) | 0.402 (8) | 0.384 (1) | 0.405* (12) | dnn | **cut dnn** (2026-10-03, Adelaide) |
| ginkgo_hic | gp+dnn | 0.576 (5) | 0.580* (18) | 0.576 (6) | 0.566 (2) | none | **keep all four evaluators** (2026-10-03, Adelaide) |
| ginkgo_pr_cho | gp+dnn | 0.523* (14) | 0.513 (4) | 0.515 (5) | 0.520 (10) | none | **keep all four evaluators** (2026-10-03, Adelaide) |
| ginkgo_titer | dnn+rf | 0.424* (17) | 0.396 (1) | 0.412 (12) | 0.411 (11) | lightgbm | **cut lightgbm** (2026-10-03, Adelaide) |
| herg | lightgbm+xgboost | 0.866 (11) | 0.867* (15) | 0.864 (3) | 0.866 (12) | none | **keep all four evaluators** (2026-10-03, Adelaide) |
| herg_ki | svm+dnn | 0.549 (8) | 0.549* (10) | 0.550 (4) | 0.550 (8) | none | **keep all four evaluators** (2026-10-03, Adelaide) |
| humanness_therapeutic | xgboost+rf | 0.969 (4) | 0.969 (5) | 0.970* (18) | 0.969 (7) | none | **keep all four evaluators** (2026-10-03, Adelaide) |
| kiba | dnn+rf | 0.720 (8) | 0.720 (11) | 0.720* (11) | 0.720 (10) | none | **keep all four evaluators** (2026-10-03, Adelaide) |
| lhasa_herg | lightgbm+xgboost | 0.869* (18) | 0.869 (12) | 0.865 (0) | 0.869 (12) | none | **keep all four evaluators** (2026-10-03, Adelaide) |
| logd | dnn+svm | 0.414* (12) | 0.415 (10) | 0.416 (8) | 0.416 (8) | none | **keep all four evaluators** (2026-10-03, Adelaide) |
| nt_H3K27me3 | dnn+lightgbm | 0.560 (13) | 0.560* (8) | 0.558 (9) | 0.558 (9) | none | **keep all four evaluators** (2026-10-03, Adelaide) |
| nt_H3K4me3 | dnn+lightgbm | 0.668* (11) | 0.667 (8) | 0.668 (11) | 0.668 (12) | none | **keep all four evaluators** (2026-10-03, Adelaide) |
| nt_H3K9ac | dnn+lightgbm | 0.521 (8) | 0.524* (12) | 0.523 (10) | 0.523 (10) | none | **keep all four evaluators** (2026-10-03, Adelaide) |
| nt_H3K9me3 | dnn+lightgbm | 0.361 (10) | 0.360 (11) | 0.362* (9) | 0.362* (9) | none | **keep all four evaluators** (2026-10-03, Adelaide) |
| nt_promoter_tata | lightgbm+svm | 0.709 (7) | 0.709 (9) | 0.710* (10) | 0.710 (10) | none | **keep all four evaluators** (2026-10-03, Adelaide) |
| qm9 | dnn+lightgbm | 0.327 (7) | 0.324* (16) | 0.328 (7) | 0.328 (7) | none | **keep all four evaluators** (2026-10-03, Adelaide) |
| suzuki | rf+lightgbm | 19.273 (6) | 19.163* (11) | 19.173 (13) | 19.253 (7) | none | **keep all four evaluators** (2026-10-03, Adelaide) |
| tcga_brca_pam50 | dnn+svm | 0.898 (0) | 0.906* (9) | 0.905 (19) | 0.905 (19) | none | **keep all four evaluators** (2026-10-03, Adelaide) |
| tcga_luad_vs_lusc | lightgbm+xgboost | 1.000* (16) | 0.999 (8) | 0.998 (0) | 1.000 (10) | none | **keep all four evaluators** (2026-10-03, Adelaide) |
| tz_her2_binary | xgboost+lightgbm | 0.827* (12) | 0.818 (4) | 0.823 (7) | 0.825 (7) | none | **keep all four evaluators** (2026-10-03, Adelaide) |
| tz_her2_multiclass | svm+xgboost | 0.687 (9) | 0.689* (7) | 0.683 (4) | 0.689 (10) | none | **keep all four evaluators** (2026-10-03, Adelaide) |


## Step 6: representation count, general search

One row is one dataset. 'all' = all 10 representations. Cut by the same rule as importance (worse than the best by more than the band AND 0 or 1 of the 30 best), on surviving configurations after every earlier cut. Fast-tier datasets store only 'all' and 3. Checked independently by a second pass: no disagreements. DECISION blank until Adelaide fills it.

| dataset | proposed cut | DECISION |
|---|---|---|
| abagym_Ang2 | 2 | **cut 2** (2026-10-03, Adelaide) |
| abagym_EGFR | none | **keep all stored rep counts** (2026-10-03, Adelaide) |
| abagym_VEGF | none | **keep all stored rep counts** (2026-10-03, Adelaide) |
| abcov_kd | none | **keep all stored rep counts** (2026-10-03, Adelaide) |
| adme_hlm | 2/3 | **cut 2/3** (2026-10-03, Adelaide) |
| adme_mdr1 | 2/3/5 | **cut 2/3/5** (2026-10-03, Adelaide) |
| adme_rlm | 2/3/5 | **cut 2/3/5** (2026-10-03, Adelaide) |
| bh_doyle_plates | none | **keep all stored rep counts** (2026-10-03, Adelaide) |
| bindingdb_kd | 3/5 | **cut 3/5** (2026-10-03, Adelaide) |
| buchwald_hartwig | 3/5/all | **cut 3/5/all** (2026-10-03, Adelaide) |
| caco2 | 2/3/5 | **cut 2/3/5** (2026-10-03, Adelaide) |
| cdr_flexibility | none | **keep all stored rep counts** (2026-10-03, Adelaide) |
| ddg_combined | 2 | **cut 2** (2026-10-03, Adelaide) |
| flab_binding_adams | none | **2, 3, 5; 'all' cut** (2026-10-07, Adelaide; was keep all 2026-10-03. The best count is not 'all' on this dataset and the search folds were taking 10 h and more) |
| flab_binding_koenig | 3/5 | **cut 3/5** (2026-10-03, Adelaide) |
| flab_expression_adams | none | **2, 3, 5; 'all' cut** (2026-10-07, Adelaide; was keep all 2026-10-03. The best count is not 'all' on this dataset and the search folds were taking 10 h and more) |
| flab_expression_koenig | 2/3/5 | **cut 2/3/5** (2026-10-03, Adelaide) |
| flab_thermo_tresanco | none | **keep all stored rep counts** (2026-10-03, Adelaide) |
| gb_human_enhancers_cohn | 3 | **cut 3** (2026-10-03, Adelaide) |
| ginkgo_acsins | 3 | **cut 3** (2026-10-03, Adelaide) |
| ginkgo_colloidal | 2 | **cut 2** (2026-10-03, Adelaide) |
| ginkgo_hic | 2/3 | **cut 2/3** (2026-10-03, Adelaide) |
| ginkgo_pr_cho | 3 | **cut 3** (2026-10-03, Adelaide) |
| ginkgo_titer | none | **keep all stored rep counts** (2026-10-03, Adelaide) |
| herg_ki | 2/3/5 | **cut 2/3/5** (2026-10-03, Adelaide) |
| herg | none | **keep all stored rep counts** (2026-10-03, Adelaide) |
| humanness_therapeutic | none | **keep all stored rep counts** (2026-10-03, Adelaide) |
| kiba | 2/3/5 | **cut 2/3/5** (2026-10-03, Adelaide) |
| lhasa_herg | none | **keep all stored rep counts** (2026-10-03, Adelaide) |
| logd | none | **keep all stored rep counts** (2026-10-03, Adelaide) |
| nt_H3K27me3 | none | **keep all stored rep counts** (2026-10-03, Adelaide) |
| nt_H3K4me3 | none | **keep all stored rep counts** (2026-10-03, Adelaide) |
| nt_H3K9ac | none | **keep all stored rep counts** (2026-10-03, Adelaide) |
| nt_H3K9me3 | none | **keep all stored rep counts** (2026-10-03, Adelaide) |
| nt_promoter_tata | 3 | **cut 3** (2026-10-03, Adelaide) |
| qm9 | none | **keep all stored rep counts** (2026-10-03, Adelaide) |
| suzuki | none | **2, 3, 5; 'all' cut** (2026-10-07, Adelaide; was keep all 2026-10-03. The best count is not 'all' on this dataset and the search folds were taking 10 h and more) |
| tcga_brca_pam50 | none | **keep all stored rep counts** (2026-10-03, Adelaide) |
| tcga_luad_vs_lusc | none | **2, 3, 5; 'all' cut** (2026-10-07, Adelaide; was keep all 2026-10-03. The best count is not 'all' on this dataset and the search folds were taking 10 h and more) |
| tz_her2_binary | none | **keep all stored rep counts** (2026-10-03, Adelaide) |
| tz_her2_multiclass | none | **keep all stored rep counts** (2026-10-03, Adelaide) |

## Step 7: distance search (general_distance), all five axes, same rules, downstream = decided pair

One row is one dataset. Budgets are 10 to 500 only. Evaluators here are knn, rf, dnn, lightgbm, svm (and a model's own type where it was downstream). Note: knn is in this search by design (the experiments that read it measure nearest-neighbour prediction), so a knn evaluator cut is flagged, not assumed. Checked independently by a second pass: no disagreements on any axis. Datasets with no distance rows: bindingdb_kd, kiba, lhasa_herg. DECISION blank until Adelaide fills it.

| dataset | budget range | importance cut | allocation cut | evaluator cut | rep count cut | DECISION |
|---|---|---|---|---|---|---|
| abagym_Ang2 | 25-100 | deeplift | none | lightgbm | 2 | **as proposed** (2026-10-03, Adelaide) |
| abagym_EGFR | 100-500 | deeplift/treeshap | none | rf | none | **as proposed** (2026-10-03, Adelaide) |
| abagym_VEGF | 100-500 | none | wglobal_power2 | none | 2/3/5 | **as proposed** (2026-10-03, Adelaide) |
| abcov_kd | 25-500 | none | none | none | none | **as proposed** (2026-10-03, Adelaide) |
| adme_hlm | 100-500 | deeplift | none | none | 2/3 | **as proposed** (2026-10-03, Adelaide) |
| adme_mdr1 | 100-500 | none | none | none | 2/3 | **as proposed** (2026-10-03, Adelaide) |
| adme_rlm | 100-500 | none | none | none | 2/3 | **as proposed** (2026-10-03, Adelaide) |
| bh_doyle_plates | 100-500 | none | none | svm | none | **as proposed** (2026-10-03, Adelaide) |
| buchwald_hartwig | 100-500 | none | none | dnn/knn/svm | 3/5/all | **as proposed, including the knn evaluator cut** (evaluator is not part of the stored-search identity; no stored-read experiment filters on a knn evaluator) (2026-10-03, Adelaide) |
| caco2 | 100-500 | treeshap | none | none | 2/3 | **as proposed** (2026-10-03, Adelaide) |
| cdr_flexibility | 10-100 | lightgbm_gain | none | svm | none | **as proposed** (2026-10-03, Adelaide) |
| ddg_combined | 100-500 | deeplift | performance_weighted/power_2/wglobal_power2 | lightgbm | 2/3 | **as proposed** (2026-10-03, Adelaide) |
| flab_binding_adams | 25-500 | deeplift | none | none | none | **as proposed** (2026-10-03, Adelaide) |
| flab_binding_koenig | 100-500 | none | none | none | 3/5 | **as proposed** (2026-10-03, Adelaide) |
| flab_expression_adams | 100-500 | none | none | none | none | **as proposed** (2026-10-03, Adelaide) |
| flab_expression_koenig | 100-500 | none | none | none | 3/5 | **as proposed** (2026-10-03, Adelaide) |
| flab_thermo_tresanco | 50-500 | deeplift | none | none | none | **as proposed** (2026-10-03, Adelaide) |
| gb_human_enhancers_cohn | 100-500 | none | none | none | none | **as proposed** (2026-10-03, Adelaide) |
| ginkgo_acsins | 50-500 | none | none | rf | none | **as proposed** (2026-10-03, Adelaide) |
| ginkgo_colloidal | 100-500 | none | none | knn | none | **as proposed, including the knn evaluator cut** (evaluator is not part of the stored-search identity; no stored-read experiment filters on a knn evaluator) (2026-10-03, Adelaide) |
| ginkgo_hic | 100-500 | deeplift | none | knn | 2 | **as proposed, including the knn evaluator cut** (evaluator is not part of the stored-search identity; no stored-read experiment filters on a knn evaluator) (2026-10-03, Adelaide) |
| ginkgo_pr_cho | 100-500 | lightgbm_gain | none | none | none | **as proposed** (2026-10-03, Adelaide) |
| ginkgo_titer | 100-500 | none | none | lightgbm/svm | none | **as proposed** (2026-10-03, Adelaide) |
| herg_ki | 100-500 | deeplift | wglobal_power2 | none | 2/3/5 | **as proposed** (2026-10-03, Adelaide) |
| herg | 100-500 | none | none | none | none | **as proposed** (2026-10-03, Adelaide) |
| humanness_therapeutic | 10-500 | none | none | none | none | **as proposed** (2026-10-03, Adelaide) |
| logd | 100-500 | none | none | none | none | **as proposed** (2026-10-03, Adelaide) |
| nt_H3K27me3 | 100-500 | none | none | none | none | **as proposed** (2026-10-03, Adelaide) |
| nt_H3K4me3 | 100-500 | none | none | none | none | **as proposed** (2026-10-03, Adelaide) |
| nt_H3K9ac | 100-500 | none | none | none | none | **as proposed** (2026-10-03, Adelaide) |
| nt_H3K9me3 | 100-500 | none | none | none | none | **as proposed** (2026-10-03, Adelaide) |
| nt_promoter_tata | 100-500 | none | none | none | 3 | **as proposed** (2026-10-03, Adelaide) |
| qm9 | 100-500 | none | none | svm | none | **as proposed** (2026-10-03, Adelaide) |
| suzuki | 50-250 | none | performance_weighted/power_2/wglobal_power2 | none | none | **as proposed except allocation cut HELD (21.006 rmse values unchecked)** (2026-10-03, Adelaide) |
| tcga_brca_pam50 | 25-500 | none | none | none | none | **as proposed** (2026-10-03, Adelaide) |
| tcga_luad_vs_lusc | 10-500 | none | none | none | none | **as proposed** (2026-10-03, Adelaide) |
| tz_her2_binary | 100-500 | none | none | none | none | **as proposed** (2026-10-03, Adelaide) |
| tz_her2_multiclass | 100-500 | none | none | none | none | **as proposed** (2026-10-03, Adelaide) |

## Step 8: augmentation (capped / uncapped)

Decision (Adelaide 2026-10-03): run augmented searches only on the datasets where augmentation clearly helped; cut it everywhere else. 'Helped' = better than none by more than the band (3% small datasets, 1% otherwise) on the surviving configurations; only ginkgo_hic and ginkgo_colloidal qualified, both on the distance search. OPEN CHECK before this is final: experiments that read augmented searches on other datasets (for example noise `base_k_aug_uncapped`) may break; the experiment search is pending. OPEN: whether the kept augmentation applies to the general search as well as the distance search for these two datasets.

| dataset | augmentation |
|---|---|
| abagym_Ang2 | **cut (no augmented searches)** |
| abagym_EGFR | **cut (no augmented searches)** |
| abagym_VEGF | **cut (no augmented searches)** |
| abcov_kd | **cut (no augmented searches)** |
| adme_hlm | **cut (no augmented searches)** |
| adme_mdr1 | **cut (no augmented searches)** |
| adme_rlm | **cut (no augmented searches)** |
| bh_doyle_plates | **cut (no augmented searches)** |
| bindingdb_kd | **cut (no augmented searches)** |
| buchwald_hartwig | **cut (no augmented searches)** |
| caco2 | **cut (no augmented searches)** |
| cdr_flexibility | **cut (no augmented searches)** |
| ddg_combined | **cut (no augmented searches)** |
| flab_binding_adams | **cut (no augmented searches)** |
| flab_binding_koenig | **cut (no augmented searches)** |
| flab_expression_adams | **cut (no augmented searches)** |
| flab_expression_koenig | **cut (no augmented searches)** |
| flab_thermo_tresanco | **cut (no augmented searches)** |
| gb_human_enhancers_cohn | **cut (no augmented searches)** |
| ginkgo_acsins | **cut (no augmented searches)** |
| ginkgo_colloidal | **keep uncapped (Adelaide 2026-10-03; on the distance search .400 -> .412 spearman)** |
| ginkgo_hic | **keep uncapped only (Adelaide 2026-10-03; on the distance search .573 -> .594 spearman)** |
| ginkgo_pr_cho | **cut (no augmented searches)** |
| ginkgo_titer | **cut (no augmented searches)** |
| herg | **cut (no augmented searches)** |
| herg_ki | **cut (no augmented searches)** |
| humanness_therapeutic | **cut (no augmented searches)** |
| kiba | **cut (no augmented searches)** |
| lhasa_herg | **cut (no augmented searches)** |
| logd | **cut (no augmented searches)** |
| nt_H3K27me3 | **cut (no augmented searches)** |
| nt_H3K4me3 | **cut (no augmented searches)** |
| nt_H3K9ac | **cut (no augmented searches)** |
| nt_H3K9me3 | **cut (no augmented searches)** |
| nt_promoter_tata | **cut (no augmented searches)** |
| qm9 | **cut (no augmented searches)** |
| suzuki | **cut (no augmented searches)** |
| tcga_brca_pam50 | **cut (no augmented searches)** |
| tcga_luad_vs_lusc | **cut (no augmented searches)** |
| tz_her2_binary | **cut (no augmented searches)** |
| tz_her2_multiclass | **cut (no augmented searches)** |

## Step 9: knn, and who reads the distance search

Source: a read of tests/experiments and the submit scripts on 2026-10-03 (subagent report, claims with file:line kept in the session, not re-verified line by line by me). knn is not one of any dataset's two models. The distance search is read for all 41 datasets (the complexity experiments and c03 read it with no dataset filter). knn is needed as a downstream model in the distance search by s02, s05, s06, s08, s10, s13 and the OPIG shared-vectors census; all similarity experiments skip the six nucleotide datasets. s02 and s08 also skip datasets over 15,000 rows (kiba, bindingdb_kd). No stored-read experiment needs knn as an INTERNAL evaluator; only framework f34 does, and it is in no submit script. Evaluator is a per-row column in the stored results, not part of the search identity (result_cache.py IDENTITY_COLS vs CONFIG_COLS), so cutting evaluators does not change which stored search is read.

Decision (Adelaide 2026-10-03): run the knn downstream precompute on the datasets below; not on the six nucleotide datasets.

| dataset | knn downstream in the distance search |
|---|---|
| abagym_Ang2 | **run** |
| abagym_EGFR | **run** |
| abagym_VEGF | **run** |
| abcov_kd | **run** |
| adme_hlm | **run** |
| adme_mdr1 | **run** |
| adme_rlm | **run** |
| bh_doyle_plates | **run** |
| bindingdb_kd | **run** (not for s02/s08: over 15,000 rows) |
| buchwald_hartwig | **run** |
| caco2 | **run** |
| cdr_flexibility | **run** |
| ddg_combined | **run** |
| flab_binding_adams | **run** |
| flab_binding_koenig | **run** |
| flab_expression_adams | **run** |
| flab_expression_koenig | **run** |
| flab_thermo_tresanco | **run** |
| gb_human_enhancers_cohn | **not needed** (all similarity experiments skip nucleotide) |
| ginkgo_acsins | **run** |
| ginkgo_colloidal | **run** |
| ginkgo_hic | **run** |
| ginkgo_pr_cho | **run** |
| ginkgo_titer | **run** |
| herg | **run** |
| herg_ki | **run** |
| humanness_therapeutic | **run** |
| kiba | **run** (not for s02/s08: over 15,000 rows) |
| lhasa_herg | **run** |
| logd | **run** |
| nt_H3K27me3 | **not needed** (all similarity experiments skip nucleotide) |
| nt_H3K4me3 | **not needed** (all similarity experiments skip nucleotide) |
| nt_H3K9ac | **not needed** (all similarity experiments skip nucleotide) |
| nt_H3K9me3 | **not needed** (all similarity experiments skip nucleotide) |
| nt_promoter_tata | **not needed** (all similarity experiments skip nucleotide) |
| qm9 | **run** |
| suzuki | **run** |
| tcga_brca_pam50 | **run** |
| tcga_luad_vs_lusc | **run** |
| tz_her2_binary | **run** |
| tz_her2_multiclass | **run** |

Distance searches not yet stored but read by experiments: bindingdb_kd, kiba, lhasa_herg.
