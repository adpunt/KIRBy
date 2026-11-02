# Chapter 5 results: checked against the 2026-10-09 results

Written by Claude on 2026-10-09, read-only. Not reviewed by the author. Every number below carries its file and field.

- VERIFIED: I read the number in a raw result file or the code in this session.
- PAGE: the number is from a table on a generated page. Those tables were rebuilt on 9 Oct from the same raw files. I did not recompute it.
- UNVERIFIED: seen only in a document.

Where the outline states a number or verdict, each entry says what the outline says, what the results say now, and one of SAME / CHANGED / NOT FOUND / NOT YET RUN.

Short paths:
- `R/` = `/Volumes/seagate/kirby_results_2026-10-09/`
- `A/` = `/Volumes/seagate/claude_analysis_2026-10-08/answers/`

## 0. What changes the outline (short list)

1. **Matched-width PCA beats the hybrid on LogD and Caco-2 with the network.** Two experiments that fit PCA the same way agree on this.
   - LogD, dnn: PCA at 1,000 and 2,000 components gives 0.3923 and 0.3920 RMSE. The hybrid gives 0.4089.
   - Caco-2, dnn: PCA at 1,000 components gives 0.2727. The hybrid gives 0.2775.
   - Sources: `R/complexity/c02_complexity_matching_grid/<ds>.json` `pca_baselines`, and `R/complexity/c11_ensemble_vs_hybrid/<ds>.json` `results` rows with method `pca_matched`. VERIFIED, 5 folds.
2. **The PCA in the feature-reduction comparison (c17) is not the same PCA as in c02 and c11.** That comparison is the source of 5.5's "it beats principal components".
   - c02 and c11 fit the model directly on the principal components. VERIFIED: `c02_complexity_grid.py:349-361`, `c11_ensemble_vs_hybrid.py:928-960`.
   - c17 standardises every component to unit variance and then fits. Its `_sel_pca` (`c17_information_theoretic_decomposition.py:687-693`) feeds `_model_score`, which applies `FeatureScaler` again (`:374-388`, called at `:937`). VERIFIED.
   - So the two PCAs differ in one way: whether each component is rescaled before the model.
   - The size of the difference: on hERG Ki with svm at 2,000 features, c17's PCA gives 0.8961 RMSE (PAGE `A/c17.md`, herg_ki section). c11's gives 0.5806. c17's PCA also gets worse as components are added, in every small-molecule table I read. c02's and c11's get better.
   - So the "beats PCA" half of 5.5's first bullet is measured against a PCA that does not match the one in c02 and c11.
3. **mRMR never ran on a small-molecule dataset.** In c17 it is capped at 200 features (`MRMR_MAX_BUDGET`, `c17...py:577`). The small-molecule budgets start at 250 or 500. `mrmr_budgets_served` is `[]` for qm9, herg_ki, caco2, logd, adme_mdr1 and buchwald_hartwig. VERIFIED. So "beats all three" in 5.5 is, on small molecules, "beats recursive elimination and the c17 PCA".
4. **Human and rat microsomal clearance are named as wins in 5.4 (ensembles) and 5.5 (feature reduction). Neither has results.** c11 and c17 have no adme_hlm or adme_rlm file. Angiopoietin-2 (abagym_Ang2) is also named in both bullets and has no c11 or c17 file. NOT FOUND.
5. **The federated split by family (5.8, fed03) reads the wrong way round in the outline.** The outline says splitting the budget evenly across families "scores lower".
   - In RMSE, lower is better. The even split (C3) has the lower RMSE than one selection over all families (C2) on Caco-2 with both models and on LogD with dnn (PAGE `A/fed03.md`, question 2).
   - So the even split is better on 3 of the 4 dataset-and-model pairs, not worse. The width caveat still applies.
6. **fed03 has two halves, and only one has results.**
   - The methods text (5.3.5) describes Part A: three real hERG sources, with arms C4, C8 and C9.
   - Part A needs lhasa_herg, which is cut. It has no result.
   - The results that exist are Part B only: Caco-2 and LogD, with the representation families as sites.
7. **fed01 has Caco-2 only, from partial files.** There are 3 dnn seeds and 2 svm seeds. There is no `caco2.json`, only `caco2.partial.jsonl` (`R/federated/fed01_noise_heterogeneity/`). LogD has no finished seed. The outline says "Only Caco-2 and LogD have federated results"; for fed01 that is Caco-2 alone.
8. **The c23 verdict has changed.** The outline says scoring candidates with the downstream model itself "wins only on SMAC colloidal stability". It now wins for both models on ginkgo_colloidal and on abagym_EGFR (PAGE `A/c23.md`). CHANGED.
9. **Four method descriptions in 5.3.4 describe code that has since changed.**
   - c11: the outline says fold_reselection at `_shared.py:1734`. The code now picks per fold with `fold_standard_selection`, on the fold's training rows.
   - c17: the outline says one random forest, with the recipe chosen on all rows. The code now uses the two decided models and a per-fold search.
   - c19: the outline says rf, lightgbm and xgboost at budget 100. The code now uses the decided models and the 2 best single representations plus the 2 best hybrids per fold.
   - c21: the decoy design is in the code, but the full run only did extra features on ginkgo_colloidal and ginkgo_hic.
   - Each of these is VERIFIED from the script docstrings.
10. **The PAM50 dataset has a held-out fold of 3 patients.** tcga_brca_pam50 fold 0 holds out 3 patients, and every method scores weighted F1 1.0 there (`A/FAULTS.md` A14; VERIFIED in c02 per-fold values below). Every PAM50 mean over folds includes that fold.
11. **Every dnn number in the core table comes from a network that does not clip its predictions.** The clip was added in commit c18ee30, 8 Oct 13:04.
    - Most c02 files were written before that commit. The exceptions are qm9 and ginkgo_hic (9 Oct) and the gp datasets (8 Oct, 17:47 to 19:28).
    - Whether the server ran the newer code for those exceptions is UNVERIFIED.
12. **The gp numbers on five datasets are to be replaced.** The datasets are abagym_Ang2, abagym_VEGF, ginkgo_acsins, ginkgo_hic and ginkgo_pr_cho (`A/FAULTS.md` A2).
    - Their c02 files were written after the gp fix (ffcd254, 8 Oct 11:03).
    - The per-fold searches that chose their hybrids ran before it. UNVERIFIED beyond FAULTS.md.

## 1. Source and selection rule for the core table

- **Source.** `R/complexity/c02_complexity_matching_grid/<dataset>.json`, field `standard_selection_summary[model]`.
  - `base1` is the best single representation and `hybrid1` the best hybrid.
  - Per-fold values are in `.per_fold[i].heldout_score`.
  - Differences per fold are in `heldout_delta_by_model[model].per_fold`, signed so that positive means the hybrid is better.
  - All VERIFIED, read this session (script in my session scratchpad, not saved to the repo).
- **How "best" is picked.** Code: `tests/experiments/_shared.py:1701` `fold_standard_selection`, called from `c02_complexity_grid.py:199`.
  - On each outer fold, one hybrid search runs on that fold's training rows only, over the dataset's decided grid with pruning off.
  - The search scores every configuration by its mean over the inner folds. There are 5 inner folds per configuration (`per_fold[i].configs[j].fold_scores` has 5 entries).
  - base1 is the representation with the best inner score. hybrid1 is the configuration with the best inner score.
  - Both are then fitted once on the fold's training rows and scored once on the held-out fold.
  - **The held-out fold is never used to choose.** This is not optimistic. base1 can be a different representation on each fold; the table lists them.
- **best_budget_per_fold (commit 8815f1e)** reads the same per-fold searches from `$KIRBY_CACHE_DIR/result_db/<ds>__ocv5f<k>s42.db`. Those files are on the server. Locally, `/Volumes/seagate/kirby_cache/result_db/` has fold 0 only, for abcov_kd, caco2 and herg_ki. So I could not run it. Its numbers are training-row scores only and it never reads the held-out fold. c02's `best_by_model` and `per_fold[i].configs` carry the same information.
- **Precompute:** not used. c02 is the per-fold, held-out version.
- **Picks that are optimistic, made on the held-out fold, in other files the chapter might cite:**
  - c11 `best_single`: `selection_scope` `heldout_fold_mean`.
  - c11 `oracle_single`: `heldout_fold`; the outline already calls it an upper bound.
  - c21 question 2, hybrid against best single: "Both are picked on the held-out folds" (`A/c21.md`).
  - c33 question 1 takes the best held-out mean among budgets on each side (`A/c33.md`), so both sides are picked on held-out scores.
  - c11's `base_top1` is the honest one. It matches c02's base1 within 0.001 (for example LogD svm: 0.4928 against 0.4930).
- **Folds.** All 24 datasets in the table have 5 of 5 outer folds scored (`n_outer_folds_scored`). VERIFIED.
- **Missing from the table:**
  - flab_binding_koenig: only `flab_binding_koenig.partial.jsonl`, fold 0. On fold 0, rf goes from esm3 at 0.4220 to a hybrid 0.0391 higher (0.4611). svm goes from ablang2 at 0.4690 to 0.0649 higher (0.5339). Spearman, 1 fold. VERIFIED.
  - herg_ic50: runs only the repeat-measurement experiments, so it has no c02 file.

## 2. Core table: best single representation against best hybrid, held out, per dataset and decided model

One row is one dataset and one of its two decided models. Scores are on the held-out fold, in the dataset's own metric. The arrow gives the direction: ↓ means lower is better, ↑ means higher is better. "hybrid minus single" is signed so that + always means the hybrid is better. SD is the standard deviation over the 5 folds. Source: `R/complexity/c02_complexity_matching_grid/<ds>.json`. VERIFIED.

#### Small molecules

| dataset | model | metric | folds | best single (mean) | single per fold | single SD | best hybrid (mean) | hybrid per fold | hybrid SD | hybrid minus single (better = +) | folds hybrid better | single representation picked per fold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qm9 | dnn | RMSE ↓ | 5 | 0.3619 | 0.3978, 0.3269, 0.3674, 0.3649, 0.3524 | 0.0257 | 0.3155 | 0.3473, 0.2980, 0.3241, 0.3153, 0.2929 | 0.0218 | +0.0464 | 5 of 5 | mordred, mordred, mordred, mordred, mordred |
| qm9 | lightgbm | RMSE ↓ | 5 | 0.3720 | 0.4118, 0.3394, 0.3807, 0.3789, 0.3493 | 0.0287 | 0.3463 | 0.3725, 0.3129, 0.3551, 0.3607, 0.3301 | 0.0242 | +0.0257 | 5 of 5 | mordred, mordred, mordred, mordred, mordred |
| herg_ki | svm | RMSE ↓ | 5 | 0.5881 | 0.6397, 0.5644, 0.5614, 0.5291, 0.6457 | 0.0518 | 0.5590 | 0.6071, 0.5477, 0.5228, 0.4952, 0.6224 | 0.0544 | +0.0290 | 5 of 5 | mhg_gnn, mhg_gnn, mhg_gnn, mhg_gnn, mhg_gnn |
| herg_ki | dnn | RMSE ↓ | 5 | 0.6031 | 0.6365, 0.5948, 0.5660, 0.5369, 0.6814 | 0.0572 | 0.5691 | 0.6035, 0.5407, 0.5394, 0.5270, 0.6349 | 0.0474 | +0.0340 | 5 of 5 | mhg_gnn, grover, chembert, grover, grover |
| caco2 | dnn | RMSE ↓ | 5 | 0.3042 | 0.2759, 0.2885, 0.3164, 0.3384, 0.3019 | 0.0243 | 0.2775 | 0.2415, 0.2633, 0.2819, 0.3184, 0.2822 | 0.0284 | +0.0267 | 5 of 5 | mhg_gnn, mhg_gnn, mhg_gnn, graph_kernel, pdv |
| caco2 | svm | RMSE ↓ | 5 | 0.3036 | 0.2760, 0.2771, 0.3066, 0.3255, 0.3326 | 0.0264 | 0.2862 | 0.2563, 0.2732, 0.2889, 0.3110, 0.3017 | 0.0220 | +0.0174 | 5 of 5 | pdv, pdv, pdv, pdv, ecfp4 |
| logd | dnn | RMSE ↓ | 5 | 0.4497 | 0.4308, 0.4659, 0.4710, 0.4443, 0.4366 | 0.0179 | 0.4089 | 0.3970, 0.4305, 0.4141, 0.4075, 0.3952 | 0.0143 | +0.0408 | 5 of 5 | mhg_gnn, mordred, mordred, mordred, mordred |
| logd | svm | RMSE ↓ | 5 | 0.4930 | 0.4758, 0.5159, 0.4832, 0.4946, 0.4956 | 0.0152 | 0.4302 | 0.4200, 0.4501, 0.4239, 0.4283, 0.4286 | 0.0117 | +0.0628 | 5 of 5 | mhg_gnn, mhg_gnn, mhg_gnn, mhg_gnn, mhg_gnn |
| adme_hlm | svm | RMSE ↓ | 5 | 0.4574 | 0.4428, 0.4584, 0.4803, 0.4673, 0.4383 | 0.0173 | 0.4267 | 0.4284, 0.4256, 0.4373, 0.4241, 0.4183 | 0.0070 | +0.0307 | 5 of 5 | pdv, pdv, pdv, pdv, pdv |
| adme_hlm | dnn | RMSE ↓ | 5 | 0.4660 | 0.4770, 0.4702, 0.4411, 0.4744, 0.4672 | 0.0144 | 0.4273 | 0.4337, 0.4174, 0.4206, 0.4393, 0.4256 | 0.0091 | +0.0386 | 5 of 5 | mhg_gnn, mhg_gnn, mhg_gnn, pdv, mhg_gnn |
| adme_rlm | svm | RMSE ↓ | 5 | 0.5209 | 0.5375, 0.5204, 0.5380, 0.5065, 0.5022 | 0.0168 | 0.4818 | 0.4935, 0.4928, 0.4867, 0.4699, 0.4661 | 0.0129 | +0.0392 | 5 of 5 | pdv, pdv, pdv, pdv, pdv |
| adme_rlm | dnn | RMSE ↓ | 5 | 0.5349 | 0.5552, 0.5279, 0.5361, 0.5295, 0.5258 | 0.0120 | 0.4987 | 0.5223, 0.4992, 0.5032, 0.4845, 0.4843 | 0.0157 | +0.0362 | 5 of 5 | mhg_gnn, pdv, mhg_gnn, mhg_gnn, mhg_gnn |
| adme_mdr1 | svm | RMSE ↓ | 5 | 0.4678 | 0.4788, 0.4600, 0.4665, 0.4485, 0.4850 | 0.0146 | 0.4370 | 0.4438, 0.4164, 0.4365, 0.4295, 0.4586 | 0.0158 | +0.0308 | 5 of 5 | pdv, pdv, pdv, pdv, pdv |
| adme_mdr1 | dnn | RMSE ↓ | 5 | 0.4732 | 0.4846, 0.4494, 0.4757, 0.4639, 0.4921 | 0.0169 | 0.4438 | 0.4664, 0.4220, 0.4353, 0.4368, 0.4583 | 0.0181 | +0.0294 | 5 of 5 | pdv, pdv, pdv, pdv, pdv |
| herg | lightgbm | AUROC ↑ | 5 | 0.8555 | 0.8395, 0.8490, 0.8620, 0.8529, 0.8744 | 0.0133 | 0.8666 | 0.8611, 0.8609, 0.8605, 0.8684, 0.8824 | 0.0094 | +0.0111 | 4 of 5 | mordred, mordred, mhg_gnn, mhg_gnn, mhg_gnn |
| herg | xgboost | AUROC ↑ | 5 | 0.8532 | 0.8369, 0.8479, 0.8618, 0.8461, 0.8732 | 0.0143 | 0.8644 | 0.8586, 0.8680, 0.8597, 0.8582, 0.8777 | 0.0084 | +0.0113 | 4 of 5 | mordred, mordred, mhg_gnn, mhg_gnn, mhg_gnn |

#### Antibodies

| dataset | model | metric | folds | best single (mean) | single per fold | single SD | best hybrid (mean) | hybrid per fold | hybrid SD | hybrid minus single (better = +) | folds hybrid better | single representation picked per fold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flab_expression_koenig | svm | Spearman ↑ | 5 | 0.7029 | 0.6983, 0.7227, 0.6820, 0.7373, 0.6741 | 0.0268 | 0.7729 | 0.7954, 0.7968, 0.7579, 0.7526, 0.7619 | 0.0214 | +0.0700 | 5 of 5 | esm3_cdr, esm3_cdr, prot_t5, esm3_cdr, prot_t5 |
| flab_expression_koenig | dnn | Spearman ↑ | 5 | 0.7163 | 0.7103, 0.6935, 0.6715, 0.7343, 0.7719 | 0.0387 | 0.7352 | 0.7564, 0.7417, 0.6946, 0.7481, 0.7354 | 0.0240 | +0.0189 | 4 of 5 | esm3_cdr, esm3_cdr, prot_t5, esm3_cdr, esm3_cdr |
| flab_thermo_tresanco | dnn | Spearman ↑ | 5 | 0.6939 | 0.7700, 0.7542, 0.5636, 0.6867, 0.6951 | 0.0813 | 0.6764 | 0.6998, 0.7819, 0.5174, 0.6672, 0.7157 | 0.0982 | -0.0175 | 2 of 5 | saprot, saprot, saprot, saprot, saprot |
| flab_thermo_tresanco | svm | Spearman ↑ | 5 | 0.6901 | 0.7550, 0.7214, 0.6572, 0.6687, 0.6482 | 0.0461 | 0.6899 | 0.6604, 0.7688, 0.6475, 0.6684, 0.7044 | 0.0489 | -0.0002 | 2 of 5 | saprot, saprot, saprot, saprot, saprot |
| abagym_EGFR | svm | Spearman ↑ | 5 | 0.2821 | 0.1114, 0.2142, 0.5164, 0.3491, 0.2197 | 0.1557 | 0.3138 | 0.2343, 0.3076, 0.5069, 0.3036, 0.2164 | 0.1154 | +0.0316 | 2 of 5 | mutation_features, abroberta_cdr, esm2_early, abroberta_cdr, abroberta_cdr |
| abagym_EGFR | dnn | Spearman ↑ | 5 | 0.1810 | 0.1694, 0.1988, 0.5201, 0.1893, -0.1727 | 0.2453 | 0.2160 | 0.1309, 0.1215, 0.5783, 0.1501, 0.0994 | 0.2033 | +0.0350 | 2 of 5 | esm3, saprot_cdr, ablang2, mutation_features, mutation_features |
| abagym_VEGF | gp | Spearman ↑ | 5 | 0.1981 | 0.5109, 0.3402, 0.3033, -0.1312, -0.0326 | 0.2696 | 0.3017 | 0.3089, 0.5166, 0.3459, 0.1322, 0.2048 | 0.1469 | +0.1035 | 4 of 5 | ablang2, esmc, saprot, abroberta_cdr, mutation_features |
| abagym_VEGF | svm | Spearman ↑ | 5 | 0.4317 | 0.6279, 0.4915, 0.7467, 0.2435, 0.0489 | 0.2843 | 0.4737 | 0.3710, 0.6817, 0.7639, 0.1274, 0.4242 | 0.2551 | +0.0419 | 3 of 5 | mutation_features, esmc, esmc, esmc, mutation_features |
| abagym_Ang2 | gp | Spearman ↑ | 5 | 0.4550 | 0.2073, 0.6126, 0.7541, 0.6403, 0.0608 | 0.3022 | 0.5049 | 0.2410, 0.7776, 0.6380, 0.4933, 0.3748 | 0.2114 | +0.0499 | 3 of 5 | saprot, mutation_features, saprot, saprot, esm3 |
| abagym_Ang2 | svm | Spearman ↑ | 5 | 0.4543 | 0.1980, 0.6713, 0.5940, 0.6101, 0.1979 | 0.2358 | 0.5216 | 0.2690, 0.7047, 0.5785, 0.4767, 0.5792 | 0.1627 | +0.0674 | 3 of 5 | saprot, abroberta_cdr, saprot, saprot, esm3 |
| humanness_therapeutic | xgboost | AUROC ↑ | 5 | 0.9511 | 0.9352, 0.9701, 0.9670, 0.9368, 0.9466 | 0.0165 | 0.9555 | 0.9139, 0.9878, 0.9652, 0.9524, 0.9580 | 0.0269 | +0.0043 | 3 of 5 | igt5, antiberty_cdr, igt5, abroberta, antiberty_cdr |
| humanness_therapeutic | rf | AUROC ↑ | 5 | 0.9659 | 0.9564, 0.9735, 0.9773, 0.9681, 0.9542 | 0.0102 | 0.9614 | 0.9311, 0.9889, 0.9701, 0.9643, 0.9527 | 0.0214 | -0.0045 | 1 of 5 | antiberty_cdr, antiberty_cdr, antiberty_cdr, igt5, antiberty_cdr |
| ddg_combined | svm | Spearman ↑ | 5 | 0.0819 | 0.1254, 0.2372, 0.1881, -0.3775, 0.2364 | 0.2609 | 0.2178 | 0.1270, 0.1166, 0.2861, 0.3217, 0.2374 | 0.0926 | +0.1359 | 4 of 5 | esm2_wt_delta, mutation_features, mutation_features, antiberty, mutation_features |
| ddg_combined | dnn | Spearman ↑ | 5 | 0.1317 | 0.3427, 0.0680, 0.0760, 0.2151, -0.0434 | 0.1494 | 0.3542 | 0.1961, 0.2050, 0.4120, 0.5834, 0.3744 | 0.1609 | +0.2225 | 4 of 5 | esm2_wt_delta, esm2_wt_delta, esm2_wt_delta, onehot, onehot |
| cdr_flexibility | xgboost | AUROC ↑ | 5 | 0.8527 | 0.7168, 0.8985, 0.8670, 0.8210, 0.9600 | 0.0912 | 0.8689 | 0.7863, 0.9173, 0.8803, 0.8809, 0.8799 | 0.0489 | +0.0163 | 4 of 5 | abroberta, antiberty, antiberty, esm2_concat, onehot |
| cdr_flexibility | lightgbm | AUROC ↑ | 5 | 0.8713 | 0.7411, 0.8960, 0.8963, 0.8691, 0.9542 | 0.0792 | 0.8669 | 0.7905, 0.9010, 0.8763, 0.8948, 0.8719 | 0.0444 | -0.0044 | 3 of 5 | esmc, ablang2, ablang2, esm2_concat, ablang2 |
| abcov_kd | lightgbm | Spearman ↑ | 5 | 0.2494 | 0.3409, -0.1547, 0.2167, 0.5362, 0.3078 | 0.2542 | 0.1654 | 0.0498, 0.1947, 0.4001, 0.4700, -0.2875 | 0.3030 | -0.0840 | 2 of 5 | abroberta, onehot, abroberta, abroberta_cdr, abroberta |
| abcov_kd | xgboost | Spearman ↑ | 5 | 0.2925 | 0.3212, 0.3578, 0.3919, 0.2838, 0.1079 | 0.1108 | 0.3050 | 0.3394, 0.3098, 0.1708, 0.5745, 0.1303 | 0.1749 | +0.0125 | 3 of 5 | blosum, onehot, esmc_cdr, abroberta, esm3 |
| ginkgo_colloidal | dnn | Spearman ↑ | 5 | 0.2766 | 0.0671, 0.1967, 0.2492, 0.4419, 0.4280 | 0.1591 | 0.2677 | 0.2915, 0.0273, 0.4067, 0.3585, 0.2546 | 0.1467 | -0.0088 | 2 of 5 | prot_t5, prot_t5, prot_t5, prot_t5, prot_t5 |
| ginkgo_colloidal | svm | Spearman ↑ | 5 | 0.3485 | 0.2699, 0.3381, 0.3902, 0.5085, 0.2359 | 0.1076 | 0.3480 | 0.1709, 0.3168, 0.4126, 0.5447, 0.2948 | 0.1397 | -0.0005 | 3 of 5 | prot_t5, prot_t5, esm2_early, esm2_concat, esm2_concat |
| ginkgo_hic | gp | Spearman ↑ | 5 | 0.6000 | 0.5442, 0.5344, 0.6338, 0.6490, 0.6388 | 0.0558 | 0.5515 | 0.5543, 0.5343, 0.4921, 0.5048, 0.6719 | 0.0716 | -0.0486 | 2 of 5 | esm2_concat, esm2_concat, esm2_concat, esm2_concat, esm2_concat |
| ginkgo_hic | dnn | Spearman ↑ | 5 | 0.5311 | 0.5187, 0.4980, 0.5698, 0.6060, 0.4631 | 0.0570 | 0.5293 | 0.5019, 0.5256, 0.3998, 0.6250, 0.5943 | 0.0879 | -0.0018 | 3 of 5 | esm2_concat, esm2_concat, esm2_concat, esm2_concat, esm2_concat |
| ginkgo_acsins | lightgbm | Spearman ↑ | 5 | 0.4631 | 0.4374, 0.5371, 0.4386, 0.2842, 0.6182 | 0.1253 | 0.5242 | 0.5626, 0.6857, 0.5572, 0.2628, 0.5527 | 0.1564 | +0.0611 | 3 of 5 | igt5, igt5, ablang2, igt5, igt5 |
| ginkgo_acsins | gp | Spearman ↑ | 5 | 0.5357 | 0.5271, 0.5446, 0.5857, 0.3353, 0.6857 | 0.1278 | 0.5547 | 0.5927, 0.5872, 0.5756, 0.3160, 0.7020 | 0.1428 | +0.0190 | 3 of 5 | ablang2, ablang2, igt5, ablang2, ablang2 |
| ginkgo_titer | dnn | Spearman ↑ | 5 | 0.3083 | 0.3862, 0.2023, 0.5015, 0.2814, 0.1701 | 0.1364 | 0.3668 | 0.2257, 0.4203, 0.5238, 0.4036, 0.2604 | 0.1226 | +0.0585 | 4 of 5 | ablang2, igt5, ablang2, esm3, esm2_entropy_top20 |
| ginkgo_titer | rf | Spearman ↑ | 5 | 0.2504 | 0.3085, 0.2884, 0.4298, 0.0214, 0.2040 | 0.1513 | 0.2706 | 0.2622, 0.2473, 0.5205, 0.0996, 0.2236 | 0.1537 | +0.0202 | 3 of 5 | igt5, igt5, ablang2, prot_t5, igt5 |
| ginkgo_pr_cho | gp | Spearman ↑ | 5 | 0.5009 | 0.4474, 0.5749, 0.5293, 0.5016, 0.4512 | 0.0539 | 0.4419 | 0.4200, 0.4324, 0.5322, 0.5095, 0.3155 | 0.0855 | -0.0590 | 2 of 5 | esmc, prot_t5, prot_t5, prot_t5, prot_t5 |
| ginkgo_pr_cho | dnn | Spearman ↑ | 5 | 0.4003 | 0.3892, 0.4448, 0.4846, 0.3818, 0.3012 | 0.0696 | 0.4133 | 0.4225, 0.4570, 0.5204, 0.3507, 0.3159 | 0.0820 | +0.0130 | 4 of 5 | prot_t5, onehot, esmc, prot_t5, prot_t5 |

#### Reaction

| dataset | model | metric | folds | best single (mean) | single per fold | single SD | best hybrid (mean) | hybrid per fold | hybrid SD | hybrid minus single (better = +) | folds hybrid better | single representation picked per fold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| buchwald_hartwig | rf | RMSE ↓ | 5 | 14.4056 | 18.8815, 15.3866, 13.0996, 10.0374, 14.6230 | 3.2338 | 15.2383 | 17.8533, 15.7597, 15.2950, 10.0142, 17.2693 | 3.1036 | -0.8327 | 2 of 5 | diff_avalon, cgr_morgan, cgr_morgan, cgr_morgan, cgr_morgan |
| buchwald_hartwig | lightgbm | RMSE ↓ | 5 | 16.1679 | 20.2242, 15.7682, 16.8262, 11.8136, 16.2072 | 3.0004 | 15.6211 | 20.0970, 14.8241, 14.9378, 11.2088, 17.0376 | 3.2638 | +0.5468 | 4 of 5 | diff_avalon, diff_mhg_gnn, diff_mhg_gnn, diff_mhg_gnn, diff_mhg_gnn |

#### Gene expression

| dataset | model | metric | folds | best single (mean) | single per fold | single SD | best hybrid (mean) | hybrid per fold | hybrid SD | hybrid minus single (better = +) | folds hybrid better | single representation picked per fold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tcga_brca_pam50 | dnn | weighted F1 ↑ | 5 | 0.8528 | 1.0000, 0.8686, 0.8500, 0.8399, 0.7056 | 0.1046 | 0.9011 | 1.0000, 0.9103, 0.8558, 0.8922, 0.8470 | 0.0611 | +0.0482 | 4 of 5 | gene_geneformer_v2, gene_geneformer_v2, gene_geneformer_v2, gene_geneformer_v2, gene_bulkrnabert |
| tcga_brca_pam50 | svm | weighted F1 ↑ | 5 | 0.8808 | 1.0000, 0.8716, 0.8578, 0.8313, 0.8435 | 0.0683 | 0.8950 | 1.0000, 0.8965, 0.8806, 0.8426, 0.8551 | 0.0624 | +0.0141 | 4 of 5 | gene_geneformer_v2, gene_geneformer_v2, gene_geneformer_v2, gene_geneformer_v2, gene_geneformer_v2 |

### What the table says, by domain

**Small molecules (8 datasets with results, 16 dataset-and-model pairs).**
- The hybrid beats the best single representation in all 16 pairs, on the mean over 5 folds.
- On the seven RMSE datasets it is better on every fold: 70 of 70 fold comparisons.
- On herg (AUROC) it is better on 4 of 5 folds with each model. On fold 2 the single representation wins: lightgbm by 0.0015, xgboost by 0.0021.
- The RMSE gains run from 0.0174 (Caco-2, svm: 0.3036 to 0.2862) to 0.0628 (LogD, svm: 0.4930 to 0.4302).
- As a share of the single representation's RMSE, that is 5.7% to 12.7%. This is a derived number; the raw pairs are in the table.
- hERG classification gains +0.0111 AUROC with lightgbm (0.8555 to 0.8666) and +0.0113 with xgboost (0.8532 to 0.8644).
- On qm9, whose labels are computed, the gain is +0.0464 RMSE with dnn (0.3619 to 0.3155) and +0.0257 with lightgbm.
- Fold spread is smaller than the gain on most small-molecule pairs. Example, LogD svm: single-representation SD 0.0152 against a gain of 0.0628.
- The exception is hERG Ki. Its single-representation SD (0.0518 svm, 0.0572 dnn) is larger than its gain (0.0290, 0.0340). The gain is still positive on all 5 folds, because the folds move together.

**Antibodies (14 datasets with results, 28 pairs).**
- The hybrid's mean is better in 18 of 28 pairs and worse in 10.
- It loses with both models on flab_thermo_tresanco (dnn -0.0175, svm -0.0002), ginkgo_colloidal (-0.0088, -0.0005) and ginkgo_hic (gp -0.0486, dnn -0.0018).
- It loses with one model on abcov_kd lightgbm (-0.0840), ginkgo_pr_cho gp (-0.0590), humanness_therapeutic rf (-0.0045) and cdr_flexibility lightgbm (-0.0044).
- Fold-level wins on antibodies are often 2 or 3 of 5. The antibody fold SDs reach 0.30 Spearman (abagym_Ang2 gp single: 0.3022).

**Reaction (buchwald_hartwig).**
- rf: the hybrid loses, 14.4056 to 15.2383 RMSE in yield %. It is better on 2 of 5 folds.
- lightgbm: the hybrid wins, 16.1679 to 15.6211. It is better on 4 of 5 folds.
- Fold SD is about 3.0 to 3.3 RMSE, larger than either difference.

**Gene expression (tcga_brca_pam50).**
- The hybrid wins with both models: dnn 0.8528 to 0.9011, svm 0.8808 to 0.8950 weighted F1.
- Fold 0 is 1.0000 for both sides and both models, because it holds 3 patients. On folds 1 to 4 the hybrid is better on 4 of 4 for both models.

### The antibody exception, named

"Hybrid gains far larger than elsewhere": these are the antibody pairs where the hybrid's mean gain is above +0.05 Spearman. For comparison, the largest small-molecule gain is 0.0628 RMSE (LogD svm), in a different unit. So the comparison across domains is only by direction and fold count, not by size.

| dataset (antibodies) | model | single → hybrid, Spearman ↑ | gain | folds hybrid better | fold SD of single | note |
|---|---|---|---|---|---|---|
| flab_expression_koenig | svm | 0.7029 → 0.7729 | +0.0700 | 5 of 5 | 0.0268 | the cleanest: 4,275 antibodies, every fold, gain larger than SD |
| ddg_combined | dnn | 0.1317 → 0.3542 | +0.2225 | 4 of 5 | 0.1494 | 429 complexes; single representation near zero on two folds |
| ddg_combined | svm | 0.0819 → 0.2178 | +0.1359 | 4 of 5 | 0.2609 | fold 3 alone contributes +0.6992 (single -0.3775 there); folds 0 and 4 differ by under 0.002 |
| abagym_VEGF | gp | 0.1981 → 0.3017 | +0.1035 | 4 of 5 | 0.2696 | gp to be replaced (FAULTS A2) |
| abagym_Ang2 | svm | 0.4543 → 0.5216 | +0.0674 | 3 of 5 | 0.2358 | fold 4 alone +0.3813 |
| ginkgo_acsins | lightgbm | 0.4631 → 0.5242 | +0.0611 | 3 of 5 | 0.1253 | 242 antibodies |
| ginkgo_titer | dnn | 0.3083 → 0.3668 | +0.0585 | 4 of 5 | 0.1364 | 239 antibodies |

Reading: only **flab_expression_koenig with svm** has a gain larger than its fold SD and better on every fold. The other large antibody gains come with fold SDs of 0.12 to 0.27 Spearman, often carried by one fold.

The same datasets also hold the hybrid's losses: abcov_kd lightgbm -0.0840, and the three losses with both models listed above. With ensembles (c11), the ensembles beat the hybrid with both models on abcov_kd, ginkgo_colloidal, ginkgo_pr_cho and ginkgo_titer (PAGE `A/c11.md`, summary and question 1).

So on antibodies the hybrid's results spread wider in both directions than on small molecules. The antibody exception the chapter should name for the hybrid is flab_expression_koenig. ddg_combined is a second, but its gain rests on near-zero single-representation scores.

### Domain balance: the small-molecule picture alone

- Every small-molecule dataset with results, both models, held out, picked on training rows: the hybrid beats the best single representation (16 of 16 pairs; 78 of 80 folds). Source: the core table.
- Against ensembles (c11, 5 folds, `results` rows `hybrid_top1`, `ensemble_s1_auto` and the "optimal" model's `ensemble_s2_auto`). VERIFIED.
  - qm9 and LogD: the hybrid beats the ensemble of the same model, with both models. qm9 dnn: hybrid 0.3174 against 0.3402 RMSE. LogD svm: 0.4305 against 0.4752.
  - The second ensemble uses each representation's own better model, so it can mix models. It beats the hybrid in two pairs. qm9 lightgbm: ensemble 0.3402 against hybrid 0.3463. LogD svm: ensemble 0.4175 against hybrid 0.4305. Against the dnn hybrids on qm9 (0.3174) and LogD (0.4086) it loses.
  - Caco-2, hERG Ki and adme_mdr1: depends on the model.
  - hERG: both ensembles win with both models (lightgbm: 0.8719 and 0.8724 AUROC against 0.8666).
- Against matched-width PCA (c02 and c11; see item 1 of section 0): with dnn, PCA beats the hybrid on LogD, Caco-2 and adme_mdr1. For adme_mdr1, c02 PCA at 1,000 components gives 0.4411 against the hybrid's 0.4438 (c02 `pca_baselines`). With svm the hybrid beats PCA on all three.
- Best budget: on LogD (dnn), adme_hlm (dnn) and flab_expression_koenig (both models), hybrid1 used 2,000 features on all 5 folds. That is the largest decided budget, so the best number of features may lie above the grid (c02 `standard_selection_summary.hybrid1.per_fold[].name`, third field). VERIFIED. c33 searched beyond 2,000 on LogD: dnn at 3,016 features scores 0.4066 RMSE (PAGE `A/c33.md` summary). That number was picked on held-out scores.

### Fill for Table 5.2's last column (best single representation per dataset and model)

Use the "single representation picked per fold" column of the core table. Where the pick changes across folds, list the representation picked most often:

- qm9: mordred (dnn and lightgbm).
- herg_ki: svm mhg_gnn; dnn grover (3 of 5).
- caco2: dnn mhg_gnn (3 of 5); svm pdv (4 of 5).
- logd: dnn mordred (4 of 5); svm mhg_gnn.
- adme_hlm: svm pdv; dnn mhg_gnn (4 of 5).
- adme_rlm: svm pdv; dnn mhg_gnn (4 of 5).
- adme_mdr1: pdv (both).
- herg: lightgbm and xgboost mhg_gnn (3 of 5).
- Antibodies: often a different pick on each fold. Exact lists are in the table.

This replaces the f26 pointer in outline 5.3.1, which used pools from before pruning.

## 3. Outline bullets, one by one

### 5.4 When the hybrid helps

**5.4 bullet 1: hybrid against best single representation (F1; c02, c22).**
- *Wins on the small-molecule datasets for both models.*
  - Outline: Caco-2, LogD, hERG Ki, hERG and the three ADME sets.
  - Results: all of those, plus qm9. 5 folds. Source: the core table. VERIFIED. **SAME** (qm9 can be added).
- *LogD: dnn 0.4497 to 0.4089, svm 0.4930 to 0.4302 RMSE.*
  - Results: dnn 0.4497 to 0.4089, svm 0.4930 to 0.4302. 5 folds.
  - Source: c02 `logd.json` `standard_selection_summary.{dnn,svm}.{base1,hybrid1}.heldout.mean`. VERIFIED. **SAME**.
- *Loses for both models on nanobody thermostability, SMAC colloidal stability and HIC retention.*
  - flab_thermo_tresanco: dnn 0.6939 to 0.6764, svm 0.6901 to 0.6899.
  - ginkgo_colloidal: dnn 0.2766 to 0.2677, svm 0.3485 to 0.3480.
  - ginkgo_hic: gp 0.6000 to 0.5515, dnn 0.5311 to 0.5293.
  - VERIFIED, 5 folds. **SAME**.
  - Three of these six losses are under 0.002 Spearman, much smaller than the fold SD (0.05 to 0.16), so "loses" means "no gain". HIC gp is to be replaced.
  - Not in the outline: the hybrid also loses with one model on abcov_kd (lightgbm), ginkgo_pr_cho (gp), humanness_therapeutic (rf), cdr_flexibility (lightgbm) and buchwald_hartwig (rf).
- c22 (also cited for F1): logd has a partial file only, 2 folds (PAGE `A/c22.md`: "logd, read on 2 folds only so far, dnn 0.4535 → 0.4138"). adme_rlm and flab_expression_koenig are also partial. Use c02 for F1, not c22.

**5.4 bullet 2: budget split over top 2 or 3 representations (c18).**
- *hERG Ki, svm, 500 features: 0.6006 to 0.5672 RMSE.*
  - Results: `R/complexity/c18_diversity_vs_depth/herg_ki.json` `per_model.svm.per_budget.500.depth_1.score` = 0.6006 and `.depth_3.score` = 0.5672. 5 folds. VERIFIED. **SAME**.
- Context the outline does not give: splitting the same budget evenly over all 10 representations scores 0.5569 there (`uniform_all`). That is better than both depth_3 (0.5672) and the search's own allocation (`natural`, 0.5593).
  - On herg_ki, `natural` loses to the even split with both models (PAGE `A/c18.md` question 3: no, no).
  - On the other small molecules `natural` beats the even split with both models on adme_mdr1, adme_rlm, herg and logd. It depends on the model on caco2 and qm9.
- c18 coverage: adme_hlm has no file. flab_expression_koenig has 2 of 5 folds and flab_binding_koenig 3 of 5, partial (PAGE `A/c18.md`).

**5.4 bullet 3: hybrid against the two ensembles (c11).**
- The two ensembles are `ensemble_s1_auto` (same model, how predictions are pooled is chosen on inner folds) and `ensemble_s2_auto` (each representation uses its own better model). File: `R/complexity/c11_ensemble_vs_hybrid/<ds>.json` `results`. VERIFIED, 5 folds.
- *Beats both on human microsomal clearance.* adme_hlm has no c11 file. **NOT FOUND** (`A/c11.md`; FAULTS A1: dnn rerun pending).
- *Beats both on hERG Ki with svm.* Hybrid 0.5596, ensembles 0.5819 and 0.5828 RMSE. **SAME**.
- *Beats both on the PAM50 subtype.*
  - dnn: hybrid 0.9011, ensembles 0.8904 and 0.8992 weighted F1.
  - svm: hybrid 0.9000, ensembles 0.8909 and 0.8992.
  - **SAME**. Fold 0 has 3 patients and is 1.0 for every method.
- *Both ensembles beat it on Angiopoietin-2 binding.* abagym_Ang2 has no c11 file. **NOT FOUND**.
- *Both ensembles beat it on anti-coronavirus affinity.* abcov_kd lightgbm: hybrid 0.1654, ensembles 0.4186 and 0.4532. xgboost: hybrid 0.3050, ensembles 0.4851 and 0.4532. **SAME**.
- *Both ensembles beat it on CHO polyreactivity.* ginkgo_pr_cho gp: hybrid 0.4434, ensembles 0.4853 and 0.5078. dnn: hybrid 0.4131, ensembles 0.4490 and 0.5078. **SAME** (gp to be replaced).
- Now also true, not in the outline: both ensembles beat the hybrid with both models on ginkgo_colloidal and ginkgo_titer (PAGE `A/c11.md`, questions 1 and 3), and on herg (see section 2).
- Coverage: c11 is missing adme_hlm, adme_rlm, flab_binding_koenig, abagym_Ang2 and ginkgo_hic.
- Method text (5.3.4) is stale. c11's current schema is `c11_per_fold_standard_selection_2026_10_06`, with `selection_scope` `fold_train_rows`, not fold_reselection.

**5.4 bullet 4: easy and hard halves (c07).**
- *LogD hard half by scaffold frequency, dnn: 0.6698 single against 0.5735 own-search hybrid.*
  - Recomputed from `R/complexity/c07_data_complexity_partitions/logd.json`, `partitions.scaffold_frequency.hard.folds[i].half.dnn.base[0].heldout_score` and `.hybrids[0].heldout_score`.
  - Means 0.6698 and 0.5735. 5 folds, 129 or 130 held-out molecules each. VERIFIED. **SAME**.
- Not in the outline: c07 searches a random 2,000 of LogD's 5,028 molecules (`n_rows_searched` 2000, `row_cap` cap2000s42). The hard half is 649 of those 2,000. The text must say so.
- On the same row, the hybrid searched on all 2,000 molecules (0.5830) is worse than the half's own search (0.5735) (PAGE `A/c07.md` logd table).
- c07 has no qm9 or adme_hlm file. The ginkgo datasets are skipped: 5 groups (FAULTS A15).

**5.4 bullet 5: where the search overfits (c33, c24).**
- *Hybrids wider than the number of training molecules are better on expression titer and SMAC colloidal stability for both models.*
  - ginkgo_titer yes/yes, ginkgo_colloidal yes/yes (PAGE `A/c33.md` question 1). **SAME**.
  - Now also yes/yes on ginkgo_acsins and abcov_kd.
  - The rule picks the best held-out budget on each side, so both sides are optimistic.
- *Worse on hERG Ki and LogD for both.* herg_ki no/no, logd no/no. **SAME**. Example: logd dnn 0.4066 RMSE at 3,016 features against 0.4157 at 8,044 (PAGE `A/c33.md` summary).
- *One fold fewer of training molecules is generally worse: hERG Ki svm 0.5596 to 0.5931 RMSE.* This comes from c24, not c33.
  - `R/complexity/c24_blueprint_transfer/herg_ki.json` `by_model.svm`: the mean of the 5 diagonal cells of `transfer_matrix` is 0.5596, and `aggregate.mean_delta` is +0.0335, giving 0.5931. VERIFIED. 5 target folds × 4 source folds. **SAME**.
  - Also LogD dnn 0.4174 to 0.4400 (PAGE).
  - The page notes the own and cross sides also pick different budgets, so they differ in two ways at once.
- Coverage: c33 is missing qm9, adme_hlm, adme_rlm, adme_mdr1, flab_expression_koenig, flab_binding_koenig, abagym_VEGF, abagym_Ang2 and ginkgo_hic. c24 is missing qm9, adme_hlm, adme_rlm, adme_mdr1, flab_binding_koenig, abagym_Ang2 and ginkgo_hic.

**5.4 bullet 6: c05 cut.** Results exist for 16 datasets (`R/complexity/c05_learning_curves/`). Cut by the author on 2026-10-08. Nothing to check.

### 5.5 Why the hybrid helps

**5.5 bullet 1: against PCA, mRMR, recursive elimination and random (c17, c15, c09).**
- *Beats all three on human and rat microsomal clearance.* No c17 file for adme_hlm or adme_rlm (`A/c17.md`: "no answers until their dnn results are re-run"). **NOT FOUND**.
- *... Caco-2, hERG Ki and LogD with both models.* Yes/yes on all three (PAGE `A/c17.md` question 1). **SAME as written**, but see items 2 and 3 of section 0. Also:
  - mRMR did not run at any of these budgets.
  - c17's PCA is rescaled per component.
  - On hERG Ki with dnn, recursive elimination beats the hybrid at 1,000 features (0.5794 against 0.6112 RMSE). Random features beat it at 500, 1,000 and 2,000 (0.5656, 0.5705, 0.5813 against 0.5751, 0.6112, 0.6200) (PAGE `A/c17.md` herg_ki table).
  - Now also yes/yes on qm9 and adme_mdr1.
- *LogD at 1,000 features, network: 0.4099 against 0.4146 for recursive elimination.*
  - PAGE `A/c17.md` logd table: rank-1 hybrid 0.4099, recursive elimination 0.4146, PCA 0.4627, mRMR not run.
  - The hybrid beat the best other method on 2 of 5 folds at that budget. **SAME**.
  - Not recomputed from raw; the page table was rebuilt from `R/complexity/c17_information_theoretic_decomposition/logd.json` on 9 Oct.
- *Beats none of the three on Angiopoietin-2 binding.* abagym_Ang2 has no c17 file (ledger status pending). **NOT FOUND**.
- *... Buchwald-Hartwig yield, nanobody thermostability, CHO polyreactivity.* no/no on all three (PAGE). **SAME**.
- Method text (5.3.4) is stale: "one downstream model, a 100-tree random forest", "recipe chosen on all rows". The current code (`RESULT_IDENTITY` `c17_per_fold_decided_models_2026_10_05`) uses the two decided models and a per-fold search. VERIFIED. The outline's "CHECK: recipe selected on all rows" is resolved in the code.
- c17 skips classification datasets (cdr_flexibility, herg, humanness_therapeutic, tcga_brca_pam50: `regression_only`). Recursive elimination uses a random forest regressor.

**5.5 bullets 2 and 3: best number of features (F2a) and of representations (F2b).**
- No number in the outline. Data for both figures: c02 `best_by_model[model].budget` and `.rep_count` (the configuration with the best mean training-row score over the 5 outer folds), and the per-fold hybrid1 label (budget is field 3, representation count field 6). VERIFIED. Listed in section 6.
- "Differs by dataset and model": c02 question 2 (PAGE `A/c02.md`) says yes on 17 datasets and no on 7: abagym_Ang2, abagym_VEGF, abcov_kd, ddg_combined, flab_expression_koenig, ginkgo_pr_cho and qm9.
- Representation count: `None` means all representations in the pool were allowed. The best configuration uses all representations on most small-molecule folds. It uses 3 on qm9 lightgbm and herg, and 2 on buchwald_hartwig on every fold. The decided_grid.json `rep_counts` field was not read this session.

**5.5 bullet 4: scoring candidates with the downstream model (c23).**
- *Wins only on SMAC colloidal stability.* Now: the matched evaluator beats every mismatched one with both models on ginkgo_colloidal **and** abagym_EGFR (PAGE `A/c23.md`). **CHANGED** (one more dataset).
- On ginkgo_colloidal svm: 0.3806 Spearman against 0.3501 and 0.3626.
- qm9 has 3 of 5 folds (partial file). flab_expression_koenig is partial.
- The c23 svm near-family evaluator is the gp with starting length scale 1 (FAULTS A3). So the svm comparison on every c23 dataset uses that gp.

**5.5 bullet 5: importance weights during training (c19).**
- *At least one of four weighting conditions beats unweighted on most datasets; which wins changes.* The page's question 1 is yes for at least one model on 23 of the 24 datasets with results. The exception is adme_mdr1 (no/no); adme_hlm has no file (PAGE `A/c19.md`). **SAME**.
- Size: qm9 dnn 0.3163 to 0.3134. LogD dnn 0.4086 to 0.4077. The largest is ginkgo_colloidal dnn, 0.2607 to 0.3571 Spearman with `feature_weighted`.
- Method text is stale (the old rf/lightgbm/xgboost list at budget 100). Now: the decided models, and the 2 best single representations plus the 2 best hybrids per fold (`c19_importance_weighted_training.py` docstring). VERIFIED.
- FAULTS A1: c19 adme_hlm/adme_rlm dnn has a fold in the thousands of RMSE. adme_hlm has no file.

### 5.5a Admitting extra features, and the decoy test

- **Files exist.** `tests/experiments/results/framework/f24_augmentation_admission_*.json`, dated 7 Sept (file mtime).
  - Hybrid files exist for **4** datasets: buchwald_hartwig, caco2, flab_expression_koenig and tcga_brca_pam50.
  - herg_ki has one single-representation file (`__single__uncapped__curated__64__rf.json`).
  - abagym_VEGF, abcov_kd, flab_thermo_tresanco and logd have timing files only.
  - So "five datasets" is four with results and one with a single file. VERIFIED by `ls`.
- *Three outer folds, three seeds per fold.* `n_folds` 3, `n_seeds` 3 in `..._caco2__hybrid__b250__power_2__uncapped__rf.json`. VERIFIED. **SAME**.
- These are local framework runs from before the 5 October per-fold redesign. The base representation is picked from a `rep_prune` solo score on the server (`base_rep_provenance`). They have not been re-run on 9 Oct.
- *Caco-2, rf, extras appended outside the budget: 407 configurations, 160 took no decoy features.* Recounted across the caco2 `*uncapped*rf*` and `*single*` files: rows with `method` uncapped, `control` real, rf status ok = 407. Rows with zero `decoy_noise_per_fold` and zero `decoy_echo_per_fold` = 160. VERIFIED. **SAME**. "208 beat no extras" and "185 beat the matched random donor" were not recomputed.
- *Buchwald-Hartwig, lightgbm, extras displacing base features: 259 configurations, 59 no decoys.* Recounted the same way with `method` capped: 259 and 59. VERIFIED. **SAME**. 108 and 50 were not recomputed.
- *0 of 36 clean for the rule with no test, eight times over*, and *a fifth to two thirds*: not recomputed. UNVERIFIED.
- *Fold spread on Buchwald-Hartwig 1.6198 (lightgbm), 2.1040 (rf).* Not recomputed. The field is `fold_spread`; for caco2 rf power_2 uncapped it is 0.0085. UNVERIFIED for BH.
- **The full-run counterpart (c21) does not test admission.** In the 9 Oct results, c21 ran extra features only on ginkgo_colloidal and ginkgo_hic, admission method `uncapped` (docstring and PAGE `A/c21.md`). Elsewhere augmentation was cut (DECISIONS Step 8). The 5.5a table can only come from the 7 Sept f24 files.
- c21 question 2 (hybrid against best single) picks both on held-out folds.

### 5.6 Blueprints

- *Stability: phi reaches 0.4 only on Caco-2 svm, hERG lightgbm, and LogD with both models.*
  - PAGE `A/c24.md`: caco2 svm 0.4238, herg lightgbm 0.4470, logd dnn 0.4203, logd svm 0.4972. Every other dataset and model is below 0.4.
  - herg_ki svm 0.3232 VERIFIED (`R/complexity/c24_blueprint_transfer/herg_ki.json` `by_model.svm.stability.nogueira_phi`). **SAME**.
- The feature sets compared differ in size from fold to fold (herg_ki svm: 1,000, 250, 250, 500 and 500 features, `stability.n_features_selected_per_fold`), because each fold picks its own budget. The text should say phi is computed over sets of different sizes.
- The deeper panel's per-feature names: c24 `stability.feature_sets_per_fold` holds the (representation, feature index) pairs per fold. Feature names are not in that field. TODO stays.

### 5.7 OPIG

- *Hybrid beats base1 on three ChEMBL targets except one; fxa dnn 0.7367 to 0.6612 MAE on all 5 folds.*
  - The exception is fxa with lightgbm (0.6854 to 0.7144) (PAGE `A/opig_qsar.md`).
  - Hybrid 0.6612 VERIFIED: `R/opig/qsar.json` `fxa.results.kirby_hybrid.models.dnn.qsar.mae.mean`. base1 0.7367 is PAGE (per-fold base1 from `per_fold_selection`).
  - 5 folds. **SAME**.
  - fxa built 8 of 10 representations; grover and smited are missing. `top2_base_detail[].score_mean` is null in the stored selection. The page reads base1's held-out scores from elsewhere in the file; not checked.
- *Shared rows: hERG Ki svm, graph_kernel 0.6109 to 0.5715 RMSE, all 5 folds, 105 to 0 molecules.* PAGE `A/opig_shared_vectors_remedy.md`, pair 2 (adds ecfp4 and mhg_gnn). **SAME**. Only herg_ki ran.
- *Census: MACCS merges 804 of 2,158 Caco-2 molecules; on two antibody sets a representation merges most.* PAGE `A/opig_shared_vectors_census.md`: aac on abagym_EGFR 432 of 659, structural on abagym_VEGF 332 of 418. **SAME**.
  - The census is a partial file (`R/opig/shared_vectors_census/opig_shared_vectors_census.partial.jsonl`). It covers only the 10 datasets alphabetically up to ddg_combined. herg_ki, logd, qm9, herg and the ginkgo and flab sets are not counted.
- *Recipe transfer LogD to hERG Ki beats hERG Ki's best single representation with both models, loses to the native hybrid.*
  - svm: base1 0.5888, native 0.5631, recipe transfer 0.5665, full transfer 0.5692.
  - dnn: 0.6035, 0.5666, 0.5889, 0.5926 RMSE.
  - 5 folds (PAGE `A/opig_blueprint_transfer.md`). **SAME**.
- *Ginkgo HIC: neither hybrid beats the single best representation.* PAGE `A/opig_ginkgo.md`: **4 of 5 folds**, ginkgo_hic only. dnn: esm2_concat 0.5529 against hybrid1 0.4728. **SAME**, but partial. The other four Ginkgo properties have no opig_ginkgo result.

### 5.8 Features chosen across separate sites

- *fed03: one selection over all families beats the best single family and best single representation on Caco-2 and LogD, both models.* yes on all 8 entries (PAGE `A/fed03.md` questions 1 and 3). **SAME**.
- *Splitting the budget evenly scores lower.* **CHANGED / wording reversed**: see section 0, item 5.
  - The even split's RMSE is below one selection's (better) on caco2 dnn, caco2 svm and logd dnn. On logd svm (one finished seed) it is below in 2 of 6 configurations.
  - The width caveat (two ways at once) stands.
- *fed02: one hybrid over five site representations beats best single site and best single representation, Caco-2 and LogD, both models; beats FLuID except Caco-2 dnn, 0.2720 against 0.2858.*
  - `R/federated/fed02_rep_diversity_silos/caco2.json` `per_model.dnn.per_seed[0].c3_fluid_quality.rmse` = 0.2720 and `.c4_kirby_hybrid.rmse` = 0.2858. VERIFIED. **SAME**.
  - Also: per-cluster selection (`c6_fluid_within_kirby_across`) = 0.2858 there, a tie with C4.
- *fed01: selection inside each site beats FLuID at every flip level, never beats the same selection on pooled molecules.* PAGE `A/fed01.md`: caco2 only, yes/yes and no/no. **SAME for Caco-2.** LogD **NOT YET RUN** (no finished seed). No `caco2.json`; the page reads `caco2.partial.jsonl`.
- *fed04: per-cluster selection matches global; separate model per cluster is worse; Caco-2 dnn 0.5675 against 0.6567 mean Spearman.*
  - Values VERIFIED in `R/federated/fed04_cluster_aware_importance/caco2.json` `per_model.dnn.per_seed[0].by_slot.hybrid1.aggregate.mean_profile_stability` (0.5675) and `..._null` (0.6567).
  - The outline's "at 500 features" is from the page. The raw field I read is the aggregate over budgets for hybrid1. UNVERIFIED that it is the 500-feature value.
  - "Matches": per-cluster features above global on caco2 yes/yes and logd no/yes, by at most 0.0088 MCC (PAGE). **SAME**.
  - logd has no `logd.json`, only partial files.
- *fed05 not run.* **SAME** (`R/federated/fed05_antibody_federated/` holds caco2.json and logd.json marked skipped; the antibody job lived inside lhasa_herg, which is cut).
- *Methods 5.3.5 fed03 Part A* (Lhasa hERG sites, C4/C8/C9): **NOT YET RUN and cannot run** while lhasa_herg is cut. The C8/C9 width CHECK in the outline then describes arms with no results.

## 4. Missing or incomplete results, per experiment (Chapter 5 experiments)

Live datasets counted: 25. That is the 26-row roster minus herg_ic50, which runs only the repeat-measurement experiments. "Missing" means no `<ds>.json` and no partial file in `R/complexity/<exp>/`. "Partial" means a partial file only. All 5 outer folds unless stated. VERIFIED by listing the folders.

| experiment | missing | partial only |
|---|---|---|
| c02 hybrid against single | none | flab_binding_koenig (fold 0) |
| c06 capacity | adme_hlm | flab_expression_koenig, flab_binding_koenig |
| c07 easy and hard halves | qm9, adme_hlm (ginkgo x5 skipped by design: 5 groups) | none |
| c09 per-budget | adme_hlm, adme_rlm, flab_expression_koenig, flab_binding_koenig, abagym_Ang2 | none |
| c11 ensembles | adme_hlm, adme_rlm, flab_binding_koenig, abagym_Ang2, ginkgo_hic | none |
| c15 | adme_hlm, adme_rlm, adme_mdr1, flab_expression_koenig, flab_binding_koenig, abagym_Ang2, ginkgo_hic | none |
| c16 | adme_rlm, flab_binding_koenig, abagym_Ang2, ginkgo_hic | none |
| c17 feature reduction | adme_hlm, adme_rlm, flab_binding_koenig, abagym_VEGF, abagym_Ang2, ginkgo_hic (4 classification sets skipped by design) | none |
| c18 depth | adme_hlm | flab_expression_koenig (2 folds), flab_binding_koenig (3 folds) |
| c19 weighting | adme_hlm | none |
| c21 extra features | adme_hlm, adme_rlm, flab_binding_koenig | none |
| c22 explainability | none | logd (2 folds), adme_rlm, flab_expression_koenig |
| c23 evaluator | adme_hlm, adme_rlm, adme_mdr1, flab_binding_koenig, abagym_VEGF, abagym_Ang2, ginkgo_hic | qm9 (3 folds), flab_expression_koenig |
| c24 one fold fewer, stability | qm9, adme_hlm, adme_rlm, adme_mdr1, flab_binding_koenig, abagym_Ang2, ginkgo_hic | none |
| c33 wide hybrids | qm9, adme_hlm, adme_rlm, adme_mdr1, flab_expression_koenig, flab_binding_koenig, abagym_VEGF, abagym_Ang2, ginkgo_hic | none |
| opig_qsar | 3 ChEMBL targets done, 5 folds | none |
| opig_shared_vectors_remedy | only herg_ki | none |
| opig_shared_vectors_census | none finished | partial, 10 datasets up to ddg_combined |
| opig_blueprint_transfer | only logd to herg_ki | none |
| opig_ginkgo | titer, pr_cho, acsins, colloidal | ginkgo_hic, 4 of 5 folds |
| fed01 | logd (no finished seed) | caco2 (partial file only: 3 dnn seeds, 2 svm seeds) |
| fed02 | none | none (caco2, logd) |
| fed03 | Part A (lhasa_herg, cut) | none for Part B (caco2, logd); logd svm 1 finished seed (PAGE) |
| fed04 | none | logd (partial only) |
| fed05 | everything (lives inside the cut lhasa_herg job) | none |

Pattern:
- adme_hlm, adme_rlm, abagym_Ang2, flab_binding_koenig and ginkgo_hic are missing from most of the experiments 5.4 and 5.5 cite.
- adme_hlm and adme_rlm are waiting on the dnn re-run (FAULTS A1).
- The outline names "human and rat microsomal clearance" and "Angiopoietin-2" as results in 5.4 and 5.5. Those results do not exist yet.

Files in `R/complexity/` that belong to cut datasets (bh_doyle_plates, tz_her2_binary, tz_her2_multiclass, nt_promoter_tata) are present in several experiment folders. Leave them out of every table.

## 5. Result files the outline cites from tests/results or tests/experiments/results

- **f24 augmentation admission** (5.5a): `tests/experiments/results/framework/f24_augmentation_admission_*.json`. **Exists.** Hybrid files for buchwald_hartwig, caco2, flab_expression_koenig and tcga_brca_pam50. One single-representation file for herg_ki. Timing-only files for abagym_VEGF, abcov_kd, flab_thermo_tresanco and logd. Files dated 7 Sept. 3 folds, 3 seeds. VERIFIED.
- **c32 allocation cross** (5.2.3): `/Volumes/seagate/kirby_cache/c32_allocation_cross/`. **Exists**, 30 files dated 29 Aug.
  - 10 dataset names: adme_hlm, adme_mdr1, adme_rlm, bh_doyle_plates (now cut), buchwald_hartwig, caco2, flab_expression_koenig, herg_ki, logd, qm9.
  - Models catboost, lightgbm and rf.
  - The outline says 9 datasets and 54 cells. Which dataset is left out was not checked.
  - Not in `tests/results` or `tests/experiments/results`.
- **f33 pruning** (5.2.3, 5.2.5): `tests/results/framework/f33_pruning/{flab_thermo_tresanco, herg_ki, humanness_therapeutic, logd, tcga_brca_pam50}.json`. **Exist**, dated 12 to 14 Sept, plus a `tcga_brca_pam50__smoke.json` and five `ANALYSIS_*.txt`. Contents not re-checked this session.
- **f26 redundancy** (5.3.1 "best single representation"): `tests/experiments/results/f26_redundancy_penalty_*.json`. **Exist** (buchwald_hartwig, flab_thermo_tresanco, ginkgo_hic, and seven herg_ki variants). The outline already says they use pools from before pruning. Replace with c02 (section 2).

## 6. Data for F2a and F2b (best number of features, best representation count)

Source: c02 `best_by_model[model]`, the configuration with the best mean training-row score over the 5 outer folds, and `standard_selection_summary[model].hybrid1.per_fold[i].name`, the hybrid each fold picked. In the name, the 3rd field is the budget and the 6th the representation count; `None` means all representations in the pool. VERIFIED.

| dataset | model | budgets searched (features) | best budget, mean over folds | budget picked on each fold | representation count picked on each fold |
|---|---|---|---|---|---|
| qm9 | dnn | 500, 1000, 2000 | 1000 | 2000, 1000, 1000, 1000, 1000 | all, all, 3, all, all |
| qm9 | lightgbm | 500, 1000, 2000 | 1000 | 1000, 2000, 1000, 500, 1000 | 3, 3, 3, 3, 3 |
| herg_ki | svm | 250, 500, 1000, 2000 | 500 | 1000, 250, 250, 500, 500 | all, all, all, all, all |
| herg_ki | dnn | 250, 500, 1000, 2000 | 2000 | 2000, 250, 2000, 250, 2000 | all, all, all, all, all |
| caco2 | dnn | 500, 1000, 2000 | 2000 | 2000, 2000, 2000, 1000, 1000 | all, all, all, all, all |
| caco2 | svm | 500, 1000, 2000 | 500 | 500, 500, 500, 500, 1000 | all, all, all, all, all |
| logd | dnn | 500, 1000, 2000 | 2000 | 2000, 2000, 2000, 2000, 2000 | all, all, all, 5, all |
| logd | svm | 500, 1000, 2000 | 1000 | 1000, 1000, 1000, 1000, 2000 | all, all, all, all, all |
| adme_hlm | svm | 500, 1000, 2000 | 1000 | 500, 1000, 500, 1000, 1000 | all, all, all, all, all |
| adme_hlm | dnn | 500, 1000, 2000 | 2000 | 2000, 2000, 2000, 2000, 2000 | all, all, all, all, all |
| adme_rlm | svm | 500, 1000, 2000 | 2000 | 2000, 2000, 2000, 2000, 1000 | all, all, all, all, all |
| adme_rlm | dnn | 500, 1000, 2000 | 1000 | 2000, 2000, 2000, 1000, 1000 | all, all, all, all, all |
| adme_mdr1 | svm | 500, 1000, 2000 | 1000 | 2000, 1000, 1000, 2000, 500 | all, all, all, all, all |
| adme_mdr1 | dnn | 500, 1000, 2000 | 2000 | 2000, 2000, 2000, 1000, 2000 | all, all, all, all, all |
| herg | lightgbm | 100, 250, 500, 1000, 2000 | 500 | 2000, 2000, 2000, 1000, 2000 | 3, 3, all, 3, 3 |
| herg | xgboost | 100, 250, 500, 1000, 2000 | 2000 | 1000, 500, 2000, 2000, 1000 | all, 3, all, 3, 3 |
| flab_expression_koenig | svm | 250, 500, 1000, 2000 | 2000 | 2000, 2000, 2000, 2000, 2000 | all, all, all, all, all |
| flab_expression_koenig | dnn | 250, 500, 1000, 2000 | 2000 | 2000, 2000, 2000, 2000, 2000 | all, all, all, all, all |
| flab_thermo_tresanco | dnn | 50, 100, 250, 500, 1000, 2000 | 1000 | 1000, 1000, 1000, 1000, 250 | all, 3, 5, all, 5 |
| flab_thermo_tresanco | svm | 50, 100, 250, 500, 1000, 2000 | 1000 | 250, 250, 250, 500, 500 | all, 2, all, all, 5 |
| abagym_EGFR | svm | 250, 500, 1000, 2000 | 2000 | 2000, 2000, 2000, 2000, 250 | 3, all, all, 5, all |
| abagym_EGFR | dnn | 250, 500, 1000, 2000 | 250 | 2000, 500, 1000, 2000, 250 | 3, all, all, all, all |
| abagym_VEGF | gp | 10, 25, 50, 100, 250, 500 | 50 | 100, 250, 50, 250, 50 | 5, 5, all, 3, all |
| abagym_VEGF | svm | 10, 25, 50, 100, 250, 500 | 100 | 250, 500, 50, 50, 50 | all, 5, all, all, all |
| abagym_Ang2 | gp | 25, 50, 100 | 100 | 100, 100, 100, 100, 25 | 3, all, 5, 3, all |
| abagym_Ang2 | svm | 25, 50, 100 | 100 | 100, 100, 100, 100, 25 | all, all, 3, 5, all |
| humanness_therapeutic | xgboost | 10, 25, 50, 100, 250, 500, 1000, 2000 | 1000 | 250, 1000, 500, 50, 500 | 2, 2, 5, 5, 3 |
| humanness_therapeutic | rf | 10, 25, 50, 100, 250, 500, 1000, 2000 | 1000 | 250, 100, 1000, 1000, 250 | 3, 3, all, all, 2 |
| ddg_combined | svm | 100, 250, 500 | 100 | 100, 500, 100, 100, 100 | 3, 5, all, all, all |
| ddg_combined | dnn | 100, 250, 500 | 100 | 250, 100, 100, 100, 250 | 3, all, 3, 3, all |
| cdr_flexibility | xgboost | 10, 25, 50, 100 | 50 | 100, 25, 25, 100, 10 | 2, 3, 3, 5, 2 |
| cdr_flexibility | lightgbm | 10, 25, 50, 100 | 100 | 50, 25, 25, 25, 10 | 2, 3, 3, 5, 3 |
| abcov_kd | lightgbm | 50, 100, 250, 500, 1000, 2000 | 2000 | 100, 100, 250, 2000, 50 | 2, 5, 3, 5, 2 |
| abcov_kd | xgboost | 50, 100, 250, 500, 1000, 2000 | 2000 | 100, 50, 250, 500, 2000 | 3, 3, 3, all, all |
| ginkgo_colloidal | dnn | 500, 1000, 2000 | 2000 | 500, 500, 2000, 2000, 2000 | all, all, all, 5, all |
| ginkgo_colloidal | svm | 500, 1000, 2000 | 2000 | 500, 1000, 500, 2000, 2000 | 5, 5, 3, all, 3 |
| ginkgo_hic | gp | 250, 500, 1000, 2000 | 1000 | 250, 500, 500, 250, 500 | all, 5, 5, all, 5 |
| ginkgo_hic | dnn | 250, 500, 1000, 2000 | 1000 | 1000, 500, 1000, 500, 1000 | all, all, all, all, all |
| ginkgo_acsins | lightgbm | 50, 100, 250, 500, 1000, 2000 | 500 | 2000, 2000, 100, 500, 500 | all, 5, all, all, 5 |
| ginkgo_acsins | gp | 50, 100, 250, 500, 1000, 2000 | 1000 | 250, 250, 1000, 500, 2000 | 2, all, 5, all, all |
| ginkgo_titer | dnn | 250, 500, 1000, 2000 | 250 | 500, 1000, 500, 2000, 500 | 5, all, 2, all, 2 |
| ginkgo_titer | rf | 250, 500, 1000, 2000 | 250 | 250, 2000, 250, 1000, 2000 | 2, all, 2, all, 5 |
| ginkgo_pr_cho | gp | 250, 500, 1000, 2000 | 2000 | 2000, 2000, 500, 2000, 1000 | all, 5, all, all, 5 |
| ginkgo_pr_cho | dnn | 250, 500, 1000, 2000 | 2000 | 2000, 250, 500, 2000, 500 | 5, all, all, all, all |
| buchwald_hartwig | rf | 500, 1000, 2000 | 500 | 500, 1000, 500, 500, 1000 | 2, 2, 2, 2, 2 |
| buchwald_hartwig | lightgbm | 500, 1000, 2000 | 500 | 1000, 1000, 500, 1000, 2000 | 2, 2, 2, 2, 2 |
| tcga_brca_pam50 | dnn | 25, 50, 100, 250, 500, 1000, 2000 | 1000 | 250, 500, 500, 2000, 250 | 3, 3, all, 3, 3 |
| tcga_brca_pam50 | svm | 25, 50, 100, 250, 500, 1000, 2000 | 250 | 100, 50, 100, 100, 500 | all, 3, 3, 3, 3 |

Reading:
- Small-molecule budgets stop at 2,000 features. LogD dnn, adme_hlm dnn and flab_expression_koenig (both models) picked 2,000 on every fold, the top of the grid.
- The training-row best (column 4) and the per-fold picks often disagree. Example: herg_ki dnn's mean-best budget is 2,000, but folds 1 and 3 picked 250.
- Representation count: buchwald_hartwig picks 2 on every fold with both models; qm9 lightgbm picks 3 on every fold. Most small-molecule folds pick all representations.
