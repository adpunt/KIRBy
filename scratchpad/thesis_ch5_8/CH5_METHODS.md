# Chapter 5 methods: the experiments, read from the code (2026-10-09)

Written by Claude on 2026-10-09 for THESIS_OUTLINE.md 5.3 and thesis.tex "Methods: experimental setup".
Read-only. Nothing in thesis.tex, the outline, the code or the answer pages was changed.

Tags:
- VERIFIED means I read it in the code or in a raw result file in this session.
- UNVERIFIED means I saw it only in a document (outline, FAULTS.md, an answer page).
- Result files are under `/Volumes/seagate/kirby_results_2026-10-09/`, written R/ below.
- Code paths are relative to `/Users/apunt/repos/KIRBy/`.
- Fold counts say how many outer folds a number covers.

---

## 0. What changes the outline or thesis.tex (read this first)

Each item is one disagreement. The fix is not proposed here.

1. **The federated hERG experiment cannot run.** The outline (5.3.5) and thesis.tex (sec:method:sites, about lines 2775-2812) describe three real hERG sources as sites. That is fed03 Part A, which runs only on `lhasa_herg`. `lhasa_herg` was cut on 2026-10-08 (`tests/experiments/orchestrate/plan.py:74-80`). R/federated/fed03_synthetic_private/ holds only caco2 and logd, which run Part B (three representation families as sites). VERIFIED. So the hERG paragraphs, diagram D5.4 and the C8/C9 CHECK describe an experiment with no result and no planned run.
2. **The antibody federated experiment (fed05) cannot run either.** Its caco2 and logd files say "computed once in the lhasa_herg job" and are skip records (R/federated/fed05_antibody_federated/caco2.json, field `reason`). VERIFIED. With lhasa_herg cut, fed05 has no job. thesis.tex still lists it as the fourth separate-site experiment.
3. **The grid in decided_grid.json is not the grid most stored results ran on.** The representation-count cuts landed on 7 Oct at 22:20-22:35 (commits 9491e39, a793f65). Most c02 files were written 6-7 Oct. Example: logd's c02 file scored 252 configurations per fold with representation counts 2, 3, 5 and all (R/complexity/c02_complexity_matching_grid/logd.json, `configs`, 5 folds). decided_grid.json now gives logd 63 (counts: all only). thesis.tex tab:decided_grid prints the current 63. VERIFIED. Seven of 27 c02 files differ (table in section 3).
4. **Outline 5.3.4 is stale for six of its eight complexity experiments.** c11 no longer uses fold reselection; c17 no longer uses one 100-tree forest; c19 is no longer fixed at budget 100 or on rf, lightgbm and xgboost; c23 has no rf_exact condition; c24 measures more than stability; c21's extra-feature arms run on two datasets only. Details in section 2. thesis.tex sec:method:comparators was corrected on 2026-10-08 and mostly matches the code.
5. **The outline's c19 citation note is wrong about the weight.** Outline line ~1290 says c19's molecule weight is n02's 1/(1 + 3 x gap). c19 computes 1 - |y_i - mean of 10 neighbours| / (max y - min y), clipped to [0.1, 1] (`tests/experiments/complexity/c19_importance_weighted_training.py:100-126`). VERIFIED. thesis.tex states the c19 formula correctly.
6. **opig_blueprint_transfer shares 7 representations, not 6, and uses 2 models, not 5.** Outline 5.3.5 says atom_pair, avalon, ecfp4, maccs, mordred, pdv and 5 downstream models. The result file says avalon, chembert, ecfp4, grover, mhg_gnn, mordred, smited, and models svm and dnn (R/opig/blueprint_transfer/opig_blueprint_transfer/logd_to_herg_ki.json, `shared_reps`, `models`). VERIFIED. thesis.tex already says seven. The seven include chembert, whose cache memory says was never checked against its builder (UNVERIFIED).
7. **Two outline 5.4/5.5 verdicts name datasets with no result in the experiment cited.** c11 has no adme_hlm, adme_rlm or abagym_Ang2 file, yet the outline says the hybrid beats both ensembles on human microsomal clearance and loses on Angiopoietin-2. c17 has no adme_hlm or adme_rlm file, yet the outline says it beats the alternatives on human and rat microsomal clearance. VERIFIED by directory listing. The rebuilt c11 page also shows n/a for both rows.
8. **mRMR is absent from every small-molecule comparison.** c17 caps mRMR at 200 features (`c17_information_theoretic_decomposition.py:577`). Every small-molecule dataset's decided budgets start at 250 or 500. So on Caco-2, LogD, hERG Ki, MDR1 and QM9, c17 compares the hybrid against PCA, RFE and random only. mRMR ran on 4 of 15 regression datasets (abcov_kd, ddg_combined, flab_thermo_tresanco, ginkgo_acsins), per each file's `mrmr_budgets_served`. VERIFIED.
9. **The 3-fold limit does not describe the stored results.** thesis.tex says four experiments run only the first three outer folds except on hERG Ki and trastuzumab. The code limits ten (c05, c06, c07, c10, c13, c15, c23, c33, n15, n26) and exempts 13 datasets (`tests/experiments/_shared.py:3929-3941`). The limit arrived 6-7 Oct, so most stored files already had 5 folds. c33 has 5 folds on 15 of 16 datasets and 3 on humanness_therapeutic; c06 has 3 on adme_rlm and qm9 and 5 elsewhere. VERIFIED (each file's `per_fold` length or `n_outer_folds_run`).
10. **Held-out features are clipped to the training range in six experiments and not in the others.** c01, c02, c05, c07, c08 and c18 clip (`c02_complexity_grid.py:127`, `c07:96`, `c05:90`, `c18:307`). c11, c17, c19, c21, c22, c23, c24, c33 and the OPIG and federated scripts do not (grep, VERIFIED). FAULTS.md A8 lists only c01, c07 and c08, so it undercounts. thesis.tex states no rule.
11. **Stored dnn and gp results straddle two code fixes.** dnn predictions are clipped to the training-label range since c18ee30 (8 Oct 13:04). The downstream gp starts its length scale at sqrt(features) since ffcd254 (8 Oct 11:03); the gp internal evaluator since 60d635a (8 Oct 15:59). Per-experiment counts of files with a `timestamp` after 8 Oct 13:04 are in section 4. A timestamp after the commit does not prove the code version, because folds resume from partial files and the server pull time is not recorded. UNVERIFIED which code each file ran.
12. **thesis.tex's dataset table and grid table list five cut datasets and omit fifteen live ones.** Section 6.
13. **f24 (the decoy test of 5.5a) is local, dated 7 Sep, and not in git.** Files `tests/experiments/results/framework/f24_augmentation_admission_*.json`, mtime 7 Sep 2026, `git log` returns nothing for them. VERIFIED. They predate the 18 Sep representation-cache fix, the 28 Sep removal of the admission gate (920bbb8) and every 5-6 Oct alignment change. Models are rf and lightgbm, not each dataset's two decided models.

Outline numbers I re-read and found SAME are listed per experiment in section 2. None of those I checked had changed.

---

## 1. thesis.tex Chapter 5, subsection by subsection

Chapter runs from line 1821 (`\chapter{Building Adaptive Hybrid Representations}`) to 3602. 137 lines in it carry `% TODO` or `[SPLIT`. Status words: current = matches the code I read; outdated = was true earlier; contradicts = disagrees with code now.

| Line | Heading | Status | What disagrees |
|---|---|---|---|
| 1832-1918 | Introduction and its four subsections | not checked against code | Background prose. |
| 1919 | Methods: the hybrid search | current | Rewritten recently. |
| 1931 | From base representations to a hybrid | mostly current | "Under label noise, training quantities use noisy labels while held-out scoring uses clean labels unless." The sentence is cut off. It also points to Chapter 6 (chap:similarity) for the noise variant; noise is now Chapter 7. The 5,000-feature pre-screen it states exists (`find_best_hybrid.py:4321`, HIGH_DIM_FILTER_THRESHOLD = 5000). `% TODO: double check this is an accurate description of what's happening with large reps`. `% TODO: Figure 5.1`. |
| 1979 | Blueprints (paragraph) | current | Field list not re-checked line by line. |
| 2026 | The search space | current | Two author TODOs: "am I keeping nanobody thermostability??" and [RESULT NEEDED] for best budget and best representation count. |
| 2042 | Feature importance scoring | not checked | |
| 2067 | Feature allocation methods | not checked | Three `% TODO: review` lines. |
| 2128 | Search configuration and internal evaluation | current | "8,160 configurations with five downstream models" recomputes exactly (section 3). "63 to 3,360" is true of the 16 rows in its table, not of the live roster: abcov_kd has 2,520 and the live maximum is still humanness 3,360. |
| 2162 | Augmentation | current for the general search | tab:aug_conditions says uncapped runs on "none". The decided grid keeps uncapped on the distance search of ginkgo_colloidal and ginkgo_hic (`_shared.py` search_on_fold comment, decided_aug_conditions), and c21 ran its extra-feature arms there. |
| 2219 | Search grid per dataset (tab:search_tiers, tab:decided_grid, tab:cut_walkthrough) | contradicts | Rows include five cut datasets. Live datasets missing. Configuration counts are the current decided_grid.json, not what most stored results ran (section 3). Fast tier named as four datasets; among live datasets the fast class is qm9 and herg (decided_grid.json `speed`). |
| 2312 | Adaptive pruning | current | Says pruning was not used. Code: every per-fold search runs unpruned (`_shared.py:1410-1428`). VERIFIED. |
| 2316 | Methods: experimental setup (opening) | contradicts | 3-fold sentence (item 9 above). "Chapter~\ref{chap:similarity} reuses this protocol with noise added" now belongs to Chapter 7. The 2+2 rule is current (`_shared.py:1697-1698`). |
| 2332 | Datasets and splitting (tab:datasets) | outdated | "16 datasets from six domains". Section 6. |
| 2410 | Representations by domain | outdated | Mentions H3K4me3 and TATA (both cut). [RESULT NEEDED] for most and least alike pairs. |
| 2659 | Models and metrics | mostly current | "Across the 16 datasets, all but the Gaussian process are used." On the live roster gp is decided on five datasets (abagym_Ang2, abagym_VEGF, ginkgo_acsins, ginkgo_hic, ginkgo_pr_cho; decided_grid.json). Contradicts. MCC paragraph for nucleotides is now about cut datasets. kNN paragraph says "every dataset except the two nucleotide ones": both are cut. |
| 2691 | Comparators, controls and ablations | mostly current | Matches c33, c11, c17, c23, c24, c05, c19 code except: does not say c24's main measurement (own fold against one fold fewer); does not say mRMR is absent at small-molecule budgets; says nothing about c18. Contains the c10 noise paragraph, which moves to Chapter 7. |
| 2763 | Transfer, mutational scans and separate sites | contradicts | Blueprint transfer: matches code. Mutational scans paragraph: opig_scaffold_hybrid, not checked here. Federated: the hERG design (Part A) cannot run (item 1); budgets 100-2,000 and LightGBM/XGBoost are lhasa_herg's, while the runs that exist (caco2, logd) use 500-2,000 and dnn/svm. The fed05 sentence describes a job that cannot run (item 2). |
| 2818-3532 | Results sections | not in scope | |
| 3533-3602 | Discussion, Limitations, Conclusions | not in scope | |

The `% [notes moved to thesis_notes_moved.md]` comment sits under every Methods heading. Most `% TODO: review, needs manual review` comments carry a date and a source file; the 2026-10-08 ones were written by an earlier chat.

---

## 2. Each experiment the outline's 5.4-5.8 cites

### 2.0 What every complexity experiment shares (code, VERIFIED)

- Outer folds: `kirby.find_best_hybrid.nested_cv_outer_folds` (`find_best_hybrid.py:10439`), seed 42, count from `config['n_splits']`. `--mode full` gives DEFAULT_N_FOLDS = 5; intermediary gives 1 (`_shared.py:9999-10000`). Groups whole; classification stratified.
- One search per outer fold, on that fold's training molecules only: `_shared.fold_standard_selection` -> `fold_searches` -> `search_on_fold` (`_shared.py:1329-1808`). Pruning off in every per-fold search (`_shared.py:1410-1428`).
- The grid of that search is the dataset's row of `tests/experiments/decided_grid.json`, applied by `decided_grid.apply_decided_grid` (`decided_grid.py:84`). A dataset with no row raises; there is no fallback.
- Inside the search: 5 inner folds (DEFAULT_N_FOLDS, `find_best_hybrid.py:3741`). Inside each inner training part, an 80/20 group-aware split: importance on the 80%, each representation's standalone score on the 20%.
- Standard selection: per decided model, the 2 best single representations (ranked by the search's inner-fold mean, `search.baselines`) and the 2 best hybrids over every budget (`STANDARD_N_BASE = 2`, `STANDARD_N_HYBRIDS = 2`, `_shared.py:1697-1698`). A second hybrid whose selected features equal the first's is skipped for the next distinct one. Each is rebuilt on the outer training molecules and scored once on the held-out fold.
- Models: the dataset's two decided models (decided_grid.json `models`). kNN only on the distance search.
- Feature scaling: `kirby.scaling.FeatureScaler`; 0/1 features left unscaled, others standardised on the training molecules.
- Augmentation: none on the general search of every dataset.
- No fold reselection, except c16 (`c16_feature_selection_as_regularization.py:826`) and c12.

### c02 complexity grid (`tests/experiments/complexity/c02_complexity_grid.py`)

- Question (docstring 2-31): do different models have different best budgets?
- Per outer fold: one search; per model, the best training-row score at each decided budget (the model x budget heatmap), the best configuration, which importance method it used, and the standard selection scored held out. Plus PCA of all representations joined, to each decided budget.
- Held-out features clipped to the training range (`:127`).
- Not run, by decision 2026-10-06: fine-tuned representations; a search at the best single representation's own width; capped augmentation (`ARMS_NOT_RUN`, `:69-82`).
- Results: 27 files, all `n_outer_folds_scored` 5. flab_binding_koenig has a partial file only. VERIFIED.
- Outline 5.4 "LogD network 0.4497 to 0.4089 RMSE, support vector regressor 0.4930 to 0.4302": SAME. R/complexity/c02_complexity_matching_grid/logd.json, `standard_selection_summary.<model>.base1.heldout.mean` and `.hybrid1.heldout.mean`, 5 folds. VERIFIED.
- Outline 5.4 "wins for both models on Caco-2, LogD, hERG Ki, hERG and the three ADME sets; loses for both on nanobody thermostability, SMAC colloidal stability and HIC retention": SAME on all ten datasets, same fields, 5 folds each. VERIFIED. QM9 also wins for both models (dnn 0.3619 to 0.3155, lightgbm 0.3720 to 0.3463 RMSE); the outline does not mention it.

### c22 explainability (`complexity/c22_explainability.py`)

- Question (2-48): can each selected feature be traced to its representation, and which representations carry the importance?
- Per fold: standard selection; each hybrid explained with SHAP: TreeExplainer on tree models, DeepExplainer on dnn (100 training molecules as background), KernelExplainer on 50 fixed held-out molecules for svm and gp (`:97-131`). Plus grouped permutation importance: one representation's features shuffled together, 5 repeats (`PERM_REPEATS`, `:88`).
- Shares per representation are computed inside each fold, then averaged over folds.
- Results: 24 json files plus partials (adme_rlm, flab_expression_koenig, logd partial only). VERIFIED by listing. Fold counts not read.

### c18 diversity against depth (`complexity/c18_diversity_vs_depth.py`)

- Question (4-6): at a fixed budget, is it better to draw features from many representations or few?
- Arms per model, per decided budget, per fold (docstring 9-26; code 430-627):
  - depth_1, depth_2, depth_3: all features from the top 1, 2 or 3 representations. Representations ranked by summed importance on the fold's training molecules.
  - natural, natural__general2: the search's 2 best hybrids at this budget, rebuilt.
  - uniform_all: equal split over the representations natural used, to natural's built width (decided 2026-10-06).
  - uniform_every_rep: equal split over every general representation, filled to the budget (back 2026-10-07).
  - The 2 best single representations, beside the arms.
- What is held fixed: the importance method. The hand-built arms use hybrid 1's method on that fold (`:435-449`). VERIFIED.
- What still differs between depth arms and natural, two things: depth arms fill the budget past the last feature with importance above zero, and natural stops there (`:285-307` comment); depth arms rank representations by summed importance, while natural ranks them by the internal evaluator's standalone score. The built widths are stored (`width_built_per_fold`). VERIFIED. On herg_ki svm at 500 features all arms built 500 features on all 5 folds, so width does not differ there.
- Held-out features clipped (`:307`).
- Outline 5.4 "hERG Ki svm at 500 features: 0.6006 to 0.5672 RMSE": SAME. R/complexity/c18_diversity_vs_depth/herg_ki.json, `per_model.svm.per_budget.500.depth_1.score` 0.6006 and `.depth_3.score` 0.5672, 5 folds. VERIFIED. Same cell: natural 0.5593, uniform_every_rep 0.5569.

### c11 ensembles against the hybrid (`complexity/c11_ensemble_vs_hybrid.py`)

- Question (4-9): does one model on the hybrid match an ensemble of one model per representation, at lower standing cost?
- Arms (50-75): base_top1/2 and hybrid_top1/2 (standard selection); best_single (picked by its mean over the held-out folds); pca_matched (all representations joined, PCA to each decided budget, `:930-990`); ensemble style 1 (one model on every representation of a set) and style 2 (each representation's own best model); `_auto` variants that also pick membership and aggregator; oracle_single (best model and representation pair read off each held-out fold).
- Ensemble choices on an inner 3-fold group split of the outer training molecules (`INNER_FOLDS = 3`, `:180`). Membership top 3 or all. Aggregators: mean, median, trimmed20, iqr_filtered, mad_filtered, weighted, topk_mean; majority_vote for hard labels.
- Head-to-head reads hybrid_top1, base_top1 and ensemble_s1_auto, all chosen on training molecules.
- Cost fields: featurisers, models deployed, bytes, fit and predict seconds.
- Disagrees with outline 5.3.4: no fold reselection (removed 2026-10-05, docstring 102-105); PCA is to each budget, not to the hybrid's width; oracle_single is not an upper bound (docstring 68-73 gives nt_H3K4me3, ensemble 0.672 against oracle 0.659 MCC). thesis.tex already says all three correctly.
- Results: 22 files, `n_outer_folds` 5. No file for adme_hlm, adme_rlm, abagym_Ang2, ginkgo_hic or flab_binding_koenig. VERIFIED.
- Outline 5.4 verdicts: "beats both on human microsomal clearance": NOT FOUND (no adme_hlm file). "Both ensembles beat it on Angiopoietin-2 binding": NOT FOUND (no abagym_Ang2 file). hERG Ki svm, PAM50, anti-coronavirus, CHO polyreactivity: not re-checked.

### c07 easy and hard halves (`complexity/c07_data_complexity_partitions.py`)

- Question (2-4): does the best configuration depend on how hard a subpopulation is?
- Datasets over 2,000 molecules are cut to about 2,000 by whole scaffold groups, seed 42 (`C07_MAX_SAMPLES`, `:70`, `:411-430`).
- Four partitions, each into two halves (`:488-583`): scaffold frequency (scaffolds at or above the 80th percentile of group size against the rest); heavy-atom count at the median; label variance within scaffold at the median (uses labels; flagged `partition_uses_labels`); feature spread within scaffold at the median. A half under 50 molecules is skipped.
- One shared fold layout on the capped rows. Per fold: the half's own search (standard selection) and an all-rows search shared by every half, its hybrids rebuilt on the half.
- 3-fold limit applies (OUTER_FOLDS_RUN c07 = 3) for files written after 7 Oct; logd's file has 5.
- Held-out features clipped (`:96`).
- Outline 5.4 "LogD hard half by scaffold frequency, network: 0.6698 single against 0.5735 RMSE": SAME. R/complexity/c07_data_complexity_partitions/logd.json, `partitions.scaffold_frequency.hard.heldout_by_model.dnn.base1.mean` 0.6698 and `.hybrid1.mean` 0.5735, 5 folds. VERIFIED. The methods text must say this is a 2,000-molecule subset of LogD (`n_rows_searched` 2000) and the hard half is 649 molecules (`n_hard`).
- FAULTS.md A13 (UNVERIFIED): abcov_kd and cdr_flexibility files say ok with zero partitions.

### c33 budget overfitting (`complexity/c33_budget_overfitting.py`)

- Question (2-21): is a budget above the molecule count a real gain or an over-promise by the search?
- One nested run per model: `nested_cv_evaluate` (`find_best_hybrid.py:10613`), per-fold search on the decided grid except budgets. Budgets are c33's own: FULL_BUDGETS (10-2,000), plus 4,000 and 8,000, plus 0.5, 0.75, 1, 1.25, 1.5 and 2 times the smallest outer training set. herg_ki: 16 budgets, smallest training set 1,129 (R/complexity/c33_budget_overfitting/herg_ki.json, `budgets`). VERIFIED.
- Bias = inner score minus held-out score, positive when the search over-promised. Also: the budget each fold picked, against the budget with the best held-out score.
- Two decided models. The near-unpenalised linear model was removed 2026-10-05.
- The docstring (55-80) says dnn, svm and rf are past their own interpolation threshold at every budget, so a flat curve is not evidence against double descent.
- Results: 16 files. 5 folds on 15, 3 on humanness_therapeutic. VERIFIED.
- Outline 5.4 titer/colloidal/herg_ki/logd verdicts: not re-checked here.

### c24 "blueprint transfer" across folds (`complexity/c24_blueprint_transfer.py`)

- What it measures (docstring 10-28): nothing is transferred. Diagonal cell = the standard per-fold search, trained on every fold but the target. Off-diagonal cell = a fresh search on the rows outside both the target and a source fold. The two differ in training size, by one fold. Molecule counts are stored per cell.
- Also Nogueira stability (`:171-260`): d = features in all representations joined (`total_features`, `:359`), one selected set per fold, per model.
- Results: 18 files, 5 folds each. VERIFIED.
- Outline 5.4 "one fold fewer: hERG Ki svm 0.5596 to 0.5931": SAME. My mean of the diagonal against the off-diagonal cells of R/complexity/c24_blueprint_transfer/herg_ki.json `by_model.svm.transfer_matrix`: 0.5596 and 0.5931, 5 target folds. VERIFIED (my arithmetic).
- Outline 5.6 "the agreement coefficient reaches 0.4 only on Caco-2 svm, hERG LightGBM, LogD both": SAME. `by_model.<m>.stability.nogueira_phi`: caco2 svm 0.424, herg lightgbm 0.447, logd dnn 0.420, logd svm 0.497. caco2 dnn is 0.399. All others below 0.33. VERIFIED.
- Outline 5.3.4 says only "c24, stability"; the fold-fewer measurement needs its own sentence.

### c17 against PCA, mRMR, RFE and random (`complexity/c17_information_theoretic_decomposition.py`)

- Question (4-6): at the same width, does the hybrid beat variance-based, mutual-information-based, model-based and random selection?
- Per fold, per decided budget, per model: the 2 best hybrids at that budget; PCA on the joined, standardised representations; mRMR (KSG mutual information, pre-filter to 2,000 features, ranking on at most 2,000 molecules, stops at 200 features, `:542`, `:577`, `:629`); RFE with a 50-tree, depth-5 forest, 10% of features removed per step (`:333-334`); random features. The four non-search methods choose once per fold and budget, then each model scores them.
- Regression only (`:441-447`). Skip files for cdr_flexibility, herg, humanness_therapeutic, tcga_brca_pam50. VERIFIED.
- Disagrees with outline 5.3.4: two decided models, not one 100-tree forest; per-fold search, not a recipe chosen on all rows.
- mRMR coverage: item 8 above.
- Outline 5.5 "LogD at 1,000 features, network: 0.4099 against 0.4146 RMSE for recursive elimination": SAME. R/complexity/c17_information_theoretic_decomposition/logd.json, `summary.heldout_score_per_budget.1000.dnn.hybrid` 0.4099 and `.rfe` 0.4146, 5 folds. VERIFIED. mRMR is NaN there.
- Outline 5.5 "beats all three on human and rat microsomal clearance": NOT FOUND (no adme_hlm or adme_rlm file).

### c15 theory diagnostics (`complexity/c15_ml_theory_depth.py`)

- Four sub-analyses (docstring 10-19): distance concentration (Beyer ratio, pairwise distances), kernel matrix condition number and spectrum, a double-descent probe sweeping budget through n = p with its own search per budget, and an empirical complexity bound.
- 2 best hybrids per model at each decided budget from the fold's search.
- It does not compare against PCA, mRMR or RFE. The outline cites it for that comparison in 5.5; that citation is wrong. VERIFIED from the docstring.
- 3-fold limit applies to files written after 7 Oct.

### c09 formal analysis (`complexity/c09_formal_analysis.py`)

- Sub-analyses (docstring 2-35): 9a variance decomposition over c02's file (no search); 9b-9d information retention, effective rank and bias-variance on each built hybrid and each single representation. Per decided budget, the 2 best hybrids at that budget.
- FAULTS.md A7 (UNVERIFIED): its question 5 compares hybrid, PCA and random features on effective rank, and PCA stops at training molecules minus one.
- The outline cites it in 5.5 for the comparison against feature reduction and in 5.5 for F2a. The first is only true of effective rank.

### c06 capacity within a model family (`complexity/c06_within_family_gradient.py`)

- Question (2-13): does the best budget rise with model capacity within one family?
- Families only where the family's model is a decided model. One search per capacity setting per fold, that setting as the only downstream model. Best budget chosen on training-row scores; held-out score at every budget stored. Spearman between capacity index and best budget.
- 3-fold limit applies to later files (adme_rlm, qm9 have 3).

### c16 feature selection as regularisation (`complexity/c16_feature_selection_as_regularization.py`)

- Question (2-8): at the best budget, does model regularisation stop mattering?
- Families: rf (max_depth sweep) and svm (C sweep) only, where decided. Heatmap: regularisation setting x budget, plus a "capped_full" width = min(pool width, 2 x molecules, 2,048).
- Uses fold reselection (`:826`): the only complexity experiment here that picks across folds' held-out scores.

### c23 internal evaluator against downstream model (`complexity/c23_evaluator_model_matching.py`)

- Question (2-8): must the internal evaluator match the downstream model?
- `nested_cv_evaluate` per model per condition; deltas paired per outer fold.
- Conditions (`MATCH_TABLE`, `:125-142`; `NN_PAIRS`, `:161-163`):
  - A matched.
  - B near family: rf-lightgbm, lightgbm-rf, xgboost-lightgbm, svm-gp, gp-svm, dnn-dnn at (64, 32).
  - C far family: rf, lightgbm and xgboost with svm; svm and gp with lightgbm; dnn with knn.
  - D dnn scored by a wider dnn (512, 256, 128).
  - E the dataset's decided evaluator list.
- Each condition picks its own budget, so widths differ and are stored.
- Disagrees with outline 5.3.4: no rf_exact condition (grep finds none in c23). thesis.tex matches the code.
- svm's near-family evaluator is gp. Stored results written before 8 Oct 15:59 used a gp evaluator with length scale starting at 1 (fixed in 60d635a, `hybrid.py:2991-2999`). FAULTS.md A3 says this is still open; current code has the fix. VERIFIED (code), UNVERIFIED (which files ran which).
- Results: 16 json files, 35-45 per-fold records each.

### c19 importance during training (`complexity/c19_importance_weighted_training.py`)

- Question (4-6): how should importance enter training?
- Per fold, per model: the 2 best single representations and the 2 best hybrids (standard selection), each under five conditions (docstring 22-28): unweighted; features x sqrt(importance), normalised to mean 1 (`:83-95`); molecule weights; both; features strictly above the median importance (`:247-251`).
- Molecule weight: 1 - |y_i - mean label of 10 nearest neighbours| / (max y - min y), clipped to [0.1, 1] (`:100-126`), neighbours in the importance-weighted, scaled space.
- Not run: molecule weighting for gp (`MODELS_WITHOUT_SAMPLE_WEIGHT`, `:177-180`) and on nominal labels.
- Importance method: the rank-1 hybrid's on that fold (`:286-298`).
- Disagrees with outline 5.3.4: not budget 100; not rf, lightgbm and xgboost; single representations included.

### c21 one representation plus extras (`complexity/c21_single_rep_augmentation.py`)

- Question (4-6): can a few extra features nudge a strong single representation?
- Conditions: A_baseline; B_aug (curated augmentation sets); C_topreps_n9 (top 9 features of every other pool representation, `DEFAULT_TOP_N = 9`, `:142`); D_both; R_rand_n9 (matched random); H0 the multi-representation hybrid at fixed recipe. Two planted decoys. Ranking: residual_correlation.
- Extra-feature arms run only where augmentation is kept: ginkgo_colloidal and ginkgo_hic, uncapped (docstring 66-79). VERIFIED in results: only those two files contain B, C, D and R arms; the other 21 hold A_baseline and H0 only.
- The outline's c21 description (5.3.4: "pool of 50, blocks of 5/10/20/40") describes the engine's admission step, not what c21 varies.

### f24 admission rules and the decoy test (`tests/experiments/framework/f24_augmentation_admission.py`)

- Question (8-20): which way of admitting extra features works, per dataset?
- Methods: none, all (no test), capped, uncapped. Axes: donor source, features per donor, ranking, size rule, margin in fold-spread units. Controls: matched random donor, pure-noise decoy, near-copy decoy (5% jitter).
- Results (`tests/experiments/results/framework/f24_*`): buchwald_hartwig, caco2, flab_expression_koenig, tcga_brca_pam50 complete for single and hybrid placements; herg_ki one rf cell only. 3 folds, 3 seeds, models lightgbm and rf (`n_folds`, `n_seeds`, `models` in each file). VERIFIED.
- Outline 5.5a says "five datasets". Four are complete; the fifth is one cell.
- Provenance: item 13 above.

### opig_qsar (`tests/experiments/opig_activity_cliff_qsar.py`)

- Question (8-11): which representation gives the best QSAR on three ChEMBL targets, and does a hybrid change activity-cliff prediction? Replicates Dablander et al. 2023.
- Targets D2 (6,333 compounds, 2,876 scaffolds), factor Xa (3,605, 1,462), Mpro (1,924, 1,087) (R/opig/qsar.json `<target>.n_compounds`, `.n_scaffolds`). VERIFIED.
- Pool: chembl_qsar's 10 general representations; 8 built, grover and smited missing (`reps_missing`). VERIFIED.
- Models lightgbm and dnn. 5 scaffold folds. Metric MAE. Standard selection per fold, search filed per target (condition_suffix).
- Outline 5.7 "network on factor Xa 0.7367 to 0.6612 MAE on all 5 folds" and "wins on every target and model except one": SAME. base1 per fold read from `per_fold_selection[*].models.<m>.top2_base_reps[0]`, scored from `results.<rep>.models.<m>.per_fold[*].held_out.mae`; hybrid from `results.kirby_hybrid`. fxa dnn 0.7367 to 0.6612, 5 of 5 folds lower. The exception is fxa lightgbm (0.6854 to 0.7144). VERIFIED (my arithmetic).
- Note: `top2_base_detail[*].score_mean` is null in the stored selection. The ranking it records cannot be checked from the file.

### opig_shared_vectors_remedy (`tests/experiments/opig_shared_vectors_remedy.py`)

- Question (3-12): can a hybrid stop a representation giving two molecules one identical feature row? Score and tie count must both move.
- Arms: A target alone; C1 target + ecfp4 + pdv (both blind to stereochemistry); C2 target + ecfp4 + mhg_gnn (mhg_gnn ties none). B and D (donor features) run only on ginkgo_colloidal and ginkgo_hic.
- Ties reported three ways: target at full width, target features the hybrid kept, whole hybrid.
- Results: herg_ki only. Target graph_kernel, 105 of 1,412 molecules in shared rows (R/opig/shared_vectors_remedy/opig_shared_vectors_remedy.json `per_dataset[0].choice`). VERIFIED.
- Outline 5.7 "graph kernel 0.6109 to 0.5715 RMSE, lower on all 5 folds, shared rows 105 to 0": SAME for C2 svm (`rows` C2__hybrid1 svm: target 0.6109, hybrid 0.5715, 5 of 5 lower; `ties_per_fold[*].built_hybrid` 0). VERIFIED.
- C1 svm also improves (0.6109 to 0.5755, 5 of 5) while 56 molecules still share a row. So the score gain does not need the ties broken. VERIFIED.

### opig_shared_vectors_census (`tests/experiments/opig_shared_vectors_census.py`)

- Counts molecules sharing an identical row, per representation, base and augmentation sets both.
- Results: partial file only, 11 datasets reached (abagym_Ang2 to ddg_combined, plus chembl_qsar empty). VERIFIED.
- Outline 5.7 "MACCS keys merge 804 of 2,158 Caco-2 molecules": SAME (`percent_molecules_sharing` 37.257, `n_molecules_sharing` 804). maccs is in Caco-2's augmentation list, not its base pool. VERIFIED.
- "On two antibody sets a representation merges most of them": SAME. abagym_EGFR aac 432 of 659; abagym_VEGF structural 332 of 418. VERIFIED.

### opig_blueprint_transfer (`tests/experiments/opig_blueprint_transfer.py`)

- Question (4-6): can settings found on LogD be reused on hERG Ki?
- Arms per target fold (docstring 17-28): base_best (standard selection), native (searched on the target fold), recipe_transfer (LogD's allocation, importance, budget and evaluator; features re-selected on hERG Ki), full_transfer (LogD's features unchanged). The source recipes come from one search on every LogD molecule, which holds no target molecule.
- 5 folds, RMSE, svm and dnn, 7 shared representations (item 6).
- Outline 5.7 verdict: SAME. svm: base_best 0.5888, native 0.5631, recipe 0.5665, full 0.5692. dnn: 0.6035, 0.5666, 0.5889, 0.5926 (`per_model.<m>.target.<arm>.mean`, 5 folds). VERIFIED.

### opig_ginkgo (`tests/experiments/opig_ginkgo_developability.py`)

- Folds: the competition's own hierarchical cluster folds, not nested_cv_outer_folds (docstring 30-41). Standard selection per fold. paratope and structural scored alone.
- Results: ginkgo_hic only, 4 of 5 folds finished (R/opig/ginkgo.json `ginkgo_hic.n_folds_finished`). VERIFIED. The docstring lists nine properties.
- Outline 5.7 "neither hybrid beats the single best representation on HIC retention": SAME on 4 folds. gp: base1 0.6173, hybrid1 0.5537, hybrid2 0.5072 Spearman (`by_model.gp.by_arm`). VERIFIED.

### fed01 noise across sites (`federated/fed01_noise_heterogeneity.py`)

- Labels made two-class: Caco-2 at log10 efflux ratio >= log10 2, LogD at >= 3.0 (`federated_helpers.py:496-520`). Metric MCC.
- 5 sites by k-means (`K = 5`, `:78`). Flip levels 0, 0.05, 0.1, 0.15, 0.2, 0.3 spread over sites from 0 to the level (`SIGMA_FRACS`, `:79`; `:805-807`).
- Arms (docstring 8-26): C0 hybrid on pooled noisy labels; base1/2; C1 pooled Morgan fingerprint; C2/C3 FLuID ensembles; C4 FLuID student; C5 federated selection; C6 C5 after each site removes flagged molecules. One search per fold on clean two-class labels, shared by every level.
- Results: partial files only. Caco-2 29 of 30 (model, seed, fold) units; LogD 4 folds of dnn seed 0. VERIFIED.
- Defect: on Caco-2, 240 site selections in C6 and 12 in C5 were refused because a site's held-out block had one group, under ten molecules or one class (`silo_failures` in R/federated/fed01_noise_heterogeneity/caco2__folds.partial.jsonl). VERIFIED. So C6 and C5 differ in two ways at once: the removal of flagged molecules, and which sites contributed.

### fed02 representations as sites (`federated/fed02_rep_diversity_silos.py`)

- Each of 5 sites holds the same molecules as one representation: ecfp4, graph_kernel, molformer, pdv, avalon on Caco-2 (`config.silo_reps`). Regression, RMSE, dnn and svm, 3 seeds, 5 folds. VERIFIED.
- Arms: best single site; FLuID similarity and quality ensembles; hybrid over all 5; hybrid over top 3; clustered variant.
- Outline 5.8 "averaging five models gives 0.2720 against 0.2858 RMSE" on Caco-2 dnn: FLuID quality 0.2721, similarity 0.2730, hybrid over all sites 0.2837 (`per_model.dnn.aggregated.<arm>.rmse.mean`, mean over 3 seeds). The direction is SAME. The 0.2858 does not match the aggregated hybrid number; it may be one budget. UNVERIFIED which.

### fed03 Part B families as sites (`federated/fed03_synthetic_private.py`)

- Three sites, one per family (`REP_FAMILIES`, `:87-110`): structural, physicochemical, learned.
- C1 best family; C2 one selection over all families; C3 one selection per family at an even share of the budget. When a family is narrower than its share, the rest goes to the others, and `same_width_as_c2` records whether the widths still match (notes, `:2150-2165`). VERIFIED.
- Budgets 500-2,000, dnn and svm, 3 seeds, RMSE.

### fed04 clusters (`federated/fed04_cluster_aware_importance.py`)

- Question (4-10): does the best allocation differ across k-means clusters of chemical space?
- Two-class labels, MCC. Caco-2 complete (30 units); LogD partial (svm seed 2 missing, no json). VERIFIED.

---

## 3. Search-space size, recomputed from the code

Rule: `_expected_config_count` (`src/kirby/find_best_hybrid.py:609-634`). `match` is not a separate evaluator: for each model it adds that model as an evaluator if it is not already listed. So (model, evaluator) pairs are a sum over models. A representation count at or above the pool size collapses to "all".

configurations = (model, evaluator) pairs x allocation methods x importance methods x budgets x distinct representation counts.

Full default grid: 5 models, evaluators match + rf + lightgbm + dnn. Pairs: rf 3, lightgbm 3, dnn 3, xgboost 4, svm 4 = 17. 17 x 5 allocations x 3 importance methods x 8 budgets x 4 counts = **8,160**. This matches thesis.tex and dataset_policy.py's 8,160. VERIFIED.

Fits per outer fold, from the loop structure (`_run_search`, `find_best_hybrid.py:6240-6600`), an estimate and not a run log:
- downstream fits = configurations x 5 inner folds;
- importance fits = 5 inner folds x importance methods x representations (computed once, reused);
- standalone evaluator fits = 5 x distinct evaluators x representations (reused);
- single-representation baselines = 5 x models x representations.

Per (dataset, outer fold), current decided_grid.json, general search:

| dataset | pool | pairs | configs | downstream fits | total fits / fold | x 5 folds |
|---|---|---|---|---|---|---|
| qm9 | 10 | 6 | 108 | 540 | 840 | 4,200 |
| herg_ki | 10 | 7 | 224 | 1,120 | 1,520 | 7,600 |
| caco2 | 10 | 7 | 252 | 1,260 | 1,710 | 8,550 |
| logd | 10 | 7 | 63 | 315 | 665 | 3,325 |
| adme_hlm | 10 | 7 | 315 | 1,575 | 2,025 | 10,125 |
| adme_rlm | 10 | 7 | 315 | 1,575 | 2,025 | 10,125 |
| adme_mdr1 | 10 | 7 | 315 | 1,575 | 2,025 | 10,125 |
| herg | 10 | 7 | 210 | 1,050 | 1,400 | 7,000 |
| flab_expression_koenig | 10 | 7 | 420 | 2,100 | 2,550 | 12,750 |
| flab_binding_koenig | 10 | 7 | 210 | 1,050 | 1,450 | 7,250 |
| flab_thermo_tresanco | 10 | 7 | 1,260 | 6,300 | 6,700 | 33,500 |
| abagym_EGFR | 10 | 5 | 600 | 3,000 | 3,350 | 16,750 |
| abagym_VEGF | 10 | 2 | 540 | 2,700 | 3,050 | 15,250 |
| abagym_Ang2 | 10 | 8 | 720 | 3,600 | 4,050 | 20,250 |
| humanness_therapeutic | 10 | 7 | 3,360 | 16,800 | 17,250 | 86,250 |
| ddg_combined | 8 | 7 | 378 | 1,890 | 2,210 | 11,050 |
| cdr_flexibility | 10 | 7 | 840 | 4,200 | 4,600 | 23,000 |
| abcov_kd | 10 | 7 | 2,520 | 12,600 | 13,050 | 65,250 |
| ginkgo_colloidal | 10 | 6 | 810 | 4,050 | 4,500 | 22,500 |
| ginkgo_hic | 10 | 7 | 840 | 4,200 | 4,650 | 23,250 |
| ginkgo_acsins | 10 | 7 | 1,512 | 7,560 | 8,010 | 40,050 |
| ginkgo_titer | 10 | 4 | 640 | 3,200 | 3,500 | 17,500 |
| ginkgo_pr_cho | 10 | 7 | 840 | 4,200 | 4,600 | 23,000 |
| buchwald_hartwig | 10 | 4 | 120 | 600 | 900 | 4,500 |
| tcga_brca_pam50 | 10 | 7 | 2,205 | 11,025 | 11,475 | 57,375 |

Sum over the 25 datasets, 5 outer folds: 540,525 fits for the general search alone. The distance search adds kNN as a third model and has its own grid; its configurations per fold sum to 37,224 over the same 25 datasets (19,617 for the general search). Computed by `scratchpad` script in this session from decided_grid.json and DATASET_REPS (`tests/experiments/complexity/complexity_helpers.py:136`). VERIFIED (arithmetic); the fit counts are an estimate from code structure.

Check against a run: herg_ki's c02 file has 224 configurations, which matches. VERIFIED.

What the stored c02 results actually searched, where it differs from decided_grid.json (`configs` list length in each c02 file, all 5 folds):

| dataset | stored | current grid | which axis differs |
|---|---|---|---|
| logd | 252 | 63 | representation counts 2, 3, 5, all against all only |
| adme_hlm | 630 | 315 | representation counts 5, all against all |
| abagym_EGFR | 800 | 600 | counts 2, 3, 5, all against 3, 5, all |
| flab_thermo_tresanco | 1,680 | 1,260 | counts 2, 3, 5, all against 3, 5, all |
| cdr_flexibility | 1,120 | 840 | counts 2, 3, 5, all against 2, 3, 5 |
| tcga_brca_pam50 | 2,940 | 2,205 | counts 2, 3, 5, all against 3, 5, all |
| ginkgo_acsins | 1,451 | 1,512 | 61 configurations missing from the stored list |

All other c02 files match the current grid. The other experiments read the same per-fold searches, so the same holds for them on these datasets. UNVERIFIED for experiments other than c02.

The outline's 5.2.2 numbers: FRAMEWORK_FINALIZATION.md's 162 and 3,060 are for a different, three-budget grid; I did not recompute them. dataset_policy.py's 816 against 8,160: 8,160 recomputes; 816 is not reproduced by any live dataset's grid.

---

## 4. Code changes that split the stored results

Count of result files with `timestamp` at or after 2026-10-08T13:04 (dnn clip commit c18ee30), per experiment. VERIFIED counts; which code ran is UNVERIFIED.

| experiment | before | after | files after |
|---|---|---|---|
| c02 | 19 | 8 | abagym_Ang2, abagym_VEGF, bh_doyle_plates, ginkgo_acsins, ginkgo_hic, ginkgo_pr_cho, qm9, tz_her2_multiclass |
| c11 | 16 | 6 | bh_doyle_plates, abagym_VEGF, ginkgo_acsins, ginkgo_pr_cho, qm9, tz_her2_multiclass |
| c17 | 13 | 6 | abagym_EGFR, abcov_kd, adme_mdr1, ginkgo_acsins, ginkgo_pr_cho, qm9 |
| c18 | 0 | 25 | every file (the summary is rewritten at the end; folds may be older) |
| c19 | 19 | 8 | |
| c22 | 0 | 24 | every file |
| c23 | 12 | 4 | |
| c24 | 15 | 3 | |
| c33 | 12 | 4 | |
| c07 | 16 | 9 | |
| c06 | 16 | 7 | |

The gp fixes (8 Oct 11:03 and 15:59) touch abagym_Ang2, abagym_VEGF, ginkgo_acsins, ginkgo_hic, ginkgo_pr_cho, and c23's svm near-family arm everywhere.

---

## 5. Citations for the Chapter 5 methods

All 147 keys cited between thesis.tex lines 1821 and 3603 exist in refs.bib. VERIFIED (grep of `\cite*{}` keys against `@type{key` lines).

Present in refs.bib and already cited in Chapter 5: Peng2005 (mRMR), Guyon2002 (RFE), Jolliffe2016 (PCA), Nogueira2018 (stability), Stahlschmidt2022 (early and late fusion), Varma2006 and Cawley2010 (selection bias), Ambroise2002 and Vabalas2019 (leakage), Bengio2004 (no unbiased k-fold variance), Kohavi1997 (wrapper), Lundberg2020 (TreeExplainer), Shrikumar2017 (DeepLIFT), McMahan2017 (FedAvg), Heyndrickx2024 (MELLODDY), Hanser2023, Hanser2025, Bassani2023 (Lhasa), Olier2018 (Meta-QSAR), Cassara2022 (federated feature selection), shu2019metaweightnet, Grinsztajn2022, Turkina2025, Kenlay2024, Vieira2026, Ahneman2018, Koenig2017, Adams2016, Chungyoun2024, Chinery2024, Prihoda2022, ValdesTresanco2023, fang2023, polaris, openadmet, Goldman2020.

Present in refs.bib, not cited in Chapter 5, and used by a Chapter 5 experiment:
- Dablander2023: opig_qsar replicates it.
- vanNiekerk2026: opig_ginkgo replicates the Ginkgo competition.
- Dehghani2022 ("The Efficiency Misnomer"): c11's cost accounting cites it.
- Beyer1999, Aggarwal2001, Radovanovic2010: c15's distance concentration and hubness.
- Belkin2019: c33's and c15's double-descent question.
- Kalousis2007: feature-selection stability, beside Nogueira2018.
- Breiman2001, Chen2016, Ke2017 (ke2017lightgbm is cited): the downstream models.
- Dietterich2000, Breiman1996: ensembles, for c11.

Missing from refs.bib:
- Lundberg2017 (SHAP, NeurIPS): c22 uses KernelExplainer and DeepExplainer, which come from that paper. Only Lundberg2020 is present.
- Krstajic2014 (nested cross-validation in cheminformatics). Not present; Varma2006 and Cawley2010 are.
- Wettschereck, Aha and Mohri 1997 (feature-weighted kNN). The outline says it is not yet in refs.bib; confirmed absent.
- Heyndrickx2023: the outline names it; refs.bib has Heyndrickx2024. Same work, check the year.
- Dataset sources with no key found: AbAgym (abagym_*), Ab-CoV (Rawat 2022; Rawat appears only as a co-author in Chinery2024), ITsFlexible (cdr_flexibility), SKEMPI 2.0 and AB-Bind (ddg_combined), Graphinity. Searched by name and title, VERIFIED absent.

---

## 6. Dataset list: outline 5.3.1 against the code (discrepancies only)

Sources: `tests/experiments/orchestrate/plan.py` (`dataset_roster()` = DATASET_REPS minus EXCLUDED_DATASETS minus CUT_DATASETS, `:61-80`, `:146-148`), `tests/experiments/dataset_policy.py` (`DATASET_REGISTRY_DATA`, `:230`), `tests/experiments/decided_grid.json`.

- Roster: 25 datasets. With herg_ic50 (a noise-only row, `plan.py:81-97`) it is the outline's 26. The outline's 26 names match the roster. VERIFIED.
- CUT_DATASETS (16): tcga_luad_vs_lusc, nt_H3K27me3, nt_H3K9ac, nt_H3K9me3, gb_human_enhancers_cohn, bindingdb_kd, kiba, flab_expression_adams, flab_binding_adams, suzuki, lhasa_herg, bh_doyle_plates, nt_H3K4me3, nt_promoter_tata, tz_her2_binary, tz_her2_multiclass. Matches the outline's cut list. EXCLUDED_DATASETS adds chembl_qsar and davis. VERIFIED.
- decided_grid.json has 43 rows. 18 are not on the roster (the 16 cut, chembl_qsar, herg_ic50). Every roster dataset has a row. VERIFIED.
- herg_ic50 has no row in dataset_policy.py's registry. The outline says domain comes from dataset_policy.py. VERIFIED.
- herg_ic50 has no DATASET_REPS entry; it borrows herg_ki's (`plan.py:81`). VERIFIED.
- tcga_brca_pam50: registry n_samples 900; the c01 outer split sums to 821 (R/complexity/c01_rep_characterization/tcga_brca_pam50.json, `_meta.outer_split.held_out_rows_per_fold` 3+79+403+142+194). The outline's 821 matches c01. VERIFIED.
- herg: registry note "TDC/Lhasa binary hERG blocker, ~10K"; the outline says 7,772 from the loader default. c01's folds sum to 7,772. VERIFIED.
- ginkgo folds: c01 uses nested_cv_outer_folds (`splitter` field); opig_ginkgo uses the competition's own fold assignments. The outline says "Ginkgo's own 5 cluster folds" for the dataset row. Whether the complexity runs' groups reproduce the competition folds is UNVERIFIED.
- One tcga_brca_pam50 outer fold holds 3 patients (c01 split). Weighted F1 on that fold rests on 3 labels. VERIFIED.
- thesis.tex tab:datasets (lines ~2345-2378) and tab:decided_grid (lines ~2253-2276) list 16 rows. Five are cut: Buchwald-Hartwig plates (bh_doyle_plates), trastuzumab-HER2 binding (tz_her2_binary), H3K4me3, TATA promoter, lung adenocarcinoma vs squamous (tcga_luad_vs_lusc). Live datasets missing from both: herg, adme_hlm, flab_binding_koenig, abagym_EGFR, abagym_VEGF, abagym_Ang2, ddg_combined, cdr_flexibility, abcov_kd, the five ginkgo sets. VERIFIED.
- Small-molecule pools carrying chembert (256 features): herg_ki, logd, qm9, adme_hlm (DATASET_REPS). qm9 carries chembert and chemberta. Matches the outline's CHECK. VERIFIED.
