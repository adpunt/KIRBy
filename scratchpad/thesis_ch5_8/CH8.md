# Chapter 8 inputs (Discussion and Conclusions)

Written by Claude on 2026-10-09 for the outline's Chapter 8 (THESIS_OUTLINE.md:2051-2131).
Read-only pass. Nothing in THESIS_OUTLINE.md, thesis.tex or the answer pages was changed.

Labels used below:
- VERIFIED: read in this session from a raw result file or from code.
- PAGE: read from a table on a rebuilt answer page (`/Volumes/seagate/claude_analysis_2026-10-08/answers/<exp>.md`, tables rebuilt 9 Oct from the result files). Not re-derived from the raw file.
- UNVERIFIED: seen only in a document or memory file.
- Experiment codes appear here only as file pointers.

Two result folders are used. They differ for n09:
- `/Volumes/seagate/kirby_results_2026-10-09/noise/n09_lnl_survey/` holds 11 datasets.
- `/Volumes/seagate/kirby_results_2026-10-09_noise_view/noise/n09_lnl_survey/` holds 13. It adds ddg_combined and ginkgo_titer. The n09 answer page reads this one.

---

## 1. What in thesis.tex Chapter 8 is out of date against the outline

thesis.tex:4740-5103, `\chapter{Discussion and Conclusions}`.

### 1.1 Structure

- thesis.tex has seven chapters. Chapter 6 is still "Hybrid Representations for Similarity and Label Noise" (`chap:similarity`, thesis.tex:3603). The outline splits it into Chapter 6 (similarity) and Chapter 7 (noise). So every `Chapter~\ref{chap:similarity}` in Chapter 8 that means the noise work now points at the wrong chapter.
- The first comment line (thesis.tex:4741) still calls it "Chapter 7".
- Sections in thesis.tex: Summary; Dataset Dependence of the Best Method; Simple Models and Small Feature Sets; Limitations; Future Directions; Conclusions.
- Sections in the outline: 8.1 Summary; 8.2 the chain of eight steps; 8.2a the central claim; 8.3 handling noise realistically; 8.4 no single method fits every dataset; 8.5 simple is often better; 8.6 limitations; 8.7 future directions.
- Missing from thesis.tex entirely: the eight-step chain (8.2), the scope defence, diagram D8.1, and the central claim (8.2a). The chapter does not end on the author's argument, which the outline says it must.
- 8.3 "Handling noise realistically" exists only as the unnumbered opening paragraph (thesis.tex:4765, the `Real assay labels are not noisy...` paragraph).

### 1.2 Vocabulary

- "LNL" appears 28 times in lines 4740-5103. The outline's naming rules (THESIS_OUTLINE.md:586-596) allow "LNL" only in Chapters 2 and 4.
- The text uses the old two-way split: "LNL methods that change training" against "noise mitigation, the subset that changes the labels" (thesis.tex Limitations and Dataset Dependence sections). The outline now has three groups: detector, noise mitigation (delete, count less, or move the label), and noise-robust training. It also has three things that pick molecules: a detector, a known flag, or every molecule.
- The Limitations recommendation ("prefer methods that change training over noise mitigation, except for noise of known structure") was written before the full runs. The outline (THESIS_OUTLINE.md:556-560) now says the censoring corrections are the strongest results and are all flag-driven. That recommendation needs rewording around detector against known flag, not training-time against data-level.

### 1.3 Numbers in thesis.tex Chapter 8 that came from preliminary runs

Every number below predates the full runs. Status against current results:

| thesis.tex Chapter 8 statement | Current results | Status |
|---|---|---|
| hERG Ki default hybrid of 250 features beat best single with SVM and LightGBM on all three folds; RF 0.0006 pKi worse | Full search, 5 folds: hERG Ki svm 0.5881 → 0.5590 RMSE, dnn 0.6031 → 0.5691, hybrid better on 5 of 5 folds for both. VERIFIED, `complexity/c02_complexity_matching_grid/herg_ki.json` key `heldout_delta_by_model` (+0.0290 svm, +0.0340 dnn); before/after values PAGE c02.md | CHANGED (models are now svm and dnn; RF not run on hERG Ki) |
| Best budget 100 features for RF to 1,000 for SVM on hERG Ki | Best training-molecule budget: svm 500, dnn 2000. PAGE c02.md Q2 | CHANGED |
| LogD search settings reused on hERG Ki beat hERG Ki's best single for 3 of 5 models | Outline 5.7 now: beats hERG Ki's best single with both models, loses to a hybrid searched on hERG Ki. Not re-read here | NOT CHECKED here (Chapter 5 file) |
| Three hERG sites: selected features 0.5264 against FCFP4 0.4690 MCC | Those three sites came from the Lhasa hERG data. lhasa_herg was cut on 8 Oct (outline 5.8, line 1465). Current site results are Caco-2 and LogD only (`federated/fed02`, `fed03`, `fed04`; fed01 partial) | CHANGED: the hERG-site numbers are no longer backed by a live run |
| kNN on hERG Ki: similarity hybrid 0.6128 against atom pairs 0.6452 RMSE | Chapter 6 file's job; s05 is the source | NOT CHECKED here |
| Constant lift beat clipped labels in 48 of 48 and 33 of 36 comparisons | Chapter 7 file's job | NOT CHECKED here |
| Nanobody thermostability: 0 of 12 methods beat the plain network at 0.5 SD, 8 of 12 at 1.5 SD | flab_thermo_tresanco has no n09 results now (n/a in every column of n09.md's table) | NOT YET RUN in the current runs |
| "Chapter 6 adds nanobody thermostability and humanness" to the noise datasets | n09 now covers 13 datasets (noise_view folder): buchwald_hartwig, caco2, cdr_flexibility, ddg_combined, ginkgo_acsins, ginkgo_colloidal, ginkgo_pr_cho, ginkgo_titer, herg, herg_ki, humanness_therapeutic, logd, qm9. VERIFIED by listing the folder | CHANGED |
| "The label-noise experiments are almost all regression; only Chapter 6's Gaussian experiments flip class labels" | n09 runs six label-flip kinds on herg, cdr_flexibility and humanness_therapeutic. VERIFIED, file names `<ds>__{uniform,class_dependent,class_imbalance,binary_asymmetric,confusion_directed,instance_noise}.json` | CHANGED. Classification under label flips is now partly done, which also weakens the first future direction |
| "results ... are per-fold scores over 3 to 5 folds" | Complexity work: 5 folds. Noise work: fold 0 only in every n09 file. VERIFIED, `folds_run: [0]` in all 55 n09 json files read | CHANGED: one fold for the noise work |
| Future direction: "whether neighbour-based cleaning does better on outliers, and whether label smoothing and sample weighting behave differently on these noise types" | n09 now runs outlier_p05, grouped_shifted, shift_only_p10 and censoring on the regression datasets (VERIFIED, file names). The question is no longer only open | CHANGED |
| Future directions order: classification first | Outline 8.7 puts more QM9 properties first | CHANGED |
| Future direction "transformed features, graph-distance features, KernelSHAP and permutation importance" | Not in outline 8.7 | Outline dropped it, or it was never carried over. DECIDE |
| External benchmarks paragraph (activity cliffs, Ginkgo competition splits, Hummer2025) | Not in outline 8.7 | DECIDE |
| Missing from thesis.tex future directions | More QM9 properties; the extra-features rule (decoy test) | ADD |

There are 22 `[RESULT NEEDED]` markers and 19 lines saying "preliminary" in thesis.tex:4740-5103 (counted with grep).

---

## 2. The chain in 8.2: each piece of "measured" evidence

### Step 2. "All 20 configurations with the lowest noise retention used a distance" (Chapter 4, 4.4.1)

- Source: `/Users/apunt/Downloads/Confirmation_Report/paper.tex:721`. Read in this session. It says the 20 worst configurations "with respect to noise robustness used distance metrics". It also says no configuration reached both clean R2 > 0.6 and retention > 80%.
- The raw results are not available locally. `~/Downloads/experiments/` holds the job scripts only. The outline (line 834) says the CSVs were on the server under `$DATA/results/comprehensive_noise_study/`.
- Status: UNVERIFIED as a measurement. It is a sentence in an earlier report. It cannot be recomputed from anything on this machine.
- QM9 HOMO-LUMO gap, 10,000 molecules (8 Mahalanobis commands used 5,000, outline line 832). Scaffold splits. Folds or replicates not stated in paper.tex:717.
- Two inconsistencies inside the outline:
  - 8.2 calls this "Step 2 is measured". Outline 7.11 (line ~1990) says the ANOVA on distance usage "is step 3 of the chain". The two places number the same evidence differently.
  - The ANOVA has no effect size anywhere (outline line 835). Only the top-20 ranking is quotable.

### Step 3. "Highest Caco-2 correlation in any single representation at 0.2292 Spearman" (Chapter 6, 6.3)

- VERIFIED. `/Volumes/seagate/kirby_results_2026-10-09/similarity/s01_similarity_baseline/caco2.json`, key `by_rep/top_torsion/minkowski_p0.5/sac_rho_mean` = 0.2292. Mean over 5 outer folds.
- What it computes: Spearman between pairwise distance and pairwise activity difference, over sampled pairs of each fold's held-out molecules (field `metric_definition`).
- Status: SAME.
- SMAC colloidal stability's highest, 0.0155 (ablang2, Minkowski p=0.5), also SAME. 5 folds.
- Outline 6.3 says "On five datasets some representations sit at or below zero." Current count is 8 of 18 datasets. Each count is representation-and-distance pairs at or below zero, out of the pairs scored:
  - ginkgo_colloidal 15 of 30, cdr_flexibility 14 of 30, tcga_brca_pam50 7 of 30, ginkgo_acsins 6 of 30, ginkgo_pr_cho 5 of 30, abagym_EGFR 2 of 30, flab_expression_koenig 2 of 30, humanness_therapeutic 2 of 30.
  - Status: CHANGED (five → eight). All but one are antibody or expression sets; no small-molecule set has any.
- Caco-2 0.2292 is not the highest anywhere. Higher bests: tcga_brca_pam50 0.4723 (gene_geneformer_v2, cosine), humanness_therapeutic 0.3885 (igt5, cosine), qm9 0.2789 (chemberta, Manhattan), flab_expression_koenig 0.2498. hERG Ki's best is 0.1170 (smited, Manhattan). The sentence in 8.2 is accurate only because it names Caco-2.
- Chapter 4 (outline 4.3, line 786) quotes different Caco-2 numbers from an older local run: atom_pair 0.225, Manhattan, 18 Sep. The current file gives top_torsion 0.2292 under Minkowski p=0.5. Chapters 4, 6 and 8 must quote the same run.
- The chain puts step 3 in Chapter 4 (4.3-4.4). Its measured evidence sits in Chapter 6 (6.3). A reader meets the number two chapters after the step it supports.

### Step 4. "Features that do not vary, hash collisions, and molecules a representation cannot tell apart" (Chapter 4, 4.2)

Three parts. Only two are measured.
- Features that do not vary: counted 2026-09-27 from `/Volumes/seagate/kirby_cache/rep_cache/<ds>/<rep>.npy` (outline 4.2.1). UNVERIFIED here, and the outline itself flags that the pipeline reads a different cache first.
- Hash collisions: NOT MEASURED. Outline 4.2.2 says "Planned statistic (not run; no code in KIRBy)". I searched `tests/` and `scratchpad/` for unfolded Morgan identifiers or bit-collision code and found none. The only "collision" code in `src/kirby/representations/molecular.py:2352-2385` counts molecules that the ChemBERTa tokeniser turns into identical token sequences. That is a different thing.
- Molecules a representation cannot tell apart: VERIFIED. `tests/results/comparisons/opig_shared_vectors_census.csv` (16 Sep), herg_ki rows. ecfp4, atom_pair, pdv and mordred each give 56 of 1,412 molecules an identical feature vector (`n_molecules_sharing`). maccs gives 377 (26.7%). This census fits no model, so it has no folds.
- Status: the sentence in 8.2 is CHANGED. "Hash collisions" must come out of "Step 4 is measured", or the statistic must be run.

### Step 8. "The best learning-with-noisy-labels method on a hybrid is never the best on either single representation, on all 9 datasets" (Chapter 7, 7.11)

Status: CHANGED. The statement is false in its strong form, and the dataset count is out of date.

What the n09 answer page tests (n09.md, question 2, script `answers/_scripts/n09.py:284-296`): is the top method on `hybrid` the same as the top method on `base_1` AND on `base_2`? "No" there means the hybrid's top method differs from at least one of the two. It does not mean it differs from both. The outline's sentence says "either", which is the stronger claim.

I recomputed both readings from the raw files. Script: `/private/tmp/claude-501/-Users-apunt-repos-KIRBy/48666903-2765-47e3-961d-0fd22e9f6645/scratchpad/n09_top.py`. It reads each file's `vs_plain_dnn_same_labels/<level>/<space>_lnl/<method>/method_mean`. It takes the best method per feature space, at every damage kind and level, level 0 included. Fold 0 only. Under each damaged level the mean is over 3 damage draws (1 under censoring). Level 0 is one fit.

Folder used: `/Volumes/seagate/kirby_results_2026-10-09_noise_view/noise/n09_lnl_survey/`. Four `grouped_shifted` files there have no `vs_plain_dnn_same_labels` key (buchwald_hartwig, ginkgo_acsins, ginkgo_colloidal, ginkgo_pr_cho), so they are skipped.

Each row is one dataset. "Combinations" counts damage kind × level, level 0 counted once per file.

| dataset | combinations | hybrid's top method differs from both singles | equals base_1's | equals base_2's | equals both |
|---|---|---|---|---|---|
| buchwald_hartwig | 8 | 2 | 6 | 0 | 0 |
| caco2 | 14 | 8 | 2 | 5 | 1 |
| cdr_flexibility | 12 | 2 | 1 | 10 | 1 |
| ddg_combined | 8 | 7 | 0 | 1 | 0 |
| ginkgo_acsins | 8 | 8 | 0 | 0 | 0 |
| ginkgo_colloidal | 8 | 6 | 1 | 1 | 0 |
| ginkgo_pr_cho | 8 | 7 | 1 | 0 | 0 |
| ginkgo_titer | 8 | 7 | 1 | 0 | 0 |
| herg | 12 | 7 | 3 | 4 | 2 |
| herg_ki | 10 | 6 | 0 | 4 | 0 |
| humanness_therapeutic | 10 | 9 | 0 | 1 | 0 |
| logd | 10 | 7 | 3 | 0 | 0 |
| qm9 | 4 | 2 | 0 | 2 | 0 |

What this says:
- "Never the best on either single representation" holds on 1 of 13 datasets: ginkgo_acsins.
- On cdr_flexibility the hybrid's top method equals base_2's in 10 of 12 combinations. That is close to the opposite of the claim.
- On buchwald_hartwig it equals base_1's in 6 of 8 (small_loss each time at level 0).
- The hybrid's top method equals BOTH singles' top method in 4 combinations: caco2 shift_only_p10 at 1.0 (distance_selection), cdr_flexibility instance_noise at 0.1 (small_loss), herg class_dependent at 0.1 (confident_learning), herg instance_noise at 0.1 (distance_selection).
- The weaker claim, "the hybrid's top method is not the top method on both singles", holds in at least half of the combinations on all 13 datasets. That is the n09 page's question 2, answered "no" on every row. VERIFIED by recomputation, and it agrees with n09.md's table.
- The 2026-10-09 folder without the noise_view additions gives the same picture on 11 datasets (it lacks ddg_combined and ginkgo_titer, and caco2 has 10 combinations, not 14).

Dataset count: outline says 9. Current: 13 in the noise_view folder, 11 in the main 2026-10-09 folder. CHANGED.

What can be written instead, in her words: on 12 of 13 datasets, the noise method that scores best with the hybrid's features is sometimes also the best on one of the two single representations. On none of the 13 is it usually the best on both. The ranking of noise-handling methods moves when the features move, but not every time. One fold.

Related sentence in 7.11, also CHANGED: "On three Ginkgo sets every method passes." Counted from n09.md's dataset table (PAGE), yes in 24 of 24 cells only on ginkgo_titer. ginkgo_pr_cho 23 of 24 (multistage_cleaning on the labels as delivered is "no"). ginkgo_colloidal 22 of 24 (early_learning and dividemix on the labels as delivered are "no").

---

## 3. "The hybrid worked unusually well on antibodies"

### 3.1 The hybrid against the best single representation (prediction)

Source: `/Volumes/seagate/kirby_results_2026-10-09/complexity/c02_complexity_matching_grid/<ds>.json`, key `heldout_delta_by_model`. VERIFIED. Positive means the hybrid is better in the dataset's own metric (sign checked against logd: dnn 0.4497 → 0.4089 RMSE is +0.0408). 5 outer folds on every dataset. The top single and top hybrid are each picked on the fold's training molecules.

Each row is one dataset and model. "Better on" counts folds where the hybrid's held-out score beat the single representation's, out of 5.

Small molecules:
| dataset | metric | model: mean gain, folds better, range over folds |
|---|---|---|
| adme_hlm | RMSE | svm +0.0307, 5/5, +0.014..+0.043; dnn +0.0386, 5/5 |
| adme_mdr1 | RMSE | svm +0.0308, 5/5; dnn +0.0294, 5/5 |
| adme_rlm | RMSE | svm +0.0392, 5/5; dnn +0.0362, 5/5 |
| caco2 | RMSE | dnn +0.0267, 5/5; svm +0.0174, 5/5 |
| herg_ki | RMSE | svm +0.0290, 5/5; dnn +0.0340, 5/5 |
| logd | RMSE | dnn +0.0408, 5/5; svm +0.0628, 5/5 |
| qm9 | RMSE | dnn +0.0464, 5/5; lightgbm +0.0257, 5/5 |
| herg | AUROC | lightgbm +0.0111, 4/5; xgboost +0.0113, 4/5 |

Antibodies:
| dataset | metric | model: mean gain, folds better, range over folds |
|---|---|---|
| abagym_Ang2 | Spearman | gp +0.0499, 3/5, -0.147..+0.314; svm +0.0674, 3/5, -0.133..+0.381 |
| abagym_EGFR | Spearman | svm +0.0316, 2/5; dnn +0.0350, 2/5 |
| abagym_VEGF | Spearman | gp +0.1035, 4/5, -0.202..+0.263; svm +0.0419, 3/5, -0.257..+0.375 |
| abcov_kd | Spearman | lightgbm -0.0840, 2/5, -0.595..+0.349; xgboost +0.0125, 3/5 |
| cdr_flexibility | AUROC | xgboost +0.0163, 4/5; lightgbm -0.0044, 3/5 |
| ddg_combined | Spearman | svm +0.1359, 4/5, -0.121..+0.699; dnn +0.2225, 4/5, -0.147..+0.418 |
| flab_expression_koenig | Spearman | svm +0.0700, 5/5; dnn +0.0189, 4/5 |
| flab_thermo_tresanco | Spearman | dnn -0.0175, 2/5; svm -0.0002, 2/5 |
| ginkgo_acsins | Spearman | lightgbm +0.0611, 3/5; gp +0.0190, 3/5 |
| ginkgo_colloidal | Spearman | dnn -0.0088, 2/5, -0.173..+0.224; svm -0.0005, 3/5 |
| ginkgo_hic | Spearman | gp -0.0486, 2/5; dnn -0.0018, 3/5 |
| ginkgo_pr_cho | Spearman | gp -0.0590, 2/5; dnn +0.0130, 4/5 |
| ginkgo_titer | Spearman | dnn +0.0585, 4/5, -0.160..+0.218; rf +0.0202, 3/5 |
| humanness_therapeutic | AUROC | xgboost +0.0043, 3/5; rf -0.0045, 1/5 |
| tz_her2_binary (cut) | AUROC | xgboost -0.1066, 1/5; lightgbm -0.0413, 0/5 |
| tz_her2_multiclass (cut) | weighted F1 | svm +0.0099, 4/5; xgboost -0.0586, 0/5 |

flab_binding_koenig has no c02 json (partial only), so it is not in the table.

What this shows:
- On small molecules the hybrid is better on 5 of 5 folds for both models on 7 datasets (adme_hlm, adme_mdr1, adme_rlm, caco2, herg_ki, logd, qm9). herg is 4 of 5 for both.
- On antibodies, one dataset and model reaches 5 of 5: flab_expression_koenig with svm.
- The largest antibody mean gains are large in Spearman. ddg_combined dnn goes 0.1317 → 0.3542 (PAGE c02.md). abagym_VEGF gp goes 0.1981 → 0.3017 (PAGE). But the folds disagree in sign. ddg_combined svm ranges from -0.121 to +0.699 across folds.
- The hybrid loses for both models on flab_thermo_tresanco, ginkgo_colloidal and ginkgo_hic, and on both cut tz_her2 sets for at least one model.

So "unusually well on antibodies" is not supported as a statement about the hybrid's prediction. In c02 the hybrid is more consistent on small molecules. On antibodies it gives some of the largest mean gains and some losses. Its folds often disagree in sign. Status: CHANGED (reversed for consistency; true only for the size of the largest means).

Where the antibody exception does hold in the current results is the noise methods, not the hybrid:
- n09 (PAGE n09.md table): every one of the 12 methods passes on ginkgo_titer, on the labels as delivered and under injected damage. ginkgo_pr_cho and ginkgo_colloidal are 23 and 22 of 24.
- n18 (VERIFIED): SMAC colloidal stability, svm on base_1, removing the 20% of antibodies the `cross_val` detector flags: 0.2740 → 0.40336 Spearman. Key `/per_space/base_1/per_detector/cross_val/0.2/per_model/svm/remove/cleaned_mean` in `noise/n18_clean_the_real_labels/ginkgo_colloidal.json`. The 0.2740 start appears in the same file. Fold 0 only (`folds_run: [0]`).
- A size caveat to state beside these. Ginkgo sets hold 197 to 242 antibodies, and each fold holds out about 50 (outline 7.8). By the usual approximation, a Spearman on 50 pairs moves by about 1/sqrt(49) = 0.14 by chance. That is my arithmetic, not a measured spread. Every Ginkgo noise number above is one fold.

### 3.2 Her hypothesis: larger molecules leave more room for distinct featurisation

The hypothesis predicts two things that can be read off existing results. Both go the other way.

(a) If antibody representations were more distinct, antibody hybrids should draw on more representations, more evenly.

Source: `complexity/c22_explainability/<ds>.json`, `per_fold[*]/models/<model>/hybrids[slot=hybrid1]/allocation_built`. VERIFIED. 5 folds × 2 models = 10 hybrids per dataset. Every pool holds 10 representations, except ddg_combined with 8.
- "Representations used" counts representations given at least one feature.
- "Effective count" is exp(entropy) of the feature shares. It equals 10 when 10 representations get equal shares and 1 when one takes everything. This summary is mine, not one in the repo.

| domain | dataset | representations used, median | largest share, median | effective count, median |
|---|---|---|---|---|
| small molecule | caco2 | 10 | 0.12 | 9.92 |
| small molecule | herg_ki | 10 | 0.12 | 9.95 |
| small molecule | adme_mdr1 | 10 | 0.13 | 9.65 |
| small molecule | adme_hlm | 10 | 0.19 | 9.16 |
| small molecule | herg | 3 | 0.35 | 2.99 |
| small molecule | qm9 | 3 | 0.43 | 2.89 |
| antibody | ginkgo_pr_cho | 10 | 0.13 | 9.23 |
| antibody | abagym_VEGF | 10 | 0.16 | 8.99 |
| antibody | ginkgo_acsins | 10 | 0.19 | 8.07 |
| antibody | ginkgo_hic | 10 | 0.22 | 7.63 |
| antibody | abagym_EGFR | 10 | 0.28 | 7.53 |
| antibody | flab_binding_koenig | 10 | 0.27 | 7.38 |
| antibody | flab_thermo_tresanco | 7 | 0.21 | 6.07 |
| antibody | abagym_Ang2 | 7.5 | 0.30 | 5.74 |
| antibody | ddg_combined | 6.5 | 0.28 | 5.64 |
| antibody | ginkgo_titer | 5 | 0.35 | 4.68 |
| antibody | ginkgo_colloidal | 5 | 0.32 | 4.53 |
| antibody | cdr_flexibility | 3 | 0.38 | 2.98 |
| antibody | humanness_therapeutic | 3 | 0.42 | 2.96 |
| antibody | abcov_kd | 3 | 0.43 | 2.94 |
| reaction | buchwald_hartwig | 2 | 0.59 | 1.96 |
| expression | tcga_brca_pam50 | 3 | 0.54 | 2.72 |

logd, adme_rlm and flab_expression_koenig have partial c22 files only and are not in the table.

Small-molecule hybrids on caco2, herg_ki and the ADME sets spread their features over all 10 representations almost evenly. Antibody hybrids range from that down to 3 representations. Antibody hybrids do not draw on more representations than small-molecule ones.

(b) If antibody representations were more distinct, their pairwise distances should agree less.

Source: `tests/experiments/results/f26_redundancy_penalty_<ds>.json`, key `redundancy/pairdist/matrix`. This is the Spearman between two representations' pairwise distances over the same pairs. VERIFIED file read; median of the off-diagonal entries computed here. These are local runs of 8 Sep on representation pools that predate the signed-off ones (outline 4.2.5 flags this).
- ginkgo_hic, 13 representations: 0.42
- flab_thermo_tresanco, 13 representations: 0.509
- herg_ki, 11 representations: 0.315
- logd, 12 representations: 0.388
- buchwald_hartwig, 17 representations: 0.452
- tcga_luad_vs_lusc (cut), 14 representations: 0.106

On these two antibody sets the representations agree with each other more than on the two small-molecule sets. That is the reverse of "more room for distinct featurisation".

What the antibody hybrids select (c22, PAGE short-answer table; VERIFIED for ddg_combined's raw allocation):
- The largest SHAP share comes from a protein or antibody language model on every antibody set: esm3, esm2_concat, prot_t5, saprot, ablang2, igt5, antiberty, abroberta_cdr, or esm2_wt_delta on ddg_combined.
- Small molecules: mordred leads on qm9, logd and herg. mhg_gnn, smited, pdv or molformer leads on caco2, herg_ki and the ADME sets.
- None of the antibody features can be named chemically. The feature-name registry resolves an embedding feature only to "representation + dimension index". VERIFIED in c22: every `top_features` entry on ddg_combined fold 0 has `interpretation.type = embedding_dim`. A blueprint on antibodies therefore says which models were used, not which residues.

Confound to state. The antibody sets are also the small ones: 141 (abcov_kd) to 672 (flab_thermo_tresanco), except the two Koenig sets at 4,275 (outline 5.3.1). The small-molecule sets are 1,412 to 9,978. Molecule size and dataset size move together, so these results cannot separate them. The two Koenig sets are the only large antibody sets. There the hybrid wins with both models, and flab_expression_koenig svm is the one antibody case better on 5 of 5 folds.

A dated memory note (`project_hybrid_root_cause_2026_07_14.md`, July) says "21/21 antibody sets underpowered", with fold spread 0.13-0.45. UNVERIFIED and on older runs, but consistent with the fold ranges above.

---

## 4. "The method generalises to any domain": every non-molecular domain in the repo

### 4.1 In the thesis runs

**Reactions: buchwald_hartwig** (3,955 reactions, yield %, RMSE, grouped by additive; models rf and lightgbm)
- Hybrid against best single, 5 folds (VERIFIED c02 `heldout_delta_by_model`): rf -0.8327 RMSE, better on 2/5 folds, range -2.646..+1.028. lightgbm +0.5468, better on 4/5. PAGE c02.md: rf 14.4056 → 15.2383, lightgbm 16.1679 → 15.6211.
- Fold spread is larger than either mean. The outline (5.5a) records a LightGBM fold spread of 1.6198 RMSE in f24.
- The hybrid draws on 2 representations (median), with diff_mhg_gnn the largest share (c22 VERIFIED allocation; PAGE SHAP 0.467 rf, 0.430 lightgbm).
- Distance against yield difference: best single 0.1351 Spearman (rxnfp, Manhattan), 5 folds. VERIFIED s01.
- Prediction from 5 nearest neighbours: hybrid worse at every setting, 14.4572 → 16.1811 RMSE at the fixed setting, better at 0 of 18 settings (PAGE s05.md).
- Against principal components, mRMR, recursive elimination: the outline (5.5) says the hybrid beats none of them on Buchwald-Hartwig. Not rechecked here.
- Noise: n09 has 8 kind×level combinations (fold 0). The method passes for most of the 12 (PAGE n09.md row). n07, n18 and n20 have rows (PAGE).
- Verdict: the hybrid runs; it neither clearly helps nor hurts with these models.

**Gene expression: tcga_brca_pam50** (821 patients, 4 subtypes, weighted F1; models dnn and svm)
- Hybrid against best single, 5 folds (VERIFIED): dnn +0.0482, better on 4/5 (the fifth equal, +0.000). svm +0.0141, 4/5, also +0.000 on the fifth. PAGE: dnn 0.8528 → 0.9011, svm 0.8808 → 0.8950.
- Against two ensembles: the hybrid beats both (outline 5.4, PAGE c11.md row).
- Distance against subtype: best single 0.4723 (gene_geneformer_v2, cosine), the highest of any dataset. 7 of 30 representation-distance pairs at or below zero. VERIFIED s01.
- Hybrid uses 3 representations (median), gene_geneformer_v2 the largest share (0.527 to 0.629 SHAP, PAGE c22).
- Noise: no completed n09 file. Only `tcga_brca_pam50__uniform.rows.jsonl` and `.passes.jsonl` exist, no json. Every noise page row is n/a. NOT YET RUN.
- Verdict: the cleanest non-molecular success for prediction. No noise results.

**Nucleotides (cut 2026-10-08, plan.CUT_DATASETS)**
- `tests/results/nucleotide_survey/*.json` (22 files, Mar-Apr 2026): single-representation baseline surveys only. No hybrid. VERIFIED (nt_promoter_tata.json holds `baselines` per representation; `metadata.n_reps` 1).
- In the current run, nt_promoter_tata has partial files only:
  - c22, fold 0 only (VERIFIED, `complexity/c22_explainability/nt_promoter_tata.partial.jsonl`): lightgbm mistral_dna 0.6268 → hybrid 0.7060 MCC; svm mistral_dna 0.6569 → 0.6842 MCC.
  - c18 has folds 0-2 partial.
  - c01-c02 precompute error file says `result_status: empty`.
- One fold, cut. Usable at most as "the search runs on DNA sequences".

### 4.2 Cut non-molecular sets with results

- tz_her2_binary and tz_her2_multiclass (antibody, trastuzumab HER2): full c02 results, 5 folds, hybrid worse with at least one model on each (table in 3.1). Cut.
- bh_doyle_plates (reactions): c02 rf +0.5557 (4/5), lightgbm -0.6527 (2/5), ranges over ±3 RMSE. Cut.
- tcga_luad_vs_lusc: `tests/results/complexity/c02_complexity_matching_grid/tcga_luad_vs_lusc.json`, 1 June, an old result format with no `heldout_delta_by_model`. Predates the current pipeline. Cut.

### 4.3 Spinoffs and side projects (memory files checked against files)

| project | memory says | what exists on disk | results? |
|---|---|---|---|
| Financial (`~/repos/kirby-finance`) | Roadmap; classical time-series models as representations; KIRBy hybrid search | Last commit 4 Apr 2026. One result file: `results/surveys/rv_forecast_quick_summary.json` (19 Mar). Task: 30-day realised volatility, 9 tickers, 11,066 rows, 16 walk-forward folds, rf and xgboost | Single-representation survey only. Best single spectral 0.0967 RMSE (rf); all four joined 0.0953 (rf). No hybrid search was run. VERIFIED |
| Genomics / nucleotides | "Not a spinoff", 22 representations × 22 tasks | See 4.1. Baseline surveys plus partial current-run files | No hybrid result beyond one fold of nt_promoter_tata |
| Autonomous vehicles (`~/repos/kirby-av`) | AV2 scenario mining; ARC jobs to download data | Last commit 7 May 2026: data download and rsync scripts. `src/` has search and representation code. No result files found | NONE. Memory's job tracker ends at data download |
| Cross-species (vet) | Vet ChEMBL targets; foundation_hybrid_vet | `tests/results/cross_species/foundation_hybrid_vet_full.json` (15 May): cow_CA4 1,250, cow_DRD2 250, cow_PTGS2 27 (3 folds), dog_PTGS2 107 (3 folds). VERIFIED | Small molecules, not a new domain. Every `hybrid/per_model/<m>/best_rmse` is None, so no hybrid score was stored. The single-representation baselines exist |
| ANNITIA (MASLD survival, Trustii challenge) | View-subset search for survival; leaderboard 0.9302 | `/Volumes/seagate/kirby/annitia/annitia_{hepatic,death}_search.json` (4 May). VERIFIED | Not find_best_hybrid: it joins whole feature views (`tests/experiments/annitia/search.py`, "mirrors find_best_hybrid pattern"). Hepatic (1,106 patients, 39 events, 5 folds): best single view nit + rsf 0.8840 C-index; best combination delta + nit 0.8914. Death (966, 76 events): best single missingness + coxnet 0.9437; best combination 0.9465. Both picked as the best of 189 conditions on the same folds, so optimistic. Gains are smaller than the fold SD (0.0617 and 0.0175) |
| ROGII (wellbore geology, Kaggle; `~/repos/kirby-rogii`) | ANNITIA-pattern playbook | Last commit 8 May. EDA outputs under `data/eda/`. `EXPERIMENTS.csv` has a header and no rows | NONE |

What can be said honestly: in the thesis runs the hybrid has been tested on small molecules, antibodies, one reaction set and one gene-expression set. It helps on the expression set and is neutral on the reaction set. The four spinoffs give no hybrid-search result. ANNITIA is a view-combination search, and its gains sit inside the fold spread. "Any domain" is not supported. "Not limited to molecules" is supported by one expression set, plus one fold of DNA promoters.

---

## 5. SMAC colloidal stability against HIC retention, same 242 antibodies

### 5.1 Same antibodies

VERIFIED as far as the files allow:
- `src/kirby/datasets/ginkgo_dev.py:80-127`: both load one GDPa1 table and keep rows with the property, both chains and a fold ID present.
- c02 `input_fingerprint`: ginkgo_colloidal and ginkgo_hic both have n_samples 242, n_groups 5, groups_hash 9830554105668594874. labels_hash differs. ginkgo_acsins has the same groups_hash, so it is the same 242 as well.
- Sequences were not compared directly.

### 5.2 "The answer is opposite" (outline 7.8 and 8.2a)

The outline sentence: on SMAC, cleaning is the strongest result in the chapter. On HIC, neither hybrid beats the single best representation and there is no cleaning result at all.

That compares a cleaning result on one dataset with the absence of any cleaning result on the other. Two things disagree here:
- The outline calls this an opposite answer. HIC has no noise results in any experiment. I searched `noise/`, `noise_full/` and the noise_view folder and found no ginkgo_hic file. The ledger lists it under complexity, still running.
- On the one question both have answered, the hybrid against the best single representation, they give the SAME answer. The hybrid loses with both models on both. VERIFIED c02, 5 folds:
  - SMAC: dnn -0.0088, svm -0.0005.
  - HIC: gp -0.0486, dnn -0.0018.

So the claim cannot be made from current results. Status: NOT FOUND as stated.

Where the two endpoints do give opposite answers, over the same 242 antibodies and the same 5 folds:
- Adding the augmentation representations in the distance search (c02 Q5, PAGE c02.md). SMAC: worse with dnn (0.3224 → 0.1593) and svm (0.3837 → 0.3430), better with kNN (0.1918 → 0.2056). HIC: better with all three (gp 0.4980 → 0.5151, dnn 0.4538 → 0.4600, kNN 0.3301 → 0.3612). I could not trace these numbers to a single key in the raw file; they come from the page builder.
- Across the complexity answer pages, I compared every yes/no column where both datasets have an answer. PAGE, script inline in this session. They differ in 31 of 79 such answers:
  - c01: 2 of 5. c02: 4 of 12. c04: 3 of 10. c06: 0 of 3. c08: 1 of 6. c09: 5 of 11. c18: 7 of 14. c19: 2 of 4. c21: 5 of 8. c22: 2 of 6.
  - Many flips are tiny. Example: c22's svm hybrid1 against base1 is "yes" on SMAC at 0.3446 → 0.3448 Spearman. A yes/no with no margin flips on noise.
- The distance-against-activity comparison cannot be made: s01 has no ginkgo_hic file.

If the author wants the pair to carry 8.2a, it needs HIC's noise results (n07, n09, n18) to finish. Until then, the honest version is "same antibodies, the hybrid loses on both, and the endpoints disagree on about two in five of the complexity questions".

---

## 6. Future directions in outline 8.7: what exists to support each

**More QM9 properties as clean controls.**
- The loader already supports 12 QM9 properties (`src/kirby/datasets/qm9.py:101`, `load_qm9(..., property_idx=4)`; names at line 128: mu, alpha, homo, lumo, gap, r2, zpve, U0, U, H, G, Cv). index 4 is the gap.
- Only the gap is registered as a dataset (`qm9`). A second property needs a new dataset entry plus its representations and runs. No results exist for any other property.
- QM9 in n09: censoring and grouped_shifted have json files. gaussian, outlier_p05 and shift_only_p10 have rows and passes files only, no json (VERIFIED by listing).

**Classification and label flipping.**
- Partly done already. n09 runs six flip kinds on herg (7,772), cdr_flexibility (328) and humanness_therapeutic (564), fold 0 (VERIFIED file names).
- n07 has herg flip rows (PAGE n07.md, herg confusion_directed at 0.05-0.4).
- fed01 runs label flips across five silos on Caco-2 (PAGE).
- The future direction should say what is still missing: flips that depend on the molecule near a class threshold, more folds, and more classification sets. Not "classification" as a whole.

**Admitting extra features (the decoy test, and extras that must displace base features).**
- Outline 8.7 says the displacing rule "has no results". This is wrong, and it contradicts outline 5.5a.
- `tests/experiments/framework/f24_augmentation_admission.py:27` defines `capped` as "extras must DISPLACE base features". Result files with `capped` in the name exist for buchwald_hartwig, caco2, flab_expression_koenig, tcga_brca_pam50 and herg_ki (VERIFIED by listing `tests/experiments/results/framework/`).
- 5.5a quotes one: Buchwald-Hartwig with LightGBM, extras displacing base features, 108 of 259 configurations beat no extras.
- What is true: augmentation is off in the full runs, and only 3 folds ran. Status: CHANGED.

**Hybrid features for generative models.**
- Built, not run. `tests/experiments/opig_generator_scoring_swap.py` (untracked in git) plus `tests/experiments/opig_generator_scoring/` (REINVENT 4 scorer, published configuration).
- Arms in the docstring: ecfp4 as the control; the hybrid's blueprint; a random matrix of the same width; the best single representation. The generator is steered to maximise predicted hERG pKi.
- No result files found under `tests/results` or the 2026-10-09 folder.

**A faster search.**
- Done in part. Per-dataset cut grids: on hERG Ki the final grid keeps 224 configurations and 1.4% of the recorded fitting time, and keeps the best configuration (thesis.tex:2288, from `/Volumes/seagate/slim_precompute/herg_ki.slim.db`; UNVERIFIED here).
- Fast tier and adaptive pruning exist in `find_best_hybrid.py` (outline 5.2.5). Pruning is off by default since 2026-09-13. The outline records a 30% cut in hours on the f33 record (130.4 to 91.5 hours over five datasets) with no best score changed; UNVERIFIED here.
- Recipe transfer as a warm start: opig_blueprint_transfer (LogD recipe on hERG Ki beats hERG Ki's best single, loses to its own search; outline 5.7) and c24.
- What is open: a warm start from a similar dataset's blueprint, and measured hours for full search against cut grid (thesis.tex:5058 `[RESULT NEEDED]`).

**A fuller federated deployment.**
- Current results are Caco-2 and LogD only.
  - fed02, fed03, fed05 have json for both. fed04 has caco2.json only.
  - fed01 has partial jsonl only; its page says Caco-2 has 3 finished dnn seeds and 2 svm, LogD none.
- lhasa_herg was cut on 8 Oct, and with it the Lhasa sites. fed05 (antibody sites) was cut by the author on 9 Oct (outline 5.8).
- The FLuID dependency and the training-set-as-transfer-pool compromise are recorded in outline Appendix E.
- So this direction stands. "A fuller deployment" now also means restoring the real-site data, since only simulated silos on public data remain.

---

## 7. Items for the author to decide (one per line)

- Step 8: rewrite the "never the best on either" sentence to the weaker form, or drop it from 8.2 and 8.2a.
- Step 4: drop "hash collisions" from the measured evidence, or run the planned statistic.
- Step 2: it rests on one sentence of the confirmation report with no raw data on this machine; say so, or retrieve the server CSVs.
- 8.2a's strongest evidence: SMAC against HIC cannot carry "opposite answer" until HIC has noise results.
- 8.7: delete "has no results" from the extra-features bullet.
- 8.7: reword classification as "molecule-dependent flips near a threshold, on more folds".
- The antibody discussion: decide whether "works unusually well" refers to the noise methods (supported, one fold) or to the hybrid (not supported for consistency).
- Domain generality: the spinoffs give no hybrid result; decide whether to mention them at all.
