# Chapter 7 results: what the 9 October noise results say, checked against the outline

Written by Claude on 2026-10-09. A reading of result files and analysis pages, not a measured result of its own. Nothing here has been written into the thesis or the outline.

## What changes the outline (summary; detail in sections 1 to 7)

1. **Reversed or wrong bullets in 7.5 to 7.11.**
   - "Retuning did not help" (7.10): it helped in 34 of 44 rows.
   - "The best method on the hybrid is never the best on either single representation, all 9 datasets" (7.11): false on caco2, cdr_flexibility and herg, of 13 datasets.
   - "No removal or shrinking method beat the damaged labels under any damage kind" (7.6): now true of Caco-2 only.
   - The "flip rate of 0.1" in 7.6 is a clipped share of 0.1 on Caco-2.
   - "Scaffold-offset methods did nothing on the real labels" (7.7): they made Caco-2 and LogD worse.
   - "Binding free energy and titer have no shuffled control" (7.8): they now do, and some cleanups pass it.
   - The SVR on SMAC colloidal is no longer the exception to "hybrid keeps its lead". Titer, humanness, flab_thermo_tresanco and abcov_kd are.
2. **Clipped-label corrections: the strongest result, with a sharper reading.** They are strong on caco2, herg_ki, logd, qm9, flab_thermo_tresanco and buchwald_hartwig (rf). Their only control deals the same lifts to the wrong clipped molecules. The correctly assigned per-molecule lift passes it in 1 of 22 rows, and on every clipped row of caco2, herg_ki, logd and qm9 the shuffled lifts score better. Raising the flagged labels by the right average amount is what works.
3. **Real labels, small molecules.** The only controlled gain larger than 0.012 RMSE is hERG Ki label smoothing with the neural network: 0.638 → 0.578 single, 0.599 → 0.578 hybrid, against a shuffled control of 0.6419 / 0.5877. Caco-2 and LogD have no strong real-label row at all.
4. **New result not in the outline.** Deleting flagged molecules on Buchwald-Hartwig's real labels beats random deletion: lightgbm hybrid 20.0823 → 18.0417 RMSE against 19.8986. VERIFIED.
5. **The antibody exception, named.**
   - ginkgo_colloidal: strongest, with controls.
   - ginkgo_titer, cdr_flexibility, abcov_kd: strong, partly controlled.
   - ginkgo_pr_cho: strong, uncontrolled.
   - Not: humanness_therapeutic, flab_thermo_tresanco, ginkgo_acsins.
   - The strong rule's +0.05 Spearman on about 53 held-out antibodies is easier to reach than 5% RMSE, so the size of the exception is inflated by the rule itself.
6. **QM9.** In the n09 run, every one of the twelve published methods makes clean QM9 worse. In QM9_CONTROL.md, deleting the molecules the n07 and n24 detectors flag never helps the hybrid (0 of 18 and 0 of 36 settings across both models). Label smoothing helps the network's hybrid by medians of about 0.003 RMSE, inside a fit-to-fit wobble of up to 0.013 (0.3484 to 0.3618).
7. **Hybrid after cleaning.** On small molecules it wins in 95% of rows, but it already won in 92% before cleaning. On other domains it is close to a coin toss before and after.
8. **Analysis-page defects.** MEETING_METHODS uses the cliff put-back control where n25 marks it unusable (caco2, herg, logd), and "cliffs" are 54% to 86% of training molecules there. Hand-written summaries on six pages are stale and are the source of several outline bullets.

## 0. Sources and how to read this file

- **Main source:** `/Volumes/seagate/claude_analysis_2026-10-08/answers/MEETING_METHODS.md` (built 9 Oct 13:45 from the result files in `/Volumes/seagate/kirby_results_2026-10-09_noise_view/noise`, header line 1 of `MEETING_METHODS_detail.md`). I parsed every one of its 4,258 table rows with a throwaway script in the session scratchpad. That script is not in the repo; it only splits the page's tables and recounts them.
- **Spot checks against raw result files** are marked VERIFIED, with the file and key. Numbers seen only on an analysis page are marked PAGE: those tables were rebuilt on 9 Oct from the raw files, but I did not re-open the raw file myself. Numbers seen only in the outline or a hand-written page summary are marked UNVERIFIED.
- **Folds:** every number below is outer fold 0 only, unless stated. Rows under added noise average 1 to 3 noise draws; real-label rows are one fit.
- **Rows:** "a row" is one line of a MEETING_METHODS table: one dataset, one noise type, one model. Each row compares the best single representation (base_1 or base_2) and the best hybrid (hybrid or hybrid__general2), before and after the method.
- **"Yes"** (the page's "does it work?"): the method improved both the single representation and the hybrid, and beat the control where the table has one. There is no minimum size: 0.228 → 0.228 rounded can count.
- **"Strong" (*)**: yes, and the smaller of the two gains is at least 5% of the starting RMSE, +0.05 Spearman, or +0.01 AUROC.
- **Three standing caveats on every count in this file.**
  1. Where a method has several settings (detector, flag rate, strength, neighbour setting), the page keeps the setting with the best held-out score after treatment. Picking on the test score makes every count lean optimistic.
  2. The single representation and the hybrid in each row are also picked after treatment (best of two each), so the comparison is twice selected on the held-out score.
  3. The neural network's untreated score moves between jobs that should be identical. QM9 example in section 5: the second hybrid scores 0.3484, 0.3568 or 0.3618 RMSE depending on the job (VERIFIED, `kirby_results_2026-10-09/noise/{n07,n09}/qm9*.rows.jsonl`, field `score_D`, level 0, fold 0). lightgbm gives the same score in every job. So small neural-network gains can be fit-to-fit wobble.

## 1. The "Overall: does it work, by group" table

### 1a. Re-derived: the page's counts reproduce exactly

The page groups by keyword matching on section titles and method names (`answers/_scripts/meeting_readable.py:436-452`, read in full this session). Recounting the 4,258 rows of MEETING_METHODS.md with the same rule gives the same numbers. Status: SAME.

| page group | yes (hybrid better / hybrid worse) | single representation only | hybrid only | no | rows | strong rows (*) |
|---|---|---|---|---|---|---|
| training with noisy labels | 653 (507 / 146) | 297 | 304 | 477 | 1,731 | 143 |
| reweighting | 544 (396 / 148) | 109 | 156 | 184 | 993 | 93 |
| noise mitigation | 691 (544 / 147) | 163 | 190 | 490 | 1,534 | 187 |

The strong-row column is not on the page; I added it from the same parse.

### 1b. Things the counts hide

- **Rows with no score are counted as "no".** 114 rows have "n/a" in a score cell. 84 are in the page's noise mitigation group (kNN label replacement on hERG, 42; n07 out-of-fold and James-Stein-style moves, 20; the two clipped-label lifts on datasets with no assay limit, 22). The other 30 are the cliff-aware weights in n02, which ran on hybrids only, so they can never be "yes". Without them: noise mitigation "no" is 406 of 1,450 rows, reweighting "no" is 154 of 963.
- **"Hybrid worse" includes ties.** The rule at `meeting_readable.py:478` counts anything but "yes" as worse. 5 rows are exact ties.
- **Many "yes" rows are tiny.** In 585 of the 1,374 "yes" rows of the outline's noise mitigation group (next subsection), the smaller of the two gains is under 1% of the starting RMSE, or under 0.01 Spearman or AUROC.

### 1c. The same rows under the outline's three groups

The outline (lines 525-656) puts a method in noise mitigation when something selects molecules (a detector, a known flag, or every molecule) and one of three actions follows: delete, count less, or move the label. Methods with no action are noise-robust training. I took each method's selector and action from `answers/_scripts/method_groups.json`, a Claude reading of the code dated 2026-10-09, not checked against the code by me. Four page sections have no entry in that file, and I classified them myself:
- DivideMix: noise-robust training.
- Label smoothing toward the nearest neighbours (n24): a detector, then move the label.
- kNN label replacement: a detector, then move the label.
- Detect and remove: a detector, then delete.

| outline group | yes (hybrid better / hybrid worse) | single only | hybrid only | no | rows | strong |
|---|---|---|---|---|---|---|
| noise mitigation (delete, count less, or move the label) | 1,374 (1,043 / 331) | 359 | 426 | 874 | 3,033 | 313 |
| noise-robust training (no action on the data) | 514 (404 / 110) | 210 | 224 | 277 | 1,225 | 110 |

Rows that move between groups:
- All 993 rows of the page's reweighting group join noise mitigation as "count less". This assumes the outline's open decision is settled the recommended way; see the OPEN POINT at outline line ~620.
- 506 rows of the page's "training with noisy labels" group move to noise mitigation:
  - the four selection methods from the published sweep of twelve: small-loss selection, confident learning, distance-based selection, multistage cleaning;
  - shrinking flagged labels toward a prediction (n32), one weight or a weight per molecule;
  - moving every label toward a Huber-fitted prediction (`huber_teacher_blend`) and the Yeo-Johnson transform (n33);
  - the measurement-count shrink (`calibrated`, n19).
- No row moves the other way.

The outline's noise mitigation group, split by what selects the molecules and what is done to them:

| selects | action | yes | strong | rows | yes share |
|---|---|---|---|---|---|
| a detector | delete | 413 | 119 | 986 | 0.42 |
| a detector | count less | 454 | 81 | 781 | 0.58 |
| a detector | move the label | 284 | 58 | 750 | 0.38 |
| a known flag | delete | 5 | 2 | 22 | 0.23 |
| a known flag | count less | 51 | 2 | 116 | 0.44 |
| a known flag | move the label | 130 | 45 | 304 | 0.43 |
| every molecule | move the label | 37 | 6 | 74 | 0.50 |

Noise-robust training splits into nothing selects (1,147 rows, 488 yes, 109 strong), a known flag (12 rows, the measurement-count feature methods on herg_ic50, 5 yes), and every molecule (66 rows, 21 yes, 1 strong).

The highest strong share is known flag, then move the label: 45 of 304 rows. Almost all of those are the clipped-label corrections. Section 2 shows that their only control says the per-molecule lift amounts do not matter.

### 1d. Where a control exists

Only 9 mitigation methods have a numeric control column. Across them, 68 rows pass both the method and the control:
- n18, shuffled suspicion scores: downweight 8 yes, label_smooth 7, knn_smooth 4, oof_smooth 5, gp_smooth 4.
- Detect and remove against random removal: 6 yes.
- n24 delete against random deletion: 21 yes.
- n30 per-molecule lift against the same lifts shuffled among the clipped molecules: 1 yes.
- n25 cliff put-back against random molecules put back: 12 yes, but see the defect in section 6.

Three columns are labelled as controls but hold no values:
- n32 shrinking, both versions: 96 rows. The n32 page says there is no random control for shrinking.
- n24 downweight, Huber weight, knn_smooth and oof_smooth: 164 rows. Only deletion has a control there.
- The n18 constant lift: 6 rows.

## 2. Noise mitigation: which methods work at all

### 2a. Per dataset, all mitigation methods pooled (outline grouping)

One row is one dataset. The columns count table rows on MEETING_METHODS.md. "Hybrid better after" is the share of all its rows where the cleaned hybrid beats the cleaned single representation.

| dataset | domain | rows | yes (share) | strong (share) | real labels: rows / yes / strong | hybrid better after |
|---|---|---|---|---|---|---|
| caco2 | small molecule | 533 | 237 (0.44) | 23 (0.04) | 105 / 28 / 0 | 0.99 |
| herg_ki | small molecule | 476 | 198 (0.42) | 32 (0.07) | 99 / 18 / 3 | 0.87 |
| logd | small molecule | 364 | 151 (0.41) | 25 (0.07) | 97 / 22 / 0 | 0.98 |
| herg (classification) | small molecule | 303 | 173 (0.57) | 48 (0.16) | 55 / 21 / 1 | 0.92 |
| qm9 | small molecule, clean labels | 44 | 9 (0.20) | 6 (0.14) | 19 / 2 / 0 | 0.77 |
| herg_ic50 | small molecule | 16 | 6 | 3 | 16 / 6 / 3 | 1.00 |
| buchwald_hartwig | reaction | 291 | 151 (0.52) | 21 (0.07) | 73 / 24 / 6 | 0.56 |
| ginkgo_colloidal | antibody | 275 | 110 (0.40) | 34 (0.12) | 73 / 29 / 18 | 0.67 |
| ddg_combined | antibody | 171 | 81 (0.47) | 14 (0.08) | 53 / 23 / 6 | 0.50 |
| cdr_flexibility | antibody | 169 | 75 (0.44) | 43 (0.25) | 35 / 12 / 7 | 0.63 |
| ginkgo_titer | antibody | 138 | 83 (0.60) | 33 (0.24) | 48 / 19 / 9 | 0.29 |
| humanness_therapeutic | antibody | 90 | 30 (0.33) | 6 (0.07) | 22 / 1 / 0 | 0.09 |
| flab_thermo_tresanco | antibody | 58 | 27 (0.47) | 8 (0.14) | 28 / 6 / 0 | 0.17 |
| ginkgo_acsins | antibody | 43 | 17 (0.40) | 2 (0.05) | 19 / 8 / 2 | 0.77 |
| abcov_kd | antibody | 36 | 16 (0.44) | 9 (0.25) | 20 / 6 / 3 | 0.31 |
| ginkgo_pr_cho | antibody | 22 | 10 (0.45) | 6 (0.27) | 6 / 4 / 2 | 0.77 |
| tcga_brca_pam50 | genomics | 4 | 0 | 0 | 4 / 0 / 0 | 0.00 |

The small-molecule regression datasets have a yes share of about 0.4. Their real labels give zero strong rows on caco2 and logd and 3 on herg_ki. The main exceptions are three antibody datasets: on cdr_flexibility, ginkgo_titer and ginkgo_colloidal the strong share is 2 to 6 times higher (section 4). Before reading that as an antibody advantage, read section 4's caveat: the strong rule is not equally hard across metrics.

### 2b. Per method


### Noise mitigation (outline grouping)

One row is one method as named on MEETING_METHODS.md. "rows" counts (dataset, noise type, model) table rows. "yes" = helped both the best single representation and the best hybrid (and beat the control where one exists). "strong" = the page's * rule. "real" = rows on labels with no noise added. Source: MEETING_METHODS.md, parsed by Claude 2026-10-09; fold 0 only.

| method (page section :: name) | selects | action | rows | yes | strong | real-label rows | real-label yes | control | datasets with a yes row |
|---|---|---|---|---|---|---|---|---|---|
| Neighbour suspicion weights with importance-scaled features (n12) :: importance-scaled features | a detector | count less | 129 | 84 | 17 | 23 | 10 | none | abcov_kd, buchwald_hartwig, caco2, cdr_flexibility, ddg_combined, flab_thermo_tresanco, ginkgo_acsins, ginkgo_colloidal, ginkgo_titer, herg, herg_ki, logd |
| Neighbour suspicion weights with importance-scaled features (n12) :: unscaled features | a detector | count less | 129 | 77 | 14 | 23 | 8 | none | abcov_kd, buchwald_hartwig, caco2, cdr_flexibility, ddg_combined, flab_thermo_tresanco, ginkgo_acsins, ginkgo_colloidal, ginkgo_titer, herg, herg_ki, logd |
| Refitting with weights from a noise detector (n13) :: detector weights | a detector | count less | 100 | 69 | 6 | 18 | 9 | none | buchwald_hartwig, caco2, cdr_flexibility, ddg_combined, ginkgo_colloidal, herg, herg_ki, logd, qm9 |
| Neighbour suspicion weights: which distance (n21) :: best distance | a detector | count less | 74 | 61 | 24 | 16 | 13 | none | buchwald_hartwig, caco2, cdr_flexibility, ddg_combined, ginkgo_colloidal, herg, herg_ki, logd |
| Other cleanups of damaged labels (n07) :: remove | a detector | delete | 106 | 51 | 18 | 18 | 4 | none | buchwald_hartwig, caco2, cdr_flexibility, ginkgo_colloidal, ginkgo_titer, herg, herg_ki, humanness_therapeutic, logd |
| Other cleanups of damaged labels (n07) :: label_smooth | a detector | move the label | 106 | 48 | 9 | 18 | 1 | none | buchwald_hartwig, caco2, cdr_flexibility, ginkgo_colloidal, ginkgo_titer, herg, herg_ki, humanness_therapeutic, logd |
| Other cleanups of damaged labels (n07) :: downweight | a detector | count less | 106 | 45 | 3 | 18 | 3 | none | buchwald_hartwig, caco2, cdr_flexibility, ginkgo_colloidal, ginkgo_titer, herg, herg_ki, humanness_therapeutic, logd |
| Cleaning before the representation search (n20) :: search and fit on cleaned labels | a detector | delete | 48 | 43 | 21 | 18 | 13 | none | buchwald_hartwig, caco2, ddg_combined, ginkgo_colloidal, ginkgo_titer, herg, herg_ki, logd |
| Other noisy-label training methods for the neural network (n09, n13) :: confident_learning | a detector | delete | 122 | 41 | 9 | 22 | 8 | none | buchwald_hartwig, caco2, cdr_flexibility, ddg_combined, ginkgo_acsins, ginkgo_colloidal, ginkgo_pr_cho, ginkgo_titer, herg, herg_ki, logd |
| Cleaning by a model's own per-molecule uncertainty (n23) :: activity_cliff_aware | a detector | delete | 52 | 38 | 9 | 8 | 2 | none | caco2, herg, herg_ki, logd |
| Cleaning by a model's own per-molecule uncertainty (n23) :: gaussian_process | a detector | delete | 52 | 36 | 13 | 8 | 2 | none | caco2, herg, herg_ki, logd |
| Cleaning by a model's own per-molecule uncertainty (n23) :: hetero_gp | a detector | delete | 52 | 32 | 2 | 8 | 3 | none | caco2, herg, herg_ki, logd |
| Cleaning by a model's own per-molecule uncertainty (n23) :: heteroscedastic_nn | a detector | delete | 52 | 31 | 5 | 8 | 4 | none | caco2, herg, herg_ki, logd |
| Label smoothing toward the nearest neighbours | a detector | move the label | 48 | 30 | 6 | 10 | 3 | none | buchwald_hartwig, caco2, herg, herg_ki, logd |
| Delete or keep the flagged molecules (n24) :: huber_weight | a detector | count less | 48 | 28 | 2 | 10 | 2 | column present, no values | buchwald_hartwig, caco2, herg, herg_ki, logd |
| Other cleanups of damaged labels (n07) :: js_shrink | a detector | move the label | 64 | 27 | 1 | 12 | 2 | none | caco2, ginkgo_colloidal, ginkgo_titer, herg_ki, logd |
| Delete or keep the flagged molecules (n24) :: downweight | a detector | count less | 48 | 27 | 2 | 10 | 1 | column present, no values | buchwald_hartwig, caco2, herg, herg_ki, logd |
| Several sources per molecule: offsets and weights (n39) :: offset_removed_within_var_weights | a known flag | move the label | 54 | 27 | 0 | 0 | 0 | none | buchwald_hartwig, caco2, ginkgo_colloidal, herg_ki, logd |
| Other noisy-label training methods for the neural network (n09, n13) :: small_loss | a detector | delete | 122 | 26 | 9 | 22 | 4 | none | buchwald_hartwig, caco2, cdr_flexibility, ddg_combined, ginkgo_colloidal, ginkgo_pr_cho, ginkgo_titer, herg_ki, humanness_therapeutic |
| Other cleanups of damaged labels (n07) :: oof_smooth | a detector | move the label | 64 | 26 | 2 | 12 | 1 | none | caco2, ginkgo_colloidal, ginkgo_titer, herg_ki, logd |
| Cleaning by a model's own per-molecule uncertainty (n23) :: ngboost | a detector | delete | 52 | 26 | 4 | 8 | 3 | none | caco2, herg, herg_ki, logd |
| Several sources per molecule: offsets and weights (n39) :: weights_n19_fitted | a known flag | count less | 54 | 25 | 0 | 0 | 0 | none | buchwald_hartwig, caco2, ginkgo_colloidal, herg_ki, logd |
| Other noisy-label training methods for the neural network (n09, n13) :: distance_selection | a detector | delete | 72 | 23 | 5 | 13 | 6 | none | buchwald_hartwig, caco2, cdr_flexibility, ddg_combined, ginkgo_acsins, ginkgo_colloidal, ginkgo_pr_cho, ginkgo_titer, herg, herg_ki, logd |
| Several sources per molecule: offsets and weights (n39) :: weights_within_var | a known flag | count less | 54 | 22 | 0 | 0 | 0 | none | buchwald_hartwig, caco2, ginkgo_colloidal, herg_ki, logd |
| Down-weighting disagreeing near-duplicate pairs (n35) :: downweight_pairs | a detector | count less | 32 | 21 | 5 | 16 | 6 | none | buchwald_hartwig, caco2, ddg_combined, flab_thermo_tresanco, ginkgo_colloidal, ginkgo_titer, herg_ki, logd |
| Delete or keep the flagged molecules (n24) :: delete | a detector | delete | 48 | 21 | 10 | 10 | 2 | random deletion | buchwald_hartwig, caco2, herg, herg_ki |
| Several sources per molecule: offsets and weights (n39) :: offset_removed | a known flag | move the label | 54 | 21 | 0 | 0 | 0 | none | buchwald_hartwig, caco2, ginkgo_colloidal, herg_ki, logd |
| Trimmed and two-learner selection, and shrinking flagged labels (n32) :: shrink, one weight | a detector | move the label | 48 | 20 | 6 | 10 | 5 | column present, no values | buchwald_hartwig, ddg_combined, flab_thermo_tresanco, ginkgo_colloidal, ginkgo_titer, herg_ki |
| Trimmed and two-learner selection, and shrinking flagged labels (n32) :: shrink, weight per molecule | a detector | move the label | 48 | 19 | 4 | 10 | 4 | column present, no values | buchwald_hartwig, caco2, ddg_combined, flab_thermo_tresanco, ginkgo_colloidal, ginkgo_titer, herg_ki |
| Changing how the model reads the labels (n33) :: huber_teacher_blend | every molecule | move the label | 32 | 19 | 4 | 8 | 2 | none | buchwald_hartwig, caco2, herg_ki, logd |
| Acting on a residual detector's flags (n31) :: downweight | a detector | count less | 30 | 18 | 2 | 6 | 3 | none | caco2, ddg_combined, herg_ki |
| Down-weighting molecules that disagree with their neighbours (n02) :: neighbour-agreement weights | a detector | count less | 30 | 16 | 3 | 30 | 16 | none | abcov_kd, buchwald_hartwig, caco2, cdr_flexibility, flab_thermo_tresanco, ginkgo_acsins, ginkgo_colloidal, ginkgo_pr_cho, ginkgo_titer, herg, qm9 |
| kNN label replacement | a detector | move the label | 106 | 16 | 3 | 18 | 1 | none | buchwald_hartwig, ginkgo_colloidal, ginkgo_titer, herg_ki, logd |
| Delete or keep the flagged molecules (n24) :: oof_smooth | a detector | move the label | 34 | 16 | 0 | 8 | 0 | column present, no values | buchwald_hartwig, caco2, herg_ki, logd |
| Other noisy-label training methods for the neural network (n09, n13) :: multistage_cleaning | a detector | delete | 122 | 15 | 6 | 22 | 4 | none | buchwald_hartwig, cdr_flexibility, ginkgo_colloidal, ginkgo_pr_cho, ginkgo_titer, humanness_therapeutic |
| Other treatments of labels clipped at an assay limit (n30) :: truncated_normal_mean | a known flag | move the label | 22 | 15 | 7 | 6 | 3 | column present, no values | buchwald_hartwig, caco2, flab_thermo_tresanco, herg_ki, logd, qm9 |
| Fixing labels clipped at an assay limit :: Censored EM | a known flag | move the label | 32 | 15 | 11 | 16 | 3 | none | buchwald_hartwig, caco2, flab_thermo_tresanco, herg_ki, logd, qm9 |
| Changing how the model reads the labels (n33) :: yeo_johnson | every molecule | move the label | 32 | 15 | 0 | 8 | 3 | none | buchwald_hartwig, caco2, herg_ki, logd |
| Delete or keep the flagged molecules (n24) :: knn_smooth | a detector | move the label | 34 | 13 | 4 | 8 | 2 | column present, no values | buchwald_hartwig, caco2, herg_ki, logd |
| Other cleanups of damaged labels (n07) :: Constant lift | a known flag | move the label | 20 | 12 | 6 | 6 | 2 | none | caco2, ginkgo_colloidal, ginkgo_titer, herg_ki, logd |
| Fixing labels clipped at an assay limit :: Constant lift | a known flag | move the label | 32 | 12 | 11 | 16 | 1 | none | buchwald_hartwig, caco2, flab_thermo_tresanco, herg_ki, logd, qm9 |
| Putting activity-cliff molecules back after a cleanup (n25) :: cliff_put_back | a detector | delete | 30 | 12 | 1 | 8 | 6 | random molecules put back | caco2, herg, herg_ki, logd |
| Acting on a residual detector's flags (n31) :: remove | a detector | delete | 30 | 12 | 4 | 6 | 2 | none | caco2, ddg_combined, herg_ki |
| Other treatments of labels clipped at an assay limit (n30) :: winsor | a detector | move the label | 32 | 10 | 0 | 16 | 5 | column present, no values | buchwald_hartwig, caco2, ddg_combined, flab_thermo_tresanco, ginkgo_colloidal, herg_ki |
| Removing a shared offset per scaffold group (n34) :: random_intercept | a known flag | move the label | 18 | 10 | 5 | 6 | 0 | none | caco2, herg_ki, logd |
| Acting on a residual detector's flags (n31) :: rewrite | a detector | move the label | 30 | 10 | 4 | 6 | 2 | none | caco2, ddg_combined, herg_ki |
| Acting on a residual detector's flags (n31) :: bayes_shrink | a detector | move the label | 30 | 10 | 2 | 6 | 1 | none | caco2, ddg_combined, herg_ki |
| Detector and repair grid (n11) :: best repair | a detector | move the label | 10 | 10 | 10 | 4 | 4 | none | ginkgo_colloidal, ginkgo_titer |
| Other cleanups of damaged labels (n07) :: gp_smooth | a detector | move the label | 18 | 9 | 1 | 4 | 1 | none | ginkgo_colloidal, ginkgo_titer |
| Other cleanups of the real labels (n18) :: downweight | a detector | count less | 25 | 8 | 3 | 25 | 8 | shuffled suspicion scores | abcov_kd, caco2, cdr_flexibility, ddg_combined, ginkgo_acsins, ginkgo_colloidal, herg, logd |
| Other cleanups of the real labels (n18) :: label_smooth | a detector | move the label | 26 | 7 | 3 | 26 | 7 | shuffled suspicion scores | abcov_kd, cdr_flexibility, ddg_combined, ginkgo_colloidal, herg, herg_ki, humanness_therapeutic |
| Removing a shared offset per scaffold group (n34) :: group_center_3plus | a known flag | move the label | 18 | 7 | 2 | 6 | 1 | none | caco2, herg_ki, logd |
| Detect and remove (checked against random removal) | a detector | delete | 26 | 6 | 3 | 26 | 6 | random removal | cdr_flexibility, flab_thermo_tresanco, ginkgo_colloidal, ginkgo_titer |
| Removing a shared offset per scaffold group (n34) :: group_center | a known flag | move the label | 18 | 6 | 2 | 6 | 0 | none | caco2, herg_ki, logd |
| Other cleanups of the real labels (n18) :: oof_smooth | a detector | move the label | 20 | 5 | 1 | 20 | 5 | shuffled suspicion scores | abcov_kd, caco2, flab_thermo_tresanco, ginkgo_colloidal, logd |
| Other treatments of labels clipped at an assay limit (n30) :: drop_clipped | a known flag | delete | 22 | 5 | 2 | 6 | 2 | column present, no values | caco2, ddg_combined, flab_thermo_tresanco, ginkgo_colloidal, logd |
| Other cleanups of the real labels (n18) :: knn_smooth | a detector | move the label | 20 | 4 | 0 | 20 | 4 | shuffled suspicion scores | buchwald_hartwig, ddg_combined, ginkgo_acsins |
| Other cleanups of the real labels (n18) :: gp_smooth | a detector | move the label | 12 | 4 | 2 | 12 | 4 | shuffled suspicion scores | ddg_combined, ginkgo_colloidal, ginkgo_titer |
| Moving training labels toward an out-of-fold prediction (n27) :: toward_prediction | every molecule | move the label | 10 | 3 | 2 | 10 | 3 | none | buchwald_hartwig, ginkgo_colloidal, herg_ki |
| Other cleanups of the real labels (n18) :: Constant lift | a known flag | move the label | 6 | 2 | 0 | 6 | 2 | column present, no values | caco2, logd |
| Using how often each molecule was measured (n19, herg_ic50 only) :: count_weighted | a known flag | count less | 4 | 2 | 2 | 4 | 2 | none | herg_ic50 |
| Using how often each molecule was measured (n19, herg_ic50 only) :: shrunk | a known flag | move the label | 4 | 2 | 1 | 4 | 2 | none | herg_ic50 |
| Using how often each molecule was measured (n19, herg_ic50 only) :: spread_pooled_weighted | a known flag | count less | 4 | 2 | 0 | 4 | 2 | none | herg_ic50 |
| Other treatments of labels clipped at an assay limit (n30) :: per_molecule_lift | a known flag | move the label | 22 | 1 | 0 | 6 | 1 | shuffled lifts | caco2 |
| Down-weighting molecules that disagree with their neighbours (n02) :: cliff-aware weights | a detector | count less | 30 | 0 | 0 | 30 | 0 | none |  |
| Using how often each molecule was measured (n19, herg_ic50 only) :: calibrated | a known flag | move the label | 4 | 0 | 0 | 4 | 0 | none |  |

### Noise-robust training (outline grouping)

One row is one method as named on MEETING_METHODS.md. "rows" counts (dataset, noise type, model) table rows. "yes" = helped both the best single representation and the best hybrid (and beat the control where one exists). "strong" = the page's * rule. "real" = rows on labels with no noise added. Source: MEETING_METHODS.md, parsed by Claude 2026-10-09; fold 0 only.

| method (page section :: name) | selects | action | rows | yes | strong | real-label rows | real-label yes | control | datasets with a yes row |
|---|---|---|---|---|---|---|---|---|---|
| Other noisy-label training methods for the neural network (n09, n13) :: uncertainty_curriculum | nothing | none | 122 | 45 | 10 | 22 | 5 | none | buchwald_hartwig, caco2, cdr_flexibility, ddg_combined, ginkgo_acsins, ginkgo_colloidal, ginkgo_pr_cho, ginkgo_titer, herg, herg_ki, logd, qm9 |
| Other noisy-label training methods for the neural network (n09, n13) :: meta_weight_net | nothing | none | 122 | 45 | 7 | 22 | 7 | none | buchwald_hartwig, caco2, cdr_flexibility, ddg_combined, ginkgo_acsins, ginkgo_colloidal, ginkgo_pr_cho, herg, herg_ki, logd, qm9 |
| Other noisy-label training methods for the neural network (n09, n13) :: heteroscedastic_gp | nothing | none | 88 | 45 | 11 | 17 | 9 | none | buchwald_hartwig, caco2, ddg_combined, ginkgo_acsins, ginkgo_colloidal, ginkgo_pr_cho, ginkgo_titer, herg_ki, logd, qm9 |
| Other noisy-label training methods for the neural network (n09, n13) :: early_learning | nothing | none | 122 | 44 | 9 | 22 | 7 | none | buchwald_hartwig, caco2, cdr_flexibility, ddg_combined, ginkgo_acsins, ginkgo_colloidal, ginkgo_pr_cho, ginkgo_titer, herg_ki, logd, qm9 |
| Other noisy-label training methods for the neural network (n09, n13) :: mentornet | nothing | none | 122 | 44 | 8 | 22 | 5 | none | buchwald_hartwig, caco2, cdr_flexibility, ddg_combined, ginkgo_acsins, ginkgo_colloidal, ginkgo_pr_cho, ginkgo_titer, herg, herg_ki, logd, qm9 |
| Other noisy-label training methods for the neural network (n09, n13) :: contrast_to_divide | nothing | none | 122 | 40 | 10 | 22 | 7 | none | buchwald_hartwig, caco2, cdr_flexibility, ddg_combined, ginkgo_acsins, ginkgo_colloidal, ginkgo_pr_cho, ginkgo_titer, herg_ki, logd, qm9 |
| Other noisy-label training methods for the neural network (n09, n13) :: evidential_kernel | nothing | none | 88 | 39 | 11 | 17 | 7 | none | buchwald_hartwig, caco2, ddg_combined, ginkgo_acsins, ginkgo_colloidal, ginkgo_pr_cho, ginkgo_titer, herg_ki, logd, qm9 |
| Changing the model instead of the labels (n40) :: stack | nothing | none | 44 | 39 | 12 | 6 | 5 | none | caco2, herg_ki, logd |
| Changing the model instead of the labels (n40) :: retuned | nothing | none | 44 | 34 | 3 | 6 | 4 | none | caco2, herg_ki, logd |
| DivideMix (training two networks on each other's clean molecules) | nothing | none | 72 | 23 | 5 | 13 | 3 | none | buchwald_hartwig, caco2, cdr_flexibility, ddg_combined, ginkgo_acsins, ginkgo_colloidal, ginkgo_pr_cho, ginkgo_titer, herg_ki, qm9 |
| Changing the model instead of the labels (n40) :: tuned_clean_reused | nothing | none | 44 | 21 | 2 | 6 | 4 | none | caco2, herg_ki, logd |
| Changing the model instead of the labels (n40) :: seed_avg | nothing | none | 22 | 19 | 9 | 3 | 2 | none | caco2, herg_ki, logd |
| Trimmed and two-learner selection, and shrinking flagged labels (n32) :: two_learner | nothing | none | 48 | 15 | 6 | 10 | 3 | random molecules | buchwald_hartwig, ddg_combined, flab_thermo_tresanco, ginkgo_colloidal, ginkgo_titer |
| Changing how the model reads the labels (n33) :: target_noise_bagging | every molecule | none | 32 | 15 | 1 | 8 | 2 | none | caco2, herg_ki, logd |
| Trimmed and two-learner selection, and shrinking flagged labels (n32) :: itlm | nothing | none | 48 | 13 | 6 | 10 | 4 | random molecules | buchwald_hartwig, ddg_combined, flab_thermo_tresanco, ginkgo_colloidal, ginkgo_titer, herg_ki |
| Huber loss under damage sized by the model's own error (n38) :: robust_loss | nothing | none | 24 | 11 | 0 | 12 | 2 | none | abcov_kd, buchwald_hartwig, ddg_combined, ginkgo_acsins, ginkgo_colloidal, ginkgo_titer, herg_ki, logd |
| Changing how the model reads the labels (n33) :: huber_dnn | nothing | none | 15 | 11 | 0 | 3 | 2 | none | caco2, herg_ki, logd |
| Changing how the model reads the labels (n33) :: c_mixup | every molecule | none | 32 | 4 | 0 | 8 | 2 | none | caco2, herg_ki, logd |
| Using how often each molecule was measured (n19, herg_ic50 only) :: ngboost_count_feature | a known flag | none | 2 | 2 | 0 | 2 | 2 | none | herg_ic50 |
| Training on every measurement instead of the mean (n29, herg_ic50 only) :: per_record | every molecule | none | 2 | 2 | 0 | 2 | 2 | none | herg_ic50 |
| Using how often each molecule was measured (n19, herg_ic50 only) :: count_feature | a known flag | none | 4 | 1 | 0 | 4 | 1 | none | herg_ic50 |
| Using how often each molecule was measured (n19, herg_ic50 only) :: invvar_and_count_feature | a known flag | none | 4 | 1 | 0 | 4 | 1 | none | herg_ic50 |
| Using how often each molecule was measured (n19, herg_ic50 only) :: xgboost_interval_aft | a known flag | none | 2 | 1 | 0 | 2 | 1 | none | herg_ic50 |


### 2c. Headline mitigation results, with numbers

Each result names the detector or flag, the action and the control. Fold 0 throughout.

**1. Raising labels clipped at the assay limit (a known flag, move the label).** This is the strongest and most consistent result in the noise work. Clipping at a share of 0.25, censored EM (`Fixing labels clipped at an assay limit`) is strong on both models on:
- caco2;
- herg_ki;
- logd;
- qm9;
- flab_thermo_tresanco;
- buchwald_hartwig with rf.
It fails on ginkgo_colloidal and ddg_combined.

Raw numbers, VERIFIED in `kirby_results_2026-10-09_noise_view/noise/n30_censored_likelihood/<ds>.json`, `per_level['0.25'][config][arm][model]`. One noise draw (`noise_draws: [0]`), fold 0.

| dataset, configuration, model | labels before clipping | clipped | constant lift | per-molecule lift | same lifts dealt to other clipped molecules (best / worst of 3) |
|---|---|---|---|---|---|
| logd, hybrid, dnn | 0.4059 | 0.6049 | 0.4604 | 0.4988 | 0.4632 / 0.4714 |
| herg_ki, base_2, dnn | 0.5891 | 0.7485 | 0.6662 | 0.7016 | 0.6704 / 0.6945 |
| caco2, hybrid, dnn | 0.2459 | 0.3042 | 0.2751 | 0.2878 | 0.2708 / 0.2750 |
| qm9, hybrid, dnn | 0.3513 | 0.5196 | 0.4503 | 0.4481 | 0.4399 / 0.4495 |

All numbers are RMSE. On logd the constant lift recovers (0.6049 - 0.4604) / (0.6049 - 0.4059) = 73% of the RMSE lost to clipping. That is for this hybrid, which is the same in every column of n30.

**The control result that matters.** The only control here deals the same per-molecule lifts to other clipped molecules (`per_molecule_lift_shuffled`, n30:55-75). The correctly assigned per-molecule lift beat it, and no cleaning, in only 1 of 22 MEETING rows (caco2, real labels). In the raw rows above, even the worst of the three shuffles beats the correct assignment on logd and herg_ki.

So what helps is knowing which labels are clipped and raising them by about the right average amount. Giving each molecule its own amount does not help. This supports the outline's claim that the flag carries the gain. No experiment tests the flag itself, for example by raising the same number of unclipped labels.

**2. Deleting flagged molecules under hERG label flips (a detector, delete, against random deletion).** VERIFIED in `n24_delete_or_keep/herg__*.json`, `per_level['0.1']`, arms `delete/remove` against `delete_at_random/seed0-4`, 3 noise draws. The 6 flip kinds × 4 configurations × 2 models × 12 detector and flag-rate settings give 576 settings:
- 412 beat no cleaning;
- 397 of those also beat the best of 5 random deletions;
- 15 did not.

Example: xgboost on the hybrid, confusion-directed flips, neighbour-label-gap detector, 10% flagged: 0.8028 → 0.8375 AUROC against 0.8179 for the best random deletion. On the real hERG labels, 2 of 96 settings beat no cleaning in each file, and 1 also beat random.

**3. Removing a shared offset per scaffold group under grouped shift (a known flag: the scaffold group; move the label).** Random intercept is strong on caco2, herg_ki and logd. VERIFIED in `n34_group_random_effects/caco2__grouped_shifted.json`, `per_condition['grouped_shifted']['1'][config][arm][model]`, 3 draws:
- caco2 dnn, hybrid: 0.3423 → 0.2800 RMSE (clean labels 0.2374);
- caco2 dnn, hybrid__general2: 0.3329 → 0.2798.

On the real labels, the plain per-group centring and random intercept made every small-molecule dataset worse:
- caco2 dnn hybrid: 0.238 → 0.263;
- logd dnn hybrid: 0.395 → 0.483.

The one real-label "yes" is centring groups of 3 or more on herg_ki dnn: 0.638 → 0.618 single, 0.612 → 0.589 hybrid. No control exists. n34 ran only on caco2, herg_ki and logd. Every other file is a skip record: "condition_not_in_n34", or no SMILES for buchwald_hartwig.

**4. Cleaning the real labels, against the same detector with shuffled scores (n18, the only real-label control).** Small molecules:
- herg_ki, neural network, label smoothing: 0.638 → 0.578 single, 0.599 → 0.578 hybrid. Shuffled control 0.6419 / 0.5877 (PAGE, MEETING_METHODS.md line 1457). The outline's 0.6380 → 0.5775 is the same result at four decimals (n18.md / n25.md level-0 tables).
- caco2 dnn downweight passes: 0.266 → 0.255, 0.238 → 0.233.
- caco2 dnn out-of-fold move passes: 0.266 → 0.254, 0.238 → 0.234.
- logd dnn downweight passes: 0.421 → 0.414, 0.396 → 0.391.
- logd dnn out-of-fold move passes: 0.421 → 0.417, 0.396 → 0.391.
- herg passes for xgboost label smoothing (0.837 → 0.846, 0.855 → 0.860 AUROC) and lightgbm downweight (0.850 → 0.854, 0.855 → 0.859).

Every caco2 and logd gain above is 0.012 RMSE or less. On caco2 that is below the 0.014 RMSE the outline gives for the cache fault's spread. The svm passes nothing on caco2, herg_ki or logd.

Antibodies: ginkgo_colloidal is the clearest case (section 4).

**5. Buchwald-Hartwig real labels, deletion against random deletion.** This is new and not in the outline. VERIFIED in `n24_delete_or_keep/buchwald_hartwig__censoring.json`, `per_level['0']`. lightgbm:
- base_2: 20.2242 → 18.3284 RMSE against best random 19.9173;
- hybrid: 20.0823 → 18.0417 against 19.8986.
Both use the neighbour-label-gap detector with 20% flagged. Random deletion alone helps base_1 (19.4875 → 18.8663), so the yield RMSE moves between fits here as well.

**6. Cleaning before the representation search (n20).** By the page rule it now passes almost everywhere (section 3, 7.9). It has no control, and its "yes" rule needs only one of about 40 cleanup, cleaning-representation and flag-rate settings to win on half the configurations.

### 2d. Does it work on the real labels with nothing added?

Small molecules, all mitigation methods pooled: 97 of 371 rows are yes (0.26), and 7 are strong. That is the lowest yes share of any noise type. By action:
- delete: 33 of 123;
- count less: 35 of 94;
- move the label: 29 of 154.

Yes share by noise type for small molecules, mitigation only (rows with an n/a score left out):

| noise | yes / rows | strong |
|---|---|---|
| none added (real labels) | 97 / 371 | 7 |
| clipped at the assay limit, 25% | 84 / 234 | 30 (28 of them from moving the label) |
| 5% outliers | 104 / 169 | 10 |
| random noise on every label | 86 / 161 | 7 |
| 10% of labels shifted | 69 / 173 | 3 |
| 10% shifted upward | 29 / 85 | 0 |
| whole chemical series shifted | 98 / 139 | 27 |
| hERG label flips, all six kinds | 152 / 236 | 47 |
| simulated multiple sources (n39) | 46 / 120 | 0 |

## 3. Outline 7.5 to 7.11, bullet by bullet

Status words:
- SAME: the number or verdict still holds.
- CHANGED: the current results differ.
- NOT FOUND: no current file carries it.
- NOT YET RUN.

Every row is fold 0 unless it says otherwise.

### 7.5 Changing a label against deleting a molecule

**7.5 bullet 1: moving labels worked on Caco-2, hERG, hERG Ki and LogD, on the real labels and under damage.**
- Outline: moving a flagged label toward its neighbours' mean or toward an out-of-fold prediction worked on the labels as delivered and under damage, on all four. hERG Ki, network on the hybrid: 0.5989 → 0.5671.
- Now, the number: VERIFIED. It is n24 label smoothing on herg_ki, real labels, dnn on the hybrid, cross_val@screen3. It appears in the outlier_p05 and shift_only_p10 files. The censoring file's own level-0 copy reads 0.5972 → 0.5706.
- Now, the verdict: on the real labels the single representation did not improve (base_2 0.5892 → 0.5965 at best). So the page scores this row "hybrid only".
- Real labels, with the shuffled-score control (n18):
  - herg_ki passes, network only.
  - caco2 and logd pass only by 0.012 RMSE or less, network only.
  - herg passes with xgboost label smoothing, by 0.009 and 0.005 AUROC.
- Under damage, n24 label smoothing passes for the network on caco2, herg_ki and logd, and for both tree models under every hERG flip kind. The svm moves RMSE by 0.0053 or less.
- Status: number SAME; verdict CHANGED. On real labels the gains are hybrid-only on hERG Ki and within noise on Caco-2 and LogD.
- Source: `n24_delete_or_keep/herg_ki__{outlier_p05,shift_only_p10,censoring}.json`, `per_level['0']['hybrid'][det][rate]['keep_rewrite/label_smooth']['dnn']`; MEETING_METHODS.md lines 3467-3500 and 1427-1530.

**7.5 bullet 2: deletion fails on the real labels and passes under hERG flips and LogD damage.**
- Outline: under the random-deletion control, deleting fails on the real labels of Caco-2, hERG, hERG Ki and LogD. It passes under hERG flips (0.8028 → 0.8375 against 0.8179), and under LogD censoring and outliers.
- Now:
  - Real labels: fails on all four. SAME.
  - hERG: VERIFIED, 0.8028 → 0.8375 against 0.8179.
  - LogD censoring and outliers: only the hybrid improves (censoring dnn hybrid 0.622 → 0.609; outliers dnn hybrid 0.585 → 0.546). The single representation either fails or loses to random deletion, so the page says "hybrid only".
  - New: Buchwald-Hartwig real labels pass with both models against random deletion (section 2c, result 5). hERG Ki under censoring is a pass with svm, but tiny (0.760 → 0.757, 0.746 → 0.745).
- Status: hERG SAME; LogD CHANGED to hybrid only; Buchwald-Hartwig is new.
- Source: `n24_delete_or_keep/herg__confusion_directed.json`, `per_level['0.1']['hybrid']['neighbor_residual@screen1']['0.1']`, 3 draws; MEETING lines 3643-3690.

**7.5 bullet 3: "198 picked methods; deleting 171 of 264 pairs; changing a label 813 of 1,320; 0.648 against 0.616."**
- No current page or script produces these counts. The definition of "picked" is not written down.
- Nearest equivalent, small molecules, real labels, MEETING rows: delete 33 of 123, move 29 of 154 (section 2d). Pooled over every dataset and noise type, the outline's mitigation group: detector-delete 413 of 986, detector-move 284 of 750.
- Status: NOT FOUND. The bullet's point, that neither action is a clear winner, still holds in the per-row counts.
- Source: section 1c above.

**7.5 bullet 4: on the Ginkgo sets, removing flagged antibodies is the strongest result in the chapter.**
- Now: removal is among the strong results on ginkgo_colloidal and ginkgo_titer. The largest gains on those two datasets now come from:
  - n11's best detector and action: colloidal dnn single 0.065 → 0.486, hybrid 0.227 → 0.518;
  - n20 cleaning before the search: colloidal dnn hybrid -0.014 → 0.434.
- Neither n11 nor n20 has a control. Removal with the shuffled control (n18): colloidal svm base_1 0.2740 → 0.4034 against 0.2782, VERIFIED.
- Status: CHANGED in wording. "Strongest controlled result" still fits colloidal removal. "Strongest result" does not.
- Source: MEETING lines 5306, 2848, 4196; `n18_clean_the_real_labels/ginkgo_colloidal.json`, `per_space.base_1.per_detector.cross_val['0.2'].per_model.svm`.

### 7.6 Which detectors find the suspect labels

**7.6 bullet 1: the out-of-fold forest residual and the neighbour-label gap drive almost every success.**
- Now: partly by construction. n07, n18, n24 and n25 offer only these two detectors. n24 calls them cross_val and neighbor_residual; n07 uses cross_val.
- Where other detectors were offered, they also pass. n23's model-uncertainty detectors pass in 31 to 38 of 52 rows each (section 2b).
- Status: CHANGED. State it as "the two detectors offered in most experiments".

**7.6 bullet 2: a Gaussian process's own predicted spread worked on hERG and hERG Ki.**
- Now: n23 gaussian_process is yes in:
  - caco2: 9 of 14 rows;
  - herg: 12 of 14;
  - herg_ki: 8 of 12;
  - logd: 7 of 12.
- 13 of those rows are strong: 6 hERG flip kinds with xgboost; 4 under grouped shift (caco2 dnn, herg_ki dnn, logd dnn and svm); 2 under outliers (herg_ki and logd, dnn); 1 under gaussian noise (logd svm). Real labels: yes on herg (1 row) and herg_ki (1 row). No control.
- Status: CHANGED. It now also works on caco2 and logd under damage.
- Source: MEETING section 22, lines 4466-4790.

**7.6 bullet 3: the three feature-only detectors work only under per-scaffold-group offsets and are undefined under random damage.**
- No current page states this. n41's files mark "detection_undefined" for gaussian and grouped_shifted damage, but for a different reason: those kinds move every training label (n41.md lines 1370-1392).
- Status: NOT FOUND, not re-checked.

**7.6 bullet 4: finding them is not fixing them.**
- Outline: on Caco-2 the Shapley-value detector under forest proximity reached 0.8106-0.8921 AUROC at a clipped share of 0.25, and 0.9577-0.9681 "at a flip rate of 0.1". No removal or shrinking method beat the damaged labels on half the pairs under any damage kind.
- Now, the AUROC: both ranges are on n32.md, Caco-2 censoring, read from a checkpoint (no final `caco2__censoring.json`). The 0.9577-0.9681 range is at a clipped share of 0.1, NOT a flip rate. Caco-2 is regression and has no flips (n32.md line 616).
- Now, the fixing claim: n32's answer table now has yes for:
  - shrinking under damage on buchwald_hartwig, ddg_combined, flab_thermo_tresanco, ginkgo_colloidal, ginkgo_titer, herg_ki and logd;
  - trimmed refits under damage on buchwald_hartwig, flab_thermo_tresanco, ginkgo_colloidal and logd.
  Caco-2 is still no everywhere.
- Status: AUROC numbers SAME (PAGE, checkpoint); the "flip rate" wording is WRONG; the "no method fixed it" claim is CHANGED and now true of Caco-2 only.
- Source: n32.md lines 600-630 and its answer table.

**7.6 bullet 5: conformal p-values flag too few molecules to act on; the floor is written into each record.**
- Now: n41 flags at q 0.05 or 0.1 on buchwald_hartwig, caco2 and logd (column 5 "yes"); herg_ki does not. Neither winsorising nor removal on the flags passes on any of the 4 datasets.
- Status: "flags too few" is CHANGED; "acting on the flags fails" is SAME; "the floor in each record" is NOT FOUND on the page.
- Source: n41.md answer table.

### 7.7 By noise type

**7.7 bullet 1: clipping is the most fixable.**
- Outline: raising clipped labels worked on Caco-2, hERG Ki and LogD. hERG Ki, network, second base representation: 0.7485 → 0.6662 at 0.25.
- Now: VERIFIED. It now also works on qm9, flab_thermo_tresanco and buchwald_hartwig (rf), and fails on ginkgo_colloidal and ddg_combined.
- Status: SAME, and wider.
- Source: `n30_censored_likelihood/herg_ki.json`, `per_level['0.25']['base_2']['constant_lift']['dnn']`, 1 draw.

**7.7 bullet 2: clipping is fixable because no detector is needed.**
- The per-molecule control result (section 2c) supports this, and adds that the per-molecule amount does not matter.
- Status: SAME, with an addition.

**7.7 bullet 3: offset per scaffold group is next most fixable; these methods did nothing on the real labels.**
- Outline: Caco-2, network on the hybrid, 0.3328 → 0.2866.
- Now: caco2 dnn hybrid__general2 0.3329 → 0.2798; hybrid 0.3423 → 0.2800 (random intercept), 3 draws. On the real labels the same methods made Caco-2 and LogD WORSE (for example 0.395 → 0.483 on LogD), not "nothing".
- Status: number CHANGED (bigger gain); "did nothing on the real labels" is CHANGED (they hurt).
- Source: `n34_group_random_effects/caco2__grouped_shifted.json`, `per_condition`; MEETING lines 5014-5090.

**7.7 bullet 4: Gaussian damage and outliers were fixed less often; most passes come from moving a label.**
- Now, small molecules (section 2d):
  - Gaussian: 86 of 161 rows yes (0.53). Delete 36, move 29, count less 21.
  - Outliers: 104 of 169 yes (0.62). Delete 43, move 39.
  - Clipping: 84 of 234 (0.36).
- The yes share is not lower than for clipping. Strong rows are rarer: 7 and 10, against 30 for clipping.
- Status: CHANGED. "Fixed less strongly", not "less often"; and deleting passes about as often as moving.

### 7.8 By dataset

**7.8 bullet 1: SMAC colloidal stability is the standout on the real labels.**
- Outline: 0.2740 → 0.4034 against 0.2782; 112 settings beat the shuffled control; "7 or 8 of 8 pairs".
- Now: 0.2740 → 0.4034 against 0.2782 VERIFIED (cross_val, flag rate 0.2). 112 is VERIFIED as the sum of the "beat control" column on n18.md's ginkgo_colloidal table. Fold 0 holds out 53 of 242 antibodies (n18.md).
- The "7 or 8 of 8 pairs" I could not reproduce. At flag rate 0.2, removal beats the shuffled control on 5 of 8 configuration-and-model pairs.
- Under damage: n07 removal works on colloidal under censoring, gaussian, outlier_p05 and shift_only_p10 (CLEANING_BY_METHOD.md, PAGE).
- Status: SAME, except the "7 or 8 of 8" is NOT FOUND.

**7.8 bullet 2: the SMAC/HIC pair.**
- Now: ginkgo_hic has no noise result file of any kind in `kirby_results_2026-10-09/noise`. Checked by listing every experiment folder.
- Status: SAME. HIC still has no cleaning result.

**7.8 bullet 3: hERG Ki is the most consistent dataset.**
- Outline: relabelling cleaned the real labels, 0.6380 → 0.5775. Putting cliff molecules back beat random on all 8 pairs.
- Now: 0.638 → 0.578 single, 0.599 → 0.578 hybrid, beats the shuffled control (PAGE).
- Cliff put-back on herg_ki real labels: dnn 0.607 → 0.591 single, 0.596 → 0.579 hybrid, against random put-back 0.6005 / 0.5949; svm passes by 0.001. n25.md reports 313 of 528 settings beating random put-back at level 0 (censoring file) and 326 of 528 (shift file).
- I could not reproduce "all 8 pairs". hERG Ki is the only dataset where the random put-back control is usable (section 6).
- Status: SAME, except "8 pairs" NOT RE-DERIVED. "Most consistent" is not supported against LogD or Caco-2: herg_ki's mitigation yes share is 0.42, against 0.41 for logd and 0.44 for caco2.

**7.8 bullet 4: hERG deleting and relabelling work under every flip kind; 397 settings beat random, 15 did not.**
- Now: 397 of 412 VERIFIED (section 2c, result 2). Relabelling has no random-deletion control. Its passes (label smoothing, every flip kind with xgboost, 6 strong rows) are uncontrolled.
- Status: SAME for deleting. Say that relabelling has no control.

**7.8 bullet 5: Caco-2 has many passes, mostly small.**
- Now: caco2 has 237 yes rows of 533 but only 23 strong (0.04). On the real labels, 0 strong.
- Status: SAME.

**7.8 bullet 6: binding free energy and expression titer have large real-label gains and no shuffled-score control.**
- Now: n18 HAS run on ddg_combined and ginkgo_titer with the shuffled control. Passing rows:
  - ddg dnn: downweight, label smoothing, kNN replacement, GP move (for example label smoothing 0.341 → 0.484 single, 0.259 → 0.287 hybrid against 0.4595 / 0.2759).
  - titer rf: GP move 0.308 → 0.364, 0.224 → 0.390 against 0.3116 / 0.2861.
  - titer dnn and rf: removal (Detect and remove section).
- Status: CHANGED. The control now exists, and some cleanups pass it.
- Source: MEETING lines 1399-1548, 4198-4199.

### 7.9 Cleaning inside the feature search

**7.9 bullet 1: all three questions are yes for both models on all five datasets that ran.**
- Outline: Caco-2, binding free energy, SMAC colloidal, expression titer, hERG Ki. Largest: SMAC colloidal, network on the second hybrid, -0.0143 → 0.5596.
- Now: 9 datasets have results. buchwald_hartwig, cdr_flexibility, herg and logd are added, with real labels only for those four.
- Of the 84 answered yes/no cells on n20's answer table, 83 are yes. The one no: buchwald_hartwig, question 2 ("search on cleaned labels too"), model 2.
- -0.0143 → 0.5596 is on n20.md line 2336 (PAGE). It is question 3 ("search on cleaned labels, fit on the labels as they are"), the James-Stein-style shrink, cleaned in base_2. The starting score is near zero.
- Caveat: a dataset is "yes" if any one of about 40 settings wins on half the configurations, so the rule passes easily.
- Status: CHANGED. Wider coverage, one no.

**7.9 bullet 2: no control has run.**
- Now: n20.md line 75: "random-flag controls exist in the code but were not run".
- Status: SAME.

### 7.10 Not cleaning at all

**7.10 bullet 1: ten network seeds and stacking beat the plain model, on Caco-2 and hERG Ki. 0.7184 → 0.6557.**
- Now: 0.7184 → 0.6557 is on n40.md line 1130 (herg_ki gaussian 1.0, dnn hybrid__general2, PAGE).
- n40 now also has LogD. Seed averaging is yes in 19 of 22 rows; stacking in 39 of 44.
- On the real labels:
  - seed averaging helps Caco-2 and LogD, and helps only the hybrid on hERG Ki;
  - stacking helps all three with svm and caco2/logd with dnn.
- Stacking against an svm baseline compares a two-model stack with the weaker single model.
- Status: SAME, and wider (LogD).
- Source: MEETING lines 4910-4996.

**7.10 bullet 2: target-noise bagging did the same.**
- Now: yes in 15 of 32 rows; the svm fails on Caco-2 (MEETING n33 target_noise_bagging).
- Status: PARTLY. It is weaker than seed averaging and stacking.

**7.10 bullet 3: weighting by each source's error variance beat a plain mean on Caco-2, hERG Ki and LogD.**
- Now: weights_within_var is yes in 22 of 54 rows; weights_n19_fitted in 25 of 54. The yes rows cover buchwald_hartwig, caco2, ginkgo_colloidal, herg_ki and logd. 0 strong rows. All are simulated sources.
- Status: SAME in direction; the effect is small. By the outline's own grouping this is noise mitigation (a known flag, count less), not "not cleaning".

**7.10 bullet 4: retuning the model on the damaged labels did not help.**
- Now: retuned is yes in 34 of 44 rows on caco2, herg_ki and logd. Even n40.md's stale summary says retuning lowered RMSE on both datasets. Under censoring it helped on only 1 or 2 of 8 pairs.
- Status: CHANGED (reversed except under censoring).
- Source: MEETING lines 4798-4853.

**7.10 bullet 5: none of these has a control.**
- Status: SAME.

### 7.11 Model and representation

**7.11 bullet 1: the network benefits far more than the SVR; the SVR often fails on Caco-2, hERG Ki and LogD.**
- Now: mitigation yes share, dnn against svm:

| dataset | dnn | svm |
|---|---|---|
| caco2 | 159 / 291 (0.55) | 78 / 242 (0.32) |
| herg_ki | 132 / 259 (0.51) | 66 / 217 (0.30) |
| logd | 98 / 203 (0.48) | 53 / 161 (0.33) |

- Status: SAME.

**7.11 bullet 2: the hybrid keeps its lead after cleaning; the cleaned hybrid often gains more; the exception is SMAC colloidal with the SVR.**
- Now, small molecules: the cleaned hybrid beats the cleaned single representation in 0.87 to 0.99 of rows.
- Now, other domains, many exceptions in the share of rows where the hybrid is better after:
  - ginkgo_titer: 0.29;
  - humanness_therapeutic: 0.09;
  - flab_thermo_tresanco: 0.17;
  - abcov_kd: 0.31;
  - buchwald_hartwig lightgbm: 26 of 114.
- Colloidal svm is now 89 better against 30 worse, so it is no longer the exception.
- "Gains more": among passing rows the hybrid gains more in 368 against 332 (small molecules) and 301 against 293 (other domains). That is close to even.
- Status: CHANGED (section 7).

**7.11 bullet 3: the best method on a hybrid is NEVER the best on either single representation, on all 9 datasets.**
- Now: n09 has 13 datasets. Its question 2 says the same method is top on the hybrid and on both single representations in:
  - caco2 at shift_only_p10 1.0;
  - cdr_flexibility at level 0 in all 6 flip files, and at 0.1 in one;
  - herg at 0.1 in 2 flip kinds.
- Status: CHANGED. "Never" is false on 3 of 13 datasets.
- Source: n09.md question 2 column.

**7.11 bullet 4: three Ginkgo sets every method passes; binding free energy only one.**
- Now (n09.md answer table, methods passing of 12):

| dataset | real labels | under damage |
|---|---|---|
| ginkgo_titer | 12 | 12 |
| ginkgo_pr_cho | 11 | 12 |
| ginkgo_colloidal | 10 | 12 |
| ddg_combined | 1 (heteroscedastic GP) | 1 (DivideMix) |
| buchwald_hartwig (new) | 10 | 11 |

- Status: SAME for damage and for ddg; "every method" on the real labels is not true for colloidal or pr_cho.

**7.11 bullet 5: the reversal, distance-based selection is the only method of the twelve that passes on hERG at a flip rate of 0.1.**
- Now: herg under damage passes 1 of 10 methods, distance_selection. SAME.
- On herg real labels, uncertainty_curriculum, dividemix and mentornet pass. The stale n09 summary lists early_learning and contrast_to_divide instead of mentornet.

**7.11 bullet 6 (comparison with Chapter 4): QM9 is not among the datasets.**
- Now: n09 qm9 final files exist for censoring and grouped_shifted. The answer table has a qm9 row: real labels 1 of 12 methods pass (confident_learning), damage 8 of 12.
- MEETING's QM9 real-label rows (n09 run) show every one of the 12 methods making QM9 worse for both the single representation and the hybrid (section 5).
- The two pages use different rules. n09.md: beat the plain network on half the feature spaces in half the files. MEETING: best single and best hybrid.
- Status: OUT OF DATE, as the outline already says.

## 4. The antibody exception, and the small-molecule picture on its own

### 4a. Which antibody datasets the methods work far better on

Share of mitigation rows that are strong (section 2a), real labels in brackets:

| dataset | strong share, all rows (real labels) |
|---|---|
| cdr_flexibility | 0.25 (7 of 35) |
| ginkgo_pr_cho | 0.27 (2 of 6) |
| abcov_kd | 0.25 (3 of 20) |
| ginkgo_titer | 0.24 (9 of 48) |
| ginkgo_colloidal | 0.12 (18 of 73) |
| caco2, for comparison | 0.04 (0 on real labels) |
| herg_ki, for comparison | 0.07 (3 on real labels) |
| logd, for comparison | 0.07 (0 on real labels) |

These are the datasets to name. The case for each, strongest first:

**ginkgo_colloidal (SMAC colloidal stability), 242 antibodies, 53 held out in fold 0.**
- It has the most controlled passes on the real labels: 7 of 14 controlled rows, 5 strong.
- svm removal, VERIFIED: 0.2740 → 0.4034 against the shuffled control's 0.2782.
- dnn with every n18 cleanup against shuffled scores, for example out-of-fold move: 0.065 → 0.319 single, 0.227 → 0.442 hybrid, against 0.1327 / 0.2228.
- Caveat: the dnn's untreated score on colloidal moves a lot between jobs. Single representation 0.065 or 0.097; hybrid__general2 -0.0143 against hybrid 0.2267. The shuffled-score control alone lifts the dnn single representation from 0.065 to 0.199 (n18 downweight).

**ginkgo_titer (expression titer).**
- Strong under the n09 survey, n20, n11 and kNN replacement, none of which has a control.
- Controlled passes on the real labels:
  - rf GP move: 0.308 → 0.364 and 0.224 → 0.390, against 0.3116 / 0.2861;
  - rf removal: 0.344 → 0.407 and 0.224 → 0.317, against 0.4007 / 0.2945;
  - dnn removal.
- After cleaning, the hybrid is worse than the single representation in 71% of rows.

**cdr_flexibility (CDR loop flexibility, AUROC).**
- Strong mostly through neighbour-based weights (n02, n12, n21), which have no control.
- Controlled passes: xgboost label smoothing (0.735 → 0.762, 0.775 → 0.804 against 0.7347 / 0.7747) and xgboost removal of 1% (0.717 → 0.732, 0.775 → 0.818 against 0.6968 / 0.7958).

**abcov_kd.** Controlled passes:
- lightgbm downweight: 0.362 → 0.431, 0.361 → 0.464, against 0.4020 / 0.4000;
- xgboost label smoothing: 0.321 → 0.586, 0.116 → 0.493, against 0.3089 / 0.2778.

**ginkgo_pr_cho.** Only n02 and n09 ran on it. Strong rows are from the n09 survey (small-loss selection and distance-based selection with the network). No control.

**ddg_combined (binding free energy).**
- n18 dnn passes the shuffled control with four cleanups. Example, label smoothing: 0.341 → 0.484 single, 0.259 → 0.287 hybrid, against 0.4595 / 0.2759.
- After cleaning, the single representation beats the hybrid in most of these rows.

**humanness_therapeutic and flab_thermo_tresanco.** These are NOT exceptions. Strong shares are 0.07 and 0.14, and on humanness only 1 real-label row passes.

**ginkgo_hic (HIC retention).** No noise results exist.

**A caveat that has to sit next to any "far better" sentence.** The strong rule asks the same +0.05 of Spearman whatever the size of the held-out set. With about 53 held-out antibodies, one fold's Spearman moves by much more than 0.05 between fits. The neural-network numbers above show it. An RMSE row on caco2 has to gain 5% of the RMSE. The two thresholds are not equally hard, so the antibody "strong" shares are inflated relative to small molecules by the rule itself. The shuffled-score control (n18) and the random-deletion control (n18 removal) are the fair comparison, and on those colloidal still stands out.

### 4b. Small molecules on their own (caco2, herg_ki, logd, herg; qm9 in section 5)

- **Real labels, with nothing added.**
  - Nothing is strong on caco2 or logd (0 of 105 and 0 of 97 rows).
  - herg_ki has 3 strong rows, none with a control.
  - The controlled real-label passes are herg_ki label smoothing with the network (0.638 → 0.578 single, 0.599 → 0.578 hybrid) and several caco2 and logd passes of 0.012 RMSE or less.
  - A claim that cleaning removes assay error from small-molecule labels rests on hERG Ki with the neural network alone.
- **Under added damage, what works:**
  - The clipped-label corrections (section 2c, result 1). On logd dnn hybrid, censored EM goes 0.605 → 0.461 against 0.406 for the labels before clipping.
  - Removing a scaffold-group offset under grouped shift (result 3).
  - Deleting flagged molecules under hERG label flips, the one controlled damage result (result 2).
  - Seed averaging and stacking (noise-robust training). On logd they give gains as large as the cleanups.
- **What does not work.** On small molecules:
  - The kNN label replacement fails on every real-label row.
  - Dropping the clipped molecules (a known flag, delete) makes caco2 and herg_ki much worse (herg_ki dnn 0.742 → 0.890).
  - Per-molecule lifting loses to its own shuffled control everywhere.
  - Conformal flagging followed by an action never passes.

## 5. QM9: the clean-label control

QM9 labels are computed, so a gain with no noise added is a change to the model, not noise removed (outline 7.4a). Two sources exist:
- MEETING_METHODS.md, built from the final result files;
- QM9_CONTROL.md, built from the per-row files (`kirby_results_2026-10-09/noise/*/qm9*.rows.jsonl`) of jobs still running.

All fold 0, held-out RMSE.

**From MEETING_METHODS.md, real labels.** Before → after, single representation then hybrid. PAGE: numbers from the final files, n09/n13/n02/n30 qm9 JSON.

| method | group (outline) | model | single | hybrid | verdict |
|---|---|---|---|---|---|
| small-loss selection (n09 run) | mitigation: detector, delete | dnn | 0.396 → 0.465 | 0.362 → 0.417 | worse |
| small-loss selection (n13 run) | mitigation | dnn | 0.403 → 0.470 | 0.364 → 0.423 | worse |
| confident learning (n09 / n13) | mitigation | dnn | 0.396 → 0.412 / 0.403 → 0.422 | 0.355 → 0.371 / 0.349 → 0.355 | worse |
| distance-based selection (n09) | mitigation | dnn | 0.396 → 0.413 | 0.362 → 0.364 | worse |
| multistage cleaning (n09 / n13) | mitigation | dnn | 0.396 → 0.455 / 0.403 → 0.470 | 0.362 → 0.402 / 0.364 → 0.382 | worse |
| early learning (n09 / n13) | robust training | dnn | 0.396 → 0.420 / 0.403 → 0.403 | 0.355 → 0.370 / 0.364 → 0.356 | worse / hybrid only |
| uncertainty curriculum (n09 / n13) | robust training | dnn | 0.396 → 0.444 / 0.403 → 0.412 | 0.355 → 0.380 / 0.364 → 0.354 | worse / hybrid only |
| meta-weight-net (n09 / n13) | robust training | dnn | 0.396 → 0.426 / 0.403 → 0.397 | 0.355 → 0.380 / 0.364 → 0.353 | worse / yes |
| MentorNet (n09 / n13) | robust training | dnn | 0.396 → 0.425 / 0.403 → 0.410 | 0.355 → 0.392 / 0.364 → 0.343 | worse / hybrid only |
| contrast-to-divide (n09 / n13) | robust training | dnn | 0.396 → 0.419 / 0.403 → 0.403 | 0.362 → 0.387 / 0.364 → 0.350 | worse / yes |
| heteroscedastic GP (n09 / n13) | robust training | dnn | 0.396 → 0.417 / 0.403 → 0.395 | 0.355 → 0.372 / 0.364 → 0.363 | worse / yes |
| evidential kernel (n09 / n13) | robust training | dnn | 0.396 → 0.426 / 0.403 → 0.398 | 0.362 → 0.379 / 0.364 → 0.372 | worse / single only |
| DivideMix (n09) | robust training | dnn | 0.396 → 0.499 | 0.362 → 0.440 | worse |
| neighbour-agreement weights (n02) | mitigation: count less | dnn | 0.396 → 0.393 | 0.355 → 0.355 | yes, tiny |
| neighbour-agreement weights (n02) | mitigation: count less | lightgbm | 0.412 → 0.421 | 0.376 → 0.375 | hybrid only |
| cliff-aware weights (n02, hybrids only) | mitigation: count less | dnn / lightgbm | n/a | 0.355 → 0.353 / 0.372 → 0.376 | hybrid only / worse |
| detector weights (n13) | mitigation: count less | dnn | 0.396 → 0.397 | 0.355 → 0.345 | hybrid only |
| detector weights (n13) | mitigation: count less | lightgbm | 0.412 → 0.409 | 0.376 → 0.371 | yes, small |
| winsorising toward a prediction (n30) | mitigation: detector, move | dnn / lightgbm | 0.396 → 0.396 / 0.412 → 0.427 | 0.355 → 0.367 / 0.372 → 0.384 | worse |
| constant lift, censored EM (n30) | mitigation: known flag | both | n/a | n/a | QM9 has no assay limit, as 7.4a expects |

Reading:
- In the n09 run, every one of the twelve published methods made clean QM9 worse for both the single representation and the hybrid.
- In the n13 re-run, four of them (meta-weight-net, contrast-to-divide, heteroscedastic GP, MentorNet on the hybrid) gain 0.001 to 0.021 RMSE. That is the size of the network's fit-to-fit wobble on QM9 (0.3484 to 0.3618, section 0).
- lightgbm gives the same untreated score in every job, so its small QM9 gains are real changes to the model: detector weights 0.376 → 0.371. They are not noise removal.

**From QM9_CONTROL.md, labels with no noise added.** Per-row files, jobs still running. PAGE; I spot-checked the n07 rows from the raw row files.

Deletion:
- n07 removal: 0 of 12 dnn settings and 0 of 6 lightgbm settings improved the hybrid. Best hybrid result 0.3618 → 0.3878 (dnn) and 0.3719 → 0.4106 (lightgbm). VERIFIED from `n07_remediation/qm9__*.rows.jsonl`, `method == 'remove'`, level 0.
- n24 removal: 0 of 18 settings improved the hybrid, for both models.
- n27 removal at increasing shares: 18 of 200 dnn hybrid settings improved, at best 0.3618 → 0.3499.
- Distance-based selection (n09): 1 of 12 hybrid settings, at best 0.3618 → 0.3524.

Moving the label:
- n07 label smoothing: 10 of 12 dnn hybrid settings improved, at best 0.3568 → 0.3395, median change +0.0031. VERIFIED: best 0.3395. lightgbm hybrid 4 of 6, best 0.3759 → 0.3715.
- n24 label smoothing: 27 of 36 dnn hybrid settings, at best 0.3568 → 0.3395; lightgbm 18 of 36.
- kNN label replacement (n07, n24): 0 of 12 and 2 of 36 dnn hybrid settings.
- Out-of-fold move (n07, n24): about half the settings, median change about 0.
- Moving every label toward an out-of-fold prediction (n27): 1 of 6 dnn hybrid settings, median -0.0276.

Counting less:
- n21 neighbour weights, dnn hybrid: 40 of 54 and 31 of 54 settings, median change +0.003.
- lightgbm almost never improves: 5 of 54 and 13 of 54.

Control:
- Random deletion (n24): 9 of 90 dnn hybrid settings improved, at best 0.3568 → 0.3440; 0 of 90 for lightgbm.

QM9 untreated baselines move between jobs (VERIFIED, `score_D` at level 0 across the row files):

| configuration, model | untreated score in different jobs |
|---|---|
| dnn hybrid | 0.3491 or 0.3550 |
| dnn hybrid__general2 | 0.3484, 0.3568 or 0.3618 |
| dnn base_1 | 0.3962 or 0.3999 |
| dnn base_2 | 0.4867 or 0.4949 |
| lightgbm, every configuration | one value |

Taken together:
- On clean labels, deleting flagged molecules reliably hurts.
- Label smoothing toward the neighbours' mean helps the network's hybrid in most settings, but by medians of about 0.003 RMSE, against a fit-to-fit wobble of up to 0.013.
- The best single setting, 0.3568 → 0.3395, is 0.017. That is just above the wobble, and it is picked on the test score.

Still missing on QM9 with no noise added: n20, n23, n25, n31, n33, n34, n35, n39, n40 and n11 have no QM9 row. Final QM9 files exist only for n02, n09 (censoring, grouped_shifted), n13 (censoring), n16, n30 and n32 (gaussian, grouped_shifted), and the non-mitigation experiments.

## 6. Defects found in the analysis pages (no edits made)

1. **n25 control used where it is unusable.** `meeting_methods.py:579-598` takes `random_exempt.arm_mean` as the cliff put-back control for every file. n25's own builder (`n25.py:91`) marks a file unusable when the random pool was smaller than the cliff set, and on caco2, herg and logd it was smaller on every pass:
   - caco2: 1,241 cliff molecules of 1,726, random pool 485 (n25.md line 115).
   - herg and logd: similar (n25.md lines 811, 1206).

   So the MEETING "random molecules put back" numbers for caco2, herg and logd compare against a random set of a different size. 9 of n25's 12 MEETING "yes" rows are on these datasets. Only herg_ki's level-0, censoring and shift_only_p10 files have equal pools.

   The same lines show that "activity cliffs" are 54% to 86% of the training molecules on caco2, herg and logd. That is most of the dataset, not a small exempted set.
2. **n18 removal on MEETING uses one flag rate per dataset.** That rate is the one with the most wins over both models (`meeting_methods.py:250-262`). For colloidal it chose 0.05, so the page shows svm 0.274 → 0.352, not the 0.2740 → 0.4034 at rate 0.2 that the outline quotes. Both are real; the page's choice is narrower.
3. **The n24 label smoothing table leaves out the neighbour-label-gap detector** (`meeting_methods.py`, section 2 note). The n24 page includes it, so the two pages can disagree on the same row.
4. **Level 0 is stored in every damage-kind file with its own network fit**, and MEETING reads level 0 from one file per dataset. hERG Ki label smoothing reads 0.5972 → 0.5706 in the censoring file and 0.5989 → 0.5671 in the outlier and shift files. Same method, same fold.
5. **The hand-written summaries are stale** in n18.md, n20.md, n32.md, n40.md, n41.md and n09.md. Each says "only caco2 has run" or similar, while its own table below now covers 4 to 13 datasets. The outline bullets that say "no method fixed it" (7.6) and "retuning did not help" (7.10) came from these summaries.
6. **The n32 page's detector AUROCs come from a checkpoint.** No final `caco2__censoring.json` exists in `n32_iterative_and_dual_selection/`.

## 7. Priority 2: is the hybrid better than the single representation after the method?

Counted over every MEETING row with both scores. "Hybrid better before" uses each row's untreated scores, and the configurations are those picked after treatment.

| group, domain | hybrid better before and after | worse before, better after | better before, worse after | worse before and after |
|---|---|---|---|---|
| mitigation, small molecules (1,704 rows) | 1,529 | 93 | 33 | 49 |
| mitigation, other domains (1,215 rows) | 527 | 132 | 124 | 432 |
| noise-robust training, small molecules (681 rows) | 568 | 46 | 34 | 33 |
| noise-robust training, other domains (544 rows) | 209 | 86 | 83 | 166 |

- On small molecules, the hybrid is better after mitigation in 95% of rows, but it was already better before in 92%. Mitigation rarely changes which one wins: it flips the answer in 126 of 1,704 rows, mostly toward the hybrid (93 against 33).
- On the other domains it is close to a coin toss before and after, and mitigation flips the answer in 256 of 1,215 rows, equally in both directions.
- Among rows where the method helped both sides, the hybrid gained more in 669 rows and the single representation in 625. "Often gains more" (outline 7.11) is not supported.
- So for Adelaide's second question: on small molecules, the hybrid's lead survives cleaning, but cleaning did not create it. On antibodies the lead is not there to keep. After cleaning, the single representation beats the hybrid in this share of rows with both scores:
  - ginkgo_titer: 71% of rows;
  - humanness_therapeutic: 89%;
  - flab_thermo_tresanco: 81%;
  - abcov_kd: 68%;
  - ddg_combined: 48%.
