# Figures for Chapters 5 to 7, and Adelaide's writing style

Written 2026-10-09 by a Claude fact-gathering agent. Read-only everywhere except this file.
Experiment codes (c02, s05, n30 ...) appear here only as file pointers.

VERIFIED means I read it in this session, in code or in a raw result file, or I ran the figure
script on the raw results. UNVERIFIED means I saw it only in a document.

How I checked the figures: I ran `tests/analysis/thesis_figures.py --figure all --datasets all
--results-root /Volumes/seagate/kirby_results_2026-10-09`. Output went to my session scratchpad,
not the repo. The script fits no model. I then opened the raw result files behind the key
numbers.

---------------------------------------------------------------------------------------------

## 0. What changes the outline (short list)

1. The outline's list of figure slots (THESIS_OUTLINE.md:483-501) no longer matches the script.
   - F1, F5 and F6 are now LaTeX tables, not dot plots. Commit f84364e made the change on
     2026-10-09 at 02:32, on Adelaide's word.
   - F2b was removed from the script in commit 493526a. The outline still lists it at lines
     486 and 1387-1388.
   - F8 now writes one file per experiment, and its rows are grouped by detector. The outline
     wants rows blocked by action (delete, count less, move the label). It also says figures
     are never organised by experiment (lines 491-495 and 623-631). So the current F8 differs
     from the outline in two ways at once: the unit of the file, and the order of the rows.
   - F10's y axis still reads "After repair". Outline line 497 renames "repair" to "action".
   - Outline lines 627-628 say "The script is not written yet". That is stale. The script has
     existed since commit 58bbd3f (2026-10-09 00:36).
2. Some text numbers checked against the 9 Oct results:
   - F1, hybrid against the best single representation: SAME. The LogD numbers match exactly.
   - F5, distance against activity difference: the hERG Ki number is SAME. But "on every
     dataset that ran" is CHANGED: on anti-coronavirus affinity the hybrid is lower.
   - F6, prediction from the five nearest neighbors: CHANGED. Binding free energy is 0.0899,
     not 0.0864. Two more datasets lose besides binding free energy: Buchwald-Hartwig and
     AC-SINS.
   - F9, the censoring ladder: hERG Ki 0.7485 to 0.6662 is SAME. It covers outer fold 0 and
     one noise draw.
3. The default dataset filter keeps 7 datasets. The script calls them "promoted": they have
   units in the noise_full stage of ledger.json. The 7 are caco2, herg, herg_ki, logd,
   ddg_combined, ginkgo_colloidal and ginkgo_titer. QM9, the ADME sets, reactions and patient
   data are left out of F1, F2a, F5 and F6. That conflicts with "QM9 = clean noise control"
   and with the domain-balance rule (outline 509-523). The script docstring calls this filter
   a placeholder.
4. F8 does not filter datasets. Its labels-as-delivered file has 4 small-molecule columns and
   8 non-small-molecule columns: 7 antibody, 1 protein. That breaks the rule "balance across
   domains, not by count".
5. Four of the ten F8 files are wider than the page with the 9 Oct results: 10.0, 9.6, 7.4 and
   7.2 inches, against 6.5. Commit 94e7b23 says "every F8 file within 6.5 in". That was true on
   the smaller 8 Oct results.
6. The page is narrower than the script assumes. The script sizes figures for 6.5 inches. The
   thesis geometry (main.tex:41) gives 21 cm minus 2 x 3 cm = 15 cm = 5.91 inches. Every figure
   shrinks to 91%, so 7 pt text in F8 prints at about 6.4 pt.
7. No Chapter 5-7 figure or table from the script is in thesis.tex yet. No `\includegraphics`
   or `\input` points at them. `figures_ch5_7/` is not on the graphicspath. The generated
   tables need the adjustbox package, which root main.tex does not load. The stage1 preamble
   does load it.
8. The old outputs in `~/repos/KIRBy/figures_ch5_7/` are stale. They were made at 01:39 and
   02:29 on 9 Oct. F1, F5 and F6 there are still PNG dot plots. The F8 CSVs there hold 2.5 to
   4.7 times fewer marks than a run on the 9 Oct results.
9. Style:
   - Her own prose has a median sentence of 19 words. Prose marked as Claude's has a median of
     22, and 22% of its sentences are over 30 words against 12% of hers.
   - She writes "we", uses generic "you", asks rhetorical questions, and uses no em dashes.
   - The Claude-text marker is `% TODO: review, needs manual review`. It appears 198 times in
     plain form, plus suffixed variants.
   - CLAUDE.md says thesis.tex has `% [SPLIT 2026-09-27]` comments. It has none: zero matches.

---------------------------------------------------------------------------------------------

## TASK A. FIGURES

### A.1 The script, in one paragraph

- Entry point: `tests/analysis/thesis_figures.py`, 976 lines. Style lives in
  `tests/analysis/thesis_figure_style.py`, 436 lines. Loaders are
  `tests/analysis/thesis_loader_{f1,f2,f3,f56,f8,f9_11}.py`, 2,509 lines in total.
- What each figure writes:
  - a figure writes `<name>.png` at 300 dpi plus `<name>.csv`, one row per mark;
  - a table writes `<name>.tex` plus `<name>.csv`.
- Default output folder: `tests/results/thesis_figures/` (thesis_figures.py:893). That folder
  does not exist.
- Default results folder: the path in `tests/results/CURRENT_RUN` (that file does not exist),
  otherwise `LAPTOP_ROOT = /Volumes/seagate/kirby_results_2026-10-08` (line 58). So the default
  reads the 8 Oct results, not the 9 Oct ones. Pass
  `--results-root /Volumes/seagate/kirby_results_2026-10-09`. VERIFIED.
- `--list` and `--dry-run` check the inputs. `--list` has a small defect: for F2a it shows
  "1 files" for c09 and c16. INPUTS_F2A holds folder names, not file patterns
  (thesis_loader_f2.py:44), so it counts the folder itself. c09 actually holds 20 .json files.
  VERIFIED.
- Style, copied from qsar_qm_models generate_paper_figures_v2.py:
  - 6.5 inch width, DejaVu Sans 11 pt;
  - seaborn "ticks" style, no legend frame;
  - colour by model family, with the marker separating variants within a family;
  - `MODEL_LABELS['dnn'] = 'NN'`;
  - CLEAN_COLOR #0072B2 is kept for clean labels only. VERIFIED.
- Labels: thesis_figure_style.py holds dataset display names (DATASET_LABELS, lines 310-328)
  and representation names (REP_LABELS). The noise names are in DETECTOR_WORDS, DAMAGE_WORDS
  and EXPERIMENT_WORDS (thesis_figures.py:467-494) and HANDLE (thesis_loader_f8.py:53).

### A.2 Slot by slot

How many outer folds each slot covers:
- complexity and similarity results: 5 outer folds, checked in n_folds of every CSV row;
- noise results: outer fold 0 only.

The noise loaders find no other folds: `one_fold()` printed no "the results also hold folds"
message. The noise CSVs record n_draws 1 or 3.

| Slot | Outline section | Result files read (9 Oct root) | Output | Folds | Status vs outline |
|---|---|---|---|---|---|
| F1 | 5.4 (outline 1354-1358) | complexity/c02_complexity_matching_grid/<ds>.json, `standard_selection_summary.<model>.{base1,hybrid1}.heldout.mean` | tab_hybrid_vs_best_single.tex/.csv, label tab:hybrid_vs_best_single | 5 | shape CHANGED (table, not dots); verdict SAME |
| F2a | 5.5 (1385-1386) | complexity/c09_formal_analysis/<ds>.json, `bias_variance.curves['<model>\|hybrid1'].by_budget[i].val_score_mean`; option `--f2a-source c16` | f02a_best_number_of_features.png/.csv | 5 | built; 10 dataset-and-model pairs missing |
| F2b | 5.5 (1387-1388) | c18 (loader still exists: thesis_loader_f2.load_f2b) | none; removed from the registry | n/a | outline still lists it: STALE outline |
| F3 | 5.6 (1421-1424) | complexity/c22_explainability/<ds>.json, slot hybrid1, `allocation_built` | f03_blueprints.png/.csv | one fold drawn (`--f3-fold 0`) | built; the deeper panel CANNOT be built |
| F5 | 6.3 (1543-1548) | similarity/s02_hybrid_similarity/<ds>.json, top-level comparisons per fold, Manhattan, unweighted, no feature limit, "knn distance hybrid - knn base rank 1" | tab_distance_tracks_activity.tex/.csv, label tab:distance_tracks_activity | 5 | shape CHANGED (table); one verdict CHANGED |
| F6 | 6.4 (1555-1558) | similarity/s05_knn_prediction/<ds>.json `summary.fixed_ref` (distance hybrid against kNN's rank-1 single representation, unweighted, Manhattan, k=5, rank) | tab_nearest_neighbour_prediction.tex/.csv, label tab:nearest_neighbour_prediction | 5 | shape CHANGED (table); verdict and one number CHANGED |
| F8 | 7.5 (1832-1836) | noise/{n07,n18,n20,n23,n24,n25,n27,n30,n34,n38}/ (.json, .partial.jsonl, _partial/, _checkpoints/) | f08_did_the_cleaning_work_<experiment words>.png/.csv, 10 files | fold 0 | CHANGED in structure; see A.3 |
| F9 | 7.7 (1852-1856) | noise/n30_censored_likelihood/<ds>.json | f09_censoring_ladder.png/.csv | fold 0, 1 draw | number SAME |
| F10 | 7.6 (1843-1847) | noise/n32_iterative_and_dual_selection/*.json and `_checkpoints/*.json` | f10_detection_vs_repair_<ds>.png/.csv, one per dataset (7 files) | fold 0, 1-3 draws | reads UNFINISHED checkpoints; cannot show the outline's Shapley claim |
| F11 | 7.7 (1866-1867) | noise/n07_remediation/<ds>__<kind>.json | f11_score_vs_damage_level.png/.csv | fold 0, 1-3 draws | methods do not follow the three-group rule |

Kept as tables on purpose, and not made by the script: admitting extra features and the decoy
counts (5.5a), where importance weighting helps (6.4), and the hybrid against ensembles (5.4).
The outline also names an "F1 variant or TABLE" for the comparison with PCA, mRMR and
recursive elimination (outline 1381). The script has no slot for it.

### A.3 Details per slot

**F1, table: hybrid against the best single base representation.**

What one row is: one dataset with one of its two decided models. Each row holds two numbers,
the best single representation's score and the hybrid's score. Both are the mean over 5 outer
folds of held-out scores. In each fold, the single representation and the hybrid are picked on
that fold's training rows. The rows sit under a header line per metric, with an arrow. The
better number in each row is in bold.

The caption reads: "Held-out score of the best single base representation and the hybrid, per
dataset and model, mean over outer folds. Bold: the better of the two."

Checks on the raw files, all VERIFIED in `complexity/c02_complexity_matching_grid/<ds>.json`:

| Dataset | Model | Best single | Hybrid | n_folds |
|---|---|---|---|---|
| LogD | dnn | 0.4497 | 0.4089 | 5 |
| LogD | svm | 0.4930 | 0.4302 | 5 |
| hERG Ki | svm | 0.5881 | 0.5590 | 5 |
| hERG Ki | dnn | 0.6031 | 0.5691 | 5 |
| hERG | lightgbm | 0.8555 | 0.8666 | 5 |
| QM9 | dnn | 0.3619 | 0.3155 | 5 |
| QM9 | lightgbm | 0.3720 | 0.3463 | 5 |
| HIC retention | gp | 0.6000 | 0.5515 | 5 |
| HIC retention | dnn | 0.5311 | 0.5293 | 5 |

The LogD and hERG Ki scores are RMSE. hERG is AUROC. The HIC retention scores are Spearman.
The LogD values sit under `standard_selection_summary.dnn.base1.heldout.mean` and
`...hybrid1.heldout.mean`.

Outline 5.4 says:
- the hybrid wins on Caco-2, LogD, hERG Ki, hERG and the three ADME sets, for both models;
- LogD goes 0.4497 to 0.4089 with the network and 0.4930 to 0.4302 with the SVR;
- the hybrid loses for both models on nanobody thermostability, SMAC colloidal stability and
  HIC retention.

Current results agree on all of it: SAME, 5 folds.

Two of those losses are smaller than 0.001:
- nanobody thermostability with SVM: 0.6901 single against 0.6899 hybrid, Spearman;
- SMAC colloidal stability with SVM: 0.3485 against 0.3480.

Losses the outline does not mention, all with 5 folds:

| Dataset | Model | Metric | Best single | Hybrid |
|---|---|---|---|---|
| Anti-coronavirus affinity | lightgbm | Spearman | 0.2494 | 0.1654 |
| Buchwald-Hartwig | rf | RMSE | 14.4056 | 15.2383 |
| CDR flexibility | lightgbm | AUROC | 0.8713 | 0.8669 |
| CHO polyreactivity | gp | Spearman | 0.5009 | 0.4419 |
| Antibody humanness | rf | AUROC | 0.9659 | 0.9614 |

These are from the CSV. Only the LogD, hERG Ki, hERG, QM9 and HIC rows above were opened in
the raw files.

flab_binding_koenig has no F1 row. Its c02 result is only a partial checkpoint with 1 fold
(dry run).

Where F1 departs from the outline:
- The outline asks for "one dot per dataset, three panels by metric". The script makes a
  table instead.
- The thesis.tex notes at thesis.tex:2832 ask for a table labelled `tab:hybrid_vs_single`.
  Its columns there are the best single representation per fold, the scores per fold, and the
  difference per fold with its mean and spread. The script's table has another label,
  `tab:hybrid_vs_best_single`. It holds fold means only and names no representation. The
  per-fold names are in the CSV columns `best_single_rep_by_fold` and `hybrid_label_by_fold`.
- Bold: the script bolds the better of each pair in every row. The outline's selective-bold
  rule (451-470) says "a few cells in some tables, not every table". It allows rule (a), best
  in the row, for Chapter 5, but the caption must say which rule is used.
- The outline's arrow rule needs the caption to say that the arrow gives the direction. The
  script's captions do not say it.

What will need to change:
- whether the table shows fold means or per-fold numbers;
- which datasets it shows;
- the label to use;
- a short caption form, `\caption[short]{long}`, which every other thesis table uses;
- dataset names need to match the thesis: the script's "QM9 HOMO-LUMO gap" against
  HOMO--LUMO in the thesis, and "Human microsomal clearance" against "human liver microsome
  (HLM) clearance" at thesis.tex:1365.

**F2a, best number of features per dataset and model.**

What it draws: one dot per dataset and model. The x axis is the budget, the number of features,
on a log2 scale. The best budget is the arg-best of c09's stored fold-mean held-out score over
c09's own budget list.

40 marks, all with n_folds 5 and no ties (CSV, VERIFIED from the run). Some examples:

| Dataset | Model | Best budget |
|---|---|---|
| hERG Ki | svm | 1000 |
| hERG Ki | dnn | 250 |
| LogD | dnn | 2000 |
| LogD | svm | 1000 |
| QM9 | dnn | 500 |
| QM9 | lightgbm | 2000 |
| PAM50 | svm | 25 |
| PAM50 | dnn | 250 |
| VEGF binding | gp | 10 |

The budget lists differ by dataset. Several datasets search only {500, 1000, 2000}. On those,
a best budget of 2000 is the largest budget searched. In the text, "best" means best among
the budgets searched; this is a reading of the CSV, not something the figure shows.

No c09 file for 10 dataset-and-model pairs: abagym_Ang2, adme_hlm, adme_rlm,
flab_binding_koenig and flab_expression_koenig, both models each (dry run, VERIFIED).

The outline's verdict "differs by dataset and by model" holds in the CSV. hERG Ki, for
example, has 1000 for the SVM and 250 for the network.

**F2b.** It was removed from the script in commit 493526a. The loader `load_f2b` still exists.
The outline still lists F2b beside F2a, at lines 486 and 1387. Adelaide decides whether F2b
comes back or the outline drops it.

**F3, blueprints.**

What it draws: one horizontal bar per dataset, split into segments by representation. It uses
slot hybrid1 on outer fold 0, with the first decided model. There is one panel per domain, and
colours restart in each domain.

172 segments. Some examples (CSV):

| Dataset | Model | Features by representation (top six shown) | Total features |
|---|---|---|---|
| LogD | dnn | mordred 223, mhg_gnn 217, grover 214, molformer 212, smited 208, ecfp4 196 | 2000 |
| hERG Ki | svm | mhg_gnn 130, pdv 109, graph_kernel 106, grover 103, atom_pair 101, mordred 98 | 1000 |
| hERG | lightgbm | mordred 604, mhg_gnn 538, graph_kernel 129 (three representations only) | 1271 |

The QM9 hybrid with dnn takes 122 features from `chembert`. Memory says `chembert` and
`chemberta` differ (256 against 384 features) and that 3 pools carry chembert without a check.
That is UNVERIFIED here.

The deeper panel the outline asks for cannot be built from disk. It needs named feature groups
for one small-molecule dataset. The loader explains why (thesis_loader_f3.py:19-24, VERIFIED in
the code):
- c22 and c24 store only the count of features per representation;
- the search cache stores only a hash of the selected feature indices.

Getting the names means re-fitting the blueprint with `tests/analysis/blueprint_trace.py`, which
costs compute.

Outline 5.6 says "Reading only, no compute". That is CHANGED: it needs compute.

Partial files: adme_rlm and flab_expression_koenig have only fold 0, and logd lacks folds 2-4.
The default fold-0 bar is unaffected.

**F5, table: distance against activity difference.**

What one row is: one dataset, with two Spearman numbers, each a mean over 5 folds. The first
is between distance and activity difference in kNN's rank-1 single representation. The second
is the same in the distance hybrid. All values are from the CSV.

| Dataset | Single | Hybrid |
|---|---|---|
| hERG Ki | 0.0729 | 0.2625 |
| LogD | 0.0765 | 0.2687 |
| QM9 | 0.1298 | 0.3779 |
| Caco-2 | 0.1517 | 0.1969 |
| hERG | 0.0257 | 0.0552 |
| Anti-coronavirus affinity | 0.0901 | 0.0376 |

Outline 6.3 says the hybrid rises more steeply "on every dataset that ran". That is CHANGED:
on anti-coronavirus affinity the hybrid is lower.

The outline's hERG Ki number is SAME. I opened the s02 file. The summary of
`s02_hybrid_similarity/herg_ki.json` exists, but I did not trace the 0.0729 / 0.2625 pair to a
single key. It comes through the loader's mean over `comparisons` per fold. VERIFIED via the
loader, not a single field.

No s02 file for abagym_Ang2, abagym_VEGF, adme_hlm, adme_mdr1, adme_rlm, flab_binding_koenig,
flab_expression_koenig and ginkgo_hic. The ledger marks them pending.

**F6, table: prediction from the five nearest neighbors.**

Same layout as F5. Values are in each dataset's own metric. All of them are VERIFIED in
`similarity/s05_knn_prediction/<ds>.json`, `summary.fixed_ref.single_score_mean` and
`hybrid_score_mean`, `n_paired_folds` 5.

| Dataset | Metric | Single | Hybrid | Status |
|---|---|---|---|---|
| CHO polyreactivity | Spearman | 0.2059 | 0.3591 | SAME |
| Binding free energy | Spearman | 0.1424 | 0.0899 | outline says 0.0864: CHANGED |
| Buchwald-Hartwig | RMSE | 14.4572 | 16.1811 | hybrid worse; not in the outline |
| AC-SINS self-association | Spearman | 0.3912 | 0.3884 | hybrid worse; not in the outline |

Outline 6.4 says "better on every dataset except binding free energy". That is CHANGED: three
datasets lose, not one.

Per-fold deltas for binding free energy: 0.212, -0.329, -0.176, 0.071, -0.039.

Other rows (CSV):

| Dataset | Metric | Single | Hybrid |
|---|---|---|---|
| hERG Ki | RMSE | 0.6280 | 0.5822 |
| LogD | RMSE | 0.6766 | 0.5831 |
| QM9 | RMSE | 0.6692 | 0.4205 |
| Caco-2 | RMSE | 0.3386 | 0.3246 |
| hERG | AUROC | 0.8143 | 0.8335 |

The similarity-hybrid rows are missing:
- on every Ginkgo set and on binding free energy, because objective='similarity' has no
  neighbor outside a held-out block of one group;
- on PAM50, because AUROC needs 2 classes and PAM50 has 4.

The loader writes each reason into the missing list.

Captions and labels use British "neighbour": `tab:nearest_neighbour_prediction`, and the F8
row names "neighbours' mean" and "neighbour-label gap". The thesis uses American spelling.

**F8, did the cleaning work.**

What one cell is: one experiment, method and dataset, either on the labels as delivered or at
the main damage level. Main level: 1.0 of the clean-label spread, 0.25 for censoring, 0.1 for
class flips. The shade is how many of the 8 model-and-configuration pairs improved. 8 pairs =
2 models x {base_1, base_2, hybrid, hybrid__general2}. A pair counts as improved if any setting
of the method improved it, after rounding to 4 decimals. Fold 0.
(thesis_loader_f8.py:1-25, VERIFIED.)

Reweighting and training-side methods are excluded by the loader. Outline line 491 asks for a
"count less" block. F8 cannot show that block as built.

One file per experiment, titled by EXPERIMENT_WORDS: "remediation grid", "labels as
delivered", "cleaning before the search", and so on. Rows are grouped under "By <detector>"
headers.

This disagrees with the outline in three ways:
- rows blocked by action, with move rows sub-ordered by target;
- rows labelled "detector, action";
- noise results never organised by experiment.

Marks per file on the 9 Oct results, with the files the redraw left in figures_ch5_7/:

| File | Marks now | Marks in figures_ch5_7/ |
|---|---|---|
| uncertainty cleaning | 320 | 75 |
| cleaning before the search | 304 | 80 |
| cliff molecules put back | 462 | 198 |
| remediation grid | 238 | 154 |
| labels as delivered | 122 | 48 |

So the files in figures_ch5_7/ are stale.

Four files are wider than the page: 10.0, 9.6, 7.4 and 7.2 inches (stderr of the run). The
script suggests choosing rows with `--f8-rows`.

n20 caco2 and n38 caco2 have no level 1.0. They are read at 0.5, as the 8 Oct analysis did.
The CSV `level` column says so.

"Move toward GP prediction" has no after-score on herg_ki, caco2, logd and buchwald_hartwig for
all 8 pairs, and on ginkgo_colloidal for 2 of 8 pairs (n07, dry run). These show as hatched
cells.

Outline 7.5 says deleting improved 171 of a possible 264 model-and-configuration pairs, and
changing a label 813 of 1,320. F8 does not compute these. They are not checked here: NOT
FOUND in the figure outputs.

**F9, the censoring ladder.**

What it draws: four steps on one axis, one panel each for Caco-2 and hERG Ki. The four steps
are the labels as delivered, after clipping, after correction by `constant_lift` (raise the
clipped labels by one constant), and after deleting the clipped molecules. Colour is the
configuration and shape is the model. Clipped share 0.25. Fold 0, n_draws 1.

All values VERIFIED from `noise/n30_censored_likelihood/{herg_ki,caco2}.json` through the
loader:

| Dataset | Model on configuration | Delivered | Clipped | Corrected | Deleted |
|---|---|---|---|---|---|
| hERG Ki | network on the second base representation | 0.5891 | 0.7485 | 0.6662 | 0.8962 |
| hERG Ki | network on the hybrid | 0.6078 | 0.7313 | 0.6684 | 0.8806 |
| Caco-2 | network on the hybrid | 0.2459 | 0.3042 | 0.2751 | 0.3403 |

All are RMSE. Outline 7.7 says hERG Ki 0.7485 to 0.6662: SAME.

Caco-2 with the network on the hybrid: outline 7.7 gives 0.3328 to 0.2866 for scaffold-group
offsets, not censoring, so it is not comparable here.

**F10, detection against action.**

What it draws: one dot per detector and action. x is the detection AUROC and y is the
held-out score after the action. One file per dataset, one panel per damage kind. The pair is
the network on the hybrid. Fold 0.

Sources for Caco-2 are `_checkpoints/caco2__*.json`. The CSV marks them "(checkpoint, unit
unfinished)". So F10 currently draws unfinished units.

Detectors drawn on every dataset:
- out-of-fold residual;
- ITLM a0.8, a0.9 and a0.95;
- two-learner, share 0.05, 0.1 and 0.2.

Seven graded detectors drive no action, so they get no y value:
- cleanlab;
- conformal;
- the Shapley value under forest proximity, k = 5, 10 and 20;
- ngboost_rescaled;
- quantile_forest_rescaled.

Outline 7.6 says the Shapley-value detector reached 0.8106 to 0.8921 AUROC on Caco-2 at a
clipped share of 0.25. F10 cannot show that number, because no action reads that detector.
The number is NOT checked here.

Caco-2 censoring at 0.25 (VERIFIED, CSV from a checkpoint):
- the out-of-fold residual's AUROC is 0.5041;
- RMSE after its shrink actions is 0.3059 to 0.3109;
- no cleaning gives 0.3042;
- clean labels give 0.2474.

Open point for Adelaide: ITLM trims high-loss molecules inside the training loop. Under the
outline's test (lines 563-571), that may be noise-robust training rather than a detector plus
an action. F10 plots it as a detector.

Labels: "After repair" (thesis_figures.py:778) and file names "detection_vs_repair". The
outline renames "repair" to "action".

**F11, score against damage level.**

What it draws: two panels, hERG Ki under Gaussian noise and under censoring, with the network
on the hybrid. Methods: no cleaning, label_smooth, oof_smooth, remove@flag_share0.1 and
js_shrink. Fold 0, 1 to 3 draws.

All RMSE (CSV, VERIFIED from the run):

| Damage | Level | No cleaning | js_shrink | label_smooth | oof_smooth | remove |
|---|---|---|---|---|---|---|
| Gaussian | 1.0 | 0.7199 | 0.6954 | 0.7120 | 0.7035 | 0.7237 |
| Censoring | 0.25 | 0.7313 | 0.7356 | 0.7395 | 0.7318 | 0.7659 |

Remove at level 0 is 0.7095 in both panels.

The no-cleaning score at level 0 differs between the panels: 0.5972 on the Gaussian file,
0.5989 on the censoring file. These are separate runs. The 0.5989 is the starting number of
outline 7.5, "hERG Ki with the network on the hybrid: 0.5989 to 0.5671".

The outline says the marks are grouped by the three groups, with noise-robust training kept
visibly apart (lines 499-501). All five default methods are mitigation, of the move or delete
kind. None is noise-robust training, and the colours are per method, not per group. This will
need to change.

### A.4 Numbers the text can take straight from the outputs

Every number below is in a CSV with a `source` column naming the raw file:

- **F1** `tab_hybrid_vs_best_single.csv`:
  - per dataset and model: `best_single_score`, `hybrid_score`, `best_single_rep_by_fold`,
    `hybrid_label_by_fold`, and `n_folds_*`.
  - The raw c02 file also stores `min` and `max` per side, for example LogD dnn base1 0.4308
    to 0.4710. The script does not export them yet. The thesis notes ask for per-fold scores
    and spread.
- **F2a** `f02a_best_number_of_features.csv`: `best_value`, `best_score`, `score_by_budget`
  (every budget), and `grid`.
- **F3** `f03_blueprints.csv`: `n_features` per representation per bar, and
  `total_features`.
- **F5 and F6** `tab_*.csv`: `single_value`, `hybrid_value`, `per_fold_single`,
  `per_fold_hybrid`, and `setting`.
- **F8** CSVs: `n_pairs_improved` of `n_pairs_run`, `pairs_improved` with before and after
  scores per pair, and `pair_settings` (the best setting).
- **F9, F10 and F11** CSVs: `value`, `detection_auroc`, `repair_value`, `no_cleaning_value`,
  `clean_value`, and `n_draws`.

### A.5 Where outputs go and how they get into thesis.tex

- No Chapter 5-7 output of this script is in thesis.tex. I checked every `\includegraphics`
  and `\input` in thesis.tex (VERIFIED, grep):
  - Chapters 2 and 3 use bare file names, for example `F2_variance_decomposition.png`.
  - Chapter 4 and Chapter 5's search figure use `figures_code_input/<name>.pdf|png`.
  - Chapters 5 and 6 contain only the hand-typed preliminary tables (tab:prelim_*,
    tab:ch6prelim_*). Chapter 7 has no figure.
- Root main.tex has `\graphicspath{{./}{thesis_appendix/noise/figures/}}` (main.tex:54). Many
  of the thesis's image files (molecular_representations.png, figures_code_input/) exist only
  under `thesis_rework/stage1_takeaways_bullets/`. That preamble has
  `\graphicspath{{./}{figures_ch3/}{thesis_appendix/noise/figures/}}` and loads adjustbox
  (preamble.tex:25, 61).
- Memory says (UNVERIFIED as a build test here): root main.tex stalls, and the stage1 main.tex
  builds. thesis_rework/stage1_takeaways_bullets/thesis.tex was written at 12:37 on 9 Oct, the
  same minute as scratchpad/thesis_code_todo/thesis.pdf. It differs from root thesis.tex
  (13:40). So the 12:37 PDF was built from a copy of thesis.tex.
- The PDF is always overwritten at `scratchpad/thesis_code_todo/thesis.pdf`; the dark copy is
  `thesis_DARK_not_for_submission.pdf` (memory reference_thesis_pdf_location).
- To insert:
  - a table: `\input{figures_ch5_7/tab_hybrid_vs_best_single}`. The .tex is a complete table
    float with caption and label. It needs adjustbox.
  - a figure: `\includegraphics[width=\textwidth]{figures_ch5_7/f09_censoring_ladder.png}`
    inside a figure float, with caption and label written by hand. The script writes no
    figure captions.
  - Either way, the output folder must be reachable from the build directory.
- The house convention in thesis.tex is
  `\includegraphics[width=\textwidth,height=0.72\textheight,keepaspectratio]{...}`, with
  `\caption[short]{long}` and labels `fig:<chapter>:<name>` or `tab:<name>`.

### A.6 Defects found in the figure code

1. `--list` miscounts the F2a inputs, because the patterns are folders
   (thesis_loader_f2.py:44).
2. The default results root is the 8 Oct folder (thesis_figures.py:58), not the 9 Oct one.
3. Sizes assume a 6.5 inch text width. The thesis text width is 5.91 inches.
4. F10 draws unfinished checkpoint units without saying so on the figure. The CSV does say so.
5. Small-molecule columns in F8 sort as "Caco-2, LogD, hERG Ki, hERG blocking". The sort is
   case-sensitive, so lowercase "h" sorts after "L".
6. British spellings appear in printed text: neighbour, neighbours' (thesis_figures.py:469-471;
   thesis_loader_f8.py:54-56; thesis_loader_f9_11.py:213, 433-440).
7. thesis_figure_style.py:345 says the ddg_combined domain ("Proteins") is "Claude's
   assignment, Adelaide to confirm".

---------------------------------------------------------------------------------------------

## TASK B. ADELAIDE'S WRITING STYLE

### B.1 Which text is hers

- Text below `% TODO: review, needs manual review` is Claude's. So is text below
  `% TODO: REVIEW MODIFICATION --`, `% CODE INPUT` and `% [CLAUDE 2026-10-09] citation fix`.
  The last one means Claude edited her sentence.
- The cleanest examples of her prose are:
  - Chapter 1 (lines 13-131);
  - Chapter 3's discussion paragraphs (896-980);
  - Chapter 4's openings (1342-1376);
  - Chapter 5's introduction (1834-1917).
- In Chapter 2, many paragraphs carry `[CLAUDE 2026-10-09] citation fix` comments (lines 15-324,
  73 such comments). I avoided those.
- Her own TODOs are free-form and first-person. Examples:
  - line 27: "% TODO: come back to this at the end and double-check the contents, this is missing
    a lot because it was written prior to results in chapters 5+";
  - line 1851: "% TODO: I'm pretty sure I'm doing late fusion but calling it ensembling, make
    sure its appropriately named below";
  - line 2027: "% TODO: am I keeping nanobody thermostability?? ...".
- Line 1651 states her rule about her old text: "% TODO: my confirmatino report was written by
  me, not you, so wherevery possible, use my original language".

### B.2 Numbers (measured this session)

I split 49 of her paragraphs and 169 Claude-marked paragraphs into sentences, by script.

| | Her paragraphs | Claude-marked paragraphs |
|---|---|---|
| Sentences | 301 | 932 |
| Median words per sentence | 19 | 22 |
| Mean words per sentence | 20.2 | 23.3 |
| Longest sentence | 52 | 77 |
| Sentences over 30 words | 36 (12%) | 207 (22%) |
| Sentences under 10 words | 30 (10%) | 53 (6%) |

Other counts in her 49 paragraphs:

| Item | Count |
|---|---|
| "we" / "We" | 46 |
| "I" | 5 |
| "you" / "your" | 24 |
| contractions | 7 ("don't", "doesn't", "That's", "it's", "we've") |
| `\citet` | 16 |
| `\citep` | 39 |
| "However," | 8 |
| semicolons | 6 |
| em dashes | 0 |
| "suggests" | 1 |
| "indicates" | 2 |
| "demonstrat-" | 1 |

### B.3 Her habits, concretely

- **"We" by default.** "I" appears for her own discovery or motivation: "as I discovered",
  "I run this hybrid search" (1894), and "I will also discuss" (22). The memory rule matches:
  "we", and "I" only for personal motivation.
- **Generic "you" and "your".** "When you pool data from different sources" (328), "you end up
  with 4,685 features" (1849), "When your dataset covers only a small region" (1360).
- **Rhetorical questions** to open a design problem. Line 1892 is a run of six questions. Line
  1862: "what features does the model prefer?"
- **A short, plain claim, then the evidence.** "Not every molecular representation is equal. In
  fact, they hold very different types of information." (1847) "ECFP4 is the worst offender."
  (1367)
- **Informal phrases she keeps:**
  - "takes a turn to look at another topic" (108);
  - "In a dream world ... Unfortunately we live in reality" (1894);
  - "That's great behavior" (1376);
  - "showed disappointing results" (108);
  - "you don't have as much to lose" (939).
- **How she introduces a method:** full name, acronym in parentheses, citation, then one
  sentence on what it computes. For example: "Quantile Regression Forests (QRFs) hold the
  distribution of training labels in each leaf and compute quantiles over all trees" (734).
  Steps get ordinal verbs: "Sort & Slice begins by sorting ... It then ``slices'' away"
  (165, partly Claude-flagged).
- **Defining terms:** "which we refer to as noise conditions" (384), "referred to as the hybrid
  representation" (1837), and "We define a constant feature to be one that ..." (1365).
- **Reporting a result:**
  - the number in its own metric, often as "X against Y": "Representation explains 8.1% of the
    variance in \aucnorm, against 31.1% of clean $R^2$" (896);
  - counts as "N of M features": "1,815 of the 2,048 features are near-constant" (1367);
  - rank movement: "falls from first to fifth" (980);
  - a list of values: "0.937 under Gaussian noise against 0.947, 0.952 and 0.955 under ..."
    (973).
  - Three decimals in Chapter 3 and four in the hybrid chapters.
  - Thousands written with commas: 1,412 and 5{,}039.
- **Hedging is light, with plain words:** "may", "can", "likely because" (896), "partially due
  to" (939), "has the potential to" (939), and "we have not looked into why" (1367). She says
  she has not checked something rather than hiding it.
- **Paragraph openings** set up the comparison or question:
  - "We compare the variance decomposition on both clean and noisy labels" (896);
  - "To investigate similarity estimation, we first look at a simple measure:" (1365);
  - "As we've seen," (941);
  - "While alternative shapes of noise may be ..." (975);
  - "Considering the two main components of a QSAR model," (886);
  - "Beyond simply combining features," (1855).
- **Citing:**
  - `\citet{X}` when the authors are the subject: "\citet{Riniker2013} benchmarked ..." (1350);
  - `\citep{...}` at the end of the clause for a fact, with several keys separated by commas;
  - figures and tables in parentheses: "(Figure~\ref{fig:grid})";
  - cross-references written as "Chapter~\ref{}", "Section~\ref{}" and
    "Additional file~\ref{}".
- **LaTeX:** quotes as ``...''. `\%` is attached to the number.
- **Words she uses:** QSAR model, predictive performance, noise robustness, noise level, noise
  condition, representation, base representation, hybrid representation, budget, downstream
  model.
- **Spelling:** mostly American (modeling, neighbor, behavior, standardized, tokenized), but
  British slips into the Chapter 5 introduction: "customises" (110), "optimisation" (1894),
  "generalised" (1909), "modelling" (1909). The rule is American.
- **Typos:** fast drafting leaves many, for example "genereate" (110), "cheminformatcs" (22),
  "representaions", "calculationsc" (1834), "wiht", "reprsentation" (1853), "aggregats",
  "tarining" (1855), "featues", "nose" (1905), "fro" (1909), "antimodies" (1915). Fix them, but
  do not rewrite the sentence around them.
- **What she does not use:** em dashes, semicolon chains, "Notably", "Importantly",
  "Crucially", "delve" (one "delve" at 1342), and the "This X indicates that" frame.

### B.4 Verbatim example sentences (all found with `grep -F` in thesis.tex this session)

1. (13) "However, attrition is high, and roughly 90\% of drugs that enter clinical trials are
   never approved, often due to poor efficacy or toxicity \citep{Sun2022}."
2. (13) "A QSAR model requires a molecular representation and some form of machine learning
   (ML) model."
3. (22) "This thesis addresses the challenge of conducting molecular property prediction when
   the underlying experimental data, or ``labels'' are noisy."
4. (22) "This thesis fills that gap."
5. (106) "We found that model family was the main factor in noise robustness, and choice of
   molecular representation matters more for predictive performance than noise robustness."
6. (106) "Strong predictive performance on clean labels does not imply robustness."
7. (108) "Chapter~\ref{chap:flaws} takes a turn to look at another topic: molecular
   similarity."
8. (110) "The actual output of the hybrid search is not the set of features, but rather the
   settings that define the feature selection method used to genereate that hybrid
   representation."
9. (151) "The success of machine learning hinges on the quality of input data."
10. (314) "To properly model experimental noise, we must first understand where the noise is
    coming from."
11. (328) "When you pool data from different sources, each source has the potential to carry a
    shared offset."
12. (734) "Quantile Regression Forests (QRFs) hold the distribution of training labels in each
    leaf and compute quantiles over all trees \citep{Meinshausen2006}."
13. (734) "While we used 100 trees for RF, we observed a noisy quantile estimate with 100 trees
    for QRF, prompting us to use 300 instead."
14. (896) "Robustness divides the variance a bit differently."
15. (896) "Representation explains 8.1\% of the variance in \aucnorm, against 31.1\% of clean
    $R^2$, while model family rises to 47.2\%."
16. (896) "The replicate nearly disappears from \aucnorm, likely because each replicate's curve
    is divided by that replicate's own clean $R^2$, so a difficult split lowers both and
    cancels out."
17. (939) "NGBoost is a standout case in the sense it is highly dependent on its paired
    molecular representation."
18. (939) "Its high robustness is partially due to the fact that when you start with a low
    predictive performance, you don't have as much to lose."
19. (939) "So when paired with the right features, NGBoost has the potential to make better
    predictions under higher quantities of label noise."
20. (941) "As we've seen, the choice of representation can have an impact on noise robustness,
    but that impact is determined by model family, represented by the interaction term between
    model and representation in Figure~\ref{fig:grid}."
21. (941) "Some models are more impacted by representation than others, though."
22. (973) "SVM reported higher \aucnorm{} values on non-Gaussian label noise, 0.937 under
    Gaussian noise against 0.947, 0.952 and 0.955 under Student-$t$, outlier and Laplace noise
    on PDV, while NGBoost lost \aucnorm{} under every one."
23. (973) "The shape of the noise does not change how much performance a model keeps, but it
    does change which model keeps the most."
24. (980) "Of the 13 base models, RF started ranked seventh, but by level 1.0 it was ranked
    first."
25. (980) "SVM on the other hand falls from first to fifth (Figure~\ref{fig:rank_level})."
26. (1342) "Furthermore, there is no single way to define similarity between compounds. This is
    where the principle begins to break down in practice."
27. (1346) "The modelability index of \citet{Golbraikh2014} scores a binary dataset by the
    proportion of nearest-neighbor pairs that share an activity class, producing a single value
    for how strongly the similarity assumption holds on that particular combination of
    representation and data."
28. (1350) "\citet{Riniker2013} benchmarked 12 fingerprints and two simple baseline fingerprints
    for similarity-based virtual screening on 88 targets from three public collections, and
    found that the screening performance of the 12 fingerprints was highly correlated."
29. (1360) "When your dataset covers only a small region of the chemical space, a lot of the
    features within these generalized representations become irrelevant, describing chemistry
    which is absent from the data at hand."
30. (1365) "To investigate similarity estimation, we first look at a simple measure: how many
    features vary within a dataset."
31. (1365) "We define a constant feature to be one that takes a single value in every molecule
    of a dataset and a near-constant feature to be one that carries the same value in at least
    95\% of the molecules."
32. (1367) "ECFP4 is the worst offender. On hERG $K_i$ (1,412 molecules) 1,815 of the 2,048
    features are near-constant."
33. (1367) "They come directly from the saved encoder output rather than the data, and we have
    not looked into why."
34. (1376) "That's great behavior if those features define an AC."
35. (1847) "Not every molecular representation is equal. In fact, they hold very different types
    of information."
36. (1847) "Descriptor-based models held 73\% of top-three positions on 11 benchmarks
    \citep{Jiang2021}."
37. (1847) "We know that the choice of representation matters, as is discussed in
    Chapter~\ref{chap:robustness}, yet this choice is usually decided by trial-and-error."
38. (1849) "One of the reasons just pooling features from different representations together
    doesn't work is dimensionality."
39. (1892) "How do you measure what features are ``best''?"
40. (1894) "In a dream world, there would be a single answer to each one of these questions that
    applies to every dataset. Unfortunately we live in reality, and as I discovered, the answer
    to every single one of these questions is ``depends on the data.''"
41. (1896) "Formally, an adaptive hybrid representation is a set of fixed number of features,
    capped at a fixed budget, selected from several base representations of the same
    molecules."
42. (1905) "If the selected featues make sense chemically for that endpoint, it's a good sign
    that the selection mechanism is finding real signal instead of fitting nose."

### B.5 Her AI-phrasing guide, every rule (~/Documents/ai_phrasing.rtf, read with textutil)

Removed or replaced:
1. "providing actionable guidance for future noise-robust QSAR development": removed, as
   corporate or consultancy language.
2. "revealed an unexpected hierarchy" becomes "shows".
3. "warranted detailed investigation": the whole phrase is removed.
4. "emerged as the most consistently robust" becomes "achieved the most consistent robustness".
   "Emerged as" is overused.
5. "The critical insight emerges from" becomes "Loss function effects vary". Dramatic framing
   is removed.
6. "This pattern suggests that": every instance is removed in favour of a direct statement.
7. "This indicates that" and "indicating that": removed. Replace with "shows that" or an em
   dash, or drop the connector.
8. "This compression indicates" becomes "Explicit heteroscedastic modeling compensates": a
   direct statement, not meta-commentary.
9. "This distributional shift indicates" becomes a direct statement.
10. "This variance differential indicates" becomes a direct statement.
11. "This variance compression failure suggests" becomes a direct statement.
12. "This weak correlation suggests" becomes "shows".
13. "This clustering indicates" becomes a direct statement.
14. "Several promising avenues emerge from these findings": removed, as executive-summary
    language.
15. "demonstrates that" becomes "shows that" in most cases.

Patterns it names:
- "indicates/suggests/demonstrates that";
- "emerged as";
- "warranted investigation";
- "The critical insight is";
- "Several X emerge";
- "providing actionable guidance".

What it asks for instead:
- simple verbs: "shows", "achieved", "produced";
- em dashes for causal links;
- no interpretive layer between the data and the conclusion;
- no meta-commentary about the finding.

Note: her own prose uses no em dashes, even though the guide allows them.

Related standing rules from memory (dated claims, UNVERIFIED as text in the repo):
- American spelling and "we" (2026-09-29).
- No runs of short sentences, minimal jargon (2026-09-30).
- Captions say only what is shown, in 2-4 sentences (2026-09-30).
- No reference to the Chapter 3 paper or to the confirmation report.

### B.6 How Claude-written text and TODOs are marked in thesis.tex (counts by grep, VERIFIED)

| Marker | Count | Meaning |
|---|---|---|
| `% TODO: review, needs manual review` | 198 plain, plus about 40 with a suffix such as `-- corrected 2026-10-08 against ...`, `(caption ...)`, `-- empty table, waiting on the full runs` | The standard mark above every paragraph or caption Claude wrote. |
| `% TODO: REVIEW MODIFICATION -- <what changed>` | 16 | An older form of the same mark, for edits to existing text. |
| `% [CLAUDE 2026-10-09] citation fix: ...` / `citation note:` | 73 / 5 | Claude changed or questioned a citation in her sentence. |
| `% CODE INPUT` / `% CODE INPUT: <reason>` / `% [CODE INPUT: ...]` | 3 + several | Text or figures put in by Claude. |
| `% [OLD TEXT replaced by CODE INPUT below]` | 2 | Her original text kept as a comment above Claude's replacement. |
| `% ---- NOTES: <heading> ----` ... `% ---- END NOTES ----` | 28 pairs | Planning notes per section, opened by `% Takeaway:` lines (45). |
| `% [notes moved to thesis_notes_moved.md]` | many | Notes moved out of the file. |
| `\todo[inline]{...}` | 12 | Visible markers in the PDF, for example "Preliminary results from local runs on hERG $K_i$ with 3 folds. These may change when the full runs finish." |
| `% TODO: Results TODO(prelim): ...` / `% TODO: Preliminary -- replace with the full ...` / `% TODO: check against final Ch...` / `% TODO: Ch3 update -- ...` | 10 / 6 / 12 / 5 | Waiting on results or on the Chapter 3 update. |
| `% TODO: manual review` | 1 (line 465) | The exact phrase the task asked about. It sits above the LNL paragraph of sec:lnl_background, which also carries a CLAUDE citation fix. |
| `% [SPLIT 2026-09-27]` | 0 | CLAUDE.md says these exist. There are none now. |

The convention to follow for new Claude prose is `% TODO: review, needs manual review` on the
line above the paragraph. Add ` -- <what was done, source>` if useful. Memory rule
feedback_thesis_notes_failures_2026_09_28 gives the same convention: "Every paragraph of prose
Claude writes gets '% TODO: review, needs manual review' above it".
