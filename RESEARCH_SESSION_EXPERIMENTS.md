# What the experiments in this thesis actually do — a code-level companion to thesis.tex

Compiled 14 October 2026 from the KIRBy repository (`github: adpunt/KIRBy`, branch `similarity-metrics-study`), the server result copies of 9–10 October 2026, and `thesis.tex` as of commit `e99788e` (10 October 2026).

## Who this is for and how to use it

You are a research session helping the author with background literature research and with splitting this thesis into separate papers. You have `thesis.tex`, and ideally `refs.bib` and `additional_files_body.tex` (the Additional files). You do not have the code.

The thesis names no experiment IDs. This document supplies them. For each experiment it gives:
- the question it answers,
- how the code works (inputs, what is varied, what is held fixed, folds, metric, control),
- whether it ran, on which datasets,
- the results that exist, as score pairs in each dataset's own metric,
- where it appears in `thesis.tex`, by `\label`, or "not in thesis".

Use the `\label` names to jump between this document and the thesis. An experiment marked "not in thesis" has code, and sometimes results, that the thesis does not report. Those are material a paper could draw on, or background the thesis leaves out.

## Thesis chapters and where their experiments come from

| Chapter | `\label` | Experiments behind it |
|---|---|---|
| 1 Introduction | `chap:intro` | none |
| 2 Background (QSAR, label noise, noise injection) | `chap:background` | none; defines the noise kinds used in Ch3 and Ch7 |
| 3 Noise robustness across representations and models | `chap:robustness` | Not in this repo. It is the NoiseInject paper, whose code is the separate repository `qsar_qm_models`. Single representations × 13 models × 7 noise kinds on QM9, LogD, Caco-2, hERG Ki. |
| 4 Limitations of the similar property principle | `chap:flaws` | `opig_shared_vectors_census` (molecules that share one feature vector); a distance study in `scratchpad/thesis_code_todo/distance/` (see the similarity section); an earlier exploratory sweep of 12 noisy-label methods on QM9 (221 configurations; `sec:lnl_motivation`), whose code this document does not cover. n09 reruns the same 12 methods on hybrids for Ch7. |
| 5 Building adaptive hybrid representations | `chap:method` | complexity suite (c01–c33), framework f* build decisions (unnamed), federated fed01–fed04 (simulated separate sites), OPIG qsar / shared_vectors_remedy / blueprint_transfer |
| 6 Similarity in an adaptive hybrid representation | `chap:similarity` | similarity suite s01–s13; c04 |
| 7 Detecting and handling label noise | `chap:noise` | noise suite n02–n41 |
| 8 Conclusions | `chap:discussion` | none |

The thesis states its own paper plan in Chapter 1: Chapters 5 and 6 are being prepared as one paper, and Chapter 7 as a separate paper. Chapter 3 is the already-written NoiseInject paper.

## The five experiment suites

All live in `tests/experiments/` of the repository.

- **Complexity** (`c*`, 30 files): does the hybrid beat single representations and other ways of reducing features, and why. Feeds Ch5.
- **Similarity** (`s*`, 12 files): does distance in a hybrid track label differences better than distance in a single representation, across six uses of distance. Feeds Ch6.
- **Noise** (`n*`, 37 files): can damaged labels be found, and does acting on them help. Feeds Ch7.
- **Federated** (`fed*`, 5 files): the hybrid search when data sit at separate sites. Collaboration context: Lhasa Limited. Feeds Ch5 and its Additional files.
- **OPIG** (`opig_*`, 15 files): experiments set up with the Oxford Protein Informatics Group, mostly antibodies and activity-cliff targets. Spread across Ch4 and Ch5.

A sixth group, **framework** (`f00`–`f40`), holds build decisions for the method itself: which allocation method, importance method, distance, objective and pruning schedule. The thesis reports some outcomes in Ch5 Methods without naming the experiments. They are listed in one line each at the end of the shared section.

## Where the numbers come from

- Results are server runs. Two copies exist: `kirby_results_2026-10-09` and the newer `kirby_results_2026-10-10b`. The thesis `% SOURCE` comments cite 10-10b. Each section says which copy a number was read from.
- Each entry says how a number was checked: against the raw file, quoted from the thesis, or quoted from a per-experiment summary page written by an earlier session (not re-checked).
- Most noise experiments are a one-fold screen (outer fold 0 only). A few were promoted to three folds. Complexity and similarity experiments run five outer folds unless stated otherwise.
- Local runs on the author's laptop were for debugging only. They are not results and are not reported.

## Author's conventions that matter when drafting papers

- The method is called the **adaptive hybrid representation** ("hybrid"). "KIRBy" is the name of the code repository only, not of the method.
- Baselines are single representations, not concatenations of them.
- No significance tests across folds or datasets. Results are per-fold scores with their spread, reported per dataset in the dataset's own metric.
- No "k of N datasets" tallies as results.
- Splits are always grouped: scaffold groups for small molecules; mutated position, sequence cluster or complex for antibodies. Never random splits.
- No patient or participant data appears in the thesis, any paper drawn from it, or this document.
- American spelling, "we".

## Contents

1. Shared machinery: the hybrid search, datasets, representations, models, splits, glossary, framework experiments
2. Complexity suite
3. Similarity suite
4. Noise suite
5. Federated suite
6. OPIG suite
7. Appendix: unverified points found while compiling this document


---

# Shared machinery

Read from the code on 2026-10-14. Paths are relative to the repo root.

## Where the thesis describes it

Chapter 5, "Building adaptive hybrid representations" (`\label{chap:method}`), Methods section, describes this machinery. Chapters 6 (similarity) and 7 (noise) reuse it with a different objective or noisy training labels.

## 1. The adaptive hybrid representation

Two files hold it. `src/kirby/find_best_hybrid.py` runs the search. `src/kirby/hybrid.py` builds one hybrid.

**Inputs.** `find_best_hybrid(base_reps, labels, groups, ...)` takes one precomputed feature matrix per representation, labels, one group label per sample, a metric, downstream models and a list of budgets.

**Budget.** The budget is the maximum number of features the hybrid may hold. It is a ceiling, not a target, so a hybrid can come back narrower.

**Per configuration, inside one fold of the search:**
1. The fold's training part is split again, 80% / 20%, by the first fold of a group-aware `GroupKFold` (`hybrid.internal_holdout_split`).
2. An internal evaluator is fitted on each representation alone on the 80% part and scored on the 20% part. That gives one score per representation.
3. Each score is rescaled so 0 is a trivial predictor and 1 is perfect. For RMSE this is `1 − RMSE/RMSE_of_predicting_the_training_mean`.
4. An optional representation count keeps only the N best-scoring representations.
5. The allocation method turns the scores into weights. Each representation gets a share of the budget proportional to its weight, capped at its width.
6. An importance method ranks features within each representation. The top features of each share are kept and concatenated.
7. The downstream model is fitted on the hybrid and scored on the fold's validation part.

The configuration's score is the mean over the search's folds. Each (downstream model, budget) pair has its own winner.

**Importance methods** (`hybrid.compute_feature_importance`). `lightgbm_gain` is LightGBM gain importance from 50 trees of depth 6. `treeshap` is mean absolute SHAP value from a 50-tree random forest of depth 10. `deeplift` is Captum DeepLIFT on a small torch network with hidden layers of 64 and 32 units. Engine default list: `['deeplift', 'lightgbm_gain', 'treeshap']`.

**Allocation methods** (`hybrid.NAMED_ALLOCATIONS`). The engine default is five:
- `performance_weighted`: weight equals the rescaled score.
- `power_2`: weight equals the score squared.
- `softmax_T1`: softmax of the scores at temperature 1.
- `rank_exp_d0.3`: weight decays exponentially with rank, decay 0.3.
- `wglobal_power2`: squared-score weights size each representation's slice of one pooled candidate list, then one global importance ranking picks the budget.

The greedy allocations still resolve by name, but they are cut.

**Internal evaluators.** Engine default: `['match', 'rf', 'lightgbm', 'dnn']`. `match` uses the same model type as the downstream model. The internal evaluator only sets each representation's share. It does not rank features within a representation.

**Objectives.** `objective='predictive'` is the default and fits the downstream model. `objective='similarity'` fits no model. It predicts each molecule's label from its five nearest neighbours in the hybrid's feature space and reports Spearman ρ against the true label (`_readacross_objective_score`). The neighbours use distance weighting, manhattan distance by default (`kirby/similarity/distance_config.py`), and group-aware folds over the held-out rows. On binary labels it returns 2 × (AUROC − 0.5). An older similarity-scoring objective is cut and raises an error.

**Searches per fold.** `tests/experiments/orchestrate/search_unit.py` builds three searches on each outer fold's training rows:
- `general` uses the predictive objective.
- `general_distance` (the "kNN search") adds kNN as a downstream model and uses its own evaluator list.
- `general` is run a second time with `objective='similarity'`.

**Defaults experiments actually use.** These come from `tests/experiments/decided_grid.json`. The author set it per dataset on 2026-10-03. `tests/experiments/build_decided_grid.py` generates it from a decisions file. `decided_grid.apply_decided_grid` narrows every search to it, and an undecided dataset raises an error. Each dataset gets two downstream models. Budgets range from 10 to 2,000 features, three to eight values per dataset. Every dataset uses no augmentation (`aug_conditions = ['none']`) in the general search. Engine defaults it overrides: budgets `[10, 25, 50, 100, 250, 500, 1000, 2000]` (`_shared.FULL_BUDGETS`) and representation counts `[2, 3, 5, None]`, where None means the whole pool.

**Pruning.** Adaptive pruning drops configurations after two folds. It is off by default (`adaptive_pruning=False`). The thesis says the per-fold searches ran without it.

**Tuning.** Downstream models are not tuned by default. `tune=None` reads the `KIRBY_TUNE` environment variable, which is off unless set.

**Caching.** Searches, per-representation scores, fold splits and per-fold hybrid scores are cached under `$KIRBY_CACHE_DIR`. A metadata header is checked on load; a mismatch recomputes.

## 2. Datasets

`tests/experiments/dataset_policy.py` registers each dataset with its task, metric and domain. The live roster is `plan.dataset_roster()` in `tests/experiments/orchestrate/plan.py`. It contains every dataset in `DATASET_REPS`, minus `EXCLUDED_DATASETS` (`chembl_qsar`, `davis`) and minus `CUT_DATASETS`. It holds 20 datasets. Sizes in samples are from thesis Table `tab:datasets`. Code records sizes for only some of them.

| Domain | Dataset (code name) | n samples | Metric | Fold groups |
|---|---|---|---|---|
| Small molecule | `qm9` | 9,978 | RMSE | scaffold |
| | `herg_ki` | 1,412 | RMSE | scaffold |
| | `caco2` | 2,158 | RMSE | scaffold |
| | `logd` | 5,028 | RMSE | scaffold |
| | `adme_rlm` | 3,053 | RMSE | scaffold |
| | `adme_mdr1` | 2,641 | RMSE | scaffold |
| | `herg` (binary, Lhasa) | 7,772 | AUROC | scaffold |
| Antibody | `flab_expression_koenig`, `flab_binding_koenig` | 4,275 each | Spearman | mutated position |
| | `flab_thermo_tresanco` | 672 | Spearman | 95% identity cluster |
| | `abagym_EGFR`, `abagym_Ang2` | 659, 397 | Spearman | mutated position |
| | `humanness_therapeutic` | 564 | AUROC | 95% identity cluster |
| | `ddg_combined` | 429 | Spearman | protein complex |
| | `ginkgo_colloidal`, `_hic`, `_acsins`, `_titer`, `_pr_cho` | 242, 242, 242, 239, 197 | Spearman | competition fold (5) |
| Reaction | `buchwald_hartwig` | 3,955 | RMSE | additive |

`herg_ic50` is an extra row for six noise experiments, searched on one row per molecule. The 19 entries of `CUT_DATASETS` include `lhasa_herg`, `suzuki`, `kiba`, `bindingdb_kd`, `adme_hlm`, `abcov_kd`, `abagym_VEGF`, `tz_her2_*` and five nucleotide sets. The federated suite still uses `lhasa_herg`, with `caco2` and `logd`. Loaders are in `src/kirby/datasets/`.

## 3. Representations

The representation list for each dataset is `DATASET_REPS` in `tests/experiments/complexity/complexity_helpers.py`. Each entry has a `general` list, which the search selects from, and an `augmentation` list of small donor feature sets. Most datasets have 10 general representations, and `ddg_combined` has 8. The code that computes them is in `src/kirby/representations/`: `molecular.py`, `antibody.py`, `protein.py`, `mutation.py`, `structural.py`, `reaction.py` and `nucleotide.py`. Computed matrices are cached to disk per dataset (`_shared._cache_rep_dict`).

- **Small molecules:** fingerprints (`ecfp4`, `avalon`, `atom_pair`, `maccs`, `top_torsion`), descriptors (`mordred`, `pdv` = RDKit descriptor vector), graph kernel, and pretrained embeddings (`mhg_gnn`, `grover`, `molformer`, `chembert`, `chemberta`, `smited`, `mol2vec`, `graphmvp`, `selformer`). The augmentation donors are 3D sets: `pharmacophore3d`, `usr`, `usrcat` and `coulomb_matrix`.
- **Antibodies:** protein language models (`esm2` and its layer variants, `esm3`, `esmc`, `prot_t5`, `saprot`). They also include antibody language models (`ablang2`, `antiberty`, `abroberta`, `igt5`), with `_cdr` variants restricted to CDR residues. Sequence encodings are `onehot`, `blosum` and `aac`. `mutation_features` and `esm2_wt_delta` record differences from the parent sequence.
- **Reactions:** `rxnfp`, `reactiont5`, `cgr_morgan` (condensed graph of reaction), `onehot_components`, and molecular representations of reactants, products or their difference (`reactant_*`, `diff_*`).

## 4. Models

Factories are `DEFAULT_MODELS` in `find_best_hybrid.py`. All run at fixed default hyperparameters.
- `rf`: 100 trees, `max_features='sqrt'`.
- `xgboost`: 100 trees, `colsample_bytree=0.5`.
- `lightgbm`: 100 trees.
- `svm`: `SVRScaledTarget`, an SVR fitted on standardised labels. Classification uses `SVC`.
- `gp`: RBF + white-noise kernel, length scale starting at √(number of features), 3 optimiser restarts. Capped at 1,000 training samples.
- `dnn`: `src/kirby/models/dnn.py`, hidden layers 256-128-64, dropout 0.2, 200 epochs, early stopping with patience 20.
- `knn`: k = 5, lowered to √(training samples) on folds under 25 samples.

`mlp` and `catboost` are switched off. Features are standardised by `kirby.scaling.FeatureScaler`, which leaves 0/1 features unscaled. Each dataset's two models are its row in `decided_grid.json`.

## 5. Splitting

There are no random splits. `_shared.cv_split` raises an error if groups are missing. It uses `GroupKFold` for regression and `StratifiedGroupKFold` with seed 42 for classification. There are three levels:
- 5 outer folds (`nested_cv_outer_folds`, `DEFAULT_N_FOLDS = 5`, seed 42).
- 5 inner folds of the search on each outer training part.
- The 80/20 internal split inside each inner training part.

The search sees only outer-training rows. The winner is rebuilt there and scored once on the outer test fold. Some experiments run only the first one or three outer folds. `_shared.OUTER_FOLDS_RUN` sets that per experiment, and most noise screens run fold 0 only.

## 6. Running on the server

`tests/slurm_scripts/worker.sh` runs `tests/experiments/worker.py`. The worker takes the next dataset from a ledger (`tests/experiments/orchestrate/ledger.py`, state in `<run>/ledger.json`, readable summary in `<run>/LEDGER.md`). Each dataset row runs its stages in order: `complexity`, `federated`, `similarity`, `noise`, `noise_full`. The `complexity` stage starts with one search unit per outer fold (`search_f0` … `search_f4`). Later experiments read those searches. Results land at `tests/results/<run>/<category>/<experiment>/<dataset>.json`.

## 7. Glossary

- **representation count**: how many of the best-scoring representations a configuration may use; None = all.
- **importance method**: how features are ranked within one representation (`lightgbm_gain`, `treeshap`, `deeplift`).
- **allocation method**: how the budget is split across representations, as a function of each one's internal-evaluator score.
- **internal evaluator**: model fitted on one representation at a time on the 80% part, scored on the 20% part; its score sets that representation's share.
- **`match`**: internal evaluator of the same type as the downstream model.
- **configuration**: one combination of budget, representation count, importance method, allocation method, internal evaluator and downstream model.
- **objective**: what the search maximises. `predictive` is the downstream score. `similarity` is Spearman ρ of each molecule's label predicted from its five nearest neighbours.
- **`general` / `general_distance`**: the predictive search, and the kNN search that adds a kNN downstream model.
- **blueprint**: the recorded settings of a winning configuration, rebuilt per fold.
- **augmentation / donor**: small feature sets added after the hybrid is built. They are added uncapped (outside the budget) or capped (replacing base features). Off in the decided grids except the kNN search on `ginkgo_colloidal` and `ginkgo_hic`.
- **noise level**: injected label noise as a fraction of the clean label standard deviation (`_DEFAULT_NOISE_LEVELS = 0, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5`). Censoring is measured instead as the fraction of labels clipped. **Level 0** is the clean-label control. Under noise the search trains on noisy labels and scores on clean ones (`label_noise`, `kirby/noise_spec.py`).

## 8. Framework (`f*`) experiments

These are in `tests/experiments/framework/`. They are build decisions, not published results. `thesis.tex` names none of them by ID. Chapter 5 reports some of their outcomes without naming them. Which Chapter 5 statement comes from which `f*` is unverified. One line each, from each script's own docstring:

- f00: where to screen out weak representations before the search.
- f01: allocation families compared; smooth weightings beat greedy.
- f02: what adaptive pruning would cut, and at what cost.
- f03: greedy step size and patience (greedy since cut).
- f04: two-phase coarse-then-fine search against the full grid.
- f05: correlation and speed of 8 importance methods; redundant ones dropped.
- f06: weaker representations as base representations or as augmentation donors.
- f07: internal evaluator choice inside greedy allocation.
- f08: design of the augmentation step.
- f09: synthesis of f00–f08 across domains.
- f10: fixes for hybrids losing on datasets under 200 samples.
- f11: allocation × internal evaluator on small datasets.
- f12: allocation × importance × evaluator with an exact budget.
- f13: should the internal evaluator match the downstream model (origin of `match`).
- f14: does greedy fill its budget (led to cutting greedy).
- f15: is pruning after fold 0 safe.
- f16: multi-stage pruning thresholds.
- f17: how importance weights are normalised before entering a distance.
- f18: which distance each scoring step uses.
- f19: which objective to optimise, and blends of objectives.
- f20: whether maximum-common-substructure fidelity works inside a fold.
- f21: what should set each representation's share under a similarity objective.
- f22: whether extra similarity feature sets beat the existing representations.
- f23: which similarity space finds wrong labels best.
- f24: which way of admitting augmentation features works.
- f25: length of the ranked augmentation candidate list.
- f26: penalising redundancy between representations in allocation.
- f27: objective × representations × budget, built through the engine.
- f28: every objective alone, then every blend.
- f29: model survey on single representations, and whether tuning pays (dnn replaced mlp).
- f30: new importance methods against the current three.
- f31: internal evaluator hyperparameters.
- f32: how representation scores are rescaled before splitting the budget (one fixed rule per metric).
- f33: recalibration of pruning thresholds on the current search space (source of the shipped schedule).
- f34: kNN neighbour weighting × distance coefficient × k, and the loss function of downstream models.
- f35: whether f34's answer survives the other scoring code path.
- f36: label-smoothing settings used by noise experiments.
- f37: how the kinds of label noise differ in their damage.
- f38: which kNN setting best flags corrupted training labels.
- f39: whether the kNN-setting screen predicts training or held-out molecules.
- f40: three ways to attribute a hybrid's predictions to representations (debug only).


---

## Complexity experiments (c01–c33): building and testing the adaptive hybrid representation

- The suite asks whether the hybrid built by the search (features drawn from several representations, a budget of features split between them) beats single representations and other ways of reducing features, and why.
- Code: `tests/experiments/complexity/c*.py`, 29 experiment files plus helpers. c10 lives in `tests/experiments/noise/` but belongs to this suite by its ID and docstring.
- `complexity_helpers.py` is the shared runner. `run_all(dataset)` loads one dataset once, applies the dataset's decided search grid (`decided_config`), and runs each experiment in its module map: c01, c02, c04–c09, c11–c13, c15–c19, c21–c24, c33. It also stamps each output with the feature count of every representation (`rep_dim_summary`) and holds the per-dataset representation lists.
- Not in that map, so not run per dataset by the pipeline: c03, c14 (cross-dataset readers of the per-dataset outputs; never ran), c20, c31, c32 (run by hand on `_alloc_harness`, July–October 2026, or never).
- Shared setting (since 2026-10-05): five grouped outer folds (scaffold groups for molecules, sequence groups for antibodies, seed 42). A hybrid search runs on each outer fold's training molecules only. Per downstream model it keeps the two best single representations and two best hybrids, refit and scored on the held-out fold.
- Each dataset uses its two decided downstream models. Metric: RMSE for small molecules and reaction yield, Spearman for antibody regression, AUROC for binary classification, weighted F1 for multiclass.
- Thesis home: Chapter "Building adaptive hybrid representations" (`\label{chap:method}`) and its Additional files (`app:add:*`, in `thesis_rework/stage1_takeaways_bullets/additional_files_body.tex`). c04 feeds one table in Chapter `chap:similarity`. `tab:experiment_register` (in `sec:method:comparators`) lists the chapter's experiments with dataset counts. thesis.tex names no experiment IDs; the mapping below comes from SOURCE comments and from matching numbers.
- Results: server copy `/Volumes/seagate/kirby_results_2026-10-09/complexity/` and the newer `/Volumes/seagate/kirby_results_2026-10-10b/complexity/`, which thesis SOURCE comments cite. Numbers below are from 10-10b unless a line says otherwise.
- Datasets: up to 25 per experiment (small molecules, Buchwald–Hartwig reactions, antibodies). The 10-10b ledger marks abagym_VEGF, abcov_kd, adme_hlm, bh_doyle_plates, cdr_flexibility, tz_her2_binary and tz_her2_multiclass as cut from the thesis; the thesis uses 20 datasets. Files written before the cuts are still reported.
- tz_her2_binary and tz_her2_multiclass are trastuzumab–HER2 antibody binding sets. Their rows are reported for every experiment except c18 and c19, whose tz_her2 files exist but were not read in this pass.

Code names against thesis names (matched by identical numbers): `flab_expression_koenig` = anti-VEGF scan expression; `flab_binding_koenig` = anti-VEGF scan binding; `flab_thermo_tresanco` = nanobody thermostability; `herg` = hERG blocking; `abagym_Ang2` = angiopoietin-2 binding. Matched by name only: `ginkgo_colloidal` = SMAC colloidal stability; `ginkgo_pr_cho` = CHO polyreactivity; `ginkgo_titer` = expression titer; `ginkgo_hic` = HIC retention; `ginkgo_acsins` = AC-SINS; `adme_rlm` = rat microsomal clearance; `adme_mdr1` = MDR1–MDCK efflux.

"Checked" in a line means the number was found in the raw JSON on 2026-10-14. "From answer page" means it comes from `/Volumes/seagate/claude_analysis_2026-10-08/answers/cNN.md` and was not re-checked.

### c01 — each representation scored on its own  (`tests/experiments/complexity/c01_rep_characterization.py`)
- **Question:** For each dataset and each of its two models, how well does every single representation predict the label when fitted alone on all its features? It also profiles each representation's dimensionality and its mutual information with the label.
- **How it works:**
  - Inputs: the dataset's pruned "general" representation list (`DATASET_REPS[ds]['general']`, 10 on most datasets, 8 on ddg_combined) and its two decided models.
  - Per representation, per model, per outer fold: features scaled on the training molecules (0/1 features left unscaled), held-out features clipped to the training range, model fitted, held-out score in the dataset's own metric.
  - Nothing is selected or searched. No budget, no hybrid.
  - Profile numbers, not scored per fold: number of principal components for 90/95/99% of variance, a maximum-likelihood intrinsic dimension, effective rank (exponential of the entropy of normalised singular values).
  - Mutual information: a k-nearest-neighbour (KSG, k=5) estimate summed over up to 50 principal components, in nats. It is computed once on all molecules, capped at a random 2,000.
  - Because it is a sum over components, it can exceed what a joint mutual information allows (code comment says so).
- **Status:** Ran on the server, 5 folds, on 27 datasets covered here (20 thesis datasets plus the 7 cut ones, including both tz_her2 sets). The 10-10b files are identical to the 10-09 files (thesis SOURCE comment; not re-checked by me beyond logd).
  - gp results on abagym_Ang2, abagym_VEGF, ginkgo_acsins, ginkgo_hic, ginkgo_pr_cho were re-run on 8 Oct after a gp fix. The old files sit in `superseded/gp_fix/`.
  - Known fault (FAULTS.md A8): c01 clips held-out features to the training range; most other experiments do not. So c01 scores and c02 scores of the same representation can differ in that one way.
  - Known fault (FAULTS.md A22): mutual information uses all molecules, while scores are per held-out fold. So "is the highest-information representation also the best held out" compares a whole-dataset number with a held-out one.
  - Known fault (FAULTS.md A1): every dnn number here predates the 8 Oct change that clips dnn predictions to the training-label range.
- **Results so far:**
  - Best single representation, mean of 5 held-out folds (from the answer page; logd checked against the raw file):
    - logd: mhg_gnn, 0.4466 RMSE with dnn (fold SD 0.0185) and 0.4930 with svm. Checked.
    - caco2: pdv, 0.2980 RMSE dnn, 0.3001 svm.
    - qm9: mordred, 0.3619 RMSE dnn, 0.3720 lightgbm.
    - herg_ki: mhg_gnn, 0.5881 RMSE svm, 0.5926 dnn.
    - adme_rlm and adme_mdr1: pdv under both models (0.5209 and 0.4678 RMSE with svm).
    - buchwald_hartwig: diff_avalon 14.19 RMSE with rf; cgr_morgan 15.15 with lightgbm.
    - flab_expression_koenig: esm3_cdr, 0.7249 Spearman svm, 0.7243 dnn.
    - flab_thermo_tresanco: saprot, 0.6939 Spearman dnn, 0.6901 svm.
    - abagym_Ang2: saprot 0.6015 Spearman with gp; antiberty 0.5774 with svm.
    - tz_her2_binary: blosum, 0.8657 AUROC with xgboost and 0.8616 with lightgbm. Fold scores with xgboost run 0.812 to 0.925. Checked.
    - tz_her2_multiclass: igt5 0.6752 weighted F1 with svm; blosum 0.7164 with xgboost. Checked.
  - The two models pick a different first representation on most antibody datasets and on buchwald_hartwig and herg. On caco2, logd, qm9, herg_ki and the adme sets they pick the same one.
  - The representation with the highest summed mutual information is the held-out best only on flab_expression_koenig under both models (answer page).
- **In thesis:** Chapter `chap:method`, sec `sec:method:results_single`, columns "Best in hindsight" and "Its score" of Table `tab:hybrid_vs_best_single`. Also Additional file `app:add:hindsight_shuffle` (Tables `tab:method:hindsight_smallmol`, `tab:method:hindsight_antibodies`). The profile numbers (intrinsic dimension, effective rank, mutual information) appear nowhere in thesis.tex (grep for "effective rank", "intrinsic dim").
- **Related:** c02 (supplies the hybrid column next to c01's in the same tables), c22 (shuffling columns of the hindsight tables), c17 (another mutual-information measure).

### c02 — hybrid against single representation, budget per model, and PCA  (`tests/experiments/complexity/c02_complexity_grid.py`)
- **Question:** Per dataset and model, does the search's top hybrid beat the top single representation on held-out molecules? Does it beat PCA of all representations joined at the same number of features? Which budget does each model's search pick?
- **How it works:**
  - Per outer fold, one hybrid search on that fold's training molecules over the dataset's decided grid, pruning off. Grid axes: budget (number of features, decided per dataset, e.g. 500/1,000/2,000 on logd, caco2, qm9), importance method, allocation method, internal evaluator, number of top representations used.
  - Standard selection per model: the 2 best single representations ("base1", "base2") and the 2 best hybrids ("hybrid1", "hybrid2"), by inner cross-validation score on the training molecules.
  - Each is refitted on the training molecules and scored on the held-out fold. Hybrids are rebuilt with that fold's own per-representation scores.
  - PCA control: all general representations joined, scaled on training molecules, held-out features clipped, PCA to each decided budget (capped at training molecules minus 1), model fitted on components.
  - Also stored from the training-molecule search only: a model × budget table of best scores, the best configuration per model, which importance method won, and the ranking of representations.
  - On ginkgo_colloidal and ginkgo_hic only, a distance-grid search with and without extra "augmentation" representations appended outside the budget.
  - Arms dropped by decision (`ARMS_NOT_RUN` in the code, 2026-10-06): fine-tuned representations, a search at one representation's own width, the "capped" augmentation condition, a distance hybrid.
- **Status:** Ran on the server, 5 folds. 10-10b holds 26 datasets covered here, including both tz_her2 sets (both cut). Against 10-09, 10-10b adds flab_binding_koenig (5 folds) and removes adme_hlm.
  - adme_hlm and the earlier adme_rlm files sit in `superseded/dnn_clip_c02_c08_2026_10_09/`. adme_hlm is a cut dataset.
  - gp datasets re-run on 8 Oct after the gp fix; old files in `superseded/gp_fix/`.
  - Known fault (FAULTS.md A6): hybrids are often built with fewer features than their budget. The answer page counts 162 of 480 held-out hybrids below budget.
  - Known fault (FAULTS.md A7): PCA components stop at training molecules minus 1, so on small datasets several budgets fit the same components.
  - Known fault (FAULTS.md A1): adme_hlm and adme_rlm inner-search dnn scores were computed before dnn predictions were clipped, and those searches were not re-run.
  - Known fault (FAULTS.md A4): c05's full-size point should equal c02's, and differs for dnn on five datasets (e.g. flab_thermo_tresanco dnn hybrid 0.6126 vs 0.6764 Spearman).
- **Results so far:** (held-out means of 5 folds; logd, caco2, qm9 checked against the 10-10b raw file)
  - logd: dnn hybrid 0.4089 vs best single (picked on training molecules) 0.4497 RMSE; svm 0.4302 vs 0.4930. Checked.
  - logd dnn: PCA at each fold's picked budget scores 0.392 RMSE, ahead of the hybrid's 0.4089. All five dnn folds picked 2,000 features. Checked.
  - caco2: dnn hybrid 0.2775 vs single 0.3042 RMSE; svm 0.2862 vs 0.3036. Checked.
  - qm9: dnn hybrid 0.3155 vs single 0.3619 RMSE; lightgbm 0.3463 vs 0.3720. lightgbm PCA at the picked budget is 0.5501. Checked.
  - herg_ki: svm 0.5590 vs 0.5881; dnn 0.5691 vs 0.6031 RMSE (thesis table).
  - buchwald_hartwig: rf hybrid 15.24 vs single 14.41 RMSE (hybrid worse); lightgbm 15.62 vs 16.17 (thesis table).
  - flab_expression_koenig: svm hybrid 0.7729 vs single 0.7029 Spearman (thesis table).
  - ddg_combined: dnn hybrid 0.3542 vs single 0.1317 Spearman (thesis table).
  - flab_thermo_tresanco: dnn hybrid 0.6764 vs single 0.6939 Spearman (thesis table).
  - ginkgo_pr_cho: gp hybrid 0.4419 vs single 0.5009 Spearman (thesis table).
  - tz_her2_binary (budgets 100/250/500): xgboost hybrid 0.7458 vs single 0.8524 AUROC; lightgbm 0.8128 vs 0.8541. Checked.
  - tz_her2_binary xgboost: on fold 4 (13 held-out antibodies) the hybrid scores 0.417 AUROC against 0.833; on the other folds it trails by 0.01 to 0.07. Checked.
  - tz_her2_multiclass (budgets 250/500/1,000): svm hybrid 0.6793 vs single 0.6694 weighted F1; xgboost 0.6579 vs 0.7164. Checked.
- **In thesis:**
  - `sec:method:results_single`: columns "Best single base representation" and "Hybrid" of Table `tab:hybrid_vs_best_single`. The register row "Does the hybrid beat the best single representation, and PCA at the same width?" (20 datasets) in `tab:experiment_register`.
  - Observation: the Caco-2 SVM row appears twice in that table's tex source.
  - PCA paragraphs at the end of `sec:method:results_ensembles`, and Additional file `app:add:pca_disagreement`.
  - Per-fold budgets: Additional file `app:add:budget_per_fold` (Table `tab:method:budget_per_fold`); fold-pick dots in Figures `fig:best_number_of_features` and `fig:best_number_of_features_antibodies` (loader `tests/analysis/thesis_loader_f2.py` reads c02).
  - The thesis says it "did not test whether these picks follow measurable properties of a dataset" (sec `sec:method:results_why`). That untested question is c03's.
- **Related:** c01 (hindsight column), c05 (full-size point uses the same searches), c08 and c09 (read c02's result file), c11 and c17 (other PCA comparisons), c06 (budget per model setting).

### c03 — do the best search settings follow dataset properties?  (`tests/experiments/complexity/c03_cross_dataset_transfer.py`)
- **Question:** Across datasets, does the budget the search picks follow measurable properties of the dataset? The docstring states this is not transfer; c24 tests transfer.
- **How it works:**
  - Loads several datasets in one process (`--datasets`, default logd, caco2, herg_ki). Each uses its own decided models, grid and pruned representations.
  - Per dataset: a profile of dataset properties (`compute_dataset_complexity`): effective rank, participation ratio, molecules per dimension, MLE intrinsic dimension, and Lorena et al. 2018 measures C1, C2, L1, S3.
  - Per dataset, per outer fold: the same standard selection as c02 (2 base, 2 hybrids per model), refitted and scored held out.
  - Winning budget per (dataset, model): the budget of each fold's top hybrid; the median over folds is used.
  - Output: one Spearman coefficient per (dataset property, model), each point one dataset. Also a split of the variance in winning budget into between-dataset and between-model parts.
  - No p-values.
- **Status:** Never ran. No result file in the 10-09 or 10-10b copy; the `cross/` folder is empty in both. The 10-10b ledger lists `cross:c03` as pending, waiting on 5 dataset complexity stages.
  - It is launched through `tests/slurm_scripts/runbook/cross_one.sh c03 <datasets>`. That script's header still describes the old design (stored whole-dataset searches, SVR and ElasticNet), which the c03 code no longer does.
  - THESIS_OUTLINE.md (author, 2026-10-09): c03 "KEPT, reported as the conditional-conclusions result for Chapter 5", to start once the signed-off Chapter 5 datasets finish.
- **Results so far:** none.
- **In thesis:** not in thesis. Sec `sec:method:results_why` states the question was not tested.
- **Related:** c02 (same per-fold selection; budgets per fold), c24 (true transfer of settings between datasets), c09 (variance of the search score by model and budget).

### c04 — feature importance used as a distance weighting  (`tests/experiments/complexity/c04_importance_as_metric.py`)
- **Question:** When each molecule's label is predicted from its nearest training molecules in all representations joined, does weighting each feature by its importance beat equal weights?
- **How it works:**
  - No downstream model and no hybrid search. Regression datasets only; classification datasets return a skip (`classification_not_supported`).
  - All pruned general representations joined into one feature set. Scaled per fold on training molecules; Manhattan distance.
  - Importance computed per fold on training molecules with each of the dataset's decided importance methods (lightgbm_gain, treeshap, deeplift; logd and qm9 have lightgbm_gain only by their decided grid).
  - Weight per feature = square root of its importance, normalised to sum to 1; each scaled feature is multiplied by it.
  - Arm 1: scikit-learn kNN regressor with k = 5, 10, 20, equal weight per neighbour, unweighted vs weighted features.
  - Arm 2: 10 nearest neighbours averaged with 1/distance weights, unweighted vs weighted.
  - Arm 3: only the 25 most important features, unweighted vs weighted. Arm 4: features picked by one importance method, weighted by another, at 25 features.
  - Comparator: each single representation alone, 10 neighbours, 1/distance, unweighted.
  - Scores per fold: R-squared, MAE, and the dataset's own metric. No shuffled-weight control.
- **Status:** Ran on the server, 5 folds, on 21 regression datasets covered here; herg, humanness_therapeutic, cdr_flexibility, tz_her2_binary and tz_her2_multiclass hold skip files (classification). 10-10b adds flab_binding_koenig against 10-09.
  - Known fault (FAULTS.md A16): c04 adme_hlm ran with lightgbm_gain only, during a 2-hour window when the code dropped the other two methods. That file is now in `superseded/c04_one_importance_method/`; adme_hlm is a cut dataset.
  - gp-affected antibody datasets were re-run 8 Oct; old files in `superseded/gp_fix/` (c04 fits no gp, so why they were re-run is not verified).
- **Results so far:** (kNN regressor, k = 10, mean of 5 folds; qm9, caco2, logd, flab_expression_koenig checked against the raw file)
  - qm9: unweighted 0.5749 vs lightgbm_gain-weighted 0.4841 RMSE. Checked.
  - logd: 0.6962 vs 0.6693 RMSE (lightgbm_gain). Checked.
  - caco2: unweighted 0.3425; lightgbm_gain 0.3365, treeshap 0.3375, deeplift 0.3428 RMSE. Checked.
  - flab_expression_koenig: unweighted 0.6629; lightgbm_gain 0.7011, treeshap 0.6973, deeplift 0.6657 Spearman. Checked.
  - 1/distance arm, qm9: 0.5691 vs 0.4802 RMSE; best single representation alone 0.6459 (answer page; first pair checked).
  - 1/distance arm, caco2: pdv alone 0.3293 RMSE beats lightgbm_gain-weighted joined features 0.3302. Checked.
  - Thesis also reports ginkgo_colloidal, where lightgbm_gain and treeshap weighting lose (0.121 and 0.122 vs 0.134 Spearman unweighted).
  - The 25-feature and pick-by-one, weight-by-another arms are not used in the thesis. On caco2 they all score 0.335 to 0.338 RMSE (checked).
- **In thesis:** Chapter `chap:similarity`, sec `sec:similarity:results_knn`, Table `tab:similarity:joined_weighting` (k = 10 kNN regressor arm only). The thesis says no shuffled-weights comparison exists.
- **Related:** s05 and s10 (similarity suite weighting comparisons), c19 (importance used in training), the importance weighting table `tab:importance_weighting` nearby (source not checked here).

### c05 — hybrid against single representation as the number of molecules shrinks  (`tests/experiments/complexity/c05_learning_curves.py`)
- **Question:** Does the hybrid's lead over the best single representation change as the training set gets smaller?
- **How it works:**
  - Sample sizes: datasets of 5,000 molecules or fewer use 10, 25, 50, 75 and 100% of molecules. Larger datasets (logd, qm9, herg) use 250, 500, 1,000 and 2,000 molecules, with no full-size point.
  - Each sample is drawn as whole scaffold or sequence groups, 3 draws (seeds) per size; 1 draw at full size.
  - Each draw is split into grouped outer folds. On each fold, one search on the training molecules; the c02-style selection of 2 base and 2 hybrids per model; refitted and scored held out.
  - Curve per model: top hybrid vs top single representation at each size, mean over draws of the fold means.
  - A three-parameter curve (score = asymptote − a/n^b) is fitted only when there are 4 or more points.
  - Datasets with fewer than 10 groups are skipped (the five ginkgo sets).
  - Since 7 Oct, datasets outside a named list of 13 run outer folds 0 to 2 only (`OUTER_FOLDS_RUN`, `ALL_FOLDS_DATASETS` in `_shared.py`).
- **Status:** Ran on the server on 12 datasets covered here with numbers: abagym_EGFR, abagym_VEGF, abcov_kd, buchwald_hartwig, caco2, cdr_flexibility, ddg_combined, flab_thermo_tresanco, herg, herg_ki, logd, qm9. The five ginkgo files are skip files.
  - 10-10b adds abagym_VEGF and the ginkgo_hic skip file against 10-09. No file for tz_her2_binary or tz_her2_multiclass in either copy; FAULTS.md B2 lists tz_her2_binary as pending. No file for abagym_Ang2 (old gp file in `superseded/gp_fix/`), adme_mdr1, adme_rlm, flab_expression_koenig, flab_binding_koenig, humanness_therapeutic.
  - Fold counts are mixed: qm9 3 folds per draw, logd 5 (written 7 Oct 13:50, before the 3-fold rule). FAULTS.md C2 lists more mismatches.
  - Known fault (FAULTS.md A4): the full-size dnn point differs from c02 on five datasets; svm, gp, rf and xgboost match c02 to 4 decimals (answer page).
  - Known fault (FAULTS.md C5): fitted asymptotes above 1 for Spearman on cdr_flexibility and flab_thermo_tresanco svm; a flat-line fit on ddg_combined svm.
  - Cut by the author: THESIS_OUTLINE.md, "CUT: whether the lead grows as the dataset shrinks (c05). Removed from the chapter 2026-10-08."
- **Results so far:** (hybrid vs best single, mean over 3 draws; logd, qm9, caco2, flab_thermo_tresanco, herg_ki checked against the 10-10b raw file)
  - logd dnn: 0.8045 vs 0.8465 RMSE at 250 molecules; 0.5007 vs 0.5769 at 2,000. 5 folds per draw. Checked.
  - qm9 dnn: 0.6716 vs 0.8659 RMSE at 250; 0.4359 vs 0.5005 at 2,000. 3 folds per draw. Checked.
  - qm9 lightgbm: 0.6942 vs 0.7473 at 250; 0.4789 vs 0.4832 at 2,000. Checked.
  - caco2 svm: 0.3468 vs 0.3548 RMSE at 215 molecules; 0.2862 vs 0.3036 at all 2,158. Checked.
  - herg_ki dnn: 0.9173 vs 0.8737 RMSE at 141 molecules (single ahead); 0.5668 vs 0.6146 at all 1,412. Checked.
  - flab_thermo_tresanco svm: 0.0789 vs 0.0949 Spearman at 67 antibodies; 0.6899 vs 0.6901 at all 672. Checked.
- **In thesis:** not in thesis (grep for learning curve, sample size, subsample found nothing). Cut 2026-10-08 per THESIS_OUTLINE.md.
- **Related:** c02 (full-size point shares its searches), c33 (search with one fold fewer of training molecules), c07 (also skips the ginkgo sets).

### c06 — budget picked at each capacity setting of one model  (`tests/experiments/complexity/c06_within_family_gradient.py`)
- **Question:** Within one model family, does a higher-capacity setting make the search pick a larger budget (more features)?
- **How it works:**
  - Families used: only those whose model is one of the dataset's two decided models. Four settings per family, each with a capacity index 1 to 4.
  - SVM (RBF, labels standardised in each fit): C = 0.01, 0.1, 1, 10. DNN: hidden layers (16), (64), (128, 64), (256, 128, 64). RF: depth 3, 5, 10, unlimited. XGBoost: depth 2, 4, 6, unlimited.
  - GP family also exists; its settings were not read here. lightgbm has no family, so datasets with lightgbm as a decided model get one family.
  - Per outer fold, per setting: a full search on the training molecules with that setting as the only downstream model, over the dataset's decided grid. The internal evaluator "match" resolves to the family.
  - Optimal budget per setting = the budget with the best training-molecule score. That hybrid is also rebuilt and scored held out.
  - Per fold: Spearman coefficient between capacity index and optimal budget, if budgets vary. No p-value.
  - Classification datasets are skipped (classifier variants not wired).
  - Since 7 Oct, folds 0 to 2 only outside the named list of 13 datasets.
- **Status:** Ran on the server on 19 datasets covered here; cdr_flexibility, herg, humanness_therapeutic and tz_her2_binary hold skip files (classification). tz_her2_multiclass has no file in either copy. 10-10b adds flab_expression_koenig (3 folds) against 10-09.
  - Fold counts: caco2 and logd 5 folds; adme_rlm 3 folds; adme_mdr1 5 folds although outside the list (FAULTS.md C2, cause not verified).
  - Known fault (FAULTS.md A1): caco2 and logd files were written 7 Oct, before dnn predictions were clipped. adme_rlm dnn fold 1 scored 16,328 to 35,918 RMSE in the 10-09 copy (answer page). The thesis SOURCE comment marks c06 "RERUN-SWAP-DNN"; which dnn files were replaced is not verified.
  - Known fault (FAULTS.md A6): 760 of 2,266 rebuilt hybrids hold fewer features than their budget (answer page).
  - gp family: searches at the two smallest length-scale settings failed on every fold of ginkgo_acsins, ginkgo_hic and ginkgo_pr_cho (answer page).
- **Results so far:** (optimal budget per fold in features; held-out RMSE at the picked budget, mean of 5 folds; caco2 and logd checked against the raw file)
  - caco2 svm: C = 0.01 picks 500 on all 5 folds and scores 0.4002; C = 10 picks 1,000 on 4 of 5 folds and scores 0.2794. Checked.
  - logd svm: C = 0.01 picks 500 on all folds, 0.8602; C = 10 picks 1,000 or 2,000, 0.4126. Checked.
  - logd dnn: 2,000 at every width on every fold but one (64 units, fold 0: 1,000). Held-out 0.4238 RMSE at 16 units, 0.4042 at 64. Checked.
  - adme_rlm svm (3 folds): 500 at C ≤ 0.1, 2,000 at C ≥ 1. Checked.
  - flab_expression_koenig svm (3 folds): 250 at C = 0.01, 2,000 on 2 of 3 folds at C = 10. Checked.
  - qm9 has only the DNN family; its picked budget does not rise with width (answer page).
- **In thesis:** Chapter `chap:method`, sec `sec:method:results_why`, one paragraph ("the budget does depend on the model's own settings"). Additional file `app:add:budget_by_capacity`, Table `tab:method:budget_by_capacity` (herg_ki, caco2, logd, adme_rlm, adme_mdr1 SVM; logd DNN). Register row "Does a smaller hybrid replace tuning the model?" (5 datasets).
- **Related:** c02 (budget per decided model), c09 (variance split by model and budget; reads c06), c16 (SVM C sweep with feature sets).

Groupings (factual, by shared question or code):
- Hybrid against single representations on the same folds: c01 (single representations, hindsight best), c02 (search's pick of both), c05 (same comparison at smaller training sets).
- Which budget the search picks and why: c02 (per model, per fold), c06 (per capacity setting within one model), c03 (against dataset properties, never ran), c09 (variance split, other part).
- Importance as a distance weighting: c04 alone in this part; shares its question with the similarity suite (s05, s10).

### c07 — Easy and hard halves of each dataset  (`tests/experiments/complexity/c07_data_complexity_partitions.py`)
- **Question:** Does the best configuration differ between an "easy" and a "hard" half of a dataset? Does the hybrid still beat the best single representation within each half?
- **How it works:**
  - Datasets larger than 2,000 rows are cut to about 2,000 rows by drawing whole scaffold groups at random (seed 42).
  - Datasets with fewer than 10 scaffold or sequence groups are skipped, with the reason stored.
  - Four ways to split the rows into two halves. `scaffold_frequency`: molecules whose scaffold is in the top 20% by count ("frequent") against the rest ("rare").
  - `molecular_complexity`: heavy-atom count at or below the median against above it (needs SMILES). `target_variance`: scaffolds with low against high label variance; flagged `partition_uses_labels=True` because the label decides the half.
  - `feature_variance`: scaffolds with low against high spread of their rows in the scaled, concatenated representations. Uses no labels. A half needs at least 50 rows.
  - One fold layout is built once on the capped rows and shared by every half. Two arms per half and fold: a search on the half's own training rows, and a search on all training rows whose hybrids are rebuilt on the half.
  - Each of the 2 single representations and 2 hybrids per model is scored on the half's held-out rows. Held-out features are clipped to their range on the training rows (c07, c01 and c08 only).
  - Also recorded per half, on all its rows: effective rank, a summed mutual information with the label, PCA components for 95% variance. Per fold it records whether the two halves' best (model, budget, importance method) differ.
- **Status:** Ran on the server. 25 dataset files in 10-10b, counting the two trastuzumab-HER2 sets; 7 hold a skip. The five ginkgo datasets have 5 sequence groups. On abcov_kd and cdr_flexibility no partition had two halves of 50 rows.
  - 10-10b adds qm9 against 10-09. qm9, adme_rlm and flab_binding_koenig ran 3 folds; the rest 5.
  - Partitions present vary: adme_mdr1 and adme_rlm have no scaffold_frequency split (every row fell in the frequent half). Antibody sets have no molecular_complexity split (no SMILES).
  - Known faults (FAULTS.md, 2026-10-08). A8: c07 clips held-out features to the training range and most other experiments do not, so c07 scores differ from c02 or c11 on the same fold in one way.
  - A6: the hybrid build stops at the last feature with importance above zero. So a hybrid can hold fewer features than its budget, and two budgets can build the same hybrid.
  - A15: c07 skips the ginkgo datasets, while c01, c04 and c06 score them over 5 folds with one sequence group held out per fold. Two rules for the same 5 groups.
  - tz_her2_binary (auc, xgboost and lightgbm) and tz_her2_multiclass (weighted F1, svm and xgboost) are trastuzumab-HER2 CDR-H3 variant sets, 2,940 rows searched each. Identical in 10-09 and 10-10b.
  - On both, the frequent-scaffold half (1,440 rows) is one fold group, so it was not scored. No SMILES, so no heavy-atom split.
  - On both, the shared fold layout leaves several halves with 0 held-out rows on some folds. tz_her2_multiclass has a held-out score only for the rare half of scaffold_frequency, 3 folds.
  - tz_her2_binary target_variance easy half: the hybrid scores 1.0 auc on its one scored fold. Not explained; possibly a fold with too few held-out rows (unverified).
- **Results so far:** In thesis, scaffold_frequency split only (Table tab:method:scaffold_halves), 5 datasets, held-out means over 5 folds (qm9: 3 folds, rare half on 2).
  - logd dnn, rare-scaffold half (649 molecules): 0.670 rmse best single representation against 0.573 rmse hybrid. Frequent half (1,351 molecules): 0.607 against 0.587. Checked against raw JSON (0.6698 and 0.5735; fold spread 0.048 and 0.057).
  - caco2 dnn, rare half (745 molecules): 0.367 against 0.327 rmse. herg_ki svm, rare half (572 molecules): 0.686 against 0.655 rmse. From the thesis table, not re-checked.
  - herg (classification) lightgbm, rare half: 0.791 against 0.810 auc. qm9 lightgbm rare half: 0.590 against 0.576 rmse. From the thesis table, not re-checked.
  - Answer page c07.md (2026-10-09, not re-checked): within a half the hybrid usually beats the best single representation; on buchwald_hartwig it does not, for either model.
  - Same page: the two halves usually pick different best configurations; the hard half does not generally pick the larger budget. Own-half search against all-rows search goes both ways by dataset and model.
  - Trastuzumab-HER2, rare-scaffold half (1,500 rows), not in thesis. tz_her2_binary lightgbm: 0.5772 auc best single representation against 0.8402 hybrid, 4 folds. Checked against raw JSON.
  - tz_her2_binary xgboost, same half: 0.8179 against 0.8346 auc, 4 folds. tz_her2_multiclass svm: 0.4377 against 0.4861 weighted F1; xgboost 0.5157 against 0.4689; 3 folds. Checked.
- **In thesis:** Chapter chap:method, sec:method:results_ensembles, Table tab:method:scaffold_halves (no SOURCE comment; matched by the logd numbers above). Register row "Does the hybrid help on rare scaffolds as well as common ones?", 5 datasets. The other three partitions, the all-rows arm and the complexity measures are not in the thesis.
- **Related:** c11 and c18 (same subsection); c02 (hybrid against best single representation on whole datasets).

### c08 — Calibration of conformal intervals, hybrid against single representation  (`tests/experiments/complexity/c08_calibration.py`)
- **Question:** Are split-conformal 90% intervals from a model on the hybrid as well calibrated, and as narrow, as from a model on the best single representation? Does the budget change that?
- **How it works:**
  - Per outer fold, per decided model: the 2 best single representations and 2 best hybrids from the fold's training-row search.
  - Each hybrid is rebuilt at four settings. "optimal": its searched budget. "underfit": 25 features, or the next lower budget in the list. "overfit": the summed width of all representations.
  - "mismatched": the dataset's other decided model, fitted on the overfit-width hybrid (decided 2026-10-06).
  - The fold's training rows are split by scaffold group into a proper-training part and a calibration part (about 20%). The model is fitted on the proper-training part only.
  - Regression: interval half-width is the calibration residual quantile at level ceil((n+1)·0.9)/n. Classification: prediction sets.
  - Reported per fold: coverage (share of held-out rows inside the interval, target 0.900), mean interval width in label units, and the held-out score.
  - Features standardised on training rows (0/1 features left alone); held-out features clipped to the training range.
- **Status:** Ran on the server, 5 folds, on all 26 dataset files in 10-10b (24 plus tz_her2_binary and tz_her2_multiclass), all `ok`. 10-10b adds flab_binding_koenig and rewrites adme_rlm against 10-09.
  - FAULTS A8 (clipping, as in c07) applies. FAULTS A1 says dnn predictions are not clipped to the training-label range in any file; c08's dnn numbers are on that list.
- **Results so far:** Thesis Table tab:method:conformal reports only single (base1) against hybrid (hybrid1, "optimal"); means over 5 folds.
  - logd svm: coverage 0.905 single, 0.898 hybrid; width 1.652 against 1.403 logD units. Checked against raw JSON (0.9047, 1.6519; 0.8982, 1.4028).
  - Same row, other settings (raw JSON, checked, not in thesis): underfit coverage 0.897, width 2.026; overfit 0.902, 1.439; mismatched (dnn on the overfit hybrid) 0.900, 1.331.
  - abagym_Ang2 svm: coverage 0.804 single against 0.720 hybrid; width 0.599 against 0.492. ddg_combined svm: 0.884 against 0.828 coverage. From the thesis table, not re-checked.
  - qm9 dnn: 0.893 against 0.891 coverage, width 1.153 against 0.999. buchwald_hartwig lightgbm: 0.869 against 0.887, width 50.0 against 58.0 yield units. From the thesis table, not re-checked.
  - Answer page c08.md (not re-checked): the searched budget does not generally give the coverage closest to 0.90; the underfit or overfit build is usually closer.
  - Trastuzumab-HER2 (classification, prediction sets at 90%; not in thesis), 5 folds, checked against raw JSON. tz_her2_binary lightgbm: coverage 0.865 single against 0.884 hybrid; mean set size 1.081 against 1.152 classes.
  - Same row, held-out score: 0.8277 auc single against 0.7781 hybrid. tz_her2_multiclass svm: coverage 0.851 against 0.849; set size 1.713 against 1.588; weighted F1 0.6171 against 0.6650.
- **In thesis:** Additional file app:add:conformal, Table tab:method:conformal (in `thesis_rework/stage1_takeaways_bullets/additional_files_body.tex`, SOURCE comment names c08). Text in chap:method sec:method:results_why. Register row "Are the hybrid's prediction intervals calibrated?", 18 datasets. Cited again in chap:discussion limitations. The underfit, overfit and mismatched settings are not in the thesis.
- **Related:** c09 (training-to-held-out gap per budget); c33 (search optimism on small datasets, which the thesis uses to explain low coverage on antibodies).

### c09 — Variance shares, information kept, effective rank, train-to-held-out gap  (`tests/experiments/complexity/c09_formal_analysis.py`)
- **Question:** How much of the variation in search score comes from the downstream model, how much from the budget? Per budget, what do the chosen features keep, and how does the hybrid score held out?
- **How it works:** Four sub-analyses.
  - 9a: a two-way ANOVA on c02's result file, no new search. One observation is one configuration of c02's grid; its value is the mean over outer folds of its training-row search score, not a held-out score.
  - 9a reports eta squared, in percent, for model, budget, their interaction and the residual. The residual is variation among configurations sharing a (model, budget): importance method, allocation method, internal evaluator, representation count.
  - 9a also runs a pairwise ICC check for redundant models or budgets, and prints budget levels that build the same width.
  - 9b, 9c, 9d: per outer fold, per decided model, per decided budget (500, 1,000, 2,000 features on logd), the 2 best hybrids at that budget and the 2 best single representations, all rebuilt on training rows.
  - 9b: on held-out rows, a summed per-component mutual information between up to 50 PCA components of the chosen features and the label, minus a 3-permutation baseline.
  - 9b also: the share of the full concatenation's variance a ridge from the chosen features predicts. The code says this is not a mutual information.
  - 9c: effective rank of the chosen features, against a PCA and a random-feature set of the same width.
  - 9d: training score, held-out score and their gap per budget. The branch that reads c06 has never found the field it needs.
- **Status:** Ran on the server, 5 folds scored, 21 complete dataset files in 10-10b. flab_expression_koenig has only a partial file; flab_binding_koenig has none. Neither trastuzumab-HER2 set has a c09 file in either copy.
  - 10-10b adds adme_rlm and abagym_Ang2 against 10-09.
  - 9a here read c02 from the server folder `run_2026_10_06`, recorded in each file's `upstream` field.
  - FAULTS A6 (budget not filled) blocks reading the budget sweep. FAULTS A7: the PCA control stops at training rows minus 1, so above that width the hybrid and PCA differ in width as well as method.
  - In 9c, the hybrid and random features are standardised before the effective rank, while PCA components are only centred: a third difference (FAULTS A7).
- **Results so far:**
  - 9a, Table tab:method:variance_shares: logd model 48.5%, budget 1.4%, interaction 1.6%, residual 48.5%. caco2: model 0.0%, budget 6.6%, interaction 43.6%, residual 49.9%. qm9: model 84.1%, budget 1.0%. All three checked against raw JSON.
  - In the thesis table, budget explains more than model on five datasets: caco2, adme_mdr1, herg, flab_thermo_tresanco, humanness_therapeutic. herg: model 14.3%, budget 51.2%. From the thesis table, not re-checked.
  - 9d, qm9 dnn rank-1 hybrid held-out: 0.3142 rmse at 500 features, 0.3148 at 1,000, 0.3154 at 2,000. Checked. The answer page gives mordred alone at 0.3615 rmse, not re-checked.
  - Answer page c09.md (not re-checked): on flab_thermo_tresanco dnn, saprot alone 0.6927 spearman against 0.6244 for the hybrid at 50 features.
  - Same page: the hybrid's effective rank usually sits above random features of equal width and below PCA. qm9 at 500 features (checked): PCA 333.4, random 242.0.
- **In thesis:** chap:method sec:method:results_why, Table tab:method:variance_shares (SOURCE comment names c09 anova.eta2). Figure fig:best_number_of_features (chap:method): its solid score-against-budget lines are c09's rank-1 hybrid held-out scores (loader `tests/analysis/thesis_loader_f2.py`). No register row in tab:experiment_register names the variance shares. 9b and 9c are not in the thesis.
- **Related:** c02 (source of 9a's observations; per-fold budget picks in the same figure), c17 (recursive elimination and random lines in the same figure), c06, c16.

### c10 — Hybrid against single representation under damaged training labels  (`tests/experiments/noise/c10_noise_robustness.py`)
- **Question:** When training labels are damaged, does a model on the hybrid lose less held-out score than one on the best single representation? Does the search pick a different recipe on damaged labels? Do the selected features stay the same?
- **How it works:**
  - Part 1: per outer fold, the 8 configurations (2 models × 2 single representations, 2 hybrids) are chosen by a search on clean training labels.
  - Each is then refitted on damaged training labels and scored on the clean held-out labels.
  - Six damage kinds on regression: gaussian, outlier_p05, grouped_shifted, censoring, shift_only_p10, sparse_dprime_up_p10. Levels by dataset group, for example 0, 0.5, 1.0, 1.5 clean-label SDs on caco2.
  - Three draws per non-zero level; censoring one draw; level 0 once. Reported: damaged-minus-clean difference per level in the dataset's metric, and a normalised area where 4 levels exist.
  - Part 2: features selected by each hybrid recipe rebuilt on damaged labels, for the first damage kind, scored with the Nogueira stability index.
  - Part 3 (restored 2026-10-07): one search per (kind, level, draw) on damaged training labels, inner folds scored on clean labels. It records whether the top recipe moves.
  - Classification datasets are skipped (stored reason `regression_only`).
- **Status:** Ran on the server, incomplete. 17 dataset files in 10-10b; 3 skipped as classification (cdr_flexibility, herg, humanness_therapeutic). Neither trastuzumab-HER2 set has a c10 file in either copy; both are classification, which c10 skips.
  - 10-10b adds adme_mdr1 and rewrites abagym_EGFR, buchwald_hartwig, ginkgo_acsins, logd and qm9 against 10-09.
  - `ok` with 5 folds: abagym_EGFR, caco2, ddg_combined, ginkgo_colloidal, ginkgo_pr_cho, ginkgo_titer, herg_ki. `ok` with 3 folds: buchwald_hartwig, logd.
  - `partial`: abcov_kd (5 folds), adme_mdr1, flab_thermo_tresanco, qm9 (3), ginkgo_acsins (2).
  - caco2, ginkgo_colloidal, ginkgo_titer and herg_ki were written by the 2026-10-06 code, which has no Part 3 (FAULTS B8). caco2 also holds the older level grid.
  - THESIS_OUTLINE.md (10-09) lists c10 among noise experiments "with results that the chapter does not use".
- **Results so far:**
  - caco2 dnn, gaussian at 1.0 SD, 5 folds: best single representation 0.3093 rmse clean, +0.0453 damaged. Hybrid 0.2779 clean, +0.0552 damaged. So damaged 0.3546 against 0.3331 rmse. Checked.
  - qm9 dnn, gaussian 1.0, 2 folds: +0.1909 rmse single against +0.1598 hybrid. From answer page c10.md, not re-checked.
  - ginkgo_colloidal dnn, gaussian 1.0: −0.1512 spearman single against −0.0642 hybrid. From the answer page, not re-checked.
  - Answer page: on buchwald_hartwig the hybrid lost less with rf and more with lightgbm.
  - Answer page: where Part 3 ran (buchwald_hartwig, ddg_combined, flab_thermo_tresanco, ginkgo_pr_cho, logd), the top configuration changed in at least half of the damaged-label searches. Not re-checked.
- **In thesis:** Not in thesis. No SOURCE comment in thesis.tex names c10, and `tests/analysis/ch7_numbers.py` does not read it. chap:robustness covers single representations and models only, not hybrids. chap:noise compares hybrid with single representation through the n-experiments, not c10.
- **Related:** chap:noise experiments that compare hybrid and best single representation under the same damage kinds; n15 (named in the c10 file as asking a close question to Part 3); c33 and c24 (fold stability index).

### c11 — One model on the hybrid against ensembles of one model per representation  (`tests/experiments/complexity/c11_ensemble_vs_hybrid.py`)
- **Question:** Does one model on the hybrid match or beat an ensemble that fits one model per representation and pools their predictions (late fusion)? At what deployment cost?
- **How it works:**
  - Per outer fold, everything is chosen on that fold's training rows. Ensemble choices use an inner 3-fold group split of the training rows.
  - Hybrid arm: the 2 best hybrids of the fold's search per model, rebuilt on training rows. Single arm: the 2 best single representations of the same search.
  - Same-model ensemble (`ensemble_s1`): the row's model on every representation. Best-model ensemble (`ensemble_s2`): each representation uses its own better model, chosen on the inner split.
  - Members are the top 3 representations by inner score, or all of them. Seven pooling rules: mean, median, 20% trimmed mean, IQR-filtered, MAD-filtered, rank-weighted, top-k mean (majority vote for hard labels).
  - The `_auto` ensembles also pick members and pooling rule on the inner split; these are the reported comparators.
  - Also run: PCA of all representations to the budget width; best single representation picked on held-out folds; an oracle single pick per held-out fold (marked not fair).
  - Cost fields per arm: featurisers needed, models deployed, model input width, model bytes, fitting and prediction time, selection fits.
- **Status:** Ran on the server, 5 folds, 24 dataset files in 10-10b, all `ok`, counting tz_her2_multiclass. tz_her2_binary has no c11 file in either copy. 10-10b adds abagym_Ang2, adme_rlm and ginkgo_hic against 10-09.
  - FAULTS A1: dnn predictions are not clipped to the training-label range. An older adme_hlm file held a 62,177.96 rmse fold; adme_hlm has no c11 file in either copy.
  - FAULTS A6 (narrow hybrids) affects c11: on abagym_EGFR all 20 hybrids hold fewer features than their budget. FAULTS A7: PCA control capped at training rows minus 1.
- **Results so far:** Held-out means over 5 folds, hybrid / same-model ensemble / best single representation, from raw `head_to_head` (checked, 10-10b).
  - logd dnn 0.4086 / 0.4167 / 0.4491 rmse. logd svm 0.4305 / 0.4752 / 0.4928 rmse.
  - qm9 dnn 0.3174 / 0.3402 / 0.3617 rmse. herg_ki svm 0.5596 / 0.5819 / 0.5888 rmse. herg_ki dnn 0.5698 / 0.5536 / 0.6035 rmse.
  - caco2 dnn 0.2776 / 0.2713 / 0.3047 rmse. buchwald_hartwig rf 15.24 / 14.65 / 14.41 rmse (yield %).
  - herg lightgbm 0.8666 / 0.8719 / 0.8555 auc. flab_expression_koenig dnn 0.7348 / 0.7854 / 0.7156 spearman.
  - abagym_Ang2 svm 0.5123 / 0.6331 / 0.4524 spearman. abcov_kd lightgbm 0.1654 / 0.4186 / 0.2494 spearman. ddg_combined dnn 0.3459 / 0.1890 / 0.1311 spearman.
  - The best-model ensemble on logd is 0.4175 rmse, matching the thesis column.
  - The thesis table's AC-SINS, Titer and PR-CHO rows carry five numbers under four numeric headers, and do not match c11. Example: AC-SINS lightgbm thesis "50, 0.323, 0.521, 0.516" against c11 hybrid 0.5242, ensemble 0.5098, best-model 0.5174. Their source is not identified.
  - Cost fields and the PCA arm are in the files but not reported in the thesis.
  - tz_her2_multiclass, weighted F1, 5 folds (checked; not in thesis): svm 0.6863 / 0.6745 / 0.6648. xgboost 0.6579 / 0.7012 / 0.7164. Best-model ensemble 0.7008.
- **In thesis:** chap:method sec:method:results_ensembles, Table tab:method:ensembles (no SOURCE comment; matched by the logd and qm9 numbers). Register row "Does one model on the hybrid beat an ensemble of one model per representation?", 19 datasets. Mentioned in the abstract, Chapter 5 conclusions and chap:discussion. Not shown: abcov_kd, cdr_flexibility, ginkgo_hic, bh_doyle_plates, abagym_VEGF.
- **Related:** c02 (hybrid against single representation and PCA); c18 (top representation alone against an even split, same subsection); c07.


### c12 — Does joining two measurement types help?  (`tests/experiments/complexity/c12_modality_ablation.py`)
- **Question:** For a dataset whose representations fall into several measurement types ("modalities"), does putting the best representation of two modalities side by side beat the best single one?
- **How it works:**
  - Representations are grouped by modality from a fixed list in the docstring.
  - Each modality's representations are scored alone, per outer fold, under the dataset's decided models.
  - A combination is two whole representations concatenated at full width. There is no hybrid, no feature selection, no budget and no search.
  - Conditions: each modality alone, every pair of modalities, and all modalities together.
  - The representation chosen for each modality on fold f is the one with the best mean held-out score over the other folds (leave-one-fold-out re-pick). The same re-pick is applied to the single-representation baseline.
  - Features are scaled with `kirby.scaling.FeatureScaler`. Records stamp `RESULT_VERSION = 'c12_aligned_2026-10-06'`.
  - The baseline is the best single representation, re-picked the same way.
- **Status:** c12 is added by `run_all()` only for one excluded dataset family. No result on any dataset covered in this document; no c12 file exists for tz_her2_binary or tz_her2_multiclass.
- **Results so far:** none.
- **In thesis:** not in thesis.
- **Related:** c18 (how many representations to draw on), c13 (adding one representation to a pool).

### c13 — Does the hybrid search take in a new embedding?  (`tests/experiments/complexity/c13_forward_compatibility.py`, helper `forward_compat_reps.py`)
- **Question:** When one learned embedding is added to a dataset's pool, does the hybrid built with it score better than without it, and better than random features in its place?
- **How it works:**
  - The "subjects" are the embeddings tested one at a time. `classify_rep` marks a pool member as a learned embedding by name; fingerprints, descriptors, one-hot and similar are "established".
  - Conditions per subject, per outer fold, per decided model, all scored on held-out rows:
    - B: the subject alone.
    - C: hybrid searched over the pool without the subject.
    - D: hybrid searched over the pool with the subject.
    - D_random: D's own hybrid, with the subject's selected features replaced by the same number of Gaussian random features. No new search is run.
    - C_established, D_established, D_random_established: the same three with only established representations in the pool.
    - Controls: a Gaussian matrix of the subject's width scored alone, and the subject with its rows shuffled against the labels, scored alone.
  - Each hybrid is the first or second best of the fold's search (`fold_standard_selection`: per model, the 2 best single representations and the 2 best hybrids).
  - Reported deltas, sign-normalised so positive means better: D − C, D − D_random, D − B, and the same on the established side. No significance tests.
  - A caveat in the docstring (author, 2026-10-07): D_random is a weaker control than a searched random matrix. An earlier searched control gave the Gaussian 0 features on every fold of herg_ki and buchwald_hartwig. The re-run with a searched control, about 110 job-hours, was not made.
  - `forward_compat_reps.py` lists published embedding models not in any pool, compiled 2026-09-10 from a literature search. Each entry carries paper, weights URL and a quoted published number; none is a measured KIRBy result. Five admission rules apply: learned embedding, 2023 or later, public weights, a published comparison, and one fixed-length vector per input. Blocks: molecular 10 models (e.g. CheMeleon, Mol-JEPA, Uni-Mol2, ChemFM), antibody 9, protein 6, nucleotide 9, reaction 5, plus others. A cut list records each excluded model and the rule it failed.
  - Since 2026-10-07 the default loads none of these candidates (`_c13_candidate_policy`, "decided 2026-10-07"). Files started earlier do include some: herg_ki holds chemfm and unimol_v2; several antibody files hold ism, igbert, mint, p_iggen, currab and abcdr_esmc. Many candidates never loaded on the server for lack of a builder or weights (FAULTS.md A21).
- **Status:** Ran on the server. Final files with 5 outer folds: abagym_EGFR, abcov_kd, buchwald_hartwig, caco2, cdr_flexibility, ddg_combined, ginkgo_colloidal, ginkgo_pr_cho, ginkgo_titer, herg_ki, logd. Final files with 3 folds: adme_mdr1, herg, humanness_therapeutic. tz_her2_binary (trastuzumab–HER2 antibody binding, AUROC) has a final file with 5 folds, written 2026-10-08; the dataset was cut from the run that day (ledger "Cut, not shown"; c13 answer page). It started after commit 47654b6, so it tests only its own pool's embeddings. No file for tz_her2_multiclass. logd and herg ran on a 2,000-row subsample (`subsampled_rows` 2000, checked). Partial files only: abagym_VEGF, flab_expression_koenig, flab_thermo_tresanco, ginkgo_acsins. No file: qm9, adme_hlm, adme_rlm, abagym_Ang2, ginkgo_hic, flab_binding_koenig. The 10-10b copy adds adme_mdr1 over 10-09. Known faults, from FAULTS.md:
  - A22: the code compares D with C, where D's search may rank a different configuration first and take 0 features from the subject. It should differ from C only by the subject. Example: ginkgo_titer dnn fold 1, C 1,000 features from 10 representations scores 0.4158 Spearman; D 500 features from 3 scores 0.2447.
  - A22: on cdr_flexibility, ginkgo_acsins, ginkgo_hic and abagym_Ang2 the established side is one representation scored alone, compared with a searched hybrid.
  - B7: the gp Gaussian and shuffled controls have no score on gp datasets.
- **Results so far:**
  - herg_ki, svm, 5 folds, RMSE, hybrid 1: with chembert D 0.5596 against D_random 0.5743; C 0.5609 (checked). With unimol_v2 D 0.5592 against D_random 0.5596 (checked).
  - logd, dnn, 2,000 molecules, 5 folds: with Uni-Mol2 D 0.4908 against D_random 0.4907 RMSE (thesis table, not re-checked). With grover C 0.4980 to D 0.4902 RMSE (answer page, not re-checked).
  - Thesis text: on hERG K_i, Caco-2, MDR1–MDCK efflux and LogD the embedding's features beat random features for every embedding or all but one. On EGFR binding and ΔΔG they win for most embeddings. On several other antibody datasets and Buchwald–Hartwig they win for fewer than half.
  - tz_her2_binary, 5 folds, AUROC, hybrid 1 (checked): with ablang2, xgboost D 0.8127 against D_random 0.7844, C 0.8170; lightgbm D 0.8128 against D_random 0.7763, C 0.7531. D scores the same for every subject (0.8127 xgboost, 0.8128 lightgbm). That is by design: for a subject already in the pool, D is the whole pool, one shared search.
  - Where D's hybrid took 0 features from the subject, D and D_random are equal by construction.
- **In thesis:** Chapter `chap:method`, one paragraph in `sec:method:results_why` (thesis.tex line ~2642). Full table `tab:method:new_embeddings` in Additional file `app:add:new_embeddings` (additional_files_body.tex line ~2352). The table compares D against D_random only, hybrid 1, 12 datasets. abcov_kd, cdr_flexibility and tz_her2_binary have final files but no rows. tz_her2_binary was cut on 2026-10-08; for the other two the reason was not checked. Register row "Does the hybrid take in a new embedding?" in `tab:experiment_register`, 12 datasets.
- **Related:** c18 (spreading a budget over representations), c17 and c15 (hybrid against random features of the same width), c12.

### c14 — Hybrid against best single representation, across domains  (`tests/experiments/complexity/c14_cross_domain_summary.py`)
- **Question:** On each dataset, does the best hybrid beat the best single representation on held-out rows, shown side by side across domains?
- **How it works:**
  - One run takes several datasets (`--tasks`). Each uses its general representation pool and its two decided models, with no augmentation and no distance search.
  - Per outer fold: every representation and model is fitted on training rows and scored on held-out rows. Then `fold_standard_selection` picks, per model, the 2 best single representations and the 2 best hybrids on training rows.
  - Reported per model: the per-fold difference of hybrid 1 (and hybrid 2) against single representation 1, and its mean over folds.
  - Across datasets: one row per dataset and model, plus a count of rows where the hybrid is ahead. No cross-dataset average and no significance test.
  - A failed per-fold search stops the run with a non-zero exit (decided 2026-10-06).
  - Four helper functions in the file are still imported by c33 and two OPIG scripts; c14 itself no longer calls them.
- **Status:** Never ran in run_2026_10_06. It is a cross-dataset job (`cross:c14` in `tests/experiments/orchestrate/plan.py`, launched by `tests/slurm_scripts/runbook/cross_one.sh`). Its planned datasets in this document's scope are logd, caco2, herg and flab_thermo_tresanco. The ledger lists it "pending" on 10-09 and "skipped" on 10-10b. No result file exists in either copy. Neither tz_her2 dataset is on its planned list. c14 is not in `run_all()`'s module map.
- **Results so far:** none.
- **In thesis:** not in thesis.
- **Related:** c02 (hybrid against best single representation, used for `tab:hybrid_vs_best_single`), c33 (imports c14's helpers).

### c15 — Distances, kernels, double descent and memorisation in hybrid spaces  (`tests/experiments/complexity/c15_ml_theory_depth.py`)
- **Question:** Do compact hybrids avoid the problems of wide feature sets: distances that all look alike, ill-conditioned kernels, a score dip near as many features as molecules, and fitting random labels?
- **How it works:** Four parts, all per outer fold, regression datasets only. Comparators are all general representations concatenated ("all features"), PCA of all features, and random subsets of all features. PCA and random are matched to each hybrid's built width (decided 2026-10-06).
  - 15a: on held-out rows, for each single representation, each model's 2 hybrids at each decided budget, and width-matched PCA. Measures each molecule's farthest/nearest distance ratio (Beyer et al. 1999), relative contrast (Aggarwal et al. 2001), distance coefficient of variation, and hubness, the skew of how often each molecule is among others' 5 nearest neighbours (Radovanovic et al. 2010).
  - 15b: RBF kernel on held-out rows (capped row count, seed 42): log10 condition number, effective rank, and kernel–label alignment. Same spaces as 15a.
  - 15c: a fresh hybrid search per fold at budgets of 0.25, 0.8, 0.9, 1.0, 1.1, 1.2 and 4.0 times the dataset's row count, capped at the total feature count. Each hybrid is scored beside width-matched PCA and random features.
  - 15d: each model on all features, on 2 hybrids per decided budget, and on width-matched PCA, scored on held-out rows. Memorisation is the training-row R² of a fit to permuted labels, floored at 0, mean over 5 permutations.
  - Models: the dataset's decided models, excluding knn; svm is dropped above 10,000 training rows.
- **Status:** Ran on the server. Files with results in 10-10b: abagym_Ang2, abagym_EGFR, abagym_VEGF, abcov_kd, adme_mdr1, buchwald_hartwig, caco2, ddg_combined, flab_thermo_tresanco, ginkgo_acsins, ginkgo_colloidal, ginkgo_hic, ginkgo_pr_cho, ginkgo_titer, herg_ki, logd, qm9. The 10-10b copy adds abagym_Ang2, adme_mdr1 and ginkgo_hic over 10-09. cdr_flexibility, herg, humanness_therapeutic and tz_her2_binary hold `skipped: regression_only` (checked). No file for tz_her2_multiclass. qm9 holds 3 of 5 folds (checked); logd holds 5. Known faults, from FAULTS.md:
  - A6: a budget names a number of features, but the build stops at the last feature with importance above zero. So some hybrids hold fewer features than the budget, and two budgets can hold the same hybrid.
  - A7: the PCA control should match the hybrid's width, but it stops at training molecules minus 1. On small datasets its score repeats across budgets. Example (checked): ginkgo_titer dnn PCA gives 0.0133 Spearman at every budget from 250 to 2,000 features.
  - A7: PCA scores NaN Spearman for svm on abagym_Ang2, abagym_VEGF, ddg_combined, flab_thermo_tresanco and ginkgo_colloidal, and for gp on gp datasets. Cause not verified.
  - C4: identical scores at different widths, e.g. buchwald_hartwig lightgbm 16.7430 RMSE at five budgets.
- **Results so far:** (15d, hybrid 1, held-out means, checked against raw files)
  - qm9, 3 folds, RMSE: dnn all features 0.3513, hybrid at 500 features 0.3179, PCA at 500 0.3699. lightgbm all features 0.3636, hybrid at 500 0.3478, PCA 0.5575.
  - logd, 5 folds, RMSE: dnn all features 0.4180, hybrid at 2,000 features 0.4086. svm all features 0.4437, hybrid at 1,000 0.4299, PCA at 1,000 0.6385.
  - herg_ki, 5 folds, RMSE: svm all features 0.5828, hybrid at 1,000 0.5576. dnn all features 0.5602, hybrid at 250 0.5712, at 1,000 0.6317.
  - ginkgo_titer, 5 folds, Spearman: dnn all features 0.2145, hybrid at 1,000 0.3382. rf all features 0.3084, hybrid at 500 0.3444, at 2,000 0.2284.
  - 15c, logd, 5 folds, budget 1,257 features, RMSE: dnn hybrid 0.4088, PCA 0.4767, random 0.4264 (checked).
  - Answer page (not re-checked): in 15a every hybrid-bearing dataset had a higher farthest/nearest ratio than width-matched PCA. In 15b hybrid kernels were mostly not better conditioned than every single representation's kernel.
- **In thesis:** not in thesis. THESIS_OUTLINE.md (2026-10-09) records that c15 appears only as a file pointer. It also records that c15's hybrid-beats-PCA result disagrees with c02 and c11, where PCA beats the hybrid with the dnn on LogD and Caco-2.
- **Related:** c02, c11 and c17 (other PCA and random-feature comparisons), c09 (budget curves and effective rank), c16 and c06 (budget against model capacity).

### c16 — Does a smaller hybrid do the job of the model's regularisation setting?  (`tests/experiments/complexity/c16_feature_selection_as_regularization.py`)
- **Question:** Does a hybrid with fewer features at the model's default setting score as well as a wide hybrid with the regularisation setting tuned?
- **How it works:**
  - Only two models have a sweep: svm (C at 0.01, 0.1, 1, 10, 100) and rf (maximum depth 3, 5, 10, no limit). Defaults are the search's own: C = 1 and no depth limit.
  - A dataset runs only if one of its two decided models is svm or rf. The other decided model is not swept.
  - Per outer fold, one search on training rows. At each decided budget, the model's 2 best hybrids are rebuilt and scored at every setting.
  - "capped_full" is a wide hybrid of min(all features, 2 × rows, 2,048) features. Its recipe is the fold's best hybrid over all budgets, rebuilt at that width.
  - The search's 2 best single representations are also fitted at full width under every setting.
  - Outputs: a held-out score grid (setting × budget); the spread across settings at each budget; and the best small-budget hybrid at the default setting against capped_full at its best setting.
  - A re-score (`fold_reselection`) picks the budget and capped_full's setting on the other folds for each held-out fold.
- **Status:** Ran on the server, 5 folds. Files with results in 10-10b: abagym_Ang2, abagym_EGFR, abagym_VEGF, adme_hlm, adme_mdr1, buchwald_hartwig, caco2, ddg_combined, flab_expression_koenig, flab_thermo_tresanco, ginkgo_colloidal, ginkgo_titer, herg_ki, humanness_therapeutic, logd. Skipped because neither decided model is svm or rf (checked for qm9 and abcov_kd): abcov_kd, cdr_flexibility, ginkgo_acsins, ginkgo_hic, ginkgo_pr_cho, herg, qm9. adme_rlm is partial. No file for tz_her2_binary or tz_her2_multiclass; FAULTS.md B2 lists tz_her2_binary as pending, never started, and both were cut on 2026-10-08. The 10-10b copy adds abagym_Ang2 and adme_rlm (partial) over 10-09. On ginkgo_colloidal capped_full is 484 features, the smallest budget, so no smaller hybrid exists to compare (checked). Known faults:
  - FAULTS.md C4 and THESIS_OUTLINE.md 5.10: svm at C = 10 and C = 100 gives identical held-out scores. Checked: on adme_hlm and adme_mdr1 the two settings match at every budget. On logd, caco2 and herg_ki they differ. Cause not checked.
  - FAULTS.md A6 applies: some hybrids are narrower than their budget.
- **Results so far:** (checked against raw files, 5 folds)
  - logd, svm, RMSE: capped_full at C = 10 scores 0.4114; hybrid at 1,000 features at C = 1 scores 0.4299.
  - adme_hlm, svm, RMSE: capped_full at C = 10 scores 0.4176; re-scored smaller hybrid at C = 1 scores 0.4250.
  - ddg_combined, svm, Spearman: re-scored smaller hybrid 0.2908 against tuned capped_full 0.2653. Per-fold differences run from −0.0706 to +0.1957.
  - buchwald_hartwig, rf, RMSE: re-scored hybrid at 1,000 features 14.9673 against capped_full 15.0444.
  - Answer page (not re-checked): the smaller hybrid did not win on the small-molecule datasets. It did win on abagym_VEGF, ddg_combined, flab_thermo_tresanco and ginkgo_titer.
- **In thesis:** not in thesis. `tests/analysis/thesis_loader_f2.py` can read c16 for the best-number-of-features figure, but the figure is drawn with the default source c09. The register row "Does a smaller hybrid replace tuning the model?" cites `app:add:budget_by_capacity`, which the brief maps to c06. THESIS_OUTLINE.md lists a c16 bullet (author agreed 2026-10-09) with the C = 10/100 caveat; no matching text was found in thesis.tex.
- **Related:** c06 (budget picked at each model capacity setting), c09 (budget curves), c15.


### c17 — Hybrid against PCA, mRMR, recursive elimination and random features at the same number of features  (`tests/experiments/complexity/c17_information_theoretic_decomposition.py`)
- **Question:** At a fixed number of features, does the hybrid's selection beat other standard ways of reducing all representations joined to that number?
- **How it works:**
  - Regression only. Classification datasets return a skip file (`reason: regression_only`), because the recursive-elimination step hard-codes a random-forest regressor.
  - All of the dataset's general representations are joined into one matrix (logd: 15,673 features from 10 representations).
  - The number of features (the budget) runs over the dataset's decided general budgets that are below the joined width (logd: 500, 1,000 and 2,000 features).
  - Six sets per fold and budget. `hybrid` and `hybrid__general2` are the best and second-best hybrids of that fold's search at that budget, per model.
  - `pca`: principal components of the joined, scaled matrix, at most training rows minus 1 components.
  - `mrmr`: greedy minimum-redundancy maximum-relevance ranking with a k-nearest-neighbour mutual-information estimator (k=5), ranked on at most 2,000 training rows. It is capped at 200 features (`MRMR_MAX_BUDGET`), so it has no score at any budget above 200.
  - `rfe`: scikit-learn recursive feature elimination with a 50-tree random forest of depth 5, removing 10% of features per step. `random`: budget features drawn with seed 42.
  - Each set is fitted with each of the two decided models and scored on held-out rows. The file also stores mutual information with the label and effective rank of each set, both read on held-out rows.
  - Code change after the runs: since commit 6796aa6 (2026-10-09 15:05) the PCA arm fits the components as projected. Every stored file except abagym_Ang2 was written before that, so their `pca` arm rescaled each component to unit variance. `RESULT_CAVEATS.md` records the author's decision not to re-run and to leave c17's PCA out of the thesis.
- **Status:** ran on server, 5 folds. Scores on 17 regression datasets in the 10-10b copy: abagym_Ang2, abagym_EGFR, abagym_VEGF, abcov_kd, adme_mdr1, buchwald_hartwig, caco2, ddg_combined, flab_expression_koenig, flab_thermo_tresanco, ginkgo_acsins, ginkgo_colloidal, ginkgo_pr_cho, ginkgo_titer, herg_ki, logd, qm9. The 10-09 copy lacks abagym_Ang2 and abagym_VEGF. Skip files for cdr_flexibility, herg and humanness_therapeutic. No file for adme_rlm, flab_binding_koenig or ginkgo_hic. Known faults:
  - mRMR has no score on any dataset whose decided budgets all exceed 200 features. That covers every small-molecule dataset and buchwald_hartwig. It scored only on abagym_Ang2, abagym_VEGF, abcov_kd, ddg_combined, flab_thermo_tresanco and ginkgo_acsins.
  - PCA: the arm scores NaN Spearman on several antibody datasets (abagym_VEGF, ddg_combined, flab_thermo_tresanco, ginkgo_acsins, ginkgo_pr_cho with gp or svm), seen in the raw files. The cause is not verified.
  - FAULTS.md A6. The hybrid stops at its last feature with importance above zero, so it can hold fewer features than the budget. The other four sets fill the budget. A hybrid-versus-other comparison can therefore differ in width as well as in how features were chosen.
  - FAULTS.md A22 states that c17 computes mutual information on all molecules. The stored files say otherwise: `measured_on: held-out rows` and `mi_unit: nats, held-out rows, mean over folds` (checked in logd.json).
- **Results so far:** fold means, 5 folds, checked in the raw 10-10b files.
  - qm9, RMSE, dnn at 500 features: hybrid 0.3148, recursive elimination 0.3396, random 0.3747. Thesis `tab:method:feature_reduction` prints 0.315, 0.340 and 0.375 (matches).
  - logd, RMSE, dnn at 500: hybrid 0.4229, recursive elimination 0.4299, random 0.4584 (matches the thesis row). svm at 2,000: hybrid 0.4316, recursive elimination 0.4443.
  - caco2, RMSE, dnn at 1,000: hybrid 0.2756, recursive elimination 0.2901, random 0.2872.
  - herg_ki, RMSE, dnn at 2,000: hybrid 0.6200, recursive elimination 0.5960, random 0.5813. svm at 2,000: hybrid 0.5647, recursive elimination 0.5719.
  - buchwald_hartwig, RMSE, lightgbm at 2,000: hybrid 16.34, recursive elimination 15.33, random 14.87.
  - ginkgo_pr_cho, Spearman, gp at 2,000: hybrid 0.4601, recursive elimination 0.5355.
  - abagym_VEGF, Spearman, svm at 10 features: hybrid 0.4055, mRMR 0.3572, recursive elimination 0.3028.
  - ddg_combined, Spearman, svm at 100: hybrid 0.2388, mRMR 0.0794, recursive elimination 0.3603.
  - Thesis text (line ~2484): hybrid has the lower error at every budget on QM9, Caco-2, LogD and MDR1; on hERG Ki with the DNN random features beat it at 500, 1,000 and 2,000.
- **In thesis:** Chapter `chap:method`, `sec:method:results_why` (text around line 2484), figures `fig:best_number_of_features` and `fig:best_number_of_features_antibodies` (loader `tests/analysis/thesis_loader_f2.py`), table `tab:method:feature_reduction` in Additional file `app:add:feature_reduction`. Method paragraph "Feature reduction at the same width" in `sec:method:comparators`. Register row "Does the gain come from several representations or from fewer features?", listed as 15 regression datasets. The PCA arm is not reported.
- **Related:** c02 and c11 (the thesis's PCA comparison), c09, c16, c18.

### c18 — Many representations or few at the same budget  (`tests/experiments/complexity/c18_diversity_vs_depth.py`)
- **Question:** At a fixed number of features, is it better to take them from the one top representation, from the top two or three, split evenly, or split the way the search split them?
- **How it works:**
  - Per outer fold, per decided model, per decided budget. One search per fold on training rows supplies the hybrids and the importance method.
  - `natural`, `natural__general2`: the search's best and second-best hybrids at that budget, rebuilt on the fold's training rows.
  - Representations are ranked on training rows by summed importance. The importance method is the one the rank-1 hybrid used; a fallback is used if the search gave no hybrid.
  - `depth_1`, `depth_2`, `depth_3`: the budget filled with the top-importance features of the top 1, 2 or 3 representations.
  - `uniform_all`: an even split over the same representations the `natural` hybrid used, to the same width as `natural`.
  - `uniform_every_rep` (added 2026-10-07): an even split over every general representation, filled to the budget.
  - The file also records the fold's two best single representations at full width (`base1`, `base2`).
  - Each set is fitted with the decided model and scored on held-out rows.
- **Status:** ran on server, 5 folds, all files carry the format `per_fold_uniform_every_rep_added_2026_10_07`. 10-09 copy has final files for 21 datasets. It has partial files only for flab_binding_koenig, flab_expression_koenig and nt_promoter_tata. The 10-10b copy adds final files for flab_binding_koenig and flab_expression_koenig. bh_doyle_plates has a file, but the 10-09 ledger lists it as cut. Known faults:
  - FAULTS.md A22, read in the code. `depth_*` take the top budget features by argsort, which pads with zero-importance features. `natural` stops at its last feature with importance above zero. So `depth_*` and `natural` can differ in width as well as in split.
  - `natural__general2` can use a different importance method from the hand-built sets, which follow the rank-1 hybrid. So the second hybrid differs from the hand-built sets in two ways at once, not one.
  - FAULTS.md B8 says c18 files predate `uniform_every_rep`. That was true of the copy FAULTS.md read (8 Oct 12:12). The 10-09 and 10-10b files are later re-runs and contain the arm (checked).
- **Results so far:** fold means, 5 folds, checked in the raw files.
  - qm9, RMSE, dnn at 500 features: top representation alone 0.3566, search's hybrid 0.3140, even split over the same representations 0.3276. Thesis `tab:add:top_rep_even_split_all` prints 0.357, 0.314 and 0.328 (matches).
  - herg_ki, RMSE, svm at 1,000: top one 0.6028, top three 0.5673, hybrid 0.5549, even split 0.5568. Answer page gives `base1` at 0.5881 (not re-checked).
  - logd, RMSE, svm at 500: top one 0.4991, top three 0.4463, hybrid 0.4384, even split 0.4402.
  - abagym_EGFR, Spearman, svm at 250: top one 0.2538, hybrid 0.3205, even split 0.3570.
  - ddg_combined, Spearman, svm at 100: top one 0.1252, hybrid 0.2402, even split 0.3579, even split over every representation 0.3680.
  - humanness_therapeutic, AUROC, xgboost at 10: top one 0.8939, top three 0.8461, hybrid 0.9412.
  - Thesis text (line ~2251): the top representation alone scores worse than the hybrid at every budget on most datasets. On small molecules an even split comes close to the search's split. On antibodies the split matters more.
- **In thesis:** Chapter `chap:method`, `sec:method:results_ensembles`, main-text table `tab:method:top_rep_even_split` (smallest budget only) and Additional file `app:add:top_rep_even_split` (`tab:add:top_rep_even_split_all`, every budget). Only `depth_1`, `natural` and `uniform_all` are reported. Register row "At the same budget, is it better to draw on many representations or few?", 20 datasets.
- **Related:** c11 (ensembles, same subsection), c16 and c17 (budget), c21 (hybrid against single representations).

### c19 — Importance scores reused during training  (`tests/experiments/complexity/c19_importance_weighted_training.py`)
- **Question:** Once the hybrid is built, does reusing its importance scores to weight features or molecules during training improve the held-out score?
- **How it works:**
  - Per outer fold, per decided model: the fold search's two best hybrids and two best single representations (`hybrid`, `hybrid__general2`, `base1`, `base2`), all ranked on the fold's training rows.
  - Budget: 100 features where 100 is a decided budget. Otherwise each hybrid is the best over every decided budget (logd: 2,000-feature hybrids).
  - Inputs are scaled. Sets wider than the training rows are replaced by principal components first (`cap_dimensions`).
  - Importance is recomputed on the training rows with the rank-1 hybrid's importance method. Weights are the square root of importance, scaled to average 1.
  - Five conditions: `unweighted`; `feature_weighted` (features times weight); `sample_weighted`; `combined` (both); `importance_pruned` (only features strictly above the median importance).
  - Sample weight of molecule i: 1 − |y_i − mean label of its 10 nearest neighbours| / label range, clipped to 0.1–1. Neighbours are by Manhattan distance in the weighted feature space.
  - The gp takes no sample weights, so `sample_weighted` and `combined` are recorded as not run for gp.
- **Status:** ran on server, 5 folds, 24 datasets in both copies (files identical): abagym_Ang2, abagym_EGFR, abagym_VEGF, abcov_kd, adme_mdr1, adme_rlm, bh_doyle_plates, buchwald_hartwig, caco2, cdr_flexibility, ddg_combined, flab_binding_koenig, flab_expression_koenig, flab_thermo_tresanco, ginkgo_acsins, ginkgo_colloidal, ginkgo_hic, ginkgo_pr_cho, ginkgo_titer, herg, herg_ki, humanness_therapeutic, logd, qm9. Known faults:
  - FAULTS.md A22: on ginkgo_hic, ginkgo_pr_cho and ginkgo_titer every hybrid is wider than the training rows, so it is replaced by principal components before weighting (from FAULTS.md, not re-checked).
  - FAULTS.md lists RMSE in the thousands on adme_rlm dnn. The stored adme_rlm file (written 10-09 04:59) has dnn RMSE 0.4992 unweighted, so the blow-up is not in this file (checked).
  - FAULTS.md B8: flab_binding_koenig ran across two decided-grid commits (not re-checked).
  - For tree models, `feature_weighted` differs from `unweighted` only where a feature of importance 0 gets weight 0 (stated in the code).
- **Results so far:** rank-1 hybrid, fold means, 5 folds, checked in the raw files.
  - qm9, RMSE, dnn: unweighted 0.3163, sample-weighted 0.3134, above-median 0.3208. Thesis `tab:method:importance_training` prints 0.316, 0.313, 0.321 (matches).
  - logd, RMSE, svm: unweighted 0.4305, feature-weighted 0.4617, combined 0.4704.
  - caco2, RMSE, dnn: unweighted 0.2799, combined 0.2747.
  - adme_mdr1, RMSE, svm: unweighted 0.4372; every weighted condition 0.4405 to 0.4453.
  - ginkgo_colloidal, Spearman, dnn: unweighted 0.2607, feature-weighted 0.3571. svm: unweighted 0.3538, feature-weighted 0.3136.
  - ginkgo_pr_cho, Spearman, dnn: unweighted 0.4699, above-median 0.3868.
  - ginkgo_titer, Spearman, rf: unweighted 0.1491, above-median 0.2247.
  - The single-representation rows (`base1`, `base2`) are stored but not in the thesis table.
- **In thesis:** Chapter `chap:method`, `sec:method:results_why`, subsubsection "Using importance during training", table `tab:method:importance_training`. Method paragraph "Importance during training" in `sec:method:comparators`. Register row "Does importance help during training?", 20 datasets.
- **Related:** n12 (importance-weighted distance, noise suite), s02 (weighting in the similarity suite), c22 (importance methods).

### c20 — How many of its own features each representation needs  (`tests/experiments/complexity/c20_signal_saturation.py`)
- **Question:** For each single representation alone, what is the smallest number of its own top features that scores as well as all of them?
- **How it works:**
  - No hybrid at any point. One representation at a time.
  - Folds: 5 group-aware folds from `_alloc_harness.make_folds` (the repo's `cv_split`), not `nested_cv_outer_folds`. So they are not the folds of the other experiments in this part.
  - On each fold's training rows the representation's features are ranked by `lightgbm_gain`. The model is scored with the top 10, 25, 50, 100, 200, 300, 500, 1,000 features and with all of them.
  - Downstream models: lightgbm and rf, both fixed in the file, not the decided models.
  - `plateau_n_features` is the smallest feature count whose 5-fold mean is within one tolerance of the full-width mean. The tolerance is the mean, over the dataset's representations, of each representation's standard deviation across its 5 full-width fold scores.
  - Standalone script: `--dataset` is required, results go to `--out/c20_signal_saturation.json`. It scores hard predictions, which would be wrong on an AUROC dataset (noted in `EXPERIMENT_COMPUTE_AUDIT_2026_09_18.md`, not re-checked).
- **Status:** never ran on the server. It is not in `run_all()`'s module map, and neither results copy has a c20 folder. An earlier version ran locally in July 2026 on herg_ki, logd, buchwald_hartwig and flab_thermo_tresanco. That run used 8 representations per dataset and a fixed tolerance. Its numbers are in `HYBRID_ALLOCATION_FINDINGS_PLAIN_2026_07_19.md`. No result file from it was found in `tests/results`. The tolerance and folds have changed since (commits up to df7c7fe, 2026-09-12). A local debug run of the current code, 29 Sep to 2 Oct 2026, is in `/Volumes/seagate/dirty_local_2026_09_28/results/c20/<dataset>/c20_signal_saturation.json`. It covers herg_ki, buchwald_hartwig, flab_expression_koenig and humanness_therapeutic. It used 3 folds and 5 or 6 representations per dataset, cut by `KIRBY_LOCAL_REPS`. These are local debug runs, not results; their numbers are not reported here.
- **Results so far:** none from current code at full settings. From the July document (dated claim, not re-checked, older code): on herg_ki, ecfp4 levelled off at 200 features under lightgbm and 100 under rf. Three buchwald_hartwig representations gave flat curves with no signal.
- **In thesis:** not in thesis (no SOURCE comment and no matching text found).
- **Related:** c05 (sweeps training rows), c16 (sweeps the hybrid's budget), c30–c32 (share `_alloc_harness`).

### c21 — Fixed-recipe hybrid against each single representation, and extra features on one representation  (`tests/experiments/complexity/c21_single_rep_augmentation.py`)
- **Question:** Does a hybrid built from a fixed recipe, with no search, beat each single representation? On one strong representation, do a few added features help more than they do on the hybrid?
- **How it works:**
  - Per outer fold (`nested_cv_outer_folds`, seed 42), per decided model, per general representation (10 per dataset).
  - `A_baseline`: the representation alone at full width, fitted with 3 model seeds averaged per fold.
  - `H0_hybrid`: a hybrid with a fixed recipe, not searched. Each part is taken from the decided grid (logd: `performance_weighted` allocation, `lightgbm_gain` importance, lightgbm internal evaluator, every representation). It is built at budgets of 500, 1,000 and 2,000 features where the grid keeps them, else at the grid's budgets.
  - Extra-feature conditions on one representation: `B_aug` adds the dataset's augmentation feature sets (antibodies: paratope, 8 features; structural, 28 features). `C_topreps_n9` adds up to 9 top features from each other representation. `D_both_n9` adds both.
  - Controls of the same width: `R_rand_n9` offers 9 random features per other representation, against C. `R_randaug` offers as many random features as the augmentation sets hold, against B.
  - Two decoy blocks are planted among the offered features: pure noise, and noisy copies of base features. The counts admitted are stored.
  - `H1_hybrid_aug` (hybrid plus extra features) is defined but off on every dataset. The stored reason is "until the owner decides".
- **Status:** ran on server, 5 folds, all files `c21_per_fold_2026-10-05`. 22 datasets in the 10-09 copy; the 10-10b copy adds adme_rlm. Datasets: abagym_Ang2, abagym_EGFR, abagym_VEGF, abcov_kd, adme_mdr1, adme_rlm, bh_doyle_plates, buchwald_hartwig, caco2, cdr_flexibility, ddg_combined, flab_expression_koenig, flab_thermo_tresanco, ginkgo_acsins, ginkgo_colloidal, ginkgo_hic, ginkgo_pr_cho, ginkgo_titer, herg, herg_ki, humanness_therapeutic, logd, qm9. Extra-feature conditions ran only on ginkgo_colloidal and ginkgo_hic, `uncapped` only. Every other file stores the reason "DECISIONS.md Step 8 cut augmentation", so only `A_baseline` and `H0` ran there. Known faults:
  - FAULTS.md A22: every extra-feature condition first drops base features of importance 0. ginkgo_hic keeps 130 to 149 of esm2_concat's 1,280 (from FAULTS.md, not re-checked). So each extra-feature condition differs from `A_baseline` in two ways at once: features dropped and features added.
  - FAULTS.md A6: the `H0` hybrid can hold fewer features than its budget.
- **Results so far:** fold means, 5 folds, checked in the raw files. "Best single" below is the representation with the best held-out mean, picked by me after the fact, not by the experiment.
  - qm9, RMSE, dnn: best single mordred 0.3603, H0 at 2,000 features 0.3176 (matches the answer page).
  - logd, RMSE, svm: best single mhg_gnn 0.4928, H0 at 1,000 0.4313.
  - caco2, RMSE, dnn: best single pdv 0.2984, H0 at 2,000 0.2758.
  - abcov_kd, Spearman, lightgbm: best single igt5 0.4032, H0 at 2,000 0.3328.
  - abagym_Ang2, Spearman, gp: best single saprot 0.6008, H0 at 100 0.4967.
  - bh_doyle_plates, RMSE, rf: best single diff_avalon 15.61, H0 at 500 17.23.
  - ginkgo_colloidal, Spearman, svm, base prot_t5: alone 0.3927. With augmentation sets 0.3508; with random features of the same width 0.3500.
  - ginkgo_hic, Spearman, dnn, base esm2_concat: alone 0.4742. With augmentation sets 0.5266; with random features of the same width 0.5250.
- **In thesis:** not in thesis (no SOURCE comment). The augmentation result at thesis line ~1597, "an earlier search of 16 datasets", is a different experiment (not verified which).
- **Related:** c18 (hybrid against single representations at a budget), c13 (adds an embedding to the pool), the framework augmentation experiment named in c21's docstring.


### c22 — the hybrid picked on each fold, scored and explained by SHAP and shuffling  (`tests/experiments/complexity/c22_explainability.py`)
- **Question:** On each outer fold, does the top-ranked hybrid beat the top-ranked single representation on held-out molecules? Which representation's features carry the hybrid's importance?
- **How it works:**
  - Inputs: the dataset's representations, its two decided downstream models, outer folds from `nested_cv_outer_folds` (5 requested).
  - Per fold, one search on the training molecules ranks single representations and hybrids; the code keeps the 2 best of each (`base1`, `base2`, `hybrid1`, `hybrid2`).
  - Each is refitted on the training molecules and scored on the held-out fold in the dataset's own metric.
  - SHAP on every hybrid, model and fold: TreeExplainer for tree models, DeepExplainer (GradientExplainer fallback) for the DNN, KernelExplainer on 50 fixed held-out molecules for SVM and GP.
  - SHAP values are summed per source representation inside each fold, turned into shares, then averaged over folds.
  - Grouped permutation importance: all of one representation's features are shuffled together across held-out molecules, 5 repeats, no refit; the drop in the metric is stored.
  - `explain_hybrid` maps each hybrid's top features back to their source representation.
- **Status:** Ran on server. Final files in 10-10b: abagym_Ang2, abagym_EGFR, abagym_VEGF, abcov_kd, adme_hlm, adme_mdr1, adme_rlm, bh_doyle_plates, buchwald_hartwig, caco2, cdr_flexibility, ddg_combined, flab_binding_koenig, flab_thermo_tresanco, ginkgo_acsins, ginkgo_colloidal, ginkgo_hic, ginkgo_pr_cho, ginkgo_titer, herg, herg_ki, humanness_therapeutic, logd, qm9, tz_her2_multiclass. All checked ones hold 5 folds. Partial only: flab_expression_koenig (1 fold, from answer page), nt_promoter_tata. 10-10b adds final files for logd and adme_rlm over 10-09. Every final file checked carries the 2026-10-07 format (SHAP on every model plus permutation). Open faults: dnn files written before the prediction cap had rmse in the thousands on adme_hlm fold 3 (FAULTS A1). gp files on abagym_Ang2, abagym_VEGF, ginkgo_acsins, ginkgo_hic and ginkgo_pr_cho started the kernel width at 1 (FAULTS A2). Many hybrids hold fewer features than their budget (FAULTS A6): 168 of 500 hybrid rows, from answer page.
- **Results so far:**
  - qm9, DNN, 5 folds: `base1` 0.3615 vs `hybrid1` 0.3163 RMSE (checked; `base1` was mordred on every fold).
  - logd, 5 folds: DNN 0.4506 vs 0.4077 RMSE; SVM 0.4928 vs 0.4305 RMSE (checked).
  - adme_rlm, 5 folds: SVM 0.5210 vs 0.4819 RMSE; DNN 0.5327 vs 0.5006 RMSE (checked).
  - flab_thermo_tresanco, 5 folds: DNN 0.6927 vs 0.6788 Spearman; SVM 0.6891 vs 0.6863 Spearman, single representation higher (checked).
  - bh_doyle_plates, 5 folds: RF 16.8715 vs 16.3158 RMSE; LightGBM 16.8977 vs 17.5504 RMSE (checked; not on answer page).
  - tz_her2_multiclass, 5 folds: SVM 0.6693 vs 0.6796 weighted F1; XGBoost 0.7164 vs 0.6579 weighted F1 (checked; not on answer page).
  - Largest SHAP share and largest shuffle drop change with dataset; on qm9 both name mordred, on ddg_combined both name esm2_wt_delta (from answer page).
- **In thesis:** Chapter `chap:method`. Additional file `app:add:hindsight_shuffle`, tables `tab:method:hindsight_smallmol` and `tab:method:hindsight_antibodies`, uses c22's permutation shares (SOURCE comments). Main text in `sec:method:results_single` cites those shares. `fig:blueprints` (`sec:method:results_blueprints`) draws `hybrid1` features per representation via `tests/analysis/thesis_loader_f3.py`. The SHAP method sentence is in `sec:method:comparators`. The main table `tab:hybrid_vs_best_single` comes from c02, not c22 (`thesis_loader_f1.py`).
- **Related:** c02 (same per-fold selection, independent fits), c01 (best single representation in hindsight), c24 (stores the same per-representation feature counts).

### c23 — must the model scoring each representation inside the search match the downstream model?  (`tests/experiments/complexity/c23_evaluator_model_matching.py`)
- **Question:** The search uses an internal evaluator, a model fitted on one single representation at a time, to set each representation's share of the budget. Does a matched evaluator give a better hybrid than one from another model family?
- **How it works:**
  - Per downstream model, `nested_cv_evaluate` runs the whole search inside each outer training fold; its winner is scored on the held-out fold.
  - Only the internal evaluator changes between conditions. A_matched: same model as downstream. B_slight_off: near family. C_strong_off: far family.
  - Pairs from `MATCH_TABLE`: rf→lightgbm/svm, lightgbm→rf/svm, xgboost→lightgbm/svm, svm→gp/lightgbm, dnn→dnn_small (64, 32)/knn, gp→svm/lightgbm.
  - D_nn_size: the DNN scored by a wider DNN (512, 256, 128). E_quick_set: the search chooses among the dataset's decided evaluators.
  - Everything else is the dataset's decided grid, pruning off, no augmentation. Each condition's search picks its own budget.
  - Folds: `first_outer_folds` runs folds 0 to 2, except datasets on a named list, which run all 5.
- **Status:** Ran on server. Final files in 10-10b: abagym_Ang2, abagym_EGFR, abcov_kd, adme_mdr1, buchwald_hartwig, caco2, cdr_flexibility, ddg_combined, flab_thermo_tresanco, ginkgo_acsins, ginkgo_colloidal, ginkgo_hic, ginkgo_pr_cho, ginkgo_titer, herg, herg_ki, humanness_therapeutic, logd, qm9. Partial: abagym_VEGF, flab_expression_koenig. qm9 and adme_mdr1 hold 3 folds. Open faults: conditions differ in width as well as evaluator on most folds. Example: flab_thermo_tresanco DNN fold 1 built 250, 2,000, 250, 50 and 1,000 features under A to E (FAULTS A22, from answer page). The gp used as the SVM's near-family evaluator still starts its kernel width at 1 in current code (FAULTS A3; the thesis SOURCE comment says the same).
- **Results so far:**
  - qm9, 3 folds, LightGBM: matched 0.3459 vs RF evaluator 0.3466 vs SVM evaluator 0.3630 RMSE (checked).
  - qm9, 3 folds, DNN: matched 0.3392 vs small DNN 0.3373 vs kNN 0.3430 RMSE (checked).
  - herg_ki SVM: matched 0.5587 vs LightGBM evaluator 0.557 RMSE (thesis table).
  - abagym_EGFR SVM: matched 0.353 vs LightGBM evaluator 0.299 Spearman (thesis text).
  - ginkgo_colloidal SVM: matched 0.3806 vs 0.3501 and 0.3626 Spearman (from answer page).
- **In thesis:** `tab:method:evaluator` in Additional file `app:add:evaluator`, 17 datasets. Text in `sec:method:results_why`, subsubsection "Dependence on the downstream model". The appendix caption says candidate hybrids are scored by the evaluator. The main text and the code (`hybrid.py`) say it scores single representations.
- **Related:** c06 (model capacity against budget), c33 and c24 (same per-fold search).

### c24 — the search rerun with one more fold of training molecules left out  (`tests/experiments/complexity/c24_blueprint_transfer.py`)
- **Question:** The file name says "blueprint transfer", but the code does not apply one fold's blueprint to another. It asks how much the held-out score moves when the search sees one fold fewer of training molecules.
- **How it works:**
  - Diagonal cell (k, k): the standard per-fold search on every fold except k, rebuilt and scored on fold k.
  - Off-diagonal cell (k, j): a fresh search on every fold except k and j, model fitted on those molecules, scored on fold j. One search serves both (k, j) and (j, k).
  - Run once per decided model. Needs at least 3 outer folds; 5 used.
  - Stored: own score and mean cross score per target fold, molecules each cell's search saw, and the Nogueira stability index over the five folds' selected features.
  - Stability: 1 when every fold picks the same features, about 0 for random picks.
- **Status:** Ran on server. Files in 10-10b: abagym_Ang2, abagym_EGFR, abagym_VEGF, abcov_kd, adme_mdr1, buchwald_hartwig, caco2, cdr_flexibility, ddg_combined, flab_expression_koenig, flab_thermo_tresanco, ginkgo_acsins, ginkgo_colloidal, ginkgo_hic, ginkgo_pr_cho, ginkgo_titer, herg, herg_ki, humanness_therapeutic, logd, qm9. 10-10b adds abagym_Ang2, adme_mdr1, ginkgo_hic and qm9 over 10-09; the files both copies share are identical (checked). Open faults: own and cross searches each pick their own budget, so each comparison differs in molecules seen and in width (FAULTS A22). Example: abcov_kd LightGBM target fold 4, own 34 features, cross 144, 2,000, 100 and 50 (from answer page).
- **Results so far:**
  - logd, 5 folds: DNN own 0.4174 vs one fold fewer 0.4400 RMSE, stability 0.420; SVM 0.4305 vs 0.4600 RMSE, stability 0.497 (checked).
  - herg_ki SVM: 0.5596 vs 0.5931 RMSE (from answer page).
  - abagym_VEGF GP: 0.2957 vs 0.3586 Spearman, one fold fewer higher (from answer page; gp file has the width-1 fault).
  - logd SVM: 372 features were selected on all five folds; 38 are Mordred and 36 ECFP4 (checked; this is `fig:blueprint_logd`).
- **In thesis:** `tab:method:one_fold_fewer` (scores and stability, 18 datasets) in `sec:method:results_overfit`. `fig:blueprint_logd` and the stability paragraph in `sec:method:results_blueprints`. `tab:method:settings_transfer` (LogD settings reused on hERG Ki) is a different experiment, not c24.
- **Related:** c33 (other half of `sec:method:results_overfit`), c22 (same feature counts per representation).

### c31 — sizing each representation's offer by its score before one joint ranking  (`tests/experiments/complexity/c31_perf_weighted_global_selection.py`)
- **Question:** Each representation offers candidate features, sized by its standalone score, and one importance ranking over all offers picks the top B. Does this beat equal offers, equal per-representation quotas, or no cap?
- **How it works:**
  - Does not call `find_best_hybrid`. It imports the engine's weighting functions (`_compute_allocation_weights`) through the shared harness `_alloc_harness`.
  - Target: the best single representation with that downstream model, at full width.
  - Weighted-global: weights computed by an inner grouped 3-fold CV on each outer training fold; offers total 2B or 4B features; one ranking keeps B.
  - Other conditions: equal quota B divided by the number of representations; equal top-N offers then one ranking; all features joined with no cap, then one ranking.
  - Primary score R²; secondary the dataset's own metric. 5 outer folds, group-block bootstrap per cell.
  - Server files: budgets 500 and 1,000; models LightGBM, RF, CatBoost; schemes linear, power_2, top_2; global ranking by lightgbm_gain or TreeSHAP.
- **Status:** Ran on server 19 to 21 August 2026 by SLURM array (`tests/slurm_scripts/run_c31_weighted_global.sh`), not through `run_all()`. Files in `/Volumes/seagate/kirby_cache/c31_weighted_global/`: adme_hlm, adme_mdr1, adme_rlm, bh_doyle_plates, buchwald_hartwig, caco2, flab_expression_koenig, herg_ki, logd, qm9. One file per dataset and model from 21 August, plus an older LightGBM-only file per dataset from 19 August with 14 weighting schemes. All files predate commit 38bf326 (4 September). Before it, the target representation and the shortlist were chosen on the same five folds that grade them. The code has changed since (merged with c30 on 12 September), so the files do not match current code. Local debug folders from late September exist for three datasets and are empty.
- **Results so far:** logd, LightGBM, 5 folds, budget 500 (checked): target mordred 0.5138 RMSE. Weighted-global, linear, offers 2B: 0.4865 RMSE. Equal quota: 0.5034. Equal top-N then ranking: 0.4896. No cap: 0.5283.
- **In thesis:** Not cited. The thesis allocation method "global pool" (`tab:alloc_methods`, `sec:method:allocation`) is the engine's `wglobal_power2`. It was added to the engine in the same commit as c31's model sweep (5a866f4).
- **Related:** c32 (crosses the same mechanism with weighting and number of representations).

### c32 — weighting × mechanism × number of representations, crossed  (`tests/experiments/complexity/c32_allocation_strategy_cross.py`)
- **Question:** An allocation is three choices: how many top representations get any budget, how the budget is weighted, and quota or one joint ranking. Which combinations beat the best single representation?
- **How it works:**
  - Same harness and engine weighting functions as c31; does not call `find_best_hybrid`.
  - Number of representations N: 2, 3, 5 or all 10. Weighting: linear, power_2, softmax T=1, rank decay 0.3. Mechanism: per-representation quota, or offers of 2B then one lightgbm_gain ranking.
  - That gives 32 combinations per cell; a cell is dataset × model × budget (500 or 1,000 features).
  - Models LightGBM, RF, CatBoost. 5 outer folds; weights from an inner grouped 3-fold CV on training molecules.
  - Each combination is compared with the best single representation per fold, in R² and the dataset's own metric, with a bootstrap.
- **Status:** Ran on server 27 to 29 August 2026 by SLURM array (`tests/slurm_scripts/run_c32_allocation_cross.sh`), not through `run_all()`. Files in `/Volumes/seagate/kirby_cache/c32_allocation_cross/`: adme_hlm, adme_mdr1, adme_rlm, bh_doyle_plates, buchwald_hartwig, caco2, flab_expression_koenig, herg_ki, logd, each × 3 models. qm9 failed in all three: the stored representations had 9,978 rows against 4,995 labels (checked). The docstring says files written before 4 September are not comparable to later ones. Before then, the target and the N shortlist were chosen on all rows of the folds that grade them. All server files are from 27 to 29 August. Local debug runs from 29 September to 2 October exist for buchwald_hartwig and herg_ki.
- **Results so far:** logd, LightGBM, 5 folds (checked): target mordred 0.5138 RMSE. Best combination at budget 500 (linear, joint ranking, N=5): 0.4819 RMSE. Worst (softmax, quota, N=10): 0.5089 RMSE. These scores are the target minus the stored sign-corrected delta, not a stored score.
- **In thesis:** Not cited. A comment in `src/kirby/find_best_hybrid.py` (line ~3711) says the five allocation methods were chosen from c32 on 2026-09-01. Those are the five in `tab:alloc_methods`.
- **Related:** c31, c18 (top representation alone against an even split).

### c33 — held-out score at budgets up to 8,000 features, and the search's own score  (`tests/experiments/complexity/c33_budget_overfitting.py`)
- **Question:** When the budget exceeds the number of training molecules, does the held-out score fall? Does the search's own score overstate the held-out score more at large budgets?
- **How it works:**
  - One search per outer fold on training molecules, decided grid apart from the budget list, pruning off, no augmentation.
  - Budgets: 10 to 2,000, plus 4,000 and 8,000. Also 0.5, 0.75, 1, 1.25, 1.5 and 2 times the smallest training-fold size.
  - `nested_cv_evaluate` rebuilds every budget's best configuration on the training molecules and scores it held out (`by_model_budget`).
  - Stored per budget: held-out mean, the search's own mean, and their difference (positive = search over-promised).
  - Per fold, the 2 best single representations and 2 best hybrids, ranked on training molecules, are also scored held out.
  - Regions (under, at or above the training-molecule count) are read from the width actually built, not the budget asked.
- **Status:** Ran on server. Files in 10-10b: abagym_EGFR, abcov_kd, buchwald_hartwig, caco2, cdr_flexibility, ddg_combined, flab_thermo_tresanco, ginkgo_acsins, ginkgo_colloidal, ginkgo_pr_cho, ginkgo_titer, herg, herg_ki, humanness_therapeutic, logd, qm9. 10-10b adds qm9; shared files are identical to 10-09 (checked). ginkgo_hic is ok in the ledger with no file (FAULTS B5). qm9 and humanness_therapeutic hold 3 folds. Open faults: widths fall below budget, often by thousands of features. Example: logd DNN fold 0, budget 8,044 built 4,394 (checked). "Picked" and "hybrid 1" of one search differ on some folds (FAULTS C4).
- **Results so far:**
  - logd DNN, 5 folds: search's own score 0.4257 vs held-out 0.4133 RMSE for `hybrid1`. Budgets picked 8,044, 4,022, 2,000, 5,028 and 3,016 against 4,022 training molecules (checked).
  - logd SVM, 5 folds: 0.4479 vs 0.4314 RMSE (checked).
  - logd DNN: best held-out budget 3,016 features at 0.4066 RMSE; budget 8,000 gave 0.4185 RMSE (checked).
  - abagym_EGFR DNN: search 0.371 vs held-out 0.152 Spearman (thesis table).
- **In thesis:** `tab:method:search_optimism` (14 datasets) in `sec:method:results_overfit`. Its "Search" and "Held out" columns match c33's `hybrid1` means (checked on logd).
- **Related:** c24 (other half of `sec:method:results_overfit`), c16 and c09 (budget sweeps), c06.

### `_cache_evidence_local.py`
A local check, not an experiment: it runs `find_best_hybrid` three times on a cut-down grid to show the result store is reused. It has no results and is not in the thesis.


### Groupings
- Hybrid against the best single representation: c02, c14 (never ran), c21 (fixed recipe), c22 (per-fold hybrid), c07 (within easy and hard halves), c09 (per budget), c05 (as the training set shrinks), c10 (under damaged training labels).
- Hybrid against other ways of getting fewer or combined features: c02 and c15 (PCA, random subsets, all features), c17 (PCA, mRMR, recursive elimination, random), c11 (ensembles of one model per representation).
- Many representations or few, and how the budget is split: c18, c31, c32, c20 (features each representation needs alone).
- Budget against the downstream model and its capacity: c06, c16, c09 (variance shares), c02 (budget per fold).
- Does the search overfit, and are its picks stable: c33 (budgets up to 8,000 features), c24 (one fold fewer, features shared across folds), c03 (settings across datasets; never ran).
- What the hybrid is made of and the search's internals: c01 (each representation alone), c22 (SHAP and shuffling per representation), c23 (internal evaluator), c13 (a new embedding in the pool), c12 (joining measurement types).
- Reusing importance scores after the search: c04 (as a distance weighting for nearest neighbours), c19 (during training).
- Uncertainty of the hybrid's predictions: c08 (conformal intervals).


---

# Similarity experiments

## Overview

This suite asks whether distance in a hybrid representation follows the label more closely than distance in a single representation. It also asks whether methods built on that distance gain from it.
The live suite is 12 scripts in `tests/experiments/similarity/` (s01–s13; s07 was deleted on 2026-09-11, commit 4ab874c).
Ten of them feed Chapter 6 (`chap:similarity`). s09 is an aggregator that never ran. s04 duplicates the cliff partition of s12.
Chapter 4 (`chap:flaws`) uses none of these scripts directly. Its similarity numbers come from `scratchpad/thesis_code_todo/distance/` (see the end of this section).
The earlier similarity-metrics study in `tests/experiments/similarity_metrics/` (about 30 scripts) is not used anywhere in thesis.tex. It is summarised as one block.

Common settings of s01–s13, read in the code and in thesis Section `sec:similarity:methods`:
- Five outer folds, grouped by scaffold family for small molecules. Everything compared is chosen by searches on that fold's training molecules only.
- Three searches per fold: the general search (per downstream model), the kNN search (kNN model, Euclidean, 5 neighbours, budgets capped at 500 features), and the similarity search.
- The similarity search is the hybrid search with `objective='similarity'`. It scores a candidate by predicting each molecule's label from its five nearest neighbours (inverse-distance weights, Manhattan), with neighbours from other scaffold groups. The score is Spearman on regression and 2(AUROC−0.5) on classification.
- Feature spaces compared: the kNN model's two best single representations, each downstream model's two best single representations and two best hybrids, the kNN search's top hybrid ("kNN hybrid"), and the similarity search's top hybrid ("similarity hybrid").
- Features are scaled on training molecules: 0/1 features unchanged, the rest standardised, held-out values clipped to the training range.
- Nearest-neighbour settings swept everywhere neighbours are used: distance (Manhattan, cosine, Minkowski p=0.5; Minkowski dropped above 5,000 molecules), 5/10/20 neighbours, rank or 1/d² neighbour weights, and importance weighting on or off.
- Two downstream models per dataset, the ones decided for that dataset in the hybrid chapter's grid.
- Thesis datasets (14): qm9, caco2, adme_mdr1, herg_ki, herg (hERG blocking, classification), logd, abagym_EGFR, flab_thermo_tresanco, ginkgo_acsins, ginkgo_colloidal, ginkgo_pr_cho, ginkgo_titer, ddg_combined, buchwald_hartwig.
- The server folders also hold abcov_kd, cdr_flexibility, flab_expression_koenig and humanness_therapeutic. These are not among the chapter's 14 datasets.
- The thesis tables cite `/Volumes/seagate/kirby_results_2026-10-10b/similarity/`. The 2026-10-09 copy lacks adme_mdr1 on every script, and lacks a finished qm9 file for s08 and s10.

Faults that apply suite-wide (from the 2026-10-08 answer pages' FAULTS.md, stated against the 10-09 copy):
- The similarity hybrid does not exist on the five ginkgo datasets. Their folds are the competition's five groups, so each held-out block inside a fold is one group and no neighbour from another group exists.
- On ddg_combined the similarity search built a hybrid on fold 0 only. Folds 1–4 store "the per-fold similarity search scored no hybrid", with no reason recorded.
- Every DNN score in the 10-09 copy predates commit c18ee30, which clips DNN predictions to the training-label range. The 10-10b timestamps were not compared against that commit here.
- Hybrids can hold fewer features than their budget, because the build stops at the last feature with importance above zero. This affects budget sweeps (s08).

---

### s01 — Distance against label difference, every single representation  (`tests/experiments/similarity/s01_similarity_baseline.py`)
- **Question:** In each single representation, does the distance between two molecules rise with the difference in their labels?
- **How it works:**
  - All 10 general representations of the dataset (8 on ddg_combined). No model is fitted and nothing is selected.
  - Per outer fold, up to 50,000 pairs of held-out molecules are sampled.
  - The score is the Spearman correlation between pair distance and absolute label difference. On classification the difference is 1 for different classes, else 0.
  - Swept: representation × distance (Manhattan, cosine, Minkowski p=0.5; herg, logd and qm9 have only Manhattan and cosine).
  - Also stored: violation rate (share of each held-out molecule's 5/10/20 nearest training molecules whose label differs by more than the median training-pair difference), SALI per distance, and 200 label shuffles as reference spread.
  - No importance weighting.
- **Status:** Ran on the server, 5 folds, all 14 thesis datasets plus the four extra ones (10-10b copy).
- **Results so far:** The best pair of representation and distance per dataset, picked on held-out molecules, mean over 5 folds (thesis `tab:similarity:single_spread`):
  - qm9 0.279 Spearman (ChemBERTa-2, Manhattan), folds 0.206 to 0.333.
  - caco2 0.229 (topological torsions), herg_ki 0.117 (SMI-TED, Manhattan), logd 0.147 (MoLFormer), herg 0.041 (MoLFormer, cosine).
  - ginkgo_colloidal 0.016 (AbLang-2, Minkowski); 15 of its 30 representation–distance pairs are at or below zero.
  - Pairs at or below zero also occur on adme_mdr1, abagym_EGFR, ginkgo_acsins and ginkgo_pr_cho (thesis comment, checked by the author against 10-10b).
  - Answer page s01.md gives the same qm9, caco2, herg_ki and herg tops from the 10-09 copy.
- **In thesis:** Ch6, `sec:similarity:results_similarity`, `tab:similarity:single_spread`.
- **Related:** s02 (same statistic in hybrids); Chapter 4 distance scripts (same statistic, all molecules, no folds).

### s02 — Distance against label difference in hybrids  (`tests/experiments/similarity/s02_hybrid_similarity.py`)
- **Question:** Does distance in a hybrid follow label difference more closely than distance in the best single representation?
- **How it works:**
  - Same Spearman statistic as s01, on up to 50,000 held-out pairs per fold.
  - Spaces: kNN model's two best single representations, each model's two best single representations, each model's two hybrids, the kNN hybrid, the similarity hybrid.
  - Each hybrid is also re-read at each decided budget of its search (no extra searches).
  - Every space is scored unweighted and importance-weighted. A hybrid's weights come from LightGBM gain refitted on the hybrid's own features. A single representation's weights come from LightGBM gain on its own features.
  - Weights use w = (s/s̄)^a, with a = 1 under Manhattan, 0.5 under cosine, 2 under Minkowski p=0.5.
  - Pairings: a model's hybrid against that model's best single representation; the kNN and similarity hybrids against the kNN model's best single representation.
- **Status:** Ran, 5 folds, 14 thesis datasets. The 10-10b files for caco2, herg, herg_ki, ginkgo_titer and ginkgo_colloidal hold unweighted distances only (thesis comment). The similarity hybrid is missing on ginkgo and on ddg_combined folds 1–4.
- **Results so far:** kNN hybrid against kNN model's best single representation, Manhattan, unweighted, mean of 5 folds (thesis `tab:distance_tracks_activity`; checked in `figures_ch5_7/tab_distance_tracks_activity.csv`, which records its source keys):
  - qm9 0.130 vs 0.378; logd 0.077 vs 0.269 (hybrid higher in all 5 folds); herg_ki 0.073 vs 0.263.
  - caco2 0.152 vs 0.197 (hybrid lower in folds 2 and 4); adme_mdr1 0.229 vs 0.288; herg 0.026 vs 0.055.
  - abagym_EGFR 0.038 vs 0.074; ginkgo_colloidal 0.0002 vs 0.050; buchwald_hartwig 0.107 vs 0.143.
  - Weighted single against weighted kNN hybrid, logd: 0.276 vs 0.309; qm9: 0.329 vs 0.460 (`tab:similarity:feature_spaces`).
  - Similarity hybrid, unweighted, caco2: 0.277, against 0.197 for the kNN hybrid and 0.229 for the hindsight-best single pair.
  - Predictive hybrids, ginkgo_pr_cho GP: best single 0.124, first hybrid 0.094, second hybrid 0.093 (`tab:similarity:predictive_hybrids`).
  - Answer page s02.md reports abcov_kd going the other way, 0.0901 to 0.0376 (not a thesis dataset; not re-checked).
- **In thesis:** Ch6, `sec:similarity:results_similarity`: `tab:distance_tracks_activity` (file `figures_ch5_7/tab_distance_tracks_activity.tex`), `tab:similarity:predictive_hybrids`, `tab:similarity:feature_spaces`.
- **Related:** s01, s08, s05.

### s03 — Label information in the features a hybrid keeps  (`tests/experiments/similarity/s03_feature_concentration.py`)
- **Question:** Do the features a hybrid keeps from a representation carry more label information than other features of the same number?
- **How it works:**
  - For each model's two best hybrids per fold, and each representation the hybrid draws from.
  - Compared blocks: the kept features, a random subset of the same size, PCA of the representation to the same number of components, and the whole representation.
  - Each block is reduced to at most 50 principal components. A one-feature Kraskov mutual-information estimate (scikit-learn) is computed per component and summed.
  - The mean over three label permutations is subtracted.
  - Reported on training molecules (a concentration check) and on held-out molecules (scaling and PCA fitted on training molecules).
  - Also stores the Gini coefficient of each importance method's scores per representation.
- **Status:** Ran, 5 folds, 14 thesis datasets. Known fault, stated in the code docstring and FAULTS.md: the summed estimate is not mutual information. Shared information is counted twice, so the sum can exceed the label's entropy (2.13 nats on a two-class label in an earlier run, against a ceiling of 0.693).
- **Results so far:** Thesis conclusion: except on qm9 and buchwald_hartwig, the kept features carry no more of this summed estimate than the same number of principal components. Answer page s03.md (10-09): on qm9 the kept features beat random, PCA and whole for both models. On herg_ki and logd they beat random for both models, but not on herg. Not re-checked here.
- **In thesis:** Ch6, one line in `sec:similarity:conclusions`; table in Additional file `app:add:similarity_label_information`.
- **Related:** none in this suite.

### s04 — Activity cliffs, ECFP4 partition, kNN and model arms  (`tests/experiments/similarity/s04_activity_cliffs.py`)
- **Question:** Are activity-cliff molecules predicted better from a hybrid than from a single representation, by nearest neighbours and by the two downstream models?
- **How it works:**
  - Cliff rule from `similarity_helpers.calibrated_cliff_rule`: two molecules are similar at ECFP4 Tanimoto ≥ 0.55.
  - The label gap is solved on each fold's training labels so that 15% of training molecules are cliffs.
  - A held-out molecule is a cliff molecule if at least two similar training molecules span more than the gap. Its own label is never read.
  - kNN arms: kNN model's two best single representations, each model's two hybrids, kNN hybrid, similarity hybrid, at every nearest-neighbour setting, unweighted and weighted.
  - Model arms: each model fitted on its two best single representations and two hybrids.
  - Scored separately on cliff and other held-out molecules, in the dataset's metric.
- **Status:** Ran, 5 folds, on the six small-molecule thesis datasets. Writes a skip file for antibody and reaction datasets ("molecular-only experiment"). s04 and s12 use the same partition function, so they flag the same molecules on SMILES datasets.
- **Results so far:** Answer page s04.md (10-09): kNN on cliff molecules, logd 0.6337 vs 0.5454 RMSE and qm9 0.6848 vs 0.6243 RMSE (single vs distance hybrid). qm9 has 7 to 13 cliff molecules per fold. Not re-checked. The thesis table is built from s12, not s04.
- **In thesis:** Not cited directly. The thesis cliff table reads s12 (`ch6_tables_gen.py` comment says s04 flags the same molecules).
- **Related:** s12 (same partition, model arms reported).

### s05 — Prediction from nearest neighbours  (`tests/experiments/similarity/s05_knn_prediction.py`)
- **Question:** Does predicting each held-out molecule's label from its nearest training molecules work better in a hybrid than in a single representation?
- **How it works:**
  - Spaces: the kNN model's two best single representations, each model's two hybrids, kNN hybrid, similarity hybrid. Each hybrid is built at the budget its per-fold search chose.
  - Every space is scored at every nearest-neighbour setting, unweighted and importance-weighted.
  - Weighted distance here uses w = s^(1/2) under every distance. A hybrid uses its search's own importance; a single representation uses importance computed on training molecules.
  - Comparison fixed before the run: kNN hybrid against kNN model's best single representation, Manhattan, 5 neighbours, rank weights, unweighted.
  - Second comparison: for each held-out fold, both sides are re-picked from the other four folds' held-out scores. The hybrid side picks from up to 216 combinations and may use weighting. The single side picks from up to 36, unweighted only.
  - So the second comparison differs between sides in two ways at once: number of combinations and access to weighting.
- **Status:** Ran, 5 folds, 14 thesis datasets. Similarity hybrid missing on ginkgo and on ddg_combined folds 1–4. FAULTS.md notes the re-pick on ddg_combined uses one fold on the hybrid side and five on the single side.
- **Results so far:** Fixed comparison, single vs kNN hybrid, mean of 5 folds (thesis `tab:nearest_neighbour_prediction`; logd checked in the raw 10-10b file):
  - qm9 0.669 vs 0.420 RMSE; logd 0.677 vs 0.583 RMSE; herg_ki 0.628 vs 0.582 RMSE; caco2 0.339 vs 0.325 RMSE.
  - adme_mdr1 0.554 vs 0.520 RMSE; herg 0.814 vs 0.834 AUROC.
  - ddg_combined 0.142 vs 0.090 Spearman; ginkgo_acsins 0.391 vs 0.388 Spearman; buchwald_hartwig 14.5 vs 16.2 RMSE (yield %).
  - Re-picked per fold (`tab:similarity:knn_picked_again`): ginkgo_colloidal 0.092 vs 0.291 Spearman; ginkgo_pr_cho 0.350 vs 0.260; logd 0.638 vs 0.565 RMSE.
  - Importance weighting, Manhattan, 5 neighbours, rank (`tab:importance_weighting`): logd single 0.677 to 0.607 RMSE weighted; hybrid 0.583 to 0.577.
- **In thesis:** Ch6, `sec:similarity:results_knn`: `tab:nearest_neighbour_prediction` (file `figures_ch5_7/tab_nearest_neighbour_prediction.tex`), `tab:similarity:knn_picked_again`, `tab:importance_weighting`; Additional file `app:add:similarity_knn_by_setting`.
- **Related:** s10 (same kNN task inside its weighting sweep), s13 (retrieval uses the same neighbours).

### s06 — Distance against prediction error  (`tests/experiments/similarity/s06_applicability_domain.py`)
- **Question:** Does a held-out molecule's distance to its nearest training molecules rise with the error a model makes on it?
- **How it works:**
  - Distance statistic: mean distance to the 5, 10 or 20 nearest training molecules, under every distance, unweighted and weighted.
  - Spaces: kNN model's two best single representations, kNN hybrid, similarity hybrid, each model's two hybrids.
  - Errors from each decided model fitted two ways: on all representations joined, and on its own first hybrid. Error is absolute residual, or 1 − P(true class).
  - Score: Spearman between distance and error, per fold.
  - Also stores split-conformal intervals (80/20 group-aware split, absolute residual) and a decile reliability table.
- **Status:** Ran, 5 folds, 14 thesis datasets. The model-on-own-hybrid error paired with that same hybrid's distance is flagged `same_space=True` in the file.
- **Results so far:** Model fitted on all representations joined, Manhattan, 5 neighbours, unweighted; single vs kNN hybrid (thesis `tab:similarity:error_distance`):
  - herg_ki SVM 0.200 vs 0.276 Spearman (checked in raw 10-10b file: 0.1996 vs 0.2761).
  - qm9 DNN 0.072 vs 0.114; herg LightGBM 0.139 vs 0.019; ginkgo_colloidal DNN 0.136 vs 0.030.
  - ginkgo_titer DNN −0.095 vs 0.110; buchwald_hartwig RF −0.002 vs −0.032.
  - Conformal results are on answer page s06.md only (herg_ki width 2.1086 to 1.9476 at 0.9129 coverage); not in the thesis.
- **In thesis:** Ch6, `sec:similarity:results_other`, `tab:similarity:error_distance`.
- **Related:** s02 (same spaces).

### s08 — Best number of features for prediction and for distance  (`tests/experiments/similarity/s08_optimal_dimension.py`)
- **Question:** Is the number of features at which a hybrid predicts best the same number at which its distances best follow label differences?
- **How it works:**
  - Prediction side: at every decided general budget, each model's top two hybrids, scored held-out in the dataset metric.
  - Distance side: at every decided distance budget, the kNN hybrid; and the similarity hybrid at each general budget.
  - On each hybrid: the s02 Spearman, kNN prediction at every setting, violation rate, Levina–Bickel intrinsic dimension (10 neighbours), and correlation with the full concatenation's distances.
  - The thesis table reads the DNN's top hybrid at each budget: its prediction score and its own distance Spearman.
- **Status:** Ran, 5 folds, 14 thesis datasets. The qm9 file was partial in the 10-09 copy and complete in 10-10b (written 9 Oct 14:51, per thesis comment). Fault A6 applies: a hybrid can hold fewer features than its budget, so two budgets can give the same hybrid.
- **Results so far:** DNN only; datasets without a DNN (herg, ginkgo_acsins, buchwald_hartwig) have no row (thesis `tab:similarity:feature_count`):
  - logd: prediction 0.415 RMSE at 500 features, 0.408 at 2,000. Distance Spearman 0.231 at 500, 0.154 at 2,000.
  - herg_ki: 0.569 to 0.575 RMSE across 250–2,000 features; Spearman 0.235 at 250, 0.135 at 2,000.
  - flab_thermo_tresanco: Spearman (prediction) peaks at 250 features (0.698); distance Spearman peaks at 50 (0.193).
- **In thesis:** Ch6, one line plus `tab:similarity:feature_count`, `sec:similarity:results_similarity`.
- **Related:** s02.

### s09 — Cross-dataset summary  (`tests/experiments/similarity/s09_cross_dataset_summary.py`)
- **Question:** For each dataset, what is "hybrid minus single representation" per fold across the outputs of s02, s04, s05, s06 and s08?
- **How it works:**
  - Fits nothing. Reads the five input files of one dataset.
  - Checks they share fold counts, row counts, kNN settings, metric, decided models and per-fold selections.
  - Writes per-fold differences, means and ranges. Never averages across datasets; no tests.
- **Status:** Never ran. The 10-09 LEDGER.md lists `cross:s09` as pending.
- **Results so far:** none.
- **In thesis:** not in thesis.
- **Related:** s02, s04, s05, s06, s08.

### s10 — Importance weighting in four models that use a distance  (`tests/experiments/similarity/s10_importance_downstream.py`)
- **Question:** Does multiplying each feature by its importance help kNN, a Gaussian process, an SVM, and q-RASAR?
- **How it works:**
  - Same spaces as s05. Each hybrid is rebuilt at every decided budget of its search.
  - Each space is fitted unweighted and with features multiplied by importance (s^(1/2)).
  - kNN at every nearest-neighbour setting.
  - GP: constant × RBF + white kernel, normalised labels; two length-scale starts (1 and the median pair distance). Regression only, up to 5,000 molecules.
  - SVM: RBF, gamma='scale', C=1, standardised labels. Regression only, up to 10,000 molecules.
  - q-RASAR: each model gets four extra features from the nearest training molecules (mean label, label spread, nearest distance, mean distance). These use the top three settings from a no-model screen on training molecules.
- **Status:** Ran, 5 folds, 14 thesis datasets. The qm9 file was partial in 10-09; 10-10b holds complete qm9 and adme_mdr1 files (written 10 Oct, thesis comment). The answer page lists s10 logd as still running in the 10-09 copy.
- **Results so far:** SVM on the kNN model's best single representation, unweighted vs weighted, mean of 5 folds (thesis `tab:similarity:svr_weighting`):
  - herg_ki 0.607 vs 0.630 RMSE (checked in raw 10-10b file: 0.6072 vs 0.6298).
  - logd 0.508 vs 0.507 RMSE; qm9 0.449 vs 0.459 RMSE.
  - ginkgo_colloidal 0.282 vs 0.158 Spearman; ginkgo_titer 0.376 vs 0.296 Spearman.
  - The GP is not tabled. Its two length-scale starts disagree and several antibody folds are NaN. The text quotes herg_ki at the fixed start, 0.5819 vs 0.7115.
  - q-RASAR on logd: the SVM improves with ten neighbours under cosine and gets worse with twenty (thesis text; no table).
- **In thesis:** Ch6, `sec:similarity:results_knn`, `tab:similarity:svr_weighting`.
- **Related:** s05; complexity experiment c04 (`c04_importance_as_metric`) supplies `tab:similarity:joined_weighting` (kNN on all representations joined, per importance method).

### s11 — Simulated Bayesian optimisation  (`tests/experiments/similarity/s11_bayesian_optimization.py`)
- **Question:** Does choosing which molecules to measure next work better when the surrogate model sees a hybrid than a single representation?
- **How it works:**
  - Starts from a set of whole groups holding at least 5% of molecules (scaffolds on herg_ki and qm9, additives on buchwald_hartwig).
  - The surrogate picks 100 molecules one at a time by expected improvement.
  - Repeated from three starting sets. Score: area under the best-label-found-so-far curve (lower is better when the target is the lowest label).
  - Conditions: ECFP4 + RBF GP, ECFP4 + Tanimoto GP, best single representation + RBF GP, PCA to 50, random projection to 50, random picks.
  - Hybrid conditions: each model's two hybrids, from a per-start-set search, with GP, GP refitted every 25 picks, NGBoost, a Bayesian neural network, and a GAUCHE GP.
  - Despite the name `hybrid_50d`, hybrids are built at their searched budget (FAULTS.md: 39 to 656 features).
- **Status:** Ran with results on qm9, herg_ki, buchwald_hartwig and adme_mdr1 (10-10b). Skipped on caco2 and logd ("best label already in the starting set" because labels are clipped at the 99th percentile), herg ("classification_degenerate") and all antibody datasets. The GAUCHE surrogate was not installed (FAULTS.md), so its conditions have no valid curves. Fault A3: the surrogate GP starts its length scale at 1.
- **Results so far:** qm9, area at 100 picks, mean of 3 start sets (checked in raw 10-10b file):
  - best single representation (GP) 260.1, sd 21.0. The best of 20 hybrid runs is NGBoost on LightGBM's first hybrid, 253.4, sd 49.3.
  - GP on DNN's first hybrid 270.5; ECFP4 RBF 324.4; random picks 341.2.
  - The thesis says the single representation was tried once, and the other 19 hybrid runs all scored worse than it. Note the 253.4 hybrid run uses a different surrogate (NGBoost) than the single representation (GP).
  - Answer page s11.md (10-09, herg_ki): rank-2 DNN hybrid refitted every 25 picks 410.59 vs raw ECFP4 428.03 (pKi × picks).
  - adme_mdr1 has a result file but the thesis does not mention it.
- **In thesis:** Ch6, one paragraph at the end of `sec:similarity:results_knn`.
- **Related:** none.

### s12 — Activity cliffs, model arms, all domains  (`tests/experiments/similarity/s12_cross_domain_cliffs.py`)
- **Question:** Does each downstream model predict held-out cliff molecules better from its first hybrid than from its best single representation, and how does that compare with other molecules?
- **How it works:**
  - Same per-fold cliff rule as s04 on SMILES datasets (ECFP4 Tanimoto ≥ 0.55; 15% of training molecules).
  - Without SMILES, similarity is 1/(1+d) on the scaled concatenation of all representations, at a fixed percentile of pairs, under each distance.
  - Model arms: each model on its two best single representations and two hybrids. Headline: first hybrid against best single representation.
  - kNN arms as in s04, with the setting picked by leave-one-out on training molecules.
  - A cliff score needs at least 10 cliff molecules in a held-out fold.
- **Status:** Ran, 5 folds, all 14 thesis datasets. On the antibody datasets almost no fold reaches 10 cliff molecules (ddg_combined: 11 of 429 held-out antibodies, per thesis). Known fault (FAULTS.md): training molecules need one similar partner across the gap, held-out ones need two. On caco2 fold 0 this gives 15.0% of training molecules against 31.0% held out.
- **Results so far:** Single vs first hybrid, mean over folds with ≥10 cliff molecules (thesis `tab:similarity:cliffs`):
  - caco2 DNN cliffs 0.340 vs 0.288 RMSE, others 0.289 vs 0.272 (checked in raw 10-10b file: 0.3396 vs 0.2881).
  - logd DNN cliffs 0.430 vs 0.388 RMSE, others 0.465 vs 0.417.
  - herg_ki SVM cliffs 0.561 vs 0.530 RMSE; herg LightGBM cliffs 0.793 vs 0.808 AUROC.
  - qm9 DNN cliffs 0.242 vs 0.280 RMSE (two folds only), others 0.358 vs 0.313. qm9 has 48 cliff molecules in all, adme_mdr1 49.
- **In thesis:** Ch6, `sec:similarity:results_cliffs`, `tab:similarity:cliffs` (six small-molecule datasets only).
- **Related:** s04.

### s13 — Retrieval and clustering  (`tests/experiments/similarity/s13_clustering_retrieval.py`)
- **Question:** In a hybrid, do a molecule's nearest neighbours have closer labels, and do k-means clusters hold more similar labels?
- **How it works:**
  - Retrieval: each held-out molecule's label gap is the weighted mean absolute label difference to its nearest training molecules (classification: 1 − weighted share in its class). Lower is better.
  - Clustering: k-means (squared Euclidean) with 3, 5, 10 clusters fitted on training molecules; held-out molecules assigned.
  - Coherence: 1 − mean within-cluster label SD / overall SD (classification: mean majority-class share). Clusters with under two held-out molecules are dropped.
  - Spaces: kNN model's two best single representations, each model's two hybrids, kNN hybrid, similarity hybrid.
  - Control: PCA of all representations joined, at 100 components and at each hybrid's width. PCA is not re-standardised after projection (fixed 2026-10-05).
  - Retrieval at every nearest-neighbour setting; the thesis table re-picks space and setting per fold from the other folds.
- **Status:** Ran, 5 folds, 14 thesis datasets. FAULTS.md: the re-pick can choose a space with no score on the held-out fold, which leaves NaN hybrid values on ddg_combined.
- **Results so far:** Retrieval, re-picked, unweighted; single / hybrid / PCA (thesis `tab:similarity:retrieval`):
  - qm9 0.601 / 0.393 / 0.570; logd 0.593 / 0.543 / 0.579; herg_ki 0.528 / 0.512 / 0.543 (label units).
  - ddg_combined 1.439 / 1.507 / 1.567; abagym_EGFR 0.271 / 0.270 / 0.261; buchwald_hartwig 12.0 / 12.1 / 16.3 (yield %).
  - Clustering coherence at 10 clusters, single vs hybrid (`tab:similarity:clustering`): qm9 0.130 vs 0.357; logd 0.100 vs 0.225; ginkgo_colloidal 0.537 vs 0.461.
- **In thesis:** Ch6, `sec:similarity:results_other`, `tab:similarity:retrieval`, `tab:similarity:clustering`; Additional file `app:add:similarity_retrieval_by_setting`.
- **Related:** s05.

---

### Chapter 4 similarity numbers — not from this suite  (`scratchpad/thesis_code_todo/distance/compute.py`, `mcs.py`, `collect.py`)
- **Question:** Over all molecules, without folds or hybrids, does distance in standard representations follow label difference, and do representations agree on neighbours and cliffs?
- **How it works:**
  - Reads cached feature matrices only. Seven datasets: the six small-molecule regression datasets of the thesis plus HLM clearance.
  - Tanimoto for binary representations; Manhattan on standardised non-constant features for the rest.
  - All-pairs Spearman between distance and absolute label difference (2,000-molecule sample when larger).
  - Five-nearest-neighbour prediction with neighbours from other scaffold groups.
  - Cliffs via `similarity_helpers.calibrated_cliff_rule` from this suite (ECFP4 Tanimoto ≥ 0.55, gap set so 15% of molecules are cliffs).
  - MCS on 500 random pairs per dataset with RDKit, 2 s timeout per pair.
- **Status:** Local computation, no folds. Not in `tests/experiments/`; lives in the repo's `scratchpad/`.
- **Results so far:** herg_ki with ECFP4: all-pairs 0.06 Spearman, five-neighbour prediction 0.72 Spearman, 211 of 1,412 molecules are cliffs at a 1.76 pKi gap (thesis `tab:flaws:distance_summary`).
- **In thesis:** Ch4, `sec:flaws:similarity`, `tab:flaws:distance_summary`, `tab:flaws:neighbor_overlap`, `fig:flaws:herg_pairs`; Additional file `app:add:flaws`.
- **Related:** s01 (same statistic, held-out pairs per fold).

---

### Similarity-metrics study (block)  (`tests/experiments/similarity_metrics/`)
None of these scripts is cited in thesis.tex. Their local JSON files are local debug runs on two or three datasets, not results, and are not reported here. Last code changes were 2026-09-04 to 2026-09-18.

The study's stated yardsticks are (1) predicting each molecule's activity from its five nearest neighbours, scaffold-grouped, scored by Spearman, and (2) Spearman between a space's pair distance and maximum-common-substructure (MCS) distance on cached pairs. ECFP4 Tanimoto is one competitor, never the reference.

Scripts grouped by question:
- **Which fingerprint × coefficient best reflects activity, on 30 MoleculeACE targets?** `build_moleculeace.py`, `learned_embeddings.py`, `run_battery_moleculeace.py`, `simeval.py`. A grid of fingerprints (ECFP4/6, FCFP4, atom pair, torsion, RDKit, MACCS, count forms, descriptors) × six coefficients, plus ChemBERTa, MoLFormer and a triplet-trained MLP over ECFP4. Scored on held-out scaffold folds with a battery (neighbour prediction, MODI, neighbourhood behaviour, SALI cliff AUROC, BEDROC, violation rate, noise detection and recovery). The design includes paired Wilcoxon tests. Status: a SLURM job was submitted 2026-07-13 per a memory note (unverified); no output exists in the 10-09 server copy or locally.
- **Do other molecular representations make better similarity spaces?** `extra_fingerprints.py` (Avalon and others), `augmented_descriptor_reps.py` (PDV, Mordred), `structural_embeddings.py` (MAP4, TDA, mol2vec), `shape_similarity.py` and `aug3d_similarity.py` (USR, USRCAT, 3D pharmacophores on ETKDG conformers), `nary_similarity.py` (iSIM and related n-ary indices), `triplet_similarity.py` (learned triplet metric on herg_ki). Local only.
- **Does stacking feature blocks into one space beat a count fingerprint?** `creative_hybrids.py`, `creative_hybrids_v2.py`, `feature_ablation.py`, `aggregate.py` (six ways to fuse several metrics), `winning_recipe_vs_standard.py`, `winning_recipe_honest.py`. The "winning recipe" is z-scored ECFP4 counts plus descriptors, shape and fine 2D blocks, Euclidean. `winning_recipe_honest.py` re-selects on one half of the molecules and scores on the other, because the earlier version selected and scored on the same MCS pairs. Local only.
- **Which features to keep for similarity rather than prediction?** `rankers.py` (RReliefF, Laplacian score, Spearman-to-label-gap or to-MCS scorers). These were the precursor to `objective='similarity'` in `find_best_hybrid` (added 2026-07-31, commit ce156d0). Local only.
- **Structure as the judge instead of activity.** `structural_yardstick.py`, `mcs_cache.py` (cached MCS distances for fixed pairs). Local only.
- **Does a similarity space chosen for structure clean noisy labels better?** `similarity_noise_cleaning.py`. Injects label noise and compares cleaning across stacked spaces. Local only.
- **Antibody and protein variant libraries described by what changed from the wildtype.** `scaffold_reps.py`, `scaffold_experiment.py` (flab koenig, adams, phillips), `scaffold_ddg_experiment.py` (single-point ddG), `scaffold_feature_isolation.py`, `base_rep_comparison.py`. `feature_type_study.py` is a library imported by `tests/experiments/opig_scaffold_similarity.py`. Local only; an open confound from mutation count in the adams and phillips libraries is recorded in a memory note (unverified).
- Support files: `_data.py`, `_heldout.py`, `_nanrank.py`, `build_local_datasets.py`, `saboteur_finder_demo.ipynb`.

**In thesis:** not in thesis. **Related:** the similarity search objective used by s02–s13 descends from this study.

---

### similarity_optimized_hybrid — similarity-search hybrid for noise handling  (`tests/experiments/similarity_optimized_hybrid.py`)
- **Question:** Does a hybrid chosen with `objective='similarity'` preserve distance–label agreement on held-out folds better than the predictive hybrid, and does it help similarity-based noise detection and cleaning?
- **How it works:**
  - Small-molecule regression datasets (usage example: logd, herg), three seeds.
  - Payoff 1: held-out distance–label Spearman and violation rate, similarity hybrid against predictive hybrid and best single representation.
  - Payoff 2: Gaussian, heteroscedastic and outlier label noise; every method in `kirby.utils.noise_mitigation_methods` (neighbour-based, geometry-based and control groups); detection precision/recall/F1 and recovery by removal or smoothing.
  - Payoff 2b: per-molecule noise scorers scored by AUROC and precision at k.
  - Payoff 3: importance-weighted distances (none, square root, squared, top-k) fed to the distance-based methods.
  - Searching by the pair-distance Spearman objective was cut on 2026-08-26; asking for it stops the run.
- **Status:** No results found in the 10-09 server copy, the ledger, or `tests/results/`. A submit script exists (`tests/slurm_scripts/submit_sim_optimized_hybrid.sh`). Whether it ever ran is unverified.
- **Results so far:** none found.
- **In thesis:** not in thesis.
- **Related:** s02, s05; noise suite (neighbour-based detectors in Chapter 7).

### structural_geometry_survey — antibody structure features added to representations  (`tests/experiments/structural_geometry_survey.py`)
This script does not belong to the similarity suite. It adds seven predicted-structure geometry feature sets (VH–VL angles, CDR shape, paratope charge and others; 6 to 162 features) to antibody base representations and fits downstream models. Result files exist in `tests/results_server/geometry_survey/` (about ten antibody datasets); they were not read for this section. Not in thesis.

---

## Groupings

- **Distance against label difference:** s01 (single representations), s02 (hybrids, weighted and unweighted), s08 (by number of features), and the Chapter 4 distance scripts (all molecules, no folds).
- **Prediction or retrieval from nearest neighbours:** s05 (prediction), s13 (retrieval and clustering), s10 (kNN part).
- **Importance weighting in distance-based models:** s10, plus the weighted arms of s02 and s05, plus complexity c04 for all representations joined.
- **Activity cliffs:** s04 and s12 (same partition function; thesis reports s12).
- **Distance as an applicability domain:** s06.
- **Label information in kept features:** s03.
- **Molecule selection:** s11.
- **Aggregation, never ran:** s09.
- **Similarity metrics and similarity-space design, not in thesis:** the `similarity_metrics/` block and `similarity_optimized_hybrid.py`.


---

# Noise suite (`tests/experiments/noise/`)

## Overview

The noise suite asks two things. Can noisy labels in a molecular dataset be found? Does acting on them improve a model's held-out score? Each method is run on the best single representations and on the adaptive hybrid representation, so every result is also a hybrid-against-single comparison.

- **Size.** 37 experiments: n01, n02, n06, n07, n09, n11–n27, n29, n30, n30b, n31–n42. n16 has a second script, `n16_analyse.py`, that reads its results. n17 and n18 also have helper folders of the same name. There is no n03–n05, n08, n10 or n28 script in the folder.
- **Thesis.** The suite feeds Chapter 7, "Detecting and handling label noise" (`chap:noise`, thesis.tex lines 3473–4994). It also uses the noise definitions of Chapter 2 (`sec:noise_injection`, `sec:noiseinject_pkg`, `tab:noise_terms`) and Chapter 4's neighbour-cleaning study (`chap:flaws`).
- **Where the thesis numbers come from.** Every number block in Ch7 has a `% SOURCE [N7-xx]` comment naming the raw file and key. `tests/analysis/ch7_numbers.py` re-reads each one from the raw files. The SOURCE lines cite `/Volumes/seagate/kirby_results_2026-10-10b`, a copy newer than the 2026-10-09 copy. The 10-10b copy also has n06 results and the MDR1–MDCK efflux dataset (`adme_mdr1`), which the 10-09 copy lacks.
- **c10.** `c10_noise_robustness.py` lives in this folder but is covered in another section. It compares how hybrid and single-representation scores fall as noise rises. It is not used in the thesis. Chapter 3 (`chap:robustness`) comes from the separate NoiseInject paper code (`qsar_qm_models`), not from this suite.
- **Debug scaffolding.** `_smoke_*`, `_loop_*`, `_probe_*`, `_local_run.py` and `_fast_loop.py` are local debug and test scripts. They produce no thesis result. `check_noise_run.py`, `coverage.py`, `noise_cache.py`, `d_vs_dprime.py` and `noise_ceiling_analyse.py` are support code.
- **Datasets with results (10-10b copy).** Small molecules: Caco-2, hERG blocking (`herg`, classification), hERG K_i, hERG IC50 (`herg_ic50`, repeat measurements), LogD, MDR1–MDCK efflux, QM9. Reaction: Buchwald–Hartwig. Antibodies: AbCoV K_d, CDR flexibility, ΔΔG (`ddg_combined`), nanobody thermostability (`flab_thermo_tresanco`), AC-SINS, SMAC colloidal stability, PR-CHO, expression titer. Results also exist for EGFR binding (`abagym_EGFR`) and antibody humanness. Thesis Ch7 Methods says it leaves both out.
- **Models.** Each dataset uses two downstream models, set per dataset (for example DNN and SVM on Caco-2, LightGBM and RF on Buchwald–Hartwig, DNN and LightGBM on QM9).
- **Folds.** Outer folds are the scaffold-group-aware folds of the hybrid search (`noise_outer_folds`). Most many-method experiments are a one-fold screen: fold 0 only (`one_fold_screen_config`). "Descriptive" experiments (c10, n01, n14, n15, n16, n26, n27, n37) run the first K folds where set. A "promoted" run (`KIRBY_NOISE_PROMOTE`, stored in `noise_full/`) adds folds 1 and 2 and the full level grid. Promoted runs exist for n07, n20, n23, n24, n25, n30 and the noise-ceiling folder.
- **Which representations.** Within each outer fold a clean-label search picks, per model, two single representations (`base_1`, `base_2`) and two hybrids (`hybrid`, `hybrid__general2`). Only the hybrid's feature ranking and budget split are redone on damaged labels. The thesis states this makes reported scores possibly too favourable (`sec:noise:methods`).

### How label noise is injected (code: `noise_helpers.py`)

- **One entry point.** `inject_condition(data, dataset, condition, level, draw, train_idx=...)` damages one fold's training labels. It returns the damaged labels, a mask of which training molecules were damaged, and a provenance record. Held-out labels are never damaged.
- **Level 0.** `inject_condition` returns the clean training labels unchanged and an all-False mask when the level is 0. The note it writes is "level 0: the labels as delivered". So level 0 means the labels as published, with whatever assay error they already carry. It is the no-injection baseline, not a clean-label oracle. Only on QM9 are level-0 labels free of label noise.
- **Six regression kinds** (`DECIDED_REGRESSION_CONDITIONS`):
  - `gaussian`: every label gets Gaussian error.
  - `outlier_p05`: as Gaussian, but a random 5% of molecules get an error three times wider.
  - `grouped_shifted`: every Bemis–Murcko scaffold family gets its own offset (small molecules only; needs SMILES).
  - `censoring`: labels above the (1 − level) quantile of the clean training labels are set to that quantile.
  - `shift_only_p10`: a random 10% of molecules are raised by one fixed amount.
  - `sparse_dprime_up_p10`: the same shift, sized in standard deviations of LightGBM's out-of-fold residual on clean labels.
- **Units of the level.** For Gaussian, outlier and grouped-shifted the level is a fraction of the clean training-label standard deviation. For censoring it is the share of training labels clipped. For `shift_only_p10` it is the shift in clean training-label standard deviations. Noise size is matched across kinds by Chapter 2's Equation `eq:dose_matching` in the NoiseInject package. A delivered size outside NoiseInject's tolerance raises an error (`DoseError`).
- **Levels.** The screen runs level 0 and 1.0 (censoring 0 and 0.25) and drops `sparse_dprime_up_p10`. A promoted run on a dataset of 5,000 molecules or fewer runs 0, 0.5, 1.0 and 1.5 (censoring 0, 0.10, 0.25, 0.40), all six kinds. The thesis gives these in `tab:noise:levels`.
- **Classification.** Datasets such as hERG blocking get label flips instead, with six flip models (`CLASSIFICATION_STRATEGIES`: uniform, class-dependent, class-imbalance, binary-asymmetric, instance-dependent, confusion-directed). The screen uses flip rates 0 and 0.10.
- **Seeds.** One seed per (condition, draw) from `noise_draw_seed`. The level and the fold are not in the seed. So the damage at level 1.0 is the damage at 0.5 scaled by two, molecule by molecule. A molecule keeps the same draw in every fold it trains in. Draws 0, 1, 2 are the replicate axis.
- **Which kinds a detector can be graded on** (`DETECTION_IS_DEFINED`). Only kinds that damage some molecules and not others: outlier, censoring, the two 10% shifts, and flips. Gaussian and grouped-shifted move every label, so there is no "damaged set" to find.

### Shared machinery

- **`controls.py`.** Builds two control versions of a cleanup's output. `random_twin` applies the same action (delete, down-weight or move) to the same number of randomly chosen training molecules. `shuffled_twin` permutes the method's per-molecule corrections across all training molecules. Both are off unless `KIRBY_NOISE_CONTROLS=1`, which is set for promoted runs only. n18's own control differs: it shuffles the detector's suspicion scores and reruns the cleanup. n24 deletes at random at five seeds. n25 puts back random non-cliff molecules.
- **`_ground_truth_providers.py`.** Used by n06 only. It supplies real disagreement between repeat measurements as ground truth instead of injected noise. Three providers are written: repeat drug–target pKd on BindingDB K_d (spread of the repeats), four published hERG curators whose blocker calls differ (`cross_curator`, on herg and herg_ki), and one ChEMBL compound measured in two papers (`chembl_multi_document`). It refuses a dataset with fewer than 50 repeat groups.
- **`similarity_space.py`.** Defines where a neighbour-based method looks for a molecule's neighbours, as an axis separate from the model's own features. A setting is a feature space plus a distance. Spaces: `base_1`, `base_2`, each with augmented features appended, the four hybrid spaces (`hybrid`, `hybrid__general2`, `hybrid_similarity`, `hybrid__distance1`), 2048-bit ECFP4 under Tanimoto, and forest proximity (one minus the share of trees in which two molecules share a leaf). Distances are Manhattan, cosine or Minkowski p = 0.5. Datasets of 5,000 molecules or fewer sweep 26 settings; larger ones sweep 13. Neighbour counts 5, 10 and 20, weighted by 1/rank or 1/d², are read off each setting. The setting is chosen by predicting a held-back 20% of training scaffold families from their neighbours (thesis `sec:noise:detection`).

### How methods are grouped

The thesis sorts every method into three groups (`tab:noise_terms`, `tab:noise:tests`):

1. **Noise detection.** A detector gives each training molecule a suspicion score. It is graded by average precision (or AUROC) against the damaged-molecule mask. Detectors (`tab:noise:detectors`): Forest residual, LightGBM residual, Neighbor-label gap, Gap over neighbor spread, Cliff-aware gap, GP spread and residual, Heteroscedastic GP spread, Variance-head network, NGBoost spread. Code names: `cross_val`, `neighbor_residual`, `local_variance_ratio`, `activity_cliff_aware`, `gaussian_process`, `hetero_gp`, `heteroscedastic_nn`, `ngboost` in `src/kirby/utils/noise_estimation.py`.
2. **Noise mitigation.** Molecules are selected, then acted on. The selector is a detector, a known flag (label at an assay limit, scaffold family, measuring source, repeat count), or every molecule. The action is delete, count less (weight), or move the label toward a target (`tab:noise:mitigation`).
3. **Noise-robust training.** Nothing is selected; the model or loss changes. This covers the twelve LNL methods of n09/n13 (eight of them; the four deleting ones count as mitigation), robust losses (n33), co-teaching (n32), model-side changes (n40), and repeat-measurement inputs (n19).

`/Volumes/seagate/claude_analysis_2026-10-08/answers/_scripts/method_groups.json` assigns each method a selector (detector, flag, none) and an action (delete, move, count less, none). It is a Claude reading of the code dated 2026-10-09, with some entries marked unconfirmed. `NOISE_BY_GROUP.md` regroups the result rows by it. The thesis Additional file `app:add:noise_bygroup` holds the per-group counts.

Each group has its own test (`tab:noise:tests`). Mitigation is compared with the same model and features with no action, plus a random or shuffled control where one ran. Robust training is compared with the same model trained normally.

### QM9 as the clean-label control

QM9 HOMO–LUMO gap labels are computed, not measured, so they carry no assay error. A method that improves QM9 at level 0 is changing the model, not removing noise. The thesis applies this in `sec:noise:res:qm9test`. `/Volumes/seagate/claude_analysis_2026-10-08/answers/QM9_CONTROL.md` lists every QM9 level-0 result (RMSE, fold 0, best setting picked on the held-out score):

- Deletion made QM9 worse in every configuration. Example, n07 Forest-residual deletion: DNN hybrid 0.3618 → 0.3878 RMSE; LightGBM hybrid 0.3719 → 0.4106.
- Neighbour-mean smoothing and weighting improved the DNN on QM9. Example, n07 neighbour-mean smoothing: DNN hybrid 0.3568 → 0.3395 RMSE. n24 down-weighting: DNN hybrid 0.3491 → 0.3428. LightGBM barely moved (hybrid 0.3759 → 0.3715).
- Random deletion of the same number of molecules moved the DNN hybrid from 0.3568 to 0.3440 RMSE at its best setting. So the DNN's own fit-to-fit spread on QM9 is about 0.01 RMSE.
- The DNN's plain second hybrid on QM9 scored 0.362 RMSE in one experiment and 0.364 in another, same fold and labels (thesis, `sec:noise:res:qm9test` caveats).

### Cleaning often helps at level 0: read damaged-level gains next to it

This pattern holds. Many cleanups improve the held-out score on the labels as delivered (level 0), with no noise injected. On some datasets the level-0 gain is as large as or larger than the gain at level 1.0. A gain at a damaged level then cannot be read as noise removed unless it exceeds the level-0 gain.

Checked here from the 10-10b n07 rows files (`n07_remediation/<dataset>__gaussian__fold_aligned_2026-10-06.rows.jsonl`), first hybrid, fold 0, draw 0, best setting per method picked on the held-out score:

| dataset, model, method | level 0: none → cleaned | Gaussian level 1.0: none → cleaned |
|---|---|---|
| SMAC colloidal, DNN, delete flagged (Spearman) | 0.2267 → 0.4122 | 0.0467 → 0.0944 |
| Expression titer, DNN, delete flagged (Spearman) | 0.1550 → 0.3743 | 0.0816 → 0.1401 |
| PR-CHO, GP, delete flagged (Spearman) | 0.4710 → 0.5304 | 0.3725 → 0.4854 |
| Buchwald–Hartwig, LightGBM, delete flagged (RMSE, yield %) | 20.0823 → 19.3820 | 21.1942 → 19.9667 |
| hERG K_i, DNN, neighbour-mean smoothing (RMSE) | 0.5972 → 0.5671 | 0.7026 → 0.6894 |
| hERG K_i, DNN, delete flagged (RMSE) | 0.5972 → 0.6204 | 0.7026 → 0.6506 |
| Caco-2, DNN, delete flagged (RMSE) | 0.2333 → 0.2499 | 0.2984 → 0.3048 |
| QM9, DNN, neighbour-mean smoothing (RMSE) | 0.3550 → 0.3484 | 0.5090 → 0.5116 |

- On the three antibody sets and Buchwald–Hartwig above, the level-0 gain is the larger one or about the same. On QM9 the DNN gains at level 0, where there is no noise.
- hERG K_i deletion is the counter-case: it hurts at level 0 and helps at level 1.0.
- On Caco-2 and LogD, deletion hurts at both levels.
- The thesis reads the antibody level-0 deletion gains as noise already in the data, because deletion makes QM9 worse and beats shuffled controls there (`sec:noise:res:qm9test`).
- Caveat (FAULTS.md A4, re-checked there): the same hybrid on the same fold and clean labels scores differently in different files. Example: Caco-2 DNN hybrid 0.2333 RMSE in `caco2__censoring.json` against 0.2474 in `caco2__grouped_shifted.json`. Gains smaller than about 0.014 RMSE on Caco-2 DNN hybrids are inside that spread.
- Caveat (FAULTS.md A5): at a damaged level, a hybrid is re-picked from the damaged labels. So its clean-label score moves with the level, though single representations do not.

### Where each method came from (`thesis_provenance/METHOD_PROVENANCE.md`)

That file sorts each method into four buckets. (a) A faithful implementation of a published method. (b) A published method with a stated change. (c) No published precedent found. (d) Provenance unclear: an AI-written report, a ported code base, or a citation that does not describe the code. Its header says it was written by Claude on 2026-10-08 and "not reviewed by the author"; the buckets are judgements, not measured results. Literature searches behind it are in `thesis_provenance/LITERATURE_SEARCH_2026-10-08.md`.

| method (thesis name / code name) | experiment | bucket |
|---|---|---|
| Neighbor-label gap / `neighbor_residual` | n07, n12, n18, n21 | b |
| Gap over neighbor spread / `local_variance_ratio` | n12, n13, n21 | b (weather-station buddy check; not on molecular labels before) |
| Cliff-aware gap / `activity_cliff_aware` | n07, n02 | c (new) |
| Forest residual / `cross_val` | n07, n18, n20, n24 | b |
| GP spread and residual / `gaussian_process` | n23 | b |
| Heteroscedastic GP spread / `hetero_gp` | n23 | d |
| Variance-head network / `heteroscedastic_nn` | n23 | a |
| NGBoost spread / `ngboost` | n23 | a |
| Delete flagged / `remove` | n07, n18, n24 | b |
| Score down-weight / `downweight` | n07, n24 | c |
| Huber weight / `huber_weight` | n07, n24 | b, d |
| Neighbor-mean smoothing / `label_smooth` | n07, n18, n24 | b |
| Replace by neighbours' mean / `knn_smooth` | n07, n18 | b |
| Forest-proximity smoothing / `prox_smooth` | n07, n20 | c |
| Move toward prediction / `oof_smooth` | n07, n18 | b, d |
| Score-scaled shrink / `js_shrink` | n07 | d |
| GP smoothing / `gp_smooth` | n07, n18 | b |
| Constant lift / `constant_lift` | n07, n30 | b |
| Censored EM, Truncated-normal mean / `censored_em` | n30 | b |
| Reliability and cliff-aware weights | n02 | c |
| Importance-weighted distance | n12, n21 | b |
| Twelve LNL methods | n09, n13 | a: evidential network; b: small-loss, confident learning, distance selection, uncertainty curriculum, DivideMix, Meta-Weight-Net, MentorNet; d: multistage cleaning, early learning, contrast-to-divide, heteroscedastic GP |
| Count, spread and precision weights | n19 | a |
| Shrink by spread / `shrink_toward` | n19 | b |
| Noise variance from 1/count regression | n19, n26 | d → c |
| Clean before search | n20 | c (new) |
| Cliff put-back / cliff exemption | n25 | c (new) |
| Replicate noise diagnostics (ICC, split-half) | n26 | a |
| Per-document offsets, within-document rank | n29 | b, d |
| Nearest-neighbour Shapley, trimmed loss, two-learner exchange (co-teaching) | n32 | b; exchange b, d; shrink-flagged d |
| Robust losses (Huber, fair, L1, Tukey) | n33 | a / b |
| Random intercept per scaffold family | n34 | b |
| Close-pair noise estimate and weights | n35 | b, d; weights d |
| Re-measure priority | n36 | b, d |
| Gamma test, SIMEX | n37 | a; SIMEX b |
| Detection-theory formulas | n38 | d |
| Source offsets and ρ-aware weights | n39 | b, d |
| Retuning, seed averaging, stacking, GP nugget | n40 | b, d |
| Conformal p-values with Benjamini–Hochberg | n41 | b; trusted calibration c |

The thesis (Chapter 1, `chap:intro`) names three methods as new: the Cliff-aware gap, cleaning before the feature search, and Cliff put-back. It calls the Gap over neighbor spread, censored EM and the random intercept per scaffold family adapted from published methods.

**Result copies named in the entries.** `10-10b` = /Volumes/seagate/kirby_results_2026-10-10b (the copy thesis Ch7 SOURCE comments cite); `10-09` = /Volumes/seagate/kirby_results_2026-10-09; `view` = /Volumes/seagate/kirby_results_2026-10-09_noise_view (files dated 10 Oct). "Checked" means read from the raw file in this session. "Configuration" means one of four feature inputs chosen per outer fold from a search on that fold's training molecules: the two best single representations (first single, second single) and the two best general hybrids (first hybrid, second hybrid). Each dataset has two decided models.

## Experiments

### n01 — Hybrid against single representations on activity-cliff molecules  (`tests/experiments/noise/n01_activity_cliff_framework.py`)
- **Question:** Does the best hybrid predict activity-cliff molecules better than the best single representations? Is its error gap between cliff and non-cliff molecules smaller?
- **How it works:**
  - Clean labels; no noise is injected.
  - Per outer fold, a search on that fold's training molecules picks, for each of the dataset's two decided models, the 2 best single representations and the 2 best hybrids (8 fitted configurations per fold).
  - Reference cliffs come from a space outside the graded features: ECFP4 Tanimoto for small molecules, heavy-chain edit distance for antibodies, reactant and product Morgan bits for reactions.
  - Neighbour cliffs are a second definition: k = 5, 10, 20 training neighbours under manhattan, cosine and minkowski p = 0.5 distances.
  - Held-out RMSE or R² (AUROC for classification) is computed separately on cliff and non-cliff held-out molecules.
  - For hERG IC50, a molecule's own repeat measurements are left out of its reference neighbours.
- **Status:** Ran on server. Result files in the 2026-10-10b copy exist for adme_mdr1, buchwald_hartwig, caco2, cdr_flexibility, ddg_combined, ginkgo_acsins, ginkgo_colloidal, ginkgo_pr_cho, ginkgo_titer, herg, herg_ki, logd and qm9. Files also exist for abagym_EGFR and humanness_therapeutic, which Ch7 excludes. Several folds per dataset; the answer page notes cdr_flexibility cliff scores are means over 5 folds and non-cliff over 4. Every file predates the 2026-10-08 cap on DNN predictions.
- **Results so far:** Answer page (2026-10-08 copy, not re-checked): on caco2, herg, herg_ki and logd the best hybrid beat both single representations on cliff molecules, for both models and both cliff definitions. Example: caco2 DNN, 0.3320 to 0.2966 RMSE on reference cliff molecules. On cdr_flexibility and ginkgo_colloidal it did not; ginkgo_pr_cho and ginkgo_titer split by model. The cliff-to-non-cliff gap moves in no single direction.
- **In thesis:** Not in thesis (no SOURCE comment cites it).
- **Related:** n14 (cliff counts), n25 (cliff put-back), c10 (same configurations scored at zero damage).

### n02 — Down-weighting molecules that disagree with their neighbours, on labels as delivered  (`tests/experiments/noise/n02_sample_weighting.py`)
- **Question:** Does weighting each training molecule by how well its label agrees with its neighbours' labels help a model on undamaged labels? Does it help single representations or hybrids more?
- **How it works:**
  - Labels as delivered only (level 0); no damage injected.
  - Five conditions: single representation unweighted or weighted; hybrid unweighted, weighted, or weighted with cliff-aware weights.
  - Neighbours are found in an importance-weighted distance; feature importance comes from the dataset's decided importance methods.
  - 10 neighbours, averaged with a rank or inverse-square counting scheme; a further sweep varies k and distance.
  - Per fold, the dataset's two decided models on their 2 best single representations and 2 best hybrids.
  - Held-out score in the dataset's metric; weighted against unweighted on the same fold.
- **Status:** Ran on server, fold 0 of 5 only. Files in 10-10b for abcov_kd, adme_mdr1, buchwald_hartwig, caco2, cdr_flexibility, ddg_combined, flab_thermo_tresanco, ginkgo_acsins, ginkgo_colloidal, ginkgo_pr_cho, ginkgo_titer, herg, herg_ki, logd, qm9. Files also exist for abagym_EGFR and humanness_therapeutic (excluded from Ch7). Open fault (FAULTS.md): the neighbour-setting sweep passed sample weights to the GP without checking, which stopped ginkgo_pr_cho on 2026-10-07; fixed in commit 069fa32, re-run status not checked.
- **Results so far:** Thesis Table tab:noise:res:smac_weights (SMAC colloidal stability = ginkgo_colloidal, DNN, fold 0, Spearman): first single 0.065 to 0.209; first hybrid 0.227 to 0.370. Checked in raw 10-10b rows: 0.0651 to 0.2093 (k = 10, rank, strength 3). The thesis notes this method has no control. Answer page: weighting never helped SVM on caco2, herg_ki or logd.
- **In thesis:** Ch7 sec:noise:res:antibodies, tab:noise:res:smac_weights (ch7_numbers N7-32).
- **Related:** n12 (importance-weighted distance), n13 (detector weights), n21 (neighbour weights under damage).

### n06 — Do noise detectors agree with real disagreement between measurements?  (`tests/experiments/noise/n06_bindingdb_ground_truth.py`)
- **Question:** Do the detectors that rank best under injected noise also rank best against real disagreement between repeat measurements of the same molecule? Do they agree better in a hybrid than in a single representation?
- **How it works:**
  - Ground truth comes from `_ground_truth_providers.py`: drug-target repeats (BindingDB), cross-curator hERG disagreement (Lhasa against ChEMBL), and ChEMBL records of one compound in several papers.
  - A dataset with no repeat structure is recorded as skipped.
  - Per outer fold, 8 spaces: each decided model's 2 best single representations and 2 best hybrids, chosen on the other folds' rows.
  - Estimators: neighbor_residual, local_variance_ratio, distance_weighted (k = 5, 10, 20; several distances; two counting schemes) and a cross-validated model residual.
  - Score: Spearman between each molecule's noise score and the spread of its group's repeat measurements. No downstream model is fitted.
  - Headline: rank correlation between the detector ordering here and the orderings from n11 and n13.
- **Status:** Ran in the 2026-10-10b copy only (absent from 10-09). Only herg has scores: fold 0 of 5, 426 compounds with IC50 values in more than one ChEMBL paper (target CHEMBL240). The other datasets are skipped because no provider found repeats. The cross-curator provider was not used because the Lhasa file was missing on the server. The transfer comparison was refused: no n11 or n13 herg result existed, and only 4 estimators run on classification, below the 5 a rank correlation needs.
- **Results so far:** None summarised. Per-molecule scores exist in `herg_per_row.npz` and were not read here.
- **In thesis:** Not in thesis.
- **Related:** n11, n13 (orderings it compares against), n19, n26 (repeat measurements).

### n07 — Cleaning damaged training labels before the model is fitted  (`tests/experiments/noise/n07_remediation.py`)
- **Question:** Once a detector flags suspect molecules, which action recovers the most held-out performance: delete, down-weight, or move the label?
- **How it works:**
  - Noise injected on training labels only, through `noise_helpers.inject_condition`, at the decided kinds and levels; held-out labels clean. Three draws per non-zero level, one under censoring.
  - Actions: remove, downweight, label_smooth, knn_smooth (move toward the 10 nearest neighbours' mean), oof_smooth (toward an out-of-fold LightGBM prediction), gp_smooth, huber_weight, js_shrink, prox_smooth, invvar_weight, constant_lift.
  - Remove flags a fixed share of training molecules (5, 10 or 20%), never the injected share.
  - Detection is graded by average precision of the damaged molecules for the cross-validated forest residual and the cliff-aware neighbour gap.
  - Per fold, the dataset's two decided models on 2 single representations and 2 hybrids. Baselines: no noise, and noisy with no cleaning.
  - gp_smooth's target was a constant per inner fold until 2026-09-21; earlier results for that action are shrinkage to a mean.
- **Status:** Ran on server, fold 0 for most datasets. Files in 10-10b for adme_mdr1, buchwald_hartwig, caco2, cdr_flexibility, ddg_combined, ginkgo_acsins, ginkgo_colloidal, ginkgo_pr_cho, ginkgo_titer, herg, herg_ki, logd, qm9 (plus humanness_therapeutic, excluded from Ch7). noise_full holds extra caco2 censoring folds. Open faults: DNN fits predate the prediction cap. The same recipe on the same clean labels scores differently between files, for example caco2 DNN hybrid 0.2333 against 0.2474 RMSE.
- **Results so far:** hERG blocking (herg), 10% confusion-matrix flips, XGBoost, fold 0, AUROC: first hybrid 0.803 to 0.836 after deleting the 10% the forest residual flags; first single 0.806 to 0.8253 (tab:noise:res:herg_flip_actions; checked in raw rows: 0.8028 to 0.8356). At level 0 on the same file, deleting 10% made the hybrid worse, 0.8534 to 0.8433; down-weighting the first single gave 0.8369 to 0.8403 (checked). QM9 with no noise added, DNN first hybrid: 0.355 to 0.393 RMSE after deletion (tab:noise:res:qm9_deletion). Detection, caco2 outliers: forest residual average precision 0.338 against chance 0.051 (tab:noise:res:detect_residual).
- **In thesis:** Ch7 sec:noise:res:action (tab:noise:res:herg_flip_actions, tab:noise:res:delete_vs_random), sec:noise:res:qm9test region (tab:noise:res:qm9_deletion), sec:noise:res:detectors (tab:noise:res:detect_residual). ch7_numbers N7-03, N7-08, N7-13, N7-14.
- **Related:** n23, n24 (delete or keep), n11 (detector by action grid), n18 (same actions on labels as delivered), n30 (constant_lift origin).

### n09 — Methods for learning with noisy labels, trained on hybrids and single representations  (`tests/experiments/noise/n09_lnl_survey.py`)
- **Question:** Does the ranking of published methods for learning with noisy labels change when the network is trained on a hybrid instead of a single representation? Do any beat the plain network?
- **How it works:**
  - One model, the DNN, because every method is a neural-network training scheme.
  - Twelve methods from `lnl_methods.py`: small-loss selection, confident learning, distance-based selection, multistage cleaning, early learning, uncertainty curriculum, DivideMix, Meta-Weight-Net, MentorNet, contrast-to-divide, heteroscedastic GP and evidential network (last two regression only).
  - Feature spaces per fold: the DNN's 2 best single representations, 2 best hybrids, and an importance-weighted copy of the first hybrid.
  - Decided damage kinds and levels; 3 draws per non-zero level; level 0 is the labels as delivered.
  - Each method is compared with the plain DNN on the same labels and configuration.
- **Status:** Ran on server, fold 0 only. Files in 10-10b for adme_mdr1, buchwald_hartwig, caco2, cdr_flexibility, ddg_combined, ginkgo_acsins, ginkgo_colloidal, ginkgo_pr_cho, ginkgo_titer, herg, herg_ki, logd, qm9 (plus humanness_therapeutic, excluded from Ch7). Open faults: on 2026-10-08, 30 of 44 files had run the search once on clean labels, not per draw as the docstring says. The heteroscedastic GP's kernel starts at length scale 1, which left other GP fits stuck; not checked here. The methods' own networks have no prediction cap.
- **Results so far:** QM9 with no noise added (level 0), fold 0: every method made the DNN worse; first hybrid plain 0.355 against 0.3677 to 0.452 RMSE (tab:noise:res:qm9_sweep). ddg_combined labels as delivered, heteroscedastic GP, first single: 0.341 to 0.433 Spearman. herg, 10% one-way flips, distance-based selection: first hybrid 0.821 to 0.825 AUROC (tab:noise:res:sweep_examples).
- **In thesis:** Ch7 sec:noise:res:sweep (tab:noise:res:sweep, tab:noise:res:sweep_examples, tab:noise:res:qm9_sweep first run). ch7_numbers N7-42 to N7-45.
- **Related:** n13 (second run of the same methods), n33, n40 (changing the model).

### n11 — Every detector crossed with every way of repairing labels  (`tests/experiments/noise/n11_estimation_remediation_grid.py`)
- **Question:** Which pairing of a noise detector and a repair rule recovers the most of the held-out score lost to damaged labels?
- **How it works:**
  - Grid: detector × repair rule (smoothing or weighting) × strength × distance × representation.
  - Repair targets: the detector's own target, an out-of-fold LightGBM prediction, a GP posterior mean, or quantile clipping.
  - Per fold, the dataset's two decided models on 2 single representations and 2 hybrids; hybrids rebuilt from each draw's damaged labels.
  - Decided damage kinds and levels; 3 draws per non-zero level.
  - Recovery = share of the clean-to-damaged score gap won back. Neighbour settings screened on level 0 and reused.
- **Status:** Partial, fold 0 only. Files in 10-10b exist for buchwald_hartwig, cdr_flexibility, ginkgo_acsins, ginkgo_colloidal, ginkgo_pr_cho, ginkgo_titer, herg; the 2026-10-08 answer page read only ginkgo_colloidal and ginkgo_titer under Gaussian noise. The docstring says molecular datasets only, yet most files are antibody sets. Open faults: the stored best recovery is 716% and 1455% because it divides by very small damage (0.0454 and 0.0230 Spearman). The hybrid's level-0 score differs between blocks of one file (ginkgo_titer DNN second hybrid 0.3893 against 0.0983).
- **Results so far:** Answer page (not re-checked): on both ginkgo sets every repair rule beats no repair for both models, at level 0 and under damage, because passing needs only one of hundreds of settings. At level 0, the best repair took ginkgo_colloidal DNN first single from 0.0651 to 0.4858 Spearman, a maximum picked on held-out molecules.
- **In thesis:** Not cited by a SOURCE comment; ch7_numbers mentions it once.
- **Related:** n07, n12, n06.

### n12 — Scaling features by importance before finding neighbours  (`tests/experiments/noise/n12_importance_weighted_distance.py`)
- **Question:** Does multiplying each feature by its importance before computing neighbour distances make neighbour-based detection better? Is that redundant with building a hybrid?
- **How it works:**
  - Importance is computed on the damaged training labels.
  - Detection compared unscaled against importance-scaled, by average precision of the damaged molecules.
  - Cleanup: suspicion weights from the neighbour gap, fitted with the dataset's two decided models on 2 single representations and 2 hybrids.
  - Neighbours also searched in the distance-search hybrid and the similarity-search hybrid, which are never fitted.
  - Rank and inverse-square counting schemes; scaffold-mates kept; decided damage kinds and levels, 3 draws.
- **Status:** Ran on server, fold 0 only. Files in 10-10b for abcov_kd, adme_mdr1, buchwald_hartwig, caco2, cdr_flexibility, ddg_combined, flab_thermo_tresanco, ginkgo_acsins, ginkgo_colloidal, ginkgo_pr_cho, ginkgo_titer, herg, herg_ki, logd, qm9 (plus humanness_therapeutic, excluded). caco2 ran the older level grid (0, 0.5, 1.0, 1.5). flab_thermo_tresanco Gaussian failed: the injector delivered 8.47% less noise than asked. Refit rows at the screen's settings are not in the copy.
- **Results so far:** Shift in 10% (fold 0, average precision, unscaled against scaled): QM9 0.538 to 0.634; buchwald_hartwig 0.465 to 0.512; most other cells move by less than 0.03 (tab:noise:res:importance_scaling). Neighbour weights, own neighbours against hybrid's neighbours, logd scaffold-family shift, DNN first single: none 0.705, own 0.753, hybrid's 0.663 RMSE (tab:noise:res:neighbor_space). Answer page: suspicion weights also beat no weighting at level 0 for the DNN on caco2 and logd.
- **In thesis:** Ch7 sec:noise:res:detectors (tab:noise:res:importance_scaling, N7-16); sec:noise:res:repr (tab:noise:res:neighbor_space, N7-47).
- **Related:** n02, n21 (neighbour distance families), n11.

### n13 — Methods under different kinds of label damage  (`tests/experiments/noise/n13_noise_distributions.py`)
- **Question:** How do method rankings and the hybrid's lead move when the kind of injected damage changes?
- **How it works:**
  - Damage kinds: gaussian, outlier_p05, grouped_shifted, censoring, shift_only_p10, sparse_dprime_up_p10 on regression; six flip kinds on hERG blocking.
  - Level 0 run once per fold and reported under every kind; 3 draws per non-zero level.
  - Track A: the 12 methods of n09, each training its own DNN, against the plain DNN on 4 configurations.
  - Track B: detectors (local_variance_ratio, GP, NGBoost) turn scores into sample weights for the dataset's two decided models.
  - Fold-chosen 2 single representations and 2 hybrids per model.
- **Status:** Ran on server, fold 0 only. Files in 10-10b for adme_mdr1, buchwald_hartwig, caco2, cdr_flexibility, ddg_combined, ginkgo_colloidal, ginkgo_pr_cho, ginkgo_titer, herg, herg_ki, logd, qm9 (plus humanness_therapeutic, excluded). Open faults: the plain DNN in Track A and the unweighted DNN in Track B disagree on the same labels (herg_ki Gaussian 1.0, base_2: 0.6946 against 0.7349 RMSE). distance_selection stores no draw, so only the last draw survives.
- **Results so far:** herg, 10% confusion-matrix flips, XGBoost, local_variance_ratio weights: first hybrid 0.803 to 0.815 AUROC; first single 0.8060 to 0.8063 (tab:noise:res:weighting). QM9 level 0 (no noise), detector weights: DNN second hybrid 0.362 untreated against 0.3488 weighted; LightGBM first hybrid 0.372 against 0.3724. QM9 level 0 Track A: about half the methods beat the plain DNN on the second hybrid, for example MentorNet 0.343 against 0.364 RMSE (tab:noise:res:qm9_sweep second run).
- **In thesis:** Ch7 tab:noise:res:weighting (N7-07), tab:noise:res:qm9_sweep second run (N7-44).
- **Related:** n09 (same methods, first run), n21, n35.

### n14 — Counting activity cliffs and residual spread before any cleaning  (`tests/experiments/noise/n14_activity_cliff_landscape.py`)
- **Question:** How many activity cliffs does each small-molecule regression set have, and are they clustered or isolated? Is the label's residual spread above the published assay-noise level of 0.54 pKi?
- **How it works:**
  - A cliff pair: structurally similar (matched pairs where available, else ECFP4 Tanimoto) and labels more than 2 log units apart.
  - Counts cliff molecules and pairs, and the size of connected groups in the cliff graph.
  - Residual diagnostics from an out-of-fold LightGBM fit: kurtosis, spread against label, residual SD in label units.
  - Also counts cliffs in each neighbour space (k = 5) per fold.
  - Trains no downstream model; it is a gate for later experiments.
- **Status:** Ran on server. Numbers exist for adme_mdr1, caco2, herg_ki, logd and qm9 in 10-10b. Antibody, reaction and classification files hold skip records ("non-molecular domain", "classification out of scope"), marked ok in the ledger. A file exists for abagym_EGFR (excluded from Ch7).
- **Results so far:** Cliff molecules, from raw 10-10b files (checked): herg_ki 119 of 1412 (largest group 40); logd 2403 of 5028 (largest group 820); adme_mdr1 803 (30.4%); qm9 23 (0.23%). caco2 has none, because its labels span 1.9976 units, under the 2-unit gap. herg_ki out-of-fold residual SD 0.5870 pKi against 0.54 published (answer page, not re-checked).
- **In thesis:** Not in thesis.
- **Related:** n01, n25 (cliff put-back), n36.

### n15 — Hybrid chosen on clean labels against hybrid chosen on damaged labels  (`tests/experiments/noise/n15_clean_vs_noisy_hybrid.py`)
- **Question:** Other noise experiments choose their configurations on clean labels; how much does that head start help? Would a configuration searched on the damaged labels do as well?
- **How it works:**
  - Two per-fold searches on the fold's training molecules: one on clean labels, one on Gaussian-damaged labels of one draw.
  - Each picks, per decided model, 2 single representations and 2 hybrids.
  - Both arms are then fitted on the damaged labels and scored on clean held-out labels.
  - Head start = clean-arm score minus damaged-arm score, per configuration.
  - Gaussian noise at level 1.0 (caco2 also 0.5 and 1.5); 3 draws.
- **Status:** Ran on server. 10-10b holds buchwald_hartwig, caco2, cdr_flexibility, ddg_combined, ginkgo_colloidal, herg, herg_ki, logd, qm9 (plus humanness_therapeutic, excluded). caco2 ran 5 folds (checked); logd, herg_ki and buchwald_hartwig ran 3; others fewer, per thesis table.
- **Results so far:** Second hybrid, held-out RMSE, clean-searched against damaged-searched (tab:noise:res:search_clean_vs_damaged): caco2 DNN fold 0 0.298 against 0.303; herg_ki DNN fold 1 0.669 against 0.779, the largest loss. On SMAC and ddg_combined the two searches swap order between folds. QM9 DNN fold 0 0.544 against 0.553.
- **In thesis:** Ch7 sec:noise:res:search, tab:noise:res:search_clean_vs_damaged (ch7_numbers N7-37).
- **Related:** n20 (cleaning before the search), n11, n12 (configurations chosen on clean labels).

### n16 — Does the ranking of features move when labels are damaged  (`tests/experiments/noise/n16_importance_under_noise.py`)
- **Question:** When training labels are damaged, does the per-feature importance ranking change more than re-scoring clean labels with a second scorer seed changes it? When a hybrid is rebuilt on damaged labels, what share of its clean-label features does it keep?
- **How it works:**
  - Trains no prediction model and cleans nothing; it records importance vectors only.
  - Per outer fold, per decided model: its 2 best single representations and 2 best hybrids, chosen by a search on that fold's training rows (8 configurations).
  - Importance methods: `lightgbm_gain`, `treeshap`, `deeplift`. Regression damage kinds only, at level 0 and the screen level (1.0; 0.25 for censoring), 3 noise draws.
  - Outputs per cell: number of features with non-zero importance, Spearman and Kendall rank agreement against level 0, top-100 overlap, Nogueira set stability.
  - Reference ("floor"): level 0 scored again with another scorer seed. Hybrids are rebuilt at each level on that level's damaged training labels.
- **Status:** Ran on server, 5 folds, on buchwald_hartwig, caco2, ddg_combined, ginkgo_colloidal, ginkgo_titer, herg_ki, logd, qm9, adme_mdr1 (10-10b copy). A file also exists for abagym_EGFR, which the thesis excludes from Ch7. Skipped on herg and cdr_flexibility (class labels).
- **Results so far:** From the answer page (10-08 copy, not re-checked): damaged labels moved `lightgbm_gain` and `treeshap` rankings more than the second seed. The `lightgbm_gain` floor is 1.0000, so any change counts as movement. A hybrid rebuilt on herg_ki labels with Gaussian noise at 1.0 kept a median 0.3905 share of its clean-label features.
- **In thesis:** not in thesis.
- **Related:** n12 (computes the same importance vectors to weight distances), n15 (which representations the search picks on clean vs noisy labels), n21.

### n17 — Cleaning against real disagreement between laboratories  (`tests/experiments/noise/n17_replicate_removal.py`; folder `n17_replicate_removal/` holds only an empty local `toy_replicates` output folder)
- **Question:** Does cleaning help when the noise is real disagreement between repeat measurements, with nothing injected?
- **How it works:**
  - One row per molecule (entity, built by `_ground_truth_providers.build_replicate_groups`). Training label: one of its measurements, chosen at random. Grading label: the mean of its other measurements.
  - Only held-out molecules with 3 or more measurements are graded; molecules with fewer still train.
  - Arms: no cleaning; n07's seven corrections (`remove`, `downweight`, `label_smooth`, `knn_smooth`, `oof_smooth`, `gp_smooth`, `constant_lift`) on the 10% the neighbour-residual detector (k=10) flags.
  - Reference arms: `ceiling` removes the molecules whose training label is furthest from the grading label (uses the answer); `floor_measured`/`floor_any` remove the same number at random.
  - Models svm and dnn on 2 best single representations and 2 best hybrids per fold; RMSE; 3 draws of which measurement trains.
- **Status:** Ran on server, fold 0 only, 3 draws, herg_ic50 only (10-09 and 10-10b copies identical). Skipped on every other dataset: no repeat measurements. The 10-08 answer page predates this and says nothing ran.
- **Results so far:** Raw file checked (`n17_replicate_removal/herg_ic50.json` summary). 41 held-out molecules were graded. No cleaning, svm hybrid: 0.7464 RMSE; `label_smooth` 0.7521, `remove` 0.8412. dnn second hybrid: no cleaning 0.6952 vs `label_smooth` 0.6691. The `ceiling` arm was worse than no cleaning in all 8 model-configuration pairs (e.g. svm hybrid 0.8072).
- **In thesis:** not in thesis.
- **Related:** n19, n26, n29, n36 (same repeat-measurement split, same dataset), n07 (source of the corrections).

### n18 — Cleaning the labels as delivered, with a shuffled-score control  (`tests/experiments/noise/n18_clean_the_real_labels.py`; folder `n18_clean_the_real_labels/` holds empty local output folders logd, qm9, toy_clean, toy_wrong)
- **Question:** Does a cleanup improve the held-out score on the labels as they arrive, with no noise injected? Is the gain larger than the same cleanup driven by shuffled detector scores?
- **How it works:**
  - Labels as delivered (`as_they_come`); held-out labels are not cleaned.
  - Detectors: `neighbor_residual` (k=10, "Neighbor-label gap" in thesis) and `cross_val` ("Forest residual").
  - Cleanups: `remove`, `downweight`, `label_smooth`, `knn_smooth`, plus regression-only `oof_smooth`, `gp_smooth`, `constant_lift`. Flag rates 1%, 5%, 20%.
  - Control: the same cleanup with the detector scores shuffled across training molecules, seeds 0, 1, 2.
  - Per fold, the dataset's two decided models on 2 best single representations and 2 best hybrids (8 configurations). QM9 is run as a clean-label control.
- **Status:** Ran on server, fold 0, on 16 datasets in the 10-10b copy (incl. qm9 and adme_mdr1). Results also exist for abagym_EGFR and humanness_therapeutic, which the thesis excludes from Ch7. Known faults (FAULTS.md A19, not re-checked): `gp_smooth` refuses above 1,000 training molecules and stores no refusal; `constant_lift` has no score on herg_ki.
- **Results so far:** Thesis (checked against 10-10b with `ch7_numbers.py`, no differences): herg_ki DNN first single, Neighbor-mean smoothing, 0.638 → 0.578 RMSE; shuffles 0.642–0.645. SMAC (ginkgo_colloidal) SVM first single, delete 20%: 0.274 → 0.403 Spearman, shuffles 0.221–0.278.
  - Level-0 check: QM9 (computed labels) also moved. DNN first single smoothing 0.3999 → 0.3927 eV RMSE; delete 20% made it worse, 0.400 → 0.461.
- **In thesis:** Ch7 \label{sec:noise:res:delivered}, \label{tab:noise:res:shuffled}, \label{sec:noise:res:antibodies}, QM9 rows used in \label{sec:noise:res:qm9test} (N7-09, N7-30, N7-31, N7-50, N7-51).
- **Related:** n07 (same cleanups on injected noise), n24 (labels as delivered with random-deletion control), n20, n25.

### n19 — What a measurement count and a measurement spread are worth  (`tests/experiments/noise/n19_measurement_count.py`)
- **Question:** On data with repeat measurements, does using each molecule's measurement count or spread improve the held-out score?
- **How it works:**
  - Reuses n17's split: one row per molecule; held-out molecules (3+ measurements) graded against the mean of measurements not used.
  - Two training labels: `single` (one random measurement) and `averaged` (mean of all its measurements).
  - Conditions: untouched; `count_weighted`, `spread_weighted`, `precision_weighted`; `shrunk` and `calibrated` (label moved toward an out-of-fold prediction by its own variance); `invvar_weighted`; `count_feature` (log count as an extra feature).
  - Model pairs: NGBoost with and without the count feature; XGBoost accelerated-failure-time point model against interval model.
  - svm and dnn on 2 best single representations and 2 best hybrids; RMSE; 3 draws.
- **Status:** Ran on server, fold 0, 3 draws, herg_ic50 only. Skipped elsewhere (no repeat measurements). Known fault (FAULTS.md, not re-checked): the fitted variance-vs-count curve has a negative slope in all 48 fits, floored to zero. So `invvar_weighted` gives every molecule the same weight and equals untouched within 0.0001 RMSE.
- **Results so far:** Thesis, averaged label, first hybrid, fold 0, three draws (checked with `ch7_numbers.py` N7-34): SVM no weighting 0.760/0.766/0.761 RMSE vs count weighting 0.728/0.722/0.707. DNN no weighting 0.712/0.701/0.699 vs log-count feature 0.656/0.665/0.632. Count weighting made the DNN worse (0.765/0.739/0.716).
- **In thesis:** Ch7 \label{sec:noise:res:herg_ic50}, \label{tab:noise:res:herg_ic50}.
- **Related:** n17, n29 (one row per measurement vs mean label, same table), n26, n39.

### n20 — Cleaning before the hybrid search  (`tests/experiments/noise/n20_clean_before_search.py`)
- **Question:** Does it matter whether cleaning reaches the feature search, or only the final model fit?
- **How it works:**
  - Four arms: search and fit on labels as given; search as given, fit cleaned (the default in the rest of the suite); search and fit cleaned; search cleaned, fit as given.
  - "Search" covers both the recipe search and the per-fold feature ranking and budget split. A cleaned search trains on cleaned labels and scores candidates on labels as given.
  - Detector `cross_val` ("Forest residual"), computed in a single representation; flag rates 5%, 10%, 20%. Cleanups: `remove`, `label_smooth`, `knn_smooth`, `prox_smooth`, `oof_smooth`, `gp_smooth`, `constant_lift`, `js_shrink`; `downweight` reaches the fit only.
  - Regimes: injected damage (3 draws) and labels as delivered. Random-flag twins are defined but run only if a job names them.
  - Per fold, two decided models on 2 best single representations and 2 best hybrids.
- **Status:** Partial. Final files for ginkgo_colloidal, ginkgo_titer, herg_ki; rows or `_partial` files for buchwald_hartwig, caco2, cdr_flexibility, herg, logd, qm9 (10-10b copy). noise_full holds caco2 and herg_ki further folds. Faults (FAULTS.md): random-flag controls never ran; `grouped_shifted` on ginkgo_colloidal failed every damaged draw but is marked ok.
- **Results so far:** Thesis, DNN, fold 0 (checked, N7-35/36): titer as delivered, first hybrid, delete 20%: no cleaning 0.155, fit only 0.390, search and fit 0.562, search only 0.423 Spearman. SMAC 25% clipped, delete 5%: 0.237 / 0.416 / 0.489 / 0.221.
  - Level-0 check: the answer page reports cleaning beat no cleaning on the labels as delivered on caco2, ddg_combined, ginkgo_colloidal and herg_ki, for both models and all three placements. There is no random control, so these gains are unread against chance.
- **In thesis:** Ch7 \label{sec:noise:res:search}, \label{tab:noise:res:clean_before_search}.
- **Related:** n15 (search on clean vs damaged labels, same section), n07, n18.

### n21 — Which distance the neighbour detectors use  (`tests/experiments/noise/n21_neighbour_distance.py`)
- **Question:** Does the distance in which nearest neighbours are found change how well neighbour-based detectors find damaged labels, and how much down-weighting by them repairs?
- **How it works:**
  - Distance families: plain scaled features; importance-weighted features (`lightgbm_gain`/`treeshap`, exponent 0.5 or 1.0, Euclidean or Manhattan); forest proximity (100 trees, min leaf 5); the hybrids' own feature spaces, rebuilt on damaged labels.
  - Detectors: `neighbor_residual` and `local_variance_ratio`, at k = 5, 10, 20 and two neighbour-weighting schemes (rank, inverse square).
  - Detection graded by average precision against the molecules whose label moved. Only damage kinds that move a subset of labels: `outlier_p05`, `censoring`, `shift_only_p10`, and flips on classification.
  - Repair: down-weight with weight 1/(1 + 3 × scaled score); top 3 settings per distance refitted after an inner hold-out screen.
  - 3 draws per damaged level, level 0 once. Two decided models, 2 best single representations and 2 best hybrids.
- **Status:** Ran on server, fold 0, on adme_mdr1, buchwald_hartwig, caco2, cdr_flexibility, ddg_combined, ginkgo_colloidal, herg, herg_ki, logd, qm9 (10-10b copy). Fault (FAULTS.md): per-draw no-cleaning scores sit in rows files the 10-08 copy lacked.
- **Results so far:** Thesis (checked, N7-15, N7-07): caco2 outliers, best average precision per family 0.337 plain, 0.352 importance-weighted, 0.336 forest proximity, 0.335 hybrid, chance 0.051. LogD 5% outliers, DNN, first hybrid: 0.588 → 0.556 RMSE with neighbour weights.
  - Level-0 check: on QM9 with no noise, DNN second hybrid 0.362 → 0.348 eV RMSE with neighbour weights. The thesis reads this as a model effect, not noise removal.
- **In thesis:** Ch7 \label{sec:noise:res:detectors}, \label{tab:noise:res:detect_families}, \label{tab:noise:res:weighting}.
- **Related:** n12 (importance-weighted distance; n21 settles its Manhattan exponent defect), n07, n13, n35.

### n22 — Do detectors find the records ChEMBL curators flagged  (`tests/experiments/noise/n22_curator_flag.py`)
- **Question:** Do the suite's detectors rank highly the records that ChEMBL curators marked suspect, with no injected noise?
- **How it works:**
  - Dataset herg_ic50 (CHEMBL240 IC50, one row per ChEMBL record). Ground truth: ChEMBL's `data_validity_comment`. 301 of 12,008 records carry one: 293 "Outside typical range", 8 "Potential transcription error".
  - Detector list imported from n11; neighbour detectors run at k = 5, 10, 20 in every neighbour setting of the dataset's group. A record's own repeat measurements are excluded from its neighbours.
  - Run in 8 feature spaces per fold (svm and dnn: 2 best single representations, 2 best hybrids). Fits no prediction model.
  - Scores: average precision, precision and recall at k (k = number flagged), chance = flagged share. Graded per record and per molecule.
  - The author's ruling in the file: the flag is outlier noise only; no claim about noise in general.
- **Status:** Ran on server, fold 0, herg_ic50 only (10-10b copy). Every other dataset is skipped (no curator flag). The 10-08 answer page predates this and says nothing ran. Partly circular: "Outside typical range" is assigned for extreme values.
- **Results so far:** Thesis (checked, inside N7-34): neighbour-label gap and forest residual reached average precision 0.14 to 0.23 across configurations, against chance 0.036.
- **In thesis:** Ch7 \label{sec:noise:res:herg_ic50} (one paragraph).
- **Related:** n11, n06 (another external ground truth), n17, n19.

### n23 — A model's own uncertainty as a detector  (`tests/experiments/noise/n23_uncertainty_cleaning.py`)
- **Question:** Can a model's per-molecule predicted spread find and clean damaged labels, compared with a neighbour-based detector?
- **How it works:**
  - Sources: `hetero_gp`, `ngboost`, `heteroscedastic_nn` (predict spread from features only), `gaussian_process` (posterior SD plus residual; "GP spread and residual"), `activity_cliff_aware` ("Cliff-aware gap", the neighbour baseline). All out of fold, 5 folds inside.
  - Actions: `drop` top 5%, 10%, 20%; `move` labels toward the source's out-of-fold prediction at strength 0.25, 0.5, 1.0 under a `proportional` or `confidence` rule; `none`.
  - The three feature-only sources are marked undefined under damage that picks molecules at random (outliers, flips), since their spread cannot see a label.
  - Conditions: `outlier_p05`, `censoring`, `shift_only_p10`, `grouped_shifted`, flips on classification; injected (3 draws) and labels as delivered.
- **Status:** Ran on server, fold 0, on adme_mdr1, buchwald_hartwig, caco2, herg, herg_ki, logd, qm9 (10-10b). noise_full adds caco2, herg, herg_ki, logd at further folds. Fault (FAULTS.md): herg_ki censoring has two records of the same labels disagreeing in 18 of 400 entries.
- **Results so far:** Thesis (checked, N7-05): hERG 10% confusion-matrix flips, XGBoost hybrid, drop top 10%: none 0.8028 AUROC, Cliff-aware gap 0.8375, GP 0.8270. LogD outliers DNN hybrid: none 0.5879 vs GP 0.5581 RMSE.
  - Level-0 check: on QM9 with no noise, dropping 10% made the DNN hybrid worse with every source (0.349 → 0.371 to 0.423 eV RMSE).
- **In thesis:** Ch7 \label{tab:noise:res:herg_flip_actions}, \label{tab:noise:res:logd_outlier_actions}, \label{tab:noise:res:qm9_deletion}, \label{sec:noise:res:detectors} (N7-05, N7-17).
- **Related:** n07, n24, n31, n32.

### n24 — Delete, rewrite or down-weight flagged molecules, against random deletion  (`tests/experiments/noise/n24_delete_or_keep.py`)
- **Question:** Once a detector flags molecules, is it better to delete them, rewrite their labels, or down-weight them? Does deletion beat deleting the same number of molecules at random?
- **How it works:**
  - Detectors `neighbor_residual` and `cross_val`; flag rates 5%, 10%, 20%.
  - Actions: delete (`remove`); rewrite (`label_smooth`, `knn_smooth`, `oof_smooth`); down-weight (`downweight`, `huber_weight`).
  - Control: delete the same number at random, seeds 0–4; reported per seed, never averaged across seeds.
  - Conditions: decided damage kinds (default `outlier_p05`), 3 draws, plus level 0 (labels as delivered).
  - Two decided models on 2 best single representations and 2 best hybrids.
- **Status:** Ran on server, fold 0 on adme_mdr1, buchwald_hartwig, caco2, herg, herg_ki, logd, qm9; noise_full holds folds 1–2 for caco2, herg, herg_ki, logd (10-10b).
- **Results so far:** Thesis (checked, N7-03/10/11): hERG confusion flips, XGBoost first hybrid, delete 10%: 0.803 → 0.838 AUROC vs best random 0.818. Buchwald–Hartwig as delivered, LightGBM, delete 20%: 20.1 → 18.0 RMSE (yield points) vs best random 19.9.
  - Level-0 check: on labels as delivered, deletion beat no cleaning and every random seed in 0 settings on caco2, herg_ki, logd (of 96 per file), 1 on herg and adme_mdr1, 31 on buchwald_hartwig. herg_ki DNN hybrid smoothing: 0.599→0.567, 0.550→0.547, 0.550→0.537 RMSE on folds 0–2.
- **In thesis:** Ch7 \label{tab:noise:res:delete_vs_random}, \label{tab:noise:res:hergki_smoothing_folds}, \label{sec:noise:res:delivered}, \label{fig:noise:score_vs_level}.
- **Related:** n07, n18, n23, n25.

### n25 — Putting activity-cliff molecules back after cleaning  (`tests/experiments/noise/n25_cliff_exemption.py`)
- **Question:** After a cleanup, does restoring molecules that sit on an activity cliff (AC) improve the held-out score more than restoring the same number of random molecules?
- **How it works:**
  - One cleanup call per pass, fitted three times: `plain`; `cliff_exempt` (every AC molecule's label, weight and row restored); `random_exempt` (same number of non-cliff molecules restored).
  - AC definition reused from n01: ECFP4 Tanimoto ≥ 0.55 and activity difference > 2 × median absolute deviation of the labels.
  - Detectors `neighbor_residual`, `cross_val`, `activity_cliff_aware`; flag rates 1%, 5%, 20%; n07's cleanups.
  - Conditions `censoring`, `shift_only_p10`, `sparse_dprime_up_p10`, `outlier_p05`, and level 0; 3 draws.
  - Headline is the within-pair difference, not separate means.
- **Status:** Ran on server, fold 0 on adme_mdr1, buchwald_hartwig, caco2, herg, herg_ki, logd (qm9 rows only); noise_full folds 1–2 for caco2 and herg_ki (10-10b). The random control cannot run where cliff molecules outnumber non-cliff ones: herg has 3,345 of 6,206 training molecules on a cliff; caco2 puts back 866–1,412 with only 314–860 in the pool.
- **Results so far:** Thesis, DNN first hybrid, delete 1% by neighbour-label gap, level 0 (checked, N7-12). herg_ki RMSE, no cleaning / delete / cliff back / random back: fold 0 0.597/0.596/0.579/0.595; fold 1 0.551/0.5681/0.5684/0.557; fold 2 0.550/0.575/0.537/0.545.
  - Level-0 check: deletion alone did not beat no cleaning on any herg_ki fold. QM9 fold 0: 0.349 → 0.382 eV, cliff back 0.372, random back 0.382.
- **In thesis:** Ch7 \label{sec:noise:res:delivered}, \label{tab:noise:res:cliff_putback}.
- **Related:** n01 (cliff definition), n14, n07, n24.

### n26 — How large is the real label noise, measured from repeat measurements  (`tests/experiments/noise/n26_noise_variance_from_repeats.py`)
- **Question:** On real labels, how large is the per-measurement noise variance, and is it larger than the model's own error? Nothing is injected.
- **How it works:**
  - Needs a loader that keeps repeat measurements; only `herg_ic50` (12,008 ChEMBL records on 9,802 molecules) qualifies; every other dataset writes a skip.
  - One row per molecule; the training label is the mean of all its measurements. Same searches and folds as n19.
  - Noise variance measured three ways: directly as pooled within-molecule variance; split-half correlation with Spearman-Brown projection, plus intraclass correlation; and by regressing each molecule's squared out-of-fold residual on 1/(measurement count) (ordinary least squares, binned, gamma GLM).
  - Converts the direct estimate into an R² ceiling (the R² a model would reach if its only error were label noise) and sets each model's out-of-fold R² against it.
  - Estimator check: injects known variance (0.1, 0.25, 0.5 squared log units) and asks whether the regression slope moves by that much.
  - Models SVM and DNN; four configurations; metric RMSE in log units.
- **Status:** Ran on herg_ic50, folds 0 to 2 of 5 (10-10b). Every other dataset is a skip by design (no repeat measurements).
- **Results so far:** herg_ic50 (checked, 10-10b): direct noise variance 0.337 squared log units from 1,218 molecules with two or more records; intraclass correlation 0.776; R² ceiling 0.445. SVM on the first single representation reached out-of-fold R² 0.430 against the mean label. The 1/count regression gave negative slopes in every configuration (SVM −0.280 to −0.083), so it does not recover a positive noise variance on real labels. The injected-variance check moved the slope by 0.11, 0.29 and 0.51 for injected 0.1, 0.25 and 0.5.
- **In thesis:** not in thesis (no SOURCE comment cites n26; sec:noise:res:herg_ic50 uses n19 and n29).
- **Related:** n19 (shares its searches), n29, n17, n36.

### n27 — What removing correct molecules costs, and how much label error reaches the test error  (`tests/experiments/noise/n27_removal_cost_and_propagation.py`)
- **Question:** With every label correct, how much does training on fewer molecules cost? And what share of variance added to training labels shows up in held-out error?
- **How it works:**
  - `removal_cost`: train on a random 25%, 50%, 70%, 80% or 90% of training molecules, 20 random draws each, every label correct; paired against training on all.
  - `injected`: Gaussian noise at the dataset's levels (0.5, 1.0, 1.5 training-label SD), 3 draws; reports (held-out MSE increase) divided by (measured injected variance).
  - `move_toward_oof`: correct labels moved 25%, 50% or 100% of the way toward a group-aware out-of-fold LightGBM prediction; same ratio.
  - Held-out molecules always graded on labels as delivered. Cleans nothing; no detector.
  - Two decided models, four configurations, all 5 outer folds.
- **Status:** Ran on all 5 folds on herg_ki, caco2, logd, qm9, buchwald_hartwig, adme_mdr1, ginkgo_colloidal (10-10b). herg writes a skip (not regression).
- **Results so far:** Checked (10-10b, first hybrid, mean of 5 folds). Dropping a random tenth of correct molecules: herg_ki DNN 0.5684 → 0.5816 RMSE, logd DNN 0.4078 → 0.4225, ginkgo_colloidal DNN 0.2005 → 0.2552 Spearman. Moving correct labels a quarter of the way to the prediction: herg_ki DNN 0.5684 → 0.5657, caco2 DNN 0.2766 → 0.2736, adme_mdr1 DNN 0.4401 → 0.4339, ginkgo_colloidal SVM 0.3587 → 0.3274 Spearman (worse); logd, qm9, buchwald got worse. So relabelling toward a prediction can lower RMSE with no noise present. The 10-08 answer page puts the injected-variance pass-through at 9% to 32% on caco2 and herg_ki (not re-checked).
- **In thesis:** not in thesis (no SOURCE comment or ch7_numbers check cites n27).
- **Related:** n24, n31 (relabel and shrink actions), n38.

### n29 — Does removing a per-document offset change what the model learns  (`tests/experiments/noise/n29_document_bias.py`)
- **Question:** On herg_ic50, does fitting one offset per ChEMBL document (paper) and removing it from training labels improve held-out prediction? Nothing is injected in the main arms.
- **How it works:**
  - Arms: `mean_label` (one row per molecule, the baseline); `per_record` (one row per measurement); `per_record_bias` (document offset from an alternating ridge fit subtracted); `mean_label_bias_corrected`; `per_record_random_effect` (offset shrunk by a method-of-moments random intercept); `document_quantile_map`; `within_document_rank` (XGBoost pairwise ranking, document as query group).
  - A known-bias check injects document offsets and asks whether the fit recovers them.
  - Offsets never reach feature selection. Models SVM and DNN; first hybrid and single representations; RMSE in log units.
- **Status:** Ran on herg_ic50, fold 0 only (10-10b; 40 rows). Every other dataset is a skip (no document per measurement).
- **Results so far:** Checked (10-10b, first hybrid, fold 0): mean label 0.7668 SVM / 0.7256 DNN; one row per measurement 0.7652 / 0.7343. Removing document offsets made both worse: per_record_bias 0.7889 / 0.7743; random intercept 0.7877 / 0.7695; quantile map 0.8678 / 0.8675.
- **In thesis:** Ch7, tab:noise:res:herg_ic50, rows "Separate run, one fit: mean label against one row per measurement" (0.767, 0.765, 0.726, 0.734), section sec:noise:res:herg_ic50 [N7-34]. The offset arms are not reported.
- **Related:** n19, n26, n39 (simulated sources with offsets), n34.

### n30 — Clipped labels: lift them, drop them, or fit a censored likelihood  (`tests/experiments/noise/n30_censored_likelihood.py`)
- **Question:** When training labels above an assay limit are recorded as the limit, which treatment recovers the lost accuracy? The clipped molecules are known, so no detector is needed.
- **How it works:**
  - Injection `censoring`: the top share (level 0.25 = 25% of labels) set to the limit; deterministic, so one draw per level. Level 0 = labels as delivered.
  - The limit is read from training labels (largest value, if ≥2 molecules and ≥1% sit on it); out-of-fold LightGBM gives a prediction per molecule.
  - Arms: `clipped` (baseline), `constant_lift`, `per_molecule_lift` and the same lifts shuffled among clipped molecules (3 seeds), `truncated_normal_mean`, `censored_em` (3 rounds with the configuration's own model), `winsor_c1/1.25/1.5`, `drop_clipped`, `xgboost_aft` where XGBoost is a decided model; `clean` is the ceiling.
  - Two decided models × four configurations per fold; hybrids rebuilt from the clipped labels.
  - Fold 0 on most datasets; logd 5 folds and caco2 extra folds in noise_full.
- **Status:** Ran (10-10b) on logd, qm9, herg_ki, caco2, buchwald_hartwig, adme_mdr1, flab_thermo_tresanco, ginkgo_colloidal, ginkgo_titer, ginkgo_acsins, ddg_combined, abcov_kd, plus abagym_EGFR (thesis excludes abagym_EGFR from Ch7). Classification sets skip.
- **Results so far:** Thesis tab:noise:res:censoring_moves_main (DNN, first hybrid, fold 0, RMSE): logd before clipping 0.406, clipped 0.605, censored EM 0.461; qm9 0.351 / 0.520 / 0.441; herg_ki 0.608 / 0.731 / 0.664 (constant lift 0.668). Further folds (tab:noise:res:censoring_folds): logd fold 0 0.404 / 0.613 / 0.480. Deleting clipped molecules was worse than keeping them on caco2, herg_ki, qm9. Shuffled lifts beat each molecule's own lift (tab:noise:res:censoring_lift). Level 0, checked (10-10b, caco2, DNN first hybrid): as delivered 0.2341, constant lift 0.2406, censored EM 0.2376, drop clipped 0.2329; the lifts do not beat no treatment there.
- **In thesis:** Ch7 sec:noise:res:censoring; fig:noise:censoring_ladder, tab:noise:res:censoring_folds, tab:noise:res:censoring_moves_main, tab:noise:res:censoring_lift; Additional file app:add:noise_censoring_moves [N7-20 to N7-25].
- **Related:** n30b, n33 (expectile loss under censoring), n32.

### n30b — Real clipping in the labels as delivered (CHO polyreactivity), plus a QM9 check  (`tests/experiments/noise/n30b_real_censoring.py`)
- **Question:** ginkgo_pr_cho has 58 of 197 labels at exactly 0 and none below. Treating 0 as a lower assay limit, do n30's treatments help?
- **How it works:**
  - Runs n30's own functions on negated labels so the floor becomes a top limit; n30's code is unchanged.
  - Arms as n30 (constant lift, per-molecule lift and shuffled lifts, truncated-normal mean, censored EM, winsor, drop clipped), against `as_delivered`.
  - New control `not_at_floor_control`: the same number of labels not at the limit, moved by the constant lift's amount, 3 seeds.
  - QM9 arm (`--inject-censoring 0.25`): clips QM9's clean labels at the top, so the truth is known; includes `clean`.
  - Two decided models (GP and DNN on ginkgo_pr_cho), four configurations, all 5 folds; Spearman on ginkgo_pr_cho, RMSE on QM9.
- **Status:** Approved by the author 2026-10-09. Ran on ginkgo_pr_cho and QM9, all 5 folds; files only in the `view` copy (dated 10 Oct), not in 10-09 or 10-10b.
- **Results so far:** Checked (`view`, first hybrid, mean of 5 folds). QM9 DNN: clipped 0.5072, censored EM 0.3980, constant lift 0.4176, clean 0.3169 RMSE; moving unclipped labels 0.518 to 0.525. ginkgo_pr_cho GP: as delivered 0.4555 Spearman; every move lowered it (censored EM 0.4495, constant lift 0.4288). DNN: as delivered 0.4114; winsor c=1 0.4360 and drop clipped 0.4247 raised it; the not-at-floor control ranged 0.3155 to 0.4338.
- **In thesis:** QM9 arm in Ch7 sec:noise:res:censoring, last paragraph ("mean RMSE went from 0.5072 ... to 0.3980"), with no SOURCE comment naming n30b. The ginkgo_pr_cho result is not in thesis.
- **Related:** n30.

### n31 — How good must a detector be before acting on its flags helps  (`tests/experiments/noise/n31_regime_and_breakeven.py`)
- **Question:** What precision (share of truly damaged molecules among those flagged) must a flagged set reach before deleting, down-weighting, relabelling or shrinking it improves the model? How does that compare with the residual detector's actual precision?
- **How it works:**
  - Damage kinds: `outlier_only_p05` (5% of molecules get wider errors, others untouched), `shift_only_p10`, `sparse_dprime_up_p10` (shift sized by the model's out-of-fold error), `gaussian`; 3 draws.
  - Builds flagged sets of 5%, 10% or 20% of training molecules with precision set to 0, 0.25, 0.5, 0.75 or 1.0 from the injector's mask; 2 seeds.
  - Actions: remove; down-weight to 0.25; rewrite to the out-of-fold LightGBM prediction; shrink toward it by an estimated weight. Break-even = precision where the gain crosses zero.
  - Also grades the residual detector's AUROC against two closed-form predictions and records residual kurtosis.
  - Two decided models, four configurations, fold 0.
- **Status:** Ran (10-10b) on caco2, herg_ki, buchwald_hartwig, ddg_combined (summary files) and logd, qm9, adme_mdr1, ginkgo_colloidal (row files only). Classification sets skip. The 10-08 answer page says nothing ran; that copy predates these files.
- **Results so far:** Thesis tab:noise:res:breakeven: caco2, shift sized by model error, top 5% flagged, precision 0.171 against break-even 0.616 for deletion. herg_ki shift in 10%, top 10%: precision 0.262 against 0.684. With wide errors in 5%, precision exceeds break-even on both datasets. Level 0: tab:noise:res:known_flags, QM9 with no noise, the same down-weighting moved DNN first single 0.396 → 0.402 RMSE, first hybrid 0.355 → 0.359 (worse). Under damage, herg_ki first single 0.689 → 0.642.
- **In thesis:** Ch7 sec:noise:res:detectors, tab:noise:res:breakeven [N7-19]; sec:noise:res:action, tab:noise:res:known_flags [N7-06].
- **Related:** n24, n27, n32, n38, n41.

### n32 — Trimmed refits, two learners picking for each other, and further detectors  (`tests/experiments/noise/n32_iterative_and_dual_selection.py`)
- **Question:** Do iterative trimming, two-learner selection, or shrinking only flagged labels undo injected damage? Do detectors from outside cheminformatics rank damaged molecules better than the plain residual?
- **How it works:**
  - Detectors, graded by AUROC and average precision against the injector's mask: out-of-fold LightGBM residual; residual divided by NGBoost, quantile-forest or conformal spread; nearest-neighbour Shapley value (k 5, 10, 20, several distances); cleanlab label quality.
  - Methods: `itlm_a0.8/0.9/0.95` (4 rounds keeping the lowest-loss share); `two_learner_share0.05/0.1/0.2` (LightGBM and SVM each pick low-loss molecules for the other; fit on the intersection); `shrink_flagged_one_weight` and `shrink_flagged_per_row`.
  - Controls: the same number of molecules removed at random, 3 seeds, for ITLM and two-learner.
  - Damage kinds: censoring, outlier_p05, shift_only_p10 (gaussian and grouped_shifted skip detection); level 0 = labels as delivered. Fold 0.
- **Status:** Ran (10-10b) on buchwald_hartwig, herg_ki, logd, ddg_combined, ginkgo_colloidal, ginkgo_titer, qm9 (damaged levels only), caco2, flab_thermo_tresanco (partial). FAULTS.md (10-08) records cleanlab missing on the server for n32; the thesis reports a cleanlab score, so this was presumably fixed (unverified).
- **Results so far:** Thesis: buchwald RF first single, shift in 10% at 1.0, two-learner share 0.1: 20.3 → 18.6 RMSE yield points, ahead of random deletion at all 3 seeds (checked: 20.34 → 18.60; random 20.28 to 20.78). The same method at level 0 (labels as delivered) took it 18.88 → 17.88 (checked), against 18.96 to 19.29 for random deletion. So about 1.0 of the 1.74-point gain under damage also appears with no damage. Detection: under censoring on buchwald, Shapley value average precision 0.391 vs 0.195 for the LightGBM residual (thesis). On herg_ki and logd at level 0 every method made the first hybrid worse (checked).
- **In thesis:** Ch7 sec:noise:res:detectors, detection paragraph and Additional file app:add:noise_detect_other [N7-17]; sec:noise:res:model, two-learner paragraph [N7-41].
- **Related:** n09 (small-loss, multistage cleaning), n23, n24, n31, n41.

### n33 — Robust losses, target transforms, weights and augmentation, with no detector  (`tests/experiments/noise/n33_robust_training.py`)
- **Question:** Leaving every label in place, does changing how the model reads labels undo injected noise? Methods come from outside cheminformatics (robust statistics, C-Mixup, target-noise bagging).
- **How it works:**
  - Label and weight arms: `yeo_johnson`, `invvar_ngboost` (weights from NGBoost spread), `c_mixup`, `target_noise_bagging`, `huber_teacher_blend` (label halfway to an out-of-fold Huber LightGBM prediction).
  - Loss swaps, each paired with the same model under squared loss: LightGBM Huber/fair/median/Tukey/expectile, XGBoost pseudo-Huber and absolute, RF absolute-error splits, DNN Huber at thresholds c × s, NGBoost Student-t vs normal.
  - Scale s = 1.4826 × median absolute out-of-fold residual, so thresholds are in units of the model's own error.
  - Six damage kinds (gaussian, outlier_p05, grouped_shifted, censoring, shift_only_p10, sparse_dprime_up_p10); 3 draws per level; level 0 once. Fold 0.
  - Baseline `noisy` (same model on damaged labels); ceiling `clean`. No random control for the label arms.
- **Status:** Ran (10-10b) on herg_ki, caco2, logd, buchwald_hartwig, qm9 (level 0 only on qm9). FAULTS.md B7: `controls_run: false` on caco2 censoring. FAULTS.md A5: a hybrid's clean-label score varies with damage level, because the hybrid is reselected per draw.
- **Results so far:** Thesis tab:noise:res:huber_blend: herg_ki Gaussian 1.0, DNN second hybrid, damaged 0.718 → Huber blend 0.681 RMSE (mean of 3 draws). On QM9 level 0 Huber blend was worse everywhere (DNN second hybrid 0.3618 → 0.3682). Level 0 on herg_ki, checked: DNN first hybrid 0.5972 → 0.5909 with Huber blend, so it also helps herg_ki with nothing injected. QM9 level 0: yeo_johnson 0.3550 → 0.3518 (checked). Thesis: DNN Huber loss helped under outliers on herg_ki (0.7413 → 0.7269, first single).
- **In thesis:** Ch7 sec:noise:res:other_kinds, tab:noise:res:huber_blend [N7-29]; sec:noise:res:model, paragraph on other robust training, Additional file app:add:noise_robust_broad [N7-41].
- **Related:** n40 (model-side regularisation), n09 (published sweep), n30.

### n34 — An offset per scaffold group: subtract the group mean or fit a random intercept  (`tests/experiments/noise/n34_group_random_effects.py`)
- **Question:** Under noise that shifts each scaffold group by a shared offset, does estimating and removing that offset help? Is shrinking it by a random intercept better than the group's mean residual?
- **How it works:**
  - Groups: the injector's scaffold groups, sliced to the fold's training molecules.
  - Arms: `group_center` (subtract group mean out-of-fold residual), `group_center_3plus` (only groups with ≥3 training molecules), `random_intercept` (offset = residual sum / (count + σ²/τ²), variances by moments).
  - Conditions: `grouped_shifted` (0.62 of noise variance at group level), `gaussian` (no group structure, the control), `none` (level 0, labels as delivered).
  - Levels 0, 0.5, 1.0, 1.5 (or 0, 1.0) training-label SD; 3 draws. Scored on clean held-out labels. Fold 0.
- **Status:** Ran (10-10b) on caco2, herg_ki, logd, qm9, adme_mdr1. buchwald_hartwig skipped (no scaffold groups). FAULTS.md B7: no random-group control in the files.
- **Results so far:** Thesis tab:noise:res:family_offsets (DNN first hybrid, level 1.0): caco2 draw 0 damaged 0.321, family mean 0.297, random intercept 0.280, clean 0.239. Level 0 (tab:noise:res:offsets_delivered): both made every dataset worse, e.g. caco2 0.233 → 0.267 / 0.244, except a small MDR1 gain (0.466 → 0.453). Control, checked (10-10b): under gaussian with no group offset, random intercept still lowered herg_ki 0.7196 → 0.6799 (mean of 3 draws), against 0.7935 → 0.694 under grouped_shifted (both means of 3 draws). So much of the gain does not need a group offset to exist.
- **In thesis:** Ch7 sec:noise:res:offsets, tab:noise:res:family_offsets and tab:noise:res:offsets_delivered [N7-26, N7-27].
- **Related:** n29, n39.

### n35 — Disagreement between near-identical molecules as a noise estimate and a weight  (`tests/experiments/noise/n35_near_duplicate_disagreement.py`)
- **Question:** Without a model, how much do labels of almost-identical molecules disagree? Does down-weighting disagreeing close pairs help?
- **How it works:**
  - Close pairs: training pairs in the lowest 1% of pairwise distance, in each neighbour setting (several distance spaces and metrics, including the distance and similarity hybrids).
  - Noise estimate σ̂² = mean squared label difference / 2 over the closest fifth of these pairs.
  - `downweight_pairs`: both molecules of a pair differing by more than 3√2 σ̂ get weight 0.25. Only the screen's top 3 settings get a refit.
  - Gaussian noise at the dataset's levels, 3 draws; level 0 once. Two decided models, four configurations, fold 0.
  - No random control (`controls_run: false`, FAULTS.md B7).
- **Status:** Ran (10-10b) on caco2, buchwald_hartwig, qm9, adme_mdr1, ddg_combined, flab_thermo_tresanco, ginkgo_colloidal, ginkgo_titer, ginkgo_acsins, abcov_kd, and abagym_EGFR (not in Ch7); herg_ki and logd have summary files only. FAULTS.md C5: a negative variance estimate under forest proximity (caco2 SVM).
- **Results so far:** Thesis tab:noise:res:weighting: buchwald LightGBM first hybrid, Gaussian 1.0, 22.0 → 20.7 RMSE yield points (best setting). QM9 level 0: DNN 0.362 → 0.350, LightGBM 0.372 → 0.390. Level 0, checked (10-10b, mean over 3 settings): caco2 DNN first hybrid 0.2474 → 0.2390, larger than its gain at level 1.0 (0.3063 → 0.3048). Buchwald LightGBM level 0: 20.08 → 20.07.
- **In thesis:** Ch7 sec:noise:res:action, weighting paragraph and tab:noise:res:weighting [N7-07]; method in sec:noise:mitigation as Close-pair weights.
- **Related:** n02, n13, n21 (weighting by neighbour scores), n26 (noise variance from repeats), n37.

### n36 — Which molecules should get a second measurement  (`tests/experiments/noise/n36_remeasure_priority.py`)
- **Question:** A lab can afford a second measurement for a fixed share of its molecules. Does choosing those molecules by a rule give a lower held-out RMSE than choosing them at random?
- **How it works:**
  - Nothing is injected. The second measurement is a real repeat that the loader holds back and then hands out.
  - Only datasets that keep repeat measurements can run: `herg_ic50` (one row per ChEMBL record) and `bindingdb_kd`. Every other dataset writes a skip record.
  - Training molecules with two or more measurements start from one, drawn at random. The chosen ones get a second; their label becomes the mean of the two.
  - Budgets: 10%, 25% and 50% of those molecules. Rules: random (3 draws, the control), the gap between the label and an out-of-fold LightGBM prediction, leverage (how many other training molecules have this one among their 5 nearest neighbours), leverage times that gap, and the sparsest molecules first.
  - The dataset's two decided models, each on 4 configurations (its 2 best single representations and 2 best hybrids, searched on that fold's training molecules). Held-out molecules are graded against the mean of all their measurements.
- **Status:** Ran on `herg_ic50` only, outer fold 0, DNN and SVM, 312 rows (same in the 10-09 and 10-10b copies). Every other dataset wrote a skip record. `bindingdb_kd` has no result file.
- **Results so far:** On `herg_ic50`, fold 0, all rules and random sit within about 0.008 RMSE of each other. Averaged over the 4 configurations, DNN random gives 0.7614 to 0.7648 RMSE across the three budgets; the rules give 0.7590 to 0.7675. SVM random gives 0.7926 to 0.7935; the rules 0.7925 to 0.7946. (Computed this session from `herg_ic50.rows.jsonl`, 10-10b.) The answer page (2026-10-08) says nothing had run; it predates these rows.
- **In thesis:** not in thesis.
- **Related:** n19 and n26 (same repeat-measurement loader and skip rule), n39 (simulated repeat measurements).

### n37 — How much of the model's error is label noise, measured without cleaning  (`tests/experiments/noise/n37_noise_budget.py`)
- **Question:** What share of each model's held-out mean squared error could be label noise? Nothing is cleaned; the share is only measured.
- **How it works:**
  - Noise estimate: the Gamma test (Stefansson, Koncar & Jones 1997). For each training molecule, its 1st to 10th nearest neighbours; a straight line of half the squared label gap against squared distance, read at zero distance. Unit: label units squared. The code calls it an upper bound, because it also holds activity differences the features cannot resolve.
  - One estimate per pruned representation of the dataset (10 on most datasets). Tanimoto distance for 0/1 features, else Euclidean on scaled features.
  - Check: Gaussian noise added to the training labels at the decided levels (0, 0.5, 1.0, 1.5 label SDs on datasets of 5,000 molecules or fewer; 0 and 1.0 above). The estimate should rise by the variance added.
  - Share: each estimate divided by the held-out MSE of each of 8 fits (2 decided models x 4 configurations).
  - SIMEX (Cook & Stefanski 1994): extra noise at 0.5, 1.0 and 2.0 times a per-fold unit, 3 draws each; held-out MSE against added variance is extended back to minus the noise estimate.
  - 5 outer folds.
- **Status:** Ran on 10-10b with 5 folds: `buchwald_hartwig`, `caco2`, `ginkgo_colloidal`, `herg_ki`, `logd`, `adme_mdr1`, `qm9`, and `abagym_EGFR` (a dataset the thesis leaves out of Ch7). Classification datasets write a skip. Open fault (FAULTS.md A22): SIMEX keeps the hybrid's features fixed at those chosen from the clean labels.
- **Results so far:** Caco-2: estimates of 0.0431 (pdv) to 0.0717 (avalon) log units squared, against held-out MSE of 0.0753 to 0.0962; shares of 0.49 to 0.59 in each configuration's own representation (answer page; pdv value checked). hERG Ki: shares of 0.28 to 0.36. QM9, whose labels carry no assay error, still gives estimates of 0.0049 to 0.1887 eV squared across representations (checked, 10-10b). Under added noise the estimate overshoots on Caco-2 (rise 0.0487 to 0.0537 against 0.0472 added at level 0.5) and falls short on hERG Ki (0.7623 to 0.8008 against 0.8129 at 1.0).
- **In thesis:** not in thesis.
- **Related:** n26 (noise variance from repeats, used here where repeats exist), n38 (noise sized by model error), n40 (`dense_oofmse` damage uses the same out-of-fold MSE unit).

### n38 — Damage sized by the model's own error  (`tests/experiments/noise/n38_noise_relative_to_model_error.py`)
- **Question:** When damage is sized in units of the model's out-of-fold residual SD rather than the label SD, does cleaning lower the held-out RMSE? Can an out-of-fold residual find the moved molecules?
- **How it works:**
  - Unit: the SD of LightGBM's out-of-fold residuals on the fold's clean training labels, in one representation (`resid_sd`).
  - Damage kinds: `sparse_dprime_up_p10` (10% of training molecules moved up by level x `resid_sd`; levels 0, 1, 2, 4 in the rows that ran), `dense_oofmse` (Gaussian with variance level x residual variance, levels 0, 0.1, 0.3, 1.0), `hetero_seen` and `hetero_hidden`. Five other sparse kinds were cut on 2026-10-06 (named in the code's CUT list).
  - Label base as a second axis: `real` labels, and `semi_synthetic` labels (a LightGBM prediction plus a resampled residual, so the true signal is known).
  - Methods: winsorise at 1.5 robust residual SDs, Huber-type robust loss (DNN only), trimmed refitting keeping the 90% lowest-loss molecules over 4 rounds (`itlm_a0.9`), and inverse-variance weights from an out-of-fold NGBoost. Random controls (3 seeds) exist for winsorise and trimming only.
  - 2 decided models x 4 configurations; the moved molecules are ranked by the out-of-fold residual and graded by AUROC and average precision.
- **Status:** Outer fold 0 only. 10-10b holds rows for `sparse_dprime_up_p10` on 12 regression datasets; `caco2`, `adme_mdr1` and `qm9` have level 0 only. `dense_oofmse` ran on `abcov_kd`, `ginkgo_acsins`, `ginkgo_colloidal` and `ginkgo_titer`. `hetero_*` have no rows. Rows also exist for `abagym_EGFR`, which the thesis leaves out of Ch7.
- **Results so far:** Thesis (N7-41): the DNN's robust loss on LogD's first hybrid went from 0.4041 to 0.3982 RMSE at level 1.0, draw 0 (clean labels 0.3918). Thesis (N7-18): the residual ranked shifted molecules above chance on every dataset; on expression titer its average precision was 0.20 to 0.25 against 0.106 by chance. Level 0 matters: winsorising helps at level 0 too. Buchwald–Hartwig LightGBM hybrid: 20.08 to 19.39 RMSE (yield points) at level 0, 20.73 to 19.73 at level 1.0. hERG Ki DNN hybrid: 0.5972 to 0.5911 at level 0, 0.6237 to 0.6004 at level 1.0 (both checked, real labels, 10-10b).
- **In thesis:** Ch7, Section `sec:noise:res:detectors` (paragraph on the residual under the shift sized by model error, N7-18) and Section `sec:noise:res:model` (robust-loss paragraph, N7-41). Method described in `sec:noise:detection`.
- **Related:** n41 (conformal flags on the same residual), n31 (shift-only damage in label SDs), n32 and n33 (trimming and robust training), n40.

### n39 — Several sources per molecule, with shared errors and an offset per source  (`tests/experiments/noise/n39_correlated_sources.py`)
- **Question:** When each training label is the mean of several simulated sources, does removing each source's fixed offset, or weighting molecules by their estimated error variance, lower the held-out RMSE compared with the plain mean?
- **How it works:**
  - Simulation on the clean labels: 8 sources; each molecule gets k of them, k drawn from the real per-molecule counts of `herg_ic50` (capped at 4) or uniformly from 1 to 4.
  - Each measurement = clean label + source offset + error. Error size = level x clean label SD (levels 0.5, 1.0, 1.5; no level 0). rho, the share of error common to all sources of one molecule: 0, 0.3, 0.7. Offset size tau: 0 or 0.5 of the error size.
  - Methods: plain mean (the score to beat), offsets fitted by a mixed model and removed, weights 1/(a + b/k) fitted from out-of-fold error (n19's fit), weights from within-molecule variance, offsets removed plus weights. Two oracle weightings know the true variances. No random control.
  - 2 decided models x 4 configurations; held-out molecules graded against clean labels.
- **Status:** Outer fold 0 only. 10-10b rows: `buchwald_hartwig`, `ddg_combined`, `ginkgo_colloidal`, `herg_ki` (levels 0.5 to 1.5), `caco2` (0.5, 1.0), `logd`, `qm9` (1.0), `adme_mdr1` (0.5). Open fault (FAULTS.md A20, from the pages, not re-checked): n19's fitted curve has a negative slope floored to zero, so `weights_n19_fitted` often equals the plain mean exactly.
- **Results so far:** Thesis: on hERG Ki the DNN on the hybrid scored 0.742 RMSE with the plain mean and 0.748 after removing offsets. Removing offsets helped most often on Buchwald–Hartwig, where offsets were large next to the label spread. On QM9 and the other datasets it helped in about half the settings or fewer. Answer page (10-09 copy, level 0.5): Caco-2 DNN hybrid, offsets removed plus weights, 0.2735 to 0.2586 RMSE.
- **In thesis:** Ch7, Section `sec:noise:res:sources` (N7-28).
- **Related:** n19 (measurement count, supplies the fitted weight), n26, n34 (offset per scaffold group), n36.

### n40 — Change the model instead of the labels  (`tests/experiments/noise/n40_model_side_regularisation.py`)
- **Question:** With the damaged labels left alone, do retuning, seed averaging, stacking or a noise-aware GP lower the held-out RMSE compared with the same model at fixed default settings?
- **How it works:**
  - Damage: `gaussian` at the decided levels (0, 0.5, 1.0, 1.5 label SDs; 0 and 1.0 on large datasets) and `dense_oofmse` (Gaussian with variance a share of LightGBM's out-of-fold MSE); other regression kinds (censoring, outliers in 5%, shifts) via `--noise-condition`. 3 draws per non-zero level.
  - Arms: `clean` (fixed model, clean labels, reference), `fixed` (the score to beat), `retuned` (default plus 20 settings scored on 3 scaffold-grouped inner folds of the training molecules), `tuned_clean_reused` (settings tuned on clean labels, then fitted on damaged ones), `seed_avg` (10 DNN seeds averaged, beside `seed_one`), RF with 1,000 trees, a non-negative stack of both models, nearest-neighbour predictions, GP with its noise term set from a noise estimate.
  - 2 decided models x 4 configurations, held-out scored against labels as delivered. No random control.
- **Status:** Outer fold 0 only. 10-10b: `caco2`, `herg_ki`, `logd`, `qm9` (most damage kinds), `buchwald_hartwig` (censoring, gaussian, outliers, shift), `adme_mdr1` (censoring only). hERG blocking writes a not-regression skip.
- **Results so far:** Thesis Table `tab:noise:res:change_model`: hERG Ki, Gaussian 1.0, DNN second hybrid 0.718 RMSE plain, 0.656 seed averaging, 0.671 stacking, 0.691 retuning; 0.608 on clean labels. Level 0 (Table `tab:noise:res:change_model_delivered`): the same configuration goes 0.612 to 0.598 with seed averaging, so part of the gain exists with no noise added. On QM9 with no noise, DNN second hybrid 0.357 plain, 0.346 seed averaging, 0.340 stacking; the thesis reads these gains as better training rather than noise found. With 25% of hERG Ki labels clipped, clean-label settings beat retuning in every configuration.
- **In thesis:** Ch7, Section `sec:noise:res:model` (N7-38 to N7-40), Tables `tab:noise:res:change_model` and `tab:noise:res:change_model_delivered`; QM9 rows in the clean-label test, Section `sec:noise:res:qm9test` (N7-50).
- **Related:** n33 (robust training), n38, n09 (survey of methods for learning with noisy labels).

### n41 — Flag suspect labels with a capped share of false flags  (`tests/experiments/noise/n41_conformal_fdr_detection.py`)
- **Question:** Can conformal p-values with Benjamini–Hochberg control (Bates et al. 2023) flag damaged labels while bounding the share of false flags? Does winsorising or removing the flagged molecules beat the same action on randomly chosen molecules?
- **How it works:**
  - Score: each training molecule's out-of-fold LightGBM residual divided by a second out-of-fold LightGBM's prediction of that residual's size.
  - p-value: rank of the score against a calibration set. Calibration sets: random 20% or 50% of the training fold, an oracle set of 50% unmoved molecules, and (thesis) molecules with three or more agreeing repeats. Benjamini–Hochberg at q = 0.05, 0.1, 0.2.
  - Actions: winsorise flagged labels to within 1.0 robust residual SD of the prediction, or remove them; each beside a random control of the same size (3 seeds).
  - Other detectors graded by AUROC: isolation forest and local outlier factor on features plus residual, and residual over the spread of 5 seeds.
  - Damage kinds: censoring, outliers in 5%, shift in 10%, `sparse_dprime_up_p10`. Gaussian and grouped shifts write "detection_undefined", because every label moves. 2 decided models x 4 configurations.
- **Status:** Outer fold 0 only. 10-10b: `buchwald_hartwig`, `caco2`, `herg_ki`, `logd`, `qm9`. hERG blocking writes a not-regression skip. Open fault (FAULTS.md A21): `cleanlab` is not installed on the server, so its detector errored (100 errors).
- **Results so far:** Thesis (N7-18): the procedure almost never flagged a molecule on the measured datasets. It flagged regularly only on QM9 with outliers or a shift injected, and acting on those flags barely changed the held-out score. At level 0 (labels as delivered) on LogD and Buchwald–Hartwig, no flagged action changed any score, so nothing beats no cleaning there (checked, 10-10b rows).
- **In thesis:** Ch7, Section `sec:noise:detection` (method) and Section `sec:noise:res:detectors` (result paragraph, N7-18).
- **Related:** n38 (same residual), n32 (unconverted conformal ranking), n09.

### n42 — Every mitigation method x every neighbour setting x every model input  (`tests/experiments/noise/n42_similarity_noise_sweep.py`)
- **Question:** For each method that changes the training data, and each neighbour setting it reads, how does the model fitted on the changed data compare with the model fitted on the damaged data?
- **How it works:**
  - One row: one model fitted on the damaged training fold and on the same fold after one method, both scored on the same held-out fold; the fit on clean labels sits beside them.
  - Methods are imported, not re-implemented, from n07 (cleanups), n24, n25 (put-back of cliff molecules, with a random put-back), n02 (weights), n23, n30 (winsorise, censoring lifts), n31, n32, n33 (NGBoost weights, C-Mixup), n35 and n41.
  - Neighbour settings: n07's spaces and distances x 5, 10, 20 neighbours x two counting schemes. All are screened by predicting a held-back fifth of the training molecules from their neighbours; the best 3 are refitted.
  - Damage kinds from the decided noise grid, at the decided levels; 3 draws per non-zero level (1 under censoring). Hybrids for model input are chosen from the noisy training labels.
  - 2 decided models x 4 configurations.
- **Status:** Outer fold 0 only. 10-10b rows: `adme_mdr1`, `buchwald_hartwig`, `caco2`, `herg_ki`, `logd`, `qm9`, hERG blocking (6 flip kinds), `cdr_flexibility` (uniform flips). Level-0 rows exist only for hERG blocking, `herg_ki`, and censoring on `buchwald_hartwig` and `logd`. No answer page.
- **Results so far:** None written up. Checked this session (10-10b, best of every method and setting, so optimistic): hERG Ki DNN first hybrid at level 0, 0.5972 RMSE plain against 0.5671 best (n07 label smoothing). At Gaussian 1.0, 0.7201 plain against 0.6180 best; clean labels 0.5958. So a large part of the best gain is already present with no damage.
- **In thesis:** not in thesis.
- **Related:** n07, n24, n25, n02, n23, n30–n35, n41 (all methods it calls); `d_vs_dprime.py` (row format).

### noise_ceiling — Extrapolating noise sweeps back to zero label error  (`tests/experiments/noise/noise_ceiling_analyse.py`)
- **Question:** If each score-against-noise-level curve were extended back past level 0, what score would exact training labels give?
- **How it works:**
  - Post-hoc reader. It runs no model and loads no dataset; it reads the noise suite's result files.
  - Treats total error as sqrt(b² + L²): L is the added level, b the labels' own error as a share of the label spread.
  - Fits four curve forms and reports the score at level 0 (measured), the fitted value at level 0, and the extrapolated value at zero total error.
  - Needs curves with 3 or more levels. Experiments whose result shape it does not know are listed as "no reader written".
- **Status:** Ran per dataset on the server (10-10b `noise/noise_ceiling/` and `noise_full/noise_ceiling/`). Only Caco-2 read any curves: 2,208, all from n23 censoring files, where the level counts labels changed, so no extrapolation is made. Every other dataset read 0 curves.
- **Results so far:** none usable. Caco-2 DNN on the first single representation, n23 censoring: 0.2761 RMSE at level 0, 0.3718 at 0.4 (checked in `caco2.txt`).
- **In thesis:** not in thesis.
- **Related:** n23 (source of its only curves), c10 (read, no curve with 3 levels).

## Groupings

- **Which action on flagged molecules (delete, weight, move):** n07, n11, n24, n18, n20, n23, n42.
- **Which detector finds damaged labels:** n07 (forest residual, cliff-aware gap), n11, n21, n23, n32, n41, n31 (precision needed before acting), n38.
- **Where neighbours are found (the feature space or distance):** n12, n21, n42, with `similarity_space.py`; n02 weights in that space.
- **Labels as delivered, no injected noise:** n18 (shuffled-score control), n02, n37, n26, n27; QM9 at level 0 in every experiment.
- **Real repeat or curated measurements:** n06, n17, n19, n22, n26, n29, n36 (mostly hERG IC50 or ChEMBL records).
- **Labels clipped at an assay limit:** n30, n30b, constant lift in n07.
- **Offsets shared by a group of molecules:** n34 (scaffold family), n39 (measuring source), n29 (ChEMBL document).
- **Changing the model or the loss instead of the labels:** n09, n13 (twelve LNL methods), n33, n40, n32 (two-learner exchange).
- **Activity cliffs and cleaning:** n01, n14, n25, cliff-aware gap in n07 and n02.
- **Damage and the hybrid search itself:** n15, n16, n20.
- **How large the noise is against model error:** n37, n38, n27, noise_ceiling.


---

## Federated (separate-site) experiments — `tests/experiments/federated/`

**Overview.** Five experiments, fed01 to fed05, simulate data held at separate sites ("silos" in the code, "sites" in the thesis). They ask whether feature selection for the adaptive hybrid representation can run at each site and still match one selection on pooled data. The external comparator is FLuID, the federated knowledge-distillation process of Lhasa Limited (cited in the thesis as Hanser2025, with Bassani2023 and Hanser2023). The code imports FLuID's own functions for Morgan fingerprints, k-means site clustering and teacher (site) models (`federated_helpers.py`). The thesis states the separate sites are simulated from two public datasets, Caco-2 and LogD (Collaborations paragraph, thesis.tex line 52; `sec:method:sites`). Lhasa's own hERG set (`lhasa_herg`) was the intended third dataset; it is listed as cut in `/Volumes/seagate/kirby_results_2026-10-09/LEDGER.md`, line 7. All runs use five outer folds grouped by scaffold family, three seeds, two downstream models (DNN and SVM), and feature budgets of 500 to 2,000 features. Since 2026-10-05 every hybrid comes from one search per outer fold on that fold's training molecules. This search runs on the training molecules of all sites together, which a real federated setting could not do; the thesis states this. Results feed Chapter 5 (`chap:method`), Section `sec:method:results_sites`, and Additional file `app:add:separate_sites`.

**Three different meanings of "silo" in this code.** This is the key fact for anyone splitting the work.
- Horizontal split by chemistry (fed01, fed04): the training molecules of one dataset are split into five sites by FLuID's k-means on Morgan fingerprints (`cluster_training_only`, k=5). A `smooth_factor` (default 0.05) randomly reassigns that fraction of molecules to other clusters. Only training molecules are clustered; held-out molecules are assigned by a k-means model refitted to the smoothed clusters.
- Vertical split by representation (fed02, fed03 Part B): every site holds the same molecules, but each computes different representations. In fed02 each site holds one representation; in fed03 Part B each holds one family of representations.
- Split by real data source (fed03 Part A, fed05): ChEMBL hERG Ki versus Lhasa's training and test files (Part A), or three different antibody datasets (fed05). Neither has server results.

**What crosses a site boundary** (`federated_feature_selection`, `federated_helpers.py` line 1610, docstring). Each site runs feature selection on its own molecules. It shares only feature indices, their importance values, and one validation score. A coordinator splits the total budget across sites. The default rule is proportional to the site scores, with at least one feature per site that succeeded (`allocate_budget_across_silos`). fed03 Part B instead uses an even split (`_split_budget`). Two things are not private in the code. The search that picks the settings (allocation method, importance method, internal evaluator) runs on all sites' training molecules pooled. The downstream model is then fitted once on the joined selected features of all training molecules, not per site. For the FLuID comparators, only site models' predictions cross a boundary; the thesis appendix states this.

**Per-dataset drivers.** `caco2.py`, `logd.py` and `herg.py` each call `federated_helpers.run_all` for one dataset with fed01–fed04. `herg.py` runs `lhasa_herg`, which is cut. `lhasa_herg_loader.py` loads Lhasa's files: training 7,772 molecules ("Lhasa Chembl Call" labels), test 4,383 molecules ("Overall Conservative Call"), and 10,000 unlabelled FLuID transfer molecules (counts from its docstring, not re-counted). Two-class cut-offs for Caco-2 and LogD are in `CLASSIFICATION_CUTOFFS`: log10 efflux ratio ≥ log10(2), i.e. P-gp substrate (756 of 2,158 molecules positive, per code comment); logD7.4 ≥ 3 (1,242 of 5,028 molecules above, per code comment); hERG pKi ≥ 6.0.

---

### fed01 — Unequal label flips across five sites  (`tests/experiments/federated/fed01_noise_heterogeneity.py`)
- **Question:** When sites differ in how many of their labels are wrong, does selecting features at each site beat FLuID's prediction ensembles?
- **How it works:**
  - Labels are made two-class (cut-offs above). Metric is MCC on the clean held-out labels, higher is better.
  - Five sites are k-means clusters of the fold's training molecules.
  - Flip levels 0, 0.05, 0.1, 0.15, 0.2, 0.3. At level s, site 1 gets 0 flipped and site 5 gets s, linearly between (`np.linspace(0, s, 5)`).
  - Arms: C0 the same hybrid selected once on all training molecules pooled; base1/base2 the two best single representations; C1 one model on pooled Morgan fingerprints; C2 FLuID distance-weighted ensemble; C3 FLuID score-weighted ensemble; C4 FLuID student model; C5 selection at each site (proportional budget split); C6 = C5 after each site removes molecules flagged by noise detectors.
  - C1–C4 use FLuID's Morgan fingerprints with the run's own model; they see no search.
  - The search runs once per fold on the clean labels and is shared by every flip level (`SEARCH_LABELS='clean'`).
  - Also: noise-detector analysis (8 detectors), correlation of each site's flip rate with the features it was given, and a `smooth_factor` sweep (0 to 0.2).
- **Status:** Partial. caco2: DNN 3 seeds, SVM 2 seeds finished, 5 folds (answer page fed01.md; thesis appendix comment). logd: no finished seed. lhasa_herg: cut. FAULTS.md: C6 gives a site 0 features in 24 to 48 of 180 builds per seed, so C6 against C5 is blocked; the thesis does not report C6. FAULTS.md also notes C2–C4 use Morgan fingerprints and combine predictions while C5 uses the dataset's representations and combines features, so they differ in two ways at once.
- **Results so far:** caco2, budget 500, hybrid1, MCC, means over seeds (thesis appendix table `tab:method:flipped_sites`, not re-checked against the raw file). DNN at 0% flipped: pooled selection 0.520, per-site selection 0.509, pooled Morgan model 0.443, distance ensemble 0.390, score ensemble 0.344, student 0.391. DNN at 30% flipped: pooled 0.4324, per-site 0.4323. SVM at 20% flipped: per-site 0.510 vs pooled 0.507. SVM at 0%: pooled 0.537, per-site 0.515.
- **In thesis:** Chapter 5, Additional file `app:add:separate_sites`, Table `tab:method:flipped_sites`; described in `sec:method:sites`.
- **Related:** fed04 (same clustering), fed03 (same per-site selection function), noise suite (flip injection via noiseInject).

### fed02 — Five representations as five sites  (`tests/experiments/federated/fed02_rep_diversity_silos.py`)
- **Question:** If each site computes a different representation of the same molecules, does one hybrid over all sites beat FLuID's ensemble of per-site models?
- **How it works:**
  - Regression on the dataset's own labels; metric RMSE, lower is better.
  - Five sites, one representation each, taken from the dataset's pruned pool. Stored in the result files: caco2 = ecfp4, graph_kernel, molformer, pdv, avalon; logd = ecfp4, mordred, molformer, chembert, avalon.
  - C1 best single site (picked on training molecules); C2 FLuID distance-weighted ensemble (8 nearest neighbours); C3 FLuID score-weighted ensemble (inverse RMSE); C4 one hybrid over all five representations; C5 hybrid over the top three sites; C6 k-means within each site (3 sub-clusters) then per-cluster selection across sites (`fluid_within_kirby_across`).
  - C4 is a pooled selection over all five representations (`_build_selected`), not a per-site selection.
  - Two best hybrids per fold and budget; two best single representations as extra baselines.
- **Status:** Ran on server, caco2 and logd, both models, 3 seeds, 5 folds (`*.json`, status ok in LEDGER.md). lhasa_herg cut. FAULTS.md: the `aggregated` block holds means over 3 seeds but per-fold lists from seed 0 only. logd's fourth site is `chembert`, not `chemberta`; whether its stored features match current code is unchecked.
- **Results so far:** RMSE, mean over 3 seeds of the 5-fold mean (checked against `fed02_rep_diversity_silos/{caco2,logd}.json`, `per_model.<model>.aggregated`, this session):
  - caco2 DNN: best single site 0.3063, C4 hybrid 0.2837, C2 0.2730, C3 0.2721, C5 0.2824, C6 0.2857.
  - caco2 SVM: best site 0.3004, C4 0.2898, C2 0.2985, C3 0.2999, C5 0.2940, C6 0.2908.
  - logd DNN: best site 0.4514, C4 0.4269, C2 0.4328, C3 0.4285, C5 0.4350, C6 0.4383.
  - logd SVM: best site 0.5065, C4 0.4468, C2 0.4949, C3 0.4898, C5 0.4535, C6 0.4501.
  - The answer page fed02.md quotes caco2 DNN as C3 0.2720 vs C4 0.2858; this differs from the aggregated values above, source of difference not checked.
- **In thesis:** Chapter 5, Additional file `app:add:separate_sites`, Table `tab:method:site_reps` (C1, C4, C2, C3 only).
- **Related:** fed03 Part B (same vertical split, by family instead of by representation).

### fed03 — Representation families as sites (Part B); real data sources as sites (Part A)  (`tests/experiments/federated/fed03_synthetic_private.py`)
- **Question:** Part B: if each site holds one family of representations, does selecting at each site match one selection over all families? Part A: does selection at sites work when the sites are real, different hERG data sources?
- **How it works:**
  - Part B (any small-molecule dataset): every representation in the dataset's pool is sorted into one of three families by `REP_FAMILIES`: structural (fingerprints, graph kernels), physicochemical (descriptors, 3D shape), learned (pretrained embeddings). Each family is one site holding the same molecules.
  - C1 best single family, picked on training molecules; C2 one selection over all families pooled; C3 one selection per family at an even share of the budget (`_split_budget`); base1/base2 the two best single representations.
  - If a family has fewer features than its share, the rest goes to the other families. C3 can therefore build fewer features than C2.
  - Metric: the dataset's own, RMSE on caco2 and logd.
  - Part A (herg_ki, herg, lhasa_herg only): site A = ChEMBL hERG Ki (1,415 molecules, pKi cut at 6.0), site B = Lhasa training (7,772), site C = Lhasa test (4,383), de-duplicated by InChIKey across sites (`dedup_across_silos`). Metric AUROC. Counts are from the docstring, not re-counted.
- **Status:** Part B ran on server, caco2 and logd, both models, 3 seeds, 5 folds (status ok). For SVM the three seeds give identical numbers (seed std 0.0, checked). Part A has no current server result: it runs only on lhasa_herg/herg datasets, and lhasa_herg is cut. The docstring records an earlier Part A result of 0.5504 MCC vs 0.4005 for the best single source at a fixed 100 features (before 2026-09-23); that run used an older design and is not in the current results. FAULTS.md: C3 builds a different number of features from C2 on every fold at 500 features, so C3 vs C2 differs in two ways at once. On logd fold 0 the structural family is avalon alone. The `sweep.silos` field in the result files says 5, though Part B has three families.
- **Results so far:** RMSE, mean over 3 seeds of the 5-fold mean (checked this session against `fed03_synthetic_private/{caco2,logd}.json`; matches thesis Table `tab:method:family_sites`):
  - caco2 DNN: best single representation 0.3052, best family 0.2877, one selection over all families 0.2839, per-family selection 0.2834.
  - caco2 SVM: 0.3037, 0.3009, 0.2847, 0.2845.
  - logd DNN: 0.4527, 0.4457, 0.4191, 0.4204.
  - logd SVM: 0.4928, 0.4598, 0.43823, 0.43822.
- **In thesis:** Chapter 5, Section `sec:method:results_sites`, Table `tab:method:family_sites` (the main separate-site experiment); method in `sec:method:sites`.
- **Related:** fed02 (vertical split by single representation); fed01 (per-site selection under flipped labels).

### fed04 — Feature selection per chemical cluster  (`tests/experiments/federated/fed04_cluster_aware_importance.py`)
- **Question:** Do different regions of chemical space need different representations, and does selecting per region help?
- **How it works:**
  - Two-class labels, MCC. Five k-means clusters of training molecules (FLuID clustering, smooth_factor 0.05).
  - Importance computed per cluster; compared between clusters by mean pairwise Spearman correlation of the representations' importance totals. Null control: random parts of the same sizes.
  - Arms: `global_hybrid` (one selection on all training molecules); `cluster_features_all_rows` (features selected per cluster, one model on all molecules); `cluster_conditional_hybrid` (a separate model per cluster, trained on that cluster only, held-out molecules routed by k-means).
  - Also a smooth_factor sweep (0 to 0.2).
- **Status:** caco2 ran, both models, 3 seeds, 5 folds. logd partial (DNN 3 seeds, SVM 2 seeds per answer page). lhasa_herg cut. FAULTS.md: fed04's folds are drawn on the two-class labels, so they are different molecules from fed01–fed03's folds (caco2 fold 0: 1,745 vs 1,726 training molecules). `global_hybrid_train_rows` stores a training-molecule score under the same key as held-out scores. caco2 seed 2 fold 3: cluster 4's selection failed and its 32 held-out molecules went to the global model.
- **Results so far:** caco2, MCC, mean over 3 seeds (thesis Table `tab:method:cluster_sites`; agreement values re-checked in `caco2.json` this session): DNN global 0.532, per-cluster selection with one model 0.529, model per cluster 0.486; SVM 0.535, 0.539, 0.499. Agreement between clusters vs random parts: DNN 0.585 vs 0.654, SVM 0.525 vs 0.618 mean Spearman.
- **In thesis:** Chapter 5, Additional file `app:add:separate_sites`, Table `tab:method:cluster_sites`.
- **Related:** fed01 (same clustering); complexity experiment c07 (the docstring says fed04 extends it).

### fed05 — Three antibody datasets as sites  (`tests/experiments/federated/fed05_antibody_federated.py`)
- **Question:** Do the fed01/fed02 questions hold on antibodies, with three different antibody datasets as sites?
- **How it works:**
  - Sites: ddg_combined (binding free-energy change, kcal/mol), abagym_EGFR (deep mutational scan binding score), flab_thermo_tresanco (melting temperature, °C). Docstring gives 518, 659 and 672 rows; FAULTS.md says ddg_combined is 429 in another stored file.
  - C1 per-site baselines and hybrids; C2 importance transfer across sites; C3 ESM2-embedding k-means within a site, importance stability across clusters; C5 pooled binding (DDG + DMS, z-scored); C6 different representation subsets per site.
  - C4 (noise heterogeneity) recorded as not run, decided 2026-10-06: the sites share no representation. C7 and C8 (FLuID ensembles) skipped: sites have different feature spaces and label units.
  - Models DNN and SVM; regression metrics.
- **Status:** Never ran. Its job runs inside the lhasa_herg job, which is cut. The caco2 and logd result files are skip markers by design (answer page fed05.md; LEDGER.md runtimes 34.6 s and 3.6 s).
- **Results so far:** none.
- **In thesis:** not in thesis.
- **Related:** fed01, fed02 (same questions, small molecules); the OPIG antibody experiments (same datasets).

---

**Groupings**
- Per-site selection vs pooled selection: fed03 Part B (main), fed01 (with flipped labels), fed04 (per cluster).
- Hybrid vs FLuID prediction ensembles: fed01 (horizontal split), fed02 (vertical split), fed05 (planned, skipped).
- Vertical split, sites hold different representations of the same molecules: fed02, fed03 Part B.
- Horizontal split by k-means clusters of one dataset: fed01, fed04.
- Real data sources as sites (no current results): fed03 Part A (Lhasa vs ChEMBL hERG), fed05 (antibodies).


---

# OPIG suite (`tests/experiments/opig_*.py`)

Fifteen scripts filed in the repo under the Oxford Protein Informatics Group collaboration. Each
one takes a published study's dataset or protocol and asks whether the adaptive hybrid
representation (features selected from several representations, budget in features) beats the
single representation the same search ranked first. Four topics: small-molecule and reaction
prediction (activity cliffs, CGR reaction features, component splits), antibody prediction
(developability, binding free energy change, humanness, binder enrichment, mutation scans),
properties of the
representation itself (molecules that share a feature row, scaffold-diff similarity spaces), and
two transfer settings (a recipe found on a large dataset reused on a small one, a generator's
in-loop predictor).

Three of these reach the thesis as results, all inside Chapter~\ref{chap:method} §Extension
studies (`sec:method:results_opig`): the activity-cliff targets, the shared-row count and its
remedy, and the recipe transfer from LogD to hERG $K_i$. The shared-row census also feeds
Chapter~\ref{chap:flaws} (`sec:flaws:ties`, `tab:flaws:ties`). The thesis Collaborations section
(thesis.tex line 48) names supervision by Garrett Morris and Lhasa Limited only; OPIG is not named
there, and "OPIG" appears nowhere in thesis.tex. Common settings: 5 outer folds grouped by
scaffold family (Ginkgo properties use the competition's own 5 cluster folds), the dataset's two
decided downstream models and decided search grid (`DECISIONS.md` 2026-10-03), pruning off, and
per fold the 2 best single representations and 2 best hybrids from one search on that fold's
training rows (`fold_standard_selection`, 2026-10-05).

## Antibody-specific representations these experiments use

Defined in `src/kirby/representations/antibody.py` and `src/kirby/representations/mutation.py`;
listed in the thesis as `tab:reps_antibodies`.

- **CDR pooling.** `get_cdr_residue_indices` numbers a chain with ANARCI under the IMGT scheme and
  returns the residue indices inside the three CDRs; if ANARCI fails to number the sequence the
  function returns every residue index, so the representation silently becomes whole-chain
  pooling. `embed_*_cdr` (ESM-2, ESM C, ESM3, ProtT5, AntiBERTy, Ab-RoBERTa, SaProt, AMPLIFY) mean
  over those residues only, heavy and light pooled separately then joined.
- **Per-region readouts.** `embed_esm2_per_cdr` pools CDR1, CDR2 and CDR3 separately and joins
  them, 3,840 features per chain. `embed_esm2_per_region` does the same for all seven IMGT regions
  (FR1–FR4, CDR1–CDR3), 8,960 features per chain. The thesis table calls these ESM-2 per CDR and
  ESM-2 per region.
- **Mutation features** (`create_mutation_features`, 48 features). Per variant, averaged over its
  mutations: wild-type residue one-hot (20), mutant residue one-hot (20), BLOSUM62 substitution
  score, change in Kyte-Doolittle hydrophobicity, change in residue weight in daltons, change in
  charge at pH 7, mutated position divided by sequence length, a heavy-chain flag, the number of
  mutations, and a flag for wild type equal to mutant.
- **Shift from parent** (`create_esm2_wt_delta`, 2,560 features). The parent sequence is
  reconstructed from the mutant and its mutation list, both are embedded with ESM-2 and mean-pooled,
  and the representation is the difference. The thesis calls it ESM-2 shift from parent.
- **Mutation-site window** (`create_esm2_mutsite`, 1,280 features): per-residue ESM-2 on the mutant,
  pooled over residues within 3 positions of each mutation.
- **Log-likelihood ratios** (`create_esm2_llr`, `create_esmc_llr`, `create_antiberty_llr`, 3
  features each): masking the residue in the parent sequence, the sum, mean and minimum of the
  mutant-minus-parent log-probability. Used as augmentation features, not as single
  representations.
- **Paratope** (8 features) and **structural** (28 features): sequence-based summaries of the CDRs
  from their boundaries, and solvent-accessible surface area plus model confidence from a predicted
  structure, plus the paratope features. Both are augmentation features; in
  `opig_ginkgo_developability` they are kept out of the search pool and scored alone.
- **Scaffold-diff representations** (`tests/experiments/similarity_metrics/scaffold_reps.py`), used
  by `opig_scaffold_similarity` and `opig_scaffold_hybrid` and registered nowhere in the pipeline:
  MutID, a one-hot over (position × mutant residue); dPhysChem, the summed per-mutation change in
  standardised residue properties, position-agnostic; dPhysChem_pos, that change placed at the
  mutated position; SubScore, 4 features (summed Grantham, summed BLOSUM distance, number of
  mutations, mean position); BaseDelta, the full-sequence one-hot minus the parent's, so -1 at the
  parent residue and +1 at the mutant.

---

### opig:qsar — hybrid on three activity-cliff targets  (`tests/experiments/opig_activity_cliff_qsar.py`)
- **Question:** On three targets rich in activity cliffs, does the hybrid predict activity better
  than the single representation the same search ranked first, and does it call a cliff better?
- **How it works:**
  - Three targets, each its own prediction task: dopamine D2 (6,333 compounds), factor Xa (3,605),
    SARS-CoV-2 main protease (1,924, from COVID Moonshot).
  - Representations: `DATASET_REPS['chembl_qsar']['general']`, 10 requested and 8 built (atom pairs,
    Avalon, ECFP4, WL graph kernel, MHG-GNN, MoLFormer, Mordred, PDV). GROVER and SMI-TED were not
    built.
  - Models: LightGBM and the DNN. 5 scaffold-grouped outer folds. Per fold and model, 2 best single
    representations and 2 best hybrids from the search on that fold's training molecules; no
    distance hybrid; 0/1 features left unscaled.
  - Prediction metric: MAE in log units of activity, lower better. Cliff metric: a matched pair
    whose activities differ by at least 2.0 log units is a cliff, pairs 1.0–2.0 apart are dropped, a
    pair is called a cliff when the two predictions differ by more than 1.5 log units, scored by
    Matthews correlation over three partitions (both molecules held out, one molecule known, all
    pairs).
- **Status:** ran on the server, `ok 22.0h`, all three targets, 5 folds (LEDGER.md line 52).
- **Results so far:** thesis `tab:method:opig_qsar`, held-out MAE in log units, mean over 5 folds,
  top-ranked single representation then hybrid: D2 LightGBM 0.519 then 0.514; D2 DNN 0.523 then
  0.499; factor Xa LightGBM 0.685 then 0.714; factor Xa DNN 0.737 then 0.661; main protease LightGBM
  0.465 then 0.452; main protease DNN 0.500 then 0.490. The answer page adds that on factor Xa with
  the DNN the hybrid is lower on all 5 folds, and that the cliff call is unsettled: on main protease
  with LightGBM the hybrid's cliff MCC is 0.3745 against atom pairs' 0.3474, the highest of the 8
  single representations, while with the DNN on the same target it is 0.0283 and below six of them
  (answer page `opig_qsar.md`, numbers not re-checked against `qsar.json` here).
- **Known faults open:** every DNN score in this copy comes from a DNN whose predictions were not
  clipped to the training-label range (FAULTS.md A1). Four main-protease pairs have cliff MCC stored
  as 0.0000 with a floor flag because no cliff was predicted (FAULTS.md).
- **In thesis:** Chapter~\ref{chap:method}, `sec:method:results_opig`, `tab:method:opig_qsar`;
  design described at `sec:method:sites`.
- **Related:** the same hybrid-against-best-single question as every framework experiment; the
  cliff half is related to the AC material of Chapter~\ref{chap:flaws}.

### opig:shared_vectors_census — how many molecules a representation cannot tell apart  (`tests/experiments/opig_shared_vectors_census.py`)
- **Question:** For each representation on each dataset, how many samples get a feature row that
  another sample also has, and what is the best score any model could then reach?
- **How it works:**
  - Reads cached representation matrices and counts repeated rows. Fits no model in this half.
  - Each count is split by cause: the same input twice (same full InChIKey, or same heavy and light
    sequence), a failed featurisation that wrote all zeros, and two genuinely different samples the
    representation merges. A fourth column counts how many of the third kind differ only in
    stereochemistry.
  - `n_still_tied_if_concatenated` counts, for a pair of representations, the samples still sharing a
    row when the two matrices are placed side by side.
  - The tie count is turned into a lower bound on error: the best prediction for a tied group is its
    mean label, giving RMSE_min and the matching best $R^2$, over the whole dataset, not a held-out
    fold.
  - A hybrid half (on by default) runs or reads a hybrid search per outer fold and counts ties in
    each built hybrid: the general search's best two per decided model, the distance search's best
    per decided model, and the similarity search's best.
- **Status:** ran on the server, status `running` at the 9 October copy, and the partial file stops
  after the ten datasets alphabetically up to ddg_combined. hERG $K_i$, LogD and the Ginkgo
  properties were never reached by this run; their numbers in the thesis come from an earlier run
  (unverified here).
- **Results so far:** thesis `tab:flaws:ties` holds the per-representation counts on hERG $K_i$,
  LogD and Caco-2. Quoted there: the WL graph kernel ties 105 of 1,412 hERG $K_i$ molecules, 56 of
  them stereoisomers and 49 different 2D structures; ECFP4, atom pairs, PDV, Mordred and SMI-TED tie
  the same 56 hERG $K_i$ molecules; MACCS keys tie the most on every dataset (804 of 2,158 on
  Caco-2) and mostly not stereoisomers; ChemBERT, GROVER, MHG-GNN, MoLFormer and GraphMVP tie none;
  joining all of a dataset's representations leaves no molecule tied. The lowest reachable RMSE on
  hERG $K_i$ is at most 0.134 p$K_i$ (MACCS keys) against a label SD of 0.915 p$K_i$. On antibodies
  the thesis reports amino-acid composition giving 432 of the 659 EGFR-scan variants a shared row.
  The answer page adds, from the 9 October run, structural merging 332 of 418 on the Ang2 scan and
  MACCS merging 57 of 3,086 on adme_hlm.
- **Known faults open:** `n_representation_merged` leaves repeated inputs out of the representation
  rows and includes them in the hybrid rows, so the two are not comparable (on the Ang2 scan, 0 for
  every representation and 38 for every hybrid). Repeated input on antibody datasets is keyed on the
  heavy and light sequence only, so on ddg_combined two mutants that differ only in the mutation are
  called the same input (FAULTS.md).
- **In thesis:** Chapter~\ref{chap:flaws}, `sec:flaws:ties` and `tab:flaws:ties`; the hybrid
  question continues in `sec:method:results_opig`.
- **Related:** `opig_shared_vectors_remedy` is its fix-and-measure companion.

### opig:shared_vectors_remedy — can a hybrid separate molecules one representation merges  (`tests/experiments/opig_shared_vectors_remedy.py`)
- **Question:** When features of partner representations are added to a representation that merges
  molecules, do both the held-out score and the number of molecules sharing a row move?
- **How it works:**
  - Target representation on hERG $K_i$: the WL graph kernel, chosen by measurement as the
    representation on the dataset's pruned list that merges the most genuinely different molecules.
  - Two partner pairs: ECFP4 with PDV, both blind to stereochemistry and both tying the same 56
    molecules; and ECFP4 with MHG-GNN, which together leave no two molecules sharing a row.
  - Per outer fold, one search on that fold's training molecules over the target plus one pair; per
    decided model the 2 best hybrids and the 2 best single representations of that pool; the target
    alone is fitted and scored by the same scorer. 5 outer folds.
  - Ties are reported three ways per fold: of the target at full width, of the target's features the
    hybrid kept, and of the whole built hybrid. The middle one separates "the partner broke the tie"
    from "the budget changed the width".
  - Added-donor conditions (B and D) run only where augmentation is kept, the distance search of
    ginkgo_colloidal and ginkgo_hic, adding paratope and structural features.
  - Before fitting, sampled rows of every rebuildable representation are rebuilt and compared with
    the cached matrix; the graph kernel is rebuilt whole because its features depend on every
    molecule.
- **Status:** ran on the server, `ok 0.6h`, hERG $K_i$ only. Conditions B and D did not run; the
  Ginkgo properties did not run (FAULTS.md). Cache check records grover, mordred and smited as not
  checked.
- **Results so far:** thesis `tab:method:shared_rows`, held-out RMSE in p$K_i$ units, mean over 5
  folds, with the count of molecules sharing a row: graph kernel alone SVM 0.611, DNN 0.624, 105
  molecules; graph kernel with ECFP4 and PDV SVM 0.575, DNN 0.580, 56 molecules; graph kernel with
  ECFP4 and MHG-GNN SVM 0.571, DNN 0.581, 0 molecules. The thesis states the SVM improvement holds
  on all 5 folds for the second pair, and reads the pattern as most of the gain coming from the
  added information rather than from breaking ties. The answer page gives the same comparison as
  0.6109 to 0.5715 RMSE (ECFP4 with MHG-GNN) and 0.6109 to 0.5755 (ECFP4 with PDV).
- **In thesis:** Chapter~\ref{chap:method}, `sec:method:results_opig`, `tab:method:shared_rows`.
- **Related:** `opig_shared_vectors_census`.

### opig:blueprint_transfer — reusing a recipe found on a large dataset on a small one  (`tests/experiments/opig_blueprint_transfer.py`)
- **Question:** Do the search settings, or the selected features themselves, found on a large
  dataset transfer to a smaller dataset of the same kind?
- **How it works:**
  - Source LogD (5,028 molecules), target hERG $K_i$ (1,412). Pool is the 7 representations the two
    share: Avalon, ChemBERT, ECFP4, GROVER, MHG-GNN, Mordred, SMI-TED.
  - Four conditions scored inside every target fold: the best single representation re-picked per
    fold; a hybrid searched on that fold's training molecules (the upper reference); the source
    recipe with features selected again on the target; the source's own selected feature indices
    applied unchanged, with only the downstream model refitted.
  - The source recipes come from one search over every LogD molecule, none of which is a target row.
    The two transferred recipes are required to have distinct selected features.
  - Models: the two decided for both datasets, the DNN and SVM. 5 scaffold-grouped folds on each
    dataset. Metric RMSE in p$K_i$ units. A source-side check scores a per-fold hybrid against the
    per-fold best single representation on LogD.
- **Status:** ran on the server, `ok 7.6h`, the one pair LogD to hERG $K_i$, 5 folds on each side.
  No other pair ran.
- **Results so far:** thesis `tab:method:settings_transfer`, held-out RMSE in p$K_i$ units, mean
  over 5 folds: SVM best single 0.589, searched on hERG $K_i$ 0.563, LogD settings with features
  reselected 0.567, LogD settings and features 0.569; DNN 0.604, 0.567, 0.589, 0.593. The thesis
  notes the SVM transfer beats the best single representation on all 5 folds and stays behind the
  search run on hERG $K_i$ itself. The answer page gives base1 as 0.5888 (SVM) and 0.6035 (DNN),
  recipe transfer minus native as +0.0034 (SVM) and +0.0223 (DNN), full transfer minus native as
  +0.0061 and +0.0259, and the source-side check on LogD as 0.4928 to 0.4362 (SVM) and 0.4513 to
  0.4089 (DNN). It also records that the target's decided grid holds a budget of 250 features,
  treeshap importance and softmax_T1 allocation that the source's grid does not, so a transferred
  recipe cannot use them.
- **In thesis:** Chapter~\ref{chap:method}, `sec:method:sites` (design, with Meta-QSAR named as the
  closest prior work) and `sec:method:results_opig`, `tab:method:settings_transfer`.
- **Related:** the separate-sites study of `sec:method:results_sites` asks the sibling question
  across sites rather than across datasets.

### opig:ginkgo — antibody developability properties  (`tests/experiments/opig_ginkgo_developability.py`)
- **Question:** Which antibody representation predicts each biophysical developability property
  best, and does a hybrid help on 94 to 242 antibodies?
- **How it works:**
  - Properties of the 2025 Ginkgo competition set (GDPa1 v1.2, 242 antibodies), each its own
    dataset: HIC retention, expression titer, CHO protein recovery, AC-SINS self-interaction, SMAC
    colloidal stability.
  - Folds are the competition's own 5 hierarchical cluster folds, not the scaffold folds used
    elsewhere. Per fold one general search on the training antibodies on the property's decided
    grid, pruning off, no augmentation; per decided model the 2 best single representations and the
    2 best hybrids, with a second hybrid skipped when its selected features equal the first's.
  - Paratope and structural features are kept out of the search pool and scored alone. No distance
    and no similarity hybrid. Metric Spearman $\rho$, higher better.
- **Status:** ran on the server, status `running` in the copy; only HIC retention has scores, on 4
  of the 5 competition folds. Titer, CHO recovery, AC-SINS and SMAC have no scores from this run.
- **Results so far:** on HIC retention neither hybrid beats the single representation the search
  ranked first. With the DNN, ESM-2 on the joined chains gives 0.5529 Spearman against the first
  hybrid's 0.4728 (answer page `opig_ginkgo.md`, 4 folds). The same representation is also the best
  of those scored alone. Models used: GP and DNN for HIC.
- **Known faults open:** every GP result on ginkgo_acsins, ginkgo_hic and ginkgo_pr_cho in this copy
  came from a GP whose kernel length scale started at 1, which left fits predicting one value and
  Spearman undefined (FAULTS.md A2).
- **In thesis:** the five Ginkgo properties and their competition folds are in the dataset table
  `tab:datasets` and carry results throughout Chapters 5 to 7, but those come from the framework and
  noise suites. This experiment's own run is not cited; the HIC comparison above appears nowhere in
  thesis.tex. The thesis reports HIC with ESM-2 on the joined chains as the best single
  representation for both its models, which agrees in direction with this run.
- **Related:** `opig_shared_vectors_remedy` (its B and D conditions target the same two Ginkgo
  properties), `opig_humanness`.

### opig:graphinity — binding free energy change on antibody mutation  (`tests/experiments/opig_graphinity_ddg.py`)
- **Question:** Can sequence-based representations plus ordinary models approach a structure-based
  equivariant graph network at predicting the change in antibody-antigen binding free energy?
- **How it works:**
  - Dataset ddg_combined, 429 measured antibody-antigen mutants from SKEMPI, label in kcal/mol.
  - Condition `experimental`: 5 outer folds on the measured mutants; per fold a search on the
    training rows gives each of the two decided models its 2 best single representations and 2 best
    hybrids.
  - Condition `transfer_foldx`: train on up to 5,000 FoldX-computed mutants and score on every
    measured mutant. Single representations only, no hybrid search.
  - 8 representations requested: mutation features, one-hot, AntiBERTy CDR, ESM-2 layer 24, ProtT5
    CDR, ESM-2 joined chains, ESM-2 shift from parent, AntiBERTy.
  - Two conditions that trained and scored inside the FoldX and Rosetta sets were removed in
    September 2026, with the datasets, because their labels were simulation output rather than
    measurements. Augmentation is cut on this dataset.
- **Status:** nothing ran to a score. The `experimental` result file holds only the error "2 of 8
  requested representations were not computed" (AntiBERTy CDR and ProtT5 CDR), stored as a finished
  result although the ledger says `running`. `transfer_foldx` is absent from the result file; its
  partial file holds 6 data lines (FAULTS.md, answer page `opig_graphinity.md`).
- **Results so far:** none.
- **In thesis:** the $\Delta\Delta G$ dataset is in `tab:datasets` and carries results in Chapters 5
  to 7 from other suites, and the thesis notes at line 2180 that in the $\Delta\Delta G$ hybrid the
  representations built for this thesis carry most of the drop in score under shuffling. This
  experiment contributes no number.
- **Related:** `opig_scaffold_similarity` uses the same multi-reference mutation data as a
  similarity space.

### opig:cgr_comparison — do reaction-difference features add anything  (`tests/experiments/opig_cgr_comparison.py`)
- **Question:** Does adding condensed-graph-of-reaction features to the pool change what the
  search finds, and does either the best single representation or the hybrid improve?
- **How it works:**
  - Two pools compared per dataset and per fold: `without_cgr`, the dataset's pruned 10 minus the
    CGR Morgan fingerprint, 9 representations; `with_cgr`, the pruned 10 plus Chytorch Graphormer and
    the CGR SMILES language model, 12 representations.
  - Datasets: Buchwald-Hartwig (3,955 reactions), the Doyle plates set (4,599), Suzuki (5,760).
    Models are the dataset's two decided ones: RF with the DNN on the plates set, RF with LightGBM
    on the other two.
  - Per group and fold, one search on the training rows; per model the 2 best single representations
    and the 2 best hybrids. The CGR difference is `with_cgr` minus `without_cgr` for each of those
    four slots, paired by fold. Metric RMSE in yield percent, lower better.
  - A representation matrix whose features are all constant, or that holds a non-finite value, is
    refused and recorded rather than scored; the cached Chytorch Graphormer matrix had been all
    zeros.
- **Status:** partial. Status `running` in the copy; the partial file holds Buchwald-Hartwig only,
  with 5 folds for `without_cgr` and 3 folds for `with_cgr`. The plates set and Suzuki have nothing.
- **Results so far:** from the partial file, read this session, Buchwald-Hartwig, held-out RMSE in
  yield percent, mean over the three folds both groups scored (0, 1, 2), `with_cgr` then
  `without_cgr`: LightGBM first hybrid 16.66 then 17.48; LightGBM first single representation 17.36
  then 17.61; RF first hybrid 16.39 then 17.03; RF first single representation 15.91 then 16.65. The
  representations picked first were the CGR Morgan fingerprint or the MHG-GNN difference in the
  `with_cgr` group and the MHG-GNN or Avalon difference in `without_cgr`. Three folds is not the
  planned five, and no spread is quoted here.
- **In thesis:** not in thesis as a result. Buchwald-Hartwig itself is in `tab:datasets` with the
  CGR fingerprint named as its best RF representation, and `tab:reps_reactions` lists the reaction
  representations; the with-and-without comparison is absent.
- **Related:** `opig_kingsmith_component_splits` uses the same reaction datasets.

### opig:humanness — human against non-human therapeutic antibodies  (`tests/experiments/opig_humanness.py`)
- **Question:** Which antibody representation best classifies a therapeutic antibody as human or
  not, and does a hybrid improve on it?
- **How it works:**
  - Dataset humanness_therapeutic, the BioPhi IMGT/mAb-DB curation, 564 antibodies, 198 human
    against 366 not, grouped into 461 clusters at 95% sequence identity. Heavy chain only.
  - Classifiers: XGBoost and RF, the two decided for this dataset. Outer fold count from the
    experiment configuration, default 5. Metric AUROC.
  - The pool is the dataset's approved representations; one-hot left the experiment in October 2026
    because it is not on that list, leaving BLOSUM62 as the one classical encoding against the
    language-model embeddings.
  - The reference study scored 8 published humanness methods by sweeping a threshold on the whole
    dataset with no split, so its AUROC values are not comparable with a cross-validated one.
- **Status:** never ran. LEDGER.md line 51 lists `opig:humanness` as `pending`.
- **Results so far:** none from this experiment.
- **In thesis:** the dataset is in `tab:datasets` (binary, AUROC, RF with AntiBERTy CDR) and carries
  framework results, for instance `tab:hybrid_vs_best_single` at 0.9659 against 0.9614 AUROC for RF.
  Those come from the framework suite, not from this file.
- **Related:** `opig:ginkgo` (antibody panels rather than mutation scans).

### opig:scaffold_hybrid — scaffold-diff features in the real pool  (`tests/experiments/opig_scaffold_hybrid.py`)
- **Question:** Put the two mutation-difference representations into the real pool beside the
  language-model embeddings: do they beat the base representations standing alone, and how many
  features does the hybrid take from them?
- **How it works:**
  - Four single-parent antibody deep mutational scan datasets with mutation metadata: Koenig
    expression and binding, Adams expression and binding. The `phillips_h3` set was cut in July
    2026.
  - Pool: the dataset's pruned 10 plus dPhysChem and dPhysChem_pos, computed in this file from the
    exact sequences the loader returned so rows cannot misalign, and registered nowhere else.
  - Per dataset and outer fold: one search on the training rows on the dataset's decided grid; per
    decided model the 2 best single representations and the 2 best hybrids; each scaffold-diff
    representation also scored alone with its rank among every representation the search scored; the
    count of each hybrid's selected features that came from the two scaffold-diff representations.
  - Metric is the dataset's own: Spearman for the regression scans, AUROC for the binarised binding
    sets.
- **Status:** never ran on the server. LEDGER.md line 53 lists `opig:scaffold_hybrid` as `pending`.
  An earlier local measurement is recorded in the file's own header (Koenig expression, 3 outer
  folds, budgets 100 and 500 features): 9 of the 15 winning recipes selected no scaffold-diff
  feature at all, the random forest took none in any fold and still beat ProtT5 by 0.0366 Spearman,
  and the largest share anywhere was 18 of 300 features. Standing alone, dPhysChem ranked fourth to
  sixth of five to eight representations and dPhysChem_pos last wherever it scored. The header warns
  that the hybrid-minus-single delta this file prints is therefore not evidence about the
  scaffold-diff representations.
- **Results so far:** only that local measurement, which the project treats as debug rather than
  result.
- **In thesis:** not in thesis. The Koenig scans are in `tab:datasets`, and the thesis's
  mutation-aware representations are the ones in `tab:reps_antibodies`, not these two.
- **Related:** `opig_scaffold_similarity` (same representations, graded as similarity spaces rather
  than fitted).

### opig:scaffold_similarity — mutation-difference representations as similarity spaces  (`tests/experiments/opig_scaffold_similarity.py`)
- **Question:** Which mutation-difference representation and distance metric gives a space whose
  distances track an independent measure of how different two variants are, and what kind of feature
  earns its place in such a space?
- **How it works:**
  - No model is fitted. Every number is a Spearman rank correlation over sampled pairs of variants,
    between the distance the representation gives the pair and the distance a substitution matrix
    gives it.
  - Three modes: representations (representation × distance metric), features (which features earn a
    place, with the variants split in two halves so the subset chosen on one half is scored on the
    other; 32,000 sampled pairs by default, about 16,000 per half), and controls.
  - Three controls. Never grade a representation with a ground truth built on its own basis, so a
    physicochemical representation is graded by BLOSUM and never by Grantham. Partial out the count
    of changed positions, because the Grantham and BLOSUM ground truths sum over positions and
    largely count differences. Refuse a distance algebraically identical to its grader; this caught
    one whose apparent 0.519 was grader against grader.
  - Datasets: every scaffold-compatible antibody and protein set in the repo. Multi-reference sets
    compare variants only within one reference and pool the pairs.
- **Status:** never ran on the server. LEDGER.md line 54 lists `opig:scaffold_similarity` as
  `pending`. A local results document, `SCAFFOLD_SIMILARITY_RESULTS.md`, dates from 19 July 2026 and
  was not re-checked here.
- **Results so far:** none on the server.
- **In thesis:** not in thesis.
- **Related:** `opig_scaffold_hybrid`; the similarity suite (`s*` experiments).

### opig:generator_scoring_swap — the features a generator's in-loop predictor sees  (`tests/experiments/opig_generator_scoring_swap.py`, support files in `tests/experiments/opig_generator_scoring/`)
- **Question:** A molecule generator keeps a property predictor in its loop and hands it one
  fingerprint by convention. Change only the features that predictor sees: do the molecules that
  come out change?
- **How it works:**
  - The generator is REINVENT 4, Apache 2.0, pinned at commit 660d2c9, unedited. Its published
    reinforcement-learning configuration is copied byte for byte into `reinvent_published/` (batch
    64 molecules per step, 100 steps per stage, sigma 128, learning rate 0.0001, their diversity
    filter and two-stage termination). Their `ExternalProcess` component calls
    `kirby_reinvent_score.py` through a pipe, so no line of theirs changes in any arm.
  - Four arms, same model class (LightGBM) on the same 1,412 hERG $K_i$ molecules, same generator
    settings, only the features differ: every ECFP4 feature (the convention, the control); the
    blueprint the search picks with `objective='predictive'`; the same number of features drawn at
    random from the same representations; and every feature of whichever single representation scores
    best alone.
  - A fifth arm steered by REINVENT's own published scoring function was dropped in September 2026,
    because that function rewards molecular weight, drug-likeness and substructure filters and
    predicts no measured property, so its arm would differ from the control in two ways at once.
  - The generator is steered to maximise predicted hERG affinity, because the model that scores the
    output molecules is a hERG blocker classifier, and both sides then move the same way.
  - The output is scored by a model fitted on different molecules: the hERG binary blocker set, 7,772
    molecules, with the 166 molecules it shares with hERG $K_i$ removed first, leaving 7,606.
- **Status:** built, never ran. It is in no ledger row and has no result files. The thesis outline
  records it as future work.
- **Results so far:** none.
- **In thesis:** not in thesis; no mention of a generator anywhere in thesis.tex.
- **Related:** `opig_blueprint_transfer` (both ask what a chosen blueprint is worth away from the
  dataset it was found on).

### opig:trastuzumab — picking binders out of a CDRH3 variant library  (`tests/experiments/opig_trastuzumab_enrichment.py`)
- **Question:** Which representation best separates binders from non-binders in a library of
  trastuzumab CDRH3 variants, and does a hybrid put more binders in the top-ranked held-out
  variants?
- **How it works:**
  - Dataset `tz_her2_binary`: CDRH3 variants of the trastuzumab framework from Chinery et al. 2024,
    524,000 variants in the source files, subsampled to 5,000 stratified by label and by edit
    distance from the wild-type loop. Only the 10-residue CDRH3 varies; the full heavy chain is
    rebuilt by splicing each variant into the framework. Label is binder or non-binder.
  - Two arms. Zero-shot ranking uses a log-likelihood ratio directly as the score, with no model
    fitted; only `esmc_llr` and `antiberty_llr` are scored that way, which is the closest thing here
    to the published protocol. Supervised classification fits a model on the representation and
    ranks held-out variants by its score.
  - 13 representations: the dataset's approved 10 (BLOSUM62, ESM-2 per CDR, ESM-2 middle layers,
    ProtT5, ESM3, ESM C, Ab-RoBERTa CDR, AntiBERTy, mutation features, AbLang-2) plus three added
    back by the author on 2026-10-07 — the two log-likelihood ratios, without which the zero-shot arm
    produces no rows, and ESM-2, which is one of the four deep-learning methods the paper's headline
    comparison needs.
  - Models: XGBoost and LightGBM, the two decided for this dataset. Per outer fold the 2 best single
    representations and the 2 best hybrids of that fold's own search; no distance hybrid, because
    nothing here computes a neighbour or a distance. 5 outer folds. Decided grid: budgets 100, 250
    and 500 features, allocation performance_weighted, power_2 and rank_exp_d0.3, importance
    lightgbm_gain, internal evaluators match, rf, lightgbm and dnn, representation counts 3 and all.
  - Because the pool here is 13 representations and a per-fold search is filed by fold partition
    only, the search is given the tier marker `with_llr_esm2` so it cannot read back or overwrite
    the rows of the standard 10-representation searches other experiments run on this dataset.
  - Metric: the share of binders among the top 1%, top 5% and top 10% of held-out variants, and that
    share divided by the fold's own binder rate, plus AUROC. The file notes that at about 1,000
    held-out variants the top 1% is about 10 variants and varies a lot.
  - The published study measured something different: the share of a designed library that a CNN
    calls a binder, from 0.13 for random designs to 0.33 for the oracle, with BLOSUM62 at 0.26 ahead
    of all four deep-learning methods on experimental surface plasmon resonance. Those numbers are
    kept in the file for context, not as a comparison.
- **Status:** ran on the server, status `running` in the 9 October copy, attempt started
  2026-10-09T11:48. The partial file holds its header line only: the 13 representation names, 5
  folds requested, and no scored row.
- **Results so far:** none.
- **In thesis:** not in thesis. No mention of trastuzumab, HER2 or `tz_her2` anywhere in thesis.tex,
  and the dataset is not in `tab:datasets`.
- **Related:** `opig_humanness` and `opig_ginkgo` (antibody panels scored with the same selection
  rule); `opig_graphinity` and `opig_scaffold_hybrid` (single-parent variant libraries).

### opig:cdr_flexibility — classifying CDR loop flexibility  (`tests/experiments/opig_cdr_flexibility.py`)
- **Cut by the author on 2026-09-23**, with the reason recorded in the file's own header: 216 CDRH3
  sequences, 41 of them rigid, and 11 germline families that collapse to 5 held-out groups. Matthews
  correlation measured 0.0000 on 11 of the 25 representation-and-model rows, meaning those models
  called the majority class for every loop; the best was AbLang-2 with XGBoost at 0.6987 ROC AUC
  against the published structure-based network's 0.82.
- The dataset itself is still live elsewhere: `cdr_flexibility` carries complexity and noise results
  in the ledger.

### opig:kingsmith_component_splits — leave-one-component-out reaction splits  (`tests/experiments/opig_kingsmith_component_splits.py`)
- **Cut by the author on 2026-09-23**, with the reason in the file's header: leave-one-component-out
  gives 4 held-out ligands and 3 held-out bases, and on the ligand axis the fold spread was 8.1 to
  15.9 RMSE in yield percent against hybrid-minus-single differences of 1.2 to 7.2, so the
  difference sits inside the spread on all five models.
- The additive axis has 23 components and would be measurable, but the file runs all four axes
  together, so it was dropped whole.

---

## Groupings

- **Hybrid against the best single representation, on someone else's protocol:** opig:qsar,
  opig:ginkgo, opig:graphinity, opig:humanness, opig:trastuzumab, opig:cgr_comparison,
  opig:kingsmith_component_splits (cut), opig:cdr_flexibility (cut).
- **Properties of the representation itself, no published protocol:**
  opig:shared_vectors_census, opig:shared_vectors_remedy, opig:scaffold_similarity.
- **Mutation-difference representations against the real pool:** opig:scaffold_hybrid,
  opig:scaffold_similarity (same two representations, one fitted, one graded on pairs).
- **What a blueprint is worth away from its own dataset:** opig:blueprint_transfer,
  opig:generator_scoring_swap.
- **Reaction datasets:** opig:cgr_comparison, opig:kingsmith_component_splits (cut).
- **Reached the thesis as a result:** opig:qsar (`tab:method:opig_qsar`),
  opig:shared_vectors_census (`tab:flaws:ties`), opig:shared_vectors_remedy
  (`tab:method:shared_rows`), opig:blueprint_transfer (`tab:method:settings_transfer`).
- **Antibody libraries built on one parent sequence:** opig:trastuzumab, opig:scaffold_hybrid,
  opig:scaffold_similarity, opig:graphinity.
- **No result exists:** opig:graphinity, opig:humanness, opig:trastuzumab, opig:scaffold_hybrid,
  opig:scaffold_similarity, opig:generator_scoring_swap, and opig:cgr_comparison beyond 3 folds of
  one dataset.


---

## Appendix: unverified points and open questions from compiling this document

These were noted while the sections above were written (14 October 2026). They are things this summary could not confirm, not findings.

**Shared machinery**
- thesis.tex names no framework (f*) experiment by ID, so which Chapter 5 statement came from which f* experiment is not verified.
- Dataset sample sizes come from the thesis dataset table; the code records sizes for only some datasets.
- These two thesis details were not checked in the code: a representation wider than 5,000 features is scored on its top 500 features, and kNN uses k = √n on folds with fewer than 25 samples.
- `herg_ic50` has no representation list of its own in `DATASET_REPS`. Plan comments say it reuses the `herg_ki` list, but that code was not opened.

**Federated**
- fed01 numbers are quoted from the appendix table `tab:method:flipped_sites` and were not re-checked against the raw partial file. fed02, fed03 and fed04 were checked against the raw JSON.
- fed02, caco2, DNN model: the answer page reports 0.2720 vs 0.2858 RMSE (score-weighted ensemble vs hybrid). The aggregated JSON block reports 0.2721 vs 0.2837 RMSE. The cause of the difference was not traced.
- The lhasa_herg cut is recorded only in LEDGER.md (line 7) and the answer pages; no dated decision file records it.
- Silo row counts for the Lhasa data and fed05 come from code docstrings and were not re-counted.

**Similarity**
- The thesis tables cite the 10-10b results copy. Six headline numbers were checked against the raw 10-10b files and all six matched. Whether the 10-10b DNN results were produced after the commit that clips DNN predictions to the training-label range was not checked.
- s11 has a result file on adme_mdr1 that the thesis never mentions.
- s11, qm9: the best hybrid run (253.4) uses an NGBoost surrogate, while the single representation (260.1) uses a GP. The pair therefore differs in two ways at once: representation and surrogate model.
- Chapter 4 similarity numbers come from `scratchpad/thesis_code_todo/distance/`, not from a script in `tests/experiments/`. Only the activity-cliff rule is shared with the s* suite.
- `similarity_metrics/` has no output in the server copy or locally. The MoleculeACE job is known only from a memory note. Whether `similarity_optimized_hybrid.py` ever ran is unknown.

**Noise**
- The thesis SOURCE comments cite `/Volumes/seagate/kirby_results_2026-10-10b`, not the 10-09 copy. Each number in the noise section names the copy it was read from.
- The 10-08 answer pages say n17, n19, n22, n31 and n36 never ran, but result files for all five exist.
- `thesis_provenance/METHOD_PROVENANCE.md` states in its own header that Claude wrote it and the author has not reviewed it. The new / modified / from-literature labels are therefore Claude's judgement, not the author's record.
- The level-0 pattern was checked only in the n07 rows files, for fold 0, draw 0, best setting. That check has no spread.
- n30b results exist only in the 10-09 noise copy. The thesis QM9 clipping numbers match n30b, but no SOURCE comment names it.
- n42 has no answer page and is not in the thesis. Its level-0 and damaged-level score pairs on hERG Ki were read from the raw rows files.

**OPIG**
- The thesis numbers for the shared-row census on hERG Ki, LogD and Caco-2 cannot come from the 9 October run, which stopped after ddg_combined. Their source run was not located.
- The CGR comparison has 5 folds without CGR and 3 folds with CGR, so the paired means use folds 0–2 only.
- The Ginkgo, Graphinity and blueprint-transfer numbers are quoted from the answer pages and were not re-checked against the raw JSON.
- `SCAFFOLD_SIMILARITY_RESULTS.md` (19 July 2026) was not read, so no scaffold-similarity numbers from it are reported.
- `opig_trastuzumab_enrichment`: the label balance and the splitting group of tz_her2_binary were not read. No answer page exists for this experiment.

**Complexity**
- c18 and c19 have tz_her2_binary and tz_her2_multiclass result files. A file-permission check blocked reading them, so their rows are missing from the complexity section.
- `tab:method:ensembles` (c11): the AC-SINS, Titer and PR-CHO rows have five values under four column headers. Their numbers also disagree with the c11 result files (example: AC-SINS, LightGBM). The source of those rows was not found.
- The 10-10b ledger lists abagym_VEGF, abcov_kd, adme_hlm and cdr_flexibility as cut, but the 10-09 ledger does not. No record says who cut them or when. Their result files are still reported in the complexity section.
- No DNN result file was traced to a run after commit c18ee30 (8 October), which clips DNN predictions to the training-label range. This applies to c06, c18, c21 and the other experiments with DNN files. Some c06 files are dated 7 October, before that commit.
- `tab:method:scaffold_halves` (c07), `tab:method:search_optimism` and `tab:method:one_fold_fewer` (c33/c24) have no SOURCE comment. They were tied to experiments only by matching a few numbers.

**Chapter 4**
- The code and result files behind Chapter 4's exploratory sweep of 12 noisy-label methods (`sec:lnl_motivation`, QM9, 221 configurations) were not located. It is not n09, which runs on hybrids and single representations from the current search.


---

