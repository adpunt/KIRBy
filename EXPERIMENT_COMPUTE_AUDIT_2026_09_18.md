# Experiment compute audit — 2026-09-18

77 experiment files, every one opened and read this session: complexity 28, similarity 12,
noise 19, OPIG 13, federated 5. Framework (f00–f33) and annitia are not in it.

Each file's entry is below, in its suite's section. Four of the files fit nothing and only read
results; they are marked as such.

---

## 1. The two cuts you named, in numbers

Measured by calling the repo's own configuration counter, `_expected_config_count` at
`src/kirby/find_best_hybrid.py:568`, on a pool of 10 representations and the 8 budgets in
`FULL_BUDGETS`.

One row is one search of one dataset.

| search | configurations | downstream model fits at 5 folds |
|---|---|---|
| full — today's default | 8,160 | 40,800 |
| `speed='fast'` | 432 | 2,160 |
| fast, and 2 downstream models | 192 | 960 |

The full grid is 5 allocation methods x 3 importance methods x 4 representation counts x 8
budgets x 17 (downstream model, internal evaluator) pairs. The 17 pairs are a sum over the 5
downstream models, not a product — `match` adds each model's own evaluator. The fast grid is
3 x 1 x 2 x 8 x 9.

The full search runs 18.9 configurations for every 1 the fast search runs. It runs 42.5 for
every 1 that a fast search cut to two downstream models runs.

The two downstream models the fast tier drops are `xgboost` and `svm`. svm is the one that costs
between O(n^2) and O(n^3) in molecules. The two importance methods it drops are `deeplift` and
`treeshap`; the f33 record quoted at `find_best_hybrid.py:2779` says deeplift was half the fast
run's hours on all five datasets it was measured on, and won 0 of the 6 (model, budget) cells on
four of them.

### Fast is already shipped on 12 datasets

`tests/slurm_scripts/run_precompute_dataset.sh:100-107` puts every dataset in one of three
classes. The 12 in `LARGE_DATASETS` precompute at `speed=fast`: qm9, logd, herg,
flab_expression_koenig, nt_H3K27me3, nt_H3K9me3, nt_H3K9ac, nt_H3K4me3, nt_promoter_tata,
gb_human_enhancers_cohn, tz_her2_binary, tz_her2_multiclass. Every other dataset precomputes at full.

Speed is part of the search identity. A fast search and a full search on the same dataset are
two different searches, not one a subset of the other — the comment at
`find_best_hybrid.py:2803` states this. So an experiment asking at full speed on one of those 12
datasets does not read the fast rows; it cold-builds.

### What forcing fast would break

Fast cuts the downstream model list to rf, lightgbm and dnn. Five federated experiments read a
stored recipe for one named downstream model and fall to a hardcoded recipe when that model is
absent from the search, without recording that they did: `fed01:220-222`, `fed02:223-227`,
`fed03:190-192`, `fed04:218-221`, and fed05. Those five already do this today for `mlp`, which is
model tier `extra` and is never auto-selected. Under fast, the `xgboost` arms would join them.

That is a defect in those five files, and it is also the reason fast cannot simply be switched on
across the suite without the model sweep being cut first.

### `speed='fast'` does nothing to the model list when the models are named

The filter that drops xgboost and svm sits inside `if models is None`, at
`find_best_hybrid.py:8221-8232`. An experiment that passes its own list of downstream models keeps
all five of them under `speed='fast'`, and the grid falls from 8,160 configurations only to 2,160
rather than to 432 — the (model, evaluator) pairs stay at 17 instead of dropping to 9.

Most experiments in this suite name their models, usually as `DOWNSTREAM_MODELS`. n20 is the
worked example: it already sets `speed='fast'` at `n20_clean_before_search.py:197` and still runs
17 pairs, because it hands the search an explicit list at n20:269.

So switching an experiment to fast is two changes, not one: the speed argument, and either
dropping the model list or shortening it.

---

## 2. Where the compute actually is

One row is one experiment. The number is downstream model fits for one dataset unless the row
says otherwise. Each figure is derived from the loops in that file, not measured on a run; the
per-experiment entry below gives the arithmetic and says which terms are upper bounds.

| experiment | model fits | what multiplies |
|---|---|---|
| opig_shared_vectors_remedy | ~493,000,000 per dataset | 512 swept comparisons, each a full grid |
| n20 | ~121,000,000 per dataset | 1,105 searches, one per cleaning setting per fold |
| opig_kingsmith | ~43,800,000 for both datasets | held-out component splits x full grids |
| c13 | ~18,700,000 per dataset | 4 pool subjects x 6 representation sets x 3 augmentation conditions |
| opig_activity_cliff_qsar | ~13,500,000 across 3 targets | per-target full grids |
| c24 | ~9,200,000 per dataset | 25 searches |
| opig_ginkgo_developability | ~7,360,000 across 5 properties | ~1,471,000 per property, on ~194 antibodies |
| n11 | ~5,100,000 per dataset | the remediation grid |
| c05 | ~3,580,000 per dataset | 5 subsample sizes x 3 seeds = 13 full searches |
| c23 | ~3,290,000 per dataset | 20 arms x 5 outer folds = 100 searches |
| c33 | ~3,200,000 per dataset | budget sweep x augmentation |
| c07 | ~2,600,000 per dataset | up to 8 unpruned full searches |
| s11 | ~2,330,000 per dataset | Bayesian optimisation loop |
| fed01 | ~1,920,000 per run on caco2 | 16 search identities: 1 clean + 5 noise levels x 3 seeds |
| n15 | ~1,900,000 per dataset | 21 live searches |
| c14 | ~1,840,000 per dataset | 5 full searches with caching off, per name in `--tasks` |
| n09 | ~1,188,000 per dataset | 80 uncached searches, 99.6% of the total |
| opig_graphinity_ddg | ~807,000 | one full grid on the experimental condition |
| c11 | ~627,150 per dataset | one full search per outer fold per augmentation condition |
| c02 | ~616,000 cold, ~52,000 warm | 5 searches, 2 of them on tiers nothing precomputes |
| opig_blueprint_transfer | ~495,000 per source-to-target pair | ~1,200 of them are arms it reports |
| fed05 | ~408,000 per dispatch | 3 antibody sets x 3 seeds = 9 searches |

c17 is not in the table and belongs beside it. It fits ~1,360 models per dataset and then runs
~10,000,000 single-feature mutual-information estimates for the mRMR ranking at
`c17_information_theoretic_decomposition.py:337`, which is ~245 hours per dataset. Its wall time
is not model fitting.

---

## 3. Three counts across the 77

**41 of the 77 run their own search.** 20 call `find_best_hybrid` and nothing else, 21 call both
`find_best_hybrid` and `create_hybrid`. 22 read stored rows only. 4 call `create_hybrid` alone,
6 touch no hybrid machinery, and 4 fit nothing at all.

**None of the 77 follows the top-2-base, top-2-hybrid shape.** 21 come part of the way: they rank
base representations or hybrids, but not both, or they report 1 hybrid per downstream model
rather than 2. The other 56 have a different shape entirely, and for most of them that is the
right answer — c17 decomposes information, s01 compares distances, opig_shared_vectors_census
counts tied rows. Each entry says in one sentence what its file does instead.

The pattern is not implemented anywhere shared. There is no top-N base representation selector
and no top-N hybrid selector in `tests/experiments/_shared.py`; every file that does anything
like it rolls its own.

**40 of the 77 compute distances between molecules or use nearest neighbours.** Of those, the
ones whose answer turns on the geometry and that are searching on the predictive objective
instead are flagged in their entries under "Hybrids it uses / should use" — c04, fed01, fed02,
fed03, fed04, fed05, n12, n21 and the s-series among them. The code name for the geometry
objective is `objective='similarity'`, which scores a hybrid by predicting each molecule's
activity from its five nearest neighbours.

---

## 4. Where each cut lands

Per-file locations are in the entries. The counts: a `speed='fast'` cut is available at a named
line in 44 of the 77 files, and a downstream evaluator cut at a named line in 61 of them.

Two cross-cutting notes on the evaluator cut. The downstream model list an experiment reads is
usually `DOWNSTREAM_MODELS` at `tests/experiments/_shared.py:539`, which holds rf, xgboost, svm,
dnn and lightgbm. The internal evaluator list inside a search is `DEFAULT_INTERNAL_EVALUATORS` at
`find_best_hybrid.py:3449`, which holds match, rf, lightgbm and dnn. They are different lists and
cutting one does not cut the other. Cutting downstream models from 5 to 2 takes the (model,
evaluator) pairs from 17 to 6; cutting the internal evaluators alongside them takes it to 4.

---

## Complexity suite (c01–c43)

Counts below are per dataset unless a line says otherwise. Where a dataset is named for the arithmetic it is herg_ki: 1,412 molecules, 10 representations, 5 augmentation representations, 5 folds.

One term recurs. A "full grid" search is 5 allocation methods x 3 importance methods x 4 representation counts x 8 budgets x 17 (downstream model, internal evaluator) pairs = 8,160 configurations. The 17 pairs are a sum over the 5 downstream models, not a product: rf 3, lightgbm 3, dnn 3, xgboost 4, svm 4.

---

### c01 — representation profile

For each of ten representations on one dataset, how low-dimensional is it, how much does it tell you about the label, and how well does it predict the label alone.

- **Hybrids** — none. The file never calls find_best_hybrid, create_hybrid or any pool read. Its only scoring call is evaluate_single_rep_cv at c01_rep_characterization.py:106, one model per fold on one representation.
- **Compute** — 10 representations x 5 downstream models x 5 folds = 250 model fits. Per representation it also runs one PCA to 50 components, one 10-neighbour dimension estimate, one 5-neighbour mutual-information estimate, and one SVD. Not excessive: there is no configuration grid, so no axis multiplies.
- **Hybrids it uses / should use** — zero hybrids; it scores single representations only. Two of its four descriptors are neighbour quantities, but they describe each representation, not a hybrid, so no similarity-objective hybrid belongs here.
- **Selection pattern** — 5 downstream models x 10 base representations = 50 cells, 0 hybrids, no ranking or shortlisting anywhere.
- **Cut: fast tier** — nothing to cut, no search.
- **Cut: two evaluators** — c01_rep_characterization.py:38. Cutting 5 downstream models to 2 takes 250 model fits to 100, a saving of 150 fits per dataset. No code change needed: `--models rf,lightgbm` overwrites the list.
- **Notes** — EXPERIMENTS_FINALIZED.md:812 says up to 500 model fits; the code caps at 250. The two neighbour descriptors run under manhattan distance while find_best_hybrid's similarity objective uses euclidean, so the two measure neighbours differently.

---

### c02 — complexity grid

On one dataset, which feature budget and which recipe wins for each downstream model, and does that hybrid beat that model's best single representation.

- **Hybrids** — its own, five find_best_hybrid calls. c02_complexity_grid.py:87, :114 and :149 are tier `general` at augmentation none, uncapped and capped; :189 is tier `general_ft`; :346 is tier `general` with a condition suffix, inside a loop over 5 downstream models x the top 2 representations by that model's score. Objective predictive, speed full on all five.
- **Compute** — one condition is 8,160 configurations x 5 folds = 40,800 downstream fits, plus 150 importance fits, 250 allocation fits and 250 baseline fits: 41,450 model fits. The width pass at :346 adds 2,040 configurations x 5 folds = 10,200 fits. PCA baselines add 8 budgets x 5 models x 5 folds = 200 fits. About 52,000 new model fits per dataset when the result database holds the precomputed rows; about 616,000 when it is empty. Excessive only in the two uncovered searches — :189 and :346 share no stored row with anything.
- **Hybrids it uses / should use** — up to 34,680 hybrid configurations evaluated, up to 25 reported, one per (downstream model, condition) by argmax of mean score. As used: c02 computes no distances between molecules.
- **Selection pattern** — 5 downstream models, all 10 base representations as baselines, 1 winning hybrid per model per condition rather than 2. The width pass at :336 is the one place the top-2-base shape already appears.
- **Cut: fast tier** — c02_complexity_grid.py:87, :114, :149, :189, :346. Because c02 passes allocation, importance and evaluator lists explicitly, fast changes only the model list and the representation counts: 8,160 configurations to 2,160. On :87, :114 and :149 that costs 0 extra fits, because the fast grid is a subset of rows already stored. Real saving is at :189, 41,450 fits to about 11,000, and at :346, 10,200 fits to about 5,100.
- **Cut: two evaluators** — two lists. Downstream models at c02:44: cutting 5 to 2 takes the width pass from 10,200 fits to 3,600–4,800 and PCA baselines from 200 to 80, about 6,000 fits saved. Internal evaluators at c02:98, :125, :160, :200, :357: cutting 4 names to 2 takes the grid from 8,160 to about 4,800 configurations, a real saving only at :200 and :357.
- **Notes** — c02:87, :114 and :149 do not read the precomputed search pickle under either pruning mode the submit script ships; the pruning metadata never matches. What makes them near-free is the per-configuration row store, whose identity carries no pruning setting, no model list and no budget list. EXPERIMENTS_FINALIZED.md:854 says 2,400 recipes per search; the code gives 8,160.

---

### c03 — cross-dataset transfer

Across several datasets, does a dataset's complexity predict which feature budget wins, how much of the spread is the downstream model against the dataset, and do two regressors that were never in the search work on the winning features.

- **Hybrids** — both. c03_cross_dataset_transfer.py:151 reads the pool for each of 5 downstream models, tier `general_distance`, objective predictive, speed full, 6 budgets to 500 features, 6 downstream models including knn. c03:316 calls create_hybrid directly. A precompute for that tier exists, but c03 cannot load either writer's pickle: the pruning metadata never matches, so every read rebuilds the search object from stored rows.
- **Compute** — warm, the 5 pool reads cost 0 new fits and the per-fold re-pick costs 0 fits. The novel-model arm is 2 regressors x 5 folds x (2R + 1) fits, where R is the blueprint's representation count, 2 to 10: 50 to 210 model fits per dataset, 150 to 630 over the 3 default datasets. Fully cold on one dataset it is about 54,700 model fits per pool member. Not excessive warm; the exposure is a dataset where only the in-suite helper has written that tier.
- **Hybrids it uses / should use** — 5 downstream models x 3 datasets = 15 hybrids read, one each, plus 5 recipes per dataset rebuilt per held-out fold. Two of its descriptors are neighbour quantities: each molecule's activity against its single nearest neighbour, and a 10-neighbour dimension estimate. Its `general_distance` lineage is not neighbour geometry — it is a predictive search with knn added as a sixth model. Passing `pool_subset='predictive_and_similarity'` at c03:157 would admit hybrids built so each molecule's activity is predicted from its five nearest neighbours, but nothing precomputes that tier under that objective, so it would cold-build. Its budgets already stop at 500 features rather than 2,000.
- **Selection pattern** — 5 downstream models, 1 hybrid and 1 base representation each: 10 model/representation pairs against the intended 20. All 10 base representations are carried but only the best per model is reported.
- **Cut: fast tier** — no find_best_hybrid call in the file; speed would have to be added to the keyword arguments at c03:151-158. That would take 10,800 configurations to 1,080, but on a dataset whose rows are stored it costs 0 extra fits, because every fast list is a subset of the full one.
- **Cut: two evaluators** — c03:150, the loop over 5 downstream models. Cutting to 2 saves almost no fits, because all five iterations resolve the same pool members; it drops the correlation table from 5 coefficients to 2 per descriptor. The real cut is c03:297: the create_hybrid at :316 does not depend on the novel model, so hoisting it above that loop halves the arm, 50–210 fits to 30–110.
- **Notes** — complexity_helpers.py:1068 writes the `general_distance` precompute with no model list, so it is scored on 5 models with no knn, while the read side asks for 6 including knn. Both writers target one filename and overwrite each other. That helper also emits a capped condition nothing ever reads. The correlation at c03:473 runs on as few as 2 datasets.

---

### c04 — importance as a distance

Does ranking a molecule's features by importance also make a better distance for finding its nearest neighbours, and does predicting activity from those neighbours improve when features are weighted or cut to the top 25.

- **Hybrids** — none. No find_best_hybrid, no create_hybrid, no pool read. The only kirby call is compute_feature_importance at c04:65.
- **Compute** — 5 folds x 3 importance methods x 10 representations = 150 model fits under `--mode full`; 30 model fits at the command-line default of 1 fold. On top sit 245 nearest-neighbour index builds, which train nothing. Not excessive: one importance computation per (fold, method) is shared by all four arms.
- **Hybrids it uses / should use** — zero hybrids; every arm works on the raw concatenation of all ten representations. This is the clearest candidate in the suite for similarity-objective hybrids, the ones whose features are chosen so each molecule's activity is predictable from its five nearest neighbours. They belong at the low-dimension arm, c04:397-401, where the budget of 25 features already matches, and at 10 and 50 features beside it. No such comparison exists in the file.
- **Selection pattern** — no downstream evaluators and no hybrids: 3 importance methods x 3 k-values of neighbour regression, a 3x3 select-by-one-weight-by-another matrix, 3 low-dimension rows, 10 single-representation retrieval rows.
- **Cut: fast tier** — nothing to cut, no search.
- **Cut: two evaluators** — no downstream evaluator list. The nearest axis is the 3 importance methods at c04:333; cutting to 1 takes 150 model fits to 50 but collapses the 3x3 matrix to one cell, which is the experiment's own result.
- **Notes** — two arms in the same function measure two different distances. The retrieval arms use manhattan through make_nearest_neighbors; the neighbour-regression arm three lines above uses sklearn's euclidean default.

---

### c05 — learning curves

As the number of training molecules grows, does the hybrid stay ahead of the best single representation, and is the gap larger when molecules are fewer.

- **Hybrids** — both. c05:131 reads the pool per downstream model, tier `general`, objective predictive, 8 budgets. c05:223 runs its own find_best_hybrid, tier `general`, speed full, augmentation strategy capped — but under a dataset name mangled to `<dataset>_n<size>_s<seed>`, which no precompute covers. c05:336 rebuilds per fold with no precomputed importance or allocation scores.
- **Compute** — 5 subsample sizes x 3 seeds, 13 searches per dataset. Each is 1,020 x B configurations, B the surviving budget count of 4 to 7, x 5 folds x 10 fits per configuration-fold: 1 downstream fit, 5 capped ladder fits, 3 ranking fits, 1 base-ranking fit. About 3,580,000 model fits per dataset under `--mode full`, about 365,000 on a dataset with no augmentation representations, about 716,000 at the command-line default of 1 fold. Excessive: 13 uncovered full-grid searches per dataset.
- **Hybrids it uses / should use** — 5 hybrids per search, one per downstream model, x 13 searches, plus 5 all-rows blueprints from the pool read. As used: c05 computes no distances between molecules.
- **Selection pattern** — 5 downstream models, 1 hybrid and 1 base representation each: 10 model/representation pairs, not 20.
- **Cut: fast tier** — c05:223-237. Fast takes allocation 5 to 3, importance 3 to 1, representation counts 4 to 2: 1,020 x B configurations to 102 x B, a factor of 10. Models and evaluators are passed explicitly at c05:225 and :230, so those two axes do not move unless those arguments are dropped.
- **Cut: two evaluators** — c05:225, models=model_names, 5 names. Cutting to 2 takes the (model, evaluator) pairs from 17 to about 6, the baselines from 250 to 100 fits per search, and the allocation pre-pass from 5 evaluators to 3. Combined with fast, one search goes from 1,020 x B to about 36 x B configurations.
- **Notes** — with KIRBY_NO_COLD_SEARCH set, all 13 searches are refused before any fit, and c05:257 catches the exception and prints a failure, so the hybrid arm fills with NaN and the curve is empty. Each of the 13 searches also opens its own result database under a name in the shared `general` namespace.

---

### c06 — within-family gradient

Within one model family, does a higher-capacity setting want more features.

- **Hybrids** — its own. c06:270 calls find_best_hybrid once per model variant, tier `c06_variant`, objective predictive, speed fast, 8 budgets, one estimator per call. Nothing precomputes that tier and the cold-search guard names it as one it will never refuse. c06:203 separately asks for a tier `general` search to fetch an importance cache.
- **Compute** — 7 families x 4 capacity settings, 24 searches on herg_ki because gp is gated out above 1,000 molecules. Each is 3 allocation methods x 1 importance method x 2 representation counts x 8 budgets x E pairs, E being 4 or 3 by family: 192 or 144 configurations, x 5 folds x 9 fits per configuration-fold. About 197,000 model fits per dataset under `--mode full`, about 39,400 at the command-line default, about 232,600 on a dataset under 1,000 molecules where gp survives. Excessive: 24 uncovered searches inside one job wall.
- **Hybrids it uses / should use** — 8 hybrids per variant, one per budget, about 192 to 224 hybrid cells per dataset. As used: c06 computes no distances between molecules, and varying k in the kNN family is a downstream model setting.
- **Selection pattern** — one downstream model per search, that variant itself, reported at 8 budgets, with no base representation read at all; the answer is a correlation between capacity index and peak budget.
- **Cut: fast tier** — already fast at c06:77 and :297. Fast buys allocation 5 to 3, importance 3 to 1, representation counts 4 to 2; the model dict and the evaluator list are explicit, so speed does not touch them.
- **Cut: two evaluators** — c06:87. Cutting the base evaluator list from 4 names to 2 takes a 4-evaluator family from 192 to 96 configurations, the allocation pre-pass from 200 to 100 fits, and the dataset from about 197,000 to about 103,000 model fits. The larger cut is the outer loop at c06:240 and :255, 24 searches per dataset.
- **Notes** — c06's own header says fast cuts importance methods to two and evaluators to `[rf, match]`; the constants give one importance method and four evaluator names, so every c06 search is twice the size its comment claims. Each search also computes 50 baseline fits c06 never reads, 1,200 wasted fits per dataset.

---

### c07 — easy and hard partitions

Does the winning recipe change between the easy half and the hard half of the same dataset, cut four different ways.

- **Hybrids** — both. c07:72 reads the pool for 4 candidate downstream models. c07:275-287 runs its own find_best_hybrid per subpopulation, tier `general`, objective predictive, speed full, 8 budgets, augmentation strategy capped. The tier string says `general`, but a subpopulation's labels and groups are not the dataset's, so no stored row applies and every configuration is paid live.
- **Compute** — up to 8 searches per dataset, 4 partitions x easy and hard. Each is 8,160 configurations x 5 folds, at 1 fit per configuration-fold without augmentation representations and about 8 with them: 41,350 or about 326,950 model fits per search. About 2.6 million model fits per dataset on herg_ki, about 332,000 without augmentation representations. Excessive: 8 unpruned full grids none of which any precompute covers.
- **Hybrids it uses / should use** — per subpopulation one cross-model best hybrid plus one per downstream model, and every base representation's score. As used: the answer rests on scores, not distances.
- **Selection pattern** — 5 downstream models, 1 hybrid recorded per model rather than 2, all 10 base representations recorded, and a top-3 base list that is stored and never used to pick anything.
- **Cut: fast tier** — c07:275. Fast takes 8,160 configurations to 432 and downstream models from 5 to 3, so one search goes from 41,350 to about 2,510 model fits and all 8 searches from about 331,000 to about 20,000. EXPERIMENTS_FINALIZED.md:1042 already labels c07 fast; the code passes no speed.
- **Cut: two evaluators** — c07:280, 4 internal evaluator names; cutting to 2 takes the grid to 4,320 configurations per search. The bigger cut is the downstream model list, which :275-287 does not pass at all: adding two names takes the grid to 2,880 configurations and baselines from 250 to 100 fits per search.
- **Notes** — with KIRBY_NO_COLD_SEARCH set, every one of the 8 searches is refused before a single fit, c07:325-328 swallows it, and the partition's main answer becomes None with no error in the JSON. c07:70 asks for an mlp blueprint that no precompute scores.

---

### c08 — calibration

Does the recipe that predicts best also give the best-calibrated uncertainty, against the same recipe at 25 features, at full width, and against a low-capacity model under a different allocation and importance method.

- **Hybrids** — precomputed. c08:131 reads the pool for 5 candidate downstream models, tier `general`, objective predictive, augmentation none, uncapped and capped, best across all 8 budgets. A precompute exists for exactly that. c08:252 then calls create_hybrid per fold per condition.
- **Compute** — 4 conditions x 5 folds x 22 fits per fold: 10 importance fits, 10 allocation-scoring fits, 1 model fit on the training half, 1 on the proper-training half. 440 model fits, plus about 140 more when the winning blueprint carries augmentation features, so about 580 per dataset. Not excessive: nothing multiplies unless a missing precompute turns a pool read into a live full grid.
- **Hybrids it uses / should use** — exactly one recipe per dataset, the best-scoring blueprint across the candidates, reused at three budgets. As used.
- **Selection pattern** — one recipe against itself at three feature budgets under one downstream model, plus a fourth arm that swaps the allocation and importance method, and no base representation at all.
- **Cut: fast tier** — nothing to cut, no search. Passing speed through c08:131 would change the search key and miss the precompute this file exists to read.
- **Cut: two evaluators** — c08:129, 5 candidate names of which mlp and knn can never resolve. Cutting to 2 saves pool lookups only, 0 fits. The list that costs fits is c08:76, the 4 conditions: dropping to 2 takes about 580 fits to about 290.
- **Notes** — the uncertainty-versus-error correlation at c08:436-437 only fires for a model exposing fitted sub-estimators, so it is NaN whenever the pool picks svm or lightgbm. The overfit condition uses the whole concatenated width as its budget and still pays its 20 ranking fits to select everything.

---

### c09 — formal analysis

Four questions in one file: how much of c02's score spread is model against budget, how much the selected features tell you about the property as the budget rises, whether they span more directions than PCA or a random subset, and whether each model's train-to-held-out gap widens with budget.

- **Hybrids** — precomputed. c09:374, :580 and :756 read the pool, tier `general`, objective predictive; a precompute covers that. None of the three passes the in-memory dictionary, so all 104 lookups per dataset take the disk path. c09:431 and :624 call create_hybrid_cached; c09:782 reaches an uncached per-fold create_hybrid.
- **Compute** — 9b is 8 budgets x 5 folds x 20 fits = 800 model fits, plus about 280 augmentation fits and 750 Ridge regressions. 9c is 0 new fits when 9b has run, about 1,080 cold. 9d is 5 models x 8 budgets x 5 folds x 21 fits = 4,200 model fits, or 0 when a c06 result file is present. Roughly 6,100 model fits plus 750 Ridge regressions with no c06 file, roughly 1,900 with one — which is the normal case. Not excessive in fits; the exposure is 104 disk lookups, any of which cold-builds a full grid.
- **Hybrids it uses / should use** — 8 recipes in 9b and again in 9c, one per budget; 40 in 9d, one per (downstream model, budget). 9b's headline number is a 5-neighbour mutual-information estimate between selected features and the property, so it is about how molecules sit near each other, while every hybrid it measures was chosen to make a downstream model predict well. It should also read similarity-objective hybrids, those whose features make each molecule's activity predictable from its five nearest neighbours, at 10, 25 and 50 features. The argument is `pool_subset='predictive_and_similarity'` at c09:374 and :580. It costs no new fits only where that dataset was precomputed with `--objectives predictive,similarity`; elsewhere it cold-builds.
- **Selection pattern** — 8 budgets against one recipe each in 9b and 9c, 5 downstream models x 8 budgets in 9d, and no base representation anywhere; the comparison arms are PCA and a random subset of the same width.
- **Cut: fast tier** — nothing to cut, no search of its own.
- **Cut: two evaluators** — c09:754, the loop over 5 downstream models. Cutting to 2 takes 9d from 40 cells to 16 and from 4,200 to 1,680 model fits, and the pool lookups from 40 to 16.
- **Notes** — 9d's helper drops metric_fn before create_hybrid, so a blueprint carrying evaluator `match` collapses to lightgbm for all 5 downstream models and the budget split is ranked by r2 or accuracy rather than the dataset's own metric. 9b and 9c also run the identical lookup loop twice, 64 pool reads where 32 would do.

---

### c11 — ensemble against hybrid

If you train one model per representation and average their predictions, does a single model on one searched fused feature matrix match that accuracy with fewer featurisers at deployment.

- **Hybrids** — both. c11:476 calls find_best_hybrid per outer fold, tier `general`, objective predictive, speed full, 8 budgets, 5 downstream models and 4 internal evaluators passed explicitly, under a nested-CV cache namespace. Nothing precomputes a nested-CV namespace, so every one is cold. c11:556 reads the pool only under `--recipe-search whole_data`. c11:822 rebuilds per fold with no precomputed scores.
- **Compute** — per cold search 8,160 configurations x 5 folds plus 650 pre-pass and baseline fits = 41,450 model fits. Searches per dataset are 5 outer folds x 1 augmentation condition, or 15 with augmentation representations. Plus 1,000 base-representation fits, 200 PCA fits, and about 4,200 hybrid-arm fits. About 212,650 model fits per dataset at the default with no augmentation representations, about 627,150 with them; about 5,400 under `--recipe-search whole_data` when the pickles are on disk. Excessive at the default: 5 to 15 cold full grids per dataset.
- **Hybrids it uses / should use** — one hybrid per (downstream model, budget, outer fold), up to 40 hybrid arms per dataset, each the best-scoring row across the augmentation conditions. As used.
- **Selection pattern** — 5 downstream models, all 10 base representations rather than 2, and 8 hybrids rather than 2: 50 base cells and 40 hybrid cells against the intended 20. Every base representation is structurally required, because the ensemble arm averages over all of them.
- **Cut: fast tier** — c11:476-490. Fast alone takes 8,160 configurations to 816, about 41,450 model fits to 4,630, a factor of 9.0. Also dropping the explicit model and evaluator lists at :478 and :482 gives 432 configurations and 2,510 fits, a factor of 16.5. The per-fold labels still miss the shipped fast precompute, so it stays cold either way.
- **Cut: two evaluators** — c11:1549 and :367, the 5-model list feeding the search, the model loop and the factories. Cutting to 2 takes the grid from 8,160 to 2,880 or 3,840 configurations by which pair survives, so about 74,000 to 98,500 model fits per dataset instead of about 212,650. Base fits fall 1,000 to 400, PCA 200 to 80, hybrid rebuilds about 4,000 to about 1,600. Cutting the 4 internal evaluator names at :482 to 2 is a separate 8,160 to 4,320 configurations.
- **Notes** — the per-fold searches sit in a bare try/except at c11:492-493. With KIRBY_NO_COLD_SEARCH set the guard fires on every one, the hybrid arm goes silently empty, and the ensemble arms still report numbers. `--budgets` is ignored: the budget list comes from the module constant and the config key is never read. The 200 per-fold rebuilds are handed no precomputed scores, so each re-scores all 10 representations.

---

### c12 — modality ablation

On a TCGA cohort, does adding a second omics assay next to gene expression predict the outcome better than the best single representation of either assay alone.

- **Hybrids** — none. No find_best_hybrid, no create_hybrid, no pool read. Every number comes from model fits inside three helpers the file owns.
- **Compute** — under the suite runner: 10 representations x 5 models x 5 folds = 250 fits, plus 10 modality pairs x 5 models x 5 folds = 250 fits and at most 250 re-pick fits, plus 25 fits for the all-modality matrix and at most 25 re-picks. About 525 model fits per cohort, up to about 800 if every re-pick misses the shared cache. Standalone with all representations it is up to about 2,825 model fits. Not excessive: the only multiplying term is the pairwise loop, and a shared fit cache caps the re-pick.
- **Hybrids it uses / should use** — zero hybrids. Arms are single representations, two full-width representations side by side, and every modality's best representation side by side. As used.
- **Selection pattern** — 5 downstream models, all 10 loaded base representations rather than 2, 0 hybrids; the comparison is 10 modality pairs plus 1 all-modality matrix against the per-fold best single representation.
- **Cut: fast tier** — nothing to cut, no search.
- **Cut: two evaluators** — c12_modality_ablation.py:735. Model count is a plain factor in all three phases, so cutting 5 to 2 takes about 525 model fits to about 210 under the runner and about 2,825 to about 1,130 standalone.
- **Notes** — the runner hands c12 the pruned 10-representation pool, not the 39-name modality catalog in the file, so only 5 of 9 modalities carry a representation and the pairwise grid is 10 pairs rather than 36. EXPERIMENTS_FINALIZED.md:1255 says three classification models under the runner; the code fits five in both paths.

---

### c13 — forward compatibility

When a newly published embedding is added to a dataset's ten representations, does the hybrid built with it beat the hybrid built without it by more than a random matrix of the same width would.

- **Hybrids** — its own, through a pool read that never hits. c13:475 reads the pool with a condition suffix that find_best_hybrid welds onto the tier, giving strings such as `general__cond-c13_whole_pool`. None is a precompute root, so nothing is precomputed and the cold-search guard is silent. Objective predictive, speed full, 8 budgets, 3 augmentation conditions per call.
- **Compute** — one pool read is 3 find_best_hybrid calls: 40,800 fits at augmentation none and 367,200 each at uncapped and capped, plus pre-passes, about 777,150 model fits. On herg_ki 4 of the ten representations are subjects, giving up to 24 unique searches: about 18.7 million model fits per dataset. Up to about 58 million if all 10 molecular candidates build, 75 searches. Both are upper bounds, because configurations selecting identical features share one fit; that discount is unverified. Excessive by a wide margin.
- **Hybrids it uses / should use** — 6 hybrids per subject, 24 on herg_ki, each a single best configuration across 8 budgets for one named downstream model. As used: c13 measures a predictive delta, and its one cosine distance is between mean feature vectors of two representations, not between molecules.
- **Selection pattern** — six hybrid conditions per new representation against that representation's own best single-model score and against a Gaussian matrix of the same width, one downstream model throughout, no base-representation ranking.
- **Cut: fast tier** — c13:474, add speed to the extra dictionary. The grid goes from 8,160 to 432 configurations, a factor of 18.9, and one pool read from about 777,150 to roughly 42,000 model fits.
- **Cut: two evaluators** — c13:474 again. Passing two downstream model names takes the (model, evaluator) pairs from 17 to 6 and the grid from 8,160 to 2,880 configurations. Also cutting the 4 internal evaluator names to 2 takes the pairs to 3 and the grid to 1,440 configurations. c13 already uses one downstream model per subject, so the model cut costs it nothing it reads.
- **Notes** — condition D for a pool subject searches the dataset's whole 10-representation pool, which the precompute has already built under tier `general`, but the condition suffix makes the tier string differ, so all 8,160 configurations are paid again. Ten candidate representations each run that identical search under ten different tier strings. Under KIRBY_SMOKE_N, c13:663-666 reassigns the shared budget list process-wide to a single 25-feature entry, so every later experiment in the same process misses every precomputed file.

---

### c14 — cross-domain summary

Across a list of datasets from different domains, does the hybrid beat the best single representation on folds neither was chosen on.

- **Hybrids** — its own. c14:257-269 calls nested_cv_evaluate, which runs one find_best_hybrid per outer fold. No tier and no cache argument are passed, and find_best_hybrid defaults caching off, so nothing is written or read and every inner search is cold on every run. Objective predictive, speed full, 8 budgets, augmentation strategy uncapped.
- **Compute** — section 1 is 10 representations x 5 models x 5 folds = 250 fits. Each of 5 outer folds runs 8,160 configurations x 5 inner folds x 9 fits = 367,200 fits plus 650 pre-pass and baseline fits. Outer scoring adds about 2,700 fits. About 1.84 million model fits per dataset, and the whole thing again per name on the tasks line. Excessive: five uncached full grids per dataset, repeated on every rerun.
- **Hybrids it uses / should use** — 25 hybrids per dataset, one per (downstream model, outer fold), plus 200 per-budget blueprints rebuilt and scored outside. As used.
- **Selection pattern** — every base representation is scored against the hybrid: 10 representations x 5 models x 5 folds in section 1, and all 10 again per model per outer fold in the baseline arm.
- **Cut: fast tier** — c14:257-269. Fast takes 8,160 configurations to 816 per outer fold; because the model and evaluator lists are explicit, the 17 pairs stay. Dropping the model list to three names at :261 at the same time gives 432 configurations.
- **Cut: two evaluators** — c14:261, 5 downstream models: cutting to 2 takes the grid to 2,880 or 3,360 configurations per outer fold, section 1 from 250 to 100 fits, and the outer baseline arm from 250 to 100 fits. c14:267, 4 internal evaluator names: cutting to 2 gives 9 pairs and 4,320 configurations.
- **Notes** — c14 is not reachable from any driver. No module map lists it, no script imports it, and EXPERIMENTS_FINALIZED.md:317 records the decision to delete it. It also saves results with no dataset argument, so two runs over different task lists overwrite each other, and its Wilcoxon test pools deltas measured in RMSE, Spearman, MCC and F1 into one signed-rank test.

---

### c15 — theory diagnostics

Do distances between molecules bunch up, do kernels go singular, does performance wobble when features approach molecules, and does cutting to a hybrid's budget fix any of it.

- **Hybrids** — both. c15:105, :343, :416, :825, :832, :1117 and :1124 read the pool, tier `general`, objective predictive, 8 budgets; a precompute exists for that. c15:652 runs its own find_best_hybrid, tier `c15c_fast`, speed fast, one budget, two estimators (rf and ridge). Nothing precomputes that tier, so the guard is silent and every configuration is paid live.
- **Compute** — its own searches are 36 configurations per budget x 5 folds x 9 fits plus 300 pre-pass fits = 1,920 model fits per budget, about 13,440 over 7 budgets. The rebuild arms add about 200 fits in 15a, about 225 in 15b, 175 in 15c and about 510 in 15d. About 14,500 model fits per dataset, provided the `general` precompute is on disk. Not excessive: its own search is already fast, single-budget and two-model.
- **Hybrids it uses / should use** — 8 hybrids in 15a, 9 in 15b, 14 in 15c, 8 in 15d, all picked by predictive score. 15a and 15b rest entirely on distances between molecules: a distance-ratio statistic, a pairwise-distance spread, the hubness skew of each molecule's five nearest neighbours, and an RBF kernel built from those same distances. They should read similarity-objective hybrids, whose features are chosen so each molecule's activity is predictable from its five nearest neighbours, weighted to 10, 25 and 50 features. The choice is made by omission at c15:105-108, :343-346 and :416-419. That precompute is opt-in behind `--with-similarity`; without it that read cold-builds, and a geometry-objective search asked this way carries representation counts it cannot compute.
- **Selection pattern** — 10 base representations scored on distance and kernel statistics with no model fitted, 8 to 9 hybrids beside them, and 1 to 2 downstream models; no top-2 selection anywhere.
- **Cut: fast tier** — already fast at c15:666, which takes its grid from 360 to 36 configurations per budget. The pool reads have no speed to set.
- **Cut: two evaluators** — c15:652-668. Its downstream model list is already two. Cutting the 4 internal evaluator names to 2 takes the (model, evaluator) pairs from 6 to 2, the grid from 36 to 12 configurations per budget, and the file from about 13,440 to about 5,200 model fits. Dropping ridge is not available: 15c exists to read a linear model at the interpolation threshold.
- **Notes** — 15d asks the pool for a ridge recipe; ridge is not in any general precompute, so the read returns nothing and falls through to rf. Every number 15d labels ridge-on-hybrid is computed on an rf-chosen recipe. 15a and 15b also omit the in-memory dictionary the driver fills, so they reload from disk.

---

### c16 — feature selection as regularisation

Once the hybrid picks how many features a model gets, does it still matter how strongly that model is regularised.

- **Hybrids** — both. c16:344-353 reads the pool per model family per budget, tier `general`, objective predictive, 8 budgets, with the model list overridden to include ridge; a `--with-ridge` precompute covers exactly that. c16:377-394 runs its own find_best_hybrid, tier `c16_fast`, one budget, one model, cache on — and passes no speed, so it runs full.
- **Compute** — the three own searches are 660 configurations x 5 folds = 3,300 downstream fits plus 1,150 pre-pass fits. The heatmap is 14 regularisation settings x 9 budgets x 5 folds = 630 fold builds at 21 fits each = 13,230. A firm floor of about 17,700 model fits per dataset when the ridge precompute is on disk, about 45,800 once augmentation and ranking fits are counted — those multipliers are unverified. Add about 153,450 model fits if the precompute is absent or the dataset is in the large class. Excessive in the two places named below.
- **Hybrids it uses / should use** — up to 27 recipes per dataset, one per (model family, budget), each the single best stored configuration for that family's downstream model at that budget. As used: c16 computes no distance between molecules.
- **Selection pattern** — 3 model families x 4 to 6 regularisation settings x 9 budgets on one recipe per (family, budget), with no base representation arm anywhere in the file.
- **Cut: fast tier** — c16:377-394. Fast takes 660 configurations to 66 and the importance pre-pass from 150 to 50 fits per search: 4,450 firm fits to 850. The pool read can take speed through its keyword arguments, but that turns a free read into a cold build; the fix there is to run the ridge precompute at the speed the read asks for.
- **Cut: two evaluators** — no downstream evaluator list. The nearest axis is the 3 model families at c16:106-150. Dropping one removes one search of 180 to 240 configurations and 180 to 270 fold builds at 21 fits each, about 4,700 to 6,900 firm fits.
- **Notes** — the comments at c16:247-251 and :362 and the tier string `c16_fast` all say fast; the call passes no speed, so the grid is 660 configurations where the name implies 66. Each of the 630 fold builds is handed no precomputed importance or allocation scores, so 20 of its 21 fits re-score the ten representations.

---

### c17 — information-theoretic decomposition

Does the hybrid pack more information about the property into each feature it keeps than PCA, mRMR, recursive elimination or a random pick.

- **Hybrids** — precomputed. c17:377-383 reads the pool once per budget, tier `general`, objective predictive, 8 budgets, models left to the search's own five. A precompute covers it. c17:404-411 rebuilds per fold with no precomputed scores.
- **Compute** — model fits are small: 840 in the hybrid arm, 40 each for PCA, mRMR and random, about 400 for recursive elimination, about 1,360 firm model fits per dataset. What costs is the mRMR greedy pass at c17:338: about 1,999,000 single-feature mutual-information estimates per fold, each about 0.088 seconds, so about 49 hours per fold and about 245 hours per dataset at 5 folds. Excessive, and not in fits.
- **Hybrids it uses / should use** — one recipe per budget, 8 in total, each the single best stored configuration for rf at that budget. This rests on neighbours: the headline is a mutual-information density estimated from each molecule's five nearest neighbours in the reduced feature space. It should test similarity-objective hybrids, whose features make each molecule's activity predictable from its five nearest neighbours, through `pool_subset='predictive_and_similarity'` at c17:377-383; that lineage already has a similarity precompute behind it, so it costs no new search. It should also weight 10, 25 and 50 features, because above 100 features the density falls for a mechanical reason the file itself records at c17:477-482.
- **Selection pattern** — one downstream model, five feature-selection strategies, one hybrid recipe per budget, and no base representation on its own.
- **Cut: fast tier** — no find_best_hybrid call. Speed could be forwarded through the pool read's keyword arguments, but that turns a free read into a cold build. The compute cut is c17:285: dropping the mRMR prefilter from 2,000 to 500 features takes about 1,999,000 estimates per fold to about 125,000, from about 49 hours per fold to about 3 hours.
- **Cut: two evaluators** — no downstream evaluator list; one random forest is hardcoded and the experiment is gated to regression datasets.
- **Notes** — the rebuild at c17:404-411 gets no precomputed scores, so 20 of its 21 fits per fold build re-score the ten representations. Row counts are capped at 2,000 molecules, so cost grows with the feature count rather than with dataset size.

---

### c18 — diversity against depth

At a fixed number of features, is it better to take them from one or two representations or to spread them across all of them.

- **Hybrids** — precomputed. c18:188-194 reads the pool once per budget, tier `general`, objective predictive, 8 budgets, models left to the search's own five. A precompute covers it, and the read is served from memory when no model list is passed.
- **Compute** — 10 importance fits for a reference ranking, then per (budget, fold) 46 selection fits and 5 downstream fits: 8 budgets x 5 folds x 51 = 2,040, so about 2,050 model fits per dataset under `--mode full` and about 420 at the command-line default. Not excessive: every arm is one random forest fit on a per-fold selection.
- **Hybrids it uses / should use** — one recipe per budget, 8 in total, each the single best stored configuration for rf at that budget. As used: c18 computes no distance between molecules.
- **Selection pattern** — one downstream model, five allocation arms per budget, one hybrid recipe per budget; the closest thing to a base arm is the single highest-ranked representation by summed importance, not the top 2 by accuracy.
- **Cut: fast tier** — no find_best_hybrid call, and forcing speed through the pool read would turn a free read into a cold build. The only lever is the budget list at c18:147-148: 8 budgets to 3 takes about 2,050 model fits to about 780.
- **Cut: two evaluators** — no downstream evaluator list; one random forest is selected by task type at c18:137.
- **Notes** — the fault this file was once flagged for is fixed: the importance method is read off the recipe at c18:208 and passed to all four hand-built arms, so only the budget split differs between them. The natural arm passes no augmentation representations, so when the winning recipe came from the uncapped or capped pool member the rebuilt hybrid is not the configuration the search scored.

---

### c19 — importance-weighted training

Does turning the hybrid's own importance numbers into training signal — rescaling features, weighting molecules by neighbour agreement, or deleting the bottom half of features — beat leaving the hybrid alone.

- **Hybrids** — precomputed. c19:155 reads the pool for rf at 100 features, tier `general`, objective predictive; a precompute covers it. Under the suite runner it reads 3 stored searches, standalone 1. c19:235 rebuilds per fold with no precomputed scores.
- **Compute** — 3 downstream models x 5 folds x 20 fits per rebuild = 300 fits, plus 3 models x 5 conditions x 5 folds x 2 fits = 150. 450 model fits per dataset, 570 to 585 when the winning recipe carries augmentation features. Not excessive: the recipe is a stored read and the fold loop is fixed.
- **Hybrids it uses / should use** — exactly one recipe per dataset, rf at 100 features. Two of its five conditions weight each training molecule by how far its label sits from the mean of its 10 nearest neighbours in the importance-weighted hybrid space, so they ask a geometry question of a feature space chosen to make rf predict well. It should also read similarity-objective hybrids, whose features make each molecule's activity predictable from its five nearest neighbours, and budgets of 10, 25 and 50 features. Both are set in the same call, c19:155-161 and the hardcoded 100 features at c19:124.
- **Selection pattern** — 5 weighting treatments of one hybrid under 3 downstream models, 15 cells, no base representations and no second hybrid.
- **Cut: fast tier** — nothing to cut, no search. The only grid that can fire is the disk-miss fallback inside the shared helper.
- **Cut: two evaluators** — c19:118. Cutting 3 downstream models to 2 takes the rebuild from 300 to 200 fits and scoring from 150 to 100, so 450 model fits become 300 per dataset. Only rf, lightgbm and xgboost accept sample weights, so the cut is taken from inside those three.
- **Notes** — the recipe is fetched for rf only and then scored under xgboost and lightgbm, so two of the three columns run rf's recipe. Importance weights depend only on model and fold but are recomputed inside the condition loop: 60 of the 150 scoring-stage fits per dataset are redundant. If the recipe's importance method is deeplift, each of the 15 fold rebuilds pays a deeplift pass over 10 representations, measured at 271 seconds per pass on caco2 at 100 features.

---

### c20 — signal saturation

For each base representation alone, how many of its top features by lightgbm_gain does it take before it scores as well as all of them.

- **Hybrids** — none. No find_best_hybrid, no create_hybrid, no pool read. It fits models directly and ranks features through a harness helper.
- **Compute** — 10 representations x 5 folds = 50 ranking fits, plus 2 downstream models x 10 representations x 5 folds x L downstream fits, L being how many of 8 feature counts fall below that representation's width plus one full-width point, so 5 to 9. About 750 to 950 model fits per dataset. Not excessive: no search and no nested condition loop.
- **Hybrids it uses / should use** — zero hybrids; every arm is one base representation truncated to its own top features. As used: no distance or neighbour is computed anywhere.
- **Selection pattern** — every base representation, 10 rather than 2, alone at up to 9 feature counts under 2 downstream models; no hybrid, so the top-2 shape does not apply.
- **Cut: fast tier** — nothing to cut, no search.
- **Cut: two evaluators** — c20_signal_saturation.py:76 and :402 set 2 downstream models, already at the target. Cutting to 1 halves the 750 to 950 downstream fits and leaves the 50 ranking fits untouched.
- **Notes** — c20 is not in the complexity dispatcher's experiment list or module map, so it only runs standalone. EXPERIMENTS_FINALIZED.md:1626-1627 calls it broken on two counts; neither holds in the code. Line 243 scores hard class labels with no prediction helper, which bites only on an AUROC dataset.

---

### c21 — single-representation augmentation

Do a handful of extra features do more when added to one strong base representation at full width, or when added to the multi-representation hybrid at a budget.

- **Hybrids** — its own, constructed rather than searched. c21:692 and c21:371 call build_fold_hybrid on blueprints written out in the file at :615-632 and :313-330. No find_best_hybrid call and no pool read anywhere, so there is no tier, no objective and no speed.
- **Compute** — single-representation arms: 10 base representations x 3 downstream models x 5 folds x 204 fits per cell = 30,600 model fits. Reference arms: 1,080 fits with no extras, 1,485 capped and 1,440 uncapped. About 34,600 model fits per dataset. Excessive: a four-deep loop with no cache and no search to fall back on.
- **Hybrids it uses / should use** — two families, both constructed and neither selected: a degenerate one-representation blueprint at full native width per arm, and 9 fixed multi-representation recipes at 500, 1,000 and 2,000 features x 3 downstream models. As used: nothing computes a distance or a neighbour. One flag anyway — the reference budgets are the top three of eight, so the hybrid is never shown at 10, 25, 50 or 100 features; the list at c21:112 is where those would be added.
- **Selection pattern** — a survey: 330 single-representation cells and 27 hybrid cells per dataset, with no ranking step anywhere.
- **Cut: fast tier** — no search. Bigger levers exist: `--base-reps` at c21:979 pins the base loop to one representation and takes 30,600 fits to 3,060; `--strategies` at :998 from two admission methods to one roughly halves what remains; `--no-controls` at :996 drops 2 of the 5 donor conditions.
- **Cut: two evaluators** — c21:512, 3 downstream models. Cutting to 2 removes a third of every fit: about 34,600 model fits become about 23,100 per dataset. On the record quoted in the engine, adding xgboost closed nothing on any of five datasets.
- **Notes** — the base representation is re-scored inside every arm although its score depends only on (representation, model, fold): 4,500 wasted fits per dataset, about 13 percent of the total, removable by hoisting one call. The reference arms use an allocation method no search ever scores, which is deliberate and stated in the file.

---

### c22 — explainability

Which base representation did each feature of the winning hybrid come from, and how much of the fitted model's SHAP importance does each representation carry.

- **Hybrids** — precomputed. c22:111 reads the pool for rf at the best budget across all 8, tier `general`, objective predictive, augmentation none, uncapped and capped; a precompute covers it. c22:162 rebuilds per fold with no precomputed scores.
- **Compute** — 0 fits for the lookup, then per fold 10 importance fits, 10 allocation-scoring fits, 2 downstream fits, and 8 more when the winning recipe carries augmentation features. 110 to 150 model fits per dataset, plus five SHAP passes that fit nothing. Not excessive: one read and a fixed fold loop.
- **Hybrids it uses / should use** — exactly one hybrid, rebuilt inside each of 5 folds, and zero base representations. As used: c22 computes no distance between molecules.
- **Selection pattern** — it explains one hybrid rather than comparing: 2 downstream models, 1 hybrid, 0 base representations.
- **Cut: fast tier** — nothing to cut, no search.
- **Cut: two evaluators** — c22:98 already names 2 downstream models. Cutting to 1 saves 5 model fits of 110 to 150.
- **Notes** — run standalone with no precompute pickle on disk, the read cold-builds three full grids; the refusal that would stop it only fires when KIRBY_NO_COLD_SEARCH is set, which the cluster submit script sets and a local run does not. The per-representation SHAP breakdown is incomplete when the hybrid carries augmentation features, which c22:302-307 prints.

---

### c23 — evaluator and model matching

Should the model used to score candidate feature sets inside a search be the same model the finished hybrid is fed to.

- **Hybrids** — its own. c23:113 and c23:317 call nested_cv_evaluate, which issues one find_best_hybrid per outer fold. Tier is passed as `general` but a condition suffix rewrites it to `general__cond-c23_<model>_<evaluator>`, which is not a precompute root, so nothing precomputes it and the guard never fires even on the cluster. Objective predictive, speed full, 8 budgets.
- **Compute** — 15 arms of 480 configurations and 5 arms of 1,440 configurations, each run on 5 outer folds, each configuration-fold costing 9 fits under the uncapped augmentation strategy. About 3,290,000 model fits per dataset through the runner; about 407,000 standalone, where no augmentation representations attach. Either way it is 100 searches per dataset. Excessive.
- **Hybrids it uses / should use** — no stored hybrid is read. Per arm, 5 outer folds x 9 hybrids = 45 rebuilds, 900 across 20 arms, each picked by the inner search's own mean. As used: c23 computes no distance between molecules. One arm of fifteen names knn as dnn's far-family internal evaluator, so there the features are chosen by five-nearest-neighbour prediction inside the build.
- **Selection pattern** — per arm 1 downstream model, 9 hybrids, and all 10 base representations on every outer fold; the base side is 5 times the intended shape and the hybrid side 4.5 times.
- **Cut: fast tier** — c23:113 and :317. Fast changes only the three recipe axes, because the model and evaluator lists are explicit: an arm goes from 480 to 48 configurations, or 1,440 to 144, and the dataset from about 3,290,000 to about 330,000 model fits.
- **Cut: two evaluators** — c23:92, 5 downstream models. Cutting to 2 leaves 8 arms of 20 and 40 searches of 100: about 3,290,000 model fits become about 1,250,000. With fast as well, about 125,000 model fits per dataset, which is 3 times one full precompute search.
- **Notes** — condition D runs nothing: its only pair names a model that is filtered out before the loop. EXPERIMENTS_FINALIZED.md:373 marks c23 fast; the code passes no speed anywhere.

---

### c24 — blueprint transfer

Does a set of features chosen without seeing one fold score as well on that fold as a set chosen without seeing it and one more.

- **Hybrids** — its own. c24:182 and c24:341 call find_best_hybrid, tier passed as `general` and rewritten by a condition suffix to `general__cond-c24_fold0` and `general__cond-c24_drop1_2`. Neither is a precompute root, so nothing precomputes either and the guard is silent. Objective predictive, speed full, 8 budgets, 4 internal evaluators, augmentation strategy uncapped.
- **Compute** — 25 searches per dataset: 5 one per fold, 20 one per ordered fold pair. Each is 8,160 configurations x 5 folds x 9 fits plus 650 pre-pass and baseline fits, about 367,850 model fits. About 9,200,000 model fits per dataset through the runner, about 1,040,000 standalone. Excessive, and 10 of the 20 pair searches duplicate rows another pair search already covers.
- **Hybrids it uses / should use** — 25 hybrids, each the best across 5 downstream models by the search's own inner mean. As used: c24 computes no distance between molecules; its stability index is a set-overlap statistic.
- **Selection pattern** — one hybrid per fold and one per fold pair, 1 downstream model per cell and 0 base representations anywhere.
- **Cut: fast tier** — c24:182 and :341. Fast takes the grid from 8,160 to 432 configurations per search, a factor of 18.9, and the dataset from about 9,200,000 to about 490,000 model fits.
- **Cut: two evaluators** — c24:187 and :346, 4 internal evaluator names. Cutting to 2 and dropping `match` takes the pairs from 17 to 10 and the grid to 4,800 configurations per search, about 9,200,000 model fits to about 5,400,000. With fast as well, 288 configurations per search and about 330,000 model fits per dataset.
- **Notes** — the rows a pair search selects are symmetric in the two folds, but the cache identity is built from the ordered pair, so two tier strings search identical rows and both run. Sorting the pair at c24:352 saves 10 searches of 25, about 3,700,000 model fits per dataset, and changes no reported number. Two printed verdicts use cutoffs of 0.02 and 0.05 in metric units applied the same way to a Spearman and to an RMSE.

---

### c31 — performance-weighted global selection

When every representation offers candidate features and the whole pile is ranked once, does sizing each offer by that representation's own score beat giving every representation the same number of slots.

- **Hybrids** — its own, and only as a check. c31:960 calls create_hybrid with a fixed allocation on fold 0 to confirm the equal-quota arm matches the engine. No find_best_hybrid call, no tier, no objective, no speed, no precompute read or write.
- **Compute** — per (dataset, model) task: 50 standalone ranking fits, 50 per-fold importance fits, 150 train-only sizing fits, 1,800 ranking plus 1,800 downstream fits in the weighted arm, 30 in the equal-quota arm, 470 in the equal-cap arm, 80 in the naive arm, 10 in the engine check. About 4,440 model fits per task, about 13,300 per dataset across 3 downstream models. About 985 of those are treeshap rankings on matrices up to 4,000 features wide, and treeshap was measured at 542 seconds per call on caco2 at 100 features, so it dominates wall time. Borderline: the weighting family grew from 3 shapes to 15 while the 47-hour task wall did not move.
- **Hybrids it uses / should use** — 442 hand-built selections per downstream model, none from the engine's search: 360 weighted-global, 6 equal-quota, 64 equal-cap, 12 naive. As used.
- **Selection pattern** — all 10 base representations at full native width for each of 3 downstream models, and 442 hand-built selections compared against one target, the single best base representation.
- **Cut: fast tier** — no search. The grid constants are the levers: dropping treeshap at c31:123 halves the weighted arm, 360 arms to 180, and removes all about 985 treeshap calls, which is the wall-time term. Cutting the weighting family at c31:151-167 back to 3 shapes takes a task from about 4,440 to about 1,700 model fits.
- **Cut: two evaluators** — no internal evaluator list. The nearest axis is the 3 downstream models at c31:118, which is also the SLURM array dimension: cutting to 2 takes 30 tasks to 20 and about 13,300 model fits per dataset to about 8,900.
- **Notes** — the file's docstring says the global-ranking importance sweep is lightgbm_gain and xgboost_gain; the code runs lightgbm_gain and treeshap. The submit script's comment says the weighting family is cut to three shapes and the task is 24 hours; the code holds fifteen shapes and the wall is 47 hours. The headline delta and its bootstrap interval are taken against a whole-data target, while an honest per-fold train-picked baseline sits beside it and is reported separately.

---

### c32 — allocation strategy cross

Crossing how many representations get budget, how the budget is split, and whether each representation keeps its own share or everything is re-ranked into one pile, which combination beats the best single representation.

- **Hybrids** — its own, and only as a check. c32:292 calls create_hybrid with a fixed allocation on fold 0 to confirm the quota arm matches the engine. No find_best_hybrid call, no tier, no objective, no speed, no precompute read or write.
- **Compute** — per (dataset, model) task: 50 standalone ranking fits, 50 per-fold importance fits, 150 train-only sizing fits, 320 downstream fits and 160 global-ranking fits across 64 arms, 10 engine-check fits. About 740 model fits per task, about 2,200 per dataset across 3 downstream models, all lightgbm, rf or catboost. Not excessive: the grid is fixed at 64 arms and the two sub-knobs that would multiply it are pinned.
- **Hybrids it uses / should use** — 64 hand-built selections per downstream model: 4 representation counts x 4 weighting shapes x 2 mechanisms at each of 2 budgets, 500 and 1,000 features. None comes from a stored search. As used.
- **Selection pattern** — all 10 base representations at full native width per downstream model, one target picked per fold from training rows, and 64 hand-built strategies against it.
- **Cut: fast tier** — no search. The grid constants are at c32:82-92; cutting any scales the 320 downstream and 160 ranking fits proportionally, but at about 740 fits per task there is little to recover.
- **Cut: two evaluators** — no internal evaluator list. The nearest axis is the 3 downstream models at c32:102, also the SLURM array dimension: cutting to 2 takes 30 tasks to 20 and about 2,200 model fits per dataset to about 1,480.
- **Notes** — EXPERIMENTS_FINALIZED.md:1924 puts the cost at 520 model fits per dataset and model; the loops give about 740. Its :1928 and :1930 bullets are both stale — the quota arm now spends exactly the budget, and both sides of the engine check use fold 0's training rows. This is the cheapest of the three allocation files and the only one whose target is picked per fold from training rows alone.

---

### c33 — budget overfitting

Once the feature budget is larger than the number of molecules, does the held-out score keep improving or does the search start picking budgets the untouched folds do not support.

- **Hybrids** — its own. c33:502-515 calls run_nested_cv, which issues one find_best_hybrid per outer fold under a per-fold cache namespace. Tier `general`, objective predictive, speed full, 5 inner folds, augmentation strategy left at the engine's uncapped default. Nothing precomputes it: the interpolation budgets are in the cache key, and nothing in the repository calls the nested-CV precompute helper.
- **Compute** — 14 budgets on herg_ki, so 14,280 configurations per outer-fold search x 5 inner folds x at least 9 fits per configuration-fold, about 642,600 fits per outer fold, plus 650 pre-pass and baseline fits. About 3.2 million model fits per dataset, 78 times one reference full grid of 41,450 fits. The outer arms add 625 fits and 375 blueprint rebuilds. Excessive: five cold full grids, each over 14 budgets, each configuration paying the augmentation ladder.
- **Hybrids it uses / should use** — per downstream model per outer fold, one headline hybrid plus one per budget: 15 x 5 models x 5 outer folds = 375 rebuilt hybrids per dataset. As used.
- **Selection pattern** — every base representation, 10 rather than 2, for each of 5 downstream models on each outer fold, alongside 15 hybrids per model: 125 model/representation pairs per outer fold against the intended 20.
- **Cut: fast tier** — c33:502-515, speed forwards through to every per-fold search. Recipe combinations go from 60 to 6, so configurations per outer-fold search go from 14,280 to 1,428 and the dataset from about 3.2 million to about 322,000 model fits. The model list is explicit, so it does not move.
- **Cut: two evaluators** — c33:502-515 passes no internal evaluator list and the helper never reads one from config, so 4 names apply. Passing 2 takes the pairs from 17 to 9 and configurations per search from 14,280 to 7,560. With fast as well, 756 configurations per search. The separate model cut at c33:425 takes the pairs to about 6 and the outer arms from 625 to 250 fits.
- **Notes** — two constants in this file are dead, and the docstring's argument rests on both. c33:424 and :425 fall back to the file's own budget and model lists only when the config keys are empty, and both dispatch paths always set them. So the two budgets added on 2026-09-10 never reach the search, and `linear` never runs — the one model whose fitted parameter count equals the budget, and the only row that can answer the double-descent question. The fix is one line each. Passing `augmentation_strategy='none'` at c33:505 would take 9 fits per configuration-fold to 1.

---

### c43_signal — buchwald anchor

On the buchwald_hartwig reaction dataset, how well does cgr_morgan predict yield under a random forest and under lightgbm on the same five folds.

- **Hybrids** — none. No create_hybrid, no find_best_hybrid, no pool read, no search of any kind.
- **Compute** — 2 downstream models x 5 folds x 1 representation = 10 model fits on 3,955 reactions. Not excessive.
- **Hybrids it uses / should use** — zero hybrids; one base representation at full native width. As used.
- **Selection pattern** — one representation under two downstream models, writing the lightgbm number to disk as the anchor c43_step3 subtracts.
- **Cut: fast tier** — nothing to cut, no search.
- **Cut: two evaluators** — c43_buchwald_signal.py:71, 2 downstream models. Cutting to lightgbm alone saves 5 of 10 fits and empties the rf fields other readers use. Ten fits is not worth cutting.
- **Notes** — it has already run: the result file holds rf RMSE 14.3664 and lightgbm RMSE 15.2302 over 5 folds at 3,955 reactions. EXPERIMENTS_FINALIZED.md:2065 and :2074 say it is broken and unrun; both are stale. It loads all ten staged representations while scoring one, so any uncached representation among the other nine is built for nothing.

---

### c43_step3 — buchwald truncation

On buchwald_hartwig, does the hybrid's truncation destroy cgr_morgan's signal, how many of the budget's features does cgr_morgan get, and does ranking all ten representations in one pool beat splitting the budget.

- **Hybrids** — its own, built directly. c43_buchwald_step3.py:193-196 calls create_hybrid inside a fold loop inside a budget loop inside a method loop. No find_best_hybrid call, so there is no tier and no speed, and the objective is the default predictive one. Budgets and allocation methods are typed into the file, not read from the shared lists.
- **Compute** — 40 fits in the saturation curve, 440 in the quota cells (20 calls x 22 fits: 10 allocation-scoring, 10 importance, 1 hybrid, 1 slice), 20 in the global counterfactual. 500 model fits on 3,955 reactions, all lightgbm. Not excessive in total, but 305 of the 500 recompute numbers that do not depend on the method or budget being swept.
- **Hybrids it uses / should use** — four hybrids, rebuilt independently on each of 5 folds, 20 create_hybrid calls: 2 allocation methods x 2 budgets of 100 and 1,000 features. No ranking rule and no stored search. As used: the file computes no distances between reactions.
- **Selection pattern** — one downstream model, one base representation at full width plus 6 truncations of it, 4 hybrids, 2 global-ranking counterfactuals.
- **Cut: fast tier** — no search. Halving the 2 allocation methods at :109 or the 2 budgets at :103 takes 500 model fits to 280.
- **Cut: two evaluators** — no downstream evaluator list. The larger saving is reuse: of the 22 fits per (cell, fold), 20 are allocation scores and importance that do not change with the allocation method or budget, and the internal carve is identical across cells. create_hybrid already returns both and accepts both back, so passing them after the first cell takes 500 model fits to 200. Hoisting the global arm's ranking out of the budget loop takes 200 to 195.
- **Notes** — the quota arm ranks each representation's features separately; the global arm ranks all ten concatenated. The importance routine variance-prescreens any matrix wider than 2,000 features and gives every feature outside that screen an importance of exactly zero. cgr_morgan alone is 2,048 features, so the concatenation is over the cap and the two sides do not see the same candidate features. The exact widths are unverified. It has not been run: no checkpoint file on disk.

---

## Similarity (s01–s13)

Twelve files run under this wave. There is no s08 gap and no s07: s07 was dropped from s09's
input list, and `_summary_keys.py:23-30` now names five inputs, not six.

Two mechanisms recur below, so they are stated once here.

A stored search is no longer a pickle. `find_best_hybrid.py:285-287` says search results live in a
SQLite row database. One row is one configuration: allocation method, budget, importance method,
model, evaluator, representation count, step size. The identity that groups rows holds tier,
objective, labels hash, groups hash, fold count, seed, metric, task type, representation pool, tune
token and rep_factories_id — and, by its own comment at `find_best_hybrid.py:8611-8613`, "never the
sweep". So a caller asking for a longer model list or a longer budget list is served every row
already stored and pays only for the remainder.

The precompute does not run at one speed. `run_precompute_dataset.sh:103-109` uses speed `fast` for
12 datasets (qm9, logd, herg, flab_expression_koenig, the 5 nucleotide sets, gb_human_enhancers_cohn,
tz_her2_binary, tz_her2_multiclass) and `full` for the other 28. A fast precompute stores 432
configurations per augmentation condition where a full read asks for 8,160 to 10,080.

---

### s01 — single-representation distance baseline

For each single representation on its own, does the distance between two molecules track how
different their activities are?

- **Hybrids** — none. The file imports no hybrid builder, and neither create_hybrid nor
  find_best_hybrid appears in it. No search, so no tier, no objective and no speed.
- **Compute** — per dataset: 10 representations x [1 dense molecules-by-molecules distance matrix
  (s01:144) + 1 dense molecules-by-molecules SALI matrix (s01:166) + 1 Spearman on 5,000 sampled
  molecule pairs (s01:159) + 200 label permutations re-scoring those same 5,000 pairs
  (s01:169-178)]. Antibody and nucleotide datasets add one Hamming sweep over 4 edit thresholds,
  which fits nothing. Headline: 0 model fits per dataset, 2,010 Spearman evaluations and 20 dense
  matrices per dataset. Not excessive: no model is fitted, and above 15,000 molecules s01:45 moves
  to a nearest-neighbour path.
- **Hybrids it uses / should use** — none, and none belong. s01 is the single-representation
  comparator that s02's hybrid arms are measured against, stated at s01:2-11.
- **Selection pattern** — 0 evaluators, 10 base representations, 0 hybrids. It reports three
  distance statistics per representation over all molecules at once, with no fold held out.
- **Cut: fast tier** — nothing to cut; no search runs.
- **Cut: two evaluators** — nothing to cut; no downstream evaluator list exists. The only length
  available is the 200 label permutations at s01:169, costing 200 Spearman calls per representation
  and 0 fits.
- **Notes** — `get_config(args)` at s01:254 resolves n_splits, budgets and models, and run() reads
  none of the three. Every number is computed on all molecules, with no rows held out. The distance
  is manhattan by default (`distance_config.py:68`), which is what
  EXPERIMENTS_FINALIZED.md:2855 flags as a change from euclidean.

---

### s02 — hybrid distance versus single-representation distance

Does a hybrid's distance between two molecules track activity differences better than the best
single representation's distance does?

- **Hybrids** — precomputed read, through `get_fitted_pool_hybrid` at s02:293-318 with
  downstream_model='knn' and pool_subset='predictive_and_similarity'. Tiers `general` and
  `general_distance`, objectives predictive and similarity, speed never passed so `full`.
  `run_precompute_dataset.sh:186-191` writes general predictive, general similarity and
  general_distance predictive. Nothing writes general_distance under objective similarity.
- **Compute** — per dataset, at 10 representations, 5 folds and 2 lineages x 8 budgets = 16 cells:
  10 importance fits in run(), then per cell 20 fits in one whole-data create_hybrid, then per cell
  per fold 20 fits in one build_fold_hybrid plus 10 importance fits. That is 16 x (20 + 5 x 30) + 10
  = 2,730 model fits. Wall time also carries 1,814 dense molecules-by-molecules matrices per
  dataset. On top, to top up the stored rows: about 28,800 knn fold-fits and 27,000 geometry
  fold-evaluations per dataset where the precompute ran full. Headline: ~2,730 model fits per
  dataset from its own code, and ~145,000 fold-fits instead of 28,800 on each of the 12 datasets
  whose precompute ran fast. Excessive on those 12, because a full-speed read asks for 10,080
  configurations per augmentation condition against 432 stored.
- **Hybrids it uses / should use** — 16 recipes per dataset, one per (lineage, budget) cell, each
  the pool's single winner for knn at that budget (`_shared.py:6464`). Every number is a distance
  between molecules, so the similarity-objective family belongs here, and s02 asks for it at
  s02:311 and sweeps budgets down to 10 features at s02:679. The general-lineage similarity rows are
  read, not rebuilt. The family with no precompute behind it is general_distance under objective
  similarity.
- **Selection pattern** — 1 evaluator ('knn' at s02:97, a distance statistic rather than a fitted
  model), 10 base representations scored unweighted and importance-weighted, 1 hybrid recipe per
  cell scored the same two ways. So 20 base arms against 2 hybrid arms in each of 16 cells.
- **Cut: fast tier** — s02:293-318. Adding speed='fast' takes the asked-for grid from 5 allocation
  methods x 3 importance methods x 4 representation counts x 8 budgets x 21 (model, evaluator)
  pairs = 10,080 configurations to 3 x 1 x 2 x 8 x 21 = 1,008. Safe everywhere: speed is not in the
  row identity, so a fast read still reuses every full-speed row stored. On the 12 fast-precompute
  datasets it asks for exactly the 432 configurations already written.
- **Cut: two evaluators** — nothing on the evaluator axis; s02 fits one model name. The two lengths
  that multiply cost are the budget list at s02:679 (8 entries) and the lineage loop at s02:683 (2
  entries). Cutting budgets to 2 takes 16 cells to 4 and 2,730 model fits to about 690. Cutting the
  10 base representations scored per fold to the top 2 removes 8 of 10 importance fits per fold and
  16 of 22 dense matrices per fold.
- **Notes** — Three defects. First, the distance lineage is forced to budgets [10, 25, 50, 100, 250,
  500] features at `_shared.py:5630` while s02 sweeps up to 2,000 features at s02:679, so 2 of the 8
  distance cells per dataset record budget 1,000 or 2,000 for a recipe built at some other feature
  count. Second, s02 has no size guard on its dense matrices, unlike s01's 15,000-molecule switch: at
  5,028 molecules each matrix is about 202 MB, at 27,791 molecules about 6.2 GB, and it builds 22 per
  fold. Third, because s02 names models explicitly, `_shared.py:5655-5657` forces the disk path and
  all 16 cells re-enter find_best_hybrid for every pool member, 160 entries per dataset; each
  re-entry is a SELECT plus cached per-fold work, so only the first is slow. Unverified: the 1,412
  molecule count for herg_ki, and whether any dataset's precompute has actually been run.

---

### s03 — are the kept features the informative ones

When a hybrid keeps some of a representation's features and throws the rest away, are the ones it
kept the informative ones?

- **Hybrids** — precomputed read, through `get_fitted_pool_hybrid` at s03:152-156 with no models, no
  pool subset and no distance. Tier `general`, objective predictive by default
  (`_shared.py:5396`), speed never passed so `full`. `run_precompute_dataset.sh:186-191` writes
  that family on every dataset, and `similarity_helpers.py:995` warms the same family in-process.
- **Compute** — per dataset, at 10 representations and 3 downstream models x 8 budgets = 24 cells:
  24 x 20 fits in the whole-data create_hybrid, plus 10 fits for the Gini figures at s03:281 = 490
  model fits. Non-fit work: at most 960 Kraskov mutual-information estimations, each on at most
  2,000 molecules over at most 50 principal components, at most 960 singular-value decompositions,
  at most 240 PCA fits. Headline: ~490 model fits per dataset where the precompute ran full, and
  about 116,000 extra fold-fits on each of the 12 datasets where it ran fast. Excessive only on
  those 12.
- **Hybrids it uses / should use** — 24 hybrids per dataset, one per (downstream model, budget)
  cell, each the pool's winner for that pair from the predictive family alone. s03 computes mutual
  information and principal components, no distances and no neighbours, so the predictive family
  fits its own statistic. Its stated purpose at s03:2-4 is to explain why hybrid similarity works,
  and s02 has read the similarity family since 2026-09-12; if the point is to explain s02's winner,
  s03 should name the same pool subset at s03:152.
- **Selection pattern** — 3 evaluators (rf, lightgbm, dnn at s03:143), all 10 base representations
  per cell, 1 hybrid per cell. Each hybrid is compared against all 10 representations rather than a
  top 2.
- **Cut: fast tier** — s03:152-156. speed='fast' takes the grid from 5 allocation methods x 3
  importance methods x 4 representation counts x 8 budgets x 17 (model, evaluator) pairs = 8,160
  configurations to 3 x 1 x 2 x 8 x 9 = 432. The pair count falls to 9 because models is None here,
  so the fast filter at `find_best_hybrid.py:8221` keeps rf, lightgbm and dnn with 3 evaluators
  each. Safe: speed is not in the row identity.
- **Cut: two evaluators** — s03:143, already 3 downstream models rather than 5. Cutting to 2 takes
  24 cells to 16, 480 create_hybrid fits to 320, and the mutual-information estimations to two
  thirds of their count. The budget list at s03:142 cuts more: 8 entries to 2 takes 24 cells to 6.
- **Notes** — Across the 22 result files on disk, 5,964 of the 5,976 stored cells holding mi_full
  hold exactly 0. The estimator returns `max(0.0, mi_raw - null)` at `_shared.py:3044`, where null
  is the mean over label shuffles, so every estimate at or below its own null floors to zero and all
  four arms floor together. The comparison s03 exists to make has not produced a number on any
  dataset. Two defects EXPERIMENTS_FINALIZED.md lists are fixed in the code: the density ratio now
  divides both sides by the component count with the unit named (s03:250-259), and per_rep is keyed
  by an explicit string (s03:93-100). Everything is measured on all molecules with the selection fit
  on all molecules; the file writes descriptive_only=True at s03:128. Two imports at s03:35-36 are
  never called, so the docstring's dependency on s02 at s03:6 is wrong.

---

### s04 — activity cliffs

On molecules that sit on an activity cliff — a close ECFP4 neighbour with a very different activity
— does a hybrid feature set predict better from five nearest neighbours than any single
representation does?

- **Hybrids** — precomputed read, through `get_fold_pool_recipes` at s04:270 with distance=True and
  pool_subset='predictive_and_similarity'. That asks for two tiers: `general_distance` under
  objective predictive, which the `dist` precompute pass writes
  (`_precompute_passes.sh:37`), and `general_distance__obj-similarity`, which nothing writes. The
  `sim` pass carries no `--distance` and the `dist` pass carries no objective. No speed is passed
  anywhere, so a missing family builds at the full grid (`find_best_hybrid.py:7764`).
- **Compute** — per dataset, at 10 representations and 5 folds: 50 importance fits for the weighted
  baselines, 100 kNN fits over 10 representations x 2 weightings x 5 folds, 10 kNN fits for the
  pool's own representation, 600 fits in 30 hybrid builds at 20 fits each, 150 importance fits for
  the weighted hybrid arm, 30 hybrid kNN fits and 5 random-forest fits. Headline: ~945 model fits
  per dataset. On top, while the similarity family has no precompute: 10,800 configurations
  enumerated per augmentation condition, of which 8,100 raise before any fit and 2,700 are evaluated
  over 5 folds, giving 135,000 per-representation geometry scorings per condition, 270,000 on
  herg_ki's two conditions. Excessive, because none of that work can be used — see Notes.
- **Hybrids it uses / should use** — 6 slots per fold, 3 from the predictive family and 3 from the
  similarity family, each ranked on the other four folds' stored scores and re-picked per fold. Every
  arm is scored with five nearest neighbours (s04:375-379) and the cliff rule is an ECFP4 Tanimoto
  neighbour rule, so similarity-objective hybrids belong here and it asks for them at s04:54. On
  feature budget it does the opposite of what the question wants: s04:262 pins 100 features for
  every arm, so the 10-, 25- and 50-feature hybrids are never looked at.
- **Selection pattern** — 1 evaluator (kNN, plus one random-forest baseline arm), 10 base
  representations at 2 weightings, 1 pool base representation at 2 weightings, 6 hybrid slots at 2
  weightings: about 34 scored arms on one model.
- **Cut: fast tier** — no find_best_hybrid call to carry speed, but kwargs at s04:270-277 are
  forwarded into it (`_shared.py:5350-5352`). speed='fast' there takes the cold grid from 5
  allocation methods x 3 importance methods x 4 representation counts x 6 budgets x 30 (model,
  evaluator) pairs = 10,800 configurations to 3 x 1 x 2 x 6 x 30 = 1,080, of which 540 are
  evaluated. A blunter lever: `KIRBY_NO_COLD_SEARCH`, forwarded by
  `submit_all_datasets.sh:196`, makes the cold build raise before any fit and drops the similarity
  family with a printed warning. Cheapest of all is dropping 'similarity' at s04:54.
- **Cut: two evaluators** — s04 fits one downstream model, so no fits move on that axis. The list
  that sizes the cold grid is DISTANCE_EVALUATORS at `_shared.py:476`, 5 names; cutting it to 2
  takes the (model, evaluator) pairs from 30 to 12 and the grid from 10,800 to 4,320
  configurations, and does not change the row identity, so stored distance rows survive. The two
  lists that size s04 itself are `_N_POOL_HYBRIDS` = 3 at s04:57 (3 to 2 takes hybrid arms from 6 to
  4 and build fits from 750 to 500) and the pool subset at s04:54.
- **Notes** — Defect, two sides disagreeing. The pool hands back similarity-family hybrids with
  `blueprint.objective='similarity'` (`find_best_hybrid.py:5122`). s04 rebuilds every arm through
  build_fold_hybrid at s04:643 passing a kNN model and a metric function. build_fold_hybrid raises
  ValueError for exactly that combination at `_shared.py:4735-4739`, because a geometry objective
  fits no downstream model. The raise is swallowed at s04:415-421 and the fold is dropped. All 3
  similarity arms lose all 5 folds and are written out as not computed, while the cold search that
  produced them is paid in full. Separately, DISTANCE_BUDGETS and DISTANCE_EVALUATORS are imported at
  s04:37 and never used. The docstring remedy does not close the gap on its own: the precompute
  forces representation counts to [None] (`precompute_hybrids.py:729`) while this read normalises to
  [2, 3, 5, None], so its pickle still misses — but the rows are matched by SELECT, so the live pass
  costs only 150 importance fits and 8,100 immediate raises.

---

### s05 — five-nearest-neighbour prediction across feature widths

Does predicting a molecule's activity from its five or ten nearest neighbours work better in a
hybrid feature space than in any single representation's space, and how does that change from 10 to
500 features?

- **Hybrids** — precomputed read, through `get_fold_pool_recipes` at s05:273, reached from s05:329,
  s05:460, s05:738 and s05:837. Same two tiers as s04: `general_distance` predictive, written by the
  `dist` pass, and `general_distance__obj-similarity`, written by nothing. No speed passed, so a
  missing family builds at the full grid. s05:9 names the command that would fill it —
  `precompute_hybrids.py <dataset> --distance --with-similarity` — and no submit script runs it.
- **Compute** — per dataset, at 10 representations, 5 folds, 6 budgets and 2 neighbour counts:
  180 hybrid builds x 20 fits = 3,600 fits, 360 hybrid kNN fits, 50 baseline importance fits, 200
  baseline kNN fits. Headline: ~4,210 model fits per dataset, of which 3,600 are the importance and
  allocation scoring inside the builds rather than the kNN evaluations the file is about. On top,
  while the similarity family is cold: 10,800 configurations per augmentation condition, 8,100
  raising, 2,700 evaluated over 5 folds, 135,000 geometry scorings per condition. Excessive, because
  the similarity arms it pays for do not build — see Notes.
- **Hybrids it uses / should use** — 6 slots per fold per budget, 3 predictive and 3 similarity,
  ranked on the other four folds' stored scores. Every number is a nearest-neighbour prediction
  (s05:379-383), so similarity-objective hybrids belong here and s05:92 asks for them. On feature
  width it already does the right thing: s05:172 sweeps 10, 25, 50, 100, 250 and 500 features.
- **Selection pattern** — 1 evaluator, kNN. It scores 10 base representations at 2 neighbour counts
  at 2 weightings (40 baseline cells) and up to 144 hybrid cells, half of them dead, then reads one
  stored reference per each of 6 distance models.
- **Cut: fast tier** — kwargs at s05:273-283 forward into find_best_hybrid. speed='fast' there takes
  the cold grid from 10,800 configurations to 1,080, of which 540 are evaluated. Safe: a fast read
  reuses every overlapping stored row, since model, evaluator, budget and representation count are
  row fields, not identity. `KIRBY_NO_COLD_SEARCH` removes the cold build entirely; dropping
  'similarity' at s05:92 does the same at the cost of the comparison.
- **Cut: two evaluators** — one model is fitted, so the evaluator axis saves no fits. The 6 distance
  models looped at s05:913 drive 25 extra pool reads and about 100 pickle loads and 0 fits; cutting
  to 2 saves those reads. The lists that set the fit count are k_values at s05:171 (2 entries to 1
  halves 3,600 build fits to 1,800, 360 hybrid kNN fits to 180 and 200 baseline kNN fits to 100),
  budgets at s05:172 (6 entries to 3 saves 1,800 build fits) and N_HYBRIDS_PER_FAMILY at s05:94 (3
  to 2 takes 3,600 build fits to 2,400).
- **Notes** — Same defect as s04, confirmed. The similarity family's blueprints carry
  objective='similarity'; s05 rebuilds them at s05:432 passing a kNN model and a metric function;
  `_shared.py:4735-4739` raises; s05:494-497 catches it and records the cell as failed. Because the
  build cache is written only on success (s05:442), the same raise fires 4 times per (budget, arm,
  fold). The headline comparison is unaffected, since FIXED_REF_FAMILY is 'predictive' at s05:100,
  but the similarity row of the summary stays empty. The wave passes `conditions` at
  `similarity_helpers.py:1209` and run() never reads it. REFERENCE_REP_BY_DATASET at s05:154-157
  hardcodes graph_kernel for herg_ki and mordred for logd.

---

### s06 — applicability domain

Does how far a molecule sits from the training set, in a hybrid's feature space, tell you how wrong
the model's prediction of it will be?

- **Hybrids** — both. `get_fitted_pool_hybrid` at s06:169-175 with distance=True and
  pool_subset='predictive_and_similarity', so tiers `general_distance` and
  `general_distance__obj-similarity`. `similarity_helpers.py:1024-1039` writes the predictive
  member with matching models, evaluators and budgets. Nothing writes the similarity member: its
  only writer, `similarity_helpers.py:1063-1099`, fires only when run_all is called with
  `precompute_objectives=`, which defaults to None at `similarity_helpers.py:852` and which no
  caller in the repository passes. So `get_precomputed_search` falls through to find_best_hybrid at
  `_shared.py:5344-5346` and the similarity member builds inside s06's job. No speed is passed, so
  full grid. Unverified from this machine: whether the predictive general_distance file is on disk
  for any given dataset.
- **Compute** — per dataset with both searches on disk: 10 fits in the whole-data rebuild, 110 fits
  in the correlation loop (5 folds x 22), 105 fits in the conformal loop (5 folds x 21, regression
  only). Headline: ~225 model fits per dataset, plus 120 nearest-neighbour index builds. The first
  time the similarity member is asked for: 13,500 hybrid rebuilds that fit no model, of which only
  450 are distinct, plus 150 importance fits where no predictive search has written this dataset's
  per-fold importance files. Excessive, because 13,050 of those 13,500 rebuilds recompute a result
  already in hand.
- **Hybrids it uses / should use** — one hybrid. Every number in the file is a mean distance from a
  held-out molecule to its 5 nearest training molecules, so similarity-objective hybrids and small
  feature budgets belong here. It asks for the similarity family at s06:175 and the request never
  lands: with two families present and two different metric names, the winner is decided by subset
  order at `_shared.py:5760-5771` and 'predictive' is named first. On budget it is unpinned rather
  than pinned — only the discarded whole-data matrix is at 100 features, and every per-fold hybrid
  floats to whichever of 10, 25, 50, 100, 250 or 500 features won for knn across the whole search.
- **Selection pattern** — 1 downstream model (random forest, s06:150-151), 1 hybrid, 1 top-1
  comparator representation, and all 10 base representations plus their 10 weighted twins as
  distance measurements.
- **Cut: fast tier** — s06:169-175 as speed='fast'. It takes the similarity member from 10,800
  configurations enumerated to 1,080, of which 540 are evaluated, and the importance pre-pass from
  150 fits to 50. It does not shorten the model or evaluator lists, because both are named
  explicitly and the fast filter sits inside `if models is None`. A larger cut is free: under a
  geometry objective the engine ignores the evaluator (`hybrid.py:2659-2663`), so 5 evaluators to 1
  removes four fifths of that member's work and changes no number. Caution: the kwarg also lands on
  the predictive member, whose stored metadata carries the three lists fast substitutes, so it would
  cold-build a second search. To cut one member only, the change goes in `_shared.py:5480-5495`.
- **Cut: two evaluators** — s06's own downstream list is already 1 model, so the 225 fits do not
  move. The search's internal evaluator list, 5 names at `_shared.py:5629`, is overridable from
  s06:175: 5 to 2 takes the similarity cold search from 2,700 to 1,080 evaluated configurations. A
  third cut needs no argument: the whole-data create_hybrid produces a matrix and feature info that
  s06 never reads, so 10 of the 225 fits buy nothing.
- **Notes** — Defect, two budgets under one name. The whole-data matrix is built at 100 features and
  never used. Every per-fold hybrid comes from the blueprint at s06:203, which is the search's best
  for knn across all six budgets, and no build passes a budget override. So the distance-error
  correlation and the conformal block are measured on a hybrid whose feature count the file neither
  sets nor records. Second defect: s06 passes a model and a metric function at s06:290-291 and
  s06:633-634, which build_fold_hybrid refuses under a geometry objective; if the pool ever returns a
  similarity blueprint, every fold raises, a bare except at s06:305-311 swallows it, and the file
  writes no correlation and no conformal block. Subset order masks this today. Third: the comparator
  representation is ranked off the search's stored baselines, which under a similarity search are
  identical across models, so it would be picked by geometry rather than accuracy.
  EXPERIMENTS_FINALIZED.md:405 puts the call at line 144; it is at line 169.

---

### s08 — feature count for distance versus feature count for prediction

Does the number of features that makes molecules' distances line up with their activity differ from
the number that makes a model predict best?

- **Hybrids** — both. `get_fitted_pool_hybrid` at s08:209-215 inside a loop over 6 budgets, with
  distance=True and pool_subset='predictive_and_similarity' (s08:131). Tier `general_distance` for
  the predictive member, written by `similarity_helpers.py:1024-1039`, and
  `general_distance__obj-similarity` for the other, written by nothing, for the same reason as s06.
  No speed passed, so full grid. The first budget iteration cold-builds the similarity member; the
  other five read what it wrote.
- **Compute** — per dataset with both searches on disk: 60 fits in 6 whole-data rebuilds, 480 fits
  in the hybrid block (6 budgets x 5 folds x 16), 300 fits in the per-representation block (10
  representations x 5 folds x 6 models), 10 whole-data importance fits. Headline: ~850 model fits
  per dataset. The cold similarity member adds 13,500 geometry fold-builds that fit no model, of
  which 450 are distinct, plus 150 importance fits where no predictive search has run. Excessive for
  the same reason as s06: the model and evaluator axes do nothing under a geometry objective.
- **Hybrids it uses / should use** — 6 hybrids, one per budget entry from 10 to 500 features, each
  the pool's winner for knn at that budget. The file rests on distances throughout: a Spearman over
  a pairwise distance matrix at s08:285 and a five-nearest-neighbour violation count at s08:286. It
  asks for the similarity family and the subset-order rule at `_shared.py:5760-5771` hands back the
  predictive family every time both are present, so `hybrid_families_used` reads ['predictive'] on
  every dataset that has a predictive search. The budget sweep is the part it gets right.
- **Selection pattern** — 6 evaluators (DISTANCE_MODELS at s08:164), 10 base representations, 6
  hybrids: 6 x 16 = 96 model/representation cells. A top-3-by-model table is reported but never used
  to filter what is evaluated.
- **Cut: fast tier** — s08:209-215 as speed='fast'. The similarity member goes from 10,800
  configurations enumerated to 1,080, of which 540 are evaluated, and its importance pre-pass from
  150 fits to 50. Model and evaluator lists do not shrink, since both are named. The free cut on that
  member is the evaluator axis, which the engine ignores under a geometry objective. Same caution as
  s06: the kwarg also makes the predictive member miss its precompute, so the per-member place is
  `_shared.py:5480-5495`.
- **Cut: two evaluators** — s08:164, 6 downstream models. Cutting to 2 takes the hybrid block from
  180 to 60 fits and the per-representation block from 300 to 100 fits: 320 of the 850 model fits
  per dataset. knn and rf both have to survive, because s09's ratio divides the knn-optimal budget by
  the rf-optimal one (s08:160-163). The search's own 5 internal evaluators are separate: 5 to 2 takes
  the similarity cold search from 2,700 to 1,080 evaluated configurations.
- **Notes** — Defect, two recipes under one budget row. The whole-data matrix, and with it
  actual_dim, mle_id, sac_rho, the violation rate and dist_corr, is built from the pool's winner at
  this budget. The per-fold numbers in the same row come from the blueprint at s08:293, which is the
  winner across budgets re-applied at this budget by override. The two can differ in allocation
  method, importance method and internal evaluator. Second: if the pool ever returns a similarity
  blueprint, the fold hybrid is selected on geometry while the six downstream models are scored on
  the dataset's metric, and the row does not distinguish them beyond a family label.
  EXPERIMENTS_FINALIZED.md's 480 fits counts downstream fits only.

---

### s09 — cross-dataset summary

Across datasets, do the hybrid-versus-best-single-representation comparisons that s02, s04, s05, s06
and s08 each already measured point the same way?

- **Hybrids** — none of its own. It imports no hybrid builder and calls no search. It opens five
  saved result files per dataset and subtracts stored numbers.
- **Compute** — 4 datasets x 5 input files read, 1 re-score of s04's stored per-fold block per
  dataset, and 1 Wilcoxon test per series over at most 12 series. Headline: 0 new model fits. Not
  excessive; nothing is fitted.
- **Hybrids it uses / should use** — none. It reports whichever hybrid each input file chose. Two of
  the five quantities it pools rest on nearest neighbours upstream — the kNN delta from s05 and the
  distance-to-error correlation from s06 — so the similarity-objective emphasis belongs in those two
  files, not here.
- **Selection pattern** — no evaluators, no base representations, no hybrids: one difference per
  dataset per comparison, with a Wilcoxon signed-rank test down each column when at least 3 datasets
  contributed.
- **Cut: fast tier** — nothing to cut; no search runs.
- **Cut: two evaluators** — nothing to cut; s09 names no evaluator.
- **Notes** — The SLURM branch that runs s09 cannot. `run_similarity_experiments.sh:80-100` calls
  `run_dataset.py similarity $DS --experiments s09`, and `run_dataset.py:155-160` validates that name
  against SINGLE_DATASET_EXPERIMENTS, which does not list s09. The runner prints an unknown-experiment
  error and exits 1 on every dataset; s09 runs only when invoked directly. Its docstring at s09:6 and
  `_summary_keys.py:10` still say six inputs and name s07; the accepted list holds five. Four of the
  five open items EXPERIMENTS_FINALIZED.md:3250-3254 lists are fixed in the file. Still true: the kNN
  delta at s09:392-411 maximises the hybrid side over budgets, augmentation conditions and neighbour
  counts while maximising the baseline side over representations only, and that asymmetric number
  enters the Wilcoxon.

---

### s10 — importance weighting for neighbour-based estimators

Does multiplying every feature by the square root of its LightGBM-gain importance improve a kNN, a
Gaussian process, an SVR, or a random forest given four neighbour-derived extra features?

- **Hybrids** — both. `get_best_search_for_downstream` at s10:1148 and `get_fold_pool_recipes` at
  s10:1213, both with distance=True and pool_subset='predictive_and_similarity' (s10:91). Four pool
  members: `general_distance` and `general_distance__obj-similarity`, each under conditions 'none'
  and 'uncapped'. The two predictive members are written by
  `similarity_helpers.py:1024-1040` with matching models, evaluators and budgets. The two similarity
  members are written by nothing: the `sim` precompute pass carries no `--distance`
  (`submit_precompute_extra_passes.sh:465-468`), so they build live. No speed is passed anywhere.
- **Compute** — per dataset, at 10 representations, 5 folds, 9 hybrid widths and 4 tasks: 50 shared
  importance fits, 650 base-representation fits, 270 create_hybrid calls at 20 fits each = 5,400
  fits, 585 hybrid downstream fits. Headline: ~6,700 model fits per dataset, ~4,700 on a
  classification dataset where the Gaussian-process and SVR tasks are skipped, and up to ~8,800 if
  the fold's recipe carries the uncapped augmentation strategy. On top, while the similarity members
  are cold: each iterates 10,800 configurations, of which 8,100 raise and 2,700 build, with no
  downstream model fitted in any. Excessive: 4,800 of the 5,400 build fits are recomputation — see
  Notes.
- **Hybrids it uses / should use** — 1 recipe per held-out fold, the pool's rank-1 for k nearest
  neighbours, rebuilt at 9 widths. The file rests on distances end to end: kNN at 5, 10 and 20
  neighbours, an RBF Gaussian process, an RBF SVR, and four q-RASAR features from the 10 nearest
  training molecules. It asks for the similarity family and cannot reach it: recipes are flattened in
  subset preference order with 'predictive' first (`_shared.py:6194`) and s10 takes the first with
  n_hybrids=1. To test similarity hybrids it needs _N_POOL_HYBRIDS above 1 at s10:94 with a separate
  arm per family, or a second pool read at s10:1213 with pool_subset='similarity_only'. On small
  budgets it is covered from 10 features up; the two largest swept widths, 1,000 and 2,000 features,
  apply a recipe at a width no search scored it at.
- **Selection pattern** — 4 estimator families, 6 settings counting three neighbour counts, all 10
  base representations rather than a top 2, and 1 recipe rebuilt at 9 widths rather than a top 2
  hybrids. Per task that is 19 rows; across four tasks, 114 rows before the weighted doubling.
  `extract_top3_base_reps` is called at s10:1175 and its output only stored.
- **Cut: fast tier** — no speed is passed; the entry points are kwargs on s10:1148 and s10:1213.
  Forcing fast on the predictive member changes its stored metadata and turns a free read into a
  cold build, so it should go on the similarity member only. There it takes the grid from 5
  allocation methods x 3 importance methods x 4 representation counts x 6 budgets x 30 (model,
  evaluator) pairs = 10,800 configurations to 3 x 1 x 2 x 6 x 30 = 1,080, and what actually builds
  from 2,700 to 540. The larger cut is not the speed tier: `_SWEEP_BUDGETS` at s10:100 back to
  [None] alone takes 270 create_hybrid calls to 30 and the total from ~6,700 to ~1,300 fits.
- **Cut: two evaluators** — the four task dispatches at s10:1404, s10:1418, s10:1433 and s10:1444.
  Dropping the Gaussian process and the SVR removes 90 create_hybrid calls (1,800 fits) and 380
  downstream fits, about 2,200 of ~6,700; classification datasets already do this. Dropping
  neighbour counts 10 and 20 at s10:312 removes a further 90 calls (1,800 fits) and 180 downstream
  fits. On the search side, DISTANCE_EVALUATORS 5 to 2 takes a similarity search from 10,800 to 4,320
  configurations; passing models=['knn'] to that member alone divides it by 6 with no loss, since no
  downstream model is fitted under a geometry objective.
- **Notes** — The 9 widths repeat identical work. Within one task and one fold only the width
  changes, yet build_fold_hybrid takes precomputed importance and precomputed allocation scores and
  s10 passes neither, so each width re-scores the same 10 representations and re-fits the same 10
  importance models. `hybrid.py:1674-1681` records that per-representation scoring as the most
  expensive part of a build, about 240 s of 336 s on flab_thermo_tresanco, and as independent of
  allocation method, augmentation settings and budget. s10 already holds the right importance numbers
  in fold_importance at s10:1390-1398. Separately, the hybrid arms have never run: all 16 result
  files under `tests/results/s10_importance_downstream/` carry an empty hybrid block, with timestamps
  from 2026-05-08 and 2026-05-09, predating the per-fold pool read and the width sweep. Five imports
  are never called. EXPERIMENTS_FINALIZED.md:3299 says the width sweep runs no search and adds one
  fit per budget per fold; each swept width is a full create_hybrid at about 20 fits, and the file
  does trigger two searches.

---

### s11 — Bayesian optimisation over feature spaces

When you pick molecules one at a time to screen, does a 50-feature hybrid space let a Gaussian
process find the good ones in fewer picks than raw 2048-feature ECFP4, a 50-component PCA, a
50-component random projection, the best single representation, or picking at random?

- **Hybrids** — its own search. `get_fold_best_search` at s11:927-936, reaching
  `find_best_hybrid` at `_shared.py:1230`. Tier `general` (set by the distance=False branch at
  `_shared.py:1270-1271`), objective predictive, speed full because nothing passes speed. Nothing
  precomputes it: the search runs on the 5% seed molecules, and the row identity carries the labels
  hash and groups hash of exactly those molecules, so no stored row matches. 3 seeds x 3 augmentation
  conditions = 9 full-grid searches per dataset.
- **Compute** — per search on condition 'none': 150 importance pre-pass fits + 250 allocation
  pre-pass fits + 250 baseline fits + 8,160 configurations x 5 folds x 1 downstream fit = 41,450
  model fits. Under 'uncapped' or 'capped' each configuration-fold costs 9 fits, not 1, giving
  367,200 fits. Per seed the 900 pre-pass and baseline fits are paid once, so a seed costs 775,500
  fits. Headline: ~2.33 million model fits per dataset on herg_ki, and ~123,000 on a dataset with no
  augmentation representations. Each fit is on about 57 training molecules, so the count is the whole
  cost. Excessive: about 7,140 of each search's 8,160 configurations are never read.
- **Hybrids it uses / should use** — 1 recipe per seed, 3 per dataset, each read back for random
  forest at 50 features and shared by all five hybrid conditions. The stated hypothesis at s11:9-12
  is that pairwise distances concentrate in high dimensions, so similarity-objective hybrids are the
  ones it should test; it builds none, because no objective is passed at s11:927-936. It should also
  sweep small feature budgets rather than pin 50 features at s11:862, since "compact beats
  high-dimensional" is currently measured at one width.
- **Selection pattern** — 11 named feature spaces compete under one surrogate each, over 3 seeds.
  No evaluator axis, 1 base representation and 1 hybrid at 50 features.
- **Cut: fast tier** — s11:927-936, but the parameter does not exist: `get_fold_best_search` and
  `search_on_fold` are keyword-only with no speed argument and no kwargs, so it must be added to
  both and forwarded at `_shared.py:1204-1227`. With it, the grid goes from 5 allocation methods x 3
  importance methods x 4 representation counts x 8 budgets x 17 (model, evaluator) pairs = 8,160
  configurations to 3 x 1 x 2 x 8 x 9 = 432. Two larger cuts need no plumbing, because both
  parameters exist: budgets=[50] takes 8,160 to 1,020 configurations, models=['rf'] takes it to
  1,440, and both together to 180 — a 45-fold cut.
- **Cut: two evaluators** — s11 passes no internal evaluator list, so 17 (model, evaluator) pairs
  apply. `get_fold_best_search` already accepts one: internal_evaluators=['rf','lightgbm'] gives 10
  pairs and takes the grid from 8,160 to 4,800 configurations per search, removing 16,800
  fold-evaluations per search and 151,200 across the 9 searches. In fits the saving is 957,600,
  because 6 of the 9 searches pay 9 fits per fold-evaluation.
- **Notes** — Under `KIRBY_NO_COLD_SEARCH` the hybrid arms disappear without an error. Tier
  'general' passes the precompute-tier test, the seed-molecule identity has 0 stored configurations
  against 8,160 wanted, find_best_hybrid raises at `find_best_hybrid.py:8839-8851` before any fit,
  s11:941-946 swallows it into a null blueprint, and s11:1153-1159 refuses every hybrid condition as
  'no_blueprint'. The result file then reads as complete with its main condition missing.
  EXPERIMENTS_FINALIZED.md:408 records the tier as fast; the code passes no speed and runs the full
  grid. The rebuild comparison is 1 build against 4: rebuild_interval defaults to 0 at s11:861 while
  the refit arm passes 25 at s11:1230.

---

### s12 — cross-domain cliffs

On molecules whose close structural neighbours have very different activity, does a hybrid predict
activity better than the best single base representation?

- **Hybrids** — precomputed read, through `get_fold_pool_recipes` at s12:371-384 with
  pool_subset='predictive_only' (s12:58) and 3 recipes per fold. Members are `general` under
  conditions none, uncapped and capped, objective predictive, all three written by
  `precompute_hybrids.py:586-590`. The file itself runs no search. The job can: under
  `similarity_helpers.run_all`, a precompute call at `similarity_helpers.py:993-1007` builds the
  general-tier grid inside this job when the store is cold. Run standalone through s12:800-825, every
  read falls through to find_best_hybrid instead.
- **Compute** — per dataset: 250 baseline fits (10 representations x 5 models x 5 folds), 75 hybrid
  builds at 20 fits each = 1,500 fits, 75 hybrid downstream fits. Headline: 1,825 model fits per
  dataset as a floor. A recipe carrying the uncapped or capped augmentation strategy adds 8 fits per
  build, up to 600 more, so the range is 1,825 to ~2,425 model fits. The cliff scan adds no fits: one
  distance matrix over at most 5,000 molecules, reused across a 5-entry percentile sweep. Not
  excessive while the general-tier precompute is on disk.
- **Hybrids it uses / should use** — 3 per (downstream model, fold), ranked by mean stored score over
  the other four folds: 75 builds per dataset. A cliff pair is two molecules whose similarity falls
  in the most-similar few per cent of all pairs, measured on the standardised concatenation of every
  representation. s12 excludes similarity-objective hybrids deliberately at s12:53-58, because the
  cliffs live in a space outside every hybrid. That holds for defining cliffs; the prediction arm on
  those molecules is a plain supervised fit, where predictive hybrids fit. What is missing is width:
  100 features is pinned at s12:341 and no budget flag exists, so 10-, 25- and 50-feature hybrids are
  never asked for.
- **Selection pattern** — 5 evaluators (rf, xgboost, svm, dnn, lightgbm at s12:261-262), all 10 base
  representations, 3 hybrids per model per fold: 13 spaces x 5 models = 65 cells against the intended
  20.
- **Cut: fast tier** — nothing to cut safely. The only path to a search is the fall-through on a
  missing stored search, and kwargs at s12:371-384 do reach it — but speed='fast' would also change
  the metadata of all three precomputed predictive members and make each miss the precompute it
  should hit.
- **Cut: two evaluators** — s12:261-262, 5 models to 2. Baselines fall from 250 to 100 fits and
  hybrid arms from 75 builds (1,575 fits) to 30 builds (630 fits): 1,825 model fits down to 730. A
  second cut at s12:411, restricting the loop over all 10 base representations to the top 2, removes
  200 of the 250 baseline fits.
- **Notes** — EXPERIMENTS_FINALIZED.md:409 says s12 calls get_fold_best_search and marks the pool
  read as not yet implemented; there is no such call in the file and s12:631 records 0 searches run by
  it. EXPERIMENTS_FINALIZED.md:473 lists s12 among files whose pool subset includes hybrids built to
  predict each molecule's activity from its five nearest neighbours; the code says predictive_only.
  Cliff membership is decided with the full-dataset labels in hand, recorded as accepted at
  s12:279-280. Non-cliff scores are computed and stored on every fold for every representation and
  hybrid, and no difference, ratio or honest block is built from them. Four imports are never called.

---

### s13 — clustering and retrieval

In a hybrid's feature space, do k-means clusters group held-out molecules with more similar activity,
and are a held-out molecule's five nearest training neighbours closer in activity, than in any single
base representation's space?

- **Hybrids** — precomputed read, through `get_fold_pool_recipes` at s13:292-302 with distance=True,
  budget=None and pool_subset='predictive_and_similarity' (s13:103). Four members:
  `general_distance` and `general_distance__obj-similarity`, each under 'none' and 'uncapped'. The
  two predictive members are written by `precompute_hybrids.py:575-577`. The two similarity members
  are written by nothing, for the same reason as s06 and s08. The file runs no search of its own; on a
  miss `get_precomputed_search` falls through to find_best_hybrid at `_shared.py:5352`.
- **Compute** — per dataset with both families on disk: 6 hybrid slots x 5 folds = 30 builds. The 15
  predictive builds cost 20 fits each = 300 model fits. The 15 similarity builds fit no downstream
  model, so 10 importance fits each = 150 fits plus 150 neighbour-geometry scorings. Add 30
  importance fits for the weighted retrieval arm. Headline: ~480 model fits per dataset, ~315 when
  the similarity family is dropped, plus about 345 KMeans fits. One cold similarity member costs
  10,800 configurations, of which 8,100 raise and 2,700 evaluate over 5 folds, giving 135,000
  per-representation geometry scorings; across two augmentation conditions, 270,000. Excessive only
  in that cold case.
- **Hybrids it uses / should use** — up to 6 per fold, 3 per family, each ranked within its own
  family on the other four folds' stored scores. s13 fits no downstream model for its own statistics
  and both are distances between molecules, so similarity-objective hybrids belong here more clearly
  than anywhere else in the wave. It already asks for them at s13:103 and for every budget from 10 to
  500 features at s13:295. The problem is that half of what it asks for has no precompute behind it.
- **Selection pattern** — no evaluator axis exists: the two statistics are k-means label coherence
  and five-nearest-neighbour activity distance, neither of which fits a supervised model. About 23
  spaces per fold are scored — 10 base representations, 1 fixed 100-feature PCA control, one
  width-matched PCA control per distinct hybrid width, and up to 6 hybrid slots.
- **Cut: fast tier** — no search is written here, but kwargs at s13:292-302 reach the cold rebuild.
  speed='fast' takes that grid from 5 allocation methods x 3 importance methods x 4 representation
  counts x 6 budgets x 30 (model, evaluator) pairs = 10,800 configurations to 3 x 1 x 2 x 6 x 30 =
  1,080. The kwarg applies to every pool member, so the two precomputed predictive distance searches
  would be rebuilt too. The fix that costs nothing is running the missing precompute:
  `precompute_hybrids.py --distance --objectives similarity`.
- **Cut: two evaluators** — no downstream evaluator list exists. The two levers that behave like one
  are n_clusters_list at s13:229 ([3, 5, 10]; dropping to 2 counts removes 115 of 345 KMeans fits)
  and N_HYBRIDS_PER_FAMILY at s13:104 (3 to 2 takes hybrid slots from 6 to 4 and builds from 30 to
  20, saving about a third of the ~480 model fits).
- **Notes** — The defect worth acting on: s13 asks for four pool members and only two are written
  anywhere. Under `KIRBY_NO_COLD_SEARCH` the other two are caught at `_shared.py:5491-5500` and
  dropped with a printed warning; without it they cost a cold distance-lineage geometry grid. Either
  way the half s13:33-34 names as its clearest case is the half that does not run.
  EXPERIMENTS_FINALIZED.md:410 marks the pool read as not yet implemented and records the tier as
  fast; the read is implemented, and nothing in the file passes speed. build_fold_hybrid at
  s13:481-486 passes no model and no metric function, so for the predictive family the accept-or-
  reject test inside create_hybrid falls back to the internal evaluator's r2_score rather than the
  metric the recipe was ranked under; for the similarity family passing neither is required. The
  weighted-retrieval importance fit at s13:552-557 uses each slot's own importance method, so on a
  deeplift or treeshap recipe it is a neural or SHAP pass rather than a tree fit.

---

## Noise — c10, n01–n23

### c10 — noise robustness curves

When you corrupt the training labels, does a hybrid lose accuracy more slowly than any single representation, and do the same features keep getting picked as the corruption rises?

- **Hybrids** — its own. `tests/experiments/noise/c10_noise_robustness.py:305-319` calls `find_best_hybrid`. It passes `tier='general'`, but any non-zero noise level makes the engine rewrite the tier to `general__noise_<spec_id>__lb<8 hex>` at `find_best_hybrid.py:8453`. No objective is passed, so predictive. `speed='fast'` at c10:313. Nothing precomputes a noise tier: `_is_precompute_tier` (`find_best_hybrid.py:561-565`) matches only the roots at `find_best_hybrid.py:558`, and the note at `find_best_hybrid.py:8120-8123` says the cold-search guard never fires on a label-noise tier. One search per (noise condition, level) is paid inside c10's own job. There is no `create_hybrid` call; the per-fold feature rebuild goes through `build_fold_hybrid` at c10:696.
- **Compute** — 11 noise conditions (`src/kirby/noise_spec.py:113`) times 5 non-zero levels each (`tests/experiments/dataset_policy.py:410`, 6 levels of which 1 is zero) = 55 searches, plus 1 clean-label search all 11 zero-level cells share (c10:293) = 56 searches per dataset. Configurations per search = 3 allocation methods x 1 importance method x 2 representation counts x 8 budgets x 3 (model, evaluator) pairs = 144. Fits per search = 50 importance pre-pass + 150 allocation pre-pass + 50 baselines + 720 downstream = 970 model fits. 56 searches x 970 fits = 54,320 fits. Phase 3 adds 30 uncached `create_hybrid` rebuilds (1 condition x 6 levels x 5 folds). **Headline: ~54,900 model fits per dataset** on herg_ki at 1,412 molecules, 10 representations, 5 folds, 8 budgets; ~44,900 fits on a dataset with no SMILES, where 2 of the 11 conditions drop (c10:193-200). The per-rebuild fit count in phase 3 is unverified: it is near 20 fits under `performance_weighted` and near 10 fits under `power_2` or `rank_exp_d0.3`. Excessive: 55 of the 56 searches are on tiers nothing precomputes, so the whole grid is paid inside one SLURM wall.
- **Hybrids it uses / should use** — 8 hybrids per (condition, level) cell, one per budget entry. For each budget, `honest_rescore` picks per reported fold whichever recipe scored best on the other folds (c10:351-368), so the hybrid is re-picked per fold. Fold 0 is excluded from both arms (c10:343-344), leaving 4 reported folds. As used: c10 measures accuracy under label corruption and computes no distances between molecules. Phase 3 compares which feature indices get selected, not which molecules are near which.
- **Selection pattern** — different. 1 downstream model (rf, c10:308), all 10 base representations on their own curves (c10:465), 1 per-fold re-picked best single representation (c10:499), 8 hybrids (c10:527). The axis it sweeps is budget, not evaluator.
- **Cut: fast tier** — already fast at c10:313. Going to the full lists would raise the grid from 144 configurations per search to 5 allocation x 3 importance x 4 representation counts x 8 budgets x 3 pairs = 1,440, and the 56 searches from ~54,900 fits to ~420,000 fits.
- **Cut: two evaluators** — the downstream model list is already one name at c10:308. The list left to cut is the internal evaluators, which c10 never passes. Adding `internal_evaluators=['rf','match']` to the call at c10:305-319 takes the (model, evaluator) pairs from 3 to 1, configurations per search from 144 to 48, and the allocation pre-pass from 150 to 50 fits — about 54,900 fits down to ~19,300 fits per dataset. The larger lever is the 56 searches, set by the 11 conditions at c10:173-174 and the 6-level grid at `tests/experiments/dataset_policy.py:410`.
- **Notes** — `EXPERIMENTS_FINALIZED.md:367` is stale: it says c10 reads a stored clean-label search and builds features with a bare `create_hybrid`. c10 does neither. `EXPERIMENTS_FINALIZED.md:1196` says 62 searches per dataset; the code gives 56. The comment at c10:285-287 says fast means 12 recipes per (model, budget) instead of 75; the fast lists give 18 and the full lists give 180. `augmentation_strategy` is never passed, so it defaults to `uncapped` (`find_best_hybrid.py:7784`) — a run with `--augment` on a dataset holding donor representations searches the uncapped condition on all 56 noise tiers.

### n01 — activity cliff framework

On pairs of molecules that look alike but have very different activity, does a hybrid have a smaller error gap between those molecules and the rest than any single representation does?

- **Hybrids** — its own. `tests/experiments/noise/n01_activity_cliff_framework.py:485-506` calls `find_best_hybrid` inside the outer-fold loop at n01:483, so 5 searches per dataset. `tier='general'`, no objective (predictive), `speed='fast'` at n01:494, `cache_namespace=nested_cv_cache_namespace(...)` at n01:504-505, which appends `ocv5f<i>s42` to the dataset key for every cache path and the result database (`find_best_hybrid.py:8204-8206`). Nothing warms those namespaces — `precompute_hybrids.py` never calls `nested_cv_cache_namespace` — so all 5 searches are cold. With `KIRBY_NO_COLD_SEARCH` set they are refused rather than run, because `general` is a protected root (`find_best_hybrid.py:558`) and the namespaced result database is empty.
- **Compute** — configurations per outer fold = 3 allocation methods x 1 importance method x 2 representation counts x 8 budgets x 17 (model, evaluator) pairs = 816. The 17 pairs come from 5 downstream models against the explicit evaluators, with `match` summed over models (`find_best_hybrid.py:568-590`). The explicit model list bypasses the fast-tier model cut at `find_best_hybrid.py:8221`, so all 5 models stay. Fits per outer fold = 50 importance pre-pass + 250 allocation pre-pass + 250 baselines + 4,080 downstream = 4,630. 5 outer folds x 4,630 fits = 23,150 fits. The scoring loop adds 250 base-representation fits (n01:671), 25 hybrid downstream fits (n01:707) and 25 uncached `create_hybrid` rebuilds (n01:698). **Headline: ~23,900 model fits per dataset** on herg_ki at 1,412 molecules, 10 representations, 5 outer folds, 8 budgets. The 25 rebuilds are counted at ~20 fits each, which is unverified and is nearer 10 fits each under an allocation method that needs no per-representation model fit. Excessive: 5 cold searches per dataset at full model width, in namespaces no precompute fills.
- **Hybrids it uses / should use** — 5 hybrids per dataset per fold, one per downstream model, each that model's best configuration across all 8 budget entries (`budget=None` at n01:521). A model the search never scored by name inherits another model's recipe, recorded at n01:532-540. This experiment rests on distances between molecules: the cliff reference is ECFP4 Tanimoto (n01:371), edit distance on heavy chains (n01:364), reaction or drug Tanimoto (n01:391, n01:399), or Euclidean on the concatenated representations (n01:342). The hybrid it grades on those cliffs is chosen under the predictive objective only — n01:485-506 passes no `objective=` and no `pool_subset=`, so hybrids whose features were chosen so each molecule's activity is predictable from its five nearest neighbours are never candidates. Those belong here, and so do small budgets: `budgets=FULL_BUDGETS` at n01:490 spans 10 to 2,000 features and the winner is taken at `budget=None`.
- **Selection pattern** — partial. 5 downstream models (n01:307-308), all 10 base representations per model (n01:668, not the top 2), 1 hybrid per model (n01:521, not the top 2) = 55 (model, representation) pairs against the intended 20. The top-3 list at n01:575 is a reporting field built from scores already paid for.
- **Cut: fast tier** — already fast at n01:494. Going to the full lists would raise each fold's grid from 816 configurations to 5 x 3 x 4 x 8 x 17 = 8,160, and the 5 searches from ~23,150 fits to ~212,000 fits.
- **Cut: two evaluators** — n01:307-308 sets the model list to the 5 names of `DOWNSTREAM_MODELS_NOISE` (`_shared.py:799`). Cutting to 2 takes the pairs from 17 to 6, configurations per fold search from 816 to 288, the allocation pre-pass from 250 to 150 fits and baselines from 250 to 100 fits. Each fold search goes from 4,630 fits to ~1,740, and the 5 searches from 23,150 fits to ~8,700. The scoring loop drops with it: 250 base-representation fits to 100, and 25 rebuilds to 10.
- **Notes** — the comment at n01:453-455 misstates the fast lists: `FAST_IMPORTANCE_METHODS` is 1 entry (`find_best_hybrid.py:3418`) and `FAST_INTERNAL_EVALUATORS` is the same 4 names as the default. Nine imports are dead, including `create_hybrid` (n01:68) and `ActivityCliffDetector` (n01:69). On TCGA and any molecular set where RDKit refuses the SMILES, the cliff reference is built by concatenating the representations being graded (n01:342-353); the file flags that itself as `cliff_reference_independent`.

### n02 — sample weighting

If you downweight training molecules whose activity disagrees with their ten nearest neighbours, does that help a single representation more than it helps a hybrid?

- **Hybrids** — precomputed. Three read sites, all `get_best_search_for_downstream`: n02:393-400 (rf), n02:443-450 (per model, the loop at n02:441), n02:840-847 (rf, predictive only, on a non-predictive family). No `find_best_hybrid` and no `create_hybrid` in the file; features are rebuilt per fold by `build_fold_hybrid` at n02:565. The read path can still become a search: on a miss, `get_precomputed_search` falls through to `find_best_hybrid(..., cache=True)` at `_shared.py:5352`.
- **Compute** — no search fits when the stored searches exist. Conditions 1-2 (n02:252-330) = 10 representations x 5 models x 5 folds x 5 fits = 1,250 model fits. Conditions 3-5 (n02:565-645) = 5 models x 5 folds, giving 25 rebuilds and 150 downstream fits. Pool reads = 7 calls over 2 or 6 members each, 0 new fits on a hit. **Headline: ~1,900 model fits per dataset** on herg_ki at 1,412 molecules, 10 representations, 5 folds, with 0 search fits provided the similarity precompute is on disk. The 25 rebuilds are counted at ~20 fits each, unverified, and are nearer 10 each under an allocation method that fits no per-representation model. Not excessive. If the similarity precompute is missing, each of the 6 pool reads cold-builds a geometry search the engine attempts at 8,160 configurations, of which 2,040 cost fits.
- **Hybrids it uses / should use** — 5 hybrids, one per downstream model, each that model's best entry across all 8 budget entries (`budget=None` at n02:445), out of 2 pool members without donor representations or 6 with them. A model with no pool entry falls back to rf's hybrid (n02:510). This rests on nearest neighbours twice: reliability is each molecule's agreement with its 10 nearest neighbours in the importance-weighted distance (n02:151-158), and the cliff-aware weights come from the same neighbourhood (n02:123). It already admits similarity-objective hybrids at n02:399 and n02:449. It does not weight small feature budgets: `budget=None` at n02:394 and n02:445 lets a 2,000-feature hybrid win the slot in which a ten-nearest-neighbour distance is then computed.
- **Selection pattern** — partial. 5 downstream models (n02:102), all 10 base representations per model (n02:252), 1 hybrid per model (n02:445) = 55 (model, representation) pairs. The top-3 at n02:755-756 is computed after the fact and prunes nothing.
- **Cut: fast tier** — nothing to cut in this file: no search of its own. A pool miss at `_shared.py:5352` passes no speed argument, so it runs the full grid. Forcing fast would have to be done in `_shared.get_precomputed_search`'s `call_kwargs` at `_shared.py:5320-5348`.
- **Cut: two evaluators** — n02:102 holds 5 model names. Cutting to 2 takes conditions 1-2 from 1,250 to 500 model fits, conditions 3-5 from 25 rebuilds and 150 downstream fits to 10 rebuilds and 60 fits, and the per-model pool reads from 5 to 2: about 1,900 fits down to ~760 fits.
- **Notes** — the comment at n02:779-781 names a model list including ridge; `WEIGHTED_MODELS` at n02:102 is rf, xgboost, lightgbm, svm, dnn and no ridge model is ever built. `EXPERIMENTS_FINALIZED.md:385` predates both the model list and the pool subset. `results['summary']` is created at n02:242 and never filled. The torch-before-LightGBM import guard at n02:70-73 exists because `lightgbm_gain` followed by `deeplift` segfaulted with exit 139; reordering n02:106 reopens that.

### n06 — BindingDB ground truth

When the same drug and protein were measured twice by different labs, does each noise-detection method flag the molecules whose two measurements disagree, in the same order it did on noise injected on purpose?

- **Hybrids** — precomputed. One read at `tests/experiments/noise/n06_bindingdb_ground_truth.py:311-320`, `get_best_search_for_downstream(..., downstream_model='rf', budget=budget, pool_subset=pool_subset)`, with `pool_subset='predictive_and_similarity'` unless `--no-similarity-arm` (n06:635-636). Features are rebuilt by `build_fold_hybrid` at n06:345, which reaches `create_hybrid` at `_shared.py:4772`. No direct `find_best_hybrid` and no direct `create_hybrid` call. That one read expands to 6 pool members on a dataset with donor representations: tier `general` for all, objective predictive for 3 and similarity for 3. No speed argument is passed on this path, so a miss runs at `speed='full'` (`find_best_hybrid.py:7764`). A precompute exists for all 6 under `noise_helpers.run_all` (`noise_helpers.py:817` and `:846`), both behind `if len(reps_data) >= 2` at `noise_helpers.py:814`, and the similarity one inside a try/except at `noise_helpers.py:845-869`. Standalone, every member takes the on-demand path at `_shared.py:5302-5352`.
- **Compute** — 3 feature spaces (1 predictive hybrid, 1 similarity hybrid, 1 base representation) x 5 folds (`N_FOLDS` at n06:104, hardcoded) x 14 detection methods (`resolve_estimators` at n06:171). Fits inside one `estimate()` call sum to 130 model fits, from `src/kirby/utils/noise_estimation.py`: ensemble 50, deep_ensemble 25, quantile_regression 15, and 5 each for hybrid, cross_val, gaussian_process, conformal, bayesian_rf, quantile_forest, heteroscedastic_nn and ngboost. 3 spaces x 5 folds x 130 fits = 1,950 fits. Hybrid builds = 2 spaces x 5 folds = 10 `create_hybrid` calls, costing 20 fits each for the predictive recipe and 10 each for the similarity one = 150 fits. **Headline: ~2,100 model fits per dataset** with torch and ngboost installed, and 0 new search configurations when both stored families are on disk. Without torch and ngboost the detector list drops to 10 names and the total falls to ~1,350 fits. The default dataset is bindingdb_kd at 30,000 rows (n06:1077-1079), so wall time per fit is well above a mid-sized molecular set. Not excessive by fit count; the exposure is the cold-search case.
- **Hybrids it uses / should use** — 2 hybrids, one per search family: the pool's winner (n06:687) and the best entry of every other eligible family from `meta['by_objective']` (n06:689-699). Both are picked for rf at the budget the search chose, through the fallback at `_shared.py:6464-6529`. They are never ranked against each other, because a predictive search's stored score is the dataset's metric and a similarity search's is a neighbour correlation. This rests on distances: every neighbour estimator and the exclusion machinery at n06:200 use nearest neighbours, and the similarity-objective hybrid is already a first-class arm. It does not ask for small feature budgets: `budget` is None (n06:623), so the best across all 8 budget entries wins. `--budget 25` or `--budget 50` at n06:1064 costs nothing extra, because the stored search holds every budget.
- **Selection pattern** — different. It fits no downstream prediction model of its own: it compares 14 noise-detection methods across 3 feature spaces with rf hardcoded at n06:313, and reports each method's rank agreement with the repeat-measurement disagreement.
- **Cut: fast tier** — adding `speed='fast'` at n06:311-320 forwards through `_shared.py:5626` into `find_best_hybrid` at `_shared.py:5350-5352`. The predictive grid goes from 8,160 configurations to 432, and the similarity grid goes from 8,160 to 432 as well — nothing collapses the model or evaluator axis under a geometry objective. It would cost here rather than save: those four lists are in the search cache key (`_make_search_meta`, `find_best_hybrid.py:673`, lists emitted at `:814-826`), so a fast request misses the full precompute on disk and builds 432 configurations instead of reading 8,160 stored ones. It saves only if `precompute_hybrids.py` runs fast on that dataset too.
- **Cut: two evaluators** — nothing to cut: n06 asks the pool for rf alone at n06:313. The costly axis is the 14-name estimator list at n06:171, where ensemble is 50 of the 130 fits per (space, fold) and deep_ensemble another 25. `KIRBY_NOISE_LIGHT_ESTIMATORS=1` (n06:180) cuts it to 4 names fitting 5 models between them, taking ~1,950 estimator fits per dataset to ~75. It also drops the headline: 4 estimators is below `MIN_ESTIMATORS_FOR_TRANSFER = 5` (n06:490), so the rank correlation with n11 and n13 is refused (n06:608-615).
- **Notes** — the cold-search guard is opt-in, set by `KIRBY_NO_COLD_SEARCH` (`find_best_hybrid.py:534-554`), so an unguarded or standalone run against a cold disk cache pays a full 8,160-configuration search per member inside the experiment's wall. n06 writes a recorded skip where no ground-truth provider finds at least 50 repeat groups (`_ground_truth_providers.py:46`, checked at n06:586-592). The comment at n06:603-607 says 4 distinct estimators survive on a classification dataset; the intersection at n06:196 resolves to 3, which is still below the 5 a rank correlation needs. `noise_helpers.py:729-736` passes `models` and `internal_evaluators` in `config=` to `precompute_hybrid_searches`, which never reads `config` — both entries are dead.

### n07 — remediation

After a tenth of the training activity values have been deliberately corrupted, which of seven cleanup methods puts back the most of the accuracy that was lost?

- **Hybrids** — precomputed. `tests/experiments/noise/n07_remediation.py:720-727` calls `get_best_search_for_downstream` once per model in the 5-name list at n07:658-660, with `budget=100` features (n07:669) and `pool_subset='predictive_and_similarity'` (n07:708). One more call at n07:824-831 when rf produced no search. Features are rebuilt by `build_fold_hybrid` at n07:1233. No direct `find_best_hybrid` and no direct `create_hybrid` call. Each lookup expands to 6 pool members on a dataset with donor representations, tier `general`, objective predictive for 3 and similarity for 3. No speed argument reaches the engine, so a miss runs the full grid. A cold standalone run costs 6 searches, not 30: the first call writes the disk cache with `cache=True` and the later 4 model calls read it back.
- **Compute** — 5 folds from `config['n_splits']` at n07:666, which is 5 on a helper run and 1 standalone at the parser's default `--mode intermediary`. Block A (n07:427, called n07:911) = 5 models x 10 representations x 5 folds = 250 model fits. Block B (n07:1189-1399) = 2 top models x 1,110 fits + 3 other models x 570 fits = 3,930 fits, of which 5 models x 5 folds x 4 builds x 20 fits = 2,000 are inside `create_hybrid`. Block C (n07:1463-1526), random forest only = 10 representations x 5 folds x 16 fits = 800 fits. Block D (n07:1544-1625) = 2 models x 2 representations x 5 folds x 70 fits = 1,400 fits. **Headline: ~6,380 model fits per dataset on a helper run at 5 folds, plus 100 `create_hybrid` builds**, and 0 new search configurations when both stored families are on disk. Standalone at 1 fold it is roughly 1,280 fits and 20 builds. Wall time is dominated by the builds: the comment at `hybrid.py:1674-1678` measures one build at 336 s on flab_thermo_tresanco, which puts 100 builds at roughly 9 hours before a strategy is scored — an extrapolation from one measurement on one dataset, not an n07 measurement.
- **Hybrids it uses / should use** — 5 hybrids, one per downstream model, each the best entry for that model at 100 features across the 6 pool members. Features are rebuilt per fold and per noise draw, twice: once against the corrupted training labels and once against the clean ones as a named oracle arm (n07:1319-1339). Four of the 7 cleanups read a molecule's neighbours: `label_smooth` and `knn_smooth` move a flagged label to the mean of its 10 nearest training molecules (n07:284-319), `downweight` reweights from a neighbour-based suspicion score, and that score comes from `ActivityCliffAwareEstimator(k=10)` at n07:217. The similarity family is admitted at n07:708 but cannot be built — see Notes. The budget is pinned at 100 features by n07:669 and never swept, so the 10-, 25- and 50-feature entries already in the stored search are never asked for. A `--budget` flag beside n07:1844, forwarded from n07:1856, costs no new search.
- **Selection pattern** — partial. Block D is the intended shape and says so at n07:1529-1535: 2 downstream models x 2 base representations = 4 pairs, all 7 cleanups. Block B runs 5 models x 1 hybrid x 2 feature spaces, Block C runs 1 model x all 10 base representations, and no block compares 2 hybrids, because `get_best_for_downstream` returns exactly one configuration.
- **Cut: fast tier** — adding `speed='fast'` at n07:720-727 forwards through `_shared.py:5626` into `find_best_hybrid`. The predictive grid goes from 8,160 configurations to 432, and the similarity grid does the same. It only saves if `precompute_hybrids.py` runs fast for that dataset, because those lists are the search cache key (`find_best_hybrid.py:814-826`). It would also change which recipe each model gets: the fast tier scores only rf, lightgbm and dnn, so xgboost and svm would return no entry and land in `models_missing_blueprint` (n07:781).
- **Cut: two evaluators** — n07:658-660 is where the 5-model list enters. Cutting to 2 takes the pool lookups from 5 to 2, Block A from 250 to 100 fits, and Block B from 3,930 to 2,220 fits with its builds from 100 to 40. Blocks C and D do not move. Total goes from ~6,380 fits to ~4,520 fits, and from 100 builds to 40. The machinery to pick the 2 is already there: `rank_models_and_reps` at n07:427 with `N_TOP_MODELS = 2` at n07:380. It reorders the file, because that ranking runs at n07:911 after the blueprints are fetched at n07:718 and needs `models_with_blueprint` from that fetch.
- **Notes** — n07 admits similarity-family recipes but cannot build one. `build_fold_hybrid` raises when a geometry objective arrives with `model=` or `metric_fn=` (`_shared.py:4732-4738`), and n07:1233-1237 passes both. The closure swallows the exception and returns None (n07:1238-1240), the fold is skipped (n07:1241-1243), and the model contributes zero rows to Block B with no entry in `models_missing_blueprint`. n06 is not exposed, because its rebuild passes neither argument. Three further defects: `results['detection']` is always empty, because the dict at n07:978 is built before the loop at n07:1311-1317 fills `detection_acc`; `EXPERIMENTS_FINALIZED.md:2346-2382` is stale in 5 places and the code wins on all 5; `rank_models_and_reps` calls `cv_split` inside its innermost loop at n07:465, re-deriving the same split 50 times per run at no fit cost.

### n09 — noisy-label learning survey

When training labels are corrupted, does swapping one plain representation for a per-fold hybrid change which of the 12 noisy-label learning methods comes out on top?

- **Hybrids** — both. `tests/experiments/noise/n09_lnl_survey.py:437` calls `find_best_hybrid` with no tier, no dataset_name, no speed and no objective: so predictive, `speed='full'`, and `cache=False`, which also skips the cold-search guard (`find_best_hybrid.py:8841-8842`). Budgets are `[50,100,250]` at n09:104, inner folds are 3 under `--mode full` (n09:433), and the model list is overridden to rf alone. Nothing precomputes this combination and nothing could: `cache=False` reads and writes no result database. Two pool reads at n09:261 and n09:269 are lookups under the suite runner; standalone they cold-call `find_best_hybrid(cache=True, budgets=FULL_BUDGETS, tier='general', models=None)` at `_shared.py:5331` and `:5352`. `build_fold_hybrid` at n09:492 rebuilds features per pass.
- **Compute** — searches = (1 draw at level 0 plus 5 levels x 3 draws) x 5 folds = 80 per dataset (n09:389-393). Configurations per search = 5 allocation methods x 1 model x 3 budgets x 3 importance methods x 3 (model, evaluator) pairs x 4 representation counts = 540. Fits per search = 90 importance pre-pass + 90 allocation pre-pass + 30 baselines + 14,580 search loop = 14,790 model fits. The search loop is 9 fits per (configuration, fold): 3 out-of-fold residual fits, 5 augmentation-ladder fits and 1 rf fit. 80 searches x 14,790 fits = 1,183,200 fits. Add 80 rebuilds at ~28 fits each = 2,240, 240 plain-DNN fits, and 2,700 noisy-label method trainings. **Headline: ~1,188,000 model fits per dataset** on herg_ki at 1,412 molecules, of which 1,183,200 sit inside the 80 uncached searches. Excessive: 80 full-grid searches share nothing with each other, and the per-fold importance and allocation caches are off because `dataset_name` is not passed (`find_best_hybrid.py:6161`).
- **Hybrids it uses / should use** — 1 hybrid per (level, fold, draw), plus the same hybrid with each feature multiplied by the square root of its importance weight (n09:554-557). The hybrid is picked at n09:455-468 as the best score over the 3 budgets the inner search recorded for rf. Three of the 12 methods read the geometry directly: `DistanceBasedSelection` (`lnl_methods.py:722`) ranks each training molecule by its distance to its 10 nearest neighbours, and `heteroscedastic_gp` and `evidential_kernel` both put a kernel over those distances. So hybrids whose features were chosen so each molecule's activity is predictable from its five nearest neighbours are the on-point test, and so are the 10-, 25- and 50-feature budgets. n09 asks for neither: the pool read at n09:261 takes the default `predictive_only` and the inner search at n09:437 passes no objective, and the budget grid at n09:104 starts at 50 features.
- **Selection pattern** — different. There is no evaluator roster and no top-2 step: it scores 1 plain DNN and 12 noisy-label methods across 3 feature sets (1 base representation, 1 hybrid, 1 weighted hybrid), giving 36 method arms plus 3 plain-DNN arms per noise level.
- **Cut: fast tier** — add `speed='fast'` at n09:437-455. The grid goes from 540 configurations to 3 allocation x 1 importance x 2 representation counts x 3 budgets x 3 pairs = 54. The evaluators do not change, and the pairs do not change, because the model list is explicit and the fast model filter sits inside `if models is None` (`find_best_hybrid.py:8221-8231`). The importance pre-pass drops from 90 to 30 fits. One search goes from 14,790 fits to 1,608, and the dataset from 1,183,200 fits to ~128,600 — a factor of about 9.2.
- **Cut: two evaluators** — no downstream evaluator list exists here; the roster is the 12 methods at n09:243 and 1 plain DNN, and the inner search already runs 1 downstream model. The nearest axis is the internal evaluator list the search leaves at 4 defaults. Passing `internal_evaluators=['rf','lightgbm']` at n09:437 takes the pairs from 3 to 2 and the grid from 540 to 360 configurations: about 4,890 fits saved per search and ~391,000 per dataset. Combined with `speed='fast'` the grid lands at 36 configurations.
- **Notes** — the docstring at n09:25 says 5 noise levels; the code sweeps 6 (`dataset_policy.py:410`). `EXPERIMENTS_FINALIZED.md` says 30 searches per dataset; the code runs 3 noise draws per fold (n09:233), giving 80. `train_groups_subset` is assigned at n09:431 inside the try opened at n09:425 and read at n09:587, n09:602 and n09:618 outside it; if the augmentation slices at n09:426-427 raise, the name still holds the previous fold's training rows and a later fold carves its checkpoint holdout on another fold's scaffold groups. The `budget` and `clean_pool_budget` fields at n09:319-320 report the clean pool's budget, not what each fold built.

### n11 — estimation and remediation grid

Which noise detector, in which feature set, with which label-repair method at which strength, wins back the most of the performance the corrupted labels cost?

- **Hybrids** — both. `tests/experiments/noise/n11_estimation_remediation_grid.py:684` reads the pool for rf at `budget=None` with `pool_subset='predictive_and_similarity'` (n11:110). `tests/experiments/noise/n11_estimation_remediation_grid.py:312` calls `create_hybrid` to apply a stored recipe, from n11:880, once per pool family per (fold, draw). No `find_best_hybrid` call in the file. The pool is 6 members: 3 augmentation conditions x 2 objective families. The predictive 3 hit the in-memory store under `noise_helpers.run_all`; the similarity 3 cannot (`_shared.py:5289-5298`) and fall through to `find_best_hybrid(..., cache=True)` at `_shared.py:5352` on tier `general`, objective similarity, at full speed. That fall-through cannot read the file the SLURM `sim` pass writes: `precompute_hybrids.py:729` forces `rep_counts=[None]`, the reader normalises to `[2,3,5,None]` (`find_best_hybrid.py:8187`), and `rep_counts` is in the cache meta (`find_best_hybrid.py:824-826`). It can read the one `noise_helpers.run_all` builds in-process at `noise_helpers.py:834-857`. A cold read enumerates 8,160 configurations over 5 folds, of which 6,120 raise at `find_best_hybrid.py:4673` before any fit and 2,040 do work.
- **Compute** — feature sets per (fold, draw) = 1 control representation + 2 pool families + 9 other representations = 12 (n11:288-347). Detector settings per feature set = 4 neighbour detectors x 4 distance metrics, plus 6, 9 or 10 non-neighbour detectors depending on whether torch and ngboost import = 22, 25 or 26 settings; `requirements-working.txt:262` pins ngboost, so 26 on the cluster. Repair rows per (feature set, detector) = 60 weighting fits + 80 or 240 smoothing fits + 20 clip fits. Per pass = 3 feature sets x 26 x 300 + 9 x 26 x 140 + 12 x 20 = 56,400 downstream fits. Passes = 6 noise levels x 5 folds x 3 draws = 90. 90 x 56,400 = 5,076,000 fits. Add 28,080 detector fits, 2,700 target fits, ~5,000 fits inside 180 rebuilds, 150 clean reference fits and 450 noisy reference fits. **Headline: ~5.1 million model fits per dataset** on herg_ki at 1,412 molecules with torch and ngboost importable; ~4.3 million with neither. Excessive by a wide margin, and the detector count is set at runtime by two imports, so which figure applies to the cluster image is unverified.
- **Hybrids it uses / should use** — 2 at most, one per objective family, from `meta['by_objective']` at n11:696-710. Each is the single best entry of its family for rf at whatever budget the search recorded, because n11:686 passes `budget=None`. Both are rebuilt per (fold, draw) on that draw's corrupted labels at n11:312. This rests on distances: 4 of its detectors score a molecule against its nearest neighbours, swept over all 4 distance metrics (n11:126, n11:951-956), and the similarity family is already admitted at n11:110. It does not ask for a small feature budget, so a 2,000-feature space can be the one a nearest-neighbour detector runs in; 10, 25 or 50 features is the on-point test. Setting `budget=` at n11:686, or building one space per small budget, is the change. Note what the hybrid does not touch: every downstream refit is on the base representation (n11:846-850, passed at n11:965), so the feature set decides only where noise is detected.
- **Selection pattern** — partial. The downstream roster is right at 5 models (n11:599), but it scores all 10 base representations as separate spaces (n11:343-347) rather than the top 2, and 2 hybrids chosen one per family rather than the top 2 by accuracy: 12 feature sets x 5 models = 60 pairs against the intended 20.
- **Cut: fast tier** — no search of n11's own, and `get_best_search_for_downstream` has no speed argument (`_shared.py:5509-5520`). Where the similarity search is not on disk under the `rep_counts` key this read produces, n11:684 cold-builds 8,160 enumerated configurations over 5 folds with no flag in this file to make it cheaper. Two levers, both outside n11: run the suite through `noise_helpers.run_all` so its own similarity precompute writes the matching key, or pass `rep_counts=[None]` on both sides so the SLURM `sim` pass's file becomes readable.
- **Cut: two evaluators** — n11:599 holds 5 model names, and every remediation row is refitted once per model (n11:491, n11:526, n11:558), so the downstream fit count is linear in the list length: 5 models to 2 takes 5,076,000 fits to 2,030,400 per dataset. The larger cut is n11:343-347, where the 9 non-control base representations account for 32,760 of the 56,400 fits per pass. Cutting those 9 to the top 2 by accuracy takes the spaces from 12 to 5 and the per-pass fits from 56,400 to ~30,780. Both cuts together: 5.08 million fits to ~1.11 million.
- **Notes** — the "~4,800 configs" caveat repeated at n11:828, n11:1239, n11:1241, n11:1402 and n11:1471 is stale by about 12x; it predates the widening from 1 downstream model to 5 (n11:593-598) and the 2 added smoothing targets (n11:181-200). `EXPERIMENTS_FINALIZED.md:2468` marks the similarity pool subset not yet implemented and `:2446` says one random forest is retrained; the code has implemented the subset and fits all 5 models. `get_precomputed_search` is imported at n11:91 and never called. The 3 noise draws are not collapsed at level 0 as n09 collapses them, so on a regression dataset n11 runs 3 identical zero-noise passes per fold.

### n12 — importance-weighted distance

Does multiplying each z-scored feature by the square root of its importance before the distances between molecules are measured make the nearest-neighbour noise score better at finding corrupted training labels?

- **Hybrids** — both. n12:509-516 reads the pool for rf at `budget=None` with `pool_subset='predictive_and_similarity'`. n12:652-665 rebuilds features once per (noise level, fold, noise draw) through `build_fold_hybrid` at n12:219-231, which reaches `create_hybrid` at `_shared.py:4772` with no precomputed importance and no precomputed allocation scores. No `find_best_hybrid` call in the file. The predictive family is served from the in-memory dict; the similarity family is refused there (`_shared.py:5289`) and resolves by calling `find_best_hybrid(tier='general', objective='similarity', 8 budgets, full speed, cache=True)` at `_shared.py:5352`, 3 members per pool read on a dataset with donor representations. The grid is 8,160 configurations per member per fold, of which 2,040 evaluate and 6,120 raise under a geometry objective. A file written by `tests/precompute_hybrids.py` is rejected on the `rep_counts` meta field (`find_best_hybrid.py:418-421`); what keeps the re-entered search from costing fits is the SQLite result store, whose identity excludes the sweep (`find_best_hybrid.py:8606-8612`).
- **Compute** — passes = 6 noise levels x 5 folds x 3 noise draws, less the 2 draws dropped at level 0.0 on a regression set = 80 (n12:599). Clean baselines = 150 fits, noisy baselines = 400 fits. Hybrid rebuilds = 80 passes x ~29 fits = 2,320 fits. Weighting importances = 80 x 2 importance methods x 2 matrices = 320 fits. Weighted refits = 80 passes x 2 importance methods x 5 conditions x 2 estimators x 3 neighbour counts x 5 models = 24,000 fits. **Headline: ~27,200 model fits per dataset for the one gaussian condition** on herg_ki at 1,412 molecules, 10 representations, 5 folds; run over all 11 regression conditions that is ~300,000 model fits per dataset. The 10 allocation-scoring fits inside each rebuild happen only when the winning recipe's allocation method routes through `allocate_performance_weighted`, which is unverified for this dataset, and the augmentation term is zero when the winning condition is `none`.
- **Hybrids it uses / should use** — exactly 1 recipe, the pool's winner for rf at `budget=None` (n12:511), rebuilt 80 times. Conditions `hybrid_imp_base` and `hybrid_imp_own` are the same features reweighted two ways, not two hybrids. This rests entirely on neighbours: all 5 conditions are distance geometries and both estimators score a molecule against its k nearest others (`noise_estimation.py:239` and `:299`). It admits the similarity family at n12:515 but never gives it an arm — with both families on disk the predictive one wins by subset order (`_shared.py:5762-5771`), so the similarity hybrid is only a recorded field. It should carry a similarity-family arm the way n13 does at n13:430-433, and a small-budget arm: `budget=None` takes the winner over 10 to 2,000 features and nothing asks for the 10-, 25- or 50-feature entries.
- **Selection pattern** — different. It compares 5 feature spaces built from 1 base representation (n12:469) and 1 hybrid recipe in 3 weighted variants, across 5 downstream models, 2 estimators, 3 neighbour counts and 2 importance methods.
- **Cut: fast tier** — no search of its own. The only search-capable call is the pool read at n12:509-516, and a `speed='fast'` kwarg there would miss the full-grid precompute and build a fresh search of 3 allocation x 1 importance x 2 representation counts x 6 (model, evaluator) pairs = 36 configurations per budget, 288 over the 8 budgets. That is a cost, not a saving.
- **Cut: two evaluators** — n12:440 holds 5 model names. Cutting to 2 takes the weighted refits from 24,000 to 9,600 fits, noisy baselines from 400 to 160 and clean baselines from 150 to 60: about 27,200 fits down to ~12,500. A second cut needs no decision: `base_unweighted` and `hybrid_unweighted` do not depend on the importance method, yet n12:666-690 re-scores both under each of the 2 methods, so 4,800 of the 24,000 refits recompute numbers already in hand.
- **Notes** — the docstrings at n12:18 and n12:359 say 3 estimators; `ESTIMATOR_FACTORIES` at n12:75 holds 2 since `activity_cliff_aware` was cut, and n12:18 also says 2 sigmas where the resolved grid is 6 levels. The comment at n12:560-561 says 30 hybrid builds; the three-draw change at n12:600-603 makes it 80. The file describes its geometry as Euclidean (n12:121-134) but builds its estimators with `metric=None` at n12:386, which resolves to the process-wide default, manhattan (`distance_config.py:68`) — the geometry it names and the geometry it measures disagree. The 0.0 level yields no recovery number at all, because the clean-to-noisy gap is zero there and recovery is NaN for the whole level (n12:777-780).

### n13 — noise distributions

When the shape of the injected label noise changes, does the same noise-learning method keep winning and does the hybrid keep its advantage over the base representation?

- **Hybrids** — both. Six pool reads per dataset: n13:385-392 for rf and n13:491-497 once per model in the 5-name list at n13:291-292, both with `pool_subset='predictive_and_similarity'` (n13:134). Three `build_fold_hybrid` sites: n13:245-250 (predictive), n13:787-795 (similarity), n13:847-853 (each model's matched recipe, de-duplicated by signature at n13:812 and n13:823-825). All reach `create_hybrid` at `_shared.py:4772`. No `find_best_hybrid` call in the file. The similarity half of each read bypasses the in-memory store and calls `find_best_hybrid(tier='general', objective='similarity', 8 budgets, full speed, cache=True)` at `_shared.py:5352`, 3 members per read. A similarity file written by `tests/precompute_hybrids.py` is rejected on the `rep_counts` meta field, so the search re-enters and is served by the SQLite result store; the in-run precompute at `noise_helpers.py:846` writes a readable key. None of the 6 geometry reads is memoized in memory (`_shared.py:5502-5505`).
- **Compute** — cells = 11 noise conditions x 6 levels = 66 (n13:337-338, n13:357-360). Passes = 66 cells x 5 folds x 3 noise draws = 990. Per pass: 5 noisy-baseline fits, ~28 fits for the predictive rebuild, ~10 for the similarity rebuild, ~28 to ~140 for the matched rebuilds, 5 honest-arm fits, 30 Track B estimator fits and 45 Track B weighted refits = about 151 model fits, plus 36 neural-network trainings of 50 epochs each. Outside the loop: 25 clean baseline fits and 5 clean DNN trainings. **Headline: ~150,000 model fits plus ~36,000 fifty-epoch neural-network trainings per dataset** on herg_ki at 1,412 molecules; ~120,000 model fits where ngboost is not importable. Excessive: nothing inside the file caps the 4 axes, though `noise_helpers.run_all` can narrow the condition axis to one from outside, giving 6 cells and 90 passes.
- **Hybrids it uses / should use** — 3 feature spaces per pass plus 1 matched hybrid per downstream model. The `hybrid` space is the pool's winner for rf at `budget=None`, `hybrid_sim` is the similarity family's own best (n13:430-433, rebuilt at n13:787), and the honest arm builds one matched-evaluator recipe per model (n13:491-509). Every one is refit on that fold's noisy training labels. Neighbours matter here: Track B's `local_variance_ratio` scores each molecule against its k nearest others (`noise_estimation.py:299`) and `distance_selection` is one of the 12 Track A methods (`lnl_methods.py:2245`), and `hybrid_sim` already gives the similarity family its own arm and win tally (n13:1307-1322). Missing is a small-budget arm: `budget=None` at n13:388 and n13:494 takes the winner over 10 to 2,000 features, and no arm asks for 10, 25 or 50 features.
- **Selection pattern** — different. Track A trains 12 label-noise methods through 1 DNN on each of 3 feature spaces; Track B scores molecules for noise in each of the 3 spaces and always refits the downstream model on the base representation with those weights. `extract_top3_base_reps` is called at n13:398 but its 3 names only go into `results['top3_base_reps']` at n13:554.
- **Cut: fast tier** — no search of its own. Forcing `speed='fast'` on the reads at n13:385 or n13:491 would swap in the fast lists, and those lists are fields of the search cache meta (`find_best_hybrid.py:8168-8190`), so it would miss the full-grid precompute and build a fresh fast search.
- **Cut: two evaluators** — n13:291-293 holds 5 model names. Cutting to 2 takes Track B's weighted refits from 45 to 18 per pass, noisy baselines from 5 to 2, honest fits from 5 to 2 and matched-recipe reads from 5 to 2: about 150,000 model fits down to ~117,000. It leaves wall time almost untouched, because Track A's 36 network trainings per pass are per space and per method, not per model. The cuts that move n13 are the space list at n13:877 and n13:921, the 12-method list at n13:144, or the 66 cells at n13:337-338 and n13:357-360.
- **Notes** — `EXPERIMENTS_FINALIZED.md:2532` describes the third space as a hand-built stack and `:2554` marks the pool-subset change not yet implemented; the code has replaced the stack with the stored similarity search and the code wins. The defect at `:2542`, where the similarity hybrid fell into the base pool of the win tally, is fixed at n13:1295. The same document at `:2552` counts 900 passes; the code adds censoring on its own 6-level grid, giving 990 on a dataset with SMILES and 54 cells where the 2 grouped conditions drop. `get_noise_estimator` is called with no groups at n13:933, so `gaussian_process` and ngboost run their internal 5-fold pass with shuffling off and scaffolds straddle those internal folds; the outer split is unaffected. On a dataset over 10,000 molecules svm caps its training rows at 10,000 and the file warns rather than dropping it (n13:297-307).

### n14 — activity cliff landscape

How many molecules in this dataset sit on an activity cliff — a structurally close partner whose activity differs by more than 2 log units — and are those cliffs clustered or scattered?

- **Hybrids** — none. Grep over the whole file for `find_best_hybrid`, `create_hybrid`, `build_fold_hybrid`, `get_best_search_for_downstream`, `evaluate_cv` and `run_nested_cv` returns nothing. No tier string, no objective, no budget list and no speed argument appears in the file. `config` is read once, at n14:404, for `n_splits` alone.
- **Compute** — 1 out-of-fold LightGBM regressor per fold, over `max(2, config['n_splits'])` folds (n14:404-405), through `oof_predict` at `src/kirby/utils/noisy_labels/core.py:150-157`. **Headline: 5 LightGBM fits per dataset under `--mode full`**, 2 under `--mode intermediary`, with 0 downstream evaluation fits and 0 hybrid builds. Not excessive. The only thing that grows is the molecule-by-molecule similarity matrix, guarded at 15,000 molecules (n14:324).
- **Hybrids it uses / should use** — none. It evaluates no hybrid and no base representation; `data['reps']` is used once, at n14:399, where every loaded representation is concatenated into one matrix for the residual diagnostic. It does rest on distances between molecules — an ECFP4 Tanimoto similarity between every pair (n14:281) — but that similarity comes from fingerprints computed off SMILES, not from a hybrid's feature space. Nothing to change. If a hybrid arm were ever added, similarity-objective hybrids and the small budget entries would be the right candidates, because the whole output is a neighbour graph.
- **Selection pattern** — different. It counts cliff pairs from an ECFP4 Tanimoto matrix plus a 2-log activity gap, reports the cliff graph's component sizes, and runs 1 out-of-fold LightGBM residual diagnostic.
- **Cut: fast tier** — nothing to cut: no search.
- **Cut: two evaluators** — nothing to cut: there is no downstream evaluator list. The only model-count knob is `n_splits` at n14:404, already 5 LightGBM fits.
- **Notes** — `resid_sd_over_kramer_floor` at n14:134-135 divides the residual spread by 0.54 pKi on every molecular regression dataset, so on logd or caco2 it compares a residual against an assay figure for a different endpoint; `EXPERIMENTS_FINALIZED.md:2587` flags the same thing. The residual diagnostic's feature space is every loaded representation concatenated (n14:399-401), which is used as a feature space nowhere else in the noise suite.

### n15 — clean versus noisy hybrid search

How much of a hybrid recipe's apparent advantage under corrupted labels comes from the recipe having been chosen while the labels were still clean?

- **Hybrids** — both. Two fetches, n15:305 (clean) and n15:309 (noisy), both landing on `get_best_search_for_downstream` at n15:175 with `speed='fast'` (n15:154), `tune=True`, `tune_n_iter=20`, `tune_knn_k=True` and an aggressive pruning schedule (n15:153-161). The clean lineage is tier `general`, objective predictive; no precompute exists for that combination, because every list `speed` selects plus the tune token is in the search cache key (`find_best_hybrid.py:673` and `:744`). The noisy lineage's tier becomes `general__noise_<spec_id>__lb<8 hex>__selnoisy` (`find_best_hybrid.py:8453` and `:8460`), which nothing precomputes — `noise_helpers.py:891` sets the noisy store to empty deliberately, and the cold-search guard never fires on it. Features are rebuilt by `build_fold_hybrid` at n15:371 with precomputed importance and allocation scores passed in.
- **Compute** — noise specs = 1 draw at level 0.0 plus 3 draws at 0.15 plus 3 draws at 0.30 = 7. Augmentation conditions in the pool = 3 under the suite runner, 1 standalone. Live noisy searches = 7 specs x 3 conditions = 21, or 7 standalone. Fast grid per search = 3 allocation methods x 1 importance method x 2 representation counts x 8 budgets x 9 (model, evaluator) pairs = 432 configurations. Tuning turns 1 downstream fit into 61 fits for rf and lightgbm, 1 for dnn. Search fits = 88,560 per search. Baselines (6,150 fits), the importance pre-pass (50 fits) and the allocation pre-pass (150 fits) are paid once per noise spec and read from disk for the other 2 augmentation conditions. **Headline: ~1.90 million model fits per dataset** on herg_ki at 1,412 molecules under the suite runner — 21 searches x 88,560 fits = 1,859,760, plus 7 x 6,350 fits = 44,450. Standalone it is 7 searches and ~664,000 fits. Excessive: every one of those searches is a cold build inside the experiment's own SLURM wall. The aggressive pruning schedule lowers the number actually fitted; F16's own calibration claims 58 fewer evaluations in every 100, unverified against this grid.
- **Hybrids it uses / should use** — exactly 2 per cell: the single best configuration for that downstream model from the clean pool and the single best from the noisy pool, each at `budget=None`, so best across all 8 budget entries. Cells producing a number: 7 noise specs x 3 downstream models = 21, of the 35 the loop attempts. As used: n15 computes no distance between molecules and uses no neighbours.
- **Selection pattern** — different. Two hybrid recipes per (noise draw, downstream model) and no base representation at all. 5 downstream model names are looped (n15:248-249), 3 of which produce a cell; 0 base representations; 2 hybrids per cell. `extract_top3_base_reps` is imported at n15:92 and never called.
- **Cut: fast tier** — already fast at n15:154, with the aggressive pruning schedule at n15:159-160. Under a suite run the clean arm throws both away — see Notes.
- **Cut: two evaluators** — n15:248-249 holds 5 model names, but cutting it saves almost nothing, because the 21 searches do not depend on `downstream_model`. The cut that bites is passing `models=['rf','dnn']` into the search kwargs at n15:153-161: with an explicit model list the speed filter does not apply (`find_best_hybrid.py:8221`), the pairs go from 9 to 6, the grid from 432 to 288 configurations, and the per-search downstream fits from 88,560 to 44,640. The other lever is the 21 searches: 7 noise draws x 3 augmentation conditions. Dropping the augmentation pool members takes it to 7 searches.
- **Notes** — under a suite run the clean arm never sees the settings n15 says both arms share. `get_precomputed_search` returns the in-memory stored search at `_shared.py:5288-5300` before reading any of those keywords, and `noise_helpers.run_all` fills that store at `noise_helpers.py:829`. So the clean arm is the full untuned precompute at 5 allocation methods x 3 importance methods x 4 representation counts x 17 pairs, and the noisy arm is a live fast tuned search at 3 x 1 x 2 x 9. The two arms differ in two ways at once, not one, so `oracle_advantage` under a suite run mixes the clean-label head start with a grid-and-tuning difference. Standalone, both arms take the disk path and the settings apply. `noise_helpers.py:114-137` records the same thing. Also: xgboost and svm produce no cell on the noisy arm (n15:314-317), so 21 cells run rather than 35; the rebuild is handed the whole 10-representation pool at n15:346 rather than the recipe's own representations, which makes `rep_jaccard` and `reps_identical` describe a field neither score depends on; `--models` does nothing; and `EXPERIMENTS_FINALIZED.md:2608` and `:392` are both stale against the code.

### n16 — importance under noise

As the labels get noisier, does the ordering of features by importance change, both for each base representation scored whole and for the features the recipe actually selects?

- **Hybrids** — precomputed. One read at n16:544-550 for rf at `budget=None`, with no speed, no tune, no models, no objective and no pool subset. Tier `general`, objective predictive, budgets the 8 entries of `FULL_BUDGETS`, evaluators at their defaults. A precompute exists for exactly that combination (`noise_helpers.py:817-829`), so under the suite it is a stored-row read. On a dataset whose precompute has not run, the same call cold-builds 8,160 configurations inside the experiment, and `KIRBY_NO_COLD_SEARCH` would refuse it. Features are rebuilt by `build_fold_hybrid` at n16:745 with no precomputed importance and no precomputed allocation scores, so every call refits both.
- **Compute** — base arm = 3 noise replicates x 5 folds x 3 scoring methods x 10 representations x 6 noise levels = 2,700 importance fits (n16:697-705). Hybrid arm builds = 3 replicates x 5 folds x 6 levels = 90 `build_fold_hybrid` calls (n16:729-731), each costing 1 importance fit and 1 allocation fit per representation the recipe names, plus an augmentation ladder when the recipe's strategy is capped or uncapped. Hybrid ranking rows = 90 builds x 3 scoring methods = 270 importance fits (n16:792-795). **Headline: ~3,000 estimator fits per dataset that can be pinned down**, plus roughly 540 fits inside the 90 rebuilds at a 3-representation recipe and an unverified augmentation-ladder term. None is a downstream evaluation fit. Not excessive.
- **Hybrids it uses / should use** — exactly 1, held fixed at every noise level and replicate by design (n16:542-560, reasons at n16:44-53). It is the single best configuration for rf across all 8 budget entries. As used: n16 computes no distance between molecules and no neighbours; it correlates two feature-importance rankings against each other.
- **Selection pattern** — different. It compares a feature-importance ranking at one noise level against the same cell's ranking at level 0.0, per representation and per scoring method: 1 model name in the file, 10 base representations scored for their importance vectors rather than their accuracy, 1 hybrid, 3 scoring methods, 6 noise levels, 3 replicates, 5 folds.
- **Cut: fast tier** — no search, and `speed='fast'` at n16:544 would make it worse. Every list `speed` derives is in the search cache key (`find_best_hybrid.py:673`), so a fast search would miss the full-grid precompute already in memory and cold-build 432 configurations where the stored one is free.
- **Cut: two evaluators** — there is no downstream evaluator list; the only model name is rf. The equivalent axis is `imp_methods` at n16:470, 3 scoring methods (n16:145), already overridable at n16:1129. Cutting to `lightgbm_gain` alone takes the base arm from 2,700 to 900 importance fits and the ranking rows from 270 to 90, and removes the dear ones: the f30 bake-off at `find_best_hybrid.py:3396-3403` records deeplift at 271 s, treeshap at 542 s and lightgbm_gain at 42 s. Which scorer's ranking moves is the axis this experiment exists to measure, so that cut needs a stated reason rather than a compute argument. The free cut is passing the recipe's stored per-fold importance and allocation scores into `build_fold_hybrid` at n16:745, the way n15 does at n15:361-380.
- **Notes** — treeshap runs in this environment: `compute_feature_importance` through the repo path returns finite vectors for lightgbm_gain, deeplift and treeshap, so the 3-method grid is really 3 methods locally and none of the fit counts drop. n16's own docstring at n16:320-334 still records the old broken state. n16 has no entry in `EXPERIMENTS_FINALIZED.md`. The hybrid arm passes the same dictionary as both the train and the val representations at n16:745-752, so `create_hybrid`'s admission holdout is carved out of the training rows alone — deliberate, and n16:670-691 subsets the pool to the recipe's own representations for the same reason.

### n17 — replicate removal

When a molecule was measured several times in different laboratories and the numbers disagree, does throwing out or rewriting the labels a detector flags beat leaving them alone?

- **Hybrids** — precomputed. One read at `tests/experiments/noise/n17_replicate_removal.py:470` for `downstream_models[0]` at `budget=None`, `pool_subset='predictive_and_similarity'` (n17:177). Features are rebuilt by `hybrid_for_fold` at n17:594, which reaches `create_hybrid` at `_shared.py:4772` at most 15 times. No `find_best_hybrid` call. The pool is 2 members without donor representations and 6 with them, tier `general` throughout. The predictive members are served from the in-memory store under `noise_helpers.run_all`; the similarity members always fall through to `find_best_hybrid(cache=True)` on disk at `_shared.py:5352`. That disk read misses the file the precompute wrote: `run_all` defaults to `default_pruning` (`noise_helpers.py:690`), whose 5 pruning keys are written into the stored meta (`find_best_hybrid.py:747-766`), while the reader passes `adaptive_pruning=False` (`_shared.py:5346`) and `_load_cache_with_meta` rejects an opt-in key the caller did not ask for (`find_best_hybrid.py:432-441`). The per-dataset SQLite result database carries no pruning field in its identity, so the rebuild re-reads the stored configuration rows rather than refitting them.
- **Compute** — 3 draws (n17:165) x 5 folds (n17:361) x 11 spaces (10 representations plus 1 hybrid) x 6 arms (n17:160) x 4 downstream models = 3,960 downstream model fits on bindingdb_kd, where svm is dropped at 30,000 rows (`find_best_hybrid.py:2597`). At 5 models it is 4,950 fits, which is the figure the file's own note at n17:101-105 states. Separately, at most 3 draws x 5 folds = 15 hybrid builds, each up to 20 model fits and 0 for the allocation half under an allocation method that does not score representations — up to 300 fits. The 165 detector runs are nearest-neighbour graphs and cost 0 model fits. **Headline: ~4,300 model fits per dataset** on bindingdb_kd. Not excessive; the exposure is the standalone cold-search case.
- **Hybrids it uses / should use** — exactly 1 per run: the best stored configuration for rf across all 8 budget entries (`_shared.py:6500-6505`), rebuilt per fold and scored under every surviving downstream model. This is a neighbour experiment: the detector is `neighbor_residual` at 10 neighbours (n17:142-143), scoring a molecule by how far its label sits from its nearest neighbours' labels, and the repair arm overwrites a flagged label with its 10 neighbours' mean (n17:262, used at n17:294-295). So hybrids whose features were chosen so each molecule's activity is predictable from its five nearest neighbours belong here, and small feature budgets with them. n17:177 admits the similarity family, but the two families are scored in different units, so the ranking at `_shared.py:5757-5772` falls to subset order and the predictive family wins whenever a predictive search exists. With only one family present the similarity recipe is returned (`_shared.py:5745-5748`). Otherwise the similarity hybrid is recorded only as a score at n17:506-512. The budget choice is `budget=None` at n17:472.
- **Selection pattern** — partial. 5 downstream models, 10 base representations taken whole and unranked (n17:389-390), 1 hybrid: 55 (model, representation) pairs against the intended 20. There is no top-2 step of any kind.
- **Cut: fast tier** — available, one argument at n17:470. `get_best_search_for_downstream`'s kwargs become `extra` (`_shared.py:5628`), are forwarded at `_shared.py:5488`, and reach `find_best_hybrid` at `_shared.py:5347-5350`. `speed='fast'` cuts the grid from 8,160 configurations to 432. Those lists are in the search cache key, so a fast call files separately and cannot read the stored 8,160-configuration precompute.
- **Cut: two evaluators** — `tests/experiments/noise/n17_replicate_removal.py:374` sets the model list, currently 5 names and already 4 on bindingdb_kd and herg_ic50. Cutting to 2 takes the headline from 3,960 to 1,980 downstream model fits per dataset; the arm build sits outside the model loop, so nothing else changes. It cannot be done from the command line today: this file reads only `config['n_splits']` (n17:361) and neither `main()` nor `noise_helpers.run_all` fills the models kwarg, so the cut needs one line of wiring at n17:374.
- **Notes** — two sides disagree on augmentation. The recipe can carry an `uncapped` or `capped` condition (recorded at n17:503) and bindingdb_kd has 3 donor representations, but the call at n17:594 passes no train or val augmentations. `hybrid_for_fold`'s docstring at `noise_helpers.py:581-584` says that rebuilds a narrower feature space than the search chose. n21 is the only caller that passes them. Also: n17 sizes its model list on `len(labels)` (n17:376) while n19 sizes it on the molecule count (n19:455), so on herg_ic50 at 12,008 measurement rows over 9,802 molecules svm is dropped in n17 and kept in n19. herg_ic50 has no `DATASET_REPS` entry, so `run_all` raises at `noise_helpers.py:745` and n17 reaches it only through the standalone CLI.

### n18 — cleaning the real labels

On the labels exactly as the loader hands them over, with nothing corrupted on purpose, does cleaning up the training labels make the held-out score better on more folds than worse?

- **Hybrids** — precomputed. One read at `tests/experiments/noise/n18_clean_the_real_labels.py:259` for `downstream_models[0]` at `budget=None`, `pool_subset='predictive_and_similarity'` (n18:137). Features are rebuilt by `hybrid_for_fold` at n18:331, reaching `create_hybrid` 5 times per dataset. No `find_best_hybrid` call. Same pool mechanism as n17: tier `general`, 2 objective families times the augmentation conditions, 8 budget entries. The similarity member always goes to disk and its pickle load is rejected on the pruning meta keys, so the search rebuilds from the SQLite result rows.
- **Compute** — 5 folds (n18:181) x 2 spaces (n18:129) x [5 baseline fits + 2 detectors x 3 flag rates x 5 downstream models x 4 cleanups] = 5 x 2 x 125 = 1,250 downstream model fits. Each `_evaluate_fold` call is exactly 1 fit. Separately, 5 hybrid builds at up to 20 fits each = up to 100 fits, conditional on the recipe's allocation and importance methods. The 20 detector runs cost 0 model fits. **Headline: ~1,350 model fits per dataset**, 1,250 of them downstream, on herg_ki at 1,412 molecules where all 5 models are kept. Not excessive per dataset, but see Notes on dataset size.
- **Hybrids it uses / should use** — exactly 1 hybrid and 1 base representation. The hybrid is the pool's best configuration for rf across all 8 budget entries, rebuilt per fold and shared by every cleanup, flag rate and detector (n18:331-344). Neighbour-based on both sides: the 2 detectors are `neighbor_residual` and `activity_cliff_aware` at 10 neighbours (n18:124-125), and one of the 4 cleanups is `knn_smooth` (n18:110), which overwrites a label with its neighbours'. The file's own header at n18:133-136 says 3 of the 4 cleanups use a molecule's neighbours' labels, which is the quantity the second objective optimises. So similarity-objective hybrids and small feature budgets belong here. n18:137 admits the similarity family, but the predictive family wins by subset order whenever a predictive candidate exists; with only one family present the similarity recipe is returned. `budget=None` at n18:261 asks for no small budget.
- **Selection pattern** — partial. 5 downstream models, 1 base representation, 1 hybrid = 10 (model, representation) pairs against the intended 20. Neither representation is picked by accuracy: the base one is `sorted(reps)[0]`, the alphabetically first name (n18:199), which the file records rather than chooses (n18:55-58). The folds are spent on the 2 detectors x 3 flag rates x 4 cleanups grid.
- **Cut: fast tier** — available, one argument at n18:259, by the same kwargs chain as n17. `speed='fast'` cuts the grid from 8,160 configurations to 432. Those lists are the search cache key, so a fast call files separately from the stored full precompute.
- **Cut: two evaluators** — `tests/experiments/noise/n18_clean_the_real_labels.py:184` sets the 5-name model list. Cutting to 2 takes each (fold, space) from 125 to 50 fits, so the headline drops from 1,250 to 500 downstream model fits per dataset. Same wiring gap as n17: this file reads only `config['n_splits']` and no caller passes a models kwarg. A second axis: `DETECTORS` at n18:124 holds 2 names, and the file's note at n18:120-123 records that on herg_ki in ecfp4 over 1,412 molecules `activity_cliff_aware` ranks the molecules the same way as `neighbor_residual` at Spearman 0.989 — dropping it halves the 120 fits per (fold, space) to 60.
- **Notes** — n18 is a neighbour experiment and is absent from `NUCLEOTIDE_SKIP_EXPERIMENTS` (`_shared.py:3336-3339`), so it runs its full 1,250 fits on the nucleotide sets listed at `submit_all_datasets.sh:57`, which hold 18,244 to 100,000 rows. Same augmentation disagreement as n17: the recipe can carry an `uncapped` or `capped` condition (n18:291) and n18:331 passes no augmentations. `--detectors` is not a command-line flag; `run()` reads `kwargs['detectors']` at n18:195 but `main()` adds only `--flag-rates` and `--base-rep`. qm9 is run deliberately as the control (n18:237), because its labels are calculated and carry no laboratory error. A failed fold is swallowed and lands as a NaN (n18:386-387). n18 has no entry in `EXPERIMENTS_FINALIZED.md`.

### n19 — measurement count

Is it worth telling the model how many times each molecule was measured and how far those measurements disagreed, as a per-molecule fit weight or by pulling an imprecise label toward what the other molecules predict?

- **Hybrids** — precomputed. One read at `tests/experiments/noise/n19_measurement_count.py:514` for `downstream_models[0]` at `budget=None`, `pool_subset='predictive_and_similarity'` (n19:164). Features are rebuilt by `hybrid_for_fold` at n19:649, reaching `create_hybrid` at most 30 times. No `find_best_hybrid` call. Pool: tier `general` under both objectives, times the augmentation conditions — 6 members on bindingdb_kd, 2 without donor representations. The predictive members hit the in-memory store under `run_all`; the similarity members always go to disk and their pickle is rejected on the pruning meta keys, so the search rebuilds from the SQLite result rows.
- **Compute** — 3 draws (n19:154) x 2 label builds (n19:133) x 5 folds (n19:425) x 2 spaces (n19:149) x [1 out-of-fold prediction + 6 conditions x 5 downstream models] = 1,800 downstream model fits plus up to 300 LightGBM out-of-fold fits on herg_ic50 at 9,802 molecules. On bindingdb_kd svm is dropped at 26,531 pairs, giving 1,440 downstream fits plus the same 300. Separately, at most 30 hybrid builds at up to 20 fits each = up to 600 fits. **Headline: ~2,700 model fits per dataset on herg_ic50** and ~2,340 on bindingdb_kd. Not excessive. The 300 out-of-fold fits are an upper bound: `oof_predict` fits one model per split from `_safe_n_splits` (`core.py:149-156`), so a fold whose training molecules span fewer than 5 scaffold groups fits fewer.
- **Hybrids it uses / should use** — exactly 1 hybrid and 1 base representation. The hybrid is the pool's best for rf across all 8 budget entries, rebuilt once per (draw, label build, fold) on the untouched training labels, shared by all 6 conditions. As used: n19 computes no distances between molecules and uses no nearest neighbours. Its weights come from measurement counts and measurement variances (n19:703-705), and the only other thing fitted in the loop is a LightGBM out-of-fold prediction (n19:680). `POOL_SUBSET` at n19:164 still names the similarity family, which loses to the predictive one by subset order whenever a predictive candidate exists, and is otherwise recorded as a score at n19:548-554.
- **Selection pattern** — partial. 5 downstream models, 1 base representation, 1 hybrid = 10 (model, representation) pairs against the intended 20, and neither representation is picked by accuracy: the base one is `sorted(reps)[0]` at n19:563. The folds are spent on 2 label builds x 6 conditions.
- **Cut: fast tier** — available, one argument at n19:514, by the same kwargs chain as n17 and n18. `speed='fast'` takes the grid from 8,160 configurations to 432. Those lists ride in the search cache key, so a fast call files separately from the stored full precompute.
- **Cut: two evaluators** — `tests/experiments/noise/n19_measurement_count.py:453` sets the 5-name model list, already 4 on bindingdb_kd. Cutting to 2 takes the downstream term from 1,800 to 720 fits on herg_ic50; the up-to-300 out-of-fold fits do not move, because `oof_predict` runs once per (draw, build, fold, space) outside the condition loop. The headline goes from 2,100 to 1,020 model fits. Same wiring gap as n17 and n18: this file reads only `config['n_splits']` and no caller passes a models kwarg.
- **Notes** — same augmentation disagreement as n17 and n18: the recipe can carry an `uncapped` or `capped` condition (n19:545) and the call at n19:649 passes no augmentations; bindingdb_kd is where it bites. herg_ic50 has no `DATASET_REPS` entry, so `run_all` raises at `noise_helpers.py:745` and n19 reaches it only standalone, where a miss on either pool member builds a whole 8,160-configuration search inside the experiment. The downstream axis is not a clean product in general: a model that raises TypeError on `sample_weight` is recorded as refusing and skipped for the 3 weighted conditions (n19:708, n19:713-714). None of the 5 default models does. n19 has no entry in `EXPERIMENTS_FINALIZED.md`.

### n20 — clean before search

When you throw out or rewrite training labels that look wrong, does it matter whether you do that before or after the recipe search picks the features?

- **Hybrids** — its own, and nothing can precompute them. `tests/experiments/noise/n20_clean_before_search.py:263` calls `find_best_hybrid` from `_run_one_search`, reached at n20:520 (the labels as they are) and n20:600 (the cleaned labels). It passes `tier='general'`, which `find_best_hybrid.py:8407-8409` rewrites to `general__trainlabels__lb<8 hex>` because `train_labels=` is passed, hashing the actual label column. No objective, so predictive. `speed='fast'`, `tune=True`, `tune_n_iter=20`, `tune_knn_k=True` and an aggressive pruning schedule, all from n20:197. Budgets are the 8 entries of `FULL_BUDGETS` and inner folds are 3 (n20:329). No two of the searches share a tier with each other or with anything `precompute_hybrids.py` writes, and `cache_namespace` is unique per cell as well (n20:519, n20:598). The cold-search guard never fires, because the rewritten tier is not a precompute tier.
- **Compute** — configurations per search before pruning = 3 allocation methods x 1 importance method x 2 representation counts x 8 budgets x 17 (model, evaluator) pairs = 816, scored over 3 inner folds. The 17 pairs come from the 5 model names n20 passes explicitly; `speed='fast'` does not cut that list, because the model filter sits inside `if models is None` (`find_best_hybrid.py:8221-8232`). Tuning turns 1 downstream fit into 61 fits for rf, lightgbm and xgboost, 25 for svm, 1 for dnn. Search fits = 102,672, tuned baselines 6,270, importance pre-pass 30, allocation pre-pass 150 = 109,122 model fits per search. Searches = 13 per (fold, pass) x 17 passes x 5 outer folds = 1,105. **Headline: ~121,000,000 model fits per dataset as an unpruned upper bound** for herg_ki at 1,412 molecules and 5 outer folds. The aggressive pruning schedule and the feature-selection dedup at `find_best_hybrid.py:4902-4915` both cut it and neither is measured here; on the schedule's own caco2 calibration of 58 fewer evaluations in every 100, the real figure would be nearer 50 million fits. The 10-representation pool size is assumed, not read. Excessive by a wide margin, and n20 prints its own upper bound of 1,360 searches at n20:435-437.
- **Hybrids it uses / should use** — 2 per (model, cleanup, flag rate) cell, both from n20's own searches: the recipe from the search on the labels as they are (n20:537) and the recipe from the search on the cleaned labels (n20:667). `budget=None` at both, so each is the winner across all 8 budget entries. Over one (fold, pass) that is 65 rebuilds. The detector is neighbour-based — `ActivityCliffAwareEstimator` at 10 neighbours, reached from n20:502 — and `knn_smooth` moves each label toward its neighbours'. But the neighbours are found in the base representation only (n20:501-502), never in a hybrid, and the answer is a count of folds won on the downstream metric. n20 cannot reach a similarity-objective hybrid at all, because it builds every recipe itself and passes no objective and no pool subset. Adding a second objective argument at n20:274 would double the 1,105 searches.
- **Selection pattern** — different. There is no base-representation arm: all 4 arms are hybrids, and the comparison is 2 hybrids (searched on the uncleaned labels against the cleaned ones) crossed with 2 label columns to fit on, over 5 downstream models, 5 cleanups and 3 flag rates.
- **Cut: fast tier** — `speed='fast'` is already set at n20:197. The remaining fast-tier cut is the model list at n20:334, handed to the search at n20:269, which defeats the fast model filter. Letting it pass `models=None`, or naming rf, lightgbm and dnn, takes the pairs from 17 to 9 and the grid from 816 to 432 configurations: per-search fits from 109,122 to ~56,950, and the dataset from ~121 million to ~63 million unpruned fits. The larger lever is `tune=True` at n20:197: untuned, one search is about 2,600 fits and the dataset about 2.9 million.
- **Cut: two evaluators** — n20:334 is the one place, because `downstream_models` is both the search's model list (n20:269) and the list every arm is scored on (n20:530, n20:628). Cutting 5 names to 2 takes the pairs from 17 to 6 and the grid from 816 to 288 configurations; per-search fits fall from 109,122 to ~28,700, and the dataset total from ~121 million to ~32 million unpruned fits. Arm fits and rebuilds drop from 5 per cell to 2.
- **Notes** — 4 of the 13 searches per (fold, pass) re-search a label column already searched. `no_cleaning`'s cleaned column is the original column (n20:553-556) and it is searched once per flag rate, giving 3 identical searches; the docstring at n20:100-105 says `label_smooth` on a regression dataset produces the same column at all 3 flag rates, giving 3 more. Because the namespace at n20:598 carries the flag rate, the duplicates cannot hit each other's cache. Keying the cleaned search on the label column rather than the flag rate would take 13 searches per (fold, pass) to 9 — from 1,105 searches to 765 — with no arm lost. Also: n20:346 picks the detector's space with `sorted(reps)[0]`, which is the alphabetical fallback `pick_noise_base_rep` was written to refuse, while n21 in the same suite calls `pick_noise_base_rep` at n21:602. And n20 uses the detector n21:112-115 records as cut on 2026-09-12.

### n21 — neighbour distance

When a detector decides a label looks wrong by comparing it to the molecule's nearest neighbours, which space should those neighbours be found in?

- **Hybrids** — precomputed. One read at `tests/experiments/noise/n21_neighbour_distance.py:614` for `downstream_models[0]` at `budget=None`, `pool_subset='predictive_and_similarity'` (n21:619), outside every loop. Features are rebuilt by `hybrid_for_fold` at n21:793, one uncached `create_hybrid` per (fold, draw). No `find_best_hybrid` call; `build_fold_hybrid` is imported at n21:105 and never used. The pool asks for 2 objectives times 1 or 3 augmentation conditions, tier `general`, at the default full speed. Both lineages are precomputed by `noise_helpers.py:818-828` and `:846-858`. The similarity members are handed `precomputed=None` at `_shared.py:5477` and always resolve from disk, where the stored meta carries 5 pruning keys the reader does not ask for, so the pickle is rejected (`find_best_hybrid.py:435-441`) and the search re-enters `find_best_hybrid`.
- **Compute** — passes = 5 noise levels x 5 outer folds x 3 corruption draws = 75 (n21:583). Spaces = 13: 2 plain distances, 8 importance-weighted, 1 forest proximity and 2 hybrid (n21:372-420). Detectors = 2 (n21:116), neighbour counts = 3 (n21:119), models = 5. Repair fits = 75 x 13 x 2 x 3 x 5 = 29,250 model fits (n21:851-859). On top: 125 clean reference fits, 375 corrupted reference fits, 75 forests of 100 trees, 150 importance fits, and 75 `create_hybrid` rebuilds at roughly 20 fits each — about 1,500 fits, an estimate not traced against `create_hybrid`'s loops. The detection reading costs no fit: it is a nearest-neighbour lookup and an average-precision score (n21:824-835). **Headline: ~31,500 model fits per dataset**, of which 29,250 are the repair arm. Not excessive. The pruning-meta mismatch costs a rejected pickle and a re-walk of the configuration list, not fresh fits, because the result database's identity carries no pruning field and serves the precompute's stored rows back (`find_best_hybrid.py:8612-8646`, `:7108-7128`).
- **Hybrids it uses / should use** — exactly 1 recipe, read once for rf at `budget=None` and rebuilt per pass under the corrupted labels. It becomes 2 of the 13 spaces: the hybrid feature space under a euclidean distance and under a manhattan one. The recipe was chosen on the clean labels of every molecule including each fold's held-out ones; the file records that at n21:609 and calls the hybrid distance a best case rather than a leak-free comparison at n21:638-645. This is the most neighbour-dependent experiment here — all 13 spaces exist only to define which molecules are near which — and it already admits similarity-objective hybrids at n21:619. Two things stop that from landing. The pool does not rank the two families against each other: their scores are in different units, so the winner is the first family the subset names (`_shared.py:5762-5771`), and predictive is first. And `budget=None` at n21:616 takes the winner across all 8 budget entries, so nothing puts weight on a small number of features — a distance over 2,000 features is a different object from a distance over 25. The change worth considering is at n21:614-619: read more than one recipe, force `pool_subset='similarity_only'` for at least one, and name a small budget, so the hybrid family enters as 2 or 3 spaces built on different numbers of features.
- **Selection pattern** — different. It fixes 1 base representation and 1 hybrid recipe and sweeps 13 distances x 2 detectors x 3 neighbour counts, scoring each by how well it finds the molecules whose labels really moved; the 5 models are recipients of a training weight, not a selection axis.
- **Cut: fast tier** — nothing to cut: no search. The only place speed could matter is the pool's disk fallback at `_shared.py:5330-5352`, which n21 does not control. If that fallback is firing, the change is to make the reader's pruning meta match the writer's; on the evidence the fallthrough is served from the result database anyway, so forcing a fast tier would change which configurations are asked for without saving fits.
- **Cut: two evaluators** — `tests/experiments/noise/n21_neighbour_distance.py:592` sets the 5-name model list. It multiplies the repair arm only. Cutting to 2 takes 29,250 repair fits to 11,700, and the dataset from ~31,500 to ~13,900 model fits. It changes nothing about the headline, because the detection reading costs no fit and does not read the model list. A larger cut sits at n21:560 and n21:566: the 3 neighbour counts and the 2 detectors multiply the repair arm the same way, and the file's note at n21:112-115 says the 2 detectors rank molecules at Spearman 0.989 on herg_ki.
- **Notes** — all 29,250 repair fits are on the same matrix: n21:854-856 fits each model on one standardised base representation with only the sample weights differing, which n21:707-711 states on purpose, so that the only thing separating two spaces is the distance. n21:614 passes `downstream_models[0]` with no comment on why the first model in the list should choose the hybrid for all 5; that is rf, by list position. The experiment refuses to run under 6 of the 11 regression noise conditions, with its reason at n21:552-559: average precision is 1.0 by construction when every training label moves.

### n22 — curator flag

Do the noise detectors rank as suspect the same hERG IC50 records that ChEMBL's own curators marked suspect, and does building a hybrid feature space change that?

- **Hybrids** — precomputed in intent, cold in practice. `tests/experiments/noise/n22_curator_flag.py:401` calls `_blueprint_for`, which reaches `get_best_search_for_downstream` at `n06_bindingdb_ground_truth.py:311` for rf with `pool_subset='predictive_and_similarity'` unless `--no-similarity-arm` (n22:691). Features are rebuilt through `n06:345` into `create_hybrid` at `_shared.py:4772`, once per (hybrid space, fold). No `find_best_hybrid` call in the file. Tier is `general` for the predictive member and `general__obj-similarity` for the similarity one (`find_best_hybrid.py:8473`); both roots are covered by the cold-search guard (`find_best_hybrid.py:558-565`). No precompute exists for this dataset: n22 runs on herg_ic50 only (n22:137), `tests/results/precompute_registry.json` holds 47 datasets whose only hERG key is herg_ki, and herg_ic50 has no `DATASET_REPS` entry. So both pool members build live at `_shared.py:5352` on the first run, with `cache=True`, which writes a validated search pickle a second run reads back.
- **Compute** — the experiment's own work: 2 hybrid spaces x 5 folds (n22:126, hardcoded) x 1 rebuild at about 6 model fits = ~60 fits, where the per-build figure of 6 is an estimate from the recipe's importance and allocation axes. Then 3 feature spaces x 5 folds x 130 detector-internal fits = 1,950 fits, the 130 summing ensemble 50, deep_ensemble 25, quantile_regression 15 and 5 each for 8 others. The cold search on top, first run only: the model list drops svm at 12,008 records, leaving 4 models and 13 (model, evaluator) pairs, and the 3-representation pool collapses the representation counts to 2 entries. Predictive member = 5 allocation x 3 importance x 2 representation counts x 8 budgets x 13 pairs = 3,120 configurations x 5 folds = 15,600 model fits. Similarity member = the same 3,120 configurations, of which 1,560 raise on every fold (`find_best_hybrid.py:4672-4680`) and 1,560 x 5 folds = 7,800 hybrid builds run without fitting any downstream model. **Headline: ~2,010 model fits of its own per run on herg_ic50 at 12,008 records, and ~15,600 more model fits plus ~7,800 no-model hybrid builds on the first run.** The importance pre-pass, allocation pre-pass and baseline fits are unverified and not counted. Excessive only because the precompute is missing.
- **Hybrids it uses / should use** — up to 2 hybrid recipes plus 1 single representation, so at most 3 feature spaces. One recipe is the pool winner; the others come from `meta['by_objective']` (n22:418-427), one per eligible objective family. Each is the single best entry its family had for rf at `budget=None`. The single representation is `extract_top3_base_reps(search)[0]` at n22:456-462, the top one ranked by rf baselines, falling back to `pick_noise_base_rep` at n22:467. This rests on distances: 4 of its 14 detectors score a record by how far its label sits from its nearest neighbours' labels. Similarity-objective hybrids are already admitted by default (n06:112). Small feature budgets are not: `budget` is always None, because n22:700 reads `--budget` and the parser defines `--budgets`, plural, at `_shared.py:7176`. Pinning 10, 25 or 50 features needs a real flag at n22:690-694 threaded to n22:389.
- **Selection pattern** — different. It fits no downstream prediction model at all: it scores each record's suspiciousness with 14 detectors in 3 feature spaces over 5 folds and grades those rankings against ChEMBL's flagged records.
- **Cut: fast tier** — nothing to cut at this file's lines: n22 makes no `find_best_hybrid` call and `get_best_search_for_downstream` takes no speed argument (`_shared.py:5509-5531`; its docstring at `:5573` still lists one and is stale). The cut has to be made one level up: give herg_ic50 a `DATASET_REPS` entry and a precompute pass, or set `KIRBY_NO_COLD_SEARCH=1` so the fall-through refuses instead of building 3,120 configurations live. A forced fast search on the predictive member would go from 3,120 configurations to 3 allocation x 1 importance x 1 representation count x 8 budgets x 9 pairs = 216, so 15,600 fold-evaluations down to 1,080. `FAST_INTERNAL_EVALUATORS` is the same 4 names as the default, so fast cuts the model list, not the evaluator list.
- **Cut: two evaluators** — the analogous list is the 14 detectors resolved at n22:380, and it cannot be cut by top-2 accuracy, because that ranking is n22's own output. The cut with a number on it is dropping the 3 dearest: ensemble, deep_ensemble and quantile_regression take 130 fits per (space, fold) to 40, so 3 spaces x 5 folds goes from 1,950 to 600 model fits, leaving 9 detectors. There is no downstream evaluator list in this file.
- **Notes** — the documented suite command at n22:86 does not run: `run_dataset.py:192` calls `run_all` with no rep names, and `noise_helpers.py:739-746` raises `ValueError: No rep list for 'herg_ic50'`. Run standalone with the parser default `--reps fast`, n22 gets 3 representations — ecfp4, maccs, pdv — while every other noise experiment runs on a 10-representation pool, so n22's hybrid is not comparable to any other file's. `config` is never read in `run()` and `N_FOLDS = 5` is hardcoded at n22:126, so `--mode intermediary` does not reduce this experiment's folds. Each detector is handed the whole fold (n22:525-526), so all 210 estimate calls run over all 12,008 records, except `gaussian_process`, subsampled to 1,000 rows. n22 is not in `EXPERIMENTS_FINALIZED.md`.

### n23 — uncertainty cleaning

Does dropping or shifting the training labels a model is least certain about make its held-out predictions better than leaving the labels alone?

- **Hybrids** — precomputed. One read at `tests/experiments/noise/n23_uncertainty_cleaning.py:498-504` for `downstream_models[0]` at `budget=None`, `pool_subset='predictive_and_similarity'` (n23:219). Features are rebuilt by `hybrid_for_fold` at n23:599-602, reaching `create_hybrid` once per (pass, fold). No `find_best_hybrid` call. Tier `general` for the predictive member and `general__obj-similarity` for the similarity one, budget None, no axis overrides, no speed argument. A precompute exists for both on herg_ki: `run_precompute_dataset.sh` covers the predictive one and `noise_helpers.py:846` runs the similarity one for every noise dataset. Standalone, `load_data` at n23:898 is called with no augment argument, so the pool asks for the `none` condition alone — 2 members in all.
- **Compute** — passes = 17: 1 as-they-come (n23:527) plus 16 injected, from 6 noise levels where level 0.0 on a regression set takes 1 of the 3 draws and the other 5 levels take 3 each (n23:347-364). Then 5 folds (n23:378) x 2 spaces (n23:517-520) x 5 uncertainty sources (n23:171-172) x 10 cleaning arms (n23:326-334) x 5 downstream models = 42,500 downstream model fits. Uncertainty estimates = 850 calls, of which 4 sources in 5 fit over 5 inner folds = 3,400 model fits. Hybrid rebuilds = 17 passes x 5 folds = 85 `create_hybrid` calls; `hybrid_for_fold` cuts the pool to the recipe's own representations first (`noise_helpers.py:589-596`), so at most 1,700 model fits is a ceiling rather than the figure. **Headline: ~46,000 to ~47,600 model fits per dataset at `--mode full`**, or ~9,600 at the parser's default `--mode intermediary`, which uses 1 fold. Excessive for what it answers: the nested loop alone is about the size of one whole full hybrid search.
- **Hybrids it uses / should use** — exactly 1 recipe, fetched once before the loops and rebuilt per fold: the single best pool entry for rf at `budget=None`, drawn from the prediction-built and distance-built families together. Every one of the 17 passes, 5 folds, 5 sources and 10 arms shares it. The base representation is the alphabetically first name in the pool (n23:404), atom_pair on herg_ki, with no accuracy ranking. Partly neighbour-based: `activity_cliff_aware` scores each molecule against its 10 nearest neighbours' labels, and when a source carries no denoised value the move target is the mean of its 10 nearest neighbours' labels (n23:283-295). The other 4 sources predict a spread from the features. Distance-built hybrids are eligible through n23:219, but only the single overall pool winner is used — the similarity recipe is never run as its own arm the way n22 runs it at n22:418-434. Small feature budgets are never asked for.
- **Selection pattern** — different. It compares 10 cleaning arms against no cleaning, paired per fold, in 2 feature spaces across 5 uncertainty sources and all 5 downstream models at every noise level, so the axis it sweeps is cleaning settings, not representations.
- **Cut: fast tier** — nothing to cut: no search. n23's only pool contact is the read at n23:498, which takes no speed argument, and the compute is entirely in its own loops.
- **Cut: two evaluators** — `tests/experiments/noise/n23_uncertainty_cleaning.py:394` sets the 5-name model list. Cutting to 2 takes the downstream fits from 42,500 to 17,000, a saving of 25,500 model fits per dataset; the rebuilds and the 3,400 uncertainty estimates do not move. Three other levers carry numbers: cutting the noise draws at n23:214 from 3 to 1 takes the 17 passes to 7 and the downstream fits to 17,500; dropping the 2 move methods to 1 (n23:199) takes the 10 arms to 7 and the fits to 29,750; both together with 2 models give 8,400 fits. Note `--models` does not reach n23: it lands in `config['models']` and n23 reads `kwargs.get('models')`, so the cut needs the `run()` call changed or the flag wired at n23:394.
- **Notes** — by the file's own reckoning most of the injected grid is undefined. `DEFAULT_NOISE_CONDITION = 'outlier_p10'` (n23:213) is not in `NOISE_KEYED_TO_STRUCTURE` (n23:188), and `LEARNS_NOISE_FROM_FEATURES` holds 4 of the 5 sources (n23:178-179), so n23:469-473 writes False for 4 sources on every row of the default run. The 16 injected passes buy a defined answer for 1 source in 5. Running `--noise-condition grouped_wider` costs the same and makes all 5 defined. n23 is not wired into any runner — it is absent from `SINGLE_DATASET_EXPERIMENTS`, `noise_helpers`, `run_dataset.py` and every SLURM script — and it is not in `EXPERIMENTS_FINALIZED.md`, so nothing currently schedules these fits.

---

## OPIG

### opig_activity_cliff_qsar — ChEMBL activity cliffs

On three ChEMBL targets, which representation predicts activity best, and does a hybrid predict the activity jumps between near-identical molecule pairs that single representations miss?

- **Hybrids** — its own. Two call sites, both `get_best_search_for_downstream`: `tests/experiments/opig_activity_cliff_qsar.py:472` (all molecules) and `:326` (once per scaffold fold, on that fold's training rows). Tier string `general`, rewritten by the engine to `general__cond-d2` and `general__cond-d2_heldout_fold0` and so on. Objective predictive, speed full, budgets all 8 entries from 10 to 2000 features. Nothing precomputes any of it: `chembl_qsar` is named in the exclusion list of every precompute driver, at `tests/slurm_scripts/submit_precompute_extra_passes.sh:157`, `submit_everything.sh:163` and `submit_all_datasets.sh:62-65`. The cold-search guard stays silent because the rewritten tier is not one of the two protected roots, so nothing warns.
- **Compute** — one search is 5 allocation methods x 3 importance methods x 4 representation counts x 8 budget entries x 17 (model, evaluator) pairs = 8,160 configurations. Each configuration is scored on 5 folds. Without augmentation that is 1 downstream fit per configuration-fold; with augmentation it is 9, being 1 downstream fit, 5 ladder scoring fits and 3 out-of-fold residual fits. Each target runs 18 searches: 1 whole-set search plus 5 per-fold searches, each expanded into the three augmentation conditions none, uncapped and capped. Per target that is about 4.14 million fits for d2 and 4.66 million each for fxa and mpro. **~13,500,000 model fits across the three targets.** That is excessive: 3 targets of 1,924 to 6,333 molecules each are paying a 13.5 million fit grid for a 40-row table.
- **Hybrids it uses / should use** — one recipe per (downstream model, fold), 25 fold-specific recipes per target, pooled into 5 hybrid rows. Each fold's search is chosen for random forest across all 8 budget entries. This experiment rests on neighbours: a matched molecular pair is two structurally near molecules, and the score is whether the predicted activity gap tracks the measured one. It never tests a similarity-objective hybrid and never steers toward a small feature count. Both are set in the same argument lists at `:472-486` and `:326-335`, which pass no `objective=`, no `pool_subset=` and `budget=None`. Adding `pool_subset='predictive_and_similarity'` there doubles the live searches from 18 to 36 per target, because no precompute exists to read.
- **Selection pattern** — different. It scores 5 downstream models x 7 representations = 35 rows per target plus 5 hybrid rows, 40 rows against the intended 20, with no top-2 cut anywhere in the file.
- **Cut: fast tier** — `tests/experiments/opig_activity_cliff_qsar.py:472-486` and `:326-335`, add `speed='fast'` to both argument lists. Per condition the grid goes from 8,160 configurations to 3 x 1 x 2 x 8 x 9 = 432, a factor of 18.9. Across all 18 searches per target that is roughly 4.66 million fits down to about 250,000. The evaluator list is passed explicitly at `:481`, so fast does not shorten it.
- **Cut: two evaluators** — `tests/experiments/opig_activity_cliff_qsar.py:466`, `_eval_list` is 4 names: match, rf, lightgbm, dnn. Cutting to `['match','rf']` takes (model, evaluator) pairs from 17 to 9 and the grid from 8,160 to 4,320 configurations per condition. A non-augmented search goes from 41,190 to 21,990 fits. The allocation pre-pass does not shrink, because `match` still pulls all five models' own evaluators in.
- **Notes** — svm is dropped from every uncapped search on d2: that target has 6,333 molecules against a 5,000-row cap at `find_best_hybrid.py:2827-2829`, so pairs fall from 17 to 13 there. `adaptive_pruning=0.25` at `:333` and `:484` changes no behaviour but does enter the cache key, splitting the cache. `EXPERIMENTS_FINALIZED.md:393` says this file calls one helper and does not call `build_fold_hybrid`; the code calls `build_fold_hybrid` at `:351` and the helper at two sites. Unverified: how far pruning cuts folds 2 to 4, and whether all 7 representations load for these targets.

### opig_blueprint_transfer — recipe transfer between datasets

Does a hybrid found on a large dataset still work on a smaller one, either its recipe re-run there or its exact selected feature positions copied over?

- **Hybrids** — its own. `find_best_hybrid` at `tests/experiments/opig_blueprint_transfer.py:188` (source dataset), `:201` (target dataset) and `:268` (once per held-out fold, called for 5 target folds at `:306` and 5 source folds at `:313`). Tier `general` at all three, objective predictive, speed full, budgets all 8 entries. Nothing precomputes it: the representation pool is the intersection of the two datasets' pools, 7 representations for logd to herg_ki, against the 10 each dataset was precomputed on. Representation names and widths sit in the search identity, so a 7-representation call cannot read a 10-representation file.
- **Compute** — one search is 5 allocation methods x 3 importance methods x 4 representation counts x 8 budget entries x 17 (model, evaluator) pairs = 8,160 configurations, scored on 5 folds at 1 fit each because augmentation is off. That is 40,800 search fits, plus 105 importance fits, 175 allocation fits and 175 baseline fits: 41,255 fits per search. Twelve searches per dataset pair gives **~495,000 model fits per source-to-target pair**, of which about 1,200 fits are the arms the experiment reports. Excessive: 99 percent of the fits are grid search for a 20-row table.
- **Hybrids it uses / should use** — as used. Four arms per downstream model: one base representation per (model, fold), one native recipe per fold from that fold's own search, one source recipe rebuilt on each target fold, and the source's exact selected feature positions applied unchanged. No distance between molecules and no nearest neighbours enter it, so similarity-objective hybrids answer a different question.
- **Selection pattern** — partial. Up to 5 downstream models, 1 base representation per (model, fold) rather than 2, and 3 hybrid arms rather than 2, giving 20 rows on the target from a different mix.
- **Cut: fast tier** — `tests/experiments/opig_blueprint_transfer.py:188`, `:201` and `:268`, add `speed='fast'` to all three calls; the third covers all ten per-fold searches. Per search the grid goes from 8,160 to 432 configurations and fits from 41,255 to 2,405. Across the twelve searches, 495,060 fits down to about 28,900.
- **Cut: two evaluators** — `tests/experiments/opig_blueprint_transfer.py:184`, `_eval_list` is 4 names, consumed at `:192`, `:206` and `:273`. Cutting to `['match','rf']` takes pairs from 17 to 9 and fits per search from 41,255 to 22,055, so 495,060 down to about 265,000 across the twelve searches.
- **Notes** — the run dies before any of this on the submit path that matters. `submit_everything.sh` exports `KIRBY_NO_COLD_SEARCH=1` at lines 288, 307 and 339 and launches this file at line 107. Lines 188 and 201 pass tier `general` with no condition suffix, so the cold-search guard counts zero stored rows for the 7-representation identity and raises, with no try/except around it. `submit_opig_experiments.sh:74-75` sets no such variable, so under that script every search builds live inside a 47:59:00 wall. `EXPERIMENTS_FINALIZED.md:5495-5496` lists two defects at `:346` that the code has already fixed. Unverified: molecule counts for logd and herg_ki, and how far pruning cuts folds 2 to 4.

### opig_cdr_flexibility — flexible versus rigid CDR3

Can sequence-based antibody representations tell a flexible CDR3 loop from a rigid one, and does a hybrid beat the best single representation?

- **Hybrids** — its own. One call site, `get_best_search_for_downstream` at `tests/experiments/opig_cdr_flexibility.py:288-306`, run once per CDR type. Tier `general`, rewritten to `general__cond-CDRH3` and `general__cond-CDRL3`. Objective predictive, speed full, budgets all 8 entries, 5 folds. No precompute is reachable, for two sufficient reasons. Tier is the first identity column, so the rewritten name matches no stored row. And the precompute loads this dataset with both loops at once and groups by 80 percent CDR3-loop identity clusters, while the experiment loads one loop and groups by IGHV germline family, so sample count, labels and groups all differ.
- **Compute** — one search is 5 allocation methods x 3 importance methods x 4 representation counts x 8 budget entries x 17 (model, evaluator) pairs = 8,160 configurations, scored on 5 folds at 1 fit each = 40,800 fits. Add 150 importance fits, 250 allocation fits, 250 baseline fits, and roughly 50 full-data refit fits from `return_fitted=True` (unverified, the refit was not counted inside the helper). The base-representation arm adds 10 representations x 5 models x 5 folds = 250 fits. Two CDR types gives **~83,500 model fits per run**. Each fit is on about 216 antibodies, so the wall is carried by deeplift and treeshap importance over wide language-model features, not by the downstream fits.
- **Hybrids it uses / should use** — one. The pool has exactly one member, so the search returns one result, and one winning recipe per downstream model is re-picked from stored per-fold scores. As used: the experiment classifies loop flexibility from whole-chain features and computes no distances between antibodies.
- **Selection pattern** — different. It scores 10 representations x 5 models x 5 folds and sets exactly 1 hybrid recipe per model beside the single best representation per model.
- **Cut: fast tier** — `tests/experiments/opig_cdr_flexibility.py:288-306`, add `speed='fast'`. Grid goes from 8,160 to 3 x 1 x 2 x 8 x 9 = 432 configurations, search fits from 40,800 to 2,160 per CDR type, the importance pre-pass from 150 to 50 fits, the allocation pre-pass from 250 to 150 fits and baselines from 250 to 150 fits. Note `:303-304` strips any `speed` key out of the smoke kwargs, so the edit has to be an explicit argument.
- **Cut: two evaluators** — two places. Inside the search, `:299-300` sets 4 evaluator names and no model list, giving 17 pairs; naming 2 models and 2 evaluators takes pairs to 4 and configurations from 8,160 to 1,920, saving 31,200 of the 40,800 search fits per CDR type. Outside the search, `MODELS_LIST` at `:91` is 5 models, used at `:355`; cutting it to 2 takes the base arm from 250 to 100 fits per CDR type.
- **Notes** — the baseline cache filename carries dataset, representation, model, metric, folds and seed but nothing about which loop, while the meta carries a labels hash. So the CDRH3 and CDRL3 runs write to the same baseline files and each one's labels make the other's a stale miss; those 250 baseline fits per CDR type are paid every run. The whole search sits in one try/except at `:282-352` that turns any failure into a flag, so a dead search leaves a results file with a full base-representation table and no hybrid. `EXPERIMENTS_FINALIZED.md:5503-5516` disagrees with the code on the representation count, the CDR types run and the exclude-folds argument; the code wins. Unverified: whether any `general` search for this dataset was ever written on disk.

### opig_cgr_comparison — condensed graph of reaction

Do reaction representations built from the condensed graph of reaction add anything that SMILES-based reaction representations do not already carry, for predicting reaction yield?

- **Hybrids** — its own. One call site, `get_best_search_for_downstream` at `tests/experiments/opig_cgr_comparison.py:358-369`, reached twice per dataset from `:645-647` and `:650-652`. Tier `general`, rewritten to `general__cond-smiles_only` and `general__cond-smiles_plus_cgr`. Objective predictive, speed full, budgets all 8 entries, 5 folds regardless of `--n-folds`. No precompute can serve these. Tier is an identity column, and the representation pool is this file's own: 20 hand-listed reaction generators plus 8 molecular variants across 3 roles, 44 representations against the 10-representation approved pools.
- **Compute** — per dataset, the individual-representation phase is 44 representations x 5 models x 5 folds = 1,100 fits. Each of the two searches is 8,160 configurations x 5 folds x 1 fit = 40,800 fits, plus an importance pre-pass of 615 or 660 fits and an allocation pre-pass of 1,025 or 1,100 fits; those are not shared between the two searches, because the pool hash differs. Baselines are 1,025 fits for the first search and 75 for the second, since the 41 shared representations hit the first search's files. Full-data refits add roughly 205 to 220 fits per search, unverified. Per dataset about 87,700 fits, and 3 datasets gives **~263,000 model fits per run**. Excessive for a two-number answer, and the individual-representation phase re-fits the same (representation, model) pairs the search already scores as baselines.
- **Hybrids it uses / should use** — two per dataset, six per run: one winner from the 41-representation pool and one from the 44-representation pool, each picked for random forest across all 8 budget entries. As used: the experiment predicts yield and measures representation-to-representation similarity with linear CKA on whole matrices, with no folds and no model. No distances between reactions and no nearest neighbours.
- **Selection pattern** — different. It is a pool-versus-pool test: all 44 representations against all 5 models, then one hybrid built from 41 representations against one built from 44.
- **Cut: fast tier** — `tests/experiments/opig_cgr_comparison.py:358-369`, add `speed='fast'`. One edit covers both pools and all three datasets. The six searches go from 244,800 search fits to 12,960, each importance pre-pass from 615-660 fits to 205-220, each allocation pre-pass from 1,025-1,100 to 615-660, and the first search's baselines from 1,025 to 615 fits.
- **Cut: two evaluators** — `tests/experiments/opig_cgr_comparison.py:358-369` passes neither evaluators nor models, so the engine defaults give 17 pairs. Naming 2 models and 2 evaluators takes pairs to 4 and configurations from 8,160 to 1,920, saving 31,200 of the 40,800 search fits per search and 187,200 across the six. Separately `MODELS_LIST` at `:101` is 5 models, read at `:605` and `:392`; cutting it to 2 takes the individual phase from 1,100 to 440 fits per dataset. Whether svm is one of the two kept decides most of the saving.
- **Notes** — a representation whose array has the wrong number of rows is skipped from the individual table at `:614-615` but is still inside the pool when the search runs at `:643`. `--n-folds` does not reach the search: it runs 5 folds while the printed table uses `min(--n-folds, number of groups)`. Each dataset is wrapped in its own try/except at `:895-903`, so one failure drops that dataset and the run reports on the rest.

### opig_ginkgo_developability — Ginkgo antibody properties

On five Ginkgo antibody developability properties, does a hybrid built inside each training fold beat the best single representation, and how do both compare to the competition's reported Spearman rho?

- **Hybrids** — its own. `find_best_hybrid` at `tests/experiments/opig_ginkgo_developability.py:321-345`, inside the per-fold loop at `:306`, inside the per-property loop at `:769`. Tier `general` plus a per-fold condition suffix, so the engine sees `general__cond-ginkgo_hold_fold0` through `__fold4`. Objective predictive, speed full, budgets all 8 entries. The five Ginkgo properties are precomputed on tier `general` over all rows, but this file searches a four-fifths training subset under a per-fold tier name, so none of that is readable. The comment at `:318-320` says so.
- **Compute** — per property per held-out fold, 5 allocation methods x 3 importance methods x 4 representation counts x 8 budget entries x 17 (model, evaluator) pairs = 8,160 configurations. Inner folds are 4, not 5: the search is handed the Ginkgo fold ids and the training rows carry 4 of the 5. Each configuration-fold costs 9 fits, being 1 downstream fit, 5 augmentation rung fits and 3 residual fits. That is 293,760 search fits, plus 120 importance, 200 allocation and 200 baseline fits. Outer scoring adds 12 representations x 5 models x 5 folds = 300 fits plus 25 hybrid fits per property. **~7,360,000 model fits across the five properties.** Each fit is on about 194 antibodies, so individually cheap, but 7.36 million fits for 5 properties of 197 to 242 antibodies each is out of proportion.
- **Hybrids it uses / should use** — one per held-out fold, 5 per property, all read from the search's random-forest winner and then scored with all five downstream models. The other four fitted results are built and discarded. As used: this is a per-property regression of a biophysical assay under fixed hierarchical cluster folds, with no distances between antibodies.
- **Selection pattern** — partial. 5 downstream models, all 12 base representations rather than the top 2, and 1 hybrid rather than 2: 60 model/representation pairs plus 5 model/hybrid pairs per property against the intended 20.
- **Cut: fast tier** — `tests/experiments/opig_ginkgo_developability.py:321-345`, add `speed='fast'` next to `tier='general'` on line 340. Configurations per fold go from 8,160 to 432, search fits per fold from 293,760 to 15,552, and the three pre-passes together from 520 to 280 fits. Fast also drops xgboost and svm from the search's own model list, since no model list is passed; both are still scored in the outer arms.
- **Cut: two evaluators** — `:302` sets `_evals` to 4 names and `:336` passes it. Cutting to `['match','rf']` takes pairs from 17 to 9, configurations per fold from 8,160 to 4,320 and search fits per fold from 293,760 to 155,520. Cutting `MODELS_LIST` at `:107` from 5 models to 2 saves only the outer arms, 325 fits per property down to 140, which is nothing next to the search.
- **Notes** — the hybrid arm is scored on at most 200 features, because anything wider than 500 features is cut to 200 at `:395`. So the budgets of 1000 and 2000 features the search pays for can never reach the reported number. `adaptive_pruning=0.25` at `:342` never takes effect; the schedule wins and the threshold in force is 0.15. The 9-fits-per-configuration figure depends on the two augmentation representations actually loading; if either is missing the strategy falls back and the search cost drops 9x. That branch is unverified. The engine re-pays 3 residual fits per configuration although a precomputed candidate ranking is accepted; the code's own comment puts that repeat at 57 percent of the grid's fits on herg_ki.

### opig_graphinity_ddg — binding free energy change

For antibody-antigen binding free energy change, how close do sequence representations plus ordinary models get to the structure-based Graphinity network, and does training on FoldX-simulated mutations transfer to measured SKEMPI values?

- **Hybrids** — its own, despite reading through the pool helper. `get_best_search_for_downstream` at `tests/experiments/opig_graphinity_ddg.py:447-462`, once, for the measured condition only. Tier `general` plus condition suffix `n<N>`, so `general__cond-n429`. Objective predictive, speed full, budgets all 8 entries. This dataset IS precomputed at speed full with pruning mode no_pruning, so this is a lookup that could hit and does not, for two reasons: the condition suffix rewrites the tier, and the pruning schedule this call carries writes a final threshold of 0.15 into the expected meta against the 0.0 the precompute stored.
- **Compute** — three pool members run, one per augmentation condition. Each is 8,160 configurations over 5 folds. Without augmentation that is 1 fit per configuration-fold, 40,800 fits, plus 520 pre-pass and baseline fits. Uncapped costs 9 fits per configuration-fold, 367,200 fits. Capped costs 8 fits at the 10-feature budget and 10 fits at each of the other 7 budgets, 397,800 fits. Outer scoring adds 9 representations x 5 models x 5 folds = 225 fits, and the transfer condition adds 9 representations x 5 models x 1 train/test pair = 45 fits. **~807,000 model fits for the measured condition**, plus 270 fits for both printed tables. Excessive against a dataset of at most 1,211 rows, and a free read exists but is blocked by two arguments.
- **Hybrids it uses / should use** — every configuration the three searches produced, on the order of 1,600 candidates per downstream model, re-picked per held-out fold. Only the pool member that won for random forest is carried forward, so two of the three searches are built and discarded. As used: this predicts a free energy change per mutation under group folds by PDB complex, with no distances and no nearest neighbours.
- **Selection pattern** — partial. The printed table is 9 representations x 5 models x 5 folds with no hybrid; the honest table uses 5 models, all 8 base representations and every searched configuration, re-picked per fold, at 0 new fits.
- **Cut: fast tier** — `tests/experiments/opig_graphinity_ddg.py:447-462`, add `speed='fast'`; it is forwarded untouched through the helper chain. Configurations per pool member go from 8,160 to 432, so 24,480 configurations across the three members become 1,296 and the fit count goes from about 807,000 to roughly 43,000.
- **Cut: two evaluators** — `:455-456` passes 4 evaluator names. Cutting to `['match','rf']` takes pairs from 17 to 9, configurations per member from 8,160 to 4,320 and the total from about 807,000 to roughly 427,000 fits. `MODELS_LIST` at `:93` governs the printed table only; cutting it to 2 takes 225 fits to 90 in the measured condition and 45 to 18 in transfer.
- **Notes** — the cheapest cut here is not the fast tier. Dropping `condition_suffix` at `:457` AND calling the pruning helper with `no_pruning` so the raw schedule matches what the precompute used makes this a stored read at near zero new fits. Dropping the condition suffix alone leaves the schedule mismatch; dropping the schedule too is worse, because a stored file carrying pruning fields is rejected when the request carries none. This was read from the code, not from a cache file on disk. Separately, the printed table and the honest table do not score the same feature spaces: one cuts any representation wider than 500 features to 200, the other only above 5,000 features, so "best single representation" means two different things.

### opig_humanness — human or non-human antibody

Which antibody representation classifies a therapeutic antibody as human or non-human best, and does a hybrid beat the best single representation on a held-out fold?

- **Hybrids** — its own. `get_best_search_for_downstream` at `tests/experiments/opig_humanness.py:309-323`. Tier `general`, objective predictive, speed full, budgets all 8 entries. A precompute for this dataset, tier and condition does exist and cannot serve this call, for one reason: `:106-111` adds `onehot` to the 10-representation approved pool, making 11, and the sorted representation names and widths are an identity column. The block that builds the precompute says so itself, that no experiment asking for the full pool will ever read it.
- **Compute** — 564 antibodies, 198 of them human, 11 representations, 5 folds. Configurations are 5 allocation methods x 3 importance methods x 4 representation counts x 8 budget entries x 17 (model, evaluator) pairs = 8,160. Pruning runs a vote pass over every configuration on folds 0 and 1, then re-runs survivors on all 5 folds, so search fits are 16,320 plus at most 40,800. Add 165 importance, 275 allocation and 275 baseline fits, plus a direct arm of 11 representations x 5 models x 5 folds = 275 fits. **Up to ~57,800 model fits on the one dataset**, with a floor of about 16,835. Modest against the rest of the suite, and the dataset is small.
- **Hybrids it uses / should use** — every configuration the one search evaluated, grouped by downstream model, with fold 0 dropped from the re-score. Per-model candidate counts are 1,440 for rf, dnn and lightgbm and 1,920 for xgboost and svm. Nothing is capped to a top N. As used: no distances between antibodies.
- **Selection pattern** — partial. 5 downstream models, all 11 base representations rather than a top 2, and every searched configuration rather than a top 2: 55 representation-model pairs against the intended 20.
- **Cut: fast tier** — `tests/experiments/opig_humanness.py:309-323`, add `speed='fast'`. Grid goes from 8,160 to 432 configurations, search fits from at most 57,120 to at most 3,024, the importance pre-pass to 55 fits, the allocation pre-pass to 165 fits and baselines to 165 fits.
- **Cut: two evaluators** — no evaluator list is named in this file; it falls through to the engine's 4. Add `internal_evaluators=['rf','dnn']` at `:309-323`: dropping `match` makes pairs a flat product, 17 to 10, configurations 8,160 to 4,800, and search fits from at most 57,120 to at most 33,600. `MODELS_LIST` at `:113` governs only the direct arm; cutting it to 2 takes 275 fits to 110 and leaves the grid untouched.
- **Notes** — under `submit_everything.sh`, which sets the no-cold-search variable at line 339, the search raises, `:332` swallows it, and the results file is written with base-representation rows only and an exit status of 0. Under `submit_opig_experiments.sh` no such variable is set and the whole grid builds. A refused run still pays 11 representations x 5 models x 5 folds = 275 baseline fits before it raises, because baselines are fitted before the refusal. `adaptive_pruning=0.25` at `:314` sets no threshold; the explicit schedule wins.

### opig_kingsmith — held-out reaction components

When a whole reaction component is held out, one additive or aryl halide or ligand or base, does a hybrid predict yield better than the best single representation on that unseen component?

- **Hybrids** — its own. `find_best_hybrid` at `tests/experiments/opig_kingsmith_component_splits.py:460-488`, once per outer fold, 90 searches across the two datasets. Tier `general` plus condition suffix `loo_<split>_fold<i>`. Objective predictive, speed full, budgets all 8 entries, augmentation strategy uncapped because both datasets have 4 augmentation representations. Nothing precomputes any of it: the search is handed the training rows of one leave-one-component-out fold, so the labels hash and groups hash differ from any whole-dataset stored row.
- **Compute** — per outer fold, 8,160 configurations. Inner folds are 5 on the additive and aryl halide splits, 3 on ligand and 2 on base, because the training group count sets them. Each configuration-fold costs 9 fits: 1 downstream fit, 5 augmentation ladder rungs and 3 out-of-fold residual fits. Every configuration is scored twice over, once on 2 folds for the pruning vote and again on all folds if it survives. One additive-fold search is at most 514,730 fits and at least 147,530. buchwald_hartwig has 22 additive, 15 aryl halide, 4 ligand and 3 base groups; bh_doyle_plates has 24, 15, 4 and 3. **Up to ~43,800,000 model fits for the default run of both datasets, floor about 12,500,000.** This is the largest single number in the suite and it runs in one 47-hour job.
- **Hybrids it uses / should use** — exactly one per held-out component: the search's fitted winner for random forest, then scored with all 5 downstream models. Ninety hybrids over the whole run. As used: no distances between reactions.
- **Selection pattern** — partial. 5 downstream models, 14 base representations all evaluated, 1 hybrid: 75 representation-model pairs per split against the intended 20.
- **Cut: fast tier** — `tests/experiments/opig_kingsmith_component_splits.py:460-488`, add `speed='fast'`. Grid goes from 8,160 to 432 configurations, one additive-fold search from at most 514,730 to at most 27,890 fits, and the whole experiment from at most 43.8 million to at most 2.4 million fits. The evaluator list is passed at `:475` and survives fast, and the fast evaluator list is the same four names anyway.
- **Cut: two evaluators** — `:457` sets `_eval_list` to 4 names, passed at `:475`. Cutting to 2 names without `match` makes pairs a flat product, 17 to 10, configurations 8,160 to 4,800, one additive-fold search from at most 514,730 to at most 303,650 fits, and buchwald_hartwig from at most 21.4 million to at most 12.6 million fits. `MODELS_LIST` at `:119` governs only the scoring arms; cutting it to 2 takes the base arm from 3,080 to 1,232 fits on buchwald_hartwig.
- **Notes** — a second cut is bigger than either of these: `augmentation_strategy='uncapped'` at `:464` is what turns 1 fit per configuration-fold into 9, and setting it to `none` cuts the run 9x on its own. Tier `general` at `:483` puts these per-fold searches on a precompute tier root while nothing can precompute them, so under `submit_everything.sh` the cold-search guard raises, `:489` swallows it, every fold pads, and the job exits 0 with no hybrid. Each padded fold still burns 250 baseline fits before the refusal. This file appears only in `submit_everything.sh`, not in `submit_opig_experiments.sh`. The hybrid's feature selection is the random-forest winner but is scored with all 5 models, while each base representation is scored with its own model.

### opig_scaffold_hybrid — mutation-difference representations in the pool

Put the two mutation-difference representations into each antibody dataset's own ten-representation pool, and ask on held-out folds whether they score higher than the base representations and whether the hybrid built from the twelve beats the best single representation.

- **Hybrids** — its own. `run_nested_cv` at `tests/experiments/opig_scaffold_hybrid.py:269`, which reaches `find_best_hybrid` once per outer fold at `find_best_hybrid.py:10329`. Tier `scaffold_cmp`, objective predictive, speed full, budgets all 8 entries, 5 inner folds and 5 outer folds. Nothing precomputes it, for two reasons: `scaffold_cmp` is not one of the two precompute tier roots, so nothing on the cluster ever wrote these rows, and the pool is 12 representations where every precompute for these four datasets built 10.
- **Compute** — per outer fold, 5 allocation methods x 3 importance methods x 4 representation counts x 8 budget entries x 17 (model, evaluator) pairs = 8,160 configurations x 5 inner folds x 1 fit = 40,800 search fits. Add 180 importance, 300 allocation and 300 baseline fits. Outer scoring is 645 fits per outer fold, because each of the 9 rebuilds per model re-scores all 12 representations to size the budget split. That is 42,225 fits per outer fold, 211,125 per dataset, and 4 datasets gives **~845,000 model fits in one job**. The search fits are an upper bound, because a repeated feature selection on the same fold and model reuses a stored score at an unmeasured rate. Excessive against the wall: one 8,160-configuration grid measures 79 hours unpruned on logd and 44 pruned, and this job runs twenty of them behind a 7-day wall.
- **Hybrids it uses / should use** — per (downstream model, outer fold), one winning recipe plus one recipe per budget entry, 9 hybrids, giving 225 hybrid evaluations per dataset. Picked by the inner search's own mean score on the outer-training rows. As used: the file trains real models and scores in the dataset's own metric, and its docstring at `:19` says outright this is not the nearest-neighbour proxy. Small budgets are already covered, since the per-budget arm scores a hybrid at each of the 8 budget entries down to 10 features.
- **Selection pattern** — different. It scores all 12 representations for each of 5 downstream models on each of 5 outer folds and 9 recipes per (model, outer fold): 105 model/representation pairs per outer fold against the intended 20.
- **Cut: fast tier** — `tests/experiments/opig_scaffold_hybrid.py:269`, add `speed='fast'` to the `run_nested_cv` call; it survives the kwargs merge and reaches the engine. Grid per outer fold goes from 8,160 to 432 configurations and the per-dataset total from about 211,000 fits to about 14,800. `EXPERIMENTS_FINALIZED.md:218` lists this file among those that should run fast; the file passes no speed argument anywhere.
- **Cut: two evaluators** — two lists. `:273` passes 4 evaluator names; naming 2 without `match` takes pairs from 17 to 10, configurations per outer fold from 8,160 to 4,800 and the allocation pre-pass from 300 to 120 fits. Separately there is no models flag, deleted at `:223-227`, but `models=['rf','dnn']` passed at `:269` overrides the pinned default: pairs 17 to 6, configurations 8,160 to 2,880, baselines 300 to 120 fits and outer scoring 645 to 258 fits per outer fold.
- **Notes** — a defect on the rebuild path. Each configuration may be restricted to the top 2, 3 or 5 representations by allocation score, and that restriction is applied only inside the config evaluator. The blueprint records the count but the rebuild never passes it on and rebuilds from every representation in the pool. So a recipe chosen on 2 representations is scored on the outer fold having been rebuilt on all 12. Separately `--distance-metric` at `:82-95` is applied process-wide and nothing on this run's path reads it, so the flag changes no number here. `--mode fast` cuts outer folds to 1 and leaves the 8,160-configuration grid untouched.

### opig_scaffold_similarity — mutation-difference distances

For variants of one wildtype antibody, which mutation-difference representation and which distance formula give pair distances that track an independent substitution-matrix distance once the plain count of differing positions is removed?

- **Hybrids** — none. No search, no hybrid, no tier, no objective, no budget list. The only feature selection is a forward selection over 14 amino-acid property scales, scored by rank correlation.
- **Compute** — 0 model fits. Per dataset under the default mode: the representation mode is 5 representations x 2 distance formulas x 5 correlations = 50 rank correlations over up to 32,000 variant pairs; the feature mode is about 180 count-partialled Spearman evaluations over roughly 16,000 pairs each; the control mode adds 2 correlations plus, when mean mutations per variant is at most 1.001, one full n x n position-gated distance matrix, which at the 5,000-variant cap is 25,000,000 pairwise distances in memory. **0 model fits**, and the docstring puts the smoke run at about 1 minute. Nothing to cut.
- **Hybrids it uses / should use** — none today. This file IS entirely distances between molecules: every number is a pair distance correlated against a substitution-matrix pair distance. It is the natural place to grade similarity-objective hybrids, the family where the recipe is chosen so each molecule's activity is predictable from its five nearest neighbours, and small hybrids of 10 to 50 features, against the same structural ground truths. The place to make that change is the representation dictionary at `tests/experiments/opig_scaffold_similarity.py:301-307`, where the five representations are named literally. That is an addition to the file, not a change to what it measures.
- **Selection pattern** — different. No downstream evaluator, no base-representation ranking and no hybrid: 5 hand-built representations x 2 distance formulas against 3 substitution-matrix ground truths, plus a forward feature selection over 14 property scales fitted on one half of the pairs and scored on the other.
- **Cut: fast tier** — nothing to cut; there is no search.
- **Cut: two evaluators** — nothing to cut; there is no downstream evaluator list.
- **Notes** — `--distance-metric` at `:90-97` declares itself as the pairwise distance for every quantity the script computes, and every distance in the file is written out inline instead. The flag can be set to anything and no number moves. Undefined correlations are written out as a measured 0.0 at `:323` and `:328`. `--n-pairs` names 32,000 pairs, but about half the drawn candidates are dropped because their two variants fall in different halves.

### opig_shared_vectors_census — molecules sharing one row

How many molecules does each representation give a byte-for-byte identical row to another molecule, split by cause, and what is the best score any model could reach once those molecules must share one prediction?

- **Hybrids** — none. No search of any kind, no tier, no objective, no budgets.
- **Compute** — 0 model fits. Per dataset it runs one row partition per representation, plus one joint partition for each representation pair, plus one running concatenation. On herg_ki that is 15 representations, so 15 row partitions and 105 pairwise comparisons. Across the default roster of 29 datasets the pairwise total is 1,954 comparisons. Two RDKit passes per molecular dataset are the slowest part. **0 new model fits.** The freshness check rebuilds up to 8 representations on evenly-spaced molecules, which is featuriser work rather than model fits; `--build-missing` is the only expensive flag.
- **Hybrids it uses / should use** — none, and there is nothing to select. The number it produces, how many molecules a representation cannot tell apart, is the degenerate case of the question a similarity-objective search asks, since two molecules on the same row sit at distance zero from each other.
- **Selection pattern** — different. No evaluators, no models and no hybrids: it enumerates every cached representation in the dataset's pool plus its augmentation donors, 15 on herg_ki, and reports counts and ceilings for each.
- **Cut: fast tier** — nothing to cut; there is no search.
- **Cut: two evaluators** — nothing to cut; it fits no downstream model.
- **Notes** — the printed floor is over the wrong set of representations. The pool row concatenates the augmentation donors as well as the pruned pool, and the print then calls that number the floor a hybrid cannot go below. A hybrid built with augmentation off selects from the 10 representations of the general pool only and never sees the 5 donors, so the printed floor is lower than the floor that applies to it. The two sides differ in one way: which representations the concatenation is over. Row identity is a byte comparison on raw float32, so two rows of NaN count as identical; the NaN row count is reported alongside.

### opig_shared_vectors_remedy — fixing shared rows

When a representation gives two different molecules the same row, does hybridising it with other representations or adding a donor's features both improve the held-out score and reduce how many molecules still share a row?

- **Hybrids** — its own. `run_nested_cv` at `tests/experiments/opig_shared_vectors_remedy.py:404-421`, which reaches `find_best_hybrid` once per outer fold. Tier `shared_vec`, objective predictive, speed from a flag defaulting to full, budgets all 8 entries, representation counts pinned to a single entry. Nothing precomputes it: `shared_vec` is not a precompute tier root, so the cold-search guard returns before counting rows and every configuration is paid live. Two further blocks: the augmentation strategy `all` is not one of the three conditions the precompute writes, and the condition suffix gives each of the 516 sweep points its own result database.
- **Compute** — per inner search, 5 allocation methods x 3 importance methods x 1 representation count x 8 budget entries x 17 (model, evaluator) pairs = 2,040 configurations, each scored on 5 inner folds. Adaptive pruning is off by default here, so the whole grid is paid on all 5 outer folds. Cost per configuration-fold has five terms: 1 downstream fit, the k-ladder at 5 rungs or 2 when a fraction is set, the candidate ranking at 0 to 3 fits, a donor-trim random forest when a per-source budget is set, and a base-feature ranking forest under the capped strategy. Mean over an uncapped or capped sweep point is 6.25 fits. Runs per dataset are 2 donor-free runs plus 516 sweep points x 3 comparisons = 1,548 swept runs. **~493,000,000 model fits per dataset at full speed, ~53,000,000 at fast.** That is the largest figure in the suite, on one dataset of 1,412 molecules, and the submit script already refuses more than one dataset per sweep submission.
- **Hybrids it uses / should use** — per run, per downstream model and per outer fold, one headline hybrid from 2,040 configurations plus 8 per-budget hybrids that are rebuilt and scored. The summary reads only the headline one, so 5 models x 5 folds x 8 budgets = 200 fits per run are paid and never appear in the output. This experiment rests entirely on whether two molecules sit at distance zero from each other in the built matrix, and every search it runs is predictive. If similarity-objective hybrids belong anywhere in the suite they belong here, added at `:404-421` alongside the existing tier and speed arguments. The small-budget end is already in the sweep but nothing weights it, and the per-budget scores that would show whether a 10-feature or 25-feature hybrid breaks the ties are computed and discarded.
- **Selection pattern** — different. Per run it scores 5 downstream models, all base representations in that run's pool on every untouched outer fold, 3 for two of the comparisons and 1 for another, then reports one hybrid per (model, fold) against one named target representation and throws the rest away.
- **Cut: fast tier** — `:410` passes the speed flag and `:561` defaults it to full; changing that default, or passing `--speed fast` which the submit script already plumbs through, takes each inner search from 2,040 to 216 configurations per outer fold. Per dataset that is about 493 million fits down to about 53 million, a factor of 9.3. Fast does not shorten the evaluator list; the pair count falls only because the model list falls from 5 to 3.
- **Cut: two evaluators** — no evaluator list is passed, so the search uses the engine's 4 names. Add `internal_evaluators=[...]` to the `run_nested_cv` call at `:404-421`, and pin the models through the existing flag defined at `:562-566`. Cutting both to 2 takes pairs from 17 to 4 and configurations from 2,040 to 480 per outer fold. With fast speed as well the grid is 96 configurations, and the per-dataset total falls to roughly 24 million fits.
- **Notes** — four things. The submit script prints 4,080 configurations per outer fold and 948 million fits per dataset at `submit_opig_shared_vectors.sh:162-165`, and the same 948 million appears at `submit_opig_experiments.sh:82`; the code pins the representation-count axis to one entry, so the real figures are 2,040 configurations and about 493 million fits. The margin axis is a no-op whenever a fraction is set, so 288 of the 516 sweep points repeat another point's behaviour under a different condition suffix, and each still pays a full run because the per-configuration cache carries the margin in its meta. The docstring promises a duplicate check that the code does not do. And one comparison at the single none sweep point is another comparison under a different name: the donor is never used, and it costs a full 52,275-fit run. Unverified: how often the feature-selection dedup removes the downstream fit.

### opig_trastuzumab_enrichment — HER2 binder enrichment

On held-out folds of 5,000 trastuzumab CDRH3 variants, which representation and downstream model put the most HER2 binders into the top 1, 5 and 10 percent of the ranked rows, and does a hybrid beat the best single representation at that binder count?

- **Hybrids** — its own, at two call sites. `find_best_hybrid` at `tests/experiments/opig_trastuzumab_enrichment.py:402`, once per outer fold, 5 searches, tier `general` plus condition suffix `enrich_fold<i>`. And `get_best_search_for_downstream` at `:644`, once on all 5,000 sequences, tier `general`. Objective predictive at both, speed full, budgets all 8 entries. No precompute serves either, for three reasons: this dataset is precomputed at fast speed while these calls ask for the full lists, the pool here is 13 representations against the precompute's 10, and `adaptive_pruning=0.25` at `:652` overrides the reader's own value, which the reader's comment says means a different search.
- **Compute** — each search is 5 allocation methods x 3 importance methods x 4 representation counts x 8 budget entries x 17 (model, evaluator) pairs = 8,160 configurations, scored on 5 internal folds at 1 fit each = 40,800 fits. Add 195 importance, 325 allocation and 325 baseline fits, plus up to 5 whole-data refits. The whole-data search is 41,650 fits; the 5 per-fold searches are about 208,250 fits. The supervised per-representation loop adds 13 representations x 5 models x 5 folds = 325 fits, the zero-shot ranking adds 0 fits, and the enrichment refit adds 25 fits. **~250,250 model fits on the one dataset of 5,000 antibody sequences**, of which the five per-fold searches are about 208,250, or five sixths of the total.
- **Hybrids it uses / should use** — two sets. The whole-data arm feeds every stored recipe into a leave-one-fold-out re-score per downstream model, so the candidate set is all 8,160 configurations' fold scores with no top-N cut. The per-fold arm uses exactly one hybrid per (outer fold, downstream model), the search's fitted winner with a fallback to random forest, 25 in total. No hybrid is chosen at a named budget anywhere in the file. As used: it ranks held-out rows by a fitted classifier's score, with no distance between sequences. Worth noting that with no budget pinned, a 2,000-feature recipe can win on 5,000 sequences with nothing preferring a narrow one.
- **Selection pattern** — different. 13 representations x 5 downstream models x 5 folds in the supervised loop, plus one hybrid per model per fold in one arm and every searched recipe in the other, rather than 2 representations and 2 hybrids per evaluator.
- **Cut: fast tier** — two places. `:402-419`, add `speed='fast'` beside the evaluator list at `:410`: each fold search goes from 8,160 to 432 configurations, and the five fold searches together from 204,000 fold-evaluations to 10,800. `:644-655`, add `speed='fast'` as an explicit keyword: same 8,160 to 432. Note the comprehension at `:653` strips a speed key only when the smoke variable is set, so outside a smoke run it strips nothing, and the comment at `:115-116` saying speed is ignored is false against the current engine.
- **Cut: two evaluators** — `MODELS_LIST` at `:103` is 5 downstream models and drives the supervised loop and the per-fold refit; cutting to 2 takes 325 fits to 130 and 25 fits to 10, and does not touch the searches. To cut the searches, `models=` must be added at `:402-419` and `:644-655`: with two models the pairs go from 17 to 6, each search from 8,160 to 2,880 configurations, and the per-fold arm from about 208,250 to about 73,500 fits. The internal evaluator list at `:375` is separate; trimming it to two names cuts the pairs term again, but only for the per-fold searches.
- **Notes** — the per-fold searches ride a condition suffix that the engine folds into the tier string, and the cold-search guard splits only on a different suffix, so it returns early. `KIRBY_NO_COLD_SEARCH` therefore cannot stop five full grids that are five sixths of this file's compute. The whole-data call would raise under that variable, but the exception is caught one level up in the pool iterator, which prints a warning and continues, leaving an empty comparison and an exit status of 0. `EXPERIMENTS_FINALIZED.md` is stale on three points and the code wins: the representation count, whether the hybrid enrichment number is computed, and how many budgets svm runs across. Unverified: whether the honest re-score collapses or duplicates configurations that differ only in representation count, since the recipe name omits that axis.

---

## Federated

### fed01 — Noise heterogeneity

Five parties hold molecules whose labels are wrong at different rates; does splitting a 100-feature budget across them beat averaging their predictions, and does dropping suspect rows help?

- **Hybrids** — Its own, both kinds. `tests/experiments/federated/fed01_noise_heterogeneity.py:189` calls `get_best_search_for_downstream`, once per noise level, 6 times per run. `create_hybrid` is called at fed01:396, :503, :640, :725 and :857. Tier: the clean level asks on `general`; the five noisy levels arrive as a NoiseSpec and `find_best_hybrid.py:8453` rewrites the tier to `<spec_id>__lb<8 hex of the corrupted label>`, with the noise seed inside `spec_id`. Objective: predictive, because nothing passes `objective=` or `pool_subset=`. Speed: never passed, so the full grid. Nothing precomputes any of the six — the noisy tiers are experiment-local, and the clean request uses binarised labels with metric `auc`, which are identity fields of the result database, so it cannot read what precompute wrote for caco2 or logd.
- **Compute** — One search costs 5 allocation methods x 3 importance methods x 4 representation counts x 8 budget entries x 17 (model, evaluator) pairs = 8,160 configurations, and 5 folds turn that into 40,800 downstream model fits. On caco2 the three augmentation conditions are 8,160 + 8,160 + 6,240 = 22,560 configurations, so 112,800 fits, plus 650 fits of importance, allocation and baseline pre-passes = 113,450 fits per search identity. There are 16 search identities per dataset: 1 clean, plus 5 noise levels x 3 noise seeds. That is 1,815,200 fits on caco2, 1,546,400 on logd and 687,120 on Lhasa hERG. The conditions add 12,015 fits per run and 108,135 fits across the 9 runs on caco2, 306,855 on Lhasa hERG. Headline: about 1.92 million model fits per run of this file on caco2, 1.65 million on logd, 0.99 million on Lhasa hERG. Every configuration count is an upper bound, because `find_best_hybrid.py:4905-4915` skips a repeated (fold, model, feature-selection) fit by an amount not readable from the code. Excessive: 97 fits in every 100 are inside cold searches whose only purpose is to hand back four strings.
- **Hybrids it uses / should use** — 1 recipe per noise level, 6 per run, each the best configuration at 100 features for one downstream model. Only 4 fields survive: allocation method, importance method, internal evaluator, augmentation strategy (fed01:218-229). This rests on distances: one condition weights the five parties by 8-nearest-neighbour Morgan similarity, another flags rows with knn_k5 plus prediction_error plus zscore, and 4 of the 8 detectors in the census are neighbour detectors. Similarity-objective hybrids belong here — features chosen so each molecule's activity is predicted from its five nearest neighbours — alongside the predictive ones. Passing `pool_subset='predictive_and_similarity'` at fed01:189 is one line, and it doubles the searches, since no similarity precompute exists for these labels. The 10-feature and 25-feature budgets are already inside every finished search, so reading a small-budget recipe costs nothing more.
- **Selection pattern** — Different shape: it compares 7 ways of building one classifier at one noise level, with 0 base representations from the pool scored alone, 1 hybrid recipe read per noise level, and 3 model families x 3 seeds run as 9 separate whole-file runs rather than as evaluators inside one.
- **Cut: fast tier** — fed01:189, add `speed='fast'`. Per pool member the grid goes from 8,160 configurations to 3 allocation methods x 1 importance method x 2 representation counts x 8 budget entries x 9 (model, evaluator) pairs = 432 configurations. On caco2 the 16 identities go from 1,815,200 fits to about 109,300; on logd to about 97,800. ⛔ It also breaks the read: fast keeps only rf, lightgbm and dnn as downstream models, so the xgboost and mlp runs find their model absent and fall silently back to the hardcoded performance_weighted / treeshap / match at fed01:220-222. Fast is safe here only if the model sweep is cut to rf first.
- **Cut: two evaluators** — Two separate lists. The model families are `FED_MODELS = ['rf','xgb','mlp']` at `federated_helpers.py:328`; cutting to 2 takes 9 runs per dataset to 6 and the condition fits from 108,135 to 72,090, leaving the searches untouched. The evaluator list inside each search is unset, so 4 default names give 17 (model, evaluator) pairs; passing `internal_evaluators=['rf','match']` at fed01:189 gives 9 pairs, 8,160 configurations down to 4,320, and about 41,450 fits per identity down to about 22,250. The larger cut is neither: `FED_SEEDS = [0,1,2]` at `federated_helpers.py:330` rides on the noise tier, so 2 seeds take 16 identities to 11 and caco2's 1,815,200 search fits to about 1,248,000.
- **Notes** — ⛔ The mlp family never gets a searched recipe and the run pays for the search anyway: mlp is model tier `extra` (`find_best_hybrid.py:2573`), excluded from auto-selected tiers, so the pool returns nothing and fed01:220-222 uses the hardcoded recipe. Three of the nine runs per dataset report a hardcoded recipe and the result file does not record it. ⛔ Under `KIRBY_NO_COLD_SEARCH` the clean noise level is refused at `find_best_hybrid.py:650` and fed01:210's bare `except` swallows it, while the five noisy levels build their full grids regardless — the guard makes the clean arm cheaper and silently different. The flip fractions are a fixed linspace from 0 to 0.2 across the five parties (fed01:711 and fed01:319), so party 1 has no flipped labels in any fold and its precision, recall and f1 are 0 by construction.

### fed02 — Representation diversity across silos

Five parties see the same molecules through different representations; does merging the representations at the feature level beat the best single representation and beat averaging the five parties' predictions?

- **Hybrids** — Both. `tests/experiments/federated/fed02_rep_diversity_silos.py:193` calls `get_best_search_for_downstream` once per run, before the fold loop, with `data=data`. `create_hybrid` is called at fed02:491, fed02:548 and inside the helper at `federated_helpers.py:2077` and `:2133`. Tier `general`, objective predictive, speed unset so the full grid, budgets the 8 full entries. On caco2 and logd every pool member is answered from memory, because `run_all` fills `data['precomputed_searches']` at `federated_helpers.py:2778` and the lookup at `_shared.py:5298-5300` checks only the augmentation condition. On Lhasa hERG there is no precompute at all, so the request goes to disk as a cold full grid over 33 representations.
- **Compute** — Search on caco2 and logd: 0 new model fits inside fed02. The three searches for those datasets are paid once by `run_all`'s precompute and shared across all five federated experiments and all 9 runs — 113,450 fits on caco2, 96,650 on logd — and only if `tests/precompute_hybrids.py` already filled the on-disk cache, which is unverified here. Search on Lhasa hERG: 8,160 configurations x 5 folds = 40,800 fits, plus 495 importance, 825 allocation and 825 baseline fits = 42,945 fits, paid once by whichever of the 9 runs arrives first. Conditions, per fold: 15 ranking fits + 5 single-representation fits + 5 identical rebuilt teacher fits + 25 cross-validated prediction fits + 11 + 7 + 76 = 144 fits, so 720 fits per run and 6,480 fits across the 9 runs. Headline: about 6,500 model fits per run of this file on caco2 or logd, about 49,400 on Lhasa hERG. Not excessive on caco2 or logd; the one cold 33-representation search on Lhasa hERG is the whole cost there.
- **Hybrids it uses / should use** — 1 recipe per run, and only 4 fields of it (fed02:212-227). Those 4 build 3 feature sets per fold: over all 5 party representations, over the 3 that won the inner ranking, and over 15 virtual parties cut by KMeans inside each representation. All 3 at 100 features (fed02:71). The searched hybrid's own feature set is never used. Distances decide two conditions: one weights the five parties by 8 nearest neighbours in each representation's own space, and the virtual-party condition clusters molecules by distance. A similarity-objective recipe is the right comparator for the clustering condition, which is the one asking whether cutting molecules up by distance helps. The omission is at fed02:193 — no `objective=` and no `pool_subset=`. The stored search already holds 10, 25 and 50 features, so small budgets cost one more read plus 3 more `create_hybrid` builds per fold.
- **Selection pattern** — Partial: 5 base representations are scored alone per fold, which is the whole party axis rather than a top-2 cut; 1 hybrid recipe becomes 3 feature sets; 3 model families x 3 seeds run as 9 whole-file runs. Per fold the comparison is 5 base representations + 3 hybrid feature sets + 2 prediction-averaging conditions = 10 rows, against the intended 4 rows per evaluator.
- **Cut: fast tier** — fed02:193, add `speed='fast'`. On caco2 and logd it saves nothing and costs: the in-memory hit returns before speed is consulted, and a fast identity is a separate file from the precomputed full one, so a miss would rebuild rather than read. It helps only on Lhasa hERG, where the cold search goes from 8,160 configurations to 432 and from 42,945 fits to about 3,315. ⛔ Same trap as fed01: the xgboost and mlp runs would fall back to the hardcoded performance_weighted / treeshap / match at fed02:223-227.
- **Cut: two evaluators** — `FED_MODELS` at `federated_helpers.py:328`; cutting to 2 takes 9 runs per dataset to 6 and the condition fits from 6,480 to 4,320, with no effect on the search. Overriding the evaluator list at fed02:193 is the wrong place, because it turns the free lookup into a cold rebuild. The cheaper cut in this file is `K_INTRA = 3` at fed02:70, which makes 5 representations x 3 sub-clusters = 15 virtual parties costing 76 of the 144 fits per fold; 2 sub-clusters give 10 virtual parties and 51 fits per fold.
- **Notes** — The mlp family gets no searched recipe here either, for the same tier reason, and the result file does not say so. The five per-representation models are fitted twice per fold on identical inputs, at fed02:361 and fed02:427 — 5 wasted fits per fold, 225 fits per dataset across the 9 runs. The quality-weighting condition runs cross-validated prediction on each party's full-width representation, so ecfp4 at 2,048 features is refitted 5 times per fold per run to produce one weighting number. The per-virtual-party cost of 5 fits is an upper bound: each of those `create_hybrid` calls runs over 1 representation, and whether an allocation-scoring fit still happens with 1 representation is unverified.

### fed03 — Synthetic private silos

Does one 100-feature selection spread across three data sources beat the best single source and beat averaging the three sources' predictions?

- **Hybrids** — Both. `tests/experiments/federated/fed03_synthetic_private.py:174` (Lhasa hERG) and `:859` (caco2, logd) call `get_best_search_for_downstream`. `create_hybrid` runs at fed03:683, :1031, :1130, :1159 and inside the helper at `federated_helpers.py:1261`. Tier `general`, objective predictive, speed unset so the full grid, budgets the 8 full entries. On Lhasa hERG nothing readable exists: the precompute for that dataset was built on a 10-representation pool while this file loads 33 base representations, and the representation pool is an identity field of the result database. On caco2 and logd `run_all` builds the searches itself at `federated_helpers.py:2768-2778` under the pruning mode the submit path sets to `aggressive_pruning`, while the stored precompute used `calibrated`; every pruning field is in the search cache key, so nothing on disk is readable and three full grids are cold-built minutes before fed03 runs.
- **Compute** — Search on Lhasa hERG: 495 importance fits + 825 allocation fits + 825 baseline fits + 8,160 configurations x 5 folds = 42,945 model fits, paid once per dataset. Search paid by `run_all` before fed03 starts: caco2 about 114,650 fits across three augmentation conditions, logd about 107,400 fits, the lower conditions coming from the per-dataset model-skip policy. Conditions on Lhasa hERG, per fold: 6 + 3 + 202 + 1 + 3 + 132 = 347 fits, so 1,735 fits per run and 15,615 fits across the 9 runs. Conditions on caco2 or logd: about 77 fits per fold, about 3,500 fits across the 9 runs; the augmentation-selection fits inside `create_hybrid` were not re-derived, so that figure is unverified. Headline: about 58,500 model fits on Lhasa hERG, about 118,100 on caco2, about 110,900 on logd. Excessive: on every dataset the searches are cold and they are more than 9 fits in every 10.
- **Hybrids it uses / should use** — Lhasa hERG evaluates 1 hybrid per fold, the per-party selection at 100 features total, plus 1 single-representation control (ecfp4) and 1 per-representation ensemble. caco2 and logd evaluate 2 hybrids per fold: 1 at 100 features over all three representation families and 1 built family by family at 33 features each. No hybrid is ranked against another anywhere. Lhasa hERG rests on distances: one condition weights each source's model by the mean distance to the 8 nearest training molecules, computed on 2,048-bit Morgan fingerprints built at fed03:441-453 and never on any hybrid, so the hybrid choice cannot move that condition today. If that arm is meant to be a KIRBy arm it should take similarity-objective hybrids — each molecule's activity predicted from its five nearest neighbours — and the 10-feature, 25-feature and 50-feature budgets rather than the fixed 100. The choice points are fed03:174-183 and fed03:611-623. The caco2 and logd part is a feature-selection comparison and predictive is right there.
- **Selection pattern** — Different shape: each run uses 1 downstream classifier family, swept to 3 families x 3 seeds = 9 runs; Lhasa hERG compares 7 named conditions of which 3 touch representations, and caco2 and logd compare 3 family concatenations against 2 hybrids, with 1 base representation scored alone on Lhasa hERG and 0 elsewhere.
- **Cut: fast tier** — fed03:174 and fed03:859, add `speed='fast'`. The grid goes from 8,160 configurations to 432 per augmentation condition, 18.9 times fewer. Worth it on Lhasa hERG, where nothing readable exists. On caco2 and logd the saving belongs to `precompute_hybrid_searches`, which has no speed parameter in its signature (`_shared.py:4921-4932`), so changing fed03:859 alone changes nothing there.
- **Cut: two evaluators** — fed03 has no evaluator list of its own. `FED_MODELS` at `federated_helpers.py:328` crossed with `FED_SEEDS` at `:330` gives 9 whole runs per dataset; cutting to 2 model families removes a third of every condition figure — 5,200 of Lhasa hERG's 15,615 fits, about 1,200 of caco2's 3,500. The mlp third is the cheapest to drop, because no search ever scores mlp and that third always lands on the treeshap fallback at fed03:190-192. A second cut sits inside the file: the per-representation ensemble at fed03:709-732 fits one classifier plus one 3-fold cross-validated prediction for each of Lhasa hERG's 33 representations, 132 fits per fold; restricting it to 2 representations leaves 8 fits per fold and saves 5,580 fits across the 9 runs.
- **Notes** — ⛔ On Lhasa hERG the three sources collapse to two. `run_all` loads the same 7,772 molecules that fed03 then loads again as its second source, deduplication by InChIKey keeps the first source in dictionary order, and the second source is emptied: its row becomes `{'error': 'too few samples'}` and its score is 0. The docstring at fed03:150-152 still describes three populated sources. Verified by reading the loader and the deduplication rule, not by running. Lhasa hERG rebuilds all 33 representations for about 12,000 SMILES at fed03:404, inside the experiment's wall, per run, unless the representation cache hits — and that call goes to the builder directly rather than through the caching wrapper. Whether all 33 names actually build is unverified: 10 of them are optional heavy builders, and any that fails is silently absent and scales every count down.

### fed04 — Cluster-aware importance

Does the 100-feature budget split differently from one cluster of training molecules to another, and does a per-cluster classifier beat one trained on the whole fold?

- **Hybrids** — Its own, both kinds. `tests/experiments/federated/fed04_cluster_aware_importance.py:196` calls `get_best_search_for_downstream` with `data=None`, which is deliberate and forces the on-disk path. `create_hybrid` runs at fed04:89, :275 and :396. Tier `general`, objective predictive, speed unset so the full grid, budgets the 8 full entries. ⛔ No precompute matches on any of the three datasets. caco2 and logd are asked with binarised labels, metric `auc` and classification while both are registered as regression with rmse, and the label hash, metric and task type are all identity fields. Lhasa hERG matches on metric but uses a 33-representation pool against the precompute's 10, and the representation pool is an identity field too.
- **Compute** — caco2, three augmentation conditions: 41,450 + 41,450 + 31,750 = about 114,650 model fits. logd: 41,450 + 34,200 + 31,750 = about 107,400 fits. Lhasa hERG, one condition: 495 importance + 825 allocation + 825 baseline + 40,800 configuration fits = 42,945 fits. Each search is paid once per dataset and read by the other 8 runs. Conditions on caco2, per fold: 5 clusters x 28 fits + 28 fits for the whole-fold build + 6 classifier fits + 5 smoothing values x 5 clusters x 28 fits = 874 fits, so 4,370 fits per run and 39,330 fits across the 9 runs; the 8 augmentation-selection fits inside each build are unverified, and without them the figures are 626 fits per fold and 28,170 across the 9 runs. Conditions on Lhasa hERG: 2,052 fits per fold, 10,260 per run, 92,340 across the 9 runs. Headline: about 153,900 model fits on caco2, about 135,200 on Lhasa hERG. Excessive: three full grids per dataset are built live to read four strings.
- **Hybrids it uses / should use** — 31 hybrids per fold: 5 cluster hybrids at smoothing 0.05, 1 whole-fold hybrid, and 25 more across the 5 smoothing values. Every one is a fresh uncached build at 100 features on one recipe. None is ranked against another by accuracy; the 5 smoothing rows are compared by mean pairwise Spearman between per-representation importance profiles, and only the 5 cluster hybrids and the whole-fold one are scored on held-out molecules. There is no base-representation arm. This rests on distances: KMeans on 2,048-bit Morgan fingerprints defines the clusters, and each held-out molecule goes to the nearest training centroid. It should be testing similarity-objective hybrids — each molecule's activity predicted from its five nearest neighbours — and the 10-feature, 25-feature and 50-feature budgets, since the claim is that a region of chemical space prefers different features. The choice points are fed04:196-205 and `BUDGET = 100` at fed04:46.
- **Selection pattern** — Different shape: 1 downstream classifier family per run and 9 runs, 31 hybrid builds per fold, 0 base representations, and the only scored comparison is 5 per-cluster hybrids against 1 whole-fold hybrid on the held-out molecules.
- **Cut: fast tier** — fed04:196, add `speed='fast'`. The grid goes from 8,160 configurations to 3 allocation methods x 1 importance method x 2 representation counts x 8 budget entries x 9 (model, evaluator) pairs = 432 per condition. On caco2 that turns about 114,650 fits into roughly 8,000; on Lhasa hERG about 42,945 into roughly 3,000. Nothing is lost, because the search is cold on all three datasets either way.
- **Cut: two evaluators** — `FED_MODELS` at `federated_helpers.py:328` crossed with `FED_SEEDS` at `:330` gives 9 runs per dataset; cutting to 2 model families removes about 13,100 of caco2's 39,330 condition fits and 30,800 of Lhasa hERG's 92,340, and leaves the search untouched. The larger cut in this file is `SMOOTH_FACTORS = [0, 0.01, 0.05, 0.1, 0.2]` at fed04:45: the smoothing sweep is 700 of caco2's 874 fits per fold and 1,650 of Lhasa hERG's 2,052, and the 0.05 value reproduces the earlier phase exactly — same clusters, same seed — so 1 of the 5 values is duplicate work, 140 fits per fold on caco2 and 330 on Lhasa hERG.
- **Notes** — ⛔ Three full searches are paid to read four strings: the augmentation strategy, the allocation method, the importance method and the internal evaluator at fed04:218-221. Everything else the 8,160-configuration grid produced is discarded. The cold-search guard cannot protect this file: `KIRBY_NO_COLD_SEARCH` raises inside `find_best_hybrid`, the pool catches it and prints a warning, and fed04:211-213 catches what is left, so the guard downgrades fed04 to the treeshap fallback rather than stopping it. No null control: nothing compares the 5 KMeans clusters against a random partition of the same sizes, so cluster-to-cluster differences in the budget split cannot be separated from subsampling.

### fed05 — Antibody federated

On three antibody sets, does each set's own searched settings beat re-using another set's selected features, and does pooling beat one feature set spanning all three?

- **Hybrids** — Its own, both kinds. `tests/experiments/federated/fed05_antibody_federated.py:256-269` calls `find_best_hybrid` directly, once per antibody set per seed, with `tier='general'`, `adaptive_pruning=0.25` and `cache=True`. `create_hybrid` runs at fed05:169, :465, :701, :811 and inside the helper at `federated_helpers.py:1261`. Objective predictive by default, speed defaults to full. ⛔ No precompute matches, in two ways. The search is handed the 80% training slice while precompute is handed every row, and the label hash is an identity field. And this file loads 49 representations under the `all` preset while precompute uses 8 to 10 names, and the representation pool is an identity field.
- **Compute** — One search: 5 allocation methods x 3 importance methods x 4 representation counts x 8 budget entries x 17 (model, evaluator) pairs = 8,160 configurations, x 5 folds = 40,800 downstream fits, plus 735 importance, 1,225 allocation and 1,225 baseline fits at a 49-representation pool = 43,985 model fits. Searches per dispatch: 3 antibody sets x 3 seeds = 9, because each seed redraws the 80/20 split and moves the label hash; the 3 model families share a seed's search. That is 395,865 fits. The eight conditions cost about 1,330 fits per run at an assumed 40 shared representations and about 11,970 fits across the 9 runs. Headline: about 408,000 model fits per dispatch, 97 in every 100 inside the nine searches. The row counts — 518, 659 and 672 molecules — are read off a comment at fed05:74-81 and are unverified. `adaptive_pruning=0.25` cuts the 40,800-fit term at run time by an amount not measured. Excessive: nine full grids are run to read three strings each.
- **Hybrids it uses / should use** — 1 feature set per (condition, antibody set), built fresh, never read from a stored search. The search contributes exactly 3 strings: allocation method, importance method, internal evaluator. The winning budget, the winning representation count and the selected features are all discarded. Most builds land on `create_hybrid`'s default of 100 features; the sub-cluster condition uses at most 30 features and the per-party condition 33 features. There are 0 individual base-representation baselines in the file. The sub-cluster condition rests on distances: it standard-scales one representation, reduces it to at most 50 PCA components and runs Euclidean KMeans to cut each antibody set into 3 sub-clusters. Whether a hybrid's geometry makes near neighbours share activity is what a similarity-objective hybrid optimises — each molecule's activity predicted from its five nearest neighbours — and the choice points are fed05:256 for a second search and fed05:465 for the per-cluster build. The other five conditions are label-prediction conditions and predictive is right for them.
- **Selection pattern** — Different shape: 1 downstream model per run, 1 hybrid per (condition, antibody set), and 0 base representations scored alone anywhere — the three diversity arms are hand-named 3-representation concatenations, not base representations.
- **Cut: fast tier** — fed05:256-269, add `speed='fast'`. The grid goes from 8,160 configurations to 3 allocation methods x 1 importance method x 2 representation counts x 8 budget entries x 9 (model, evaluator) pairs = 432, and one search from 43,985 fits to 3,875 at a 49-representation pool. Across the nine searches that is about 35,000 fits instead of about 396,000. ⛔ Fast cuts the downstream model list to rf, lightgbm and dnn, so 6 of the 9 runs — the xgb arm and the mlp arm — would use the hardcoded fallback recipe. ⛔ Fast also stops `adaptive_pruning=0.25` acting on the allocation axis, because the allocation vote needs more methods searched than fast searches.
- **Cut: two evaluators** — fed05:256-269 passes no `internal_evaluators=`, so the search takes 4 default names and 17 (model, evaluator) pairs. Passing `internal_evaluators=['rf','lightgbm']` drops it to 10 pairs and 4,800 configurations per search. The larger cut in the same place is the model list: fed05 reads exactly 1 key out of the finished search, and of the 3 names the sweep asks for only rf and xgboost can ever be there. Passing `models=['rf','xgboost']` drops the pairs to 7 and the grid to 3,360 configurations; both cuts together give 4 pairs and 1,920 configurations, just under a quarter of the current grid, with nothing fed05 reads lost.
- **Notes** — ⛔ The mlp arm never gets a searched recipe, because mlp is model tier `extra` and no search with an unset model list scores it; 3 of the 9 runs use the hardcoded performance_weighted / treeshap / match, and the recorded recipe is empty rather than saying so. ⛔ Sequence-identity clusters are honoured in the 80/20 splits, the pooled group vector and the per-party carve, and dropped in the four `create_hybrid` calls at fed05:169, :465, :701 and :811, which pass no groups — so each representation's share of the 100-feature budget is decided against a random 20% carve with identity clusters on both sides of it. The search runs over up to 49 representations and every condition then builds on the intersection across the three antibody sets, so the recipe is searched on one set of representations and applied to a smaller one. The comment at fed05:266-268 says "up to 3,000 recipes"; the grid is 8,160 configurations. `EXPERIMENTS_FINALIZED.md:383` says cache is off and `:5236` says the tier is fast; the code passes `cache=True` and no speed, so both lines are stale.
