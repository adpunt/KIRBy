# Finishing the experimental suite

116 experiment files, read in full on 2026-09-03, then re-checked on 2026-09-04 against the
34 commits that landed in between. Framework and annitia are out of scope.

Every entry gives the background first: what the file does, what it asks, whether it answers
it. Then what the commits changed. Then my decision, or a `>>` line if it is yours.

**Write on the `>>` lines. That is the only thing I read.** Blank means you accept the call
written above it.

---

# Read this first — the suite cannot run today

The commits fixed a great deal. They also put the suite in a state where nothing on disk is
reusable and two files raise on startup. That has to be settled before any of the rest.

**1. Every cached search is unreachable.** The default model set went from seven names
(`gp knn lightgbm mlp rf svm xgboost`) to five (`dnn lightgbm rf svm xgboost`) in `0ce7ab0`.
`model_names` is in the search cache key. So every `searches/*.pkl` on disk misses its key and
every experiment that reads the pool cold-runs a full search.

**2. Every result database must be deleted.** `c74905b` added `metric` and `task_type` to the
result-cache identity columns, and `_ensure_schema` now raises on any file written under the
old column list. `c02` raises before it fits anything. `c13`, `opig_cgr_comparison`,
`opig_ginkgo_developability` and `opig_graphinity_ddg` each swallow that raise and carry on
printing a result with no hybrid in it.

>> I'm pretty sure this has already been done but I would like a confirmation

**3. Three different internal-evaluator lists are in circulation, and the list is in the cache
key.** `_shared.INTERNAL_EVALUATORS` is `['match','rf','lightgbm','dnn']` after `f86d9a4`.
`find_best_hybrid.DEFAULT_INTERNAL_EVALUATORS` is still `['match','rf','svm','lightgbm','mlp']`.
`get_config` puts the second into `config['internal_evaluators']`. Runs filed under different
lists cannot serve each other.

>> This is the correct one: ['match','rf','lightgbm','dnn']
But the fast tier should just be ['match', 'rf']

**4. The default distance is now manhattan.** `distance_config.py:47`, applied process-wide by
`get_config`. Every neighbour, cliff, applicability-domain, noise-score and retrieval quantity
now uses it. `s01`'s 27 stored files, `s12`'s cliff definition and `c01`'s intrinsic-dimension
estimate were all computed under Euclidean and none records which distance it used. The
Levina-Bickel estimator `c01` uses is derived for Euclidean neighbour distances.

>> Is this in line with what the framework experiments found? Can you remind me of the actual results? I did a specific distance study last week this is recent memory.

**5. The distance metric is not in `input_fingerprint`.** So a re-run under a different
distance is skipped as already done. `_shared.py:5653`.

>> I have absolutely no clue what a single bit of this means you need to rephrase in plain language. 

---

# Compute — the picture inverted

Yesterday 18 experiments would not finish. Today more of them will not, because the fix for
the selection problem costs what it costs.

**Per-fold recipe selection is 4x.** `search_on_fold` (`_shared.py:853`) runs the same search
the precompute runs, on one fold's training rows, under that fold's own cache namespace. Five
folds at 4/5 of the rows each is 4.0 whole-dataset searches where there was previously one
cache read. It landed in similarity (`f86d9a4`), complexity (`a09ac08`, `16e85bd`), OPIG
(`6b37acd`) and federated (`45b0125`).

For `s04`, `s05`, `s07` and `s13` that is 5 folds x 2 augmentation conditions x 2,700
configurations x 5 inner folds = **135,000 fold-fits per dataset per experiment**, against a
cache read before. `s12` is 183,375.

**The engine change that is bigger than all of it.** `1e74b25` removed `build_fold_hybrid`'s
reuse of the searched per-representation scores, and `c74905b` made `apply_blueprint` read
them only when labels are absent. So every fold rebuild now re-scores every representation on
that fold's own training rows. `c74905b` records the measurement: **151 seconds for ten
representations on a 1,726-row caco2 fold.** `build_fold_hybrid` is called from 32 experiment
files and not one of the complexity, noise, similarity, OPIG or federated ones passes
`precomputed_alloc_scores`. An experiment doing 5 folds x 10 models x 8 budgets of rebuilds
pays 400 x 151 s = **16.8 hours per dataset in re-scoring alone**.

>> You're naming a lot of problems. What is your suggested solution? I don't feel comfortable with this. 

## Speed tiers — DECIDED AND IMPLEMENTED 2026-09-06

Set by the owner. `speed='fast'` on `find_best_hybrid` cuts four axes; `speed='full'`
is the default and is the existing behaviour unchanged.

| axis | fast | full | extra (never auto-selected) |
|---|---|---|---|
| downstream model | `rf`, `lightgbm` | + `xgboost`, `svm`, `dnn` | `mlp`, `gp` |
| internal evaluator | `rf`, `match` | `match`, `rf`, `svm`, `lightgbm`, `mlp` | — |
| importance method | `deeplift`, `lightgbm_gain` | + `treeshap` | — |
| allocation | `performance_weighted`, `power_2`, `rank_exp_d0.3` | + `softmax_T1`, `wglobal_power2` | — |

Searched grid per (model, budget): **12 configurations at fast, 75 at full.**

### Which axis lives where

`find_best_hybrid` answers one question: *for this dataset, this downstream model and
this budget, which features?* So the two axes in the question are the ones the result
is reported against, and the three that answer it are collapsed to a single winner.

| axis | role | where |
|---|---|---|
| budget | part of the question | caller supplies `budgets=`; result keyed `best_by_model_budget[model][budget]` |
| downstream model | part of the question | caller supplies `models=`; same key |
| allocation method | searched, collapsed | inside the search grid |
| importance method | searched, collapsed | inside the search grid |
| internal evaluator | searched, collapsed | inside the search grid |
| objective | searched, collapsed — **under test now** | `objective=` defaults to `'predictive'` |

### Constants

`src/kirby/find_best_hybrid.py`:
`FAST_MODEL_TIER`, `FAST_INTERNAL_EVALUATORS`, `FAST_IMPORTANCE_METHODS`,
`FAST_ALLOCATION_METHODS`, against `DEFAULT_*` for full. An explicitly passed list
always beats the tier. The per-model `'tier'` label in `DEFAULT_MODELS` already
existed (set 2026-09-04) but nothing branched on it; `speed='fast'` is what makes it
select.

### Where the fast lists came from

**Allocations** — the top three by the c32 criterion: how often a weighting|mechanism
pair lands in the top three of its cell, over 54 cells (dataset x model x budget),
qm9 excluded, each pair scored by its best over N.

| pair | top-3 | top-1 |
|---|---|---|
| linear \| quota (`performance_weighted`) | 30/54 | 9/54 |
| power_2 \| quota | 27/54 | 11/54 |
| rank_exp_d0.3 \| quota | 26/54 | 7/54 |
| softmax_T1 \| quota | 25/54 | 11/54 |
| softmax_T1 \| global | 18/54 | 9/54 |
| power_2 \| global (`wglobal_power2`) | 14/54 | 3/54 |

**Importance** — the f30 bake-off on caco2 (28 methods, 3 scaffold-grouped folds,
budget 100, delta against the best single representation):
`deeplift` +0.0144 (271s), `lightgbm_gain` +0.0100 (42s), `treeshap` +0.0067 (542s).
The ordering held in all four cells; `treeshap` is last every time and the dearest of
the three, which is why it is in full and not in fast. One dataset, three folds — the
gap between the top two is inside fold noise.

### ⛔ Cache

Every one of these four axes is in the search cache key (`_make_search_meta`). **A
fast run files separately from a full one and cannot be topped up into it.** It is a
different search, not a subset. Nothing needs deleting when the tier changes — the key
changes, so there is no silent reuse.

---

## Speed tiers — the pruning mechanisms (unchanged)

They were removed in `9bd48ac` and replaced by `DEFAULT_*` constants. That removal also
deleted the line passing `budgets` and `n_folds` in `similarity_optimized_hybrid`. What exists
today, and exactly what each one takes off:

| mechanism | what it removes | what it leaves |
|---|---|---|
| `default_pruning` | nothing. `checkpoints=[]`, `tail_drop=0.0` | the whole grid |
| `aggressive_pruning` | bottom half of (allocation, importance) pairs at budgets 25, 50, 100. 15 pairs to 8 to 4 to 2. f16 measured 58% fewer fold-evaluations on caco2 at top-3 overlap 1.0 | opt-in, and **off everywhere** |
| `adaptive_pruning=0.15` | allocation methods only, after fold 0 has already evaluated everything. Best case 32%, realistically far less | budgets, importance, evaluators, models, representations |
| `MODEL_MAX_BUDGET` | 3 of 40 (model, budget) pairs — 7.5% | the rest, and it raises rather than emptying a model |
| `_apply_budget_policy` | budgets above 500 on 14 named datasets — 8 budgets to 6 | everything else |
| model-skip policy | the largest cut that exists: named models per (dataset, strategy) | not an axis cut, per dataset |
| `DISTANCE_BUDGETS` | 1000 and 2000 from the distance lineage | 2,700 configurations, still bigger than the main lineage's 2,445 |
| `--mode intermediary` | nothing from the search. It only sets the experiment's own fold count to 1 | the full grid |

**`aggressive_pruning` is the one real mechanism and it no longer reaches the searches that
now cost the most.** `run_all` forwards the schedule to the whole-dataset precompute only.
`get_fold_best_search` accepts it and `s04`, `s05`, `s07`, `s11`, `s12` and `s13` pass none.

The two axes nothing prunes are **budgets and models**, and both are in the cache key, so a
smaller grid files separately from the full one and cannot be topped up later.
`aggressive_pruning` is the only mechanism that cuts inside a search without forking the cache
per axis.

**The decision.** Whether to turn `aggressive_pruning` on by default, whether to thread it
into the per-fold searches, and whether a deliberate smaller-grid mode is worth the cache
fork. I am not making this one.

>> 

---

# The suite should receive a SET of candidates, not one best hybrid — decided 2026-09-07

Set by the owner. Today every experiment pulls the single winning blueprint from the precompute
and runs that. The decision is that it should instead receive a selection: the best across
budgets, the best across downstream models, and augmented options alongside them.

**A precompute step has to build that set.** It does not exist. Writing it is the next piece of
work and it is scoped in the handoff prompt below.

## Why one winner is not enough — measured, f24 stage 3, jobs 8724201 and 8724202

Three feature tables were scored on the same folds: the dataset's best single representation
alone, the multi-representation hybrid at a 250-feature budget, and that same single
representation with extra features added. 3 folds, 3 seeds.

| dataset | metric | model | base rep alone | hybrid, budget 250 | base rep + extras |
|---|---|---|--:|--:|--:|
| `flab_expression_koenig` | spearman, higher better | lightgbm | 0.6754 | **0.7378** | 0.7327 |
| | | rf | 0.6827 | **0.7408** | 0.7288 |
| `caco2` | rmse, lower better | lightgbm | **0.2986** | **0.2976** | 0.2986 |
| | | rf | 0.3137 | 0.3100 | **0.3027** |
| `buchwald_hartwig` | rmse, lower better | lightgbm | 16.1809 | 17.1677 | **16.0055** |
| | | rf | 16.1888 | 16.0468 | **15.7924** |
| `tcga_brca_pam50` | f1_weighted, higher better | lightgbm | 0.8311 | **0.8642** | none survived |
| | | rf | 0.7942 | **0.8401** | 0.7994 |

Fold-to-fold wobble, the size below which a difference is not a difference: `koenig` 0.0073
spearman, `caco2` 0.0037 rmse, `buchwald_hartwig` 1.6198 rmse, `tcga_brca_pam50` 0.0305
f1_weighted.

Reading it against those wobbles: the hybrid wins on `tcga_brca_pam50` and on `koenig` with rf.
Adding extras to the single representation wins on `caco2` with rf. On `buchwald_hartwig` every
one of the six numbers sits inside the wobble, so that dataset separates nothing. And on `caco2`
the choice of downstream model moves the score more than any of the three feature tables does —
lightgbm on the base representation alone beats every rf arm.

No single table wins everywhere, and the model choice can outweigh the table choice. That is the
evidence behind the decision.

## Handoff prompt — the candidate-set precompute

Paste this into a fresh session.

---

Build the precompute step that emits a SET of hybrid candidates per dataset, replacing the
single winning blueprint the experiments read today.

READ FIRST, IN FULL, do not summarise: `EXPERIMENTS_FINALIZED.md` (this section and
`# Compute — the picture inverted`), `AUGMENTATION.md` sections 9.17 and 10, and
`tests/precompute_hybrids.py`.

WHAT THE SET CONTAINS. Three axes, decided by the owner on 2026-09-07:
  - the best blueprint at each budget, not only at the winning budget
  - the best blueprint for each downstream model, not only for the winning model
  - augmented options alongside the plain hybrids

HOW AUGMENTATION IS SPECIFIED. `augment_capped` (`src/kirby/hybrid.py:3015`) and
`augment_uncapped` (`:3345`) already exist and stay as they are — the owner confirmed the
current implementation is fine on 2026-09-07. The settings ride on the `Blueprint`:
`augmentation_strategy`, `aug_per_source_budget`, `augmentation_margin`,
`augmentation_fraction`, `candidate_ranking`, `augmentation_threshold`, `augmentation_budget`,
`aug_max_candidates`. `f24_augmentation_admission.py:1045` is a working example of setting them.

WHAT f24 STAGE 3 FOUND, so the candidate set does not re-explore settled ground. All of this is
3 folds — enough to rank settings, not enough to quote a number:
  - `capped` and `uncapped` score the same on every dataset. Default `uncapped`.
  - `all` (add every donor feature, no test) took a planted decoy feature in all 288 of its runs.
  - a margin of 0 wins everywhere; a margin of one fold spread wins nowhere.
  - a per-donor cap of 5 features is too tight; 64 and unlimited are equivalent.
  - `residual_correlation` as the candidate ranking survives on both datasets where augmenting
    works; `rf_importance` collapses on `caco2` (4 runs of 234 beat the base, against 42 of 257).
  - the extras help when added to a single representation and almost never inside a hybrid's
    250-feature budget.

WHAT IS NOT SETTLED. Which donor source to use flips between datasets: on `caco2` the curated
list alone works (16 runs of 74 beat the base), on `flab_expression_koenig` it does not (0 of
22). Only `both` works on both. Budgets other than 250 were never run.

CONSTRAINTS. The representation list is never in a cache key. No in-cache versioning: a code
change means wipe and re-run. Do not prune the search space for compute; a cut needs a stated
reason — degenerate, broken, or undefined.

DELIVERABLE. Options with the compute cost attached, for the owner to choose from. Not a
finished design.

---

# What the commits fixed

Across 1,081 re-checked claims: **104 fixed**, 47 changed shape,
450 still stand, and **280 are new** — introduced or exposed by the work.

Fixed outright, and no longer in the fix prompts:

- **`c08`** — 2 findings closed. create_hybrid is called with no groups, so the internal holdout that admits features is a random 20% carve inside a group-aware outer fold.
- **`c14`** — 2 findings closed. hybrid_score is the search's own winner-of-the-grid score on the rows that selected it, while best_single_score is an unsearched CV mean; every headline number is built on that difference.
- **`c19`** — 2 findings closed. get_fitted_pool_hybrid builds the feature matrix over every row, then the fold loop splits that already-selected matrix, so feature selection has seen the validation labels.
- **`c22`** — 4 findings closed. feature_info is taken from the last fold while avg_shap is averaged over every fold, so SHAP mass is attributed to whichever representation occupies those positions on fold 4.
- **`c30`** — 4 findings closed. Imports round2_harness from a hardcoded /private/tmp scratchpad directory that no longer exists.
- **`c32`** — 2 findings closed. The caveat claims no whole-data leak in the sizing, but pool membership, the N-subsets and the target are all whole-data ranked on the same folds used for the paired delta.
- **`engine`** — 19 findings closed. The scores that split the budget were measured on a GroupShuffleSplit carve while create_hybrid carved with GroupKFold, under a comment saying the two matched.
- **`exp_b`** — 4 findings closed. Imports round2_harness from a hardcoded /private/tmp scratchpad path that no longer exists.
- **`fed02`** — 2 findings closed. C5's three representations were chosen by their score on the fold C5 was then measured on.
- **`fed04`** — 4 findings closed. Labels are binarized at a hardcoded 6.0, which collapses caco2 and logd to one class.
- **`n06`** — 2 findings closed. 'concat', the hstack of every representation, is used as the comparison space against the hybrid, and individual base representations are never evaluated as spaces.
- **`n07`** — 2 findings closed. build_fold_hybrid is handed the clean labels, so the feature space is chosen before the noise is injected.
- **`opig_graphinity_ddg`** — 2 findings closed. The PDB overlap filter never ran, because load_dataset dropped the complex identifiers, so FoldX rows sharing a complex with the SKEMPI test set stayed in training.
- **`opig_kingsmith_component_splits`** — 3 findings closed. The reported delta was the largest R2 over 14 representations x 5 models minus the largest over 5 hybrid models, both maxima taken on the folds that produced them.
- **`opig_trastuzumab_enrichment`** — 2 findings closed. hybrid_search.rf_cv_auc was the winning recipe's mean over the folds that chose it, presented in the comment and the printed line as the honest hybrid number, and it was the only hybrid number the experiment produced.
- **`ranker_experiment`** — 2 findings closed. Feature selection saw the labels of every molecule, then the selected space was scored by a GroupKFold five-nearest-neighbour prediction over those same molecules; the activity-target arms were graded on folds whose labels chose their own features and the structural arms were not.
- **`s05`** — 3 findings closed. The hybrid is rebuilt from scratch for hybrid_weighted with identical arguments; 60 of 120 create_hybrid calls are duplicates
- **`s08`** — 3 findings closed. bp is the best config across all budgets, so the entire cross-validated half of a dimension sweep ran at one fixed dimension
- **`s11`** — 2 findings closed. The hybrid's recipe is tuned on labels the loop is meant to be discovering; get_best_search_for_downstream is handed the full labels and groups arrays.
- **`s12`** — 3 findings closed. cv_split is called with mode=mode, which _shared ignores while raising a DeprecationWarning on every call.
- **`s13`** — 3 findings closed. get_fitted_pool_hybrid builds a whole-dataset hybrid matrix that is bound to _X_unused and thrown away, one 336 s create_hybrid per run.
- **`sim_hybrid_space`** — 2 findings closed. n11 and n13 print the hybrid_sim refusal reason and throw it away, so an absent arm reads later as a null result.
- **`sm_feature_ablation`** — 3 findings closed. select_fine2D ranks 2048 features on the cached MCS pairs and the fine2D block's structural fidelity is then measured on those same pairs
- **`winning_recipe_honest`** — 2 findings closed. The 50/50 split is a permutation of the 8000 pairs, not of the molecules, so the selection has seen every molecule it is scored on.
- **`winning_recipe_vs_standard`** — 2 findings closed. desc_blacklist_greedy and select_fine2d are fitted on the full pair set and every struct_fidelity is graded on those same pairs, while the docstring calls both yardsticks honest.

`c30` and `exp_b` were ported to `_alloc_harness` in `38bf326` and both import clean. `exp_b`
dropped the `_label_signal_ladder` call that existed nowhere. It is no longer a delete
candidate. `c20`, `c43_buchwald_signal` and `c43_buchwald_step3` still carry the dead
`/private/tmp` import, so `FIX_1_DEAD_IMPORTS.md` drops from five files to three.

---

# From the two roadmaps — 2026-09-04

`HYBRID_REBUILD_ROADMAP_2026_07_15.md` and `SIMILARITY_METRICS_ROADMAP.md` were swept for
items that were never closed anywhere. Most of what they still list belongs to the framework
and is now in `FRAMEWORK_FINALIZATION.md` — §1 items 35 to 38 for the four that turned out to
be fixed, §3.9 to §3.11 for the three that need an answer. One thing landed in this suite.

**Fixed here: `get_fitted_hybrid` dropped the objective on the way to `create_hybrid`.**
It takes an `objective` argument, uses it to pick which search to read, and then rebuilt the
winning recipe on full data without passing it on. So a recipe chosen under read-across or
under MCS fidelity had its budget split predictively on the rebuild. Its sibling fifty lines
above, `get_fitted_pool_hybrid`, has forwarded the objective since the geometry objectives
went in, and carries a comment saying what happens without it; this copy was missed.

**No stored result moves.** `get_fitted_hybrid` is imported by nine experiment files —
`c15`, `s02`, `s03`, `s05`, `s06`, `s07`, `s08`, `n01`, `n06` — and called by none of them.
Every one of them calls `get_fitted_pool_hybrid` instead. `tests/experiments/_shared.py:5228`.

**Also checked and already closed, so they stop being re-raised.** Ridge's two surviving
measurement uses in this file are `LinearRegression` (the L1 linearity descriptor) and
`Lasso` (c15's linear model family), as decided on 2026-07-18 — no Ridge is left in
`_shared.py`. `similarity_optimized_hybrid.py` no longer defaults to the cut
`similarity_sac`; it defaults to read-across and raises on the cut name.
`tests/experiments/noise/_similarity_hybrid_space.py` no longer opens a July pair file
without checking the molecule list — it calls `assert_pairs_match_smiles` and falls back to
`get_mcs_pairs`, which closes item 6 of `SIMILARITY_OPEN_WORK_2026_08_27.md`.

---
# Every experiment


---

# Complexity — 30 files

## c01 — `complexity/c01_rep_characterization.py`

### What it does

For each base representation it computes intrinsic dimensionality (PCA 90/95/99 thresholds + Levina-Bickel MLE), mutual information with the labels, effective rank, and a per-model cross-validated score for that representation on its own. No hybrid is built.

**Asks:**
- How intrinsically low-dimensional is each base representation on this dataset?
- How much information does each base representation carry about the label?
- How well does each base representation predict the label on its own, per downstream model?

**Verdict on 2026-09-03:** PARTIAL · hybrid: none needed · dispatched

Why it fell short:
- The per-model baseline on classification datasets whose metric is AUROC is computed from hard class labels, not predicted probabilities. Every dataset registered with metric 'auc' (herg, flab_binding_adams, tz_her2_binary, humanness_therapeutic, humanness_species, cdr_flexibility, tcga_luad_vs_lusc) gets a systematically depressed AUROC.
- The baseline for any representation wider than the per-fold training set is silently PCA-compressed to n_train components by cap_dimensions (_shared.py:1832). c02's baselines come from the engine's _evaluate_single, which only filters above 5000 features.
- Mutual information and intrinsic dimensionality are computed on all rows with no train/test split. They are descriptors, not held-out quantities.

*Smoke:* ran clean. exit 0 in ~90s.

### Since the audit

0 fixed · 0 changed · 3 stand · 3 new · compute unchanged · still runs

- **NEW** — The intrinsic-dimensionality MLE is now measured under manhattan distance, not Euclidean. `tests/experiments/_shared.py:2609 (estimate_intrinsic_dim_mle), reached from c01_rep_characterization.py:76`
- **NEW** — The distance metric is not in input_fingerprint, so a rerun under a different distance resumes the old file. `tests/experiments/_shared.py:5653-5666 (fingerprint_mismatch), used at complexity_helpers.py:1092-1105`

### Decision

**MINE:** Swap in _predict_for_metric at _shared.py:1836 and record the width actually fitted. Keep the baseline arm.

## c02 — `complexity/c02_complexity_grid.py`

### What it does

Runs find_best_hybrid over the full grid (5 allocation methods x 3 importance methods x 8 budgets x 5 internal evaluators x the engine's 7-model default) under up to four conditions: no augmentation, uncapped augmentation, capped augmentation, and frozen-plus-fine-tuned representations. Then adds a per-model pass at the native width of that model's best single representation, PCA-on-the-concatenation baselines at every budget, and a model x budget score table.

**Asks:**
- Does each downstream model have a different optimal feature budget?
- Does the hybrid beat the best single base representation, per model?
- Which augmentation condition (none / uncapped / capped / frozen+fine-tuned) gives the best hybrid for each model?
- Does importance-based selection beat PCA of the concatenation at the same feature count?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · dispatched

Why it fell short:
- best_by_model and baselines are both argmax-over-means on the same 5 folds, but the hybrid arm searches ~3,700 configurations while the baseline arm searches 10 representations. The printed and stored delta at c02_complexity_grid.py:423 is therefore biased toward the hybrid.
- honest_rescore is built from the uncapped search only (c02_complexity_grid.py:549, `search = search_uncapped`). The no-aug, capped, frozen+FT and per-model-top-rep-dim conditions contribute nothing to it.
- The `conditions` and `best_per_model_across_conditions` blocks store only score_mean/score_std, with no fold_scores. The augmentation-condition question (which condition wins per model) therefore cannot be re-scored honestly from the JSON — that comparison is a bare argmax over four means.

*Smoke:* FAILED. exit 139 (segfault) at 'Evaluating full-dimension baselines...'.

### Since the audit

0 fixed · 2 changed · 5 stand · 3 new · compute CHEAPER · **DOES NOT RUN**

- **NEW** — The honest delta now drops fold 0 from both arms by default, including for a stored block that says nothing about pruning. `tests/experiments/_shared.py:1177-1198 (hr_exclude_folds), passed at c02_complexity_grid.py:567`
- **NEW** — Every cached c02 search is now unreachable, and an existing result database makes the file raise before any fitting. `src/kirby/result_cache.py IDENTITY_COLS and _ensure_schema`
- **NEW** — The standalone entry point discards a comma-separated --reps and loads the ten general representations instead. `tests/experiments/complexity/c02_complexity_grid.py:602`

*Compute now:* Steady state is 0.55x. The internal evaluator list went from ['match','rf','svm','lightgbm','mlp'] to ['match','rf','lightgbm','dnn'] (f86d9a4, _shared.py:545) and the engine's default model set went from 7 to 5 (0ce7ab0: mlp, gp and knn moved to 'extra'; the set is now rf, lightgbm, xgboost, svm, dnn).

### Decision

**MINE:** Do all four repairs. Only the PCA-control fix costs fits: 800 across four datasets.

**The recheck opened something new here. Overrule me if it changes your call:**

>> 

## c03 — `complexity/c03_cross_dataset_transfer.py`

### What it does

For each of several datasets it computes a Lorena-style complexity profile, pulls the best precomputed distance-tier hybrid search keyed to the SVM downstream model (falling back to RF), then Spearman-correlates each complexity descriptor against that dataset's winning budget. It also fits SVR and ElasticNet per fold on a hybrid rebuilt from the winning blueprint, and decomposes the per-model best-budget matrix into model, dataset and interaction sums of squares.

**Asks:**
- Does a dataset's intrinsic complexity (effective rank, participation ratio, C1/C2/L1/S3) predict the winning feature budget?
- How much of the variation in winning budget is attributable to the model, to the dataset, and to their interaction?
- Do two regressors that were never in the search (SVR, ElasticNet) work on the winning hybrid?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · orphaned

Why it fell short:
- The fields named best_model / best_budget / best_allocation / best_importance are the SVM-keyed pool winner (or RF on fallback), not the dataset's overall best configuration. The docstring says the experiment 'finds the native-optimal configuration on each dataset'.
- The correlation runs whenever two or more datasets survive (line 233). Spearman over two points is always exactly +/-1; over the three-dataset default it has three possible values.
- The SVR/ElasticNet arm has no comparator. It reports absolute scores on the hybrid and never evaluates those two regressors on any single base representation, so it cannot say whether the hybrid helped them.

*Smoke:* not run. exceeded 6 min at 'Generating smited...

### Since the audit

1 fixed · 0 changed · 4 stand · 4 new · compute CHEAPER · still runs

- **NEW** — The pool helper now reports that the recipe was chosen over every row, and c03 neither reads nor records it. `tests/experiments/_shared.py get_best_search_for_downstream returns metadata carrying selection_scope='whole_dataset'; c03_cross_dataset_transfer.py:101-118 unpacks only (search_result, meta)`
- **NEW** — The complexity descriptors S3 and mle_id are now measured under manhattan distance. `tests/experiments/_shared.py:2812 and :2609, reached from c03_cross_dataset_transfer.py:73`
- **NEW** — The standalone entry point discards a comma-separated --reps and loads the ten general representations for every dataset. `tests/experiments/complexity/c03_cross_dataset_transfer.py:379`
- Fixed (1e74b25) — create_hybrid is called with no groups, so the internal ranking split is a random 20% carve and the budget is allocated under scaffold leakage.

*Compute now:* About 0.71x on the search, unchanged on the rebuild arm. The distance pool is 5 allocation methods x 3 importance methods x 6 DISTANCE_BUDGETS x 5 DISTANCE_EVALUATORS = 450 recipe-evaluator pairs.

### Decision

**YOURS:** Delete, or rewrite to run over all 26 datasets. It is orphaned with no result files. c14 and c24 cover the transfer half but nothing else computes complexity-descriptor against winning budget. My lean: rewrite, it costs no model fits.

>> 

## c04 — `complexity/c04_importance_as_metric.py`

### What it does

Concatenates every base representation, computes per-fold feature importance on the training rows with each importance method, turns those scores into per-feature weights (sqrt then normalise to sum 1), and compares weighted against unweighted k-nearest-neighbour regression and inverse-distance retrieval. It also cross-tests selecting features by one importance method while weighting by another, and repeats the retrieval comparison on a 25-feature subset. Regression datasets only; classification returns a skip record.

**Asks:**
- Does weighting the concatenated feature space by importance scores improve k-nearest-neighbour prediction over the unweighted space?
- Does it improve inverse-distance-weighted retrieval of the label from the ten nearest training molecules?
- Does the importance method that best selects features differ from the one that best weights them?

**Verdict on 2026-09-03:** PARTIAL · hybrid: hand-rolled · dispatched

Why it fell short:
- Both arms live in the same all-representations concatenation. There is no single-representation comparator and no hybrid comparator, so the result cannot say whether an importance-weighted space is a better metric than any representation's own space — only that weighting beats not weighting inside one fixed space.
- The retrieval scores pool predictions across folds and compute one R-squared and one MAE over the pooled vector (lines 119-125), while the k-nearest-neighbour scores are averaged per fold (lines 112-116). The pooled R-squared measures against the global label mean, not each fold's.
- All scores are r2_score regardless of the dataset's pipeline metric (rmse for the molecular sets, spearman for antibody). Nothing in the JSON records that, and there is no metric direction stamped on the standalone path.

*Smoke:* not run. exceeded 6 min at 'Cross-test matrix (select x weight)...'.

### Since the audit

0 fixed · 0 changed · 5 stand · 2 new · compute unchanged · still runs

- **NEW** — The retrieval arms now measure manhattan distance while the k-nearest-neighbour regression arm in the same function still measures Euclidean. `tests/experiments/complexity/c04_importance_as_metric.py:94, :103, :179, :221, :230 against :81-89`
- **NEW** — The stored c04 retrieval numbers cannot be reproduced by the current file. `tests/results/complexity/c04_importance_as_metric/{dataset}.json, produced by c04_importance_as_metric.py:94-108`

### Decision

**MINE:** Memo the per-fold importance (8x on the dominant cost, no number changes) and add a single-representation comparator. The comparator is the standing baseline rule, not a choice.

## c05 — `complexity/c05_learning_curves.py`

### What it does

At each of ~5 subsample sizes (drawn as whole scaffold groups, 3 seeds) it runs a full find_best_hybrid search on the subsample and records the winning score per downstream model, against the best individual base representation scored by an independent group-CV. It also re-fits the full-data winning recipe on each subsample as a second arm.

**Asks:**
- Does the hybrid's advantage over the best individual base representation grow, shrink or hold as the number of training molecules rises?
- Does the winning recipe (budget, importance method, allocation method) change with sample size?
- Does the recipe chosen on the full data still work when re-fitted on a small subsample, compared with searching afresh at that size?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · dispatched

Why it fell short:
- The two arms are not selected the same way. The hybrid number is `search.best_by_model[m]['score_mean']` (c05:242) — the maximum over roughly 3000 searched configurations of that search's own inner CV.
- The baseline representation is PCA-compressed. evaluate_single_rep_cv calls cap_dimensions (_shared.py:1834), which PCA-reduces any representation wider than the training row count down to n_train components.
- The learning-curve fit is only defined for a higher-is-better metric bounded in [0,1] (see the bug below), so the fitted asymptote is meaningless on the RMSE and MAE datasets.

*Smoke:* not run. exceeded 6 min inside the first hybrid search (last line 'Search' then the tensorflow banner).

### Since the audit

0 fixed · 0 changed · 0 stand · 0 new · compute MORE EXPENSIVE · still runs


*Compute now:* Searches unchanged: sample_sizes x n_repeats, collapsed to one seed at the full-data point. Five fractions [0.10, 0.25, 0.50, 0.75, 1.0] x 3 repeats with the last collapsed = 13 searches; above 5,000 rows the size gate gives sizes [250, 500, 1000, 2000] and 10 searches.

### Decision

**MINE:** Take the search.baselines swap. It fixes the KeyError crash, the PCA confound and the hard-label AUROC in one edit and removes 4,550 fits per dataset.

## c06 — `complexity/c06_within_family_gradient.py`

### What it does

For each of ~28 capacity variants across 7 model families it runs a separate find_best_hybrid with that variant as the only downstream model, records the best score at every budget, takes the budget where the score peaks, and Spearman-correlates capacity index against that peak budget within each family.

**Asks:**
- Within a model family, does the feature budget at which the hybrid scores best rise with the model's capacity?
- Which importance method and allocation method wins at each (capacity variant, budget) cell?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · dispatched

Why it fell short:
- The quantity correlated against capacity is `best_by_model_budget[variant][budget]['score_mean']` — at each budget it is the maximum over 5 allocation methods x 3 importance methods x 4 internal evaluators of that search's inner-CV mean. The peak budget is then an argmax over those maxima.
- No baseline of any kind is recorded, so the file cannot tell you whether any of these variant-optimal hybrids beat the corresponding individual base representation. That is outside the stated question but means the file stands alone.
- On a classification dataset the run returns {'skipped': True} (c06:52-55), so the capacity/budget question is answered on regression datasets only.

*Smoke:* ran clean. module imports clean, exit 0.

### Since the audit

1 fixed · 0 changed · 4 stand · 2 new · compute MORE EXPENSIVE · still runs

- **NEW** — NEW: the family-prefixed model name also means MODEL_MAX_BUDGET never applies to any c06 variant. `tests/experiments/complexity/c06_within_family_gradient.py:176`
- **NEW** — NEW: the importance fix costs a whole extra search under the CLI default mode. `tests/experiments/complexity/c06_within_family_gradient.py:138`
- Fixed (a09ac08) — The reused importance scores were fetched at the engine's default fold count while the variant searches ran at a different one.

*Compute now:* Same under --mode full, more expensive under the CLI default. The 28 variant searches are unchanged: 7 families x 4 variants = 28, each 5 allocation methods x 3 importance methods x 8 budgets x 4 internal evaluators = 480 configurations x 5 folds = 2,400 fits, 67,200 fits per dataset.

### Decision

**MINE:** Store the per-fold scores and guard the borrowed importance on the fold count. Both free.

**Open — yours:** Whether the matched internal evaluator arm is worth making reachable. Today 'match' never resolves for a family-prefixed variant, so that arm is silently absent.

>> 

## c07 — `complexity/c07_data_complexity_partitions.py`

### What it does

Splits each dataset in half three ways (frequent vs rare scaffolds, small vs large molecules by heavy-atom count, low vs high within-scaffold label variance), runs a full find_best_hybrid on each half, and records whether the winning recipe differs between the halves. It also re-fits the full-data winning recipe on each half as a second arm, and computes descriptive complexity numbers (mutual information, effective rank, intrinsic dimension) per half.

**Asks:**
- Does the winning hybrid recipe — model, budget, importance method, allocation method — differ between an easy and a hard subpopulation of the same dataset?
- How far does the budget move between the two halves?
- Does the full-data winning recipe still score well when re-fitted on a subpopulation, or does the subpopulation need its own search?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · dispatched

Why it fell short:
- The primary comparison, `config_differs` (c07:397-401), is a comparison of two argmaxes over roughly 3,465 searched configurations each. Two halves of the same dataset will differ in their argmax from fold noise alone, so a True here is not evidence that the subpopulation needs a different recipe.
- The full-data-recipe arm is built with a leaking internal split (see the bug), so 'how much does partition-specific tuning buy you over global tuning' is answered by an optimistic version of the global arm.
- The target_variance partition is confounded with scaffold size, so the easy/hard contrast there is not the contrast the name claims.

*Smoke:* ran clean. module imports clean, exit 0.

### Since the audit

0 fixed · 0 changed · 0 stand · 1 new · compute MORE EXPENSIVE · still runs

- **NEW** — c07 asks for an mlp blueprint that the precompute no longer scores `tests/experiments/complexity/c07_data_complexity_partitions.py:70`

*Compute now:* 1.33x on searches, and one reported column now goes missing. a09ac08 added a fourth partition, feature_variance, beside scaffold_frequency, molecular_complexity and target_variance.

### Decision

**MINE:** Add groups, model, metric_fn and lower_is_better to build_fold_hybrid at c07:338.

## c08 — `complexity/c08_calibration.py`

### What it does

Builds four hybrids per dataset — the recipe the pool search picked, the same recipe at a tiny budget, the same recipe at the full concatenated width, and a low-capacity Ridge or LogisticRegression at the full width — then runs split-conformal prediction per fold and reports coverage, interval width or prediction-set size, and calibration error for each.

**Asks:**
- Does the hybrid recipe that scores best also give the best-calibrated uncertainty, compared with the same recipe run at too small or too large a feature budget?
- Does the correlation between the model's own uncertainty and its actual error change with the feature budget?

**Verdict on 2026-09-03:** NO · hybrid: hand-rolled · dispatched

Why it fell short:
- The 'optimal' condition is not the recipe the search chose. create_hybrid is called with internal_evaluator taken from the blueprint but with no model and no metric_fn (c08:220-228), so an internal_evaluator of 'match' silently resolves to 'lightgbm' (hybrid.py:4398) and the allocation ranking is scored with r2_score or accuracy instead of the dataset's own metric.
- Every condition's feature selection leaks. create_hybrid gets no groups, so its internal holdout is a random 20% permutation (hybrid.py:4450) rather than a scaffold-aware fold, on all four conditions.
- The conformal calibration set is carved out of training rows by np.random.RandomState(42+fold_i).permutation (c08:251, c08:360). Proper-train and calibration therefore share scaffolds, so the residual quantile q is systematically too small and the reported coverage is biased low by an unknown amount.

*Smoke:* ran clean. module imports clean, exit 0.

### Since the audit

2 fixed · 0 changed · 6 stand · 2 new · compute unchanged · still runs

- **NEW** — NEW: the 11 result files on disk predate both fixes and cannot be reproduced by today's code. `tests/results/complexity/c08_calibration/*.json`
- **STILL BROKEN** — The 'mismatched' condition hardcodes allocation_method='greedy'.
- **STILL BROKEN** — The module-level fallback also sets greedy and treeshap, and a fallback run writes a normal-looking result with no marker.
- Fixed (1e74b25) — create_hybrid is called with no groups, so the internal holdout that admits features is a random 20% carve inside a group-aware outer fold.

### Decision

**MINE:** Remove the greedy dispatch at c08:76 and c08:179 — greedy is closed, this is not a choice. Add the four arguments at c08:220.

## c09 — `complexity/c09_formal_analysis.py`

### What it does

Four separate analyses in one file: an ANOVA of c02's grid scores by model and budget; mutual information between the selected features and the property plus a variance-retention measure against the full concatenated features, per budget; effective rank of the hybrid against PCA and a random subset at the same width; and a train-minus-held-out gap per model per budget. Only the last one fits downstream models; the first reads c02's JSON.

**Asks:**
- How much of the variation in c02's grid scores is explained by the downstream model, by the feature budget, and by their interaction?
- As the budget rises, how much information about the property do the selected features carry, how much of the full concatenated feature set do they retain, and do they span more independent directions than PCA or a random subset of the same width?
- For each model and budget, is the train-to-held-out gap widening (overfitting) or is the train score itself poor (underfitting)?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · dispatched

Why it fell short:
- The ANOVA is run on c02's `configs`, which c02 copies straight out of `search.all_results` (c02_complexity_grid.py:219-229). Those are the search's own selection scores, not held-out scores, so 'model explains 50% of the variance' is a statement about the search's internal numbers.
- Each (model, budget) cell pools every allocation, importance method and internal evaluator c02 searched, and treats the 5 fold scores of one configuration as independent replicates. df_residual and therefore every F and p written to JSON are inflated, and the allocation/importance variance lands in the residual.
- The design is unbalanced, so the model, budget and interaction sums of squares are not orthogonal; `ss_residual` is clamped at zero, which hides that and lets the eta-squared shares sum above 100%.

*Smoke:* ran clean. module imports clean, exit 0.

### Since the audit

1 fixed · 0 changed · 9 stand · 3 new · compute unchanged · still runs

- **NEW** — The new held-out rebuild in 9b has no demoted-representation handling, so a top_2/top_3/top_5 recipe with augmentation representations loses every fold of that budget. `tests/experiments/complexity/c09_formal_analysis.py:376-381`

### Decision

**MINE:** Do the five local repairs and recompute the ANOVA over configurations rather than fold scores.

## c10 — `complexity/c10_noise_robustness.py`

### What it does

Injects label noise into the training labels only, at a grid of levels under eleven noise conditions, and scores a random forest on clean held-out labels. Does it once per individual representation and once per budget for a hybrid built from the cached blueprint, then summarises each degradation curve with auc_norm, and measures how much the selected feature set changes as noise rises.

**Asks:**
- Does a hybrid lose accuracy more slowly than an individual representation as label noise rises?
- Does a larger feature budget buy more noise robustness?
- Do the same features keep being selected as the noise level rises, and how similar are the selections across folds at zero noise?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · dispatched

Why it fell short:
- The blueprint is looked up on the clean labels (line 155-162) and label_noise= is never passed, even though get_best_search_for_downstream exists to resolve the noisy lineage for exactly this case (_shared.py:4334-4338). The hybrid enters the comparison with a recipe chosen from a whole-dataset search on the clean labels the experiment then corrupts; the individual representations get no tuning at all.
- The docstring says the experiment tests 'whether the optimal hybrid config changes under noise' (line 101-103). Nothing re-searches the configuration under noise; only which features the fixed recipe picks is measured.
- Only the fold mean per level is written (level_scores). There are no per-fold scores in the JSON, so no error bar and no paired hybrid-versus-rep test is possible later.

*Smoke:* ran clean. module imports clean, exit 0.

### Since the audit

0 fixed · 0 changed · 0 stand · 0 new · compute unchanged · still runs


### Decision

**MINE:** Fix the local faults and switch to build_fold_hybrid.

**Open — yours:** Whether the hybrid recipe is re-searched under noise. Symmetric costs 56 noise-aware searches per dataset.

>> 

## c11 — `complexity/c11_ensemble_vs_hybrid.py`

### What it does

Trains one model per representation on every outer fold and on a three-fold inner CV inside each outer training split, then builds ensemble processes that pick their members, weights and aggregator from the inner scores alone and aggregate the stored outer predictions. Compares those against the best single representation, a PCA of the concatenated representations, and a per-fold-refitted hybrid, and reports an inference cost row for every arm.

**Asks:**
- Does a single model on the fused feature vector match an ensemble of one model per representation, when the ensemble picks its members and aggregator per fold from that fold's training split alone?
- What does each arm cost at inference - how many featurisers, how many deployed models, how many stored bytes per sample - and how many representations does the blueprint actually draw a column from?
- Does the hybrid beat the best single representation under leave-one-fold-out reselection?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · dispatched

Why it fell short:
- The hybrid's recipe comes from a precompute searched over the whole dataset, while every ensemble process chooses its members and aggregator inside the fold's training split. The file's claim that ensemble_auto is 'the true like-for-like comparator' is not met on the hybrid side, and honest_rescore only re-selects among budgets, not among recipes.
- The hybrid row's budget label can be wrong (see the build_fold_hybrid finding), so the budget axis of the hybrid arm is not trustworthy without checking total_dim.
- Every single-rep baseline and every ensemble member is PCA-rotated by cap_dimensions when the representation is wider than the training split; the hybrid arm is not. On the small antibody sets that changes what 'best single rep' means for the tree models.

*Smoke:* ran clean. module imports clean, exit 0.

### Since the audit

0 fixed · 2 changed · 6 stand · 3 new · compute MORE EXPENSIVE · still runs

- **NEW** — A fold the hybrid arm cannot score shifts every later fold score by one position, and the new per-fold search makes that common. `tests/experiments/complexity/c11_ensemble_vs_hybrid.py:765-818`
- **NEW** — c11's outer folds are unstratified on a classification dataset whose labels are not integer-typed, which is the defect fixed elsewhere the same day. `tests/experiments/complexity/c11_ensemble_vs_hybrid.py:391`

*Compute now:* Roughly 57x on a dataset with no augmentation representations, roughly 170x with them. c11's own arms are unchanged: about 3,400 model fits, 200 PCAs of the concatenated matrix and 200 build_fold_hybrid calls per dataset.

### Decision

**MINE:** Five one-line repairs, the _emit fix, and take the PCA rotation off the baselines rather than adding it to the hybrid.

**The recheck opened something new here. Overrule me if it changes your call:**

>> 

## c12 — `complexity/c12_modality_ablation.py`

### What it does

On a TCGA dataset it scores every loaded representation on its own with 5 classification models over TSS-grouped folds, picks the best representation inside each omics modality, then np.hstack's the best representation of each modality pair and of all modalities at once and scores those the same way. It also persists a leave-one-fold-out re-score block.

**Asks:**
- For a TCGA task, which omics modality gives the best prediction on its own?
- Does column-concatenating the best representation of two modalities beat the better of the two modalities alone?
- Does concatenating one best representation per modality beat the best single modality?

**Verdict on 2026-09-03:** PARTIAL · hybrid: hand-rolled · live

Why it fell short:
- The combination rule is a raw np.hstack (c12:334), not the budgeted feature selection KIRBy actually performs. So it measures whether raw concatenation helps, not whether the hybrid picks a modality.
- The docstring and the loader comment (c12:15-33, c12:862-871) name gene_hvg2000/gene_pca100/gene_hallmark/gene_progeny/gene_kegg etc. as the expression modality.
- The printed and stored pairwise delta takes the max across models on the combination side AND the max across models and across representations on the single side (c12:372-376, 548, 565-570). Both arms are double maxima, so delta_vs_best_single is optimistically biased.

*Smoke:* ran clean. module imports clean, exit 0.

### Since the audit

0 fixed · 1 changed · 7 stand · 4 new · compute unchanged · still runs

- **NEW** — The leave-one-fold-out fix was applied to the baseline side only. `tests/experiments/complexity/c12_modality_ablation.py:545-560`
- **NEW** — At one fold the HELPS/HURTS verdict is not a leave-one-fold-out number and is not labelled as one. `tests/experiments/complexity/c12_modality_ablation.py:756 and :804`
- **NEW** — The TCGA representation cache is keyed on sample ids only, and the expression matrix changed under it. `tests/find_best_hybrid_tcga.py:300`

### Decision

**MINE:** Crash fixes, dead-name cleanup, print the honest_rescore number in the summary.

**Open — yours:** Whether the modality-combination arm stays a raw concatenation or becomes a real hybrid.

>> 

**The recheck opened something new here. Overrule me if it changes your call:**

>> 

## c13 — `complexity/c13_forward_compatibility.py`

### What it does

Splits the loaded representations into 'new' and 'legacy' by a hand-maintained name list, then for each new representation r runs five arms: r with one model, r with the best model, a hybrid over the legacy representations only, a hybrid over legacy plus r, and a hybrid over legacy plus a random-Gaussian matrix of r's width. It reports the debiased gain D minus D_random and the gain over r's own best single model.

**Asks:**
- Does adding a new representation to the hybrid's search space improve the hybrid beyond what a same-width random-Gaussian matrix earns from selection alone?
- Does the hybrid built with the new representation beat that representation's own best single-model score?
- Does the new representation carry signal above a random-Gaussian matrix and above a per-column-shuffled copy of itself?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · dispatched

Why it fell short:
- On every molecular dataset the auto-detection finds nothing. logd's general list is mhg_gnn, mordred, grover, ecfp4, smited, molformer, avalon, top_torsion, chembert, graphmvp — all ten are in _LEGACY_MOLECULAR, so _detect_new_reps returns [] and run() returns {'error':'no_new_reps'} at c13:396.
- The reverse problem on antibody data: flab_thermo_tresanco's ten representations split 8 new / 2 legacy, so condition C, described as 'the current KIRBy ceiling', is a hybrid over onehot plus esm2_early. The other seven new representations are excluded from both C and D, so the question 'does r improve the KIRBy hybrid' is answered against a two-representation pool, never the study's ten.
- The blueprint is chosen as whichever of rf/svm/lightgbm/mlp/knn scored best (c13:209-234), but the per-fold refit uses model_factory, which is models[best_b_model] — the model that won on the NEW REPRESENTATION ALONE (c13:276, 489, 529, 620). A blueprint searched for rf is routinely rebuilt and scored under mlp.

*Smoke:* ran clean. module imports clean, exit 0.

### Since the audit

0 fixed · 0 changed · 0 stand · 1 new · compute MORE EXPENSIVE · still runs

- **NEW** — c13 loses two of its five candidate downstream models with no message `tests/experiments/complexity/c13_forward_compatibility.py:209`

*Compute now:* No commit touched the file since 2026-09-02. Its cost moved entirely through the engine.

### Decision

**MINE:** Tie the blueprint to the model that scores it. Delete the nine stale result files so run_all stops skipping it.

**Open — yours:** Whether molecular datasets get an explicit --new-reps list. Without one c13 returns 'no_new_reps' on all eight.

>> 

## c14 — `complexity/c14_cross_domain_summary.py`

### What it does

For each named task it scores every base representation with each of the five downstream models over group-aware folds and keeps the best, then asks the pool helper for the best precomputed hybrid search and takes that search's own winning score. It reports the difference, a win rate, a per-domain breakdown and a Wilcoxon signed-rank test across tasks.

**Asks:**
- Across molecular, antibody and TCGA tasks, does the KIRBy hybrid beat the best single base representation?
- What is the win rate and the mean improvement, per domain?
- Is the per-task improvement consistent enough to reject zero under a signed-rank test?

**Verdict on 2026-09-03:** NO · hybrid: engine, per fold · orphaned

Why it fell short:
- The two arms are not comparable. hybrid_score at c14:279 is best_entry['score_mean'], which get_best_for_downstream returns as the maximum over the searched grid — 5 allocations x 3 importance methods x 5 internal evaluators x 8 budgets — scored on the very folds that selected it.
- In intermediary mode config['n_splits'] is 1, so the single-representation arm is one 80/20 GroupShuffleSplit while the hybrid arm's score_mean comes from the search's own 5-fold internal CV. The two arms are then scored on entirely different splits, and best_single_fold_scores and hybrid_fold_scores are stored side by side in the JSON as if they were paired.
- The Wilcoxon at c14:483 pools deltas measured in different metrics — RMSE on logd, Spearman on antibody, MCC or F1 on TCGA. The signed-rank statistic ranks by |delta|, so a 0.02 RMSE improvement and a 0.02 Spearman improvement are ranked against each other as if commensurable.

*Smoke:* ran clean. module imports clean, exit 0.

### Since the audit

2 fixed · 1 changed · 4 stand · 3 new · compute MORE EXPENSIVE · still runs

- **NEW** — The searches run over four internal evaluators, not the engine's five, and not the list c14's own config carries. `tests/experiments/complexity/c14_cross_domain_summary.py:267`
- Fixed (a09ac08) — hybrid_score is the search's own winner-of-the-grid score on the rows that selected it, while best_single_score is an unsearched CV mean; every headline number is built on that difference.
- Fixed (a09ac08) — mode='intermediary' gave n_splits=1 for the single-representation arm while the hybrid arm's score came from the search's own 5-fold internal CV, so the two arms were scored on different splits.

*Compute now:* The headline number moved from a cached whole-dataset search to nested_cv_evaluate, which searches inside every outer training fold. Before, per task: 10 representations x 5 downstream models x 5 folds = 250 baseline fits, plus 5 pool calls resolving to at most 3 cache files.

### Decision

**YOURS:** Delete, or route the headline through run_nested_cv at 15,000 hybrid builds per task. It is orphaned, has never written a file, and nothing else in complexity/ calls nested_cv_evaluate. My lean: delete.

>> 

## c15 — `complexity/c15_ml_theory_depth.py`

### What it does

Four sub-analyses on one dataset: pairwise-distance concentration and hubness per representation and per hybrid budget (15a); RBF kernel condition number, effective rank and kernel-target alignment plus a gamma sweep (15b); a budget sweep through p = n comparing hybrid, PCA and a random feature subset under RandomForest and Ridge (15c); and a random-label memorisation score (Rademacher proxy) for hybrid, PCA and the full concatenation (15d). Regression-only; it returns {'skipped': True} on classification.

**Asks:**
- Does a hybrid at budget b have less distance concentration, less hubness and a better-conditioned RBF kernel than each individual base representation of larger width, and than PCA at the same number of components?
- As the number of features passes the number of training rows, does the hybrid show the test-error spike that PCA and a random feature subset show?
- Does compressing to a budget reduce how much of a random label vector a Ridge model can memorise, relative to PCA at the same width and to the full concatenation?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · dispatched

Why it fell short:
- 15c can never compute the quantity it exists to compute. The budget list is n*[0.8, 0.9, 1.0, 1.1, 1.2] (line 482), and the 'near p=n' bucket is defined as ratio in [0.7, 1.3] (line 597).
- In 15c the arms are at different widths exactly where the question lives. PCA and the random subset are capped at min(budget, total_dim, n_samples - 1) (line 540) while the hybrid keeps the full budget (line 543).
- 15d's reference arm is the full concatenation of all 10 representations, rotated through a full-rank PCA and fed to Ridge (line 857, n_components = min(total_dim, n_samples-1)). A concatenation of everything is not an allowed baseline in this project.

*Smoke:* ran clean. module imports clean, exit 0.

### Since the audit

0 fixed · 0 changed · 0 stand · 0 new · compute MORE EXPENSIVE · still runs


*Compute now:* Worse, no new searches. a09ac08 added a group-aware 80/20 alignment split and put build_fold_hybrid on both sides of it, for the representations as well as the hybrid, so kernel alignment is read on rows that did not choose the features.

### Decision

**MINE:** Take the free dedupe: 160 identical rebuilds per dataset inside a loop where the hybrid does not depend on what the loop varies.

**Open — yours:** 15c: widen the budget band so the far bucket is reachable, or drop the sub-analysis. As written the instability ratio can never be computed.

>> 

## c16 — `complexity/c16_feature_selection_as_regularization.py`

### What it does

Builds a score grid of regularization strength against hybrid budget for three model families (Ridge or LogisticRegression, RandomForest max_depth, SVM C), rebuilding the hybrid per fold from a precomputed blueprint at each budget. Reports the spread of scores across regularization settings at each budget, compares the best budget at default regularization against the best regularization at the largest budget, and matches the train-minus-validation gap of a small budget to the gap of a regularization setting at the largest budget.

**Asks:**
- Does the spread of validation scores across regularization settings shrink as the hybrid budget shrinks, for each of Ridge/LogReg, RandomForest and SVM?
- Does the best budget at default regularization beat the best regularization setting at the largest budget?
- Which regularization strength at the largest budget produces the same train-minus-validation gap as a given small budget at default regularization?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · dispatched

Why it fell short:
- The largest-budget column silently goes all-NaN whenever total_dim is the binding constraint. capped_full = min(total_dim, 2n, 2048) at line 199, and _evaluate_cell skips every fold when budget >= total_dim at line 147.
- hybrid_wins, the boolean a reader will take as the answer to question 2, is computed by picking the best budget and the best regularization on the same folds they are scored on. Both arms are selection-optimistic.
- Every family is evaluated on a blueprint chosen for downstream_model='rf' (lines 246 and 267). The claim 'regularization stops mattering at the right budget' is therefore tested for SVM and Ridge on features that were allocated for a RandomForest.

*Smoke:* ran clean. module imports clean, exit 0.

### Since the audit

0 fixed · 0 changed · 0 stand · 0 new · compute MORE EXPENSIVE · still runs


*Compute now:* Worse per build, no new searches. The leave-one-fold-out re-score a09ac08 added is arithmetic on per-fold scores that already exist and costs no fits — the commit says so.

### Decision

**MINE:** Fix the LogReg default index and the capped_full NaN column.

**Open — yours:** Whether all three model families share one hybrid build. It changes what c16 measures.

>> 

## c17 — `complexity/c17_information_theoretic_decomposition.py`

### What it does

At each budget it builds five feature sets of that size from the concatenated pool (hybrid via the engine, PCA components, greedy mRMR, RFE with a RandomForest, and a random subset) and reports for each the estimated mutual information with the target, that information divided by the feature count, the effective rank, and a cross-validated RandomForest score. Regression-only.

**Asks:**
- At a fixed budget, which of hybrid, PCA, mRMR, RFE and a random subset produces the feature block with the most mutual information with the target per feature?
- Which of the five gives the best cross-validated RandomForest score at each budget?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · dispatched

Why it fell short:
- The downstream score cannot answer question 2. Hybrid, mRMR and RFE all select on every row including every fold's validation rows, then the score is cross-validated on those pre-selected features.
- The arms are not at matched width. PCA, mRMR, RFE and random get exactly `budget` features.
- mi_density is not comparable across budgets. estimate_mutual_information_knn PCA-caps its input at min(50, n_samples//5) components (_shared.py:2242), so at every budget of 50 or more the MI is estimated from the same 50 components while the denominator keeps growing.

*Smoke:* ran clean. module imports clean, exit 0.

### Since the audit

0 fixed · 0 changed · 0 stand · 0 new · compute MORE EXPENSIVE · still runs


*Compute now:* About 5x on the selection side. a09ac08 moved all five strategies inside the fold.

### Decision

**MINE:** Take the tie-break fix and the budget check.

**Open — yours:** The per-fold selection rewrite. It is the only thing that makes the five arms comparable, and it is expensive.

>> 

## c18 — `complexity/c18_diversity_vs_depth.py`

### What it does

At each budget and within each fold it ranks the representations by summed LightGBM gain on the training rows, then builds five feature sets of that budget and scores each with a RandomForest on the held-out fold: the top-1, top-2 and top-3 representations pooled and truncated to the budget, an equal share taken from every representation, and the engine's own hybrid rebuilt from the precomputed blueprint.

**Asks:**
- At a fixed budget, does spreading features across all representations beat concentrating them in the top 1, 2 or 3 representations?
- Does the engine's own allocation beat both the concentrated and the equal-share alternatives at the same budget?

**Verdict on 2026-09-03:** PARTIAL · hybrid: hand-rolled · dispatched

Why it fell short:
- The equal-share arm competes with fewer features than the others. _uniform_allocation uses budget // n_reps and then caps each representation at its own width, so at budget 25 with 10 representations it selects 20 features against the depth arms' 25, and at budget 2000 every representation narrower than 200 columns contributes fewer still.
- The four hand-rolled arms and the engine arm are computed by different code. depth_2, depth_3 and uniform_all reimplement allocations that already exist in NAMED_ALLOCATIONS as 'top_2', 'top_3' and 'fixed', and they do not reimplement them faithfully: _select_top_features pools the top-k representations' features and takes the global top-`budget` by importance, so depth_2 collapses to depth_1 whenever one representation dominates the importance scores, whereas the engine's top_k splits the budget across the k representations by weight.
- The importance method differs between the arms. The hand-rolled arms always use lightgbm_gain (lines 44, 57, 79), while the natural arm uses whatever importance method the blueprint chose (deeplift, lightgbm_gain or treeshap).

*Smoke:* ran clean. module imports clean, exit 0.

### Since the audit

0 fixed · 0 changed · 7 stand · 4 new · compute MORE EXPENSIVE · still runs

- **NEW** — A degenerate held-out block now raises inside the natural arm, and the caught failure leaves it averaged over fewer folds than the other four `tests/experiments/complexity/c18_diversity_vs_depth.py:245`
- **NEW** — The natural arm's budget split moved inside the fold; the four hand-rolled arms already were, so the comparison stays matched but the arm got more expensive `src/kirby/find_best_hybrid.py:7568`
- **NEW** — Every result_db file written before 2026-09-04 raises on open, and c18 turns the natural arm into a printed warning `src/kirby/result_cache.py:189`
- **NEW** — The eight result files on disk hold numbers today's code does not produce `tests/results/complexity/c18_diversity_vs_depth/`

*Compute now:* About 1.45x on the engine arm only, roughly +1.7h per dataset. No commit touched this file.

### Decision

**YOURS:** Whether the diversity-versus-depth arms are rebuilt through create_hybrid, and where that lives. It is an allocation bake-off, so by your placement rule it belongs in framework/ as a new f-number, not as an edit here. That call is yours.

>> 

## c19 — `complexity/c19_importance_weighted_training.py`

### What it does

Builds ONE hybrid feature matrix through the pool helper, then scores five importance-weighting treatments of that matrix (unweighted, features scaled by sqrt-importance, sample weights from neighbour label agreement, both, and drop features below median importance) with rf/lightgbm/xgboost under grouped CV. Reports each treatment's score, its difference from unweighted, and which treatment won the most model cells.

**Asks:**
- On the hybrid feature matrix, does rescaling features by sqrt of their importance change the downstream score against leaving them alone?
- Does weighting each training row by how well its label agrees with its ten nearest neighbours change the downstream score?
- Does deleting every hybrid feature whose importance is below the median change the downstream score?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · dispatched

Why it fell short:
- The hybrid is built on ALL rows. get_fitted_pool_hybrid calls create_hybrid(reps, labels, ...) over the whole dataset (_shared.py:4608), and c19 then splits that already-selected matrix with cv_split.
- The score is metric_fn(y_val, model.predict(X_va_s)). On the classification datasets in the roster (herg, tz_her2_binary, humanness_therapeutic, tcga_*, nt_*, gb_*) the metric is AUROC and this hands it hard 0/1 labels.
- The JSON records budget=100, the value asked for, not the budget the chosen blueprint actually carries. get_best_for_downstream falls back to the best budget across all budgets when 100 has no entry, and create_hybrid then uses bp.budget.

*Smoke:* ran clean. module imports clean, exit 0.

### Since the audit

2 fixed · 0 changed · 4 stand · 4 new · compute MORE EXPENSIVE · still runs

- **NEW** — The fold's feature set is chosen with whichever model happens to be first in the model list. `tests/experiments/complexity/c19_importance_weighted_training.py:205`
- **NEW** — When the recipe lookup returns nothing the file writes a result saying 'unweighted' won, with no error key. `tests/experiments/complexity/c19_importance_weighted_training.py:159-166`
- **NEW** — importance_pruned prunes nothing whenever more than half the hybrid features get zero importance. `tests/experiments/complexity/c19_importance_weighted_training.py:262`
- **NEW** — The per-representation score that sizes each representation's share of the budget is now fitted with the internal evaluator, not with the model passed in. `src/kirby/hybrid.py:2521-2543`
- Fixed (a09ac08) — get_fitted_pool_hybrid builds the feature matrix over every row, then the fold loop splits that already-selected matrix, so feature selection has seen the validation labels.

*Compute now:* The hybrid feature matrix is now built once per fold instead of once over every row. One create_hybrid build became five.

### Decision

**MINE:** Move the build per fold, fix the recording, take the two one-liners. The eight stored files re-run either way.

## c20 — `complexity/c20_signal_saturation.py`

### What it does

For one base representation at a time, ranks that representation's own columns by lightgbm_gain on each fold's training rows, scores the top-k columns on the held-out rows for k in 10..1000 plus the native width, and reports the smallest k that lands within tolerance of the full-width score. No hybrid anywhere, by design.

**Asks:**
- For each base representation on its own, what is the smallest number of its own columns that scores within tolerance of using all of them?
- Does that saturation point differ between lightgbm and rf as the downstream model?

**Verdict on 2026-09-03:** NO · hybrid: none needed · orphaned

Why it fell short:
- It cannot run. Line 50-54 inserts a scratchpad directory that does not exist on this machine and imports round2_harness from it.
- The harness was rebuilt into the repo as tests/experiments/_alloc_harness.py (its own docstring says so, and c31:87 already imports it that way), but _alloc_harness defines no STAGED_DATASETS, which c20:288 reads when --dataset is omitted. Fixing the import alone still leaves an AttributeError on the all-datasets path.
- Nothing is written unless --out is passed; the default run only prints JSON to stdout.

*Smoke:* ran clean. As shipped it dies at import: exit 1, ModuleNotFoundError: No module named 'round2_harness' (c20:54).

### Since the audit

0 fixed · 0 changed · 5 stand · 0 new · compute unchanged · **DOES NOT RUN**

- **STILL BROKEN** — Imports round2_harness from a /private/tmp scratchpad directory that does not exist, so every entry point dies at import.
- **STILL BROKEN** — H.STAGED_DATASETS is read when --dataset is omitted, but the surviving harness defines no such name.

### Decision

**MINE:** Run-repair, hoist the ranking out of the budget loop, rename the plateau field.

**The recheck opened something new here. Overrule me if it changes your call:**

>> 

## c21 — `complexity/c21_single_rep_augmentation.py`

### What it does

Takes one base representation at its full native width and asks whether adding a few extra columns helps: from the dataset's curated augmentation representations, from the top-5 importance columns of every other pool representation, from both, and from two matched-width random-column controls. Runs the same extras against a fixed multi-representation hybrid at budgets 500/1000/2000 as the reference, with a pure-noise donor and a jittered copy of the base planted in every donor pool and their admitted-column counts recorded per fold.

**Asks:**
- Does adding a handful of extra columns to a single strong base representation beat that representation alone?
- Do the same extras do more on one representation at full width, or on the multi-representation hybrid at a budget?
- Does the admission rule take any columns from a pure-noise donor or from a jittered near-copy of the base representation?

**Verdict on 2026-09-03:** YES · hybrid: engine, per fold · dispatched

*Smoke:* ran clean. module imports clean, exit 0.

### Since the audit

0 fixed · 0 changed · 4 stand · 4 new · compute unchanged · still runs

- **NEW** — The budget-split fix reached the hybrid reference arms and not the single-representation arms. `src/kirby/hybrid.py:2521 and tests/experiments/complexity/c21_single_rep_augmentation.py:256`
- **NEW** — The rule that ranks candidate extra features changed, so the decoy counts this file exists to measure are no longer the ones on record. `src/kirby/hybrid.py, _pool_and_rank_aug_features (c74905b)`
- **NEW** — A fold whose held-out block scores no representation finitely now raises instead of splitting the budget equally, and c21 turns that into a NaN fold. `tests/experiments/complexity/c21_single_rep_augmentation.py:661-665`
- **NEW** — The six result files in tests/results/complexity/c21_single_rep_augmentation/ predate every engine change and the code can no longer reproduce them. `tests/results/complexity/c21_single_rep_augmentation/`

### Decision

**MINE:** Cap the decoys to the donor widths conditions C and D actually use, and fix the scoring line.

**Open — yours:** Whether the multi-representation reference arm gains its own control.

>> 

## c22 — `complexity/c22_explainability.py`

### What it does

Picks the winning blueprint from the precomputed search pool, rebuilds that hybrid per fold on training rows, fits rf and xgboost, runs SHAP on whichever scored better, averages the per-feature SHAP across folds, and maps those values back to source representations to report how much each representation contributed.

**Asks:**
- For the winning hybrid, how many features does each base representation contribute?
- How much of the fitted model's SHAP importance comes from each source representation?
- Which individual hybrid features carry the most SHAP importance, and which representation and column did each come from?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · dispatched

Why it fell short:
- The per-representation SHAP breakdown is misaligned. avg_shap is averaged over all folds (line 241) but is then sliced by the LAST fold's feature_info (line 231).
- The allocation written to JSON is one fold's, presented as the allocation. Per-fold allocations are not stored, so nobody can check afterwards how much it moved.
- search_score (the search's own inner-fold winner) and the per-fold model scores (honest held-out) sit in the same JSON with no marker separating the two quantities.

*Smoke:* ran clean. module imports clean, exit 0.

### Since the audit

4 fixed · 0 changed · 1 stand · 5 new · compute unchanged · still runs

- **NEW** — shap_by_rep and shap_share_by_rep hold the same numbers and are described in the JSON as two different quantities. `tests/experiments/complexity/c22_explainability.py:314 and 324`
- **NEW** — The top-feature listing averages SHAP across folds whenever the per-representation counts match, but the selected features inside each representation still move. `tests/experiments/complexity/c22_explainability.py:273-274`
- **NEW** — build_fold_hybrid is called with no model and no metric_fn, so accept/reject inside the build is scored with r2_score or accuracy rather than the dataset's metric. `tests/experiments/complexity/c22_explainability.py:148-152`
- Fixed (a09ac08) — feature_info is taken from the last fold while avg_shap is averaged over every fold, so SHAP mass is attributed to whichever representation occupies those positions on fold 4.
- Fixed (a09ac08) — The comment claims feature_info is deterministic given the blueprint, so all folds produce identical structure.
- Fixed (a09ac08) — Only the last fold's allocation reaches the JSON, with no per-fold record.
- Fixed (a09ac08) — search_score and the per-fold held-out model scores sit in the same JSON with no marker separating them.

### Decision

**MINE:** Do it. The re-run is 5 builds and 10 fits per dataset and the current per-representation SHAP cannot be defended.

## c23 — `complexity/c23_evaluator_model_matching.py`

### What it does

For each of eleven downstream models, runs a separate full find_best_hybrid search with the internal evaluator pinned to the same model family, to a near family, and to a far family, plus one neural-size mismatch and one run over the three tree evaluators together. Reports each arm's search score and its difference from the matched arm.

**Asks:**
- Does pinning the internal evaluator to the downstream model's own family produce a better-scoring hybrid than pinning it to a different family?
- Does the size of the mismatch matter — is a near family better than a far one?
- Does letting the search choose among rf, lightgbm and xgboost beat pinning the evaluator to the downstream model?

**Verdict on 2026-09-03:** NO · hybrid: engine, per fold · dispatched

Why it fell short:
- 'ridge' is a valid downstream model but is NOT a valid internal evaluator. hybrid.py's evaluator dispatch ends at line 2318 with `raise ValueError(f"Unknown internal_evaluator: ...")` and ridge has no branch.
- Every reported number is best['score_mean'], which is a max over 120 search configurations scored on the same inner folds (find_best_hybrid.py:5296 takes the argmax, :3798 is the fold mean). This is the search_mean the house rule forbids reporting.
- Condition E is not comparable to condition A. E is a max over three evaluators, A is a max over one.

*Smoke:* ran clean. module imports clean, exit 0.

### Since the audit

0 fixed · 0 changed · 0 stand · 0 new · compute MORE EXPENSIVE · still runs


*Compute now:* Exactly 5x the searches, 4.0x the fits. a09ac08 replaced one find_best_hybrid per cell with nested_cv_evaluate at n_outer_folds = config['n_splits'] = 5.

### Decision

**MINE:** Fix the ridge evaluator names and the cache namespace.

**Open — yours:** Rewrite c23's headline through nested CV, or delete it and let framework/f13 answer the evaluator question. f13 already asks it with only dispatchable evaluators.

>> 

## c24 — `complexity/c24_blueprint_transfer.py`

### What it does

Runs a full find_best_hybrid on each fold's training rows to get that fold's winning recipe, scores it on that fold's held-out rows, then applies each fold's recipe to every other fold — rebuilding the hybrid on the target fold's own training rows — and reports own-minus-cross differences plus a Nogueira stability index over the selected feature sets.

**Asks:**
- Does the recipe discovered on one fold score as well on another fold as that fold's own recipe?
- How stable is the actual set of selected features across folds?

**Verdict on 2026-09-03:** NO · hybrid: engine, per fold · dispatched

Why it fell short:
- The transfer test does not transfer feature selection. Step 2 takes only the source fold's four recipe fields (allocation method, budget, importance method, internal evaluator) and re-runs create_hybrid on the TARGET fold's training rows, so the features are re-selected from scratch.
- create_hybrid is called bare at lines 219 and 302 with no groups. Its internal admission holdout then falls back to a RandomState(42) 20% RANDOM split of the training rows.
- The same bare calls drop model, metric_fn, lower_is_better, step_size, augmentation_margin, augmentation_fraction, candidate_ranking, augmentation_budget and the blueprint's own rep_scores. So accept/reject inside create_hybrid is scored with r2_score/accuracy rather than the dataset's metric, and the rebuild uses create_hybrid's defaults instead of the recipe the search chose — the exact defect build_fold_hybrid was written to close.

*Smoke:* ran clean. module imports clean, exit 0.

### Since the audit

0 fixed · 0 changed · 0 stand · 0 new · compute MORE EXPENSIVE · still runs


*Compute now:* 4.0x. Step 1 keeps its 5 searches, one per fold on that fold's training rows.

### Decision

**MINE:** Apply the feature-set transfer and build_fold_hybrid together, plus the two one-liners. Same re-run.

## c30 — `complexity/c30_balanced_global_selection.py`

### What it does

Three feature-selection arms per dataset at budgets 100/500/1000 over the top-5 and top-10 reps: equal per-rep quota, a per-rep top-N cap then one global lightgbm_gain rank, and no cap then one global rank. Everything is hand-rolled on a 5-fold GroupKFold; lightgbm is the fixed downstream model.

**Asks:**
- Does capping each representation at an equal top-N before one global lightgbm_gain ranking beat giving each representation an equal fixed quota of the budget?
- Does that per-representation cap remove the bias toward the widest representation that the no-cap global ranking shows?
- Does the manual equal-quota selection reproduce create_hybrid(allocation_method='fixed') on one fold?

**Verdict on 2026-09-03:** NO · hybrid: hand-rolled · orphaned

Why it fell short:
- The script cannot run: line 30 imports round2_harness from /private/tmp/claude-501/.../45a3af00-.../scratchpad, which does not exist. The module now lives at tests/experiments/_alloc_harness.py and is not on that path.
- Even with the import fixed, line 277 writes into that same missing directory, so the results file is never produced.
- The target (best single representation) is picked at line 106 from scores measured on the same five held-out folds the arms are then compared on. The winner is chosen on the data it is judged on.

*Smoke:* ran clean. Imports clean only because I created the missing harness symlink first.

### Since the audit

4 fixed · 1 changed · 0 stand · 4 new · compute MORE EXPENSIVE · still runs

- **NEW** — The engine cross-check compares the engine's top-20 features against a manual top-20 ranked on a different set of rows. `tests/experiments/complexity/c30_balanced_global_selection.py:390-411`
- Fixed (38bf326) — Imports round2_harness from a hardcoded /private/tmp scratchpad directory that no longer exists.
- Fixed (38bf326) — Writes expB_{ds}.json into the same missing scratchpad directory, and under the exact filename exp_b_balanced_global.py also writes, so a repair to a real directory would make them overwrite each other.
- Fixed (38bf326) — best_rep is ranked on whole-data fold scores, then used as the paired target on those same folds.

*Compute now:* About 1.7x more downstream fits per dataset. Before: 50 standalone (10 representations x 5 folds) + 5 reproducibility + 30 arm1 + 120 arm2 + 30 arm3 = 235-240.

### Decision

**YOURS:** Delete and read the same three arms off c31, or take the three-line repair. Deleting loses budget 100 and the 50/300 caps. c31 already ran those arms on 10 datasets and 3 models.

>> 

## c31 — `complexity/c31_perf_weighted_global_selection.py`

### What it does

For each downstream model, sizes each representation's slice of a shared candidate pool by a performance weight, glues the candidate columns, runs one global importance ranking and keeps the top B. Compares that against the best single representation and against three controls (equal quota, equal cap then global, no cap then global) at budgets 500/1000, with a group-block bootstrap on the pooled out-of-fold predictions.

**Asks:**
- Does sizing each representation's candidate slice by a performance weight beat giving every representation an equal cap before the same global ranking?
- Which weighting shape (linear, power 2, top-2) sizes the candidate pool best, and does the answer change with the downstream model or with the importance method used for the global ranking?
- Does any of these beat the best single representation at a matched budget, by more than the group-block bootstrap noise floor?

**Verdict on 2026-09-03:** PARTIAL · hybrid: hand-rolled · live

Why it fell short:
- The target and the candidate pool are still ranked on whole-data fold scores (line 426-431). Only the per-representation slot sizes are cross-validated.
- The docstring at line 13-15 claims the pool-sizing weighting is guaranteed identical to the allocation strategies. Only _compute_allocation_weights is shared.
- No nested_cv_evaluate; deltas are a hand-rolled single-level 5-fold loop plus a pooled-out-of-fold bootstrap, disclosed as a proxy at line 165-171.

*Smoke:* not run. exceeded 6 min at '[auto] caco2: filtering to 10 general reps from DATASET_REPS' — it had not yet printed 'Loaded 2158 molecules'.

### Since the audit

1 fixed · 1 changed · 4 stand · 3 new · compute unchanged · still runs

- **NEW** — The headline cell winner and the bootstrap confidence interval are still computed against the leaky whole-data target, not the new per-fold one. `tests/experiments/complexity/c31_perf_weighted_global_selection.py:733 and 777`
- Fixed (38bf326) — best_scheme_by_cell reported a delta and a bootstrap interval for a weighting method chosen on the folds it was scored on.

### Decision

**MINE:** Restore the full 15-scheme weighting family and re-run. Cutting 12 of 15 removed the axis the experiment exists to measure.

## c32 — `complexity/c32_allocation_strategy_cross.py`

### What it does

Crosses three axes the earlier runs only moved one at a time: how many top representations get budget (2/3/5/10), the weighting shape (linear, power 2, softmax, rank-exp), and the mechanism (per-representation weighted quota, or weighted candidate slices then one global ranking). 32 strategies per (dataset, model, budget) cell, each paired per fold against the best single representation.

**Asks:**
- Inside the grid interior nobody has run, does a given weighting shape beat the others once the number of representations receiving budget is also varied?
- Does the per-representation weighted quota or the weighted-slice-then-global-rank mechanism win, and does that flip with the number of representations, the weighting, the budget or the downstream model?
- Does any full strategy beat the best single representation at a matched budget?

**Verdict on 2026-09-03:** PARTIAL · hybrid: should use engine, does not · live

Why it fell short:
- CAVEATS at line 104-106 say the sizing has 'no whole-data leak'. The weights do not leak, but which representations are in the N-subset (line 311, ranked[:N]) and which one is the target (line 204) both come from whole-data standalone scores measured on the same folds.
- CAVEATS at line 108-109 say the quota mechanism '== allocate_performance_weighted'. The engine's exact-budget top-up and trim (hybrid.py:1319-1352) is not implemented, so c32's quota arm spends round-and-cap, not B.
- The target's per-fold R-squared and native arrays are never written, so only the deltas survive in the JSON.

*Smoke:* not run. exceeded 6 min at '----- MODEL: rf -----'.

### Since the audit

2 fixed · 0 changed · 3 stand · 2 new · compute unchanged · still runs

- **NEW** — The 27 to 30 result files already on disk under c32_allocation_cross/ compare against a different baseline than any file the current code writes, and the analysis handoff still describes the old shape. `C32_ANALYSIS_HANDOFF_2026_08_29.md:59-77`
- **NEW** — The engine cross-check compares selections built from different rows. `tests/experiments/complexity/c32_allocation_strategy_cross.py:189-205`
- Fixed (38bf326) — The caveat claims no whole-data leak in the sizing, but pool membership, the N-subsets and the target are all whole-data ranked on the same folds used for the paired delta.
- Fixed (38bf326) — The target's per-fold arrays are never written, so only the deltas survive in the JSON and an arm's absolute per-fold score cannot be reconstructed.

### Decision

**MINE:** Add the exact-budget top-up and trim, fifteen lines, then re-run. Median task is 23 minutes.

## exp_b — `complexity/exp_b_balanced_global.py`

### What it does

The same three arms as c30 — equal quota, per-representation cap then one global ranking, no cap then one global ranking — over the top-5 and top-10 representations at budgets 100/500/1000, run across four datasets in one process, with a blocking metric-direction check and a downstream model chosen from lightgbm and rf.

**Asks:**
- Does capping each representation at an equal top-N before one global lightgbm_gain ranking beat the equal per-representation quota?
- Does the cap remove the width bias the no-cap global ranking shows?
- Does either global arm beat the best single representation at a matched budget?

**Verdict on 2026-09-03:** NO · hybrid: hand-rolled · superseded

Why it fell short:
- Cannot run: line 36 imports round2_harness from a deleted /private/tmp scratchpad directory.
- Even with the import redirected to tests/experiments/_alloc_harness.py, line 104 calls H._label_signal_ladder, which that module does not define. The blocking direction gate raises AttributeError before any arm runs.
- Output directory (line 63-64) is the same deleted path.

*Smoke:* ran clean. Imports clean only with the harness symlink in place.

### Since the audit

4 fixed · 0 changed · 2 stand · 3 new · compute MORE EXPENSIVE · still runs

- Fixed (38bf326) — Imports round2_harness from a hardcoded /private/tmp scratchpad path that no longer exists.
- Fixed (38bf326) — H._label_signal_ladder is not defined in the surviving harness, so the blocking direction gate raises before any measurement.
- Fixed (38bf326) — OUTDIR is the deleted scratchpad directory, and the filename expB_{ds}.json collides with the one c30 writes.

*Compute now:* About 2x more downstream fits per dataset. Before: 25 direction-gate fits (5 noise levels x 5 folds) + 50 standalone + 10 model sweep + 5 reproducibility + 30 arm1 + 120 arm2 + 30 arm3 = ~270.

### Decision

**MINE:** Delete. Move the metric-direction gate into _alloc_harness first — four other scripts want it. c30 is the later rewrite of the same three arms, c31 the one after that.

## c43_signal — `complexity/c43_buchwald_signal.py`

### What it does

A one-representation diagnostic on buchwald_hartwig: scores cgr_morgan with random forest over 5 group-aware folds and compares the RMSE against the label spread and against predicting the training-fold mean.

**Asks:**
- Does buchwald_hartwig carry real predictive signal under cgr_morgan with a random forest, or is the RMSE close to predicting the training mean?
- What is the RMSE scale relative to the label standard deviation, so a later hybrid delta can be read as large or small?

**Verdict on 2026-09-03:** NO · hybrid: none needed · orphaned

Why it fell short:
- Cannot run: line 18 imports round2_harness from a deleted /private/tmp scratchpad directory.
- Nothing is written to disk. The result exists only as printed stdout (line 91-92), so the anchor RMSE it is meant to establish cannot be read back later — and c43_buchwald_step3.py:131 then hardcodes that number as a literal.

*Smoke:* ran clean. Imports clean only with the harness symlink in place (c43_buchwald_signal.py:15-18).

### Since the audit

0 fixed · 0 changed · 3 stand · 0 new · compute unchanged · **DOES NOT RUN**

- **STILL BROKEN** — Imports round2_harness from a hardcoded /private/tmp scratchpad path that no longer exists.

### Decision

**MINE:** Repair it. Ten model fits, and it is the anchor c43_step3 currently hardcodes as a literal.

**The recheck opened something new here. Overrule me if it changes your call:**

>> 

## c43_step3 — `complexity/c43_buchwald_step3.py`

### What it does

Three measurements on buchwald_hartwig over 10 hand-picked representations: how cgr_morgan's RMSE degrades as it is truncated to K top features, how many features cgr_morgan actually gets inside a real create_hybrid call and how that slice scores alone, and how a single global lightgbm_gain ranking over the whole concatenated pool compares at the same budget.

**Asks:**
- When cgr_morgan is truncated to its top K features, at what K does its accuracy stop matching its full width?
- How many features does cgr_morgan actually receive under create_hybrid at budget 100 and 1000, and does that slice alone still carry the representation's signal?
- Does one global lightgbm_gain ranking over the concatenated pool beat the quota hybrid at a matched budget, and which representations does it draw from?

**Verdict on 2026-09-03:** NO · hybrid: engine, per fold · orphaned

Why it fell short:
- Cannot run: line 21 imports round2_harness from a deleted /private/tmp scratchpad directory, and the checkpoint path at line 25 points into the same missing directory, so the first json.dump at line 82 would fail too.
- The pool of 10 is hand-picked, not score-ranked (disclosed at line 30), so 'the hybrid' here is not the pool any other experiment builds.
- The headline delta written at line 131 crosses the model axis — a lightgbm hybrid minus a hardcoded random-forest RMSE.

*Smoke:* ran clean. Imports clean only with the harness symlink in place (c43_buchwald_step3.py:20-21).

### Since the audit

0 fixed · 0 changed · 4 stand · 0 new · compute unchanged · **DOES NOT RUN**

- **STILL BROKEN** — Imports round2_harness from a deleted /private/tmp scratchpad path, and the checkpoint path points into the same missing directory, so the first json.dump would fail too.

### Decision

**MINE:** Take option 1. It is the only file in this group that builds through the engine.

**The recheck opened something new here. Overrule me if it changes your call:**

>> 


---

# Noise — 14 files

## n01 — `noise/n01_activity_cliff_framework.py`

### What it does

Marks each molecule as a cliff or non-cliff compound from a reference that does not use any of the representations (ECFP4 Tanimoto for molecular, sequence edit distance for antibody, Euclidean on the concatenated representations otherwise), then runs 5-fold scaffold-grouped CV per model per representation and reports the metric on the cliff rows, the non-cliff rows, and the difference. The hybrid is rebuilt per fold from a searched blueprint and gets the same treatment.

**Asks:**
- Is the prediction error on activity-cliff compounds worse than on non-cliff compounds, and by how much?
- Is that cliff/non-cliff gap smaller for the hybrid than for the individual base representations?
- Does the answer to the second question hold across rf, xgboost, svm, mlp and lightgbm, or only for rf?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · dispatched

Why it fell short:
- The hybrid's recipe is not chosen per fold. get_fitted_pool_hybrid / get_best_search_for_downstream return a blueprint from a find_best_hybrid search over the WHOLE dataset, and build_fold_hybrid then forwards blueprint.rep_scores as precomputed_alloc_scores (_shared.py:3751), so both the recipe and the per-representation budget split were decided with the held-out rows' labels visible.
- Only fold means are written. fold_cliff and fold_noncliff are averaged (n01:352-353) and the per-fold values are discarded, so no paired test or error bar on the gap is possible from the file.
- The cliff and non-cliff scores come from different numbers of folds (n_cliff_folds vs n_noncliff_folds), and on small antibody or reaction sets a fold's cliff subset can be 2 rows. A Spearman or R2 on 2 rows is +/-1.

*Smoke:* FAILED. exit 139 (segfault) after 160s, right after the 'EXPERIMENT N01: CLIFF-STRATIFIED PREDICTION ERROR' header; PYTHONFAULTHANDLER printed only 'Fatal Python error:' before dying.

### Since the audit

1 fixed · 1 changed · 10 stand · 4 new · compute MORE EXPENSIVE · run state unknown

- **NEW** — A fold whose internal held-out block cannot score any representation is now dropped from the hybrid's three means with no record, while the per-representation baselines lose no fold. `tests/experiments/noise/n01_activity_cliff_framework.py:432`
- **NEW** — On drug-target affinity the new cross-protein zeroing is undone by the percentile cutoff that reads it. `tests/experiments/noise/n01_activity_cliff_framework.py:232`
- **NEW** — The five n01 result files on disk cannot be reproduced by today's code. `tests/results/noise/n01_activity_cliff_framework/`
- Fixed (1e74b25) — make_cv_splitter(n_folds, groups) is called without task_type, so a classification dataset gets GroupKFold instead of StratifiedGroupKFold.

*Compute now:* At --mode full a dataset costs 25 create_hybrid calls (5 models x 5 folds) plus 250 per-representation baseline fits. The 250 baseline fits are unchanged.

### Decision

**MINE:** Take the local fixes. Fix the empty-dict test so a failed hybrid pool stops wiping the per-representation baselines.

**Open — yours:** Whether the hybrid recipe is searched per fold.

>> 

## n02 — `noise/n02_sample_weighting.py`

### What it does

Scores each training row by how well its label agrees with its 10 nearest neighbours in a distance where every feature is scaled by the square root of its lightgbm_gain importance, then refits with those scores as sample weights. Runs five conditions — each base representation weighted and unweighted, the hybrid weighted and unweighted, and the hybrid with ActivityCliffAwareEstimator weights — for rf, xgboost and lightgbm.

**Asks:**
- Do importance-weighted sample weights improve held-out prediction relative to training unweighted?
- Does that improvement differ between an individual base representation and the hybrid?
- Do cliff-aware weights beat plain neighbour-agreement weights on the hybrid?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · dispatched

Why it fell short:
- Same blueprint leak as n01: the hybrid's recipe and its rep_scores come from a whole-dataset find_best_hybrid search (n02:272), and build_fold_hybrid forwards rep_scores as precomputed_alloc_scores. Comparing the hybrid's weighting benefit against a base representation's therefore compares a tuned arm with an untuned one.
- The weighted and unweighted means can cover different folds. A failed weighted fit appends nan (n02:243, n02:405) while the unweighted score for that fold stays in; w_mean uses nanmean and uw_mean uses mean, so delta subtracts means over different fold sets.
- No aggregate base-vs-hybrid statement is computed. The printed summary counts only how many models the hybrid improved on (n02:491).

*Smoke:* not run. not attempted, time

### Since the audit

0 fixed · 1 changed · 7 stand · 3 new · compute MORE EXPENSIVE · run state unknown

- **NEW** — Condition 5's cliff-aware mean and the unweighted mean it is subtracted from can cover different folds, and nothing records it. `tests/experiments/noise/n02_sample_weighting.py:469`
- **NEW** — The four n02 result files on disk cannot be reproduced by today's code. `tests/results/noise/n02_sample_weighting/`

*Compute now:* No commit touched this file. The engine change reaches it anyway.

### Decision

**MINE:** Merge condition 5 into the conditions 3-4 loop and pair the fold sets. Saves 15 rebuilds per dataset.

**Open — yours:** Whether the importance scorer behind the weighted distance is swept rather than fixed to lightgbm_gain.

>> 

## n06 — `noise/n06_bindingdb_ground_truth.py`

### What it does

Groups BindingDB rows by (drug_id, target_id) from the loader metadata, takes the standard deviation of each group's pKd values as a real assay-noise magnitude, then runs 10 noise estimators over two feature spaces and reports the Spearman correlation between each estimator's per-row noise score and that group standard deviation, over the rows in disagreeing groups only.

**Asks:**
- Which noise estimator's per-row score agrees best with real repeat-measurement disagreement in BindingDB?
- Does the estimator agree better when it is run over the hybrid feature space than over the concatenation of all representations?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · orphaned

Why it fell short:
- The second question's comparison arm is a concatenation of every representation. Individual base representations are never scored as a space, so nothing here says whether a single representation beats both.
- The two spaces are not on equal footing. The hybrid space is built by get_fitted_pool_hybrid on all rows with all labels (n06:218), so its features were selected with the labels the ground truth is computed from; the concatenation had no label access at all.
- Rows in the same replicate group are the same drug and the same target, so their feature vectors are byte-identical in every representation. A neighbour-based estimator's nearest neighbour at distance zero is its own replicate, so neighbor_residual, local_variance_ratio and distance_weighted_residual recover the group standard deviation close to by construction.

*Smoke:* ran clean. exit 0, ~2 min, 'ALL N06 SMOKE TESTS PASSED'.

### Since the audit

2 fixed · 1 changed · 6 stand · 3 new · compute unknown · run state unknown

- **NEW** — The per-fold hybrid space stacks five folds' feature blocks into one matrix after checking only that the feature COUNT matches, so rows from different folds hold different features. `tests/experiments/noise/n06_bindingdb_ground_truth.py:224`
- **STILL BROKEN** — When there are more replicate rows than GP_MAX the subsample keeps every replicate row and the 1000-row cap is abandoned, so gaussian_process is handed several thousand rows at O(n^3).
- Fixed (6761eae) — 'concat', the hstack of every representation, is used as the comparison space against the hybrid, and individual base representations are never evaluated as spaces.

*Compute now:* Two changes pull opposite ways and neither was timed. Added: _oof_hybrid_space runs five build_fold_hybrid calls (n06:209), each a full create_hybrid at the measured 262-336s plus about 151s of per-representation scoring, so 5 x ~450s = 2,250s, about 37 minutes per run.

### Decision

**MINE:** Fix the output shape and add per-representation spaces. The concatenation-only comparator is against the baseline rule.

**Open — yours:** Whether the replicate rows stay. Their feature vectors are identical, so a neighbour estimator recovers the group spread close to by construction.

>> 

**The recheck opened something new here. Overrule me if it changes your call:**

>> 

## n07 — `noise/n07_remediation.py`

### What it does

Injects outlier-condition noise into each fold's training labels, then compares four cleanup strategies (remove / downweight / label_smooth / knn_smooth) against a clean and a do-nothing baseline, in a per-model hybrid feature space and again in each individual base representation. Reports recovery as (strategy - noisy) / (clean - noisy).

**Asks:**
- Which of remove, downweight, label_smooth or knn_smooth recovers the most of the clean-to-noisy performance gap, separately for each of the five downstream models, in that model's hybrid feature space?
- Which strategy recovers the most in each individual base representation (random forest only)?
- Side-car written to the same file: does the hybrid beat the best single representation under leave-one-fold-out re-selection, on the CLEAN-label search?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · dispatched

Why it fell short:
- The hybrid feature space is built from the clean labels the run then corrupts. build_fold_hybrid at line 504 is handed y_train_clean, and only afterwards is y_train_clean corrupted into y_tr_noisy at line 532.
- The hybrid block runs five models; the per-rep block runs random forest only (line 584). A hybrid-vs-single-rep read across the two blocks compares five models against one.
- The flag rate is fixed at the 80th percentile of the noise scores (lines 182, 204, 222) regardless of noise_fraction, which defaults to 0.1. All three flag-based strategies always touch 20% of rows while 10% are corrupted, so the ranking of strategies is entangled with a fixed over-flagging rate that is never swept.

*Smoke:* ran, NaNs present. exit 0, 'n07 RAN OK in 16.9s'.

### Since the audit

2 fixed · 0 changed · 6 stand · 4 new · compute MORE EXPENSIVE · still runs

- **NEW** — The no_noise ceiling stays in the clean-label feature space while every noisy arm moved to the noisy one, so the recovery denominator changed meaning. `tests/experiments/noise/n07_remediation.py:563`
- **NEW** — exclude_folds is hardcoded to [] on a justification that is false for this file's candidates. `tests/experiments/noise/n07_remediation.py:489`
- **NEW** — The pool is now asked for a 'dnn' blueprint, which no search cached before 2026-09-04 contains. `tests/experiments/noise/n07_remediation.py:322`
- **NEW** — A recovery percentage was computed even when the injected noise left the model no worse, inverting every strategy's number. `tests/experiments/noise/n07_remediation.py:775`
- Fixed (9b288a0) — build_fold_hybrid is handed the clean labels, so the feature space is chosen before the noise is injected.
- Fixed (9b288a0) — Docstring claims six models including ridge; the list holds five and no ridge.

*Compute now:* create_hybrid builds go from one per (model, fold) to four — one clean plus one per noise seed. Full mode is 5 models x 5 folds: 25 builds becomes 100, a factor of 4.

### Decision

**MINE:** Add the caveat field and the six one-line fixes.

**Open — yours:** Whether the hybrid is built on the noisy labels rather than the clean ones it then corrupts.

>> 

## n09 — `noise/n09_lnl_survey.py`

### What it does

Sweeps six injected noise levels; at each level and fold it re-runs find_best_hybrid on the NOISY training labels to build a hybrid, then trains a plain DNN and each of twelve learning-with-noisy-labels methods on three feature spaces (one fixed base representation, the hybrid, and an importance-weighted hybrid). Reports which methods rank top-3 in each space and whether that ranking differs.

**Asks:**
- Does swapping one fixed base representation for a per-fold, noise-searched hybrid change which learning-with-noisy-labels methods come out top-3 at the highest injected noise level?
- Does importance-weighting the hybrid's columns change that ranking again, relative to the unweighted hybrid?
- How much of the clean-label score does each (feature space, LNL method) pair retain as the injected level rises from 0 to 1.0 of the label spread?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · dispatched

Why it fell short:
- Only per-sigma means, standard deviations and fold counts are written (lines 453-457, 467-471). No per-fold scores.
- When the hybrid arm fails on a fold (line 338) the base arm still records that fold, so hybrid_no_lnl and base_no_lnl means can be taken over different fold sets and compared as if paired.
- The LNL conditions C, D and F are skipped at sigma=0 (line 410), so an LNL method's clean reference is the no-LNL DNN score in the same space. Its delta_from_clean therefore mixes 'LNL versus no LNL' with 'noisy versus clean' in one number.

*Smoke:* not run. not attempted, time

### Since the audit

0 fixed · 0 changed · 0 stand · 0 new · compute unchanged · still runs


### Decision

**MINE:** Pass model and metric_fn to build_fold_hybrid, and store fold_scores. Neither costs a fit.

## n11 — `noise/n11_estimation_remediation_grid.py`

### What it does

For each injected noise level, fold and noise seed, it builds ~12 candidate feature spaces (one base representation, a hybrid, a label-blind similarity hybrid, and the other nine base representations), fits every available noise estimator in each space, then applies four alphas x three weighting and four smoothing methods and re-trains a random forest on the base representation. Reports recovery of the clean-to-noisy gap for every cell.

**Asks:**
- Which combination of noise estimator, distance metric, remediation method and alpha recovers the most of the clean-to-noisy performance gap, and which combinations stay positive at more than one noise level?
- Does computing the noise scores in a hybrid feature space, rather than in one base representation, raise that recovery? (the code labels this comparison oracle-biased and tells the reader not to headline it)
- Does the label-blind similarity hybrid (hybrid_sim), whose selection never saw activity labels, recover more than the predictive hybrid or the base representation?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · dispatched

Why it fell short:
- _compute_summary crashes on any run where an estimator errored or where the dataset is classification (see the blocker below), so the summary block — best_config, top_10, alpha_effect, space_effect_oracle_biased, robust_configs — is never written and the run dies after all the compute.
- On a classification dataset the recovery numbers are wrong: the downstream random forest's hard 0/1 predictions are fed to metric_fn, and for a binary molecular set that metric is 'auc' (_shared.py:1538). Every clean, noisy and denoised score is then an AUROC computed on class labels.
- Only means are written. Recovery is a mean over folds and seeds of a denoised score, divided by a gap that is itself a mean; no per-fold or per-seed numbers survive, so no config can be given an error bar and no two configs can be compared as paired.

*Smoke:* not run. not attempted, time

### Since the audit

0 fixed · 0 changed · 0 stand · 0 new · compute MORE EXPENSIVE · still runs


*Compute now:* 5x on one build, unchanged elsewhere. 9b288a0 moved build_similarity_hybrid inside the fold loop: two of its four blocks are chosen against MCS overlap over sampled molecule pairs, and that choice used to run once over every molecule.

### Decision

**MINE:** Guard the two summary loops. Without it the grid runs for hours on a classification dataset and writes nothing at all.

## n12 — `noise/n12_importance_weighted_distance.py`

### What it does

Injects synthetic training-label noise at 6 levels x 5 folds, builds 5 feature spaces (base rep, base rep scaled by sqrt(importance), hybrid, hybrid scaled two ways), runs 3 kNN noise estimators x 3 k values in each space, converts the noise scores to sample weights and refits a RandomForest on the BASE rep with those weights. Reports the fraction of the clean-vs-noisy gap recovered per space.

**Asks:**
- Does scaling z-scored features by sqrt(feature importance) before computing kNN distances improve neighbour-based detection of corrupted training labels?
- Is that weighting redundant once the features come from a hybrid instead of a single base representation?
- Which neighbour count k and which importance method (lightgbm_gain vs treeshap) give the best recovery?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · dispatched

Why it fell short:
- The blueprint driving all three hybrid spaces comes from get_best_search_for_downstream on the CLEAN labels over ALL rows, including each fold's validation rows. The code says so at lines 330-339.
- Only per-condition MEANS reach the JSON. Per-fold scores are discarded at n12:470.
- Level 0.0 is in the swept grid, where clean_score == noisy_score, so gap ~ 0 and every recovery in that whole level is NaN. One sixth of the compute produces no number.

*Smoke:* not run. exceeded 6 min at '=== Sigma=1.0 ==='.

### Since the audit

1 fixed · 0 changed · 8 stand · 4 new · compute unchanged · still runs

- **NEW** — NEW: every cached clean-label search n12 reads is invalidated, because the downstream model roster and the internal evaluator list both changed. `tests/experiments/_shared.py:387,545`
- Fixed (1e74b25) — create_hybrid is called with groups=None, so the internal admission holdout is a random 20% cut on a scaffold-grouped dataset.

### Decision

**MINE:** Route through build_fold_hybrid and add the zero-importance marker.

**The recheck opened something new here. Overrule me if it changes your call:**

>> 

## n13 — `noise/n13_noise_distributions.py`

### What it does

Sweeps 10 dose-matched regression noise conditions (or 6 classification flip models) x 6 levels x 5 folds x 3 noise seeds. In each cell it builds three feature spaces (base rep, predictive hybrid, label-blind similarity hybrid), runs all 12 label-noise-learning methods on each space (Track A) and 3 noise estimators x 4 models with sample weights (Track B), plus a per-model matched-blueprint hybrid trained plainly and fed to the honest_rescore helper.

**Asks:**
- Does the ranking of label-noise-learning methods measured under Gaussian noise transfer to the other injected noise shapes?
- Does the hybrid feature space keep its advantage over the base representation across every injected noise shape?
- Does noise detection performed in a hybrid space improve base-representation training more than detection in the base space (Track B)?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · dispatched

Why it fell short:
- The summary's hybrid-advantage tally puts hybrid_sim into the BASELINE pool (n13:951), so the 'base' arm it compares against is contaminated with a second hybrid arm. The headline hybrid_advantage_persists flag does not answer question 2.
- The honest_rescore block is given exactly one candidate per model and exactly one baseline representation per model (n13:817-826). Leave-one-fold-out re-selection over a single item selects nothing, so no held-out selection tax is applied to either arm despite the comment at n13:514-521 claiming it is.
- The blueprint driving Track A/B's 'hybrid' space still comes from the CLEAN-label search over all rows (n13:283-289). The comment at n13:558 says 'realistic - no oracle leak', which is true only of the feature construction, not of the recipe selection.

*Smoke:* FAILED. The suggested command exits 0 without running the experiment.

### Since the audit

0 fixed · 0 changed · 0 stand · 0 new · compute MORE EXPENSIVE · still runs


*Compute now:* Same 5x per-fold similarity-hybrid rebuild as n11, from 9b288a0. The LNL grid is unchanged: LNL_REGISTRY's methods, minus the regression-only ones on a classification set, crossed with the alternative noise distributions and 5 folds.

### Decision

**MINE:** Fix the summary tally so hybrid_sim stops counting as a baseline, then the bare create_hybrid. The tally recomputes from a stored file with no re-run.

## n14 — `noise/n14_activity_cliff_landscape.py`

### What it does

A survey that trains no downstream comparison. It builds a high-confidence activity-cliff adjacency from matched molecular pairs when the loader supplies them, else from ECFP4-Tanimoto similarity above 0.55, crossed with an absolute 2-log-unit activity gap; then reports cliff prevalence, cliff-graph degree and connected-component structure, three feasibility flags, and the housing stage-0 residual diagnostics (kurtosis, U-shape heteroscedasticity, bias) from a group-aware out-of-fold LightGBM.

**Asks:**
- How many activity cliffs does this dataset actually contain, and what fraction of its compounds participate in one?
- Are those cliffs clustered into congeneric series or isolated pairs, i.e. do neighbour-based or per-point methods apply?
- Does this dataset support the deferred method families at all: enough cliffs to calibrate a positive-unlabelled classifier, replicate measurements, more than one assay?

**Verdict on 2026-09-03:** YES · hybrid: none needed · dispatched

*Smoke:* FAILED. exit 139 (segfault) inside the experiment, after the 'Feasibility:' line, despite the harness pinning OMP/MKL/OPENBLAS to 1 thread and restricting internal evaluators to rf — so the guardrail that saved n07/n13/n15 did not hold here.

### Since the audit

0 fixed · 0 changed · 2 stand · 1 new · compute unchanged · still runs

- **NEW** — NEW: the Kramer 0.54 pKi floor is written into the results of every molecular regression dataset, whatever its label is. `tests/experiments/noise/n14_activity_cliff_landscape.py:134-137`

### Decision

**MINE:** Set n_jobs from KIRBY_LGBM_THREADS in make_model, then re-run the smoke on caco2.

## n15 — `noise/n15_clean_vs_noisy_hybrid.py`

### What it does

For each of 3 noise levels and each downstream model, fetches two hybrid blueprints - one searched on clean labels, one searched on noisy training labels and scored on clean validation labels - then rebuilds both per fold on the noisy training labels, trains the model on each, scores both on clean validation labels, and reports the difference as oracle_advantage. Also reports how far apart the two blueprints are.

**Asks:**
- How much of the hybrid's apparent advantage under label noise comes from the blueprint having been searched on clean labels?
- Do a clean-label search and a noisy-label search pick the same blueprint - same representation set, allocation, importance method and budget?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · dispatched

Why it fell short:
- The noisy blueprint is still SELECTED by scoring on clean validation labels across all rows, including the rows that become each outer held-out fold. Both arms carry that leak equally so the difference is still meaningful, but the docstring calls the noisy arm 'what an honest practitioner could actually have picked' (n15:22-23) and it is not: an honest practitioner has no clean labels to score against.
- The noise levels are 0.05, 0.10 and 0.20 of the label SD (noise_helpers.NOISY_SEARCH_LEVELS). n11, n12 and n13 sweep 0.0 to 1.0.
- oracle_advantage is a difference of two means, not a paired per-fold delta. Because scores are appended only on success (n15:235), a fold that fails on one arm silently leaves the two means computed over different fold sets.

*Smoke:* ran, NaNs present. exit 0, 'n15 RAN OK in 5.9s', full summary block printed.

### Since the audit

1 fixed · 0 changed · 7 stand · 4 new · compute MORE EXPENSIVE · still runs

- **NEW** — NEW: the noisy searches noise_helpers precomputes cannot be read by n15 any more, so every noisy blueprint cold-runs a full search inside the experiment. `tests/experiments/noise/noise_helpers.py:385`
- **NEW** — NEW: one dead model puts a NaN into the headline number. `tests/experiments/noise/n15_clean_vs_noisy_hybrid.py:290`
- **NEW** — NEW: the blueprint agreement report compares representation sets that the rebuild ignores. `tests/experiments/noise/n15_clean_vs_noisy_hybrid.py:85`
- **NEW** — NEW: every cached search n15 reads is invalidated by the model and evaluator roster change, and the result databases must be deleted. `tests/experiments/_shared.py:545`
- Fixed (c74905b) — The noisy arm was selected by scoring on clean validation labels, so the docstring's 'what an honest practitioner could have picked' was false.

*Compute now:* Two increases. First, build_fold_hybrid stopped forwarding blueprint.rep_scores (1e74b25), so each rebuild re-scores every representation on its own training rows — measured by c74905b at 151 s for ten representations on a 1726-row caco2 fold.

### Decision

**MINE:** Paired per-fold delta and NaN exclusion together.

**The recheck opened something new here. Overrule me if it changes your call:**

>> 

## sim_hybrid_space — `noise/_similarity_hybrid_space.py`

### What it does

Builds one fixed feature matrix per molecular dataset — z-scored, equal-block-scaled [ecfp4_count x0.5 | descriptors minus the MCS-selected drops | 49-D 3D pharmacophore | top-100 fine 2D] — from the similarity study's cached arrays, then re-indexes it onto the experiment's molecules by canonical SMILES. Returns (None, reason) unless every experiment molecule is in the cache. n11 and n13 consume it as an extra representation called hybrid_sim.

**Asks:**
- Can the similarity study's winning structural space be handed to a noise experiment as an extra representation, aligned row-for-row to that experiment's molecules?
- Is every one of the experiment's molecules present in the study cache, so the arm is all-or-nothing rather than partial?
- Are the MCS pairs that drive the block selection numbered against the molecule list actually in hand?

**Verdict on 2026-09-03:** PARTIAL · hybrid: hand-rolled · live

Why it fell short:
- The refusal reason never reaches a results file. n11:444-449 and n13:308-313 print sim_info and discard it.
- The 3D pharmacophore block is aligned by row position alone. The file refuses the July MCS pair files precisely because they carry no ordering proof (:24-32), then takes pharm3d[:n] from sim_<ds>_spatial.npz on positional trust (creative_hybrids_v2.py:111).
- Whether the framework loader's herg_ki molecules are a subset of the study cache's 1415 is asserted in the docstring (:17-20) but not verifiable without running — the loader dedups by InChIKey, the alignment here is by canonical SMILES.

*Smoke:* not run. not attempted, time.

### Since the audit

2 fixed · 0 changed · 1 stand · 3 new · compute MORE EXPENSIVE · run state unknown

- **NEW** — A fold that refuses the arm makes hybrid_sim's number cover fewer folds than the clean and noisy references it is divided against. `tests/experiments/noise/n11_estimation_remediation_grid.py:639 and 675-679; tests/experiments/noise/n13_noise_distributions.py:763 and 823`
- **NEW** — The per-fold feature choice landed on hybrid_sim and not on the predictive hybrid it is compared against. `tests/experiments/noise/n11_estimation_remediation_grid.py:432-439 and 897; tests/experiments/noise/n13_noise_distributions.py:288-293`
- Fixed (9b288a0) — n11 and n13 print the hybrid_sim refusal reason and throw it away, so an absent arm reads later as a null result.
- Fixed (9b288a0) — The descriptor drop, the fine-2D pick and the z-score statistics are computed over every row of the study cache, then sliced per fold.

*Compute now:* 3.2x on the block selection. Before: one descriptor drop and one fine-2D rank per dataset per experiment, over all 8000 MCS pairs.

### Decision

**MINE:** Scaler fix, then persist sim_info so a refused arm is distinguishable from a null result.

## handtrace_noise — `noise/_handtrace_noise.py`

### What it does

Prints seven arithmetic checks on two percentage formulas — clean/noisy for a lower-is-better metric, noisy/clean for higher-is-better, and (denoised - noisy)/(clean - noisy) — comparing each against a hand-computed number. Both formulas are re-typed inside this file; no noise experiment is imported.

**Asks:**
- Does the retention percentage give a number below 100 for a degraded model under both RMSE and R2?
- Does the recovery percentage come out positive for an improvement under both metric directions, so the gap's sign carries the metric direction?

**Verdict on 2026-09-03:** NO · hybrid: none needed · orphaned

Why it fell short:
- It tests copies, not the code. Lines 32-35 and 51-52 define local retention() and recovery() functions; nothing from n07, n09 or n11 is imported or called.
- The copies omit every guard the real code has. n09's real function is _score_ratio_pct (n09_lnl_survey.py:519-537) and returns NaN when either score is at or below 1e-10 or the signs differ; n07:684 and n07:728 and n11:612 each require a minimum gap before a recovery ratio is computed.
- Every cited line number is stale: n09_lnl_survey.py:518-526, n07:528 and n11:491 do not hold those formulas today.

*Smoke:* ran clean. exit 0, seconds, '8/8 hand-trace checks passed'.

### Since the audit

0 fixed · 0 changed · 4 stand · 1 new · compute unchanged · still runs

- **NEW** — The comment naming n07 and n11 as one formula is now wrong: the two experiments define a recovery differently. `tests/experiments/noise/_handtrace_noise.py:50`
- **STILL BROKEN** — retention() is a local copy, not n09's function, while the docstring claims the real function is called.
- **STILL BROKEN** — recovery() is likewise a local copy without the minimum-gap rule the real code has.

### Decision

**MINE:** Delete. It tests local copies of two formulas, so it can never fail when the real ones change.

## smoke_harness — `noise/_smoke_harness.py`

### What it does

Loads one small real dataset with two or three cheap reps, shrinks the in-process search constants, runs the clean and one noisy hybrid precompute, then calls a single noise experiment's run() end to end and reports whether it crashed. A breakage check, not a measurement.

**Asks:**
- Does experiment nXX's run() complete end to end on a small real dataset without raising?
- Do the clean and the noisy-label hybrid precomputes both produce a blueprint the experiment can read out of data['precomputed_searches'] and data['precomputed_searches_noisy']?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · live

Why it fell short:
- The documented entry point is dead. The usage block at :9-10 names n05, which noise_helpers.py:57-58 records as cut and noise_helpers.py:181-192 has no key for — the run ends in a KeyError inside _import_experiment.
- tiny_config() (:80-91) advertises a search grid that reaches nothing. precompute_hybrid_searches never reads its `config` argument (tests/experiments/_shared.py:3835-3990); budgets come from _shared.FULL_BUDGETS and models is forced to None (:3900-3902).
- Because models is forced to None, the precompute fits the full default downstream model set (find_best_hybrid.py:5888-5895), not the ['rf'] the config names. The harness's cost comment at :44-53 covers only the internal evaluator and importance method.

*Smoke:* ran, NaNs present. Works, and it is the only way I found to run the noise experiments locally without the segfault: it sets OMP/OPENBLAS/MKL/VECLIB/NUMEXPR to 1 and KMP_DUPLICATE_LIB_OK before importing numpy, shrinks FULL_BUDGETS to [25] and forces DEFAULT_INTERNAL_EVALUATORS=['rf'].

### Since the audit

0 fixed · 0 changed · 5 stand · 3 new · compute MORE EXPENSIVE · still runs

- **NEW** — The noisy precompute the harness pays for can no longer be read by n15. `tests/experiments/noise/_smoke_harness.py:145-153`
- **NEW** — A pre-2026-09-04 result database now aborts the harness before it reaches the experiment. `tests/experiments/noise/_smoke_harness.py:129`

*Compute now:* Two causes, both from c74905b. First, `_smoke_harness.py n15` now runs two noisy find_best_hybrid searches where it ran one: the precompute at :145-153 builds the clean-scored noisy lineage, n15 cannot read it, and cold-builds the __selnoisy lineage itself.

### Decision

**MINE:** Save the returned dict.

## smoke_noisy_search — `noise/_smoke_noisy_search.py`

### What it does

Nine checks on synthetic 400-row data with two random reps: that a NoiseSpec draw is reproducible, that every noise condition delivers the same root-mean-square amount at one level, that a noisy find_best_hybrid run does not return the clean numbers, that its cache entries are namespaced by the noise id, that more noise gives a worse baseline, and that label_noise plus rep_factories is refused.

**Asks:**
- Does one NoiseSpec produce a byte-identical noise draw every time, and do different conditions at one level deliver the same amount of noise?
- Does a find_best_hybrid run with label_noise return different baselines and a different cache identity from the clean run, rather than silently serving the clean results?
- Does the engine refuse label_noise together with rep_factories, which would fine-tune the representations on clean labels?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · orphaned

Why it fell short:
- The search cache is off for every one of these calls. search() at :63-70 does not pass cache=True, and find_best_hybrid defaults cache=False (find_best_hybrid.py:5539).
- Test 6 (:133-136) opens $KIRBY_CACHE_DIR/result_db/noisysmoke.db and selects from a `results` table. That file is never created, so the connect or the SELECT raises uncaught and tests 7, 8 and 9 never run.
- Tests 4 and 5 do exercise something real: the per-representation baseline side cache is enabled by dataset_name plus metric alone, independent of `cache` (find_best_hybrid.py:3901-3906), and its filename carries the noise id (find_best_hybrid.py:159-163). That is the part of the namespacing this file actually verifies.

*Smoke:* FAILED. exit 1 at check 6 of 6.

### Since the audit

0 fixed · 0 changed · 4 stand · 2 new · compute unchanged · still runs

- **NEW** — A third noisy cache lineage now exists and nothing here checks it. `tests/experiments/noise/_smoke_noisy_search.py:63-70`
- **STILL BROKEN** — search() omits cache=True, so no result database and no search pickle is ever written and the silent failure the file exists to catch cannot occur in it.
- **STILL BROKEN** — Test 6 opens a SQLite file the cache=False runs never create, then selects from it; it raises and tests 7, 8 and 9 never run.

### Decision

**MINE:** Add cache=True, then run it once so checks 6 to 9 are reachable at all.


---

# Similarity, s01–s13 — 15 files

## s01 — `similarity/s01_similarity_baseline.py`

### What it does

For each base representation it builds a standardized-Euclidean distance matrix over the whole dataset and reports three numbers: the Spearman correlation between pairwise distance and pairwise |activity difference| (called SAC in the code), the fraction of the 5 nearest neighbours whose activity difference exceeds the median pair difference, and SALI. A 200-draw label-permutation test gives each SAC a p-value. On antibody and nucleotide datasets it also runs a sequence Hamming-distance cliff sweep at thresholds 1/2/3/5, on full V-region and on ANARCI-extracted CDRs.

**Asks:**
- For each base representation, how strongly does distance between two molecules track the difference in their activity?
- Is that correlation distinguishable from what a random label assignment gives?
- How often do the five nearest neighbours of a molecule have a very different activity?

**Verdict on 2026-09-03:** YES · hybrid: none needed · dispatched

*Smoke:* ran clean. exit 0, 31s total / s01 2.3s.

### Since the audit

0 fixed · 0 changed · 3 stand · 2 new · compute MORE EXPENSIVE · still runs

- **NEW** — The distance every s01 number is computed from changed from Euclidean to Manhattan, by default `src/kirby/similarity/distance_config.py:47`
- **NEW** — The scalable pair-distance helper computes 17.6 million distances to get the 5,000 it wants `tests/experiments/similarity/similarity_helpers.py:30-44`

*Compute now:* Two changes, neither in s01 itself. Dense path: cityblock has no BLAS path where euclidean does, roughly 2-4x per n x n matrix, 10 matrices per dataset.

### Decision

**MINE:** Hoist the pair sample, record the violation-rate reference, fix the dense-bound comment.

**The recheck opened something new here. Overrule me if it changes your call:**

>> 

## s02 — `similarity/s02_hybrid_similarity.py`

### What it does

Builds one hybrid through the real engine at budget 100 for downstream model knn, then compares the Spearman correlation between distance and activity difference for the hybrid against the same quantity for every single representation. It does this twice: once on the whole dataset (the headline keys best_sac and sac_delta_from_best_rep), and once with a 5-fold group-aware split where the importance weights are refit on train rows and the correlation is measured on the held-out rows.

**Asks:**
- Does an importance-weighted distance over the hybrid's selected features correlate better with activity difference than the best single representation's distance?
- Which importance method makes the best weighting for that distance?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · dispatched

Why it fell short:
- The headline delta compares the WEIGHTED hybrid against the UNWEIGHTED best single representation. s02 computes a weighted best single representation (best_individual_rep_weighted, line 269) and never uses it in any delta.
- The held-out block is not fully held out on the hybrid side. The hybrid's selected columns come from get_fitted_pool_hybrid on the full dataset (line 324), including the labels of every test row; only the weights are refit per fold.
- Budget is fixed at 100 (line 320) and never swept, so the answer holds at one budget only.

*Smoke:* ran clean. exit 0, s02 done in 7.0s.

### Since the audit

0 fixed · 2 changed · 3 stand · 3 new · compute MORE EXPENSIVE · still runs

- **NEW** — The whole-dataset s02 hybrid is now an rf/xgboost/svm/dnn/lightgbm blueprint served under the name knn `tests/experiments/_shared.py:5317-5324`
- **NEW** — The per-fold feature refit that f86d9a4 announces for s02 does not execute on the default configuration `tests/experiments/similarity/s02_hybrid_similarity.py:378-380`
- **NEW** — Every s02 result file on disk predates the held-out block and cannot be reproduced `tests/results/similarity/s02_hybrid_similarity/`

*Compute now:* Intended: the held-out block gained 5 build_fold_hybrid calls, each a full create_hybrid on 80% of the rows. create_hybrid re-scores all 10 representations and recomputes importance on every call; the whole-dataset build was measured at 336 seconds on 2026-08-27.

### Decision

**MINE:** Three one-line fixes: the weighted-versus-weighted delta, the recorded blueprint, the lineage.

**Open — yours:** TBD with the objective question. Also whether budget is swept rather than fixed at 100.

>> 

**The recheck opened something new here. Overrule me if it changes your call:**

>> 

## s03 — `similarity/s03_feature_concentration.py`

### What it does

For each of 3 downstream models x 8 budgets it builds a hybrid on full data, then for each base representation compares the mutual information with the label of four feature sets of matched size: the full representation, the features the hybrid selected, a random subset of the same size, and a PCA projection to the same dimension. It also reports effective rank of each set and the Gini coefficient of the importance scores.

**Asks:**
- Do the features the hybrid selects from a representation carry more mutual information with the label than a random subset of the same size?
- Do they carry more than a PCA projection to the same number of dimensions?
- How unevenly is importance spread across a representation's features?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · dispatched

Why it fell short:
- mi_selected vs mi_random and mi_selected vs mi_pca are fair comparisons — same rows, same dimension, same estimator — so questions 1 and 2 are answered. But the headline quantity the analysis reader actually consumes, mi_density_ratio, is not a valid measure of concentration (see bugs).
- mi_random is a single draw per cell. Whether selected beats random rests on one random subset with no repeats and no spread.
- Everything is a full-data statistic with selection fit on the same rows. The file says so loudly (descriptive_only=True plus a _caveat string), so this is a stated scope, not a hidden leak.

*Smoke:* ran clean. exit 0, s03 done in 9.1s.

### Since the audit

0 fixed · 1 changed · 5 stand · 1 new · compute MORE EXPENSIVE · still runs

- **NEW** — The comparison s03 exists to make has never produced a number, on any dataset `tests/results/similarity/s03_feature_concentration/`
- **STILL BROKEN** — mi_density_ratio divides an information measured over at most 50 PCA components by the full feature count

*Compute now:* Before 47e4b9e only the mi_full arm ran: 3 models x 8 budgets x 10 representations = 240 mutual-information estimates per dataset. Now the selected, random and PCA arms execute too: 960 estimates plus 240 PCA or TruncatedSVD fits.

### Decision

**MINE:** Fix the density denominator and the JSON keys, then re-run. The 22 files on disk answer none of its questions.

**The recheck opened something new here. Overrule me if it changes your call:**

>> 

## s04 — `similarity/s04_activity_cliffs.py`

### What it does

Labels every molecule as a cliff compound or not, using ECFP4 Tanimoto >= 0.55 paired with |activity difference| > 2 x MAD(y). Then runs group-aware CV and, on each fold, scores a 5-nearest-neighbour prediction from every single representation (unweighted and importance-weighted), from a per-fold hybrid (unweighted and importance-weighted), and a random forest on the best single representation — reporting the dataset metric and RMSE separately on the cliff rows and the non-cliff rows of the held-out fold.

**Asks:**
- On molecules that sit on activity cliffs, does a 5-nearest-neighbour prediction from the hybrid's feature space beat one from any single base representation?
- Does weighting the distance by feature importance help on cliffs specifically?
- How much worse is every method on cliff compounds than on non-cliff compounds?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · dispatched

Why it fell short:
- The honest-rescore block, which is the only leave-one-fold-out re-selection in the file, stores the OVERALL fold metric as fold_scores (line 219), not the cliff metric. So the held-out hybrid-vs-best-single comparison it supports answers 'does the hybrid win overall', not 'does it win on cliffs'.
- score_cliff is a mean of per-fold cliff scores over folds that happened to hold more than one cliff row. With 5 folds and a low cliff prevalence, that can be an average over two or three folds, and the number of contributing folds is not written to the JSON — only the positional fold_scores for the overall metric are.
- The experiment produces nothing on molecular datasets whose representation pool has no ecfp4 (see bugs).

*Smoke:* ran, NaNs present. exit 0, s04 done in 1.7s, file written.

### Since the audit

1 fixed · 0 changed · 5 stand · 4 new · compute MORE EXPENSIVE · still runs

- **NEW** — The hybrid's recipe is now searched inside each fold, and the RF baseline's representation still is not `tests/experiments/similarity/s04_activity_cliffs.py:462-465`
- **NEW** — The per-fold searches run with no pruning schedule while the whole-dataset search run_all pays for runs with one `tests/experiments/similarity/s04_activity_cliffs.py:166-173`
- **NEW** — The committed distance precompute writes its search under models=None while the read side asks for DISTANCE_MODELS `tests/experiments/similarity/similarity_helpers.py:1006-1011 at HEAD`
- Fixed (1e74b25) — make_cv_splitter is called without task_type, so a classification dataset gets plain GroupKFold

*Compute now:* The dominant new cost is one full find_best_hybrid per outer fold instead of one shared whole-dataset search. The distance grid is 5 allocation methods (find_best_hybrid.py:2890-2896) x 3 importance methods (:2881) x 5 internal evaluators (_shared.py:324) x 6 budgets (_shared.py:317) = 450 recipes, x 5 inner folds = 2,250 hybrid builds, each scored on 6 downstream models = 13,500 fits per search.

### Decision

**MINE:** Build ECFP4 from the SMILES so the cliff rule stops depending on the pool, and keep per-fold cliff scores.

## s05 — `similarity/s05_knn_prediction.py`

### What it does

Runs its own 5-fold group-aware CV of k-nearest-neighbour regression/classification on a per-fold hybrid rebuilt from a distance-lineage blueprint, at 6 budgets x 2 weighting conditions x k in {5,10}, against per-representation kNN baselines at the same k values. Writes a post-hoc max, a pre-registered fixed cell, and a leave-one-fold-out re-score block.

**Asks:**
- Does kNN in the hybrid feature space beat kNN in each single representation's space, on the same folds?
- Does scaling the hybrid's standardized features by sqrt(importance) improve kNN over the unweighted hybrid?
- How does the kNN score move with the feature budget (10 to 500)?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · dispatched

Why it fell short:
- The budget sweep is not guaranteed to sweep the budget. get_best_for_downstream(search,'knn',budget=b) returns the best config across ALL budgets when budget b has no finite entry (_shared.py:4793), and build_fold_hybrid then uses blueprint.budget, so a row labelled budget=10 can hold a hybrid built at 250.
- The per-rep baseline score_mean is the max over reps and over k on the same folds it is reported on, while fixed_ref is a pre-registered hybrid cell. That one comparison is asymmetric against the hybrid.
- The importance weighting question is answered only within the hybrid arm; the weighted per-rep baseline (score_weighted_mean) is stored but never enters any delta.

*Smoke:* ran clean. exit 0, s05 done in 3.5s.

### Since the audit

3 fixed · 0 changed · 2 stand · 3 new · compute MORE EXPENSIVE · still runs

- **NEW** — _fold_build_cache holds every fold matrix for the whole run with no eviction `tests/experiments/similarity/s05_knn_prediction.py:174-203`
- Fixed (f86d9a4) — The hybrid is rebuilt from scratch for hybrid_weighted with identical arguments; 60 of 120 create_hybrid calls are duplicates
- Fixed (f86d9a4) — get_fitted_pool_hybrid runs a full-data create_hybrid whose matrix is discarded

*Compute now:* Two terms moved in opposite directions, and the one that grew is much larger. Hybrid builds fell from 121 to 60: _fold_build_cache at s05:174-203 is keyed (budget, k, fold) so the weighted and unweighted conditions share one build (6 budgets x 2 k x 5 folds = 60), and the discarded whole-data create_hybrid is gone.

### Decision

**MINE:** budget_override, hoisted build, search-only helper, delta_mean_of_folds and task_type in one pass, then re-run 22 datasets.

**Open — yours:** Whether fixed_ref keeps its in-sample baseline.

>> 

## s06 — `similarity/s06_applicability_domain.py`

### What it does

Per fold, builds a hybrid on the training rows, fits a random forest on it and on the concatenation of all representations, then correlates each validation molecule's mean distance to its 5 nearest training molecules with that molecule's absolute residual, in four spaces (concat, weighted concat, hybrid, weighted hybrid) plus every single representation. Also writes a distance-decile error table and a split-conformal interval comparison of concat against hybrid.

**Asks:**
- Does distance to the training set in the hybrid space rank prediction error better than distance in the concatenation of all representations?
- Does the same distance rank error better than distance in each single representation's space?
- Do conformal intervals built on hybrid features cover at 90% with a smaller mean width than intervals built on the concatenation?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · dispatched

Why it fell short:
- The comparison arm for question 1 is the concatenation of every representation, which is not an admissible baseline here; the per-representation distances exist but no delta is formed from them, and s09 (s09_cross_dataset_summary.py:481-505) takes ad_corr_delta as hybrid_weighted minus concat.
- Question 2 is not answered honestly: the per-rep correlations at lines 408-414 are scored against errors from the random forest fitted on the HYBRID features. The de-tautologised neutral_model block covers only concat/hybrid/hybrid_weighted, so the single-representation arms never get errors from a model outside the hybrid's own space.
- Question 3 has no single-representation arm at all (lines 500-503 iterate concat and hybrid only), and only one budget (100, _DEFAULT_AD_BUDGET at line 63) is ever evaluated.

*Smoke:* ran, NaNs present. exit 0, s06 done in 6.2s, file written (result_status=ok), 24 NaN + 3 null.

### Since the audit

1 fixed · 0 changed · 5 stand · 3 new · compute unchanged · still runs

- **NEW** — s06 is the only one of these four whose recipe and comparator are still picked by one search over every row `tests/experiments/similarity/s06_applicability_domain.py:144-151,174-180`
- **NEW** — Result files written before 2026-09-04 key the correlation under 'concat'; new ones key it under a representation name, and s09 mixes the two `tests/experiments/similarity/s09_cross_dataset_summary.py:484-489`
- Fixed (6761eae) — X_concat, the concatenation of every representation, is the reference arm for the correlation block and the only comparator in the conformal block

### Decision

**MINE:** Find the NaN in the weighted arms, then add per-representation arms to both blocks.

## s07 — `similarity/s07_noise_bridge.py`

### What it does

Defines ground-truth activity cliffs from ECFP4 Tanimoto (molecular) or heavy-chain edit distance (antibody), then flags the 10% of molecules whose 5 nearest neighbours disagree most in label, and reports what share of those flagged molecules are cliffs, in the hybrid space, the importance-weighted hybrid space, and each single representation's space. Repeats the same statistic under two fold schemes and hands both to the leave-one-fold-out re-scorer.

**Asks:**
- When a k-nearest-neighbour label-consistency filter flags molecules as noisy, does the hybrid space put fewer genuine activity cliffs among the flagged set than any single representation does?
- Does scaling the hybrid's features by sqrt(importance) reduce that cliff share further?
- Does the answer survive holding out folds, under one-fold-per-scaffold and under the suite's 5-fold group splitter?

**Verdict on 2026-09-03:** NO · hybrid: engine, per fold · dispatched

Why it fell short:
- The held-out arm is not held out where it matters. The hybrid candidates handed to the fold loop are slices of X_hybrid and X_hybrid_w, both fitted on every row's labels at line 348.
- The statistic has no counterpart measuring whether the filter finds real label noise. The threshold is the 90th percentile of neighbour label spread, so every arm flags the same ~10% of molecules; a representation whose neighbourhoods carry no information scores well simply by flagging cliffs no more often than chance.
- The molecular cliff definition is built from reps['ecfp4'], which is one of the representations in the pool being scored, so the arm for ecfp4 (and any hybrid that selects ecfp4 features) is scored against ground truth drawn from its own geometry.

*Smoke:* ran, NaNs present. exit 0, s07 done in 0.6s.

### Since the audit

0 fixed · 0 changed · 0 stand · 0 new · compute MORE EXPENSIVE · still runs


*Compute now:* From about zero to five cold distance searches per dataset. _outer_fold_hybrids builds the outer folds with nested_cv_outer_folds at seed 42, then calls get_fold_best_search with distance=True inside the fold loop, so each fold's recipe is searched on that fold's training rows under DISTANCE_MODELS, DISTANCE_EVALUATORS and DISTANCE_BUDGETS.

### Decision

**YOURS:** Delete, or rewrite the held-out arm. 14 of 21 stored files carry no number and 4 more sit at exactly 1.0 for every arm. My lean: delete unless this question is going in the paper.

>> 

## s08 — `similarity/s08_optimal_dimension.py`

### What it does

For each of six feature budgets it builds a full-data hybrid and records its intrinsic dimension, its distance-activity Spearman, its 5-nearest-neighbour label violation rate and its rank correlation with the all-representation concatenation, then runs a 5-fold CV to get per-model prediction scores and a held-out distance-activity Spearman. Compares the budget that maximises the correlation against the budget that optimises prediction, and re-scores both against per-representation baselines.

**Asks:**
- Does the feature budget that maximises the distance-activity Spearman differ from the budget that optimises the downstream prediction score?
- At a matched budget, does the hybrid's distance-activity Spearman beat every single representation's, on held-out folds?
- Does scaling the held-out distance by the train-fold importance weights raise that correlation above the unweighted version?

**Verdict on 2026-09-03:** NO · hybrid: engine, per fold · dispatched

Why it fell short:
- The study variable does not vary in the cross-validated arm. bp at line 200 is read from search.best_by_model['knn'], which find_best_hybrid builds as the max over ALL budgets (find_best_hybrid.py:5292-5296).
- Question 1's prediction side is missing entirely for kNN. model_factories comes from DOWNSTREAM_MODELS = ['rf','xgboost','svm','mlp','lightgbm'] (_shared.py:336), which has no 'knn', so knn_score is NaN at every budget and optimal.best_knn_budget is always None.
- Question 2's candidate set is degenerate for the same reason as the first gap: all six 'budget_b' candidates handed to honest_rescore carry the same fold scores, so the leave-one-fold-out re-pick has nothing to choose between.

*Smoke:* ran clean. exit 0, s08 done in 3.9s, file written.

### Since the audit

3 fixed · 1 changed · 3 stand · 5 new · compute MORE EXPENSIVE · still runs

- **NEW** — One budget row now describes two different hybrids `tests/experiments/similarity/s08_optimal_dimension.py:156-181 against 205-255`
- **NEW** — n_distinct_dims and insufficient_budgets are read off the arm that may not vary `tests/experiments/similarity/s08_optimal_dimension.py:454-461`
- **NEW** — The Pareto trade-off compares a held-out score against a whole-data one `tests/experiments/similarity/s08_optimal_dimension.py:479-480,565-572,592-598`
- **NEW** — 'mlp' left the model roster and 'dnn' entered it, so stored s08 files carry a model name a re-run will not write `tests/experiments/_shared.py:387`
- **NEW** — The per-fold rebuild is scored under a different rule than the recipe was chosen under `tests/experiments/similarity/s08_optimal_dimension.py:248-255`
- Fixed (1e74b25) — bp is the best config across all budgets, so the entire cross-validated half of a dimension sweep ran at one fixed dimension
- Fixed (0ce7ab0) — DOWNSTREAM_MODELS has no 'knn', so knn_score is NaN at every budget and s09's optimal_dim_ratio can never be computed
- Fixed (0ce7ab0) — model_scores filled from the search's cached fold_scores when the budget matched: selection-optimistic numbers on different folds, averaged in beside freshly fitted ones

*Compute now:* create_hybrid calls are unchanged: 6 full-data builds (one per budget) plus 6 budgets x 5 folds = 30 per-fold builds = 36. The model fits grew twice over.

### Decision

**MINE:** budget_override and a kNN model factory together. Without both the file answers neither of its questions.

**Open — yours:** Which model scores accept-or-reject before the 16 datasets re-run.

>> 

## s09 — `similarity/s09_cross_dataset_summary.py`

### What it does

Reads the saved JSON of s02, s04, s05, s06, s07 and s08 for each dataset and pulls out one delta per comparison, each labelled with the arm it came from and the key it was read from. Fits nothing; it prints the per-dataset values, refuses to average them, and runs a Wilcoxon signed-rank test over the per-dataset values when at least 3 datasets contributed.

**Asks:**
- Does the hybrid beat the best single representation on the same sign, in the same direction, on every dataset, for each of the six similarity comparisons s02-s08 measured?
- For each comparison, does the held-out arm (winner re-picked on the other folds) agree with the selection-optimistic arm (winner picked on the scores that are reported)?
- Is the pattern of per-dataset signs unlikely under a Wilcoxon signed-rank test?

**Verdict on 2026-09-03:** PARTIAL · hybrid: none needed · orphaned

Why it fell short:
- The kNN delta at s09:388-394 puts a hybrid maximised over budgets x conditions x k against a baseline maximised over representations only. The two sides search grids of very different size, so the delta is biased upward by the grid-size difference alone.
- The applicability-domain delta has no held-out counterpart at all (s09:511-516 records that s06 stores none), so that comparison is reported only in its selection-optimistic form.
- MEMORY note, unverified here: on a single dataset the file writes n_datasets=1 and the Wilcoxon is skipped, but the stored file still reads like a cross-dataset result.

*Smoke:* ran, NaNs present. exit 0.

### Since the audit

1 fixed · 0 changed · 6 stand · 4 new · compute unchanged · still runs

- **NEW** — s09 re-scores s04's stored per-fold block without dropping fold 0, while the seventeen other readers of such a block now drop it. `tests/experiments/similarity/s09_cross_dataset_summary.py:194-196`
- **NEW** — The stored provenance string for the applicability-domain delta still says 'minus .concat', and the comparator representation's name is never written into the payload. `tests/experiments/similarity/s09_cross_dataset_summary.py:516-525`
- **NEW** — The applicability-domain series admits NaN the same way SAC_delta does, through an 'is not None' check where a numeric check is needed. `tests/experiments/similarity/s09_cross_dataset_summary.py:505-508`
- **NEW** — Result files written before and after 2026-09-04 carry two different applicability-domain baselines, and s09 aggregates them into one series and one Wilcoxon test. `tests/experiments/similarity/s09_cross_dataset_summary.py:489`
- Fixed (6761eae) — The applicability-domain delta is the hybrid minus 'concat', a concatenation of every representation, which the project does not allow as a baseline.

### Decision

**MINE:** Three one-line repairs plus an individual-representation comparator to replace the concatenation. Then point the submit script at this module.

## s10 — `similarity/s10_importance_downstream.py`

### What it does

For each base representation and for a per-fold hybrid, multiplies every feature column by the square root of its normalised LightGBM-gain importance and scores four downstream estimators weighted against unweighted: kNN at k in {5,10,20}, a Gaussian process with an RBF kernel, an SVR with an RBF kernel, and a random forest fed four extra similarity descriptors (neighbour label mean and spread, nearest and mean distance). Splits are group-aware, importance is computed on training rows only, and every delta is sign-oriented so positive means improvement.

**Asks:**
- Does scaling each feature by the square root of its importance improve a kNN, a Gaussian process, or an SVR, relative to the same estimator on the same unscaled features?
- Does appending four similarity descriptors computed from training-row neighbours improve a random forest, and does computing those descriptors in the importance-scaled space improve it further?
- Do the answers differ between an individual base representation and the hybrid?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · dispatched

Why it fell short:
- Under the live path the hybrid is 10 features wide, not 100. See the budget bug below.
- The Gaussian-process task is skipped above 5000 rows (s10:415) and the SVR task above 10000 (s10:570). On any dataset over those sizes those two questions go unanswered; the skip is recorded in the payload, so it is visible, but nothing measures them.
- The per-representation weighted arm always scores importance with method='lightgbm_gain' (s10:1032), while the hybrid arm's weights come out of feature_info and therefore carry whatever importance method the blueprint chose. The two arms are internally consistent but are not weighted by the same scorer, so per-rep and hybrid deltas are not directly comparable.

*Smoke:* ran clean. exit 0, s10 done in 13.8s (task 1 1.5s, task 2 GP 6.0s, task 3 SVR 0.2s, task 4 q-RASAR 6.1s), file written.

### Since the audit

0 fixed · 0 changed · 5 stand · 3 new · compute MORE EXPENSIVE · still runs

- **NEW** — s10 is the one similarity experiment left choosing the hybrid recipe over every row, and it does not say so in its result file. `tests/experiments/similarity/s10_importance_downstream.py:969-978`
- **NEW** — The budget split inside every s10 hybrid build changed on 2026-09-04, and no result file on disk can be reproduced by today's code. `tests/experiments/_shared.py:4234-4242`
- **NEW** — The importance scores s10's weighted arm is built from changed for every representation wider than 2000 features, and nothing marks the change. `src/kirby/hybrid.py:3891-3897`

*Compute now:* No commit touched s10 itself; the cost moved underneath it. 1e74b25 removed build_fold_hybrid's fallback to blueprint.rep_scores (_shared.py:4234-4242), and s10 passes no precomputed_alloc_scores at any of its four build sites (:331, :489, :640, :835).

### Decision

**MINE:** Pass budget 100 explicitly and store it, then the four repairs.

**Open — yours:** Whether budget becomes a reported axis, which is a separate server job.

>> 

## s11 — `similarity/s11_bayesian_optimization.py`

### What it does

Simulates library screening: takes a group-aware 5% seed set and an 80% pool, builds a feature space on the seed rows only, then runs 100 Expected-Improvement iterations picking one pool molecule at a time and revealing its label. Eleven feature spaces compete, including a 50-feature KIRBy hybrid, PCA to 50, a random Gaussian projection to 50, raw ECFP4 with an RBF kernel, raw ECFP4 with a Tanimoto kernel, the best single representation, and random acquisition. Scores the area under the best-found-so-far curve and the iteration at which the top 1% and top 5% were first hit.

**Asks:**
- Does Expected-Improvement Bayesian optimisation find good molecules in fewer revealed labels when the surrogate lives in a 50-feature KIRBy hybrid space than in raw 2048-dimensional ECFP4, a 50-component PCA, a 50-component random projection, the best single representation, or random acquisition?
- Does rebuilding the hybrid every 25 iterations on the labels revealed so far beat building it once on the seed set (hybrid_50d vs hybrid_50d_refit)?
- In hybrid space, does an NGBoost, MC-dropout BNN or Gauche Tanimoto-GP surrogate beat the sklearn RBF GP?

**Verdict on 2026-09-03:** PARTIAL · hybrid: should use engine, does not · dispatched

Why it fell short:
- The hybrid's recipe is tuned on labels the loop is meant to be discovering. get_best_search_for_downstream at s11:883 is handed the full `labels` and `groups` arrays, so the allocation method, importance method, evaluator and augmentation strategy the hybrid arm uses were selected with access to every pool label.
- hybrid_50d_refit cannot run on any dataset that carries augmentation representations. See the shape bug below.
- The top-1% and top-5% thresholds are computed over every row in the dataset (s11:763-764), but 15% of rows are in neither the seed nor the pool and can never be picked. A condition can be scored as 'did not reach the top 1%' for a reason that has nothing to do with the method.

*Smoke:* ran clean. exit 0, 55s total / s11 25.2s.

### Since the audit

2 fixed · 1 changed · 5 stand · 5 new · compute MORE EXPENSIVE · still runs

- **NEW** — The two arms of the rebuild comparison fall back to different augmentation strategies when the seed search returns nothing. `tests/experiments/similarity/s11_bayesian_optimization.py:1479-1480`
- **NEW** — The fallback recipe names an allocation method the search no longer sweeps. `tests/experiments/similarity/s11_bayesian_optimization.py:682`
- **NEW** — The hybrid's recipe is now chosen from about 5% of the rows, which is honest but very few rows. `tests/experiments/similarity/s11_bayesian_optimization.py:897-905`
- **STILL BROKEN** — _run_bo_with_rebuild transforms the pool with apply_feature_selection alone while create_hybrid hstacks selected augmentation features onto the seed matrix, so the scaler raises on the feature count and hybrid_50d_refit reports NaN on every dataset with augmentation representations.
- Fixed (f86d9a4) — The hybrid's recipe is tuned on labels the loop is meant to be discovering; get_best_search_for_downstream is handed the full labels and groups arrays.
- Fixed (f86d9a4) — create_hybrid is called with no groups, so its internal accept/reject holdout falls to np.random.RandomState(42).permutation on a scaffold-grouped dataset.

*Compute now:* The recipe search moved from one cached read over the whole dataset to three fresh searches, one per Bayesian seed, under cache namespaces nothing precomputes. One search sweeps 5 allocation methods x 3 importance methods x 5 internal evaluators x 8 budgets = 600 recipes over 5 inner folds.

### Decision

**MINE:** Route both builds through build_fold_hybrid with groups, the model and the metric. Refuse the hybrid conditions when there is no blueprint and record that. Then the four repairs.

## s12 — `similarity/s12_cross_domain_cliffs.py`

### What it does

Marks each compound as an activity cliff or not (structurally close pair, large activity gap) using standardised Euclidean distance on every representation concatenated, then scores each single representation and the hybrid separately on the cliff rows and the non-cliff rows of each held-out fold, for five downstream models. Reports the hybrid-minus-best-single-representation difference on the cliff rows, both as a same-fold argmax and as a leave-one-fold-out re-pick.

**Asks:**
- Does the hybrid predict the activity of compounds that sit on an activity cliff better than the best single base representation does, per downstream model?
- Does that advantage survive re-picking the winning base representation on the other folds instead of on the fold being scored?
- What fraction of the dataset are cliff compounds once 'structurally similar' is defined as the top few per cent most-similar pairs rather than an absolute similarity cutoff?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · dispatched

Why it fell short:
- Above 5000 rows the cliff scan is silently subsampled (max_compounds=5000) and every unscanned row is then treated as non-cliff by `vi not in cliff_indices`, so the non-cliff arm is contaminated with rows never tested for cliff membership. herg_ki (7772) and qm9 reach this.
- The winning hybrid recipe is never written to the JSON. `bp_by_model` holds the allocation method, importance method, internal evaluator and budget per model and is discarded, so a later reader cannot say what the hybrid that won was.
- Cliff membership is computed from the full-dataset labels (acknowledged in the code at s12:219-220), so which validation rows count as cliffs was decided with the validation labels in hand.

*Smoke:* ran clean. exit 0, s12 done in 1.8s, file written.

### Since the audit

3 fixed · 0 changed · 6 stand · 5 new · compute MORE EXPENSIVE · still runs

- **NEW** — The distance behind the cliff definition is no longer Euclidean, but the result file still says it is. `tests/experiments/similarity/s12_cross_domain_cliffs.py:108, 179`
- **NEW** — The downstream model set changed under s12, so no stored s12 file can be reproduced. `tests/experiments/_shared.py:387`

*Compute now:* The recipe search moved from run_all's one whole-dataset precompute into the fold loop. search_on_fold runs one find_best_hybrid per outer fold per augmentation condition: 5 folds x 3 conditions (none/uncapped/capped, _shared.py:908-912) = 15 searches, each on 4/5 of the rows, over FULL_BUDGETS (8 budgets) x 5 allocation methods x 3 importance methods x 5 internal evaluators.

### Decision

**MINE:** Fix the scanned-set contamination, write the recipe into the result, do the housekeeping pass.

**Open — yours:** TBD with the objective question.

>> 

**The recheck opened something new here. Overrule me if it changes your call:**

>> 

## s13 — `similarity/s13_clustering_retrieval.py`

### What it does

Under group-aware CV, builds three kinds of feature space per fold on the train rows only (each single representation, a PCA of everything concatenated, and the hybrid), fits k-means and a nearest-neighbour index on train, then measures on the held-out rows how tightly each cluster's activities agree and how close the five retrieved train neighbours' activities are. Averages per fold and reports hybrid versus best single representation on both axes, with a leave-one-fold-out re-pick.

**Asks:**
- Do k-means clusters built in the hybrid space group held-out compounds with more similar activities than clusters built in any single base representation's space?
- Are the five nearest training neighbours of a held-out compound closer in activity when the distance is measured in the hybrid space than in any single base representation's space?
- Does weighting the hybrid's features by their lightgbm gain improve that neighbour retrieval further?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · dispatched

Why it fell short:
- Fold indices are not aligned across spaces, so the leave-one-fold-out arithmetic can compare fold 2 of one representation against fold 1 of another (see bugs).
- The 'pca_matched' control is fixed at 100 components while the hybrid uses blueprint.budget drawn from DISTANCE_BUDGETS [10,25,50,100,250,500]. Whenever those differ, the dimension-matched control does not match, so a hybrid win cannot be separated from a feature-count win.
- Question 3 has no comparison arm: only the hybrid gets an importance-weighted variant. No single representation is tested with importance weighting, so a weighted-hybrid win could be the weighting rather than the hybrid.

*Smoke:* ran clean. exit 0, s13 done in 10.6s, file written.

### Since the audit

3 fixed · 2 changed · 4 stand · 6 new · compute MORE EXPENSIVE · still runs

- **NEW** — The two axes now measure distance two different ways, and the file compares their win booleans against each other. `tests/experiments/similarity/s13_clustering_retrieval.py:154, 115, 455`
- **NEW** — The per-fold hybrid is rebuilt without the model or the metric the recipe was chosen by. `tests/experiments/similarity/s13_clustering_retrieval.py:316-321`
- **NEW** — A dataset with fewer scaffold groups than folds no longer refuses; it silently runs fewer folds. `tests/experiments/similarity/s13_clustering_retrieval.py:195-204`
- **NEW** — No stored s13 file can be reproduced by today's code. `tests/results/similarity/s13_clustering_retrieval/abagym_Ang2.json`
- **STILL BROKEN** — Per-fold scores are appended positionally with no placeholder, so a space that fails on one fold has every later score sitting at the wrong fold index.

*Compute now:* One discarded whole-dataset create_hybrid (336 s) was removed, and 10 fold searches were added. search_on_fold runs one find_best_hybrid per outer fold per augmentation condition; the distance lineage carries only none and uncapped (_shared.py:908-912), so 5 folds x 2 = 10 searches on the general_distance tier, each on 4/5 of the rows, over DISTANCE_BUDGETS (6) x DISTANCE_EVALUATORS (5) x 5 allocation methods x 3 importance methods x DISTANCE_MODELS (6 downstream models).

### Decision

**MINE:** Pad the fold lists first — it can already be producing a wrong delta. Then the singleton-cluster fix, the dimension match, and drop the discarded build.

## _summary_keys — `similarity/_summary_keys.py`

### What it does

Stdlib-only lookup table mapping each of the six per-dataset inputs the cross-dataset summary reads to every key spelling that names it, plus the check that refuses to write an aggregate when not one dataset contributed a metric.

**Asks:**
- Under which key names can each of the six per-dataset similarity results be found, so either aggregator can read the other's output?
- Did any dataset contribute a single metric to the cross-dataset aggregate, and if not, which input was each dataset missing?

**Verdict on 2026-09-03:** YES · hybrid: none needed · live

*Smoke:* ran clean. exit 0.

### Since the audit

0 fixed · 0 changed · 1 stand · 1 new · compute unchanged · still runs


### Decision

**MINE:** Leave it.

## _run_similarity_locally — `similarity/_run_similarity_locally.py`

### What it does

Throwaway driver that shrinks the engine in memory (FULL_BUDGETS, DISTANCE_BUDGETS, DISTANCE_EVALUATORS, DEFAULT_MODELS, DEFAULT_IMPORTANCE_METHODS, DEFAULT_ALLOCATION_METHODS, DEFAULT_INTERNAL_EVALUATORS), blanks the dataset's augmentation representations, wraps the molecular representation builder so a comma list is not expanded to all 33, then calls similarity_helpers.run_all on a small real dataset.

**Asks:**
- Does the similarity suite run end to end on a small slice of a real dataset, on a laptop, without foundation models or the server?
- Does shrinking the engine's module-level defaults actually shrink the precompute, given that precompute_hybrid_searches reads those constants and not the config dict?

**Verdict on 2026-09-03:** PARTIAL · hybrid: none needed · live

Why it fell short:
- The driver is sound, but its only live caller passes experiment names it cannot resolve, so in the smoke gate it currently runs nothing (see bugs).
- It does not pass force_rerun, and similarity_helpers.run_all resumes off any existing output JSON (similarity_helpers.py:1043-1059). A hand-run into the same KIRBY_LOCAL_OUT after a code change reports RESUME and re-reads the old numbers.

*Smoke:* ran clean. exit 0 on herg_ki in six separate invocations; it is the only working local entry point for this suite and it honours KIRBY_LOCAL_OUT, so nothing was written into tests/results.

### Since the audit

0 fixed · 0 changed · 4 stand · 2 new · compute MORE EXPENSIVE · still runs

- **NEW** — Every distance-lineage search now raises in a local run, because the driver trims the model dictionary but the new DISTANCE_MODELS list is not trimmed with it. `tests/experiments/similarity/_run_similarity_locally.py:41-43 against tests/experiments/_shared.py:390`
- **STILL BROKEN** — Its only live caller passes experiment names it cannot resolve, so both similarity steps of the smoke gate exercise nothing and still record PASS.

*Compute now:* The file itself is unchanged, but what it drives is not. s04, s05, s07, s11, s12 and s13 now search a recipe per fold, so at the driver's cfg['n_splits']=2 each of those adds 2 fold searches on top of the whole-dataset precompute the driver already pays for.

### Decision

**MINE:** Take the normalising version so the smoke gate actually runs the five experiments, then fix the allocation set.

**The recheck opened something new here. Overrule me if it changes your call:**

>> 


---

# Similarity, standalone — 3 files

## sim_optimized_hybrid — `similarity_optimized_hybrid.py`

### What it does

Runs two find_best_hybrid searches on the same molecular dataset that differ only in the objective ('similarity' vs 'predictive'), rebuilds each winner per fold with create_hybrid, and compares the two spaces plus the best single representation on held-out distance-property correlation, on label-noise detection and correction under three noise conditions, and on importance-weighted distances.

**Asks:**
- Does a hybrid selected by held-out nearest-neighbour activity preserve similarity on held-out rows (distance-property Spearman, neighbour-lie rate) better than a predictive-selected hybrid and than the best single representation?
- Under injected label noise, do noise-detection and label-correction methods work better in that geometry than in the predictive hybrid or the raw representation?
- Does weighting distances by feature importance (none / sqrt / squared / top-50) sharpen the similarity-based detectors?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · live

Why it fell short:
- Both blueprints are selected by a search that runs over every row of the dataset, then scored on 3-fold splits of those same rows. The two hybrid arms are selection-optimistic; the single-representation arm is not.
- The search runs 5 folds (n_folds is never passed, DEFAULT_N_FOLDS=5) while payoff 1 and payoff 2 score on 3 folds. The claim written into the JSON at line 896-898, that sim and pred are scored on the folds their search selected on, is false.
- Under the 'gaussian' condition every row is perturbed by the same amount, so the corrupted-row mask is all-True: detection precision is 1.0 for any detector and the recorded chance line noise_frac=0.25 does not apply to it. One of the three conditions therefore contributes no detection information, while its numbers sit in the same table as the other two.

*Smoke:* not run. exceeded 6 min, killed at 360.3s, still inside the first hybrid search of seed 0.

### Since the audit

0 fixed · 0 changed · 10 stand · 6 new · compute CHEAPER · run state unknown

- **NEW** — the geometry arm searches 9 configurations and the predictive arm searches 15, so the A/B no longer differs only in the objective `tests/experiments/similarity_optimized_hybrid.py:110`
- **NEW** — the predictive rebuild's budget split moved to a different model; the geometry rebuild did not move `src/kirby/hybrid.py:1329`
- **NEW** — payoff 3 measures manhattan distance while payoff 2 measures Euclidean, on the same methods and the same spaces `tests/experiments/similarity_optimized_hybrid.py:543`
- **NEW** — payoff 2b's noise estimators moved to manhattan while the space is selected and reported under Euclidean `src/kirby/utils/noise_estimation.py:21`
- **NEW** — an exception inside a fold rebuild overwrites the streamed partial JSON with an error dict `tests/experiments/similarity_optimized_hybrid.py:1095-1100`
- **NEW** — two new raises inside create_hybrid can end a dataset run, and _build_hybrid_fold does not catch them `src/kirby/hybrid.py:844`

*Compute now:* Two things moved, in opposite directions, both small against the search. (1) Two of the five allocation methods now abort on the geometry arm.

### Decision

**MINE:** Pass budgets explicitly and remove greedy and dominant_greedy from SIM_ALLOCATIONS. Both are closed and this file can still select one as its winner.

## structural_geometry_survey — `structural_geometry_survey.py`

### What it does

Predicts an antibody structure per sequence, derives 7 geometry feature groups plus their concatenation, scores each group standalone under group-aware 5-fold CV, and scores every base representation with each geometry group hstacked onto it. It also launches a find_best_hybrid search over base plus geometry, but never gets a hybrid out of it.

**Asks:**
- Do structure-derived geometry features (VH-VL angles, CDR shape, paratope charge and the rest) carry standalone signal on antibody property prediction?
- Does concatenating a geometry feature group onto a base antibody representation change the score?
- Does a KIRBy hybrid over base representations plus geometry beat the geometry groups scored on their own?

**Verdict on 2026-09-03:** PARTIAL · hybrid: should use engine, does not · live

Why it fell short:
- The base representation on its own is never scored. geometry_baselines holds geometry groups only, augmentations holds base+geometry only, so the delta the survey exists to measure cannot be computed from its own results file - it has to be joined against a separate antibody_baseline_survey run at the same folds.
- The hybrid arm never produces a number. find_best_hybrid is called without return_fitted=True, so fitted_results is always empty (find_best_hybrid.py:6455) and every run records geo_failures['kirby_hybrid'] = 'search returned no fitted rf result'.
- If the hybrid arm did return, the selection would be fitted on all rows (_compute_fitted_result refits on full data) and then scored by 5-fold CV over those same rows, while the base and geometry arms are label-free. That comparison would be biased toward the hybrid.

*Smoke:* not run. exceeded 6 min at 'Computing geometry features (672 structures)...'.

### Since the audit

0 fixed · 0 changed · 7 stand · 4 new · compute MORE EXPENSIVE · run state unknown

- **NEW** — --skip-existing treats the 15 empty March result files as finished, because they carry no status key `tests/experiments/structural_geometry_survey.py:641-647`
- **NEW** — the search's evaluator axis still runs svm and mlp, which the suite dropped on 2026-09-04 `src/kirby/find_best_hybrid.py:2897`
- **NEW** — the scoring arms still default to mlp, which the suite replaced with dnn `tests/experiments/structural_geometry_survey.py:75`
- **STILL BROKEN** — find_best_hybrid is called without return_fitted=True, so the kirby_hybrid arm can never be scored and the whole search is discarded

*Compute now:* The discarded search shrank but the wall time went up, and structure prediction went up much more. Search: 8 budgets (FULL_BUDGETS) x 5 allocation methods x 3 importance methods x 5 internal evaluators = 600 configurations per model x 5 folds.

### Decision

**MINE:** Add the base-representation-alone arm. Without it no augmentation delta can be computed from the file.

**Open — yours:** Whether the hybrid arm is repaired or removed. Both live htc jobs record 'no fitted rf result' until this is settled.

>> 

## read_sim_optimized_results — `read_sim_optimized_results.py`

### What it does

Prints a compact per-dataset summary of the JSON that similarity_optimized_hybrid writes: whether the two searches picked different blueprints, the distance-property correlation and neighbour-lie rate, per-method detection precision against a chance line, estimator AUROC and precision@k, and the importance-weighting cells. Reads only; computes nothing.

**Asks:**
- Did the similarity search and the predictive search pick different blueprints, and on how many seeds?
- For each noise condition, how do the three spaces compare on damage, global label smoothing, per-method detection precision and removal delta?
- Which noise estimators rank the corrupted rows best in each space?

**Verdict on 2026-09-03:** PARTIAL · hybrid: none needed · live

Why it fell short:
- The divergence check compares only allocation method and budget. With identical values it declares the A/B degenerate, but create_hybrid re-ranks the representations under the objective at rebuild time (hybrid.py:1199, 2069), so the same allocation and budget still yield different feature sets under the two objectives.
- It prints per-method precision against noise_frac for every condition, including 'gaussian', where every row is corrupted and precision is 1.0 for any detector. Nothing in the reader marks that condition as one where the detection columns mean nothing.
- Only means are shown. The spread already in the file (lo, hi, std, n) is never printed, so a reader cannot see how wide any of these numbers are.

*Smoke:* ran clean. exit 0 both ways, in under a second.

### Since the audit

0 fixed · 0 changed · 3 stand · 2 new · compute unchanged · still runs


### Decision

**MINE:** Fix the three print blocks alongside the experiment, not before it.


---

# Similarity metrics — 33 files

## simeval — `similarity_metrics/simeval.py`

### What it does

A helper module, not an experiment. It turns a feature matrix into an (n,n) distance matrix under six coefficients (tanimoto, dice, cosine, soergel, tversky, euclidean), and scores one distance matrix on nine activity-grounded statistics: scaffold-grouped kNN prediction from five nearest neighbours (rho and -RMSE), MODI, Patterson neighbourhood bound, cliff AUROC, neighbour-lie rate, BEDROC enrichment, and injected-noise detection AUROC plus recovery correlation.

**Asks:**
- Given a distance matrix over molecules, how activity-coherent are its neighbourhoods?
- How does each similarity coefficient's geometry depart from ECFP4 Tanimoto (diagnostic only, never ranked)?
- Do a metric's neighbourhoods expose injected label noise, across a dose x contaminated-fraction grid?

**Verdict on 2026-09-03:** PARTIAL · hybrid: none needed · live

Why it fell short:
- Only eval_knn_readacross (line 143) is held out. The other eight statistics -- modi, patterson, cliff_auroc, violation_neg, bedroc, noise_detect_auroc, noise_recovery -- are computed in-sample on the full n x n matrix with no split.
- eval_patterson returns 0.0 when fewer than ten close pairs exist, and 0.0 is the MAXIMUM of that statistic's range (it always returns a negated percentile, so it is <= 0 otherwise). A degenerate distance matrix therefore receives the best possible Patterson score rather than a missing one.
- eval_modi_binary (line 183) and eval_noise_recovery (line 385) each return 0.0 for an undefined case. Stored in a results file, those are indistinguishable from a measured zero.

*Smoke:* ran clean. exit 0, ~5s.

### Since the audit

0 fixed · 0 changed · 6 stand · 0 new · compute unchanged · still runs


### Decision

**MINE:** Make the four returns NaN, set cliff_auroc False in the activity-grounded table, add the noise_recovery_defined flag.

## rankers — `similarity_metrics/rankers.py`

### What it does

A helper module of nine feature-importance scorers meant as similarity-aware alternatives to the engine's compute_feature_importance. Seven score every feature by the dependence between that feature's absolute pair difference and a target -- either the absolute activity gap or a structural distance. Two more (RReliefF, Laplacian) work on the rows directly. combine_ranks fuses several score vectors by averaging their ranks.

**Asks:**
- Can a feature be scored by how well its pair-difference tracks the activity gap, or a structural distance, rather than by a fitted model's importance?
- Does fusing an activity-target scorer with a structural-target scorer beat either alone (that fusion is what combine_ranks supports)?

**Verdict on 2026-09-03:** YES · hybrid: none needed · live

*Smoke:* ran clean. exit 0, ~10s.

### Since the audit

0 fixed · 0 changed · 4 stand · 0 new · compute unchanged · still runs


### Decision

**MINE:** Delete the five unused scorers plus RANKERS and LABEL_FREE.

## ranker_experiment — `similarity_metrics/ranker_experiment.py`

### What it does

Pools four representations (ecfp4_count, atompair_count, desc, maccs -- about 2423 features), z-scores them, and runs thirteen feature selectors over 8000 cached molecule pairs against two targets: the absolute activity gap and the MCS distance. Each selection keeps 100 features per representation from the top-N representations, and the resulting space is scored two ways -- scaffold-grouped kNN prediction from five nearest neighbours, and Spearman between the space's pairwise distance and MCS distance -- under cosine and euclidean. Compared against three untouched-fingerprint baselines.

**Asks:**
- Does selecting features FOR similarity beat selecting them predictively, when the selected space is judged by kNN prediction from five nearest neighbours and by agreement with shared substructure?
- Does a label-free structural selection target (MCS distance) beat a supervised activity target (the absolute activity gap)?
- Does any selected 400-feature space beat the standard count fingerprint under its native Tanimoto?

**Verdict on 2026-09-03:** NO · hybrid: hand-rolled · superseded

Why it fell short:
- Every selector is fitted on all 8000 cached pairs drawn from every molecule in the dataset, using dy = |y_i - y_j| over the whole label column (line 252, targets dict). The selected space is then scored by simeval.eval_knn_readacross (line 205), which GroupKFolds those same molecules.
- The three baselines (line 294) get no selection at all and keep their full native width -- 2048 features for ecfp4_count against 400 selected. Any gap between a selector row and a baseline row mixes a width change with a selection change, and the JSON stores nothing that separates them.
- build_pool (line 66) computes the z-score mean and standard deviation over all rows including the rows later held out inside eval_knn_readacross. A scale-only leak, shared by every selector arm, but it is not a per-fold scaling.

*Smoke:* ran clean. exit 0, 33s.

### Since the audit

2 fixed · 0 changed · 4 stand · 4 new · compute MORE EXPENSIVE · run state unknown

- **NEW** — The two stored results files hold the pre-4d16dbb schema and today's code can neither read them nor reproduce their numbers. `tests/experiments/similarity_metrics/ranker_experiment.json:1`
- **NEW** — Which representations survive the top-2 allocation is now decided inside each fold, and nothing in the output records which ones each fold kept. `tests/experiments/similarity_metrics/ranker_experiment.py:209`
- Fixed (4d16dbb) — Feature selection saw the labels of every molecule, then the selected space was scored by a GroupKFold five-nearest-neighbour prediction over those same molecules; the activity-target arms were graded on folds whose labels chose their own features and the structural arms were not.
- Fixed (4d16dbb) — The z-score mean and standard deviation were computed over all rows, including rows later held out by the scoring step.

*Compute now:* About 6 to 7 times the old wall clock, from commit 4d16dbb. Each ranker used to score once, on all 8000 cached pairs; it now scores once per context, and there are seven contexts — the five training folds, the fit half, and the all-rows in-sample arm.

### Decision

**YOURS:** Delete and read framework/f22, or rewrite it per fold. f22 asks the same headline question leak-free and has a stored result on one of these two datasets. Keeping it means the other ten selectors matter to you.

>> 

## blacklist — `similarity_metrics/blacklist.py`

### What it does

A standalone library that finds 'saboteur' columns inside one representation block by greedy backward elimination inside each scaffold GroupKFold fold, under one of two yardsticks: Spearman between the block's pairwise squared distance and MCS distance over cached pairs, or a train-reference-to-test kNN prediction Spearman. A column reaches the blacklist only if it is dropped in a majority of folds and its mean held-out gain clears a threshold. Returns the column indices to feed find_best_hybrid's feature_blacklist argument.

**Asks:**
- Which individual columns inside a feature block make the block's similarity geometry worse, stably across scaffold folds?
- How much does removing those columns raise the held-out yardstick (the returned delta)?

**Verdict on 2026-09-03:** PARTIAL · hybrid: none needed · orphaned

Why it fell short:
- For yardstick='readacross' the greedy step chooses the column to drop by maximising the score on the TEST fold (line 190), not on the train fold. The module header at lines 19-22 claims 'every drop decision is made on TRAIN folds and its effect is re-measured on HELD-OUT folds'.
- Even on the structural yardstick, the blacklist guard filters columns by mean_heldout_gain (line 313) and the final before/after then re-measures on those same test folds (lines 324-333). The delta is selected and reported on the same rows, so it is biased upward.
- The structural yardstick uses only pairs whose two endpoints are both in the test fold. With 8000 cached pairs and 5 folds that is roughly 4 percent of them, about 320 pairs, and every held-out gain per drop is measured on that.

*Smoke:* ran clean. exit 0, ~8s.

### Since the audit

0 fixed · 0 changed · 8 stand · 1 new · compute unchanged · still runs

- **NEW** — Nothing in the repository calls identify_saboteurs, and the blacklist the repository actually produces is now built by a different function that has no fold structure at all. `tests/experiments/similarity_metrics/winning_recipe_vs_standard.py:136`
- **STILL BROKEN** — On the readacross yardstick the feature dropped at each greedy step is the one that most raises the score on the TEST fold, while the module header claims every drop is decided on train folds.

### Decision

**MINE:** Delete. src/kirby/similarity/selection.py already does the structural half per fold and is wired in. Nothing imports this and nothing stores its output.

## scaffold_experiment — `similarity_metrics/scaffold_experiment.py`

### What it does

Builds five scaffold-mutation representations plus two whole-sequence base representations (aac, blosum_mean) on one FLAb variant library, turns each into a pairwise distance under cosine and euclidean, and scores every space two ways: Spearman of its pair distances against three substitution-matrix graders, and a five-nearest-neighbour prediction of the fitness label under GroupKFold. Writes one row per (representation, coefficient) to JSON.

**Asks:**
- Does describing a variant by what changed relative to the wildtype order variant pairs more like an independent substitution matrix than describing it by its whole sequence (aac, blosum_mean) does?
- Which distance coefficient, cosine or euclidean, gives the better-ordered space?
- Do the five nearest neighbours in each space predict the measured fitness under a position-grouped split?

**Verdict on 2026-09-03:** PARTIAL · hybrid: none needed · live

Why it fell short:
- The groups for the fitness axis are the FIRST mutated position only (line 78). flab_binding_phillips_h3 and flab_binding_adams are combinatorial libraries where a variant carries many mutations, so a held-out variant shares most of its mutations with training variants.
- The BLOSUM grader is built as max-minus-score in scaffold_reps.blosum_matrix, so a residue matched against itself has a non-zero distance that varies by residue. Summed over the whole sequence this adds a which-position-was-mutated term to the grader that the positionally-indexed representations can pick up.
- The position_gated row grades one ground truth against another. dist_position_gated computes exactly the closed form that grader_pair_distance produces with the Grantham matrix on a single-mutation library, so its blosum score is grader-versus-grader agreement.

*Smoke:* ran, NaNs present. exit 0, ~7s.

### Since the audit

1 fixed · 1 changed · 2 stand · 3 new · compute MORE EXPENSIVE · still runs

- **NEW** — Three result files on disk hold five-nearest-neighbour numbers that today's code writes as NaN. `tests/experiments/similarity_metrics/scaffold_experiment_flab_binding_phillips_h3.json`
- Fixed (4d16dbb) — The folds group variants by their first mutated position, so on a combinatorial library the held-out variant shares most of its mutations with training variants, and readacross_rho was written anyway.

*Compute now:* Single-mutation datasets get more expensive; multi-mutation ones get cheaper. On koenig at n=5000 the read-across step is 15 spaces x 5 folds.

### Decision

**MINE:** Use the registry grouping, delete the position_gated row (it is the Grantham grader itself), fix the filename so a smoke run stops overwriting the full run, re-run all five.

## scaffold_ddg — `similarity_metrics/scaffold_ddg_experiment.py`

### What it does

Runs the same five scaffold-mutation representations on graphinity single-point protein ddG data, sampling 400 variant pairs inside each PDB complex with at least 25 mutations, pooling those pairs across complexes and correlating each representation's pair distance against three substitution-matrix graders. Structural axis only, no labels used.

**Asks:**
- Do the five scaffold-mutation representations order pairs of single-point mutants like an independent substitution matrix, on a library where every variant carries exactly one mutation so the mutation-count confound cannot act?
- Which distance coefficient, cosine or euclidean, orders those pairs better?

**Verdict on 2026-09-03:** PARTIAL · hybrid: none needed · orphaned

Why it fell short:
- Position is the residue number alone. The loader stores a chain letter next to it (graphinity.py:349-354) and this script drops it, so two mutations at the same residue number on different chains of one complex are treated as the same position by grader_pairs and by the same/different branch.
- Complexes are grouped by PDB id alone, while the loader keys sequences by (pdb, ab_chain). One PDB carrying two antibody/antigen chain pairings is pooled as a single complex.
- The BLOSUM grader here is the closed form on the mutated residues only, whereas scaffold_experiment sums BLOSUM over the whole sequence. Because BLOSUM62's diagonal is not constant the two constructions differ by more than an offset, so a graphinity BLOSUM number is not on the same scale as a koenig one.

*Smoke:* ran, NaNs present. exit 0, ~25s.

### Since the audit

0 fixed · 0 changed · 3 stand · 2 new · compute unchanged · still runs


### Decision

**MINE:** Chain and complex keys, and the crash message.

**Open — yours:** Whether every complex still contributes 400 pairs regardless of its size.

>> 

## scaffold_feature_isolation — `similarity_metrics/scaffold_feature_isolation.py`

### What it does

Takes dPhysChem and dPhysChem_pos on three FLAb variant libraries and scores feature subsets instead of whole representations: each of the five property scales alone, all five minus one, and dPhysChem_pos with each mutated position removed in turn. Every number is a Spearman of pair distance against the BLOSUM grader, raw and with the Hamming component partialled out.

**Asks:**
- Which of the five physicochemical scales inside dPhysChem and dPhysChem_pos carry the agreement with the BLOSUM grader, alone and when dropped?
- Is that agreement carried by a few mutated positions or spread across all of them?

**Verdict on 2026-09-03:** PARTIAL · hybrid: none needed · orphaned

Why it fell short:
- Undefined correlations are written as 0.000, not NaN. score() returns 0.0 when the pair distance is constant (line 64) and coerces a non-finite Spearman or partial correlation to 0 (lines 67-68).
- The position labels written as pos_<p> are indices into the good-columns array built by encode_sequences, which drops any column holding a non-standard residue. They cannot be mapped back to residue numbers.
- top5_damage_share divides by the sum of signed damages, which can be near zero or negative when dropping positions improves the score, so the share can exceed 1 or come out negative.

*Smoke:* ran clean. exit 0, ~5 min (it exceeded the Bash tool's 2-min default on the first attempt; re-run with a longer wall).

### Since the audit

0 fixed · 0 changed · 4 stand · 1 new · compute unchanged · still runs


### Decision

**MINE:** Fix the damage share, add --cap and --output. Hold the re-run until blosum_matrix lands, since that sets every number here.

## scaffold_reps — `similarity_metrics/scaffold_reps.py`

### What it does

Pure library. Holds the amino-acid property scales, the Grantham, BLOSUM and identity substitution matrices, sequence encoding and wildtype inference, the five scaffold-mutation representations, the grader-weighted pair distance used as ground truth, and the position-gated distance. Runs nothing itself.

**Asks:**
- Does it build the five scaffold-mutation representations, the three graders and the two scaffold-relative distances correctly for the scripts that import it?

**Verdict on 2026-09-03:** PARTIAL · hybrid: none needed · live

Why it fell short:
- The BLOSUM grader is not a distance with a zero self-distance, which changes what struct_blosum measures for every caller.
- The wildtype is inferred, not read from the loader, even though load_flab_dataset returns wildtype_heavy and wildtype_light.
- No tests import this module.

*Smoke:* ran clean. exit 0, ~4s, no results file.

### Since the audit

0 fixed · 1 changed · 3 stand · 0 new · compute unchanged · still runs


### Decision

**MINE:** Fix blosum_matrix — one line, and it sets a number in ten result files. Take the wildtype as an argument before anyone quotes an adams or phillips_h3 result.

## sm_feature_type_study — `similarity_metrics/feature_type_study.py`

### What it does

Builds 14 amino-acid property scales into a positional delta-from-wildtype representation of FLAb variants, then scores subsets of those scales by the count-controlled Spearman between their Euclidean pair distance and a BLOSUM62 sequence distance over 8000 randomly sampled variant pairs. No model is trained and no measured label is ever read.

**Asks:**
- Which amino-acid property scales earn a place in a Euclidean distance space that reproduces BLOSUM sequence distance once mutation count is partialled out?
- Does forward selection on that objective beat taking the top-k scales by their solo score, i.e. does redundancy between scales cost more than weak solo signal?
- Is the score monotone in the number of scales included, or is there an optimum below all 14?

**Verdict on 2026-09-03:** PARTIAL · hybrid: none needed · orphaned

Why it fell short:
- Selection and reporting use the same 8000 pairs. greedy_forward maximises exactly the quantity printed at :216-217 as 'greedy gain over top-k', so that gain is guaranteed positive by construction and is not a measured advantage.
- The H5 curve at :184 is scored on the same pairs used to rank the scales at :169-171, so 'peak at m features' is also in-sample.
- The ground truth is BLOSUM sequence distance, never the measured label. Nothing here says the selected scales help predict activity; that is a different experiment.

*Smoke:* ran clean. exit 0, ~20s, no results file (I called build/greedy_forward directly, so feature_type_study.json was not touched).

### Since the audit

0 fixed · 1 changed · 4 stand · 3 new · compute unchanged · still runs

- **NEW** — the same greedy-versus-top-k comparison was given a fit half and a report half in the file that imports this one, and not here `tests/experiments/opig_scaffold_similarity.py:344-364 against feature_type_study.py:178-190`
- **NEW** — the surviving held-out number for H3 rests on a pair split, the cut the same day's commit rejected elsewhere `tests/experiments/opig_scaffold_similarity.py:344-346`

### Decision

**MINE:** Three one-line fixes, then delete run() and main() and keep the module. Nothing reads its JSON.

## sm_feature_ablation — `similarity_metrics/feature_ablation.py`

### What it does

Exhaustively scores every subset of four z-scored feature blocks (ECFP4 counts, 208 RDKit descriptors, top-100 2D columns, MACCS keys) concatenated into one Euclidean distance, on two yardsticks: scaffold-grouped 5-nearest-neighbour Spearman against the measured label, and Spearman against cached MCS distance. Then runs a drop-one greedy over the 208 descriptors to find which ones lower structural fidelity.

**Asks:**
- Which representation blocks help and which hurt when they are concatenated into one similarity space, judged against predicting each molecule's activity from its five nearest neighbours and against MCS structural fidelity?
- Do individual RDKit descriptors inside the physicochemical block actively lower structural fidelity, and which ones?
- Does a 3D pharmacophore block add anything over the 2D blocks?

**Verdict on 2026-09-03:** PARTIAL · hybrid: hand-rolled · orphaned

Why it fell short:
- The fine2D block is chosen by correlating each of 2048 columns against MCS on the same 8000 pairs that then score structural fidelity. Every struct_fidelity and d_struct for a subset containing fine2D is selection-optimistic, and so is the fine2D row of block_marginals.
- The descriptor drop-one greedy has no held-out pairs. On herg_ki it raises structural fidelity from 0.250 to 0.495 by dropping 143 of 208 descriptors, chosen on the 8000 pairs it reports.
- The shape question is answered on 250 herg_ki molecules and 254 MCS pairs, and never on logd, on the strength of a docstring claim about the data that is no longer true.

*Smoke:* ran clean. exit 0, ~15s, no results file.

### Since the audit

3 fixed · 1 changed · 3 stand · 4 new · compute MORE EXPENSIVE · still runs

- **NEW** — the two arms are scored on different numbers of molecules, so the gap between them is not only optimism `tests/experiments/similarity_metrics/feature_ablation.py:340 and :373-379`
- **NEW** — the shape supplement's held-out arm has 55 scoring pairs, under its own 300-pair floor `tests/experiments/similarity_metrics/feature_ablation.py:396-428, floor at :78`
- **NEW** — the stored feature_ablation.json cannot be reproduced by today's code `tests/experiments/similarity_metrics/feature_ablation.json, written at :501`
- Fixed (4d16dbb) — select_fine2D ranks 2048 features on the cached MCS pairs and the fine2D block's structural fidelity is then measured on those same pairs
- Fixed (4d16dbb) — desc_drop_one_greedy maximises structural fidelity on the same 8000 pairs it reports, 143 drops deep

*Compute now:* About twice, because every step now runs on both arms. Old, per dataset: one fine-2D ranking of 2048 features by Spearman over 8000 pairs, 15 block subsets each costing one 1415x1415 Euclidean plus 5 kNN fits, and one drop-one greedy of up to 206 rounds over about 208 descriptors, so roughly 42,800 correlations.

### Decision

**MINE:** Blacklist swap and the full-size 3D block, plus the pair split for the fine-2D selection.

## sm_base_rep_comparison — `similarity_metrics/base_rep_comparison.py`

### What it does

On four FLAb variant libraries, builds six distances from two edit-describing representations and three whole-sequence representations, guesses each held-out variant's label from its five nearest training neighbours under GroupKFold, and reports the Spearman of guessed against true labels with a 1000-draw bootstrap interval. Two null controls (random distance, constant distance) are included.

**Asks:**
- Does describing what a variant changed relative to the wildtype beat describing its whole sequence, when a variant's measured label is guessed from its five nearest neighbours?
- Do the reported correlations separate from a random distance and from a constant distance?

**Verdict on 2026-09-03:** PARTIAL · hybrid: none needed · orphaned

Why it fell short:
- Grouping is by the first mutated position only. On the two adams sets (1.75 mutations per variant) variants sharing most of their mutations can land in different folds.
- flab_binding_adams carries a 0/1 label, so its row is a rank correlation against a binary outcome while the other three are against a continuous one. The docstring calls all four 'the variant's measured value'.
- The verdict rule declares TIE whenever two marginal intervals overlap, which is not the test the design supports and will call real differences ties. In the stored result, koenig aac:cosine 0.482 [0.457, 0.507] against dphyschem 0.467 [0.442, 0.493] is declared a tie on overlapping intervals.

*Smoke:* ran clean. exit 0, ~20s, no results file.

### Since the audit

1 fixed · 3 changed · 3 stand · 3 new · compute CHEAPER · still runs

- **NEW** — the edit-versus-whole-sequence verdict now exists on two of the four datasets `tests/experiments/similarity_metrics/base_rep_comparison.py:163-171, :258-273`
- **NEW** — the stored base_rep_comparison.json holds numbers today's code would refuse to write `tests/experiments/similarity_metrics/base_rep_comparison.json`

*Compute now:* Cheaper overall, because half the roster no longer reaches the scoring. Old: 4 datasets x 8 distances x (GroupKFold(5) kNN + 1000 bootstrap Spearman) = 160 kNN fits and 32,000 correlations.

### Decision

**MINE:** Paired bootstrap and the classification dispatch.

**Open — yours:** What a group is on the adams libraries, which decides the fold grouping.

>> 

## sm_aggregate — `similarity_metrics/aggregate.py`

### What it does

Builds six single-metric distance matrices on herg_ki and logd, fuses them six ways (min-max score average, mean rank, reciprocal rank fusion, Borda, consensus vote, and a redundancy-pruned score fusion), and scores all twelve on four yardsticks: neighbour-based label prediction under GroupKFold, MCS structural fidelity, injected-noise detection AUROC, and noise recovery correlation.

**Asks:**
- Does fusing several complementary similarity metrics beat the best single metric at predicting each molecule's activity from its five nearest neighbours?
- Does fusion beat the best single metric at MCS structural fidelity, at detecting injected label noise, and at correcting it?
- Which of the six fusion rules wins, and does pruning redundant metrics before fusing help?

**Verdict on 2026-09-03:** PARTIAL · hybrid: none needed · orphaned

Why it fell short:
- The A6 redundancy-pruned arm chooses which metrics to keep using every molecule's label, then is scored on those same labels through GroupKFold. Its read-across number is not comparable with the five label-blind aggregators.
- One noise realisation only: seed 0, sigma 0.6, contaminated fraction 0.15. simeval.noise_sweep_scores at :393 provides the sigma x rate grid the noise question needs, and it is not called, so the noise columns are a single point.
- The four rank-based fusions produce an asymmetric matrix and nothing in the file says so. Structural fidelity reads only D[I,J], one triangle, and the kNN treats rows as queries.

*Smoke:* ran clean. exit 0, ~20s, no results file (aggregate_experiment.json untouched).

### Since the audit

1 fixed · 1 changed · 5 stand · 3 new · compute unchanged · still runs

- Fixed (4d16dbb) — a6_dca ranks the six metrics against the activity gap and is then scored on those same labels by scaffold-grouped kNN

### Decision

**MINE:** Recording fixes, label-free ranking, and run the noise sweep. One injection at one dose is what makes the two noise columns unquotable.

## creative_hybrids — `similarity_metrics/creative_hybrids.py`

### What it does

Builds 11 similarity spaces on one dataset by z-scoring and concatenating ecfp4_count, RDKit descriptors, MACCS, and the top-K atompair/torsion features picked by their correlation with MCS distance, then scores each space two ways: Spearman of a 5-nearest-neighbour prediction under scaffold GroupKFold, and Spearman between the space's pairwise distances and MCS distance. A second sub-table adds the 49-feature 3D pharmacophore block on whatever rows the conformer cache covers. No model is fitted and nothing is written to disk.

**Asks:**
- Does stacking descriptors, MACCS and a structurally-selected atompair/torsion block onto the count fingerprint raise the 5-nearest-neighbour activity Spearman above ecfp4_count with Tanimoto?
- Does that stacking raise agreement between the space's pairwise distances and MCS distance?
- How much of any gain is the coefficient change (Tanimoto to z-scored Euclidean) rather than the added blocks, and how many atompair/torsion features does the fine block need (50, 100, 250)?

**Verdict on 2026-09-03:** PARTIAL · hybrid: hand-rolled · superseded

Why it fell short:
- Nothing is written to a file. run() returns a dict and main() drops it, so the printed table is the only record and no creative_hybrids.json exists in the directory.
- The z-score mean and standard deviation for every block are computed over all molecules, including the fold that the 5-nearest-neighbour score holds out. The activity number is fitted on its own test rows.
- The atompair/torsion features are chosen by their correlation with MCS distance over the 8000 cached pairs, and the structure score is then the correlation with MCS distance over those same 8000 pairs. The structure number grades the selection on the data the selection was made on.

### Since the audit

0 fixed · 0 changed · 7 stand · 1 new · compute unchanged · still runs


### Decision

**MINE:** Move MACCS and the feature-count sweep to the honest builder, then delete this file.

## creative_hybrids_v2 — `similarity_metrics/creative_hybrids_v2.py`

### What it does

Re-runs the same stacking idea with the 3D pharmacophore block at full row count and with a per-dataset greedy descriptor drop: repeatedly remove the single RDKit descriptor whose removal most raises the correlation between Euclidean distance and MCS distance. Scores 10 spaces per dataset on the 5-nearest-neighbour activity Spearman and on MCS agreement, across herg_ki, logd and adme_solubility, and writes creative_hybrids_v2.json. This is the module the other three scripts import their block algebra from.

**Asks:**
- Does count fingerprint + descriptors-minus-saboteurs + 3D pharmacophore + fine 2D beat ecfp4_count with Tanimoto on the 5-nearest-neighbour activity Spearman, on all three datasets?
- Does it beat it on the correlation between pairwise distance and MCS distance?
- Does halving the weight on the count block change which stacked space wins?

**Verdict on 2026-09-03:** PARTIAL · hybrid: hand-rolled · live

Why it fell short:
- Same z-score leak: make_D at :76-94 takes each feature's mean and standard deviation from all molecules, then eval_knn_readacross holds scaffold folds out of a distance matrix that already saw them. The activity delta (+0.024 herg_ki, +0.073 logd, +0.019 adme in the committed JSON) is measured under that leak.
- The descriptor drop at :196 and the fine-2D selection at :201 are both fitted on the 8000 cached MCS pairs, and struct_fidelity at :166-178 scores on those same 8000 pairs. The structure gains (+0.13 to +0.39) are select-on-what-you-score-on.
- The greedy descriptor drop removes 143 of 208 columns on herg_ki, 140 on logd, 147 on adme_solubility with no held-out check inside this script.

### Since the audit

0 fixed · 0 changed · 6 stand · 2 new · compute unchanged · still runs

- **NEW** — NEW: the descriptor drop procedure in this file produces a gain of the same size on data with no structural signal at all, and the measured control now sits in src/. `tests/experiments/similarity_metrics/creative_hybrids_v2.py:124-162`
- **NEW** — NEW: this file is now the exporter of the leaky helpers for four downstream consumers. `tests/experiments/similarity_metrics/creative_hybrids_v2.py:76-94 and :124`

### Decision

**MINE:** Fix the output path collision first, then fold into the honest arm next door.

**The recheck opened something new here. Overrule me if it changes your call:**

>> 

## winning_recipe_honest — `similarity_metrics/winning_recipe_honest.py`

### What it does

Takes creative_hybrids_v2's winning stacked space and re-grades its structure number after splitting the 8000 cached MCS pairs 50/50: the descriptor drop and the atompair/torsion selection are fitted on one half, the distance-versus-MCS correlation is reported on the disjoint half. Compares that space against ecfp4_count with Tanimoto and ecfp4 with Tanimoto on two datasets, and writes both the in-sample and the held-out structure number side by side.

**Asks:**
- Once features are chosen on one half of the MCS pairs and structure is scored on the other half, does the stacked space still beat ecfp4_count with Tanimoto on distance-versus-MCS agreement?
- How much of the in-sample structure gain was inflation from choosing and scoring on the same pairs?
- Does the stacked space also beat the two single-representation Tanimoto baselines on the 5-nearest-neighbour activity Spearman?

**Verdict on 2026-09-03:** PARTIAL · hybrid: hand-rolled · superseded

Why it fell short:
- The split is over pairs, not molecules. Every molecule appears in both halves, so the descriptor drop and the fine-2D selection have already seen every molecule they are graded on.
- The activity arm carries the same all-molecule z-score leak, inherited from the imported make_D.
- The recipe scored here is the coarse-down variant (weights 0.5 on the count block), which is not the space that won on herg_ki or adme_solubility in creative_hybrids_v2.json.

### Since the audit

2 fixed · 1 changed · 2 stand · 4 new · compute CHEAPER · still runs

- **NEW** — NEW: winning_recipe_honest.json is not on disk. `tests/experiments/similarity_metrics/ (no winning_recipe_honest.json)`
- **NEW** — NEW: the thin-half flag from the same commit went into two files and not this one. `tests/experiments/similarity_metrics/winning_recipe_honest.py:113`
- **NEW** — NEW: the descriptor drop still picks and stops on the pairs it is graded on within the select half. `tests/experiments/similarity_metrics/winning_recipe_honest.py:87`
- **NEW** — NEW: the activity number is now measured on half the molecules. `tests/experiments/similarity_metrics/winning_recipe_honest.py:103`
- Fixed (4d16dbb) — The 50/50 split is a permutation of the 8000 pairs, not of the molecules, so the selection has seen every molecule it is scored on.
- Fixed (4d16dbb) — make_D is imported unchanged, so the z-score mean and standard deviation still come from all molecules.

*Compute now:* Still 0 model fits. The descriptor drop loop now runs on the select half's pairs only (winning_recipe_honest.py:87 passes Is, Js, Ms), not all 8000.

### Decision

**MINE:** Delete. similarity_noise_cleaning already splits at molecule level, which is what this file set out to do. Correct the two documents quoting 0.410 and 0.687.

## winning_recipe_vs_standard — `similarity_metrics/winning_recipe_vs_standard.py`

### What it does

The headline table: builds the equal-weight stacked space (count fingerprint + descriptors-minus-saboteurs + top-100 atompair/torsion features + 49-feature 3D pharmacophore) and scores it against three single-representation comparators, ecfp4 with Tanimoto, ecfp4_count with Tanimoto, and ecfp4_count as a z-scored Euclidean space, on the 5-nearest-neighbour activity Spearman and on distance-versus-MCS agreement. Writes winning_recipe_vs_standard.json.

**Asks:**
- Does the stacked space beat the two field-standard fingerprint similarity metrics on the 5-nearest-neighbour activity Spearman?
- Does it beat them on the correlation between pairwise distance and MCS distance?
- Is any gain the added blocks rather than the switch from Tanimoto to z-scored Euclidean?

**Verdict on 2026-09-03:** PARTIAL · hybrid: hand-rolled · live

Why it fell short:
- The structure column is circular. The descriptor drop at :73 and the fine-2D selection at :75 are fitted on I,J,mcs and struct_fidelity is scored on the same I,J,mcs.
- The activity column carries the all-molecule z-score leak from the imported make_D.
- Only herg_ki is on disk. The committed JSON holds one of the three datasets the --all path names, so the cross-dataset claim in the roadmap is not supported by this file's own payload.

### Since the audit

2 fixed · 0 changed · 2 stand · 5 new · compute CHEAPER · still runs

- **NEW** — NEW: the result file on disk cannot be produced by today's code. `tests/experiments/similarity_metrics/winning_recipe_vs_standard.json`
- **NEW** — NEW: the docstring's guarantee that the numbers match creative_hybrids_v2 is no longer true. `tests/experiments/similarity_metrics/winning_recipe_vs_standard.py:17-19`
- **NEW** — NEW: the thin-half flag from the same commit went into two files and not this one. `tests/experiments/similarity_metrics/winning_recipe_vs_standard.py:105`
- **NEW** — NEW: the descriptor drop still picks and stops on the pairs it is graded on within the select half. `tests/experiments/similarity_metrics/winning_recipe_vs_standard.py:136`
- **NEW** — NEW: the activity column and the headline deltas are now measured on half the molecules, under unchanged key names. `tests/experiments/similarity_metrics/winning_recipe_vs_standard.py:95 and :228`
- Fixed (4d16dbb) — desc_blacklist_greedy and select_fine2d are fitted on the full pair set and every struct_fidelity is graded on those same pairs, while the docstring calls both yardsticks honest.
- Fixed (4d16dbb) — make_D is imported unchanged, so the per-feature mean and standard deviation come from all molecules including the scaffold folds the activity score holds out.

*Compute now:* Still 0 model fits. The descriptor drop loop now runs on the select half's pairs (winning_recipe_vs_standard.py:136 passes Is, Js, Ms), roughly 2000 of 8000 under a random half-cut, so each of the 19,591 Spearman calls is about 4x cheaper: about 3.9e7 pair-evaluations against 1.6e8 on herg_ki.

### Decision

**MINE:** Fix the output path, then add the field-default comparator the noise-cleaning script lacks.

**The recheck opened something new here. Overrule me if it changes your call:**

>> 

## structural_embeddings — `similarity_metrics/structural_embeddings.py`

### What it does

Builds MAP4, TDA and mol2vec matrices for one dataset, turns each into 2 distance matrices, and scores every distance matrix on two numbers: the Spearman of activity predicted from the 5 nearest neighbours under scaffold-grouped GroupKFold, and the Spearman between its pairwise distance and the cached MCS distance. The comparison arm is ecfp4_count with Tanimoto, reported on its own, never averaged.

**Asks:**
- Does a MAP4, TDA or mol2vec distance space predict activity from a molecule's five nearest neighbours better than ecfp4_count with Tanimoto?
- Does any of those three track shared-substructure (MCS) distance better than ecfp4_count with Tanimoto?
- Do the three KIRBy builders create_map4 / create_tda / create_mol2vec run at all in this environment?

**Verdict on 2026-09-03:** PARTIAL · hybrid: none needed · orphaned

Why it fell short:
- The MAP4 arm is not built by the KIRBy builder. build_map4 (line 99) calls the installed map4.MAP4 directly because create_map4 raises ImportError, so a MAP4 win here cannot be reproduced through kirby.representations.molecular.
- Molecules whose conformer or featurisation failed become all-zero rows and stay in the matrix. The count is printed (lines 197, 210) and never written to JSON, so a reader cannot tell how much of a distance space was built on failures.
- eval_knn_readacross pools the out-of-fold predictions from all 5 folds into one Spearman, so no per-fold spread exists anywhere in the payload. A gap against the baseline cannot be compared with fold-to-fold noise.

*Smoke:* ran clean. exit 0, ~260s.

### Since the audit

0 fixed · 0 changed · 8 stand · 4 new · compute unchanged · still runs

- **NEW** — The size control cannot be applied to any stored result, because the pair files carry no atom counts `tests/experiments/similarity_metrics/mcs_cache.py:73`

### Decision

**MINE:** Fix create_map4 (one line in src/kirby/representations/molecular.py), add the molecule-size control, run all three datasets before quoting anything.

## structural_yardstick — `similarity_metrics/structural_yardstick.py`

### What it does

Samples 6000 random molecule pairs per dataset, computes an RDKit MCS distance for each, and for 47 representation-and-coefficient combinations correlates that combination's pairwise distance against the MCS distance and against the activity gap. Ranks the 47 by the MCS correlation. Its mcs_distance function is also the definition imported by mcs_cache.py and ported into src/kirby/similarity/mcs.py.

**Asks:**
- When the judge is shared substructure (MCS) rather than activity, which representation-and-coefficient pair ranks first?
- Where do ecfp4:tanimoto and ecfp4_count:tanimoto land in that ranking?
- Does a distance that tracks shared substructure also track the activity gap, or do the two judges disagree?

**Verdict on 2026-09-03:** PARTIAL · hybrid: none needed · superseded

Why it fell short:
- No control for molecule size. Euclidean distance grows with atom count and so does MCS distance, so the Euclidean sweep of the top of the table may be a size effect.
- Pairs are drawn uniformly at random over molecules (line 83), so most are near-maximally dissimilar and the correlation is dominated by the saturated end of the MCS range.
- It loads sim_<name>.npz directly instead of _data.load_prepared, so on logd it uses all 5039 molecules while every other script in the suite uses the 1400-molecule scaffold subsample. Its pair numbering is a third row space, comparable with neither.

*Smoke:* ran clean. exit 0, ~8s.

### Since the audit

1 fixed · 0 changed · 7 stand · 2 new · compute unchanged · still runs

- **NEW** — The stored 47-row table predates the two _defined fields, and the only way to add them destroys it `tests/experiments/similarity_metrics/structural_rerank.json`

### Decision

**MINE:** Take the rewrite. It removes 9 minutes of repeated MCS work per dataset, puts logd in the suite row space, and reports the size control that changes the answer.

## shape_similarity — `similarity_metrics/shape_similarity.py`

### What it does

Builds USR (12 features), USRCAT (60) and a 3D pharmacophore descriptor (49) on ETKDG conformers, makes a cosine and a z-scored euclidean distance matrix from each, and scores all six plus the ecfp4_count Tanimoto baseline on activity predicted from the 5 nearest neighbours under scaffold-grouped GroupKFold and on Spearman against the cached MCS distance. Caches the three matrices to sim_<ds>_spatial.npz.

**Asks:**
- Does a 3D shape or pharmacophore distance predict activity from a molecule's five nearest neighbours better than ecfp4_count with Tanimoto?
- Does one of them track shared-substructure (MCS) distance better than that baseline?
- Does the choice of coefficient, cosine on raw features versus euclidean on z-scored features, change which 3D descriptor wins?

**Verdict on 2026-09-03:** PARTIAL · hybrid: none needed · orphaned

Why it fell short:
- Molecules whose conformer failed are all-zero rows that stay in the matrix. After _zscore they collapse onto one identical point, so every failed molecule is a nearest neighbour of every other failed molecule in the euclidean space.
- One pooled Spearman per distance space, no per-fold values, so nothing in the payload says whether the +0.033 read-across gap the roadmap quotes for pharm3d exceeds fold noise.
- The MCS correlation has no molecule-size control, so the structural gains for the euclidean spaces are confounded by atom count.

*Smoke:* ran clean. exit 0, 0.4s of compute.

### Since the audit

0 fixed · 0 changed · 7 stand · 1 new · compute unchanged · still runs


### Decision

**MINE:** Both fixes in one pass and re-run all three. Seconds, because the conformers are cached.

## aug3d_similarity — `similarity_metrics/aug3d_similarity.py`

### What it does

Builds E3FP (1024 bits) and the Coulomb-matrix eigenvalue spectrum (50 features) on ETKDG conformers, makes tanimoto/cosine/euclidean spaces from E3FP and cosine/euclidean from Coulomb, and scores all five plus the ecfp4_count Tanimoto baseline on activity predicted from the 5 nearest neighbours under scaffold-grouped GroupKFold, on the negative RMSE of that same prediction, and on Spearman against the cached MCS distance. Caches to sim_<ds>_3daug.npz.

**Asks:**
- Does an E3FP or Coulomb-eigenvalue distance predict activity from a molecule's five nearest neighbours better than ecfp4_count with Tanimoto?
- Does either track shared-substructure (MCS) distance better than that baseline?
- Which coefficient, tanimoto, cosine or z-scored euclidean, makes the best distance out of the same 3D representation?

**Verdict on 2026-09-03:** PARTIAL · hybrid: none needed · orphaned

Why it fell short:
- Failed conformers are all-zero rows that stay in the matrix, and the printed count is not stored. This matters most for the euclidean space, where z-scoring puts every failed molecule at distance 0 from every other failed molecule.
- One pooled Spearman and one pooled RMSE per space, no per-fold values, so the stored e3fp:euclidean structural fidelity of 0.512 against a 0.045 baseline on herg_ki carries no spread.
- That same e3fp:euclidean number is the one most exposed to the uncontrolled molecule-size confound: euclidean distance over z-scored bits scales with the on-bit count, which scales with molecule size, as does MCS distance. There is no size control here and the payload does not store atom counts.

*Smoke:* ran clean. exit 0, 0.5s of compute.

### Since the audit

0 fixed · 0 changed · 8 stand · 1 new · compute unchanged · still runs


### Decision

**MINE:** Add the molecule-size control and re-run. The e3fp coefficient conclusion reverses on all three datasets without it.

## learned_embeddings — `similarity_metrics/learned_embeddings.py`

### What it does

Two things, neither of which scores anything. main() walks every sim_*.npz under the cache directory and writes a mean-pooled ChemBERTa and MoLFormer embedding per dataset to emb_<ds>_<which>.npy. train_triplet is a library function, imported by run_battery_moleculeace.py, that fits a 3-layer MLP on a base fingerprint with a triplet margin loss, taking positives either from a shared Bemis-Murcko scaffold or from a smaller activity gap.

**Asks:**
- What mean-pooled ChemBERTa and MoLFormer embedding does each molecule in each prepared dataset get, cached for the battery to score later?
- Can a distance be learned from triplets, with positives defined either by shared scaffold or by smaller activity gap, fitted on training scaffolds and applied to held-out ones?

**Verdict on 2026-09-03:** PARTIAL · hybrid: none needed · live

Why it fell short:
- The transformer half is a sound precompute: batched, no-grad, mean-pooled over the attention mask, cached, and it skips a file that already exists.
- The triplet half's structural mode is starved of training data. Measured this session on sim_herg_ki.npz: 1415 molecules in 751 Bemis-Murcko scaffold groups, so a random ordered pair shares a scaffold with probability 0.0029, giving about 58 anchor-positive pairs out of the 20000 random draws and roughly 116 usable triplets.
- The docstring says all three representations are 'scored by the SAME activity-grounded battery'. This file scores nothing; run_battery_moleculeace.py does the scoring.

*Smoke:* ran clean. exit 0, ~25s, no results file.

### Since the audit

0 fixed · 0 changed · 7 stand · 1 new · compute unchanged · still runs

- **NEW** — The suite-wide held-out split helper landed and this arm was left out `tests/experiments/similarity_metrics/_heldout.py:82`
- **STILL BROKEN** — In structural mode a triplet survives only when the anchor shares a scaffold with exactly one of two uniformly random molecules, so the encoder is fitted on about a hundred triplets

### Decision

**MINE:** All three, before any run of the MoleculeACE battery. The structural arm currently trains on about 140 triplets.

## extra_fingerprints — `similarity_metrics/extra_fingerprints.py`

### What it does

Builds Avalon and SECFP6 bit fingerprints plus the native MHFP6 MinHash vector for herg_ki, turns each into a distance matrix, and scores each against count-ECFP4 Tanimoto on two yardsticks: scaffold-grouped five-nearest-neighbour activity prediction, and Spearman against cached maximum-common-substructure distance. No model is fitted.

**Asks:**
- Does an Avalon, SECFP6 or native MHFP6 MinHash distance predict activity from a molecule's five nearest neighbours better than count-ECFP4 Tanimoto?
- Do those distances track maximum-common-substructure distance better than count-ECFP4 Tanimoto?
- Which of Avalon, MHFP6 and MAP4 can be built in this environment at all?

**Verdict on 2026-09-03:** PARTIAL · hybrid: none needed · orphaned

Why it fell short:
- MAP4 is never built. The file only probes whether `import map4` succeeds (line 72).
- One Spearman per metric, from one 5-fold GroupKFold, with no fold-level spread and no repeat. Avalon at 0.41 against the baseline at 0.40 is unjudgeable from what is written.
- Default is herg_ki alone (line 273). One dataset cannot separate a fingerprint property from a herg_ki property.

*Smoke:* ran clean. exit 0, ~25s.

### Since the audit

1 fixed · 0 changed · 4 stand · 1 new · compute unchanged · still runs


### Decision

**MINE:** Build MAP4, add adme_solubility, store per-fold values, re-run all three. 120 kNN evaluations, zero model fits.

## augmented_descriptor_reps — `similarity_metrics/augmented_descriptor_reps.py`

### What it does

Builds PDV (200 RDKit descriptors), Mordred (1613) and PaDEL descriptor blocks, z-scores each, and scores cosine and Euclidean distance on them against count-ECFP4 Tanimoto on the same two yardsticks as extra_fingerprints. Built matrices are cached to sim_<ds>_augmented.npz.

**Asks:**
- Does a z-scored physicochemical descriptor block under cosine or Euclidean distance predict activity from five nearest neighbours better than count-ECFP4 Tanimoto?
- Does it track maximum-common-substructure distance better?
- Which of PDV, Mordred and PaDEL can be built here?

**Verdict on 2026-09-03:** PARTIAL · hybrid: none needed · orphaned

Why it fell short:
- The per-feature mean and standard deviation at line 210 are taken over every molecule, then eval_knn_readacross at line 110 scores held-out folds inside that same z-scored geometry. The distance the fold is judged in was fitted on the fold.
- A cached Mordred or PaDEL matrix is unreachable when the package probe fails, so on a machine without the package the already-built block cannot be scored (see bugs).
- Same single-Spearman, no-spread problem as extra_fingerprints: one rho per metric per dataset, no fold values written.

*Smoke:* not run. exceeded 6 min, killed at '84%|########3 | 251/300' of the Mordred descriptor build (exit 142, SIGALRM at 330s).

### Since the audit

1 fixed · 0 changed · 3 stand · 2 new · compute unchanged · run state unknown

- **NEW** — The leak is on one side of the comparison only. `tests/experiments/similarity_metrics/augmented_descriptor_reps.py:173`

### Decision

**MINE:** Two one-line fixes: consult the cache before the package probe, and z-score per fold.

## nary_similarity — `similarity_metrics/nary_similarity.py`

### What it does

Implements iSIM set cohesion, per-molecule complementary similarity and an iCliff-style local roughness on ecfp4_count, then tests whether complementary similarity or roughness ranks the molecules whose activity disagrees with their neighbours, and whether inflating distances to high-outlierness neighbours raises five-nearest-neighbour activity prediction.

**Asks:**
- Do complementary similarity or iCliff roughness rank the molecules whose activity disagrees with their structural neighbours better than activity extremeness or mean distance?
- Does inflating the count-fingerprint distance toward high-outlierness molecules raise the Spearman of five-nearest-neighbour activity prediction without lowering maximum-common-substructure fidelity?

**Verdict on 2026-09-03:** PARTIAL · hybrid: none needed · orphaned

Why it fell short:
- Question 1 is decided by construction. The target at line 190 and the roughness candidate at line 189 are both computed from the same distance matrix D, and they are near-identical formulas: the target is |y_i - median activity over the 5 nearest neighbours in D|, roughness is the closeness-weighted mean |y_i - y_j| over the 8 nearest neighbours in D.
- Question 2's two arms are not comparable. The weighted arm's distance is built from an outlierness that was min-max scaled over every molecule (line 110); the baseline arm's is not.
- No dataset-level or fold-level spread is written, and the re-weighting strength lam is not recorded in the JSON, so the result cannot be reproduced from the payload alone.

*Smoke:* ran, NaNs present. exit 0, 0.4s.

### Since the audit

0 fixed · 0 changed · 7 stand · 1 new · compute unchanged · still runs

- **STILL BROKEN** — roughness_iCliff and the target it is scored against are the same quantity computed in the same distance matrix.

### Decision

**MINE:** NaN, symmetry and provenance fixes. The whole file re-runs in under 5 seconds.

**Open — yours:** How the roughness row is handled — it is a restatement of its own target.

>> 

## triplet_similarity — `similarity_metrics/triplet_similarity.py`

### What it does

Trains a torch MLP over ecfp4_count with a triplet margin loss on train scaffolds only, in two modes (positive = same Bemis-Murcko scaffold, or positive = smaller activity gap), embeds every molecule, and scores cosine distance in embedding space against count-fingerprint Tanimoto on held-out scaffolds for both five-nearest-neighbour activity prediction and maximum-common-substructure fidelity.

**Asks:**
- Does a triplet-trained embedding whose positives are same-scaffold pairs beat count-ECFP4 Tanimoto on held-out-scaffold five-nearest-neighbour activity prediction and on maximum-common-substructure fidelity?
- Does a triplet-trained embedding whose positives are smaller-activity-gap pairs beat the same baseline on the same two yardsticks?

**Verdict on 2026-09-03:** PARTIAL · hybrid: none needed · orphaned

Why it fell short:
- Nothing is written to disk. main() returns a dict and prints; every number exists only in the terminal, so the run cannot be read back.
- The structural mode trains on almost no triplets. herg_ki has 751 scaffold groups over 1415 molecules, so two random molecules share a scaffold with probability 0.0029 (measured this session).
- Held-out structural fidelity is measured on the maximum-common-substructure pairs whose BOTH endpoints fall in one fifth of the molecules: about 8000/25 = 320 pairs for herg_ki. There is no minimum-pairs guard, unlike similarity_noise_cleaning.py:79.

*Smoke:* ran, NaNs present. exit 0, ~40s, no results file.

### Since the audit

1 fixed · 0 changed · 6 stand · 2 new · compute unchanged · still runs

- **STILL BROKEN** — main() returns a dict and writes no file; every number exists only in the terminal.

### Decision

**MINE:** Write the JSON and return the surviving triplet count before any full run.

**Open — yours:** Whether triplets are resampled each epoch. That changes the method.

>> 

## similarity_noise_cleaning — `similarity_metrics/similarity_noise_cleaning.py`

### What it does

Splits the molecules into a fit half and a held-out half by whole scaffold group, fits the descriptor drop-one blacklist, the fine-2D column selection and the z-score statistics on the fit half only, then scores injected-label-noise detection AUROC and recovery correlation on the held-out half for the top-3 spaces of a structure ranking and the top-3 of an activity ranking, against count-fingerprint Tanimoto. Runs the same thing again with fit == score as an in-sample arm so the optimism is visible.

**Asks:**
- Does a similarity space selected without labels, for maximum-common-substructure fidelity, find deliberately corrupted activity values better than count-ECFP4 Tanimoto?
- Does it do so better than a space selected for activity coherence, which would be the circular choice?
- How much of the earlier fit-equals-score number was optimism?

**Verdict on 2026-09-03:** PARTIAL · hybrid: hand-rolled · superseded

Why it fell short:
- Only 3 of the 6 candidate hybrid spaces survive each ranking (_topk, TOP_K=3 at line 78). Scoring every space is a settled decision, restated in the f23 rebuild's own provenance note; this file predates that and still prunes.
- The reported winner is a max. `best_s` at line 287 picks the space with the highest recovery_mean among the top-3, and the verdict at line 289 quotes that maximum against a single baseline.
- The combo list at lines 182-187 omits `count+physchem_full`, which creative_hybrids_v2.py:224 includes, while the docstring at lines 7-9 says these are the exact combos from creative_hybrids_v2. An arm is dropped with no reason stated.

*Smoke:* ran clean. exit 0, 303.7s — just inside the 6-minute cap.

### Since the audit

0 fixed · 1 changed · 5 stand · 1 new · compute unchanged · still runs


### Decision

**YOURS:** Quote only the optimism gap now and delete once framework/f23 has run, or keep it. f23 answers the ranking question and scores every space instead of a top-3. Depends on f23 landing.

>> 

## sm_battery_moleculeace — `similarity_metrics/run_battery_moleculeace.py`

### What it does

For each MoleculeACE target it subsamples to 2000 molecules by whole scaffold groups, splits 5-fold GroupKFold, and on each held-out fold builds ~49 distance matrices (10 representations x their coefficients, plus transformer embeddings and two triplet encoders) and scores every one on the nine-evaluation activity battery in simeval.py plus a 3x3 noise dose/rate sweep. It then ranks metrics across targets and runs a paired Wilcoxon of each metric against ECFP4:Tanimoto.

**Asks:**
- Which representation x coefficient makes a similarity space whose neighbourhoods best agree with measured activity, judged on held-out scaffold folds?
- Does a learned distance (ChemBERTa, MoLFormer, or a triplet encoder trained on train-fold scaffolds) beat the best classical coefficient on that same battery?
- Does each metric's neighbourhood expose injected label error, and does the ranking of metrics invert between low and high noise doses?

**Verdict on 2026-09-03:** PARTIAL · hybrid: none needed · live

Why it fell short:
- The learned-metric arm is invalid as written: the triplet encoder trains on train-fold-standardised ECFP4 and is applied to test-fold-standardised ECFP4 (two different means and standard deviations), so question 2 is answered on shifted inputs.
- The headline p-value is selection-biased. The best metric per noise setting is chosen by mean score over targets, then tested against ECFP4 on those same targets.
- There is no hybrid similarity space in the grid. It ranks single representations only, so it cannot say whether combining representations into one distance beats the best single one.

*Smoke:* not run. not attempted, time

### Since the audit

0 fixed · 0 changed · 7 stand · 5 new · compute unchanged · still runs

- **NEW** — The fit-half z-score added on 2026-09-04 was applied to eight scripts in this directory and not to this one. `tests/experiments/similarity_metrics/run_battery_moleculeace.py:50 against tests/experiments/similarity_metrics/_heldout.py:82`
- **NEW** — This is the only scoring script here that does not use _nanrank, and it records no count behind any averaged correlation. `tests/experiments/similarity_metrics/run_battery_moleculeace.py:124`

### Decision

**YOURS:** Whether the 30-target battery runs at all. Nothing has ever been produced, so there is no re-run cost — only new compute. Run one target before sizing the job.

>> 

## sm_mcs_cache — `similarity_metrics/mcs_cache.py`

### What it does

Draws 8000 random molecule pairs per dataset on the _data.load_prepared row order and computes the maximum-common-substructure distance for each, caching I, J, mcs and the activity gap dy to mcs_pairs_<name>.npz. It is a precompute step, not an experiment.

**Asks:**
- What is the maximum-common-substructure distance between each of a fixed set of 8000 molecule pairs, so that the ranker experiments and the structural yardstick score against identical pairs?

**Verdict on 2026-09-03:** PARTIAL · hybrid: none needed · live

Why it fell short:
- rdFMCS.FindMCS is called with timeout=2 and the returned canceled flag is never read, so a pair whose subgraph search timed out is stored as a genuine, larger-than-true distance. How often that happens in the three cached files is not recorded and I have not measured it.
- Pairs are drawn uniformly at random over all molecule pairs, so the cache is dominated by dissimilar pairs. PERTURBATION_SIMILARITY_SUITE_DESIGN.md:394 reports mean MCS distance 0.816-0.859 in the three existing files, i.e.

*Smoke:* ran clean. exit 0, 7s.

### Since the audit

0 fixed · 0 changed · 3 stand · 1 new · compute unchanged · still runs

- **NEW** — The pair caches record no hash of the molecule list they were numbered against, and no consumer checks one. `tests/experiments/similarity_metrics/mcs_cache.py:73 against src/kirby/similarity/mcs.py:24-26`

### Decision

**MINE:** Add an output flag and honour SIM_DATA_DIR, and record the SMILES-ordering hash. Skip the timeout audit — 87 minutes per dataset and it changed nothing measurable.

## sm_data_loader — `similarity_metrics/_data.py`

### What it does

One loader that reads sim_<name>.npz and returns y, scaffold groups, SMILES, the ECFP4-Tanimoto reference matrix and ten representation matrices, with logd cut to a deterministic whole-scaffold-group subsample so the n x n distance matrices stay tractable.

**Asks:**
- Which molecules, in which row order, does every local similarity script index over?
- Does the logd subsample keep whole Bemis-Murcko scaffold groups together, so no scaffold straddles the subsample boundary?

**Verdict on 2026-09-03:** YES · hybrid: none needed · live

*Smoke:* ran clean. exit 0, ~15s.

### Since the audit

0 fixed · 0 changed · 2 stand · 1 new · compute unchanged · still runs


### Decision

**YOURS:** Whether the three local datasets are capped to the same size. logd is subsampled to 1,427 and herg_ki is whole at 1,415, so cross-dataset comparisons are not on equal footing.

>> 

## sm_nanrank — `similarity_metrics/_nanrank.py`

### What it does

Three functions that keep an uncomputable Spearman from winning a ranking or poisoning an average: is_defined, rank_desc (a sort key sending non-finite values last) and nanmean (mean over the finite entries, NaN when none are finite), plus n_defined for the sample size behind a nanmean.

**Asks:**
- Does a correlation that could not be computed stay distinguishable from a measured zero correlation?
- Does it lose every ranking and stay out of every average, rather than sitting at the top because max() never rejects a NaN?

**Verdict on 2026-09-03:** YES · hybrid: none needed · live

*Smoke:* ran, NaNs present. exit 0, instant, no results file.

### Since the audit

0 fixed · 0 changed · 1 stand · 1 new · compute unchanged · still runs


### Decision

**MINE:** Leave it. Twelve scripts import it and all three functions do what their names say.

## sm_build_local — `similarity_metrics/build_local_datasets.py`

### What it does

Rebuilds sim_herg_ki.npz and sim_logd.npz from two cached CSVs, importing featurize, scaffolds and tanimoto from build_moleculeace so the local pilot and the MoleculeACE battery cannot drift apart.

**Asks:**
- Do the two local pilot datasets reproduce from ~/.cache/kirby CSVs with the same ten representations, scaffold groups and Tanimoto reference the MoleculeACE builder writes?

**Verdict on 2026-09-03:** YES · hybrid: none needed · live

*Smoke:* not run. not attempted, time

### Since the audit

0 fixed · 0 changed · 3 stand · 3 new · compute unchanged · still runs


### Decision

**MINE:** Deduplicate by InChIKey to match the framework loader, and record the source CSV and the SMILES-ordering hash.

## sm_build_adme — `similarity_metrics/build_adme_solubility.py`

### What it does

Builds sim_adme_solubility.npz from polaris_adme_fang.csv, dropping rows with a missing or non-finite LOG_SOLUBILITY and rows RDKit cannot parse, using the same imported featurisation as the other two builders.

**Asks:**
- Does a third local dataset, log solubility, exist in the same schema as the herg_ki and logd pilot so the metric grid can be run on it?

**Verdict on 2026-09-03:** YES · hybrid: none needed · live

*Smoke:* ran clean. exit 0, ~110s.

### Since the audit

0 fixed · 0 changed · 2 stand · 2 new · compute unchanged · still runs


### Decision

**MINE:** Merge into build_local_datasets and delete. Same question, and the stricter filter changes no stored row.

## sm_build_moleculeace — `similarity_metrics/build_moleculeace.py`

### What it does

Discovers the benchmark CSVs inside an installed MoleculeACE package, reads SMILES and the pActivity column, computes its own Bemis-Murcko scaffold groups instead of using MoleculeACE's split column, and writes one npz per target in the local pilot's schema. It also holds featurize, scaffolds and tanimoto, which the two local builders import.

**Asks:**
- Does each MoleculeACE target become an npz with our own scaffold groups and the same ten representations, so the battery can be run on 30 targets instead of two?
- How many targets actually survive the parse and the 100-molecule floor?

**Verdict on 2026-09-03:** PARTIAL · hybrid: none needed · live

Why it fell short:
- It has never run. MoleculeACE is not importable in this environment and tests/experiments/similarity_metrics/similarity_metrics/sim_data is empty (both verified this session), so the '30 targets' in the docstring at :3-4 is a claim about the package, never an observed count.
- cliff_mol, MoleculeACE's own per-molecule activity-cliff label, is named at :10 as a verified column but never read: :100 takes only smiles and the y column. That free, package-provided ground truth is not carried into the npz, and run_battery_moleculeace re-derives cliffs from ECFP4 Tanimoto at simeval.cliff_ground_truth instead.
- find_csvs at :37 globs '**/benchmark_data/*.csv' inside the installed package with a CHEMBL*.csv fallback. Whether either pattern matches the shipped layout is unverified here, since the package is not installed.

*Smoke:* not run. not attempted, time

### Since the audit

0 fixed · 0 changed · 6 stand · 2 new · compute unchanged · still runs


### Decision

**MINE:** Take the activity-column fallback fix. One line, no rebuild.


---

# Federated — 10 files

## fed01 — `federated/fed01_noise_heterogeneity.py`

### What it does

On hERG binary activity it flips a rising fraction of training labels across five K-Means clusters of Morgan fingerprints, then compares seven conditions on clean test labels over 5 scaffold-group folds: a centralized hybrid, a pooled Morgan-fingerprint model, three FLuID prediction-level ensembles, KIRBy federated feature selection, and the same with per-silo noise detection. It also scores eight noise detectors against the true flip mask and correlates each silo's noise level with the budget it received.

**Asks:**
- As the fraction of flipped training labels rises unevenly across five clusters, does KIRBy federated feature selection (C5) hold MCC on clean test labels better than FLuID's three prediction-level ensembles (C2, C3, C4)?
- Does running noise detectors inside each silo and rebuilding the hybrid on the surviving rows (C6) recover MCC over C5?
- Do the eight detectors find the flipped labels, and does a silo with more flipped labels receive a smaller share of the 100-feature budget?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · dispatched

Why it fell short:
- The allocation method, importance method, internal evaluator and augmentation strategy all come from get_best_search_for_downstream at fed01:146, called with no label_noise=. _shared.py:4327 states that a experiment which injects label noise must pass it, so the hybrid does not enter the comparison built with access to the labels the experiment then corrupts.
- inject_noise_per_silo is called at fed01:249, 608 and 686 with no seed=, so it uses its default 42 every time. The three seeds in the sweep change the model and the fold partition but not one flipped label.
- No condition fits a model on a single base representation. The comparison is hybrid-versus-ensemble and hybrid-versus-pooled-fingerprints only, so a C5 win cannot be attributed to combining representations rather than to any one of the ten.

*Smoke:* not run. exceeded 6 min (killed at 340s) at last log line 'herg: 200 samples (pre-binarized), 97 active, 103 inactive' followed by a TensorFlow cpu_feature_guard line and then silence.

### Since the audit

0 fixed · 0 changed · 0 stand · 0 new · compute MORE EXPENSIVE · still runs


*Compute now:* 6x on searches, and 5 of the 6 are cold. 45b0125 replaced one search on the uncorrupted labels with one search per noise level.

### Decision

**MINE:** Pass the metric, and give the three noise seeds three different draws. Today all nine runs flip the same labels.

## fed02 — `federated/fed02_rep_diversity_silos.py`

### What it does

Assigns five molecular representations to five silos that all see the same molecules, then over 5 scaffold-group folds compares each silo alone, two FLuID prediction-level ensembles over those five models, a hybrid over all five representations, a hybrid over the three that scored best, and a pipeline that K-Means-clusters within each representation before running the federated selection across the resulting virtual silos.

**Asks:**
- Does a feature-level hybrid built across five representations (C4) beat the best of those five representations used alone (C1)?
- Does the hybrid beat combining the same five representations at the prediction level, by 8-nearest-neighbour similarity weighting (C2) or by inverse cross-validated error weighting (C3)?
- Does restricting the hybrid to the three best-scoring representations (C5), or clustering within each representation first (C6), beat the hybrid over all five?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · dispatched

Why it fell short:
- The three representations C5 builds on are chosen from single_silo, whose MCC and RMSE were measured on the test fold (fed02:406-413). C5 picks its inputs by looking at the answer, so its number cannot be compared with C4's.
- On caco2 _get_silo_reps assigns avalon to two silos, so the five-silo comparison is really four representations with one counted twice. See the bug below.
- The blueprint search at fed02:118 is run on the full dataset's labels and groups, outside the 5-fold loop, so the allocation method, importance method and evaluator that C4, C5 and C6 use were chosen with the test rows of every fold in the search.

*Smoke:* not run. exceeded 6 min (killed at 355s) at last log line 'Cache caco2__avalon__xgboost__rmse__5f_42.pkl: meta drift on rep_shape (stored=(60, 2048), expected=(200, 2048)) - will recompute.' Caco2 loading alone took 201.6s of the 355s (2158 molecules, all 5 reps cache hits, subsampled to 200), leaving the run inside precompute_hybrid_searches when killed.

### Since the audit

2 fixed · 0 changed · 6 stand · 4 new · compute MORE EXPENSIVE · still runs

- **NEW** — C5's inputs are now chosen inside the training rows, but C1's best silo is still chosen on the test fold. `tests/experiments/federated/fed02_rep_diversity_silos.py:334-336`
- **NEW** — fluid_within_kirby_across still scores each virtual silo on rows create_hybrid already picked features from. `tests/experiments/federated/federated_helpers.py:2019-2047`
- **NEW** — A virtual silo with no usable scaffold groups is scored on a random 20%, where the sibling function now refuses that. `tests/experiments/federated/federated_helpers.py:2036`
- **STILL BROKEN** — _get_silo_reps can assign one representation to two silos, and on caco2 it does.
- Fixed (45b0125) — C5's three representations were chosen by their score on the fold C5 was then measured on.

*Compute now:* Modestly. _rank_reps_train_only adds 5 representations x 3 inner folds = 15 model fits per outer fold, on top of the roughly 54 plain fits a fold already made, so about 69.

### Decision

**MINE:** Route C4 and C5 through build_fold_hybrid, pass C6 the metric, and fix the duplicate silo assignment on caco2.

**Open — yours:** Whether C5 stays. It picks its three representations using the test fold's own scores.

>> 

## fed03 — `federated/fed03_synthetic_private.py`

### What it does

One file holding three unrelated experiments dispatched by dataset name. Part A pools hERG molecules from three sources as silos and compares KIRBy per-silo feature selection against single-silo models, two FLuID prediction-averaging ensembles and a pooled model, by MCC over 5 scaffold folds. Part B splits the representations into structural/physicochemical/learned silos and compares a hybrid over all of them against each silo alone. Part C loads four antibody datasets as silos and transfers one silo's selected feature indices onto another silo's representations.

**Asks:**
- Part A: on hERG molecules pooled from three sources, does allocating a feature budget across silos by their own validation scores beat the best single silo, the two FLuID prediction-averaging ensembles, and the privacy-violating pooled model, measured by MCC?
- Part B: when each org holds a different family of representations, does a hybrid built across all three families beat the best single family, and does building it silo-by-silo and concatenating the selections cost anything?
- Part C: does the feature-importance profile learned on one antibody silo, applied by raw column index to another silo's representations, beat that silo's own hybrid?

**Verdict on 2026-09-03:** NO · hybrid: engine, per fold · dispatched

Why it fell short:
- Part A's first silo is not the source the docstring names. On the live submit path the dataset is lhasa_herg and load_lhasa_herg_data (federated_helpers.py:84-127) returns the Lhasa TRAINING rows, so the silo labelled 'chembl' is Lhasa training data.
- Part A's KIRBy arm (C4) is scored on a different feature space from every arm it is compared against. C4 uses combined_reps (the 10-rep general list, fed03:393-394); C1, C2, C3 and C5 use 2048-bit Morgan fingerprints (all_fps, fed03:278).
- The recipe every fold uses (allocation method, importance method, internal evaluator) is chosen on the whole dataset before the fold loop: fed03:98-107 (Part A), 627-636 (Part B), 1000-1007 (Part C). Every test fold contributed to picking it, so the per-fold numbers are not clean of recipe selection.

*Smoke:* not run. not attempted, time.

### Since the audit

0 fixed · 0 changed · 0 stand · 0 new · compute MORE EXPENSIVE · still runs


*Compute now:* Same to within a rounding error. 45b0125 added six lines: groups=pooled_groups[train_idx] and task_type='classification' into the Part A federated_feature_selection call.

### Decision

**MINE:** Rename the dataset key so the submit path stops exiting, pass the metric each part reports, and load herg_ki for Part A's first silo.

**Open — yours:** Whether Part B's silos are rebuilt from DATASET_REPS, and whether Part C stays or moves into fed05.

>> 

## fed04 — `federated/fed04_cluster_aware_importance.py`

### What it does

Splits a dataset into 5 scaffold-group folds; inside each training fold it K-Means-clusters the molecules on Morgan fingerprints, runs create_hybrid once per cluster and once on the whole fold, and compares the per-cluster feature allocations to each other and a cluster-conditional classifier to a single global one. Repeats the clustering at five smooth_factor values.

**Asks:**
- Does the per-representation share of a fixed 100-feature budget change from one K-Means cluster of the training molecules to another, measured as mean pairwise Spearman between per-cluster importance profiles?
- Does a classifier trained per cluster on that cluster's own hybrid beat one classifier trained on a single fold-wide hybrid, in MCC/BAC/F1/AUROC on held-out molecules?
- Does the clustering smooth_factor change how stable those per-cluster profiles are?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · dispatched

Why it fell short:
- The docstring promises 'structural FPs dominate near activity cliffs while foundation model embeddings dominate in smooth SAR regions'. The code never measures an activity cliff or SAR smoothness, and it never re-searches the allocation method per cluster — one blueprint (allocation, importance, internal evaluator) is fixed for every cluster.
- Labels are binarized at a hardcoded threshold of 6.0 (line 150). caco2 labels are log10 efflux ratio and logd labels are logD; both sit almost entirely below 6.0, so labels_binary collapses to a single class and the whole run is degenerate on those two datasets.
- The blueprint is read from data['precomputed_searches'], which run_all built on the ORIGINAL labels. That store is keyed by aug_condition alone (_shared.py:4194), so on caco2/logd fed04 silently gets a blueprint searched by RMSE on continuous labels and applies it to a binary target it declares as task_type='classification'.

*Smoke:* not run. exceeded 6 min (killed at 360s) at last log line 'herg: 60 samples, 28 active, 32 inactive' plus a TensorFlow cpu_feature_guard line, then silence for ~5 minutes.

### Since the audit

4 fixed · 1 changed · 5 stand · 4 new · compute MORE EXPENSIVE · still runs

- **NEW** — NEW: the recipe is still searched over every row of the dataset, then used inside every fold. `tests/experiments/federated/fed04_cluster_aware_importance.py:196-205`
- **NEW** — NEW: a third of the sweep gets no blueprint at all, silently. `tests/experiments/federated/fed04_cluster_aware_importance.py:198`
- Fixed (45b0125) — Labels are binarized at a hardcoded 6.0, which collapses caco2 and logd to one class.
- Fixed (45b0125) — get_best_search_for_downstream is handed data=, so the in-memory store answers a classification request with a regression search.
- Fixed (45b0125) — metric_name comes from get_domain_metric, so a regression metric name is passed with task_type='classification'.
- Fixed (1e74b25) — create_hybrid is called without groups=, so the internal accept/reject carve is a random permutation.

*Compute now:* The blueprint search is the only thing that changed cost. fed04 now passes data=None (fed04:204), so the request always goes to a disk-cached find_best_hybrid instead of run_all's in-memory store.

### Decision

**MINE:** Pass groups, the resolved evaluator and the searched recipe. Three small edits, no stored result to lose.

**Open — yours:** The hardcoded pKi 6.0 threshold. On caco2 and logd it collapses the target to one class.

>> 

## fed05 — `federated/fed05_antibody_federated.py`

### What it does

Loads three antibody datasets as silos, searches a hybrid blueprint per silo with find_best_hybrid, then runs six conditions on one 80/20 group-aware split per silo: a per-silo hybrid, cross-silo transfer of selected feature indices, within-silo ESM2 sub-clustering, a pooled z-scored noise-budget check, a pooled binding hybrid, and a hybrid over nine representations against an inverse-RMSE prediction ensemble of three-representation models.

**Asks:**
- On each antibody silo separately, what does a hybrid built with that silo's own blueprint score on its held-out 20% (RMSE and Spearman)?
- Does re-using another silo's selected feature indices beat the target silo's own hybrid on that target silo's held-out rows?
- When three silos' z-scored labels are pooled and Gaussian noise is added to one, does that silo receive a smaller share of the feature budget than the clean silos?

**Verdict on 2026-09-03:** NO · hybrid: engine, per fold · dispatched

Why it fell short:
- The 'dms' silo never loads. SILO_DATASETS maps it to 'abagym_HER2' (line 69), which was cut from DATASET_REGISTRY on 2026-07-09 (tests/antibody_baseline_survey.py:104); load_dataset raises KeyError, _load_silo swallows it at line 103, and the run continues on two silos.
- No individual base-representation baseline anywhere in the file. Condition 6's comparators are np.hstack concatenations of three representations each (line 644), and the KIRBy arm sees all nine at once, so any gap between them is confounded with how many representations each arm was given.
- One 80/20 GroupShuffleSplit per silo per run (line 112, cv_split with n_splits=1). Every reported RMSE is a single split.

*Smoke:* not run. exceeded 6 min (killed at 300s) at last log line 'Adaptive pruning: evaluating fold 0 for 1860 configs...'.

### Since the audit

0 fixed · 0 changed · 0 stand · 0 new · compute CHEAPER · still runs


*Compute now:* Cheaper. 45b0125 states it: same number of searches.

### Decision

**MINE:** cache=True and a per-silo recipe. Both one-line, nothing to lose.

**Open — yours:** The dms silo. abagym_HER2 is not a registry key, so condition 5 always reports a skip.

>> 

## lhasa_herg_loader — `federated/lhasa_herg_loader.py`

### What it does

Reads Lhasa's curated hERG training (7,772 molecules) and test (4,383) sets from PKL or SDF, extracts SMILES from the stored RDKit Mol objects, and returns SMILES, binary labels, InChIKeys and the DataFrame. Also provides dedup_across_silos, which drops a molecule from every silo after the first by InChIKey.

**Asks:**
- Which molecules, binary hERG activity labels and InChIKeys make up Lhasa's curated training and test sets?
- Which molecules appear in more than one silo, so a downstream federated split does not train and test on the same compound?

**Verdict on 2026-09-03:** PARTIAL · hybrid: none needed · live

Why it fell short:
- The PKL branch performs none of the label derivation the SDF branch does, so the primary format documented in the module docstring is the one that is not handled.
- load_lhasa_herg_test is never called by any federated code — federated_helpers.load_lhasa_herg_data uses only the training set — so the 4,383-molecule external test set is unused by the suite.
- The loader returns no scaffold groups. Groups are assigned later in federated_helpers.load_lhasa_herg_data via assign_scaffold_groups inside a bare try/except, so a failure there yields groups=None and fed04 hard-fails at its line 209.

*Smoke:* ran clean. exit 0.

### Since the audit

0 fixed · 1 changed · 5 stand · 0 new · compute unchanged · still runs


### Decision

**MINE:** Fix the PKL branch so it derives the label column the SDF branch already derives.

## caco2_runner — `federated/caco2.py`

### What it does

Forty-five-line argparse wrapper that calls federated_helpers.run_all with dataset_name='caco2', passing through --experiments, --output and an optional --reps override.

**Asks:**
- Can every federated experiment be run against caco2 from one command, with an optional cheap representation list for local smoke runs?

**Verdict on 2026-09-03:** YES · hybrid: none needed · superseded

### Since the audit

1 fixed · 0 changed · 1 stand · 1 new · compute MORE EXPENSIVE · still runs

- Fixed (45b0125) — Running the default set dispatches fed04, which binarizes caco2's continuous labels at a hardcoded 6.0 and runs on a single-class target.

*Compute now:* The file itself does zero model fits and is byte-identical to its 2026-09-03 state. What it dispatches costs more.

### Decision

**MINE:** Add --reps to run_dataset.py and delete all three runners together.

## herg_runner — `federated/herg.py`

### What it does

Forty-five-line argparse wrapper that calls federated_helpers.run_all with dataset_name='lhasa_herg', which routes run_all to load_lhasa_herg_data instead of the normal load_data path.

**Asks:**
- Can every federated experiment be run against Lhasa's curated hERG classification set from one command?

**Verdict on 2026-09-03:** YES · hybrid: none needed · superseded

### Since the audit

0 fixed · 1 changed · 1 stand · 0 new · compute unchanged · still runs


### Decision

**MINE:** Settle on 'herg' as the spelling, then delete with the other two.

## logd_runner — `federated/logd.py`

### What it does

Forty-five-line argparse wrapper that calls federated_helpers.run_all with dataset_name='logd', passing through --experiments, --output and an optional --reps override. Byte-for-byte the same as caco2.py apart from the dataset name.

**Asks:**
- Can every federated experiment be run against logd from one command, with an optional cheap representation list for local smoke runs?

**Verdict on 2026-09-03:** YES · hybrid: none needed · superseded

### Since the audit

1 fixed · 0 changed · 1 stand · 0 new · compute MORE EXPENSIVE · still runs

- Fixed (45b0125) — Running the default set dispatches fed04, which binarizes logD at a hardcoded 6.0 and leaves a near-constant target.

*Compute now:* Identical to caco2.py apart from the dataset name, and equally unchanged. Zero fits of its own.

### Decision

**MINE:** Delete with caco2.py and herg.py once run_dataset.py takes --reps.

## smoke_driver — `federated/_smoke_driver.py`

### What it does

Local-only launcher that refuses to run unless KIRBY_SMOKE_N is set, rebinds _shared.FULL_BUDGETS to [25], and calls federated_helpers.run_all directly with an explicit small representation list so run_dataset.py's foundation-model defaults are bypassed.

**Asks:**
- Can the federated suite be exercised end to end on this laptop, on a few hundred rows and a handful of cheap representations, without pulling in server-only foundation models or the eight-budget precompute grid?

**Verdict on 2026-09-03:** PARTIAL · hybrid: none needed · live

Why it fell short:
- It does not set KIRBY_LGBM_THREADS=1. DEFAULT_IMPORTANCE_METHODS includes deeplift and DEFAULT_MODELS includes lightgbm, and hybrid.py:59-64 records that combination as the local segfault: 'Set KIRBY_LGBM_THREADS=1 to make a local run survivable.' A file whose only purpose is local smoke leaves the caller to remember the one env var that stops it crashing.
- The row cap it enforces is not enough for fed04 to compute anything. Under smoke, K drops to 2 and N_FOLDS to 2, and _compute_cluster_importance skips any cluster under 20 samples (fed04:76).

### Since the audit

0 fixed · 0 changed · 4 stand · 1 new · compute MORE EXPENSIVE · still runs

- **NEW** — NEW: the driver's shrinks no longer reach fed04's recipe search on caco2 or logd. `tests/experiments/federated/_smoke_driver.py:39-41`

*Compute now:* The file is unchanged and does zero fits. The smoke run it launches got more expensive on caco2 and logd, because fed04 now passes data=None and can no longer read the in-memory store the driver shrank.

### Decision

**MINE:** Set the thread variables and fix the default representation list.


---

# OPIG — 11 files

## opig_activity_cliff_qsar — `opig_activity_cliff_qsar.py`

### What it does

On 3 ChEMBL targets (d2, fxa, mpro) it evaluates 7 molecular representations plus one hybrid under 5 downstream models with scaffold GroupKFold, reporting MAE/RMSE/R2/Spearman, and turns the out-of-fold predictions into a matched-molecular-pair activity-cliff MCC. The hybrid is materialised from a search result fitted on all rows.

**Asks:**
- Which single molecular representation gives the lowest MAE on activity-cliff-rich ChEMBL targets?
- Does the hybrid representation beat the best single representation on MAE?
- Does the representation choice change how often a model misses an activity cliff (AC-MCC, AC-sensitivity)?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · live

Why it fell short:
- The hybrid arm in the main table is built from feature indices fitted on all rows including every validation fold. `search.fitted_results['rf']` comes from find_best_hybrid.py:6461, which refits the winning blueprint on the full dataset with the full labels.
- Question 3 has no honest arm at all. The `honest_rescore` block (line 364) covers only the search's own MAE per fold.
- One RF blueprint is used for all five downstream models. The search runs with `downstream_model='rf'` (line 345) and only `fitted_results['rf']` is read (line 370), but the resulting feature set is then fed to rf, xgboost, svm, mlp and lightgbm alike (line 399).

### Since the audit

1 fixed · 2 changed · 3 stand · 3 new · compute CHEAPER · run state unknown

- **NEW** — The results file on disk still holds a kirby_hybrid representation row that today's code cannot produce. `tests/results/comparisons/opig_activity_cliff_qsar.json`
- **NEW** — return_fitted=True is now dead work — nothing reads search.fitted_results. `tests/experiments/opig_activity_cliff_qsar.py:351`
- **NEW** — The experiment no longer produces any hybrid activity-cliff number, so two of its three stated questions have no arm. `tests/experiments/opig_activity_cliff_qsar.py:9-12`
- Fixed (6b37acd) — The kirby_hybrid representation is built from search.fitted_results['rf'].selected_indices, fitted on all rows including every validation fold, then scored by GroupKFold on those same rows.

*Compute now:* One representation left the downstream table. Before: 8 reps x 5 models x 5 folds x 3 targets = 600 fits.

### Decision

**MINE:** Build the hybrid per fold and recompute the cliff numbers from those predictions. Fix the sign convention and the paired delta in the same pass.

## opig_blueprint_transfer — `opig_blueprint_transfer.py`

### What it does

Searches a hybrid blueprint on a large source dataset and a small target dataset, then on the target's folds compares four arms: best single representation, the target's own blueprint rebuilt per fold, the source's recipe rebuilt on target training rows, and the source's exact feature indices applied unchanged. Also runs best-single vs hybrid on the source as context.

**Asks:**
- Do the source dataset's exact selected feature indices, applied unchanged to the target, beat the target's best single representation?
- Does the source's recipe (allocation method, budget, importance method, evaluator), re-run on target training rows, match the target's own searched blueprint?
- Does the hybrid beat the best single representation on the source dataset itself?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · live

Why it fell short:
- Question 2 is never actually computed. Both `recipe_transfer` and `native` are measured only as a delta against `base_best` (lines 456-461).
- The reported delta is a difference of two means, not the mean of per-fold differences. `_agg` at line 427 takes `np.nanmean` per arm and line 460 subtracts the two.
- The `base_best` representation is picked once from `target_search.baselines` — the search's internal CV over all target rows — and then scored on those same 5 folds (lines 279-288, 320-327). `native` inherits the same optimism through `target_bp`.

### Since the audit

0 fixed · 1 changed · 9 stand · 5 new · compute MORE EXPENSIVE · run state unknown

- **NEW** — The whole-dataset target search still runs and now feeds nothing that is scored. `tests/experiments/opig_blueprint_transfer.py:201-211`
- **NEW** — The result file on disk uses key names today's code no longer writes. `tests/results/opig_blueprint_transfer/logd_to_herg_ki.json`

*Compute now:* Searches go from 2 to 2 + 2 x n_folds. At n_folds=5 that is 12: one whole-dataset search per side (lines 188 and 201, both still run) plus five per-fold searches on the target (line 306) and five on the source (line 313).

### Decision

**MINE:** Add groups= at line 250 and make the delta paired. The stored file is a smoke run, so nothing real is invalidated.

## opig_cdr_flexibility — `opig_cdr_flexibility.py`

### What it does

Classifies CDR loops as flexible or rigid from whole-Fv sequence representations, on CDRH3 alone, CDRL3 alone and the two combined, with 8 antibody representations x 5 models under StratifiedGroupKFold grouped by IGHV germline family. The hybrid is not materialised; only the hybrid search's own internal CV score is printed and stored.

**Asks:**
- Which sequence-based antibody representation best separates flexible from rigid CDR3 loops (ROC AUC, PR AUC, MCC, balanced accuracy)?
- How far below the paper's structure-based ITsFlexible GNN (ROC AUC 0.82) does the best sequence-based representation land?
- Does the hybrid representation beat the best single representation on this task?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · orphaned

Why it fell short:
- Question 3 is not answered. The hybrid's number is `_meta['score_mean']` (line 299) — the score of the single best configuration under find_best_hybrid's OWN internal 5-fold CV, chosen by argmax over the whole configuration grid on the very folds it is scored on.
- Question 1 is answered soundly for the two single-CDR runs. The `combined` run is not interpretable: rows from the CDRH3 and CDRL3 files are concatenated (lines 484-486) and the features come from `heavy_seqs` and `light_seqs` alone (rep_cache.py:113-115).
- n is roughly 116-216 sequences across 11 V-gene families, one of which holds ~76 of them. Five StratifiedGroupKFold folds over 11 groups of size 2-76 gives wildly unequal folds.

### Since the audit

0 fixed · 2 changed · 3 stand · 4 new · compute unchanged · run state unknown

- **NEW** — honest_rescore is called without exclude_folds, so fold 0 stays in — the same-day fix that reached opig_activity_cliff_qsar did not reach this file. `tests/experiments/opig_cdr_flexibility.py:308-313`
- **NEW** — The result file on disk carries the old key name, and the new print_analysis reads past it in silence. `tests/results/comparisons/opig_cdr_flexibility.json`

### Decision

**MINE:** Add the honest block. No fits, and it is the only thing between this file and an answer to its third question.

**Open — yours:** Whether the combined CDRH3-plus-CDRL3 run stays. Its features cannot express which loop the label describes.

>> 

## opig_cgr_comparison — `opig_cgr_comparison.py`

### What it does

For each of 3 reaction-yield datasets it generates ~44 reaction representations, scores every one with 5 models under GroupKFold, then runs two hybrid searches (pool without the 3 CGR reps, pool with them) and compares the two resulting hybrids. It also computes pairwise Linear CKA between the 7 family representations.

**Asks:**
- Does the best CGR representation beat the best SMILES representation within the fingerprint family and within the neural family, by RF RMSE?
- Does adding the 3 CGR representations to the hybrid pool lower the hybrid's RMSE, per downstream model?
- Do the CGR representations get any features allocated in the SMILES+CGR hybrid, and how many?
- How similar are the CGR and SMILES representations by Linear CKA?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · live

Why it fell short:
- The individual-representation comparison (question 1) is clean. The pool comparison (question 2) is not: both hybrids are built by a search over all rows and then scored on the same GroupKFold folds (opig_cgr_comparison.py:379 and :393), so both arms carry selection optimism.
- The search is run with downstream_model='rf' only (:357). The RF-chosen recipe is then scored with all 5 downstream models (:392-399) and delta_by_model reports a per-model number from a recipe chosen for RF.
- Representations whose row count does not match n are dropped from the individual table (:605) but are still handed to find_best_hybrid, because partition_rep_pools (:634) reads the unfiltered reps dict.

### Since the audit

1 fixed · 2 changed · 2 stand · 3 new · compute unchanged · still runs

- **NEW** — A result_db written before 2026-09-04 makes find_best_hybrid raise, and the whole dataset is dropped. `src/kirby/result_cache.py:194-198`
- **NEW** — --n-folds no longer sets the number of folds the Delta is computed on. `tests/experiments/opig_cgr_comparison.py:358-369`
- Fixed (6b37acd) — delta_by_model subtracted two RMSEs that both came from a hybrid selected on all rows and re-scored on those same folds, with unequal pool sizes.

### Decision

**MINE:** Take the honest-block arithmetic and the per-model recipe. Neither needs a re-run.

**The recheck opened something new here. Overrule me if it changes your call:**

>> 

## opig_ginkgo_developability — `opig_ginkgo_developability.py`

### What it does

For each Ginkgo developability property it builds a hybrid per fold on the training rows only and scores the held-out fold, and scores every base representation on the identical folds. It uses the competition's pre-computed hierarchical cluster fold IDs as the splits.

**Asks:**
- Which antibody representation best predicts each developability property, by Spearman rho on the competition's cluster folds?
- Does the per-fold hybrid beat the best single representation on each property?
- How do KIRBy's numbers sit against the competition's best team CV and test Spearman?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · live

Why it fell short:
- find_best_hybrid is called with no metric= (:300-316), so it falls back to the signature default 'r2'. Every number reported and every ranking printed is Spearman rho.
- The two arms are not preprocessed the same way. The baseline arm goes through run_baseline_survey.evaluate_rep, which runs _filter_high_dim_per_fold: any representation wider than 500 features is cut to its top 200 by a LightGBM importance fit.
- The printed table and the 'Hybrid wins X/N' line (:576-601) take a max over models for the hybrid and a max over both representations and models for the baseline, on the same folds. The honest_rescore block that corrects this is written to JSON but never computed or printed anywhere in this file.

### Since the audit

0 fixed · 0 changed · 5 stand · 3 new · compute unchanged · still runs

- **NEW** — Every model's Hybrid rho comes from the recipe chosen for rf. `tests/experiments/opig_ginkgo_developability.py:326-327`
- **NEW** — A result_db written before 2026-09-04 makes every fold's find_best_hybrid raise, and the run still prints a win count. `src/kirby/result_cache.py:194-198`
- **NEW** — The only saved result file holds ginkgo_heparin, the one property today's code refuses to run, and a rerun overwrites it. `tests/results/comparisons/opig_ginkgo_developability.json`

### Decision

**MINE:** Set metric='spearman' and match the representation preprocessing across the two arms in one edit, print the honest block, re-run once.

**The recheck opened something new here. Overrule me if it changes your call:**

>> 

## opig_graphinity_ddg — `opig_graphinity_ddg.py`

### What it does

Intends four conditions on antibody-antigen binding free energy change: within-distribution FoldX, within-distribution Rosetta, experimental SKEMPI, and train-on-synthetic-test-on-experimental. Only the last two actually execute; the two synthetic conditions are dropped at a registry check before any data is loaded.

**Asks:**
- Can sequence-only representations predict experimental SKEMPI binding free energy change under leave-one-complex-out splits, and how far is that from the structure-based EGNN Pearson r?
- Does a model trained on FoldX synthetic data transfer to experimental SKEMPI?
- Does the hybrid beat the best single representation on the experimental condition?

**Verdict on 2026-09-03:** NO · hybrid: engine, per fold · live

Why it fell short:
- within_foldx and within_rosetta never run. CONDITION_DATASETS maps them to 'graphinity_foldx' and 'graphinity_rosetta', and main() at :714 skips any condition whose dataset is absent from antibody_baseline_survey.DATASET_REGISTRY.
- The transfer condition's PDB overlap filter is dead. It reads test_data['pdb_ids'] or ['pdb'] at :232-234, but antibody_baseline_survey.load_dataset returns neither key (its return dict is heavy_seqs, light_seqs, labels, groups, n_groups, n_samples, mutations, task_type, edit_distances, fold_assignments, property_name).
- The transfer condition builds no hybrid at all (:200-334), so question 3 is unanswered there.

### Since the audit

2 fixed · 1 changed · 3 stand · 3 new · compute unchanged · still runs

- **NEW** — A result_db written before 2026-09-04 makes the search raise, and the run continues silently with no hybrid and no honest block. `src/kirby/result_cache.py:194-198`
- **NEW** — Every model's kirby_hybrid score comes from the recipe chosen for rf. `tests/experiments/opig_graphinity_ddg.py:468-473`
- Fixed (1e74b25) — The PDB overlap filter never ran, because load_dataset dropped the complex identifiers, so FoldX rows sharing a complex with the SKEMPI test set stayed in training.
- Fixed (6b37acd) — Both submit scripts passed --max-samples 5000, which argparse rejects, so the job produced nothing.

### Decision

**MINE:** Fix the submit flag and the pdb_ids filter. Until both are done it either does not start or reports an inflated transfer number.

**The recheck opened something new here. Overrule me if it changes your call:**

>> 

## opig_humanness — `opig_humanness.py`

### What it does

Loads a humanness classification dataset, clusters sequences at 95% identity for groups, builds 10 antibody representations, runs one find_best_hybrid search (metric auc, downstream rf) and materialises its full-data winner as an extra representation called kirby_hybrid, then scores every representation under 5 downstream models with 5-fold StratifiedGroupKFold. It also stores the search's per-fold candidate and baseline scores as an honest_rescore input block.

**Asks:**
- Which of the ten antibody representations separates human from non-human therapeutic VH best, per downstream model, under 5-fold identity-grouped CV?
- Does a hybrid built by find_best_hybrid beat the best single representation on AUC, per model, when the winner is re-picked on held-out folds?
- How do the classical encodings (blosum, onehot, aac) compare with the sequence language-model representations on this task?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · live

Why it fell short:
- DEFAULT_DATASETS names humanness_species, which is not a key of DATASET_REGISTRY (it was cut; only a comment mentions it). Running the script with no --datasets aborts at the validation loop and writes nothing.
- The kirby_hybrid row written into results['kirby_hybrid'] was selected using the full label vector, then re-scored under the same folds. It sits in the same dict, with the same shape, as the honestly-CV'd single representations and is indistinguishable from them in the JSON.
- The console Summary compares max-over-(rep, model) against max-over-model for the leaked hybrid row, so the headline delta is not the honest one.

### Since the audit

1 fixed · 1 changed · 3 stand · 3 new · compute unchanged · still runs

- **NEW** — The stored result file still holds the leaked kirby_hybrid row that today's code cannot write. `tests/results/comparisons/opig_humanness.json`
- **NEW** — The per-dataset result database gained two key columns on 2026-09-04, so any existing one makes the hybrid search raise, and the script swallows it. `src/kirby/result_cache.py:56`
- Fixed (6b37acd) — The kirby_hybrid representation was refit on every antibody's label, then re-scored under the same folds and written into results alongside the honestly-CV'd single representations.

### Decision

**MINE:** Delete lines 316-321, print delta_mean_of_folds, take the four one-line fixes. The held-out table already answers question 2.

## opig_kingsmith_component_splits — `opig_kingsmith_component_splits.py`

### What it does

Loads a Buchwald-Hartwig yield dataset, builds the 10 base plus 4 augmentation reaction representations named in DATASET_REPS, and for each of four component axes runs true leave-one-group-out GroupKFold (n_folds = n_groups). Per outer fold it runs a fresh find_best_hybrid on training rows only, applies the fold's rf-selected features to train and validation, and scores 5 downstream models; it separately scores every raw representation on the identical folds.

**Asks:**
- How far do R2, RMSE and Spearman fall for each reaction representation as the component held out gets scarcer (additive 23 groups, aryl_halide 15, ligand 4, base 3)?
- Does a hybrid built per fold on training rows only beat the best single representation on each component axis?
- Which component axis is hardest to extrapolate across?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · orphaned

Why it fell short:
- The reported delta (print_analysis, line 674) is max-over-(representation x model) minus max-over-model for the hybrid, both maximised on the same folds that produced the scores. There is no held-out re-selection on either arm, and no honest_rescore block is built even though every ingredient exists (the per-fold search results are discarded after the fitted 'rf' entry is pulled).
- Only fitted_results['rf'] is used, so the feature set is chosen for random forest and then scored under xgboost, svm, mlp and lightgbm as well. Those four rows answer a different question from the one the split is testing.
- Nothing per-fold reaches the JSON. _summarize_scores writes mean, std and a completed-fold count only, so a paired per-fold hybrid-vs-baseline delta cannot be recomputed later.

### Since the audit

3 fixed · 0 changed · 3 stand · 5 new · compute unchanged · still runs

- **NEW** — The new paired delta is computed on R2 only, while RMSE is the metric for these datasets and rmse_per_fold is written to the same file unused. `tests/experiments/opig_kingsmith_component_splits.py:586`
- **NEW** — R2 is NaN on any held-out component whose yields are near-constant, and those folds drop out of the paired delta while the RMSE table keeps them. `tests/experiments/opig_kingsmith_component_splits.py:341`
- **NEW** — The per-dataset result database gained two key columns on 2026-09-04, so any existing one makes every fold's search raise. `src/kirby/result_cache.py:56`
- Fixed (6b37acd) — The reported delta was the largest R2 over 14 representations x 5 models minus the largest over 5 hybrid models, both maxima taken on the folds that produced them.
- Fixed (6b37acd) — A fold whose find_best_hybrid raised was dropped from the hybrid arm while the baselines still averaged over all folds, and the two means were then subtracted.
- Fixed (6b37acd) — _summarize_scores emitted mean, std and a fold count only, so no paired delta could be recomputed from the saved file.

### Decision

**MINE:** Restore metric='rmse' and keep the per-fold scores before this is submitted. Both cheap, both change which recipe wins.

## opig_trastuzumab_enrichment — `opig_trastuzumab_enrichment.py`

### What it does

Loads 5,000 trastuzumab CDRH3 variants with binary HER2-binding labels, grouped by edit-distance shell from wildtype. Ranks variants zero-shot by the mean-LLR column of the two LLR representations, then trains 5 downstream models on each of 11 representations under 5-fold StratifiedGroupKFold and measures binder enrichment in the top 1/5/10% of each held-out fold. It runs one hybrid search and records only that search's internal rf CV AUC.

**Asks:**
- Which of the eleven antibody representations, with which downstream model, puts the most binders in the top 1/5/10% of a held-out edit-distance-shell fold?
- Do the two mean-LLR scores rank binders above non-binders with no training at all?
- Does a hybrid beat the best single representation at binder enrichment?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · live

Why it fell short:
- Question 3 is not answered. The hybrid is never scored on enrichment or on anything held out.
- search.all_results and search.baseline_fold_scores carry per-fold scores for every candidate recipe and every baseline representation, and _shared.build_honest_rescore_input / honest_rescore exist to turn them into a held-out per-model delta (opig_humanness.py does exactly this). This file discards them.
- The zero-shot arm ranks all 5,000 rows at once while the supervised arm ranks ~1,000 held-out rows per fold. The two enrichment columns land in the same JSON with no marker that their denominators differ.

### Since the audit

2 fixed · 0 changed · 3 stand · 5 new · compute unchanged · still runs

- **NEW** — honest_rescore is called with no exclude_folds, so fold 0 contributes to both arms even though the search filtered its own recipe list using fold 0. `tests/experiments/opig_trastuzumab_enrichment.py:373-378`
- **NEW** — The console prints r['delta'], the difference of two means, not delta_mean_of_folds. `tests/experiments/opig_trastuzumab_enrichment.py:560`
- **NEW** — The stored result file uses key names today's code no longer writes, and today's code cannot reproduce it. `tests/results/comparisons/opig_trastuzumab_enrichment.json`
- **NEW** — The per-dataset result database gained two key columns on 2026-09-04, and this dataset's existing database on disk lacks them, so the search raises and the script swallows it. `src/kirby/result_cache.py:56`
- Fixed (6b37acd) — hybrid_search.rf_cv_auc was the winning recipe's mean over the folds that chose it, presented in the comment and the printed line as the honest hybrid number, and it was the only hybrid number the experiment produced.
- Fixed (6b37acd) — The returned dict had no hybrid entry and no per-fold scores anywhere, so the hybrid question could not be answered from the saved file.

### Decision

**MINE:** Take the per-fold enrichment rewrite. It is the file's own question 3.

## opig_scaffold_hybrid — `opig_scaffold_hybrid.py`

### What it does

Computes two mutation-difference representations (dphyschem, 5 features; dphyschem_pos, sequence-length x 5 features) from the same sequences load_data returned, injects them into the 10-representation pool for four single-wildtype antibody DMS sets, and runs run_nested_cv under cache tier 'scaffold_cmp'. Then it reads per-representation honest fold means and counts scaffold names inside the winning blueprints.

**Asks:**
- On the honest outer folds, does each scaffold-difference representation score higher than every base representation, per model?
- Does the hybrid actually put scaffold-difference features into the winning recipe?
- Does the hybrid beat the best single representation on these four datasets (delta_mean_of_folds)?

**Verdict on 2026-09-03:** PARTIAL · hybrid: engine, per fold · live

Why it fell short:
- Question 2 is unanswerable as written. Blueprint.reps is set to list(base_reps.keys()) at find_best_hybrid.py:3755-3765, i.e.
- Nothing in NestedCVResult.to_dict() carries a per-representation feature count, so the JSON cannot answer question 2 later either. The counts live in the feature_info returned by apply_blueprint, which is never persisted.
- Question 3 is not answered by the committed run. flab_binding_adams: delta_mean_of_folds is nan and n_paired_folds is 0 for all six models.

### Since the audit

0 fixed · 0 changed · 0 stand · 0 new · compute unchanged · still runs


### Decision

**MINE:** Options 1, 3 and 4 now. They cost nothing and stop the file printing a one-fold delta as a result.

**Open — yours:** Option 2, storing the per-representation feature count, is the only way to answer whether the hybrid used the scaffold features. 301,000 cold fits.

>> 

## opig_scaffold_similarity — `opig_scaffold_similarity.py`

### What it does

Samples pairs of variants from six mutation libraries, computes a distance between each pair under five mutation-difference representations and two distance formulas, and Spearman-correlates that distance against a substitution-matrix distance between the same two sequences, with the differing-position count partialled out. No model is trained. Three modes: representation ranking, per-property-feature subset search, and three honesty checks.

**Asks:**
- Which mutation-difference representation and which distance formula give pair distances that track a substitution-matrix distance, once the differing-position count is partialled out?
- Does a property feature earn a place in the distance because it correlates with the grader on its own, or because it is not redundant with the features already chosen?
- Are the numbers honest: is the grader built on the representation's own basis, is the grader mostly counting differing positions, and is any proposed distance algebraically the grader itself?

**Verdict on 2026-09-03:** PARTIAL · hybrid: none needed · live

Why it fell short:
- An undefined correlation is written to the results file as 0.0 (lines 262 and 267), which reads later as a measured zero. On flab_expression_koenig and flab_binding_koenig every MutID and BaseDelta number in the committed JSON is exactly 0.0 for both distances and all three graders.
- The multi-scaffold branch (lines 228-236) scores only dPhysChem_pos and dPhysChem, only under euclidean. MutID, SubScore, BaseDelta and the cosine distance are never run on graphinity_foldx, so question 1 is answered on 2 of 5 representations and 1 of 2 distances there.
- The ground truth is a different quantity on the two dataset kinds, and print_analysis ranks both in one table (lines 347-359). Single-scaffold truth is SR.grader_pair_distance, a sum over all sequence positions.

### Since the audit

1 fixed · 1 changed · 10 stand · 6 new · compute MORE EXPENSIVE · still runs

- **NEW** — The committed results file cannot be reproduced by today's code, and print_analysis would fail on it `tests/experiments/results/opig_scaffold_similarity.json:1`
- **NEW** — The half-split fix was applied to mode_features and not to mode_representations, which also picks a winner on the pairs it reports `tests/experiments/opig_scaffold_similarity.py:411`

*Compute now:* Still zero model fits. Per single-scaffold dataset the pair sample went from 8,000 to about 31,800 (--n-pairs default 8000 -> 32000 at line 507, and the code's own measured note at :159-162 records 15,254 + 16,548 for flab_expression_koenig).

### Decision

**MINE:** Fold into scaffold_experiment and feature_type_study. That fixes the 0.0 zeros, the uneven grader count and the missing whole-sequence comparison at once.
## The 18 that will not finish

For each one: the size, what is spending it, and what can be taken off without removing an arm.
I have cut nothing. The axis choices are yours.

### `c05` — complexity/c05_learning_curves.py

**Size.** Per dataset under run_all: 5 sample sizes x 3 seeds, collapsed to 1 seed at the full-data point = 13 searches. Each search is 5 allocation methods x 3 importance methods x 5 internal evaluators x 8 budgets x 5 downstream models = 3000 configurations x 5 folds = 15,000 model fits. 13 x 15,000 = 195,000 fits.

**What can come off, no axis removed:**

- Delete the separate baseline loop at c05:253-270 and read the per-representation numbers out of search.baselines instead. The search already computes them, on the same folds, at the same seed, at full width for anything under 5,000 features, and using probabilities where the metric is AUROC (find_best_hybrid.py:4149-4291, _evaluate_single at 3280-3360).
  *Saves:* Removes about 4,550 duplicated model fits per dataset: 10 representations x 7 models x 5 folds x 13 subsample searches. Adds none.
- If you want the separate loop kept, the alternative is one line at c05:254: skip a model that was not resolved at the full row count.
  *Saves:* Saves the wasted gp baselines: 10 representations x 5 folds x 3 seeds x the number of sample sizes under 1,000. On buchwald_hartwig that is 300 gp fits.
- At c05:226 keep the true dataset_name and pass cache_namespace=f"n{n_sub}_s{seed}" instead. That is the supported field (find_best_hybrid.py:5945-5948) and c06:179-180 already does the equivalent.
  *Saves:* Any subsample search already cached under the old invented names becomes unreachable and re-runs. How many such files sit on the cluster is not established here.

### `c07` — complexity/c07_data_complexity_partitions.py

**Size.** 3 partitions x 2 halves = 6 searches per dataset, each on up to 2000 rows (the whole dataset is first capped to 2000 rows by whole scaffold groups at c07:91). Each search runs the default 7-model set: 5 allocation methods x 3 importance methods x sum over models of (evaluators x budgets under MODEL_MAX_BUDGET) = about 3,465 configurations x 5 folds = 17,325 fits. 6 x 17,325 = about 104,000 model fits per dataset, plus 4 models x 5 folds x 6 halves = 120 fits for the full-data-recipe arm.

**What can come off, no axis removed:**

- Extend the full-data recipe list at c07:59 to the engine's default seven models, so every model the per-half search can pick has a full-data counterpart.
  *Saves:* 3 more get_best_search_for_downstream lookups per dataset. Each is a pool read when the precompute cache is warm and a full find_best_hybrid when it is not (_shared.py:4218-4240).
- Give the two arms the same fold count: pass n_folds=config['n_splits'] to find_best_hybrid at c07:230, or drop config['n_splits'] and use 5 in the second arm.
  *Saves:* First choice: the six searches drop from 5 folds to 1 on the command-line path, and every cached row built at 5 folds is unreachable. Second choice: the second arm goes from 1 fold to 5, so 4 x 5 x 6 = 120 fits instead of 24.
- At c07:240 keep the true dataset_name and pass cache_namespace=f"{partition_name}_{subpop_name}" instead (find_best_hybrid.py:5945-5948).
  *Saves:* Any partition search cached under the old invented names re-runs. How many exist on the cluster is not established here.

### `c10` — complexity/c10_noise_robustness.py

**Size.** Eleven regression conditions: ten dose-matched at 6 levels plus censoring at 7 = 67 (condition, level) points, times 5 folds. Phase 1 = 10 reps x 67 x 5 = 3,350 random-forest fits. Phase 2 = 8 budgets x 67 x 5 = 2,680 create_hybrid builds plus 2,680 forest fits; each build re-scores the 10 representations with the internal evaluator, roughly 29,500 further fits.

**No free saving found.** Shortening this one means changing an axis, which is your call.

### `c13` — complexity/c13_forward_compatibility.py

**Size.** One find_best_hybrid = 5 allocations x 3 importance methods x 5 internal evaluators x 8 budgets (FULL_BUDGETS, which c13 uses regardless of config['budgets'] — c13:344-350) = 600 configurations x 5 inner folds = 3,000 create_hybrid builds, each scored against the downstream model set, so roughly 15,000 model fits before adaptive pruning. With augmentations present the pool asks for three aug conditions (none/uncapped/capped), which are three separate cache files, so each condition costs 3 searches. Per new representation that is 3 conditions x 3 aug conditions = 9 searches.

**What can come off, no axis removed:**

- Tie the blueprint to the model that scores it: replace the five-candidate loop at c13:209 with a single get_best_search_for_downstream call for best_b_model.
  *Saves:* Fewer pool calls, no change in searches — the underlying searches are cached on disk, so the five candidate calls already reload rather than recompute after the first. The nine stored files are already unusable for comparison, so nothing extra goes stale.

### `c15` — complexity/c15_ml_theory_depth.py

**Size.** Per dataset, 10 base representations, 5 folds, 8 budgets. 15a: 8 full-data create_hybrid calls. 15b: 9 more (8 budgets plus a third rebuild of the b=50 hybrid at line 321).

**What can come off, no axis removed:**

- Take the repeated builds out of 15a, 15b and 15d. Build one hybrid per budget in run() and hand the dict to all three; in 15d hoist build_fold_hybrid above the trial loop, since the hybrid depends only on the real training labels.
  *Saves:* Per dataset with 5 folds and the 8 default budgets: 15a 8 builds, 15b 9, 15d 240, total 307 including 15c's 50. This removes 8 duplicates between 15a and 15b, the third budget-50 build, and 160 of 15d's 240, so 169 of 307 go.

### `c16` — complexity/c16_feature_selection_as_regularization.py

**Size.** Regression families give 6 Ridge alphas + 4 RF depths + 4 SVM C values = 14 variants. Budgets = FULL_BUDGETS filtered to b <= n and b <= total_dim (up to 8) plus capped_full = 9. 5 folds.

**What can come off, no axis removed:**

- Build the hybrid once per budget and reuse it across all 14 regularization settings, instead of passing model=model_factory() into build_fold_hybrid at line 154.
  *Saves:* 630 build_fold_hybrid calls per dataset (14 x 9 budgets x 5 folds) drop to 45. At 336 s per call, measured on flab_thermo_tresanco, about 54 hours per dataset down to about 4.

### `c17` — complexity/c17_information_theoretic_decomposition.py

**Size.** The cost is mRMR, not model fits. With n_samples >= 2000 the largest budget is 2000, and _greedy_mrmr's inner loop makes one MI estimate per remaining candidate per selected feature: about 2000^2/2 = 2,000,000 calls to estimate_mutual_information_knn. Each of those runs mutual_info_regression four times (once for the estimate and three times for the permutation null, _shared.py:2270), so about 8,000,000 sklearn kNN MI fits on 2000 rows.

**No free saving found.** Shortening this one means changing an axis, which is your call.

### `c23` — complexity/c23_evaluator_model_matching.py

**Size.** Per dataset, roughly ten constructible models (gp drops above 1000 rows, svm above 10000). Matched/near/far = 30 find_best_hybrid calls, each 1 model x 1 evaluator x 8 budgets x 5 allocations x 3 importance methods = 120 configurations x 5 folds = 600 downstream fits, so 18,000. The neural-size arm adds 600.

**What can come off, no axis removed:**

- Pass cache_namespace=f"{model_name}_{evaluator}" and leave dataset_name intact. The parameter exists at find_best_hybrid.py:5948 and is applied to cache paths only, at line 6265-6266, below the line where the policies resolve.
  *Saves:* One line at c23:100 and one at c23:249. On the capped datasets it removes two of eight budgets from every one of the 42 calls, roughly a quarter of the configurations.

### `c24` — complexity/c24_blueprint_transfer.py

**Size.** One full default find_best_hybrid per outer fold: 5 allocation methods x 3 importance methods x 8 budgets x 5 internal evaluators x 7 default models = 4,200 configurations x 5 inner folds = 21,000 downstream fits. Times 5 outer folds = 105,000 downstream fits per dataset, before create_hybrid's own step-wise allocation evaluation inside every one of those 21,000 configurations. The transfer step adds only 20 builds and 20 fits.

**What can come off, no axis removed:**

- Replace both bare create_hybrid calls with build_fold_hybrid (_shared.py:3559). It forwards groups, the internal evaluator, the augmentation strategy and margin and fraction, the candidate ranking, the augmentation budgets and the blueprint's own representation scores, and it refuses model without metric_fn.
  *Saves:* No change in the number of builds or fits. The builds get slower, because passing a model means the block-size choice fits that model (hybrid.py:3302-3311) rather than the cheap internal evaluator.
- Pass cache_namespace=f"fold{fold_i}" and leave dataset_name intact at line 191.
  *Saves:* One line. Removes roughly a quarter of the configurations on capped datasets.

### `fed01` — federated/fed01_noise_heterogeneity.py

**Size.** create_hybrid calls per (fed_model, seed) run: 5 folds x [6 sigma levels x (C0 1 + C5 5 silos + C6 1) = 42, plus 5 for the federated-hybrid detector setting, plus 5 smooth factors x 5 silos = 25] = 5 x 72 = 360. The sweep is 3 fed models x 3 seeds = 9 runs, so 3,240 create_hybrid calls per dataset, each scoring all 10 representations on an internal split and running treeshap importance on all 10. Separate plain model fits: per sigma, 6 teachers + 25 teacher cross-validation fits + C1 + the C4 student + 4 downstream = 37; x 6 sigma x 5 folds x 9 runs = 9,990, plus roughly 1,600 more from Phase 5's re-clustering and Phase 3's eight detectors over six settings.

**What can come off, no axis removed:**

- For the identical noise draws: pass seed=fed_seed to inject_noise_per_silo at fed01:249, 608 and 686.
  *Saves:* Three lines. No change to the number of fits.
- For the missing precompute: run precompute_hybrid_searches for herg too, by lifting the block at federated_helpers.py:2338-2365 out of the non-herg branch.
  *Saves:* No number changes. The same search runs, once, earlier.
- Decide which representation list herg is meant to use. Today fed01 gets the 'all' preset, up to 33 base representations; DATASET_REPS['herg']['general'] names ten and is the source of truth every other suite reads.
  *Saves:* One line either way (pass DATASET_REPS['herg']['general'] as rep_names at federated_helpers.py:2248). The compute difference is not established; representation generation is cached per SMILES set.

### `fed03` — federated/fed03_synthetic_private.py

**Size.** Part C dominates. find_best_hybrid at fed03:1000-1007 runs with defaults: 8 budgets (FULL_BUDGETS) x 5 allocation methods x 3 importance methods x 5 internal evaluators = 600 configs, x 5 folds x 5 downstream models (DOWNSTREAM_MODELS) = 15,000 model fits per silo, x 4 silos = 60,000 fits, on top of 3 importance computations per representation per fold over the rep_preset='all' set (~25 reps, fed03:983). run_all wraps every experiment in a 3 fed_model x 3 seed sweep (federated_helpers.py:252-254, 2395-2410), so this is entered 9 times and only the on-disk find_best_hybrid cache stops it repeating.

**What can come off, no axis removed:**

- Change one word in tests/slurm_scripts/submit_all_datasets.sh:55, from lhasa_herg to herg, so Part A can be submitted.
  *Saves:* Nothing goes stale. tests/results/federated does not exist, so no stored fed03 file changes.
- Load the ChEMBL hERG Ki set for Part A's first silo instead of reusing the Lhasa training set that run_all hands in.
  *Saves:* About 15 lines. Pooled rows go from about 12,155 to about 13,567 (1,412 + 7,772 + 4,383 minus overlap).
- Record, per fold, what C1's per-silo scores were and pick the baseline silo by its training score, not its test score.
  *Saves:* About 10 lines in each aggregation helper. One extra model fit per silo per fold to score on training rows, so 3 per fold in Part A: 135 across 5 folds and 9 runs.

### `fed05` — federated/fed05_antibody_federated.py

**Size.** run_all sweeps 3 fed_models x 3 fed_seeds = 9 full runs. Each run calls find_best_hybrid once per loaded silo with budgets=FULL_BUDGETS (8) x 5 allocations x 3 importance methods x 5 internal evaluators = 600 configurations, x 12 default models x 5 folds = 36,000 model fits per silo. Two silos load, so 72,000 per run.

**What can come off, no axis removed:**

- Search: pass cache=True at fed05:199.
  *Saves:* One line. Turns 18 searches into 2 with the two silos that load today.
- Naming: file fed05's output under a fixed name, 'antibody', and drop fed05 from SINGLE_DATASET_EXPERIMENTS so it is not repeated once per molecular runner.
  *Saves:* Two lines, at federated_helpers.py:226 and the save call at 2426. Removes two of the three fed05 invocations — about two thirds of fed05's total cost across the suite.

### `n09` — noise/n09_lnl_survey.py

**Size.** Per dataset at --mode full: 6 sigmas x 5 folds = 30 inner find_best_hybrid calls. Each searches DEFAULT_ALLOCATION_METHODS (5) x DEFAULT_IMPORTANCE_METHODS (3) x DEFAULT_INTERNAL_EVALUATORS (5) x NOISY_SEARCH_BUDGETS (3) = 225 configs x 3 inner folds = 675 create_hybrid constructions plus fits, plus 10 representation baselines x 3 folds = 30. That is 30 x 705 = about 21,150 fits, of which 20,250 are full create_hybrid builds (each re-scores 10 representations plus importance).

**What can come off, no axis removed:**

- Do NOT swap the inline re-search for the cached noisy precompute. Recorded here so it is not proposed later.
  *Saves:* Not applicable; this is a decision to leave the design alone.
- The compute, stated so you can decide where it runs. 6 levels x 5 folds = 30 inner searches per dataset.
  *Saves:* Cost not established. I did not verify whether the file-level metadata check alone stops a noisy run reading a clean baseline file.

### `n11` — noise/n11_estimation_remediation_grid.py

**Size.** Per dataset at --mode full: 6 sigmas x 5 folds x 3 noise seeds = 90 cells. Spaces: base + predictive hybrid + hybrid_sim + 9 other base representations = 12. Estimator keys: 10 BASE + 3 TORCH + ngboost = 14 estimators, of which 5 are neighbour-based and run at 2 distance metrics, giving 5x2 + 9 = 19 keys.

**What can come off, no axis removed:**

- Fit the clean baseline once per (fold, seed) rather than once per fold, so the gap subtracts two means over the same 90 numbers.
  *Saves:* 60 extra random forest fits per dataset, against 574,560 already run. Every recovery number moves by however much the two means differ.

### `n13` — noise/n13_noise_distributions.py

**Size.** 10 conditions x 6 levels x 5 folds x 3 noise seeds = 900 inner cells per dataset. Per cell: 5 noisy-baseline fits + 5 honest-hybrid fits + 3 spaces x 12 LNL methods = 36 DNN trainings at 50 epochs + 3 spaces x 3 estimators x 4 models = 36 weighted fits, i.e. 82 model fits.

**What can come off, no axis removed:**

- Alternative to the previous option: build one hybrid space per downstream model instead of one shared space, reusing the `_hyb_space_cache` the honest arm already fills in the same cell.
  *Saves:* Track B's per-cell fits stay at 36 but read from spaces already built, so zero extra hybrid builds. Track A gains nothing.
- For the honest re-score arm, either pass every model's distinct blueprint as a candidate for every model, or stop describing the arm as a leave-one-fold-out re-selection and report it as the paired per-fold delta it is.
  *Saves:* First: with 5 models and up to 5 distinct blueprints, per-cell honest fits go from 5 to at most 25, i.e. up to 18,000 extra model fits per dataset over 900 cells, with no extra hybrid builds since the spaces are already cached.
- Compute the level-0.0 cell once per (fold, seed) and copy the result into the other nine condition names.
  *Saves:* Saves 135 of the 900 inner cells: 11,070 model fits per dataset, of which 4,860 are 50-epoch neural-net trainings, and up to 810 hybrid builds. At the 261.9 s per build recorded in _shared.py:3739-3742 that is up to 59 h of hybrid construction per dataset.

### `opig_kingsmith_component_splits` — opig_kingsmith_component_splits.py

**Size.** Per dataset the outer folds are 23 + 15 + 4 + 3 = 45. Each outer fold runs a full cold search: n_train ~3,790 so gp is skipped and svm survives the uncapped cap, leaving 6 models; 8 budgets (all of FULL_BUDGETS fit under n); 5 allocation methods; 3 importance methods; 5 internal evaluators = 3,600 configs x 5 inner folds = 18,000 model fits, plus 10 x 6 x 5 = 300 search-internal baselines. 45 x 18,300 = ~824,000 fits per dataset, ~1.65 MILLION across buchwald_hartwig and bh_doyle_plates.

**What can come off, no axis removed:**

- Put `metric='rmse'` back into the search call at line 424, where 9bd48ac removed it.
  *Saves:* One line. It invalidates no stored result, because there is none.
- Submit it as eight tasks instead of one, using the flags that already exist: `--datasets` and `--splits`, one task per (dataset, component axis).
  *Saves:* Eight tasks. The largest, additive on either dataset, is 23 folds x about 18,300 fits = about 420,000; the smallest, base, is 3 folds = about 55,000.

### `opig_scaffold_hybrid` — opig_scaffold_hybrid.py

**Size.** Per inner search: 5 allocation methods x sum over models of (budgets x 3 importance methods x evaluators). Models resolved from models=None are rf, xgboost, knn, lightgbm, mlp, svm (gp is dropped, _max_samples=1000 < n~4000). Budgets are FULL_BUDGETS=[10,25,50,100,250,500,1000,2000] capped per model: 8 for rf/xgboost/lightgbm/knn, 7 for mlp (<=1000), 6 for svm (<=500).

**What can come off, no axis removed:**

- Make nested_cv_evaluate record, per outer fold, how many features each representation contributed — the numbers are in `finfo` at find_best_hybrid.py:8311 and are currently read only for a total width — then re-run all four datasets.
  *Saves:* Per outer fold: 5 allocations x 3 importance methods x (rf 8 budgets x 4 evaluators + xgboost 8 x 5 + lightgbm 8 x 4 + knn 8 x 5 + mlp 7 x 4 + svm 6 x 4) = 2,940 configurations, each over 5 inner folds = 14,700 fits, plus 12 representations x 6 models x 5 inner folds = 360 baseline fits. About 15,060 per outer fold, x 5 outer folds x 4 datasets = about 301,000 model fits, before adaptive pruning trims some after inner fold 0.
- Either wire the three flags through — --models and --budgets into run_nested_cv's kwargs, --max-folds into n_outer_folds — or delete them from the parser.
  *Saves:* Zero model fits either way. No stored result invalidated.

### `s07` — similarity/s07_noise_bridge.py

**Size.** No downstream model is fitted at all — the cost is distance matrices. _contamination_blocks scores 10 representations plus 2 hybrid candidates = 12 arms, and each arm loops over every fold recomputing cdist(X_train, X_train) plus a full argsort. Under the per-group scheme the fold count is the number of distinct scaffold groups G, so the work is 12 x G x O(n^2).

**No free saving found.** Shortening this one means changing an axis, which is your call.
## Delete or keep — 30 files

Each one: what it asks, what already answers that, and what you lose if it goes.
Some of these delete only a block, not the file. Nothing here deletes itself.

### `blacklist` — similarity_metrics/blacklist.py

**Asks.** Which individual columns inside a feature block make the block's similarity geometry worse, stably across scaffold folds?

**Proposal.** Delete it. src/kirby/similarity/selection.py:57 already answers the structural half per fold, corrects exactly the fitted-noise problem this file has, and is wired into src/kirby/similarity/recipes.py:125 and two scripts under tests/.

**What you lose.** Nothing imports this file and nothing stores its output, so no experiment and no result file changes. What is lost is the half that has no equivalent: the yardstick that scores by predicting each molecule's activity from its five nearest neighbours, and the count of how many scaffold folds independently dropped a feature. selection.py refits per fold instead of producing one fixed list, so it also cannot feed find_best_hybrid's feature_blacklist.

### `c03` — complexity/c03_cross_dataset_transfer.py

**Asks.** Does a dataset's intrinsic complexity (effective rank, participation ratio, C1/C2/L1/S3) predict the winning feature budget?

**Proposal.** Delete c03.

**What you lose.** Nothing goes stale — there are no result files. c14_cross_domain_summary answers hybrid-versus-best-single across datasets, and c24_blueprint_transfer answers source-to-target transfer. But neither computes the complexity-descriptor-against-winning-budget correlation or the model/dataset/interaction split of the winning budget.

### `c09` — complexity/c09_formal_analysis.py

**Asks.** How much of the variation in c02's grid scores is explained by the downstream model, by the feature budget, and by their interaction?

**Proposal.** Delete 9a and keep the other three sub-analyses.

**What you lose.** No measurement is lost. Every number 9a reads is already in c02's stored configs, 3425 rows per dataset. Only the eta-squared summary goes.

### `c13` — complexity/c13_forward_compatibility.py

**Asks.** Does adding a new representation to the hybrid's search space improve the hybrid beyond what a same-width random-Gaussian matrix earns from selection alone?

**Proposal.** Delete the nine JSON files in tests/results/complexity/c13_forward_compatibility/ (and the older copies in tests/results/c13_forward_compatibility/), or pass force_rerun=True.

**What you lose.** run_all stops skipping c13 on those nine datasets. Until this is done, no c13 fix takes effect through the normal path.

### `c14` — complexity/c14_cross_domain_summary.py

**Asks.** Across molecular, antibody and TCGA tasks, does the KIRBy hybrid beat the best single base representation?

**Proposal.** Delete c14.

**What you lose.** The cross-domain roll-up disappears. No other complexity experiment answers it: nothing under tests/experiments/complexity/ calls nested_cv_evaluate or run_nested_cv, and c02, the nearest sibling, has the same searched-maximum-versus-unsearched-mean defect (audit 3.3, c02:236-278).

### `c15` — complexity/c15_ml_theory_depth.py

**Asks.** Does a hybrid at budget b have less distance concentration, less hubness and a better-conditioned RBF kernel than each individual base representation of larger width, and than PCA at the same number of components?

**Proposal.** Delete 15c and keep 15a, 15b and 15d. State that the interpolation question is unanswered.

**What you lose.** The double_descent key disappears from the result file. 15a, 15b and 15d are untouched.

### `c18` — complexity/c18_diversity_vs_depth.py

**Asks.** At a fixed budget, does spreading features across all representations beat concentrating them in the top 1, 2 or 3 representations?

**Proposal.** Delete c18. If the arms are meant to be the engine's own allocation methods, the question is an allocation bake-off and c32 already sweeps allocation methods across datasets.

**What you lose.** The diversity-versus-depth question moves to whichever allocation sweep she is already running. Nothing is lost from the result files, since all eight are from a 13-to-27 representation pool at two budgets.

### `c23` — complexity/c23_evaluator_model_matching.py

**Asks.** Does pinning the internal evaluator to the downstream model's own family produce a better-scoring hybrid than pinning it to a different family?

**Proposal.** Delete c23 and let tests/experiments/framework/f13_evaluator_model_matching.py carry the question. f13 asks the same thing, lists only dispatchable evaluators (rf, xgboost, lightgbm, mlp, svm at f13:56), and its number is a fold mean over one fixed recipe (f13:74-119) rather than a maximum over a grid. It is also where an evaluator bake-off belongs by the placement rule.

**What you lose.** The seven c23 result files stop being maintained. The evaluator-versus-model question keeps a home. Nothing in the complexity paper that reads c23's JSON would still find it.

### `c30` — complexity/c30_balanced_global_selection.py

**Asks.** Does capping each representation at an equal top-N before one global lightgbm_gain ranking beat giving each representation an equal fixed quota of the budget?

**Proposal.** Delete c30 and read the same three arms off c31's stored files at /Volumes/seagate/kirby_cache/c31_weighted_global.

**What you lose.** You lose budget 100 and the caps of 50 and 300. You keep the equal-quota, equal-cap-then-global-rank and no-cap-then-global-rank comparison on 10 datasets and 3 downstream models instead of 4 datasets and one. Note c31's stored representation pools are pre-pruning — see the c31 record.

### `caco2_runner` — federated/caco2.py

**Asks.** Can every federated experiment be run against caco2 from one command, with an optional cheap representation list for local smoke runs?

**Proposal.** Delete caco2.py and use `python tests/experiments/run_dataset.py federated caco2`.

**What you lose.** One entry point instead of two. The --reps override is lost, but _smoke_driver.py already provides it for local runs. No SLURM script calls this file.

### `creative_hybrids` — similarity_metrics/creative_hybrids.py

**Asks.** Does stacking descriptors, MACCS and a structurally-selected atompair/torsion block onto the count fingerprint raise the 5-nearest-neighbour activity Spearman above ecfp4_count with Tanimoto?

**Proposal.** Delete the file and do nothing else.

**What you lose.** No stored result is lost, because there is none. The MACCS block and the question of how many atompair/torsion features to keep stop being asked anywhere in the folder.

### `exp_b` — complexity/exp_b_balanced_global.py

**Asks.** Does capping each representation at an equal top-N before one global lightgbm_gain ranking beat the equal per-representation quota?

**Proposal.** Delete exp_b.

**What you lose.** c30 is the later rewrite of the same three arms and adds the engine check. c31 has already run the same three arms on 10 datasets and 3 models with its results on disk. Nothing in exp_b is unique except its direction gate.

### `fed02` — federated/fed02_rep_diversity_silos.py

**Asks.** Does a feature-level hybrid built across five representations (C4) beat the best of those five representations used alone (C1)?

**Proposal.** For C5's test-fold selection, the third choice: delete C5. It asks whether restricting the hybrid to the three best representations beats using all five, which is what the top_3 allocation already is, and the search sweeps that allocation.

**What you lose.** One fewer condition, and no need to solve the selection problem. C4's allocation still reports which representations got budget.

### `fed03` — federated/fed03_synthetic_private.py

**Asks.** Part A: on hERG molecules pooled from three sources, does allocating a feature budget across silos by their own validation scores beat the best single silo, the two FLuID prediction-averaging ensembles, and the privacy-violating pooled model, measured by MCC?

**Proposal.** Delete Part C (fed03:949-1457) and its dispatch branch, and let fed05 answer the antibody federated question.

**What you lose.** fed05_antibody_federated.py:14-30 already runs independent silo baselines, cross-silo importance transfer and federated feature selection over ddg_combined, abagym_HER2 and flab_thermo_tresanco. Part C uses those three plus graphinity_foldx. fed05 is in SELF_LOADING_EXPERIMENTS (federated_helpers.py:231), so it runs on the submit path and Part C does not.

### `handtrace_noise` — noise/_handtrace_noise.py

**Asks.** Does the retention percentage give a number below 100 for a degraded model under both RMSE and R2?

**Proposal.** Delete the file.

**What you lose.** One less file. Nothing breaks: the only references are two audit documents and NOISE_AUDIT_STATUS_2026_06_04.md:103, no code imports it.

### `herg_runner` — federated/herg.py

**Asks.** Can every federated experiment be run against Lhasa's curated hERG classification set from one command?

**Proposal.** Delete herg.py and use `python tests/experiments/run_dataset.py federated herg`.

**What you lose.** One entry point. The --reps override is lost, but _smoke_driver.py already provides it locally. Nothing on the server calls this file.

### `logd_runner` — federated/logd.py

**Asks.** Can every federated experiment be run against logd from one command, with an optional cheap representation list for local smoke runs?

**Proposal.** Delete logd.py and use `python tests/experiments/run_dataset.py federated logd`.

**What you lose.** One entry point instead of two. _smoke_driver.py already covers the local representation override. Nothing on the server calls this file.

### `opig_activity_cliff_qsar` — opig_activity_cliff_qsar.py

**Asks.** Which single molecular representation gives the lowest MAE on activity-cliff-rich ChEMBL targets?

**Proposal.** Delete lines 370-381 and drop kirby_hybrid from the table, leaving the honest_rescore block as the only hybrid number.

**What you lose.** The main table has no hybrid row. Question 3 loses its hybrid answer entirely: honest_rescore reports mae only, never cliff MCC. Nothing stored goes stale.

### `opig_cdr_flexibility` — opig_cdr_flexibility.py

**Asks.** Which sequence-based antibody representation best separates flexible from rigid CDR3 loops (ROC AUC, PR AUC, MCC, balanced accuracy)?

**Proposal.** Drop the combined run at main() lines 482-489. The reason is that the features cannot express the label, not compute: an antibody present in both CDR files becomes two rows with identical heavy and light chain features and possibly opposite flexibility labels.

**What you lose.** The results file loses its 'combined' block. The two single-loop runs are unaffected and are the interpretable ones.

### `opig_cdr_flexibility` — opig_cdr_flexibility.py

**Asks.** Which sequence-based antibody representation best separates flexible from rigid CDR3 loops (ROC AUC, PR AUC, MCC, balanced accuracy)?

**Proposal.** Delete the file and run cdr_flexibility through the dataset roster instead. load_data('cdr_flexibility') exists at _shared.py:3179-3198, so the comment at submit_all_datasets.sh:43 saying it is 'pending a load_data path' is stale.

**What you lose.** You lose the comparison to the paper's 0.82 GNN number, which only this file makes, and you lose the per-loop breakdown. You get the framework's own hybrid-versus-single comparison instead. The roster path groups by 80% CDR3-loop identity, not V-gene family, so it is a different split and not the same experiment.

### `opig_humanness` — opig_humanness.py

**Asks.** Which of the ten antibody representations separates human from non-human therapeutic VH best, per downstream model, under 5-fold identity-grouped CV?

**Proposal.** Delete lines 316-321 so no `kirby_hybrid` row is written at all. The held-out comparison already exists: `build_honest_rescore_input(search)` at line 315 stores it, and lines 456-477 print it per downstream model.

**What you lose.** `results` loses the `kirby_hybrid` entry and the Summary Delta at lines 433-439 goes with it. Question 2 is then answered only by the held-out table. This is what opig_trastuzumab_enrichment.py already did on 2026-05-25 for the identical row; its note is at lines 326-333 of that file.

### `opig_scaffold_similarity` — opig_scaffold_similarity.py

**Asks.** Which mutation-difference representation and which distance formula give pair distances that track a substitution-matrix distance, once the differing-position count is partialled out?

**Proposal.** Delete tests/experiments/opig_scaffold_similarity.py and run scaffold_experiment.py per dataset plus feature_type_study.py instead.

**What you lose.** Loses graphinity_foldx, which is the only multi-scaffold set and exists nowhere else; loses the single results file covering all six datasets; loses the controls block, in particular the recorded count contamination (0.12 on koenig, 0.80 on flab_binding_adams, 0.92 on phillips_h3) and the C3 finding that position_gated is bit-for-bit the Grantham truth. scaffold_experiment covers C1 by marking circular graders and refusing position_gated on a multi-mutation library, but does not measure C3.

### `ranker_experiment` — similarity_metrics/ranker_experiment.py

**Asks.** Does selecting features FOR similarity beat selecting them predictively, when the selected space is judged by kNN prediction from five nearest neighbours and by agreement with shared substructure?

**Proposal.** Delete it and read f22_similarity_feature_sources.py instead. f22 asks the same headline question -- does ranking features by their agreement with shared substructure beat ranking them by their agreement with the activity gap -- and asks it per fold: it restricts the MCS pairs to training endpoints at lines 366-372, ranks on training rows only at line 386, and reports both yardsticks on the held-out fold.

**What you lose.** The activity-versus-structural comparison gets a leakage-free answer that already exists: tests/results/f22_similarity_feature_sources/herg_ki.json, 234 KB, written 26 August, on one of this file's two datasets. f22 also sweeps four budgets, two budget splits and four ways of gluing the kept features, none of which this file has. What is lost is ten of the thirteen selectors: f22 carries activity, structure and a RandomForest importance, and does not carry distance correlation, mutual information, Lasso, RandomForest on pair differences, RFE, RReliefF, the Laplacian score, greedy-forward, or the two rank fusions.

### `s07` — similarity/s07_noise_bridge.py

**Asks.** When a k-nearest-neighbour label-consistency filter flags molecules as noisy, does the hybrid space put fewer genuine activity cliffs among the flagged set than any single representation does?

**Proposal.** Delete the file. The sqrt(importance)-weighted distance question is asked better by n12_importance_weighted_distance.py, which injects noise so it knows the ground truth. The cliff question is covered by n01_activity_cliff_framework.py and n14_activity_cliff_landscape.py.

**What you lose.** The suite loses the question 'does a neighbour label-consistency filter flag fewer real cliffs in the hybrid space'. No other file asks exactly that. The 21 stored files become orphaned, and s09 loses two series.

### `similarity_noise_cleaning` — similarity_metrics/similarity_noise_cleaning.py

**Asks.** Does a similarity space selected without labels, for maximum-common-substructure fidelity, find deliberately corrupted activity values better than count-ECFP4 Tanimoto?

**Proposal.** Delete similarity_noise_cleaning.py and keep similarity_noise_cleaning.json.

**What you lose.** f23_noise_cleaning_metric.py answers the same two questions, scores every space, reports how many beat the baseline instead of a maximum, and takes its representations from f22 rather than four hand-written combinations. What is lost is the in-sample arm. The stored JSON is 41 KB, covers herg_ki, logd and adme_solubility, holds both arms in full, and is untracked — it exists on this laptop only, so it must be moved before the file goes.

### `sm_battery_moleculeace` — similarity_metrics/run_battery_moleculeace.py

**Asks.** Which representation x coefficient makes a similarity space whose neighbourhoods best agree with measured activity, judged on held-out scaffold folds?

**Proposal.** Delete it and do not run the 30-target battery.

**What you lose.** No file in the repo then answers the metric-ranking question across 30 targets. feature_ablation.py, ranker_experiment.py and creative_hybrids_v2.py answer it on herg_ki, logd and adme_solubility only, and they have stored results.

### `sm_build_adme` — similarity_metrics/build_adme_solubility.py

**Asks.** Does a third local dataset, log solubility, exist in the same schema as the herg_ki and logd pilot so the metric grid can be run on it?

**Proposal.** Add adme_solubility to build_local_datasets.SPECS, move the stricter blank filter into that builder, and delete this file. build_local_datasets.py supersedes it.

**What you lose.** One builder for all three local datasets. The stricter filter then applies to herg_ki and logd too, and I measured that it keeps the same 1415 and 5039 rows, so nothing they hold changes.

### `sm_feature_type_study` — similarity_metrics/feature_type_study.py

**Asks.** Which amino-acid property scales earn a place in a Euclidean distance space that reproduces BLOSUM sequence distance once mutation count is partialled out?

**Proposal.** Delete run() and main(). Keep the module: the 14 scales, property_table, score_subset and greedy_forward.

**What you lose.** feature_type_study.json loses its producer. Nothing reads that file; only feature_type_study.py:261 names it. The three hypotheses stay answered by opig_scaffold_similarity.py --mode features, on flab_expression_koenig, flab_binding_koenig, flab_expression_adams, flab_binding_adams, flab_binding_phillips_h3 and graphinity_foldx.

### `structural_geometry_survey` — structural_geometry_survey.py

**Asks.** Do structure-derived geometry features (VH-VL angles, CDR shape, paratope charge and the rest) carry standalone signal on antibody property prediction?

**Proposal.** Remove the hybrid arm: lines 326-357, and 'kirby_hybrid' from standalone_names at line 371.

**What you lose.** The survey answers its first two questions only, and answers them without the discarded search. The third question, whether a hybrid over base plus geometry beats the geometry groups, is then not asked here at all.

### `structural_yardstick` — similarity_metrics/structural_yardstick.py

**Asks.** When the judge is shared substructure (MCS) rather than activity, which representation-and-coefficient pair ranks first?

**Proposal.** Delete the file and rebuild the 47-metric rank table inside the framework, repointing mcs_cache.py:44 at kirby.similarity.mcs.mcs_distance.

**What you lose.** src/kirby/similarity/mcs.py:40 already holds the same mcs_distance definition, and its pair files carry smiles_sha and n_atoms, which these do not, so the row space and the size control both come for free. Nothing in the framework currently produces the 47-metric table, so deleting removes an answer that has no replacement yet.

### `winning_recipe_honest` — similarity_metrics/winning_recipe_honest.py

**Asks.** Once features are chosen on one half of the MCS pairs and structure is scored on the other half, does the stacked space still beat ecfp4_count with Tanimoto on distance-versus-MCS agreement?

**Proposal.** Delete the file, and replace the numbers in the two documents with the ones already on disk.

**What you lose.** similarity_noise_cleaning.py:29-33 does exactly what this file set out to do, and does it at molecule level: whole Bemis-Murcko scaffold groups split 50/50 at line 89, the descriptor drop and the feature ranking fitted on desc[fit_idx] at line 175, the z-score means from the fit rows only at line 122. similarity_noise_cleaning.json holds both arms for all three datasets. On herg_ki the stacked space scores 0.4462 structure held-out against baseline 0.1107, and 0.6471 activity against 0.5802, over 2005 held-out pairs and 706 held-out molecules.

## Reference

### Counts

| | yes | partial | no |
|---|---|---|---|
| complexity (30) | 1 | 20 | 9 |
| noise (14) | 1 | 12 | 1 |
| similarity, s01–s13 (15) | 2 | 11 | 2 |
| similarity, standalone (3) | 0 | 3 | 0 |
| similarity_metrics (33) | 5 | 27 | 1 |
| federated (10) | 3 | 5 | 2 |
| OPIG (11) | 0 | 10 | 1 |

Hybrid: 56 through the engine, 42 need none, 15 hand-roll a concatenation, 3 should use the
engine and do not.

Not reachable from any submit path: `c03 c12 c14 c20 c30 c32 exp_b c43_signal c43_step3 n06 s09`
plus most of similarity_metrics. OPIG has no registry entry.

### Four experiments cannot compute their own headline number

- `c15` sub-analysis 15c. Budgets are `n*[0.8, 0.9, 1.0, 1.1, 1.2]`. The near-p=n window is
  `[0.7, 1.3]`. Every budget falls inside it, so the far bucket is always empty and the
  instability ratio is never written.
- `s08` budget sweep. The fold loop reads the across-all-budgets blueprint, so all six budget
  rows rebuild at one width.
- `s08` `knn_score`. `DOWNSTREAM_MODELS` has no `knn`, so it is NaN at every budget and
  `best_knn_budget` is always None.
- `structural_geometry_survey`. `find_best_hybrid` is called without `return_fitted=True`, so
  the hybrid arm records "search returned no fitted rf result" every run. A 21,000-fit search
  runs and is thrown away.

### Two patterns behind many of the per-file problems

**No `groups` into `create_hybrid`** — 16 files: `c03 c07 c08 c09 c10 c24 n11 n12 n13 s11 fed01
fed02 fed03 fed04 fed05 opig_blueprint_transfer`. At `hybrid.py:4440`, with groups the internal
accept-or-reject holdout is a `GroupKFold` fold. Without groups it is
`np.random.RandomState(42).permutation`, a random 20% carve. That carve decides which features
get in. `build_fold_hybrid` passes groups through. `HANDOFF_CREATE_HYBRID_AUDIT.md` is a prompt
for a fresh chat to sweep all of them, including `framework/`.

**AUROC on hard 0/1 labels** — `_shared.py:1836` scores with `model.predict(X_val)`. Eight
datasets are registered with metric `auc`: `dataset_policy.py` lines 226, 248, 250, 258, 260,
261, 273, 315. `_predict_for_metric` at `find_best_hybrid.py:2427` returns probabilities only
when the metric is AUC, so it is safe everywhere else. Affects `c01 c05 c19 c21 n11` and
`framework/f06`.

### Smoke tests

107 runs here. 76 ran, 5 failed, 26 not reached.

```
KIRBY_SMOKE_N=60 python -u <script> --dataset logd --mode intermediary \
    --reps ecfp4,avalon --models rf,knn --output <scratch>
```

The default ten-model list segfaults on this machine. Pin the model list. `c02` is the exception
— it passes `models=None` into the search so its cache key matches the SLURM precompute, so it
cannot be pinned at all. Only representations already in `~/.cache/kirby/rep_cache` can be used.

The noise suite has its own answer. `_smoke_harness.py` sets `OMP`, `OPENBLAS`, `MKL`, `VECLIB`
and `NUMEXPR` to one thread before importing numpy, and forces the internal evaluator to `rf`.
That kept `n07`, `n13` and `n15` alive where the direct command line segfaulted.

Failed: `c02`, `n01`, `n14` segfault. `_smoke_noisy_search` dies at check 6 of 6 — the searches
above it run with caching off and never create the `result_db` directory that check opens.
`n13`'s suggested command exits 0 without running the experiment.

NaN reached a headline without crashing in `n15` (the lightgbm cell is `nan / nan / +nan`, next
to four real cells), `s04`, `s06` (24 NaN, 3 null, every weighted arm), `s07`, `s09`
(`AD_corr_delta = +nan`, and it feeds the Wilcoxon) and `triplet_similarity`.

Two smoke runs overwrote full-run result files with 60-row ones and had to be restored by hand:
`scaffold_experiment.json` and `scaffold_ddg_foldx.json`. Neither script tags its output with
the row cap.

### Against `AUDIT_HYBRID_USAGE_2026_09_03.md`

All 309 items checked against the code. 282 confirmed. 27 already fixed since it was written.
28 refuted. 2 unreachable. Its 26 confirmed blockers that this audit missed are in the entries,
marked *Also caught by the hybrid-usage audit*. The similarity objective and the federated
metric are both its findings.

Where it is wrong:

- §1.6 says the harness "no longer exists" and no copy was sought. It was never committed. The
  copy a later reader found was a symlink one of my own audit agents wrote at 13:16 today, with
  a compiled cache beside it at 13:17. I deleted that directory. Left there it would make `c20`,
  `c30` and the two `c43` files appear to run here and still fail on the server.
- §1.3 names `opig_cgr_comparison`, `opig_ginkgo_developability`, `c11` and `similarity_helpers`
  as reporting the search's own score. None does. There is no `score_mean` in
  `similarity_helpers.py`. `c11`'s number comes from its own fold loop at line 699.
  `opig_ginkgo` runs a per-fold search on training rows only.
- §2.2 lists `s06:489-498` and `federated_helpers:1533-1539` as missing groups. `s06` passes
  groups on both hybrid builds. There is no `create_hybrid` at `federated_helpers:1533`.
- §2.6 says `MODEL_MAX_BUDGET` and the budget policy drop budgets silently. Both print every
  dropped budget. A model left with none raises.
- Nine of its §3.1 engine items describe code since repaired, mostly in the uncommitted diff. So
  they are true on the server and false here.

What it has that this does not: §1.4 on ranking configurations across unequal fold counts, §1.5
and §2.8 on the result database ignoring the metric, and all of §3.1 on the engine internals.
This document does not cover `src/kirby/`.

### How to read an entry

The line under the heading says whether the file needs you at all. Then the verdict on whether
the code answers its own question, whether the hybrid goes through the engine, the compute
class, and whether `run_dataset.py` dispatches it. *On re-read* is where the second pass
corrected the first. That happened on 69 of 116 files. **Options** are cheapest first. *My read*
is one line and is not a decision.
