# Logic audit — what is still open, 2026-09-12

**What this is.** An AI audit run 2026-09-10 against commit `98eeac4`, re-checked against the
tree on 2026-09-12. Nothing here is a measured experimental result. Every item says what was
read. Items marked *measured* were produced by running a few lines in that session on synthetic
inputs, with the inputs stated — they are not runs of the pipeline.

42 findings. 6 are fixed. 36 are open.

---

## Fixed since 2026-09-10 — do not re-raise

1. **860 uncommitted engine lines.** `find_best_hybrid`, `hybrid`, `result_cache` and `_shared`
   are committed now. `git diff HEAD` shows 38 added / 4 removed in `hybrid.py` only.
2. **The per-metric budget-split origin rule is committed.** It was in the working tree only.
   `FRAMEWORK_FINALIZATION.md` §3.5 and the memory index both still say it is not built — the
   documents are stale, the code is right.
3. **c10** moved to `noise/` and passes `label_noise=spec`. It no longer rebuilds from a recipe
   searched on clean labels.
4. **opig ginkgo** now names 5 properties. `ginkgo_tm2`, `ginkgo_purity`, `ginkgo_aggregation`
   and `ginkgo_heparin` are gone.
5. **n09's clean-recipe fallback** now records `_recipe_source='clean_pool_fallback'` per fold.
   The fallback still happens; it is no longer silent.
6. **f11's allocation name** — fixed in this session. It asked for `min_floor_0.05`, which is not
   in `NAMED_ALLOCATIONS`, so the arm raised. `min_floor` defaults to floor 0.05, so the rename
   is the same allocation. All five names now dispatch.

---

## Open — the engine

### E1. The similarity objective and its budget split are hardcoded Euclidean
`src/kirby/utils/noise_mitigation_methods.py:357`, twin at `src/kirby/find_best_hybrid.py:1879`.
`activity_readacross` builds the Gram matrix and takes the square root. It has no metric in its
signature. `compute_distance_matrix` and `make_nearest_neighbors` both honour the process-wide
setting, whose default is manhattan. `hybrid.py:308` calls `activity_readacross` to split the
budget. So all four distance names give one identical correlation and one identical split.

### E2. Four feature-picking points, two without the positive-importance cut
`src/kirby/hybrid.py:2569` and `:3120` have no cut. `:2347` and `:5119` do. On caco2 with ecfp4,
201 of 2048 features score above zero, so a quota of 400 takes 199 features ordered by their
position in the matrix. The comment at `:2347` says "one of the two selection points"; there are
four. *Was three on 09-10 — a fourth appeared since.*

### E3. The hybrid arm and the single-representation arm differ in four ways
`src/kirby/find_best_hybrid.py:4043` against `:4549`. Width is the question. The other three: the
baseline path skips `StandardScaler` for an all-0/1 matrix and the hybrid path never does; the
baseline fold mean is rounded to 4 decimal places and the hybrid's is not; fold 0 cuts the
configuration list and is then one of the five folds in the hybrid's mean. ecfp4, avalon, maccs,
atom_pair, top_torsion and rdkfp are all 0/1.

### E4. A representation over 5000 features is ranked at 500 and reported at full width
`src/kirby/find_best_hybrid.py:4029`. Per fold it is cut to 500 features, 200 under 1000
molecules. Narrower representations are ranked at full width. `_outer_fit_score` applies no cut.
Exposed pools: `humanness_therapeutic` and both `tz_her2` sets (7,680 features),
`ginkgo_pr_cho` (17,920).

### E5. Three of eight importance methods pre-screen to 2000 features
`src/kirby/hybrid.py:4359`, `:4660`, `:4727`. Tree Shapley, boosted-tree gain and lightgbm gain
screen by spread first. Permutation, kernel Shapley, deeplift, integrated gradients and gradient
Shapley do not. On a 4800-feature representation, 2800 features get importance exactly 0.0 under
the first three and can never be selected at any budget.

### E6. The budget split reads area-under-curve off hard class labels
`src/kirby/hybrid.py:3037`. Exactly one place in the file passes `graded_predictions_for_clf=True`
(`:270`). Applies to the 8 area-under-curve datasets. Two representations whose rankings differ
but whose thresholded labels agree split the budget equally.

### E7. The redundancy penalty and the objective it protects disagree twice
`src/kirby/hybrid.py:1254`. The penalty is Euclidean over 5000 sampled molecule pairs. Neither
live objective samples pairs. The docstring at `:1227` says it is measured on the same pairs the
objective is scored on; the only consumer of that sampler is the scorer that was cut. Reaches f26.

### E8. `min_floor` repairs its rounding by a rule no other allocation uses
`src/kirby/hybrid.py:1751`. Every other named allocation hands leftover features out by largest
fractional part, one each. `min_floor` gives the whole shortfall to the best-weighted
representation up to its width. Not in the default five.

### E9. The per-representation baseline cache key omits the task type and the splitter identity
`src/kirby/find_best_hybrid.py:690`. The other three cache layers record the task type; the search
layer and the result database record the splitter. Two runs over the same rows that differ only in
declared task type write and read one file.

### E10. Augmentation matrix widths are in no cache key
`src/kirby/find_best_hybrid.py:956` records `subset_widths` = the widths of the *representations*.
`aug_hashes` is in `ignore_keys` at `:359`. No `aug_widths` field exists anywhere.
`EXPERIMENTS_CLOSED.md:109` records this Fixed and describes the augmentation side.

### E11. The block divisor is the square-root rule under a manhattan default
`src/kirby/similarity/recipes.py:106`. `preprocessing.py:14-18` says sqrt(width) flattens
Euclidean and not manhattan, and that manhattan needs division by the width itself.
`distance_config.block_divisor_for` exists and is called by nothing outside its own module and
`stack_blocks`. The same six lines z-score the 1024-feature counting fingerprint, which
`preprocessing.py:216` says not to do. Reaches the 11 `sim_*` recipes on 8 molecular datasets;
only f27 requests them.

### E12. Three task-type flags are discarded after being threaded in — latent
`src/kirby/hybrid.py:4101` (`augment_uncapped`; `augment_capped` keeps it), `:5916` and `:5988`
(`create_hybrid` fallbacks), `:2738` (`_evaluate_allocation`'s prediction branch). Underneath,
`_is_classification_task` rejects at 10 or more distinct labels and `_detect_task_type` accepts up
to 20. **Latent**: needs a dataset of 10 or more classes and I found none registered — the largest
is 5 (`tcga_brca_pam50`). Worth a guard, not a number that is wrong today.

---

## Open — the shared layer

### S1. The framework runs on a typed representation list, not the signed-off pool
`tests/experiments/_shared.py:3189`. Selected by `--framework`, which **13** submit scripts now
pass (was 10). Current overlap, recomputed 2026-09-12:

| dataset | typed | pool | in both |
|---|---|---|---|
| buchwald_hartwig | 17 | 10 | 4 |
| abagym_EGFR | 13 | 10 | 5 |
| flab_thermo_tresanco | 14 | 10 | 6 |
| tz_her2_binary | 13 | 10 | 6 |
| ginkgo_hic | 13 | 10 | 7 |
| tcga_brca_pam50 | 14 | 10 | 7 |
| logd | 12 | 10 | 7 |
| herg_ki | 11 | 10 | 8 |
| tcga_luad_vs_lusc | 14 | 10 | 9 |
| flab_pk_jain_fcrnrt | 13 | — | no pool |
| flab_pk_jain_heprt | 13 | — | no pool |

`cgr_morgan` is absent from the buchwald run. logd's typed list names `mol2vec`, `pdv`,
`chemberta` and `graph_kernel` — four of the five in that dataset's cut column in
`REP_PRUNING_LOG.md`. f26's `equal` arm divides one budget by the representation count.

### S2. The per-fold recipe lookup substitutes across models and budgets
`tests/experiments/_shared.py:1207` against `:5253`. The whole-dataset door refuses a recipe
belonging to a model the search never scored. The per-fold door makes the identical call with no
such check, then stamps the requested model and budget over what came back (`:1226-1232`). Five
similarity files use the unguarded door. A budget request that is absent returns the best over all
6 budgets, written under the requested one.

### S3. The resume fingerprint omits the model list and the budget list
`tests/experiments/_shared.py:5884`. Both are command-line settings that land in the same config.
A rerun with more models is skipped as already done. Shared by complexity, similarity and noise.

### S4. The linear-predictability descriptor is zero at 501 molecules or fewer
`tests/experiments/_shared.py:3049`. Least squares scored on the rows it was fitted on, input cut
to `min(500, n-1)` components. *Measured*: 4e-16 at 200, 400 and 501 molecules on labels drawn
independently of the features; 0.075 at 800; 0.115 at 1415. Feeds c03's descriptor correlations.

---

## Open — the suites

### Complexity
- **c19** — `feature_weighted` cannot move a score. Models are filtered to rf, lightgbm and
  xgboost, all axis-aligned. *Measured*: max prediction difference 0.0 on all three with
  all-positive weights. Where a weight is zero the feature is destroyed, not down-weighted
  (difference 0.697), so the treatment is a deletion of the zero-importance features — which is
  what the fifth treatment already does. `c19:249`.
- **c17** — the hybrid arm is handed the augmentations; the four arms it is compared against draw
  only from the base representations. The headline divides by the realised feature count, so both
  halves move. c18 passes `augmentations_train=None` for exactly this reason. `c17:406` vs `:267`.
- **c18** — the engine arm's recipe comes from one whole-dataset search; the four hand-built arms
  select inside the fold. Two differences, not one. `c18:188` vs `:235`.
- **c02** — the principal-component control repeats one score across budget labels. **141 model
  rows in 22 of 26 stored files.** `abagym_HER2` random forest reads 0.126880 at 250, 500, 1000
  and 2000. `ginkgo_heparin` repeats across five budgets for all six models. `c02:384`.
- **c01** — **513 of 567 stored information values are exactly 0.0** across 26 files. Files dated
  1–2 June; the estimator was replaced 3 June, and its own docstring says the earlier one made
  every such value zero. The current estimator returns 0.065 nats on comparable input.
- **c09 / c08** — the best-recipe loop covers 3 of the 5 models any search scores. It asks for
  `mlp` (tier `extra`, never scored) and never asks for `xgboost` or `dnn`. `c09:372`, `:578`,
  `c08:129`.

### Similarity
- **s08** — a distance correlated against a distance over a superset of its own features.
  *Measured* on structured noise: 0.575 at 10 features, 0.984 at 500. `s08:290`.
- **s06** — the "de-tautologised" neutral error model is fitted on every representation glued
  together, which contains both arms' features. That sentence is written into the results file.
  `s06:138`.
- **s12** — 2-molecule strata admitted, where a rank correlation is forced to ±1. `s12:352`.
  Same guard in `s04:264`.

### similarity_metrics
- **base_rep_comparison.json** — one-hot equals the CONSTANT control to all 17 digits on both
  koenig sets (−0.04430496411531469 and −0.047833718489563526). Still true today. No
  `degenerate_*` key in any stored row, though `EXPERIMENTS_CLOSED.md:289` says they go into the
  JSON. `scaffold_experiment.json` carries that same constant-control value on 5 of 15 rows and
  has no control arm.
- **aggregate.py** — 6 fusion rules reported, 4 distinct. Mean-rank and Borda are one rule under a
  positive shift; the duplicate-dropping rule drops nothing so it equals the plain average.
  Identical on all four measures on logd in the stored file. `aggregate.py:89`, `:98`.
- **simeval.py** — Dice and Tanimoto order every pair identically on a binary fingerprint, so 7 of
  47 battery metrics duplicate another 7. Soergel was removed from the same dictionary for exactly
  this reason. `simeval.py:131`.

### Noise
- **One sweep grid, two meanings.** `(0.0, 0.15, 0.3, 0.5, 0.75, 1.0)` for every dataset. On a
  regression set a level multiplies the label spread. On a two-class set it is handed in as
  `target_flip_rate`, so 1.0 inverts every training label. n09 takes its headline at the largest
  level; n11 requires recovery at every level above zero. `src/kirby/noise_spec.py:346`.

### Federated
- **fed03 Part C** — gated on two binding-change silos; `graphinity_foldx` appears zero times in
  `_shared.py` and `registry.py`. The pooled result key is never written, and the entry names that
  key as the answer. `fed03:1192`.

---

## Open — documents contradicted by the code

- `FRAMEWORK_FINALIZATION.md` §3.5 and the memory index say the budget-split rule is not built.
  It is built and now committed.
- `EXPERIMENTS_CLOSED.md:121` says `git diff HEAD` is empty for the five engine files. That was
  false on 09-10 and is true now — the entry was right by accident, not by evidence.
- `EXPERIMENTS_CLOSED.md:109` describes an augmentation-width cache fix; the code records
  representation widths (E10).
- `EXPERIMENTS_CLOSED.md:289` says the degenerate flags go into the JSON; no stored row has one.
- `EXPERIMENTS_FINALIZED.md` still carries **21** `NOT YET IMPLEMENTED` markers. At least c13's is
  wrong — the 108-name list is deleted and `forward_compat_reps.py` is committed.
- `EXPERIMENTS_FINALIZED.md:1499`, `:1503` name `summary.avg_mi_density`, `summary.best_strategy`
  and `downstream_r2` for c17. The code writes none of the three.
- Two entries stop mid-sentence with a warning whose reason never follows —
  `EXPERIMENTS_FINALIZED.md:4768` and `:5424`.

---

## Refuted — do not spend time on these

- A stored difference field being 0.0 on 11 datasets of 11. It is 0.0 on **3 of 23**; the other 20
  carry a real difference.
- The mutual-information estimator returning 0.0 on a linear label. It returns 0.065 nats.
- c21's uncapped noise decoy. Already a recorded decision in that entry.
- f15's splitter missing the task type. All 6 of its datasets are regression, so it does not
  diverge today.
- Greedy allocators, and the cut scorer, in any form.

---

## What was not covered

- The augmentation admission path — `augment_capped`, `augment_uncapped` — was read by one sweep
  and not re-read. `create_hybrid_cached` and `result_cache.py` beyond its columns were not read
  line by line.
- `similarity_metrics/` was the thinnest suite: 29 of 34 files, 4 in part, 9 of 15 result files
  checked field by field.
- Of `opig_*.py`, 10 of 11 were read in their comparison and summary sections only.
- In `framework/`, f06, f07, f14, f15, f16, f23 and f25 were not read line by line.
- Three engine claims were never verified and so are not listed above: two-phase refinement
  dropping six settings the first phase ran under; a caller-supplied estimator dictionary making
  every stored configuration both reload and refit; rows stored before the lost-fold refusal being
  ranked beside full-length rows.
- Nothing in the pipeline was executed. The four *measured* items are short synthetic checks.
