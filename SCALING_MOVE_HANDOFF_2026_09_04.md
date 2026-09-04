# Handoff: move the z-scoring from the finished matrix to each representation

Written 2026-09-04, branch `similarity-metrics-study`, working tree at `862baf9`.
Two sessions were editing `hybrid.py` and `find_best_hybrid.py` that day and every
line number moved by several hundred lines in one afternoon. **Find things by
function name, not by line number.** The numbers below are only where they sat at
`862baf9`.

## What is wrong now

Z-scoring happens at the last possible moment, on the finished matrix, immediately
before a model or a distance. It is written out by hand in 324 places outside
`tests/experiments/framework/`: 25 in `src/kirby/` and 299 across `tests/`. There
is no shared function, so the 324 copies do not agree with each other.

Two consequences, both measured:

**Binary fingerprints are z-scored in one arm and not the other.** `_evaluate_single`
in `src/kirby/find_best_hybrid.py` skips `StandardScaler` when every value is 0 or 1
(the `is_binary = (X.dtype == np.uint8 or ...)` line), and that is the path every
single-representation baseline takes. `_evaluate_hybrid_config` always scales, and
so do the three scaling steps inside `_evaluate_baselines` and the outer-fold
scorer `_outer_fit_score`. `ecfp4`, `avalon`, `maccs`, `atom_pair`, `top_torsion`
and `rdkfp` are all 0/1 float32 and are in every molecular roster in
`DATASET_REPS_REFERENCE.md`. So on those datasets the baseline arm and the hybrid
arm are measured in different spaces.

Measured on 1000 rows: after z-scoring, one differing bit present in 0.2% of
molecules contributes 1001 units of squared distance; one present in half of them
contributes 4. `src/kirby/similarity/preprocessing.py` (`standardise`) already
carries the same measurement and says not to do it — 301 against 4, correlating
0.474 with plain Euclidean.

**A single check on the finished matrix cannot fix it.** A hybrid mixing `ecfp4`
with `mordred` is not all-0/1, so a whole-matrix check says "not binary" and
z-scores the bits anyway. That is the ordinary case.

## What to build

Scale each representation on its own, before anything is glued. The rule is one
line: if every value in the representation is 0 or 1, leave it alone; otherwise
z-score it per feature, with the mean and spread taken from training rows only.

Two functions do the gluing, both in `src/kirby/hybrid.py`:

- `concatenate_features` — builds the training-side matrix from `base_reps` +
  importance scores + an allocation.
- `apply_feature_selection` — rebuilds it on new rows from `base_reps` +
  `feature_info`.

Both take a dict keyed by representation and both know which representation every
feature came from. Neither knows which rows are training rows, so the training
mean and spread have to be passed in and returned, the way
`kirby.similarity.preprocessing.stack_blocks` already does with its `stats`
argument. Reuse `stack_blocks` if it fits; do not write a second copy of it.

Everything downstream then receives an already-scaled matrix and its own
`StandardScaler` is deleted.

## The traps

**Scaling twice cancels the first one.** A per-feature `StandardScaler` run over an
already-scaled matrix puts every feature back at variance 1. Measured on the
federated fix earlier the same day: a 2560-feature representation glued to a
200-feature one supplied 91.8% of one squared distance, 46.6% after correction,
and 91.8% again after a second `StandardScaler`. Every deleted scaler has to
actually be deleted, not left in place.

**Do not lose the missing-value fill.** `_impute_nan`
(`src/kirby/find_best_hybrid.py`) runs *after* scaling and takes its medians from
training rows. `StandardScaler` ignores NaN and passes it through, so the order
matters. Keep the fill after the scaling, on the glued matrix.

**Widths are still weights in a distance.** This refactor does not fix that, and
must not silently start fixing it. `block_divisor` /
`minkowski_block_divisor` in `src/kirby/similarity/preprocessing.py` are the width
correction and they are a separate decision — `stack_blocks(..., equalise=False)`
is the behaviour that matches today.

**Count fingerprints.** `create_ecfp4_count` and `create_atompair_count` exist in
`src/kirby/representations/molecular.py` and should not be z-scored either, but
neither appears in any roster in `DATASET_REPS_REFERENCE.md`. A value check will
not catch them. If they ever join a roster, the rule needs a per-representation
declaration rather than a value check.

**Trees do not care.** RandomForest, LightGBM, CatBoost and XGBoost split on a
threshold, and a per-feature shift and stretch moves the threshold with it. Every
scaler that only ever feeds a forest is already doing nothing, so those can be
deleted without changing a single number. The places where deleting one *does*
change a number are the ones feeding a GP, MLP, SVR, KNN, or any distance.

## Scope

Glue-function callers: 119, across 45 files.
Hand-written scaling to remove: 324 outside `framework/`.

Per directory:

| where | scaling sites |
|---|---|
| `src/kirby/hybrid.py` | 17 |
| `src/kirby/find_best_hybrid.py` | 7 |
| `src/kirby/screen.py` | 1 |
| `tests/experiments/complexity` | 57 |
| `tests/experiments/similarity` | 44 |
| `tests/experiments/noise` | 31 |
| `tests/experiments/annitia` | 24 |
| `tests/experiments/opig_*.py` | 14 |
| `tests/experiments/_shared.py` | 9 |
| `tests/experiments/federated` | 3 |
| `tests/experiments/pretrain_rep` | 2 |
| `tests/experiments/similarity_metrics` | 1 |
| other `tests/experiments/*.py` | 17 |
| `tests/*.py` | 74 |
| `tests/archive` | 22 |

**Do not touch `tests/experiments/framework/`** — 53 sites, and f00–f15 are frozen.

## Order to do it in

The engine first, because everything that runs through `find_best_hybrid` inherits
the fix without being edited. Ten scaling steps decide published numbers. Line
numbers are as of `862baf9` and will have moved — go by the function name.

In `src/kirby/find_best_hybrid.py`:

| function | what it scales | line at 862baf9 |
|---|---|---|
| `_evaluate_single` | one representation — already has the check | 3555 |
| `_evaluate_hybrid_config` | the hybrid | 4038 |
| `_evaluate_baselines` | geometry score of a base representation | 4517 |
| `_evaluate_baselines` | geometry score of a per-fold representation | 4627 |
| `_evaluate_baselines` | model score of a per-fold representation | 4663 |
| `_outer_fit_score` | the held-out fold | 7997 |

In `src/kirby/hybrid.py`:

| function | what it scales | line at 862baf9 |
|---|---|---|
| `blended_rep_scores` | each representation before a blend component | 341 |
| `_pairdist_vectors` | each representation before a pair distance | 902 |
| `_evaluate_allocation` | one allocation's features | 2142 |

And `compute_distance_matrix` in `src/kirby/utils/noise_mitigation_methods.py`
(line 116) — the distance every noise detector uses.

Then `tests/experiments/_shared.py`, which the suites share. Then the suites, one
at a time.

## How to check it

Before touching anything, record the current scores for one molecular dataset where
the best single representation is a fingerprint, on a scale-sensitive model (GP,
MLP, SVR or KNN) and on LightGBM. After the change: the LightGBM numbers must not
move, and the scale-sensitive numbers must move only for representations that hold
0/1 features.

Assert directly that a hybrid built from `ecfp4` + `mordred` comes out with the
`ecfp4` columns still holding only 0 and 1, and the `mordred` columns at mean 0.

Run `tests/experiments/noise/_smoke_n06.py` and the other `_smoke_*.py` files under
`tests/experiments/noise/` — they run in seconds and cover the noise paths.

## After it lands

Every cached result becomes invalid. Under the wipe-and-rerun rule that means
deleting `result_db/<ds>.db` and the matching JSONs for every molecular dataset,
then re-running. Antibody and genomic representations are continuous, so their
numbers should be unchanged — check that before wiping them.
