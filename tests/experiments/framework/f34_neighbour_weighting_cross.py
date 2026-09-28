#!/usr/bin/env python3
"""f34 — how much a neighbour counts, crossed with which coefficient decides
which molecules are near.

Two settings decide every nearest-neighbour number in this repo.

  SETTING A, how a neighbour's label is counted. Take a molecule, find its k
  nearest training molecules, average their activities. Either every neighbour
  counts the same, or a close one counts more than a far one.

  SETTING B, which coefficient decides what "near" means. manhattan, euclidean,
  cosine or a fractional Minkowski at p = 0.5.

Today each folder answers them differently, and nobody chose that. Counted from
the tree on 2026-09-27: similarity_metrics/ counts close neighbours more in 3
places, noise/ counts every neighbour the same in 3, similarity/ reads an
environment variable in s05 and s10 and lets s04 count every neighbour the same.

THE TWO HAVE NEVER BEEN CROSSED. s04:372, s05:416 and s10:586 fit sklearn on raw
features and pass no coefficient, so sklearn takes its own default of Minkowski
at p = 2, which is Euclidean, whatever KIRBY_DISTANCE_METRIC says. So every
result about setting A was measured with setting B held at Euclidean, and every
result about setting B was measured somewhere the neighbour weighting never
reached. This file crosses them.

WHAT IT IS NOT. f18 swept the distance coefficient at the six places the ENGINE
computes a distance, and found minkowski p=0.5 best on all three of its scores
where it is defined, cosine the best neighbour coefficient and the worst
structural one, and manhattan never last on two of the three. f34 must not redo
that. f18 covers where the engine CHOOSES a space's features; f34 covers the
kNN that SCORES a space once it is finished. Every recipe f34 reads was searched
under distance_metric='manhattan', so f34 varies the coefficient used to find
neighbours in a finished space, not the coefficient that built it.

WHAT IT MEASURES, per dataset, on scaffold-grouped held-out folds:

  6 weighting schemes x 4 coefficients x 6 neighbour counts = 144 cells, for
  each of 6 representations.

The neighbour counts are 5, 7, 10, 15, 20 and 25. s05 scores 5 and 10 when
KIRBY_S05_K_VALUES is unset (s05_knn_prediction.py:180); s10 scores 5, 10 and 20
when KIRBY_S10_K_VALUES is unset. Sweeping all six means f34's answer covers
every count either file already runs, and each is reported separately rather
than collapsed.

THE SIX REPRESENTATIONS, per dataset:
  - the three best base representations, ranked for a kNN on the stored search
    this file already opens. NOT read from a stored s02 result file: on
    2026-09-27 the two stored s02 results for caco2 named different
    representations from each other (tests/results_local_sim said pdv,
    ecfp4_count, ecfp4 and tests/results_local_s02fam said pdv, atom_pair,
    top_torsion), and both are local runs.
  - ecfp4 as a fixed reference, because it is the field default.
  - two hybrids at a budget of 100 features, one from the `general` lineage and
    one from `general_distance`, each built per fold from the recipe the pool
    ranked first on the OTHER folds. No search runs; every recipe comes out of
    the stored result database.

NO SEARCH RUNS AND NO AXIS IS CUT. The grid is complete: every weighting scheme
against every coefficient against every neighbour count. A cell that cannot be
computed is recorded in `skipped_cells` with its reason, never left blank.

Usage:
    python tests/experiments/framework/f34_neighbour_weighting_cross.py --check
    python tests/experiments/framework/f34_neighbour_weighting_cross.py --smoke
    python tests/experiments/framework/f34_neighbour_weighting_cross.py --datasets caco2
"""
import os

# The three datasets with a full local cache -- result_db, rep_cache and
# searches -- all live under this root. Confirmed present 2026-09-27.
os.environ.setdefault('KIRBY_CACHE_DIR', '/Volumes/seagate/kirby_simrun_cache')
os.environ.setdefault('KIRBY_LGBM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# ⛔ NO SEARCH MAY RUN. get_best_search_for_downstream and get_fold_pool_recipes
# read the stored result database through find_best_hybrid, which COMPUTES any
# configuration the grid asks for and the database does not hold. On 2026-09-28
# a run of this file without this line spent minutes searching and wrote 175 new
# rows into result_db/caco2.db before it was stopped. KIRBY_NO_COLD_SEARCH makes
# find_best_hybrid raise instead, and 1.0 means it raises on ANY gap rather than
# the 0.5 the bare flag gives. Lower it only to see how short a lineage is.
os.environ.setdefault('KIRBY_NO_COLD_SEARCH', '1.0')

import sys
import time
import argparse
import warnings
import traceback
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT / 'tests'))

from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.neighbors import (  # noqa: E402
    KNeighborsRegressor, KNeighborsClassifier)

from experiments._shared import (  # noqa: E402
    load_data, get_domain_metric, save_experiment_results,
    extract_top3_base_reps, get_best_search_for_downstream,
    get_fold_pool_recipes, build_fold_hybrid, fold_hybrid_scoring_kwargs,
)
from kirby.find_best_hybrid import nested_cv_outer_folds  # noqa: E402
from kirby.similarity import distance_config  # noqa: E402
from kirby.similarity import neighbour_weighting  # noqa: E402

# The three datasets with a full local cache.
DATASETS = ('herg_ki', 'caco2', 'adme_mdr1')

# The seven representations every stored recipe for these three was searched
# over. This is part of the result database's key, so asking for a different
# set misses the cache and starts a cold search.
LOCAL_REP_POOL = ('atom_pair', 'avalon', 'ecfp4', 'ecfp4_count', 'maccs',
                  'pdv', 'top_torsion')

# Setting B. The four canonical names, straight from the module that owns them,
# so a name added there is swept here without an edit.
COEFFICIENTS = distance_config.DISTANCE_METRICS

# Setting A. Same rule.
WEIGHTING_SCHEMES = neighbour_weighting.WEIGHTING_SCHEMES

# Neighbour counts. 5 and 10 are s05's own; 5, 10 and 20 are s10's; 7, 15 and 25
# come from the handoff. The union, so no count either file runs is missing.
K_VALUES = (5, 7, 10, 15, 20, 25)

# How many features each hybrid is built with.
BUDGET = 100

# The field default, carried as a fixed line every reader recognises.
REFERENCE_REP = 'ecfp4'

# The two hybrid lineages, and what each is called in the result file.
# distance=False reaches tier 'general', distance=True reaches
# 'general_distance'. Both hold rows at a budget of 100 features for all three
# datasets; read off result_db on 2026-09-27.
HYBRID_LINEAGES = (('hybrid_general', False), ('hybrid_general_distance', True))

# ⛔ ASK THE POOL FOR WHAT THIS FILE READS, NOT FOR THE WHOLE LINEAGE.
#
# get_fold_pool_recipes with distance=True sets internal_evaluators to
# DISTANCE_EVALUATORS, budgets to DISTANCE_BUDGETS and models to
# DISTANCE_MODELS on the loader underneath it. That is 10,800 configurations
# for caco2, and find_best_hybrid computes every one the database does not
# hold. f34 reads ONE downstream model at ONE budget, so it names them
# instead. All three are setdefault underneath, so what is passed here wins.
#
# The importance method, the allocation methods and the representation counts
# are named for the same reason, and this is where a choice was made. Counted
# from result_db on 2026-09-28, the stored rows at model='knn' and a budget of
# 100 features are RAGGED, identically on all three datasets:
#
#     performance_weighted  lightgbm_gain   whole pool        stored
#     performance_weighted  lightgbm_gain   3 representations stored
#     power_2               lightgbm_gain   whole pool        stored
#     power_2               lightgbm_gain   3 representations stored
#     performance_weighted  deeplift        whole pool        stored
#     performance_weighted  treeshap        whole pool        stored
#     -- every other pairing was never run --
#
# Asking for all three importance methods across both allocation methods and
# both representation counts asks for 12 and would set 6 searching.
# lightgbm_gain is the only importance method that ran under both allocation
# methods at both representation counts, so it is the only complete rectangle
# inside what exists. The pool therefore ranks 4 recipes per fold and this
# file takes the top one.
#
# ⛔ THAT IS A NARROWING OF WHICH RECIPE THE TWO HYBRIDS ARE BUILT FROM, and
# it is recorded in every result file as `pool_ask`. It does NOT narrow f34's
# own axes -- all 6 weighting schemes, 4 coefficients and 6 neighbour counts
# are scored on every representation. Widening it means running the missing
# configurations first, through tests/precompute_hybrids.py.
POOL_ASK = {
    'models': ['knn'],
    'budgets': [BUDGET],
    'importance_methods': ['lightgbm_gain'],
    'allocation_methods': ['performance_weighted', 'power_2'],
    'rep_counts': [None, 3],
}

# Which internal evaluator each lineage's stored rows carry. 'general' was run
# with rf and 'general_distance' with knn; asking for the wrong one finds
# nothing and starts a search.
POOL_EVALUATOR = {'hybrid_general': ['rf'], 'hybrid_general_distance': ['knn']}

# The partition the stored searches cross-validated over: 5 folds, seed 42,
# group-aware. This file's fold i and the pool's fold i must cover the same
# rows or the per-fold recipe read means nothing.
FOLD_SEED = 42
N_FOLDS = 5

# Both search families are eligible; the pool returns them in preference order
# and the family each recipe came from is recorded.
POOL_SUBSET = 'predictive_and_similarity'


# ---------------------------------------------------------------------------
def _knn(task_type):
    return (KNeighborsClassifier if task_type == 'classification'
            else KNeighborsRegressor)


def score_one_cell(X_tr, y_tr, X_va, y_va, *, k, scheme, coefficient,
                   task_type, metric_fn):
    """Fit one kNN under one (weighting scheme, coefficient, neighbour count).

    The estimator is built exactly the way s04, s05 and s10 will build theirs at
    step 2 of the handoff's plan -- sklearn's own `weights` and `metric`
    arguments -- so this is the reference the cached path below is tested
    against. f35 scores through here, not through the cache.
    """
    k_eff = min(k, len(y_tr) - 1)
    if k_eff < 1:
        raise ValueError(f'{len(y_tr)} training molecules cannot supply '
                         f'{k} neighbours')
    model = _knn(task_type)(
        n_neighbors=k_eff,
        **neighbour_weighting.sklearn_kwargs(scheme),
        **distance_config.sklearn_kwargs(coefficient))
    model.fit(X_tr, y_tr)
    return float(metric_fn(y_va, model.predict(X_va))), int(k_eff)


# ---------------------------------------------------------------------------
# ONE DISTANCE MATRIX PER (representation, coefficient, fold), NOT PER CELL.
#
# A KNeighbors estimator fitted on raw features recomputes every pairwise
# distance inside each predict, so fitting one per cell recomputes the same
# distances 36 times over: 6 weighting schemes x 6 neighbour counts, all of
# which want the same distances under the same coefficient. Measured on caco2
# atom_pair, 1726 training molecules by 2048 features and 432 held out:
#
#   coefficient       one fit+predict   one cdist   36 cells   36 cells from
#                       on raw features               direct    one cdist
#   manhattan                   0.97s       8.50s      34.9s         9.5s
#   euclidean                   0.11s       6.90s       3.8s         7.8s
#   cosine                      0.08s      15.77s       3.0s        16.9s
#   minkowski_p0.5             14.43s     175.78s     519.5s       176.7s
#
# 561 seconds per (representation, fold) the first way against 211 the second,
# so 16 hours against 6 for the three datasets. The fractional Minkowski is
# 84% of either, because sklearn and scipy both run it brute force -- it
# breaks the triangle inequality, which distance_config.is_true_metric records.
#
# ⛔ THIS IS THE `precomputed_matrix` PATH, the one similarity_metrics/
# simeval.py:161 already uses. It is NOT the raw-features path s04, s05 and
# s10 will adopt at step 2. Whether the two give the same answer is exactly
# what f35 measures, so f34's winner has to be read as the winner ON THIS
# PATH until f35 says the choice does not matter.
#
# A cheaper third way -- one neighbour SEARCH at the largest neighbour count,
# sliced for each smaller one -- was tried and dropped on 2026-09-28. caco2
# carries duplicated fingerprints, so molecules tie at exactly equal distance;
# when a tie straddles position k a 25-neighbour search and a 5-neighbour
# search keep different members of it. Measured on 40 held-out molecules with
# three identical training rows: 1 molecule differed under every
# distance-dependent scheme and 10 differed under rank.
# ---------------------------------------------------------------------------
def distance_matrices(X_tr, X_va, coefficient):
    """(train-by-train, held-out-by-train) distances under one coefficient.

    Computed with scipy under the name distance_config maps the coefficient
    to, so 'minkowski_p0.5' travels as minkowski with p=0.5 rather than
    collapsing to the Euclidean number scipy returns when nobody passes p.

    A non-finite distance is replaced by the largest finite one in the
    training block, which puts that pair as far apart as anything the fold
    holds rather than letting one NaN take out the whole row.
    """
    name, kw = distance_config.scipy_metric(coefficient)
    d_tr = cdist(X_tr, X_tr, metric=name, **kw)
    d_va = cdist(X_va, X_tr, metric=name, **kw)
    finite = d_tr[np.isfinite(d_tr)]
    fill = float(finite.max()) if finite.size else 1.0
    return (np.nan_to_num(d_tr, nan=fill, posinf=fill, neginf=0.0),
            np.nan_to_num(d_va, nan=fill, posinf=fill, neginf=0.0))


def score_precomputed_cell(d_tr, y_tr, d_va, y_va, *, k, scheme, task_type,
                           metric_fn):
    """One cell, off a distance matrix the caller already computed."""
    k_eff = min(k, len(y_tr) - 1)
    if k_eff < 1:
        raise ValueError(f'{len(y_tr)} training molecules cannot supply '
                         f'{k} neighbours')
    model = _knn(task_type)(n_neighbors=k_eff, metric='precomputed',
                            **neighbour_weighting.sklearn_kwargs(scheme))
    model.fit(d_tr, y_tr)
    return float(metric_fn(y_va, model.predict(d_va))), int(k_eff)


def _standardise(X_full, train_idx, val_idx):
    """z-score the features, fitting the scaler on this fold's TRAIN rows only.

    s05 and s10 both standardise before their kNN, so f34 does too -- a
    coefficient measured on unscaled fingerprints next to a 200-feature
    descriptor set is measuring the widths, not the coefficient.
    """
    sc = StandardScaler().fit(X_full[train_idx])
    tr = np.nan_to_num(sc.transform(X_full[train_idx]), nan=0.0,
                       posinf=0.0, neginf=0.0)
    va = np.nan_to_num(sc.transform(X_full[val_idx]), nan=0.0,
                       posinf=0.0, neginf=0.0)
    return tr, va


# ---------------------------------------------------------------------------
def read_pool(reps, labels, groups, *, dataset, data, metric_name, task_type,
              domain, n_folds):
    """Every recipe and representation name f34 needs, out of the stored pool.

    Returns (top3, fold_blueprints, notes). `fold_blueprints` is
    {lineage: {fold: blueprint}}. Nothing here runs a search.
    """
    notes = {'warnings': [], 'errors': {}}

    # -- the three best base representations, ranked for a kNN --------------
    top3 = []
    try:
        search, meta = get_best_search_for_downstream(
            reps, labels, groups,
            downstream_model='knn', budget=BUDGET,
            metric=metric_name, task_type=task_type, domain=domain,
            dataset_name=dataset, data=data,
            distance=True, pool_subset=POOL_SUBSET,
            internal_evaluators=POOL_EVALUATOR['hybrid_general_distance'],
            **POOL_ASK)
        if search is not None:
            top3 = [(n, float(s)) for n, s
                    in extract_top3_base_reps(search, model='knn')]
            notes['top3_from'] = {
                'objective': (meta or {}).get('objective'),
                'tier': (meta or {}).get('tier'),
                'metric_name': (meta or {}).get('metric_name'),
                'ranked_for_model': 'knn',
            }
        else:
            notes['errors']['top3'] = (
                'the pool returned no search for a kNN at a budget of '
                f'{BUDGET} features')
    except Exception as e:
        notes['errors']['top3'] = f'{type(e).__name__}: {e}'

    # -- one recipe per lineage per held-out fold ---------------------------
    fold_blueprints = {name: {} for name, _ in HYBRID_LINEAGES}
    fold_recipes = {name: {} for name, _ in HYBRID_LINEAGES}
    for lineage, distance_flag in HYBRID_LINEAGES:
        for fold_i in range(n_folds):
            try:
                pool = get_fold_pool_recipes(
                    reps, labels, groups,
                    fold_i=fold_i, n_outer_folds=n_folds,
                    downstream_model='knn', budget=BUDGET,
                    dataset_name=dataset, data=data,
                    metric=metric_name, task_type=task_type, domain=domain,
                    distance=distance_flag, pool_subset=POOL_SUBSET,
                    n_hybrids=1,
                    internal_evaluators=POOL_EVALUATOR[lineage],
                    **POOL_ASK)
            except Exception as e:
                notes['errors'][f'{lineage}/fold{fold_i}'] = \
                    f'{type(e).__name__}: {e}'
                continue
            for w in (pool.get('warnings') or []):
                notes['warnings'].append(f'{lineage} fold {fold_i}: {w}')
            if (pool.get('n_folds_in_pool') is not None
                    and int(pool['n_folds_in_pool']) != n_folds):
                notes['warnings'].append(
                    f"{lineage} fold {fold_i}: the pool holds "
                    f"{pool['n_folds_in_pool']} fold scores and this run splits "
                    f"into {n_folds}, so fold {fold_i} is not the same rows")
            hybrids = pool.get('hybrids') or []
            if not hybrids:
                notes['errors'][f'{lineage}/fold{fold_i}'] = (
                    f'the pool held no kNN recipe at a budget of {BUDGET} '
                    f'features on the other folds')
                continue
            top = hybrids[0]
            fold_blueprints[lineage][fold_i] = top['blueprint']
            fold_recipes[lineage][fold_i] = {
                'objective': top['objective'], 'tier': top['tier'],
                'aug_condition': top['aug_condition'],
                'allocation_method': top['allocation_method'],
                'importance_method': top['importance_method'],
                'internal_evaluator': top['internal_evaluator'],
                'budget': top['budget'],
                'metric_name': top['metric_name'],
                'score_other_folds': top['score_other_folds'],
                'n_other_folds': top['n_other_folds'],
            }
            print(f"    {lineage} fold {fold_i}: tier={top['tier']}, "
                  f"{top['objective']} family, alloc={top['allocation_method']}, "
                  f"imp={top['importance_method']}, "
                  f"ev={top['internal_evaluator']}, "
                  f"budget={top['budget']} features "
                  f"({top['metric_name']}={top['score_other_folds']:.4f} on "
                  f"{top['n_other_folds']} other folds)", flush=True)
    notes['fold_recipes'] = fold_recipes
    return top3, fold_blueprints, notes


# ---------------------------------------------------------------------------
def fold_matrices(space, reps, labels, groups, fold_splits, *, blueprints,
                  metric_fn, lower_is_better, task_type):
    """Yield (fold_i, X_train, X_val, y_train, y_val) for one representation.

    For a base representation the matrix is the representation itself, z-scored
    on this fold's training rows. For a hybrid it is built from this fold's own
    recipe, which the pool ranked on the other folds, and then z-scored the
    same way.
    """
    is_hybrid = space in dict(HYBRID_LINEAGES)
    for fold_i, (train_idx, val_idx) in enumerate(fold_splits):
        y_tr, y_va = labels[train_idx], labels[val_idx]
        if not is_hybrid:
            X_tr, X_va = _standardise(reps[space], train_idx, val_idx)
            yield fold_i, X_tr, X_va, y_tr, y_va, None
            continue
        blueprint = blueprints.get(space, {}).get(fold_i)
        if blueprint is None:
            yield (fold_i, None, None, y_tr, y_va,
                   'no recipe from the pool for this fold')
            continue
        try:
            X_tr_h, X_va_h, _feat = build_fold_hybrid(
                {n: r[train_idx] for n, r in reps.items()},
                {n: r[val_idx] for n, r in reps.items()},
                y_tr, blueprint,
                budget_override=BUDGET,
                groups=(groups[train_idx] if groups is not None else None),
                # A recipe searched under a geometry objective fits no
                # downstream model and build_fold_hybrid RAISES when a model
                # arrives with one. This returns an empty dict for those and
                # the arguments unchanged for a predictive recipe.
                **fold_hybrid_scoring_kwargs(
                    blueprint,
                    model=_knn(task_type)(n_neighbors=min(K_VALUES[0],
                                                          len(y_tr) - 1)),
                    metric_fn=metric_fn, lower_is_better=lower_is_better))
        except Exception as e:
            yield (fold_i, None, None, y_tr, y_va,
                   f'building the hybrid raised {type(e).__name__}: {e}')
            continue
        sc = StandardScaler().fit(X_tr_h)
        yield (fold_i,
               np.nan_to_num(sc.transform(X_tr_h), nan=0.0, posinf=0.0,
                             neginf=0.0),
               np.nan_to_num(sc.transform(X_va_h), nan=0.0, posinf=0.0,
                             neginf=0.0),
               y_tr, y_va, None)


def sweep_space(space, reps, labels, groups, fold_splits, *, blueprints,
                metric_fn, lower_is_better, task_type, skipped):
    """The full 144-cell grid for one representation.

    Returns {f'{scheme}|{coefficient}|k{k}': {...}}, one entry per cell, each
    holding the per-fold scores and their mean and spread.
    """
    cells = {}
    n_cells_per_fold = (len(WEIGHTING_SCHEMES) * len(COEFFICIENTS)
                        * len(K_VALUES))
    for (fold_i, X_tr, X_va, y_tr, y_va, why) in fold_matrices(
            space, reps, labels, groups, fold_splits, blueprints=blueprints,
            metric_fn=metric_fn, lower_is_better=lower_is_better,
            task_type=task_type):
        if X_tr is None:
            skipped.append({'representation': space, 'fold': fold_i,
                            'n_cells': n_cells_per_fold, 'reason': why})
            continue
        for coefficient in COEFFICIENTS:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    d_tr, d_va = distance_matrices(X_tr, X_va, coefficient)
            except Exception as exc:
                skipped.append({
                    'representation': space, 'fold': fold_i,
                    'coefficient': coefficient,
                    'n_cells': len(WEIGHTING_SCHEMES) * len(K_VALUES),
                    'reason': f'computing the distance matrix raised '
                              f'{type(exc).__name__}: {exc}'})
                continue
            for k in K_VALUES:
                for scheme in WEIGHTING_SCHEMES:
                    key = f'{scheme}|{coefficient}|k{k}'
                    try:
                        score, k_eff = score_precomputed_cell(
                            d_tr, y_tr, d_va, y_va, k=k, scheme=scheme,
                            task_type=task_type, metric_fn=metric_fn)
                    except Exception as exc:
                        skipped.append({
                            'representation': space, 'fold': fold_i,
                            'weighting_scheme': scheme,
                            'coefficient': coefficient, 'n_neighbours': k,
                            'n_cells': 1,
                            'reason': f'raised: {type(exc).__name__}: {exc}'})
                        continue
                    if not np.isfinite(score):
                        skipped.append({
                            'representation': space, 'fold': fold_i,
                            'weighting_scheme': scheme,
                            'coefficient': coefficient, 'n_neighbours': k,
                            'n_cells': 1,
                            'reason': 'the metric returned a non-finite score'})
                        continue
                    e = cells.setdefault(key, {
                        'weighting_scheme': scheme,
                        'coefficient': coefficient,
                        'n_neighbours': k, 'fold_scores': [],
                        'n_neighbours_used': []})
                    e['fold_scores'].append(score)
                    e['n_neighbours_used'].append(int(k_eff))
    for e in cells.values():
        arr = np.asarray(e['fold_scores'], dtype=np.float64)
        e['mean_of_folds'] = float(arr.mean())
        e['spread_of_folds'] = float(arr.max() - arr.min())
        e['std_of_folds'] = float(arr.std())
        e['n_folds'] = int(arr.size)
    return cells


# ---------------------------------------------------------------------------
def rank_cells(cells, lower_is_better):
    """Every cell ordered best-first, as a list of (key, entry)."""
    return sorted(cells.items(),
                  key=lambda kv: (kv[1]['mean_of_folds']
                                  * (1 if lower_is_better else -1)))


def summarise(per_space, lower_is_better, metric_name):
    """The three questions f34 exists to answer, for one dataset.

    1. Which cell wins, and by how much over second place, in this dataset's own
       metric, with the raw pair printed rather than a ratio.
    2. Is the winner the same under every representation?
    3. Do the two settings interact -- is the best coefficient the same under
       every weighting scheme?
    """
    out = {}

    # -- 1 and 2: the winner, per representation ----------------------------
    winners = {}
    for space, cells in per_space.items():
        if not cells:
            continue
        ranked = rank_cells(cells, lower_is_better)
        (k1, e1) = ranked[0]
        entry = {
            'cell': k1,
            'weighting_scheme': e1['weighting_scheme'],
            'coefficient': e1['coefficient'],
            'n_neighbours': e1['n_neighbours'],
            f'winner_{metric_name}': e1['mean_of_folds'],
            'winner_spread_over_folds': e1['spread_of_folds'],
            'n_cells_ranked': len(ranked),
        }
        if len(ranked) > 1:
            (k2, e2) = ranked[1]
            entry.update({
                'runner_up_cell': k2,
                f'runner_up_{metric_name}': e2['mean_of_folds'],
                'gap_to_runner_up': abs(e1['mean_of_folds']
                                        - e2['mean_of_folds']),
            })
        winners[space] = entry
    out['winner_per_representation'] = winners

    distinct = {w['cell'] for w in winners.values()}
    out['same_winner_across_representations'] = {
        'holds': len(distinct) == 1,
        'n_representations': len(winners),
        'distinct_winning_cells': sorted(distinct),
        'note': ('one cell wins on every representation'
                 if len(distinct) == 1 else
                 f'{len(distinct)} different cells win across '
                 f'{len(winners)} representations'),
    }

    # -- 3: do the two settings interact? -----------------------------------
    # For each weighting scheme, which coefficient is best when averaged over
    # the neighbour counts and the representations. If that coefficient is the
    # same under all six schemes the two axes are separable and could have been
    # swept one at a time. If it is not, they interact, which is the reason the
    # grid is crossed.
    sign = 1 if lower_is_better else -1
    best_coefficient_per_scheme = {}
    for scheme in WEIGHTING_SCHEMES:
        by_coefficient = {}
        for cells in per_space.values():
            for e in cells.values():
                if e['weighting_scheme'] != scheme:
                    continue
                by_coefficient.setdefault(e['coefficient'], []).append(
                    e['mean_of_folds'])
        if not by_coefficient:
            continue
        means = {c: float(np.mean(v)) for c, v in by_coefficient.items()}
        best = min(means, key=lambda c: means[c] * sign)
        best_coefficient_per_scheme[scheme] = {
            'best_coefficient': best,
            f'mean_{metric_name}_per_coefficient': means,
            'n_cells_averaged_per_coefficient': {
                c: len(v) for c, v in by_coefficient.items()},
        }
    chosen = {v['best_coefficient'] for v in best_coefficient_per_scheme.values()}
    out['axes_interact'] = {
        'best_coefficient_per_weighting_scheme': best_coefficient_per_scheme,
        'same_coefficient_under_every_scheme': len(chosen) == 1,
        'distinct_coefficients': sorted(chosen),
        'note': ('the best coefficient is the same under all six weighting '
                 'schemes, so the two settings could have been swept one at a '
                 'time' if len(chosen) == 1 else
                 f'{len(chosen)} different coefficients win depending on the '
                 f'weighting scheme, so the two settings interact and crossing '
                 f'them was necessary'),
    }

    # -- the same question the other way round ------------------------------
    best_scheme_per_coefficient = {}
    for coefficient in COEFFICIENTS:
        by_scheme = {}
        for cells in per_space.values():
            for e in cells.values():
                if e['coefficient'] != coefficient:
                    continue
                by_scheme.setdefault(e['weighting_scheme'], []).append(
                    e['mean_of_folds'])
        if not by_scheme:
            continue
        means = {s: float(np.mean(v)) for s, v in by_scheme.items()}
        best_scheme_per_coefficient[coefficient] = {
            'best_weighting_scheme': min(means, key=lambda s: means[s] * sign),
            f'mean_{metric_name}_per_scheme': means,
        }
    out['best_weighting_scheme_per_coefficient'] = best_scheme_per_coefficient

    # -- per neighbour count, never collapsed -------------------------------
    per_k = {}
    for k in K_VALUES:
        rows = [(key, e) for cells in per_space.values()
                for key, e in cells.items() if e['n_neighbours'] == k]
        if not rows:
            continue
        rows.sort(key=lambda kv: kv[1]['mean_of_folds'] * sign)
        per_k[str(k)] = {
            'best_cell': rows[0][0],
            f'best_{metric_name}': rows[0][1]['mean_of_folds'],
            'n_cells': len(rows),
        }
    out['best_cell_per_neighbour_count'] = per_k
    return out


# ---------------------------------------------------------------------------
def run_dataset(dataset, args):
    t0 = time.time()
    print(f'\n=== f34 {dataset} ===', flush=True)
    data = load_data(dataset, rep_names=list(LOCAL_REP_POOL))
    reps = data['reps']
    labels = np.asarray(data['labels']).ravel()
    groups = data.get('groups')
    groups = None if groups is None else np.asarray(groups)
    task_type = data['task_type']
    domain = data.get('domain', 'molecular')
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    missing = [r for r in LOCAL_REP_POOL if r not in reps]
    print(f'  {len(labels)} molecules, {len(reps)} representations, '
          f'metric {metric_name} ({"lower" if lower_is_better else "higher"} '
          f'is better), {task_type}')
    for name in sorted(reps):
        print(f'    {name}: {reps[name].shape[1]} features')
    if missing:
        print(f'  MISSING from the pool: {missing}. The stored recipes were '
              f'searched over all {len(LOCAL_REP_POOL)}, so the result '
              f'database key will not match and a cold search would start.')
        raise RuntimeError(f'{dataset}: representations missing: {missing}')

    fold_splits = nested_cv_outer_folds(reps, labels, groups, args.folds,
                                        FOLD_SEED, task_type)
    print(f'  {len(fold_splits)} folds, seed {FOLD_SEED}, group-aware — the '
          f'same partition the stored searches cross-validated over')

    if args.base_reps_only:
        # Every representation of the pool, named outright. No search is read,
        # so nothing here depends on how much of one the result database
        # holds. Seven representations rather than the three a ranking would
        # keep, because dropping four to save time would be a cut with no
        # reason behind it.
        top3, blueprints, notes = [], {}, {
            'warnings': ['--base-reps-only: the pool was not read, so there '
                         'are no hybrid representations and no ranking of the '
                         'base ones'], 'errors': {}}
        spaces = [n for n in LOCAL_REP_POOL if n in reps]
        print(f'  --base-reps-only: sweeping all {len(spaces)} representations '
              f'of the pool, no search read')
    else:
        print('  reading the pool (no search runs):', flush=True)
        top3, blueprints, notes = read_pool(
            reps, labels, groups, dataset=dataset, data=data,
            metric_name=metric_name, task_type=task_type, domain=domain,
            n_folds=len(fold_splits))
        if top3:
            print('    three best base representations, ranked for a kNN: '
                  + ', '.join(f'{n}={s:.4f}' for n, s in top3))
        else:
            print('    the pool gave no ranking of the base representations')

        base_names = [n for n, _ in top3 if n in reps]
        if REFERENCE_REP in reps and REFERENCE_REP not in base_names:
            base_names.append(REFERENCE_REP)
        spaces = list(base_names) + [n for n, _ in HYBRID_LINEAGES
                                     if blueprints.get(n)]
    print(f'  {len(spaces)} representations to sweep: {spaces}')

    skipped = []
    per_space = {}
    for space in spaces:
        ts = time.time()
        cells = sweep_space(space, reps, labels, groups, fold_splits,
                            blueprints=blueprints, metric_fn=metric_fn,
                            lower_is_better=lower_is_better,
                            task_type=task_type, skipped=skipped)
        per_space[space] = cells
        print(f'    [{space}] {len(cells)} cells scored in '
              f'{time.time() - ts:.1f}s', flush=True)

    results = {
        'experiment': 'f34_neighbour_weighting_cross',
        'dataset': dataset,
        'n_molecules': int(len(labels)),
        'representation_pool_asked_for': list(LOCAL_REP_POOL),
        'representation_widths': {n: int(r.shape[1])
                                  for n, r in sorted(reps.items())},
        'n_folds': len(fold_splits),
        'fold_seed': FOLD_SEED,
        'budget_features': BUDGET,
        'weighting_schemes': list(WEIGHTING_SCHEMES),
        'coefficients': list(COEFFICIENTS),
        'neighbour_counts': list(K_VALUES),
        'n_cells_per_representation': (len(WEIGHTING_SCHEMES)
                                       * len(COEFFICIENTS) * len(K_VALUES)),
        'representations_swept': spaces,
        'base_representations_only': bool(args.base_reps_only),
        'pool_ask': dict(POOL_ASK, internal_evaluators=POOL_EVALUATOR),
        'top3_base_representations': top3,
        'pool_notes': notes,
        'scoring_path': 'precomputed_matrix',
        'scoring_path_note': (
            'One distance matrix per (representation, coefficient, fold) with '
            "scipy cdist, then sklearn metric='precomputed'. This is the path "
            'similarity_metrics/simeval.py:161 uses, NOT the raw-features path '
            's04, s05 and s10 will adopt. f35 measures whether the two agree.'),
        'searched_under_distance_metric': 'manhattan',
        'what_this_does_not_answer': (
            'Every recipe read here was searched under '
            "distance_metric='manhattan', so this varies the coefficient used "
            'to find neighbours in a finished space, not the coefficient the '
            'engine used to choose that space. f18 covers the engine.'),
        'cells_by_representation': per_space,
        'skipped_cells': skipped,
        'n_skipped_cells': int(sum(s['n_cells'] for s in skipped)),
    }
    results['summary'] = summarise(per_space, lower_is_better, metric_name)
    results['elapsed_seconds'] = round(time.time() - t0, 1)

    print_dataset(results, metric_name, lower_is_better)
    save_experiment_results(results, args.output,
                            'f34_neighbour_weighting_cross',
                            dataset=dataset, metric_name=metric_name,
                            lower_is_better=lower_is_better)
    return results


def print_dataset(results, metric_name, lower_is_better):
    s = results['summary']
    direction = 'lower is better' if lower_is_better else 'higher is better'
    print(f'\n  [{results["dataset"]}] winner per representation. One row is '
          f'one representation; the score is the mean over '
          f'{results["n_folds"]} folds in {metric_name} ({direction}).')
    print(f'    {"representation":24s} {"weighting":15s} {"coefficient":16s} '
          f'{"k":>3s} {metric_name:>10s} {"2nd":>10s}')
    for space, w in s['winner_per_representation'].items():
        second = w.get(f'runner_up_{metric_name}')
        print(f'    {space:24s} {w["weighting_scheme"]:15s} '
              f'{w["coefficient"]:16s} {w["n_neighbours"]:>3d} '
              f'{w[f"winner_{metric_name}"]:>10.4f} '
              + (f'{second:>10.4f}' if second is not None else f'{"n/a":>10s}'))
    print(f'    {s["same_winner_across_representations"]["note"]}')
    print(f'\n  [{results["dataset"]}] do the two settings interact? '
          f'{s["axes_interact"]["note"]}')
    for scheme, block in s['axes_interact'][
            'best_coefficient_per_weighting_scheme'].items():
        means = block[f'mean_{metric_name}_per_coefficient']
        print(f'    {scheme:15s} best coefficient {block["best_coefficient"]:16s} '
              + '  '.join(f'{c}={v:.4f}' for c, v in sorted(means.items())))
    if results['n_skipped_cells']:
        reasons = {}
        for sk in results['skipped_cells']:
            reasons[sk['reason']] = reasons.get(sk['reason'], 0) + sk['n_cells']
        print(f'\n  {results["n_skipped_cells"]} cells skipped:')
        for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f'    {n} cells: {reason}')


# ---------------------------------------------------------------------------
def check(datasets, folds, base_reps_only=False):
    """Say what the cache holds for each dataset, without scoring anything.

    Prints the representations, the metric, and how many of the folds each
    hybrid lineage has a recipe for. A lineage with no recipe on any fold means
    the sweep would run on fewer than six representations, and that is the
    thing to see before the grid starts, not twenty minutes into it.
    """
    bad = []
    for ds in datasets:
        print(f'\n=== {ds} ===')
        try:
            data = load_data(ds, rep_names=list(LOCAL_REP_POOL))
        except Exception as e:
            print(f'  POOL MISSING — {type(e).__name__}: {e}')
            bad.append(ds)
            continue
        reps = data['reps']
        labels = np.asarray(data['labels']).ravel()
        groups = data.get('groups')
        groups = None if groups is None else np.asarray(groups)
        metric_name, _fn, lower = get_domain_metric(data)
        widths = ', '.join(f'{k}={v.shape[1]}' for k, v in sorted(reps.items()))
        print(f'  {len(labels)} molecules, {len(reps)} representations '
              f'({widths}), metric {metric_name}, '
              f'{"lower" if lower else "higher"} is better')
        missing = [r for r in LOCAL_REP_POOL if r not in reps]
        if missing:
            print(f'  MISSING: {missing}')
            bad.append(ds)
            continue
        fold_splits = nested_cv_outer_folds(reps, labels, groups, folds,
                                            FOLD_SEED, data['task_type'])
        if base_reps_only:
            print(f'  --base-reps-only: {len(reps)} representations to sweep, '
                  f'{len(fold_splits)} folds, no search read')
            continue
        top3, blueprints, notes = read_pool(
            reps, labels, groups, dataset=ds, data=data,
            metric_name=metric_name, task_type=data['task_type'],
            domain=data.get('domain', 'molecular'), n_folds=len(fold_splits))
        print(f'  three best base representations for a kNN: '
              f'{[n for n, _ in top3] or "none"}')
        for lineage, _flag in HYBRID_LINEAGES:
            got = len(blueprints.get(lineage, {}))
            mark = '' if got == len(fold_splits) else '   <- INCOMPLETE'
            print(f'  {lineage}: a recipe on {got} of {len(fold_splits)} '
                  f'folds{mark}')
            if got == 0:
                bad.append(ds)
        for w in notes['warnings']:
            print(f'  [warn] {w}')
        for key, err in notes['errors'].items():
            print(f'  [error] {key}: {err}')
    return sorted(set(bad))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--datasets', nargs='+', default=list(DATASETS))
    ap.add_argument('--folds', type=int, default=N_FOLDS)
    ap.add_argument('--check', action='store_true',
                    help='say what the cache holds, score nothing')
    ap.add_argument('--base-reps-only', action='store_true',
                    help='sweep every representation of the pool by name and '
                         'read no search at all. Use this while the result '
                         'database is too sparse to hand back a hybrid recipe; '
                         'the answer then covers the base representations '
                         'only, and the two hybrid lineages are missing.')
    ap.add_argument('--smoke', action='store_true',
                    help='caco2 only, one neighbour count — minutes, not a grid')
    ap.add_argument('--output',
                    default=os.environ.get(
                        'F34_OUT',
                        str(_ROOT / 'tests' / 'results' / 'framework')))
    args = ap.parse_args()

    from experiments import dataset_policy  # noqa: F401  (fills the registry)

    if args.check:
        bad = check(args.datasets, args.folds, args.base_reps_only)
        if bad:
            print(f'\nNOT READY: {", ".join(bad)}')
            return 2
        print('\nevery dataset has its representations and a recipe per fold '
              'for both lineages')
        return 0

    if args.smoke:
        global K_VALUES
        args.datasets = ['caco2']
        K_VALUES = (5,)
        print('smoke: caco2, one neighbour count, '
              f'{len(WEIGHTING_SCHEMES) * len(COEFFICIENTS)} cells per '
              f'representation')

    failed = []
    for ds in args.datasets:
        try:
            run_dataset(ds, args)
        except Exception:
            traceback.print_exc()
            failed.append(ds)
    if failed:
        print(f'\nFAILED: {failed}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
