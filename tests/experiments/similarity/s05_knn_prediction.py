#!/usr/bin/env python3
"""S05: kNN Prediction with Hybrid Distance.

Paper: Similarity
Question: does kNN in the hybrid feature space beat kNN in each single rep's space?
Dependencies (2026-09-12): the pool. The distance-lineage searches must already
be on disk under BOTH objectives, predictive and similarity, or the first pool
read builds the missing family live inside this run:
    python tests/precompute_hybrids.py <dataset> --distance --with-similarity

CV evaluation with per-budget hybrid conditions: hybrid_unweighted, hybrid_weighted.
Per-rep kNN baselines (the real baselines). Budget sweep over DISTANCE_BUDGETS
to show how dimensionality affects kNN performance.

Which number to report. `posthoc_max` is the MAX over budgets x pool families x
pool ranks x conditions x k scored on the folds it was picked on — an upper
bound, never the headline. `fixed_ref` pins the pool family, the rank within it,
and the hybrid's budget, condition and neighbour count in advance.
Both subtract the same thing: the reference defined below, which since
2026-09-12 is one (representation, downstream model) pair per model — the best
single representation for that model, not one representation alone. Both
headline deltas subtract the kNN pair, and both carry a delta against every
other model's pair under 'delta_by_model'. The number to report is
`fold_reselection_result`'s delta: for each held-out fold it re-picks both arms on
the OTHER folds, and s09 stores it as knn_delta_heldout.

Until 2026-09-11 both subtracted `best_rep_score` instead — whichever of the ten
representations scored best, at whichever of the two neighbour counts scored
best, on the same five folds the hybrid was scored on. That is a maximum over
twenty choices measured on the numbers it is compared against, so the hybrid had
to beat a maximum rather than one alternative, and the comparison was stacked
against it by an unknown amount. The user replaced it with a representation
named before the run, then on 2026-09-12 with the (representation, model) pair.

Where the recipe comes from (2026-09-12 decision). s05 stops calling
get_fold_best_search. It reads the pool: for each fold, the best single
representation plus the top three hybrids for its downstream model ('knn'),
chosen from the OTHER folds' scores. Similarity hybrids are in its subset, so
the pool is asked for 'predictive_and_similarity' and returns three hybrids per
family — six hybrid arms per fold, each carried through the grid as its own arm.
Nothing is searched here any more: the pool reads searches that already exist.

2026-10-02, the owner's set: the three hybrids per family are now kNN's top 2
from the general searches plus its top 1 from the distance searches
(get_rule_hybrids), not the top 3 from the distance searches. Arms are named
'<family>_<slot>_<condition>_k<k>' with slot general1, general2 or distance1.
The single representations still come from the distance read, and the base
side adds the second-ranked one per fold ('pool_base_rep2'). Empty slots and
why: results['rule_missing'].

The two families store their scores in two different metrics (a predictive
search stores the dataset's own metric, a similarity search stores a Spearman
correlation between each held-out molecule's activity and what its five nearest
neighbours predict). Those two metrics decide only which three hybrids each
family offers. Every arm below is re-scored in this file by its own kNN
cross-validation in the dataset's own metric, so the arms are comparable to each
other and to the baselines in every number this file writes.

Before 2026-09-12 the recipe — allocation method, importance method, internal
evaluator — was searched inside each fold on that fold's training rows
(2026-09-04), and before that picked once by a search over every row, which made
it the best of 75 combinations measured on the same 5 folds reported here.

The importance-WEIGHTED condition is genuinely applied (sqrt-importance scaling
of the standardized features) but empirically does NOT beat the unweighted
hybrid downstream — that is a real result, not a wiring bug.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, get_fitted_hybrid, get_best_for_downstream,
    flatten_weights, flatten_fold_weights, make_cv_splitter,
    get_domain_metric, build_fold_hybrid,
    fold_hybrid_scoring_kwargs,
    fold_reselection, get_fold_pool_recipes, get_rule_hybrids,
    importance_feature_weights,
    DISTANCE_BUDGETS, DISTANCE_EVALUATORS, DISTANCE_MODELS,
    hr_exclude_folds,
)
from kirby.find_best_hybrid import nested_cv_outer_folds
from kirby.similarity.distance_config import (
    sklearn_kwargs as _sklearn_distance_kwargs)

# 2026-09-12 decision: s05 stops calling get_fold_best_search and reads the pool
# instead. The seed below now names the folds only. The pool's stored per-fold
# scores come from each search's own cross-validation, which is 5 folds at seed
# 42, group-aware — the same partition nested_cv_outer_folds builds here. If the
# two differ, the pool's fold i is not this file's fold i; get_fold_pool_recipes
# compares the fold counts and says so in its warnings, which are printed and
# persisted below.
_FOLD_SEARCH_SEED = 42

# Which search families the pool may serve. The decision: similarity hybrids are
# in s05's subset. 'predictive_and_similarity' returns N_HYBRIDS_PER_FAMILY
# hybrids from each family, so the grid carries six hybrid arms per fold.
POOL_SUBSET = 'predictive_and_similarity'
# THE OWNER'S SET (top 2 general + top 1 distance per model per family,
# 2026-10-02): the three hybrids per family are kNN's top 2 recipes from the
# general searches plus its top 1 from the distance searches, read with
# get_rule_hybrids. Until that date they were the top 3 from the distance
# searches only. An arm is (family, slot), slot 'general1', 'general2' or
# 'distance1'.
N_HYBRIDS_PER_FAMILY = 3
RULE_SLOTS = ('general1', 'general2', 'distance1')
# The family the pre-registered comparison (summary.fixed_ref) names. It is the
# family s05 read before this change. The pool does NOT rank the two families
# against each other — their stored scores are in two different metrics — so the
# fixed cell names one family in advance instead of taking a maximum over both.
# Every family's own fixed cell is written under summary.fixed_ref_by_family.
FIXED_REF_FAMILY = 'predictive'
# The slot the pre-registered cell names: the top recipe from the distance
# searches, which is what 'rank 1' meant before 2026-10-02 (this file read the
# distance searches only). Kept so the pre-registered comparison is the same
# recipe it always was; the general slots are scored beside it.
FIXED_SLOT = 'distance1'

from kirby.hybrid import apply_feature_selection, compute_feature_importance
from experiments.similarity._summary_keys import (
    SELECTION_RULE as _SELECTION_RULE)


# =============================================================================
# THE REFERENCE — one (representation, downstream model) pair per model
# =============================================================================
# 2026-09-12, owner's decision. What the hybrid is subtracted from is the best
# single representation for the dataset AND the downstream model — the pairing
# of the two, not the representation alone. So there is one reference per model
# in DISTANCE_MODELS, and every table below carries the model it belongs to.
#
# How each pair is scored. Nothing new is fitted. The per-fold scores are the
# stored search's own baseline scores for that representation under that model,
# which get_fold_pool_recipes returns ranked for whichever downstream model it
# is asked for. The representation is picked for each held-out fold on the
# OTHER folds' scores, so no fold picks the representation it is then scored
# on. Only a pool family whose scores are in this dataset's own metric is read,
# because every other number in this file is in that metric.
#
# The kNN pair is the one exception, and it is not a different recipe: where
# this file has already scored that representation at REFERENCE_N_NEIGHBOURS
# neighbours on the same fold, that score is used instead of the stored one, so
# the kNN reference and the hybrid cells come out of the same fits. Each fold
# records which of the two it used.
#
# The headline deltas (summary.posthoc_max, summary.fixed_ref, and s09's
# knn_delta_fixed_ref) subtract the kNN pair, because every hybrid cell in this
# file is a kNN score. Every other model's pair is in summary.reference_by_model
# with its own delta.
#
#   five neighbours. Five is the neighbour count every other neighbour
#   prediction in the similarity suite uses (s01, s02, s04, s06, s08 all
#   pass k=5).
#
# The held-out number (fold_reselection_result, s09's knn_delta_heldout) is
# unaffected by any of this — it re-picks both arms on the other folds and
# never reads the reference.
REFERENCE_N_NEIGHBOURS = 5

# dataset -> a base representation that OVERRIDES the pool's choice.
#
# Until 2026-09-12 this map WAS the reference, and a dataset absent from it got
# no reference comparison at all. It is now an override only: name a
# representation here and every model's reference is that representation rather
# than the best one for that model. Pass reference_rep='...' to run(), or
# --reference-rep on the command line, to override for one run.
#
# Provenance: the step-1 solo scores in REP_PRUNING_LOG.md — every base
# representation scored on its own with lightgbm under 3-fold scaffold-group
# cross-validation, the study the user cut each pool from. Only two datasets in
# that log name a winning representation. Both names below are in their
# dataset's pool of ten.
REFERENCE_REP_BY_DATASET = {
    'herg_ki': 'graph_kernel',   # top of its step-1 ranking, solo spearman +0.745
    'logd': 'mordred',           # top of its step-1 ranking, solo spearman +0.894
}


# =============================================================================
# EXPERIMENT
# =============================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """Downstream task validation: kNN with distance-aware hybrid selection
    across multiple budgets."""
    print("\n" + "=" * 70)
    print("EXPERIMENT S05: kNN PREDICTION WITH HYBRID DISTANCE")
    print("=" * 70)

    # The neighbour counts this file scores, one arm each. KIRBY_S05_K_VALUES
    # is a comma-separated list; unset leaves the 5 and 10 written here.
    _env_k = os.environ.get('KIRBY_S05_K_VALUES', '').strip()
    k_values = kwargs.get(
        'k_values',
        [int(v) for v in _env_k.split(',') if v.strip()] if _env_k else [5, 10])
    # sklearn's `weights` for every kNN fitted below. 'uniform' averages the k
    # neighbours' labels, which is what this file did; 'distance' averages them
    # weighted by one over the distance. KIRBY_S05_KNN_VOTE; unset gives
    # 'uniform'.
    knn_vote = kwargs.get(
        'knn_vote', os.environ.get('KIRBY_S05_KNN_VOTE', 'uniform').strip()
        or 'uniform')
    budgets = kwargs.get('budgets', DISTANCE_BUDGETS)
    n_folds = config['n_splits']

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    augmentations = data.get('augmentations')

    task_type = data['task_type']
    is_clf = task_type == 'classification'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    results = {'per_budget': {}, 'per_rep_baselines': {},
               'selection_rule': _SELECTION_RULE}

    # The three settings the kNN arms are built with, stored so a result file
    # says which arm produced its numbers without anyone reading the
    # environment the run had.
    from kirby.similarity.preprocessing import importance_exponent_override
    _exp = importance_exponent_override()
    results['knn_settings'] = {
        'k_values': list(k_values),
        'knn_vote': knn_vote,
        'importance_exponent': 0.5 if _exp is None else _exp,
        'basis': ("k_values are the neighbour counts scored, one arm each. "
                  "knn_vote is sklearn's weights argument: uniform averages "
                  "the k neighbours' labels, distance averages them weighted "
                  'by one over the distance. importance_exponent is the power '
                  'the importance is raised to before it multiplies the '
                  'z-scored features; 0 makes every feature weigh the same.'),
    }

    # ── The reference override, resolved before anything is scored ──
    # The reference itself is one (representation, model) pair per downstream
    # model, built after the baselines are scored. What is resolved here is the
    # optional override: a representation named up front that replaces the
    # pool's choice for every model. See the block at the top of this file.
    _passed_ref = kwargs.get('reference_rep')
    reference_override = _passed_ref or REFERENCE_REP_BY_DATASET.get(dataset_name)
    reference_source = None
    override_error = None
    if _passed_ref:
        reference_source = 'passed to this run as reference_rep'
    elif reference_override:
        reference_source = ('named for this dataset in REFERENCE_REP_BY_DATASET '
                            'at the top of this file')
    if reference_override is not None and reference_override not in reps:
        override_error = (
            f"the named reference representation '{reference_override}' is not "
            f"one of the {len(reps)} representations loaded for "
            f"'{dataset_name}'. Loaded: {', '.join(sorted(reps))}. The pool's "
            f"best representation per model is used instead.")
        reference_override = None
        reference_source = None
    if override_error:
        print(f"  [warn] {override_error}")
    if reference_override is not None:
        print(f"  Reference representation for every model, overriding the "
              f"pool: {reference_override} ({reference_source}).")
    else:
        print(f"  Reference: the best single representation for each "
              f"downstream model, chosen per fold on the other folds' scores.")

    # ── The pool, read once per (fold, budget) ──
    # 2026-09-12 decision: s05 stops calling get_fold_best_search. It reads the
    # pool — the best single representation plus the top three hybrids for its
    # downstream model ('knn'), chosen for each fold from the OTHER folds'
    # scores. Similarity hybrids are in its subset. No search runs here: the
    # pool reads the searches already on disk and ranks their stored per-fold
    # scores, so fold i's recipe was chosen without fold i's scores.
    print(f"  Reading the pool per fold "
          f"(subset={POOL_SUBSET}, {N_HYBRIDS_PER_FAMILY} hybrids per family, "
          f"evaluators={DISTANCE_EVALUATORS}, budgets={budgets})...")

    # ── Per-budget hybrid kNN evaluation ──
    sample_X = next(iter(reps.values()))
    # The folds come from nested_cv_outer_folds at 5 folds and seed 42, the same
    # partition each stored search cross-validated over, so this file's fold i
    # and the pool's fold i are the same rows. It is the same group-aware split
    # make_cv_splitter built before: GroupKFold, or StratifiedGroupKFold on a
    # classification dataset.
    folds_b = nested_cv_outer_folds(reps, labels, groups, n_folds,
                                    _FOLD_SEARCH_SEED, task_type)

    _pool_cache = {}
    _pool_warnings_seen = set()
    # True as soon as any pool member filtered its own candidate list on fold 0.
    # It decides which folds the per-fold reselection drops from BOTH arms, below.
    pool_fold0_pruned = None
    results['pool_read'] = {
        'pool_subset': POOL_SUBSET,
        'n_hybrids_per_family': N_HYBRIDS_PER_FAMILY,
        'downstream_model': 'knn',
        'selection_scope': None,
        'selection_note': None,
        'n_folds_in_pool': None,
        'objectives': None,
        'warnings': [],
        'errors': {},
    }

    def _fold_pool(fold_i, budget, model='knn'):
        """The pool for one held-out fold, or None when the read threw.

        Ranked on the other folds' scores. A budget names which stored
        configurations are eligible; it does not filter the base
        representations, whose per-fold scores carry no budget.

        `model` is 'knn' for every hybrid arm in this file — that is the only
        model anything here fits. The reference block reads it once per model
        in DISTANCE_MODELS, for the stored baseline scores of that model alone;
        only the kNN read feeds the run-level pool status and the fold-0
        pruning flag the per-fold reselection uses.
        """
        nonlocal pool_fold0_pruned
        ck = (fold_i, budget, model)
        if ck in _pool_cache:
            return _pool_cache[ck]
        try:
            pool = get_fold_pool_recipes(
                reps, labels, groups,
                fold_i=fold_i, n_outer_folds=len(folds_b),
                downstream_model=model, budget=budget,
                dataset_name=dataset_name, augmentations=augmentations,
                distance=True, data=data,
                metric=metric_name, task_type=task_type,
                domain=data.get('domain', 'molecular'),
                pool_subset=POOL_SUBSET,
                # This read now serves the single representations (and the
                # fold-0 pruning flag) only; the hybrid arms come from
                # _fold_rule below (owner's set, 2026-10-02).
                n_hybrids=1)
        except Exception as e:
            print(f"  [warn] pool read failed (fold {fold_i}, model={model}, "
                  f"budget={budget}): {type(e).__name__}: {e}")
            results['pool_read']['errors'][f"fold{fold_i}_{model}_b{budget}"] = (
                f"{type(e).__name__}: {e}")
            _pool_cache[ck] = None
            return None
        if model == 'knn':
            _p = pool.get('fold0_pruned')
            _p = True if _p is None else bool(_p)
            pool_fold0_pruned = (_p if pool_fold0_pruned is None
                                 else (pool_fold0_pruned or _p))
            results['pool_read']['selection_scope'] = pool.get('selection_scope')
            results['pool_read']['selection_note'] = pool.get('selection_note')
            results['pool_read']['n_folds_in_pool'] = pool.get('n_folds_in_pool')
            results['pool_read']['objectives'] = list(pool.get('objectives') or [])
        # When the asked-for budget matched no stored configuration, the pool
        # ranked over every budget it holds instead. Every row this pool feeds
        # is labelled with the budget that was asked for, so without this the
        # label and the hybrid can disagree. Record which (fold, budget) pairs
        # that happened on, and over which budgets the ranking actually ran.
        if pool.get('budget_relaxed'):
            results['pool_read'].setdefault('budget_relaxed', {})[
                f"fold{fold_i}_{model}_b{budget}"] = {
                    'asked_for': budget,
                    'ranked_over': list(pool.get('budgets_ranked_over') or []),
                }
        for w in (pool.get('warnings') or []):
            if w not in _pool_warnings_seen:
                _pool_warnings_seen.add(w)
                results['pool_read']['warnings'].append(w)
                print(f"  [pool fold {fold_i}, {model}] {w}")
        _pool_cache[ck] = pool
        return pool

    # The owner's set (top 2 general + top 1 distance per model per family,
    # 2026-10-02) for one held-out fold and budget, ranked on the other folds.
    # Every slot it could not fill is written to results['rule_missing'] with
    # the reason; before that date a missing family simply had no arm.
    _rule_cache = {}
    results['rule_missing'] = {}

    def _fold_rule(fold_i, budget):
        ck = (fold_i, budget)
        if ck in _rule_cache:
            return _rule_cache[ck]
        try:
            rule = get_rule_hybrids(
                reps, labels, groups,
                fold_i=fold_i, n_outer_folds=len(folds_b),
                downstream_model='knn', budget=budget,
                dataset_name=dataset_name, augmentations=augmentations,
                data=data, metric=metric_name, task_type=task_type,
                domain=data.get('domain', 'molecular'),
                pool_subset=POOL_SUBSET)
        except Exception as e:
            print(f"  [warn] rule read failed (fold {fold_i}, "
                  f"budget={budget}): {type(e).__name__}: {e}")
            results['pool_read']['errors'][f"rule_fold{fold_i}_b{budget}"] = (
                f"{type(e).__name__}: {e}")
            results['rule_missing'][f"fold{fold_i}_b{budget}"] = [
                f"the read raised {type(e).__name__}: {e}"]
            _rule_cache[ck] = None
            return None
        if rule.get('missing'):
            results['rule_missing'][f"fold{fold_i}_b{budget}"] = list(
                rule['missing'])
        for w in (rule.get('warnings') or []):
            if w not in _pool_warnings_seen:
                _pool_warnings_seen.add(w)
                results['pool_read']['warnings'].append(w)
                print(f"  [rule fold {fold_i}] {w}")
        _rule_cache[ck] = rule
        return rule

    # The budget under which the base representations are read. Any budget gives
    # the same answer — the budget filters hybrid configurations only — so the
    # first one is used and no extra pool read is paid for.
    _base_budget = budgets[0] if budgets else None

    # Fold 0's pool, kept only to report which base representations rank best
    # there. Named for the fold it came from: ranked on folds 1..4, not a
    # whole-dataset ranking.
    # The key name 'search_status' is kept because the per-fold reselection block at
    # the end of this file reads it; what it now records is the pool read.
    results['search_status'] = {'ok': True}
    pool0 = _fold_pool(0, _base_budget)
    if pool0 is None:
        results['search_status'] = {
            'ok': False,
            'error_type': 'PoolReadFailed',
            'error': (results['pool_read']['errors']
                      .get(f"fold0_knn_b{_base_budget}", 'pool read failed')),
        }
    else:
        results['top3_base_reps_first_fold'] = []
        results['pool_first_fold'] = {}
        for _obj, _fam in (pool0.get('by_objective') or {}).items():
            _ranked = (_fam.get('base_reps_ranked') or [])[:3]
            if _ranked:
                print(f"  Top-3 base reps for kNN, fold 0, {_obj} family "
                      f"(ranked on the other folds, "
                      f"{_fam.get('metric_name')}): "
                      + ', '.join(f"{b['rep']}={b['score_other_folds']:.4f}"
                                  for b in _ranked))
                results['top3_base_reps_first_fold'].extend(
                    [(b['rep'], b['score_other_folds']) for b in _ranked])
            # The arms' recipes on fold 0: the owner's set, not this read's.
            _rule0 = _fold_rule(0, _base_budget) or {}
            _hyb = (((_rule0.get('by_objective') or {}).get(_obj) or {})
                    .get('hybrids') or [])
            for _h in _hyb:
                print(f"  Fold 0 kNN hybrid, {_obj} family {_h['slot']}: "
                      f"alloc={_h['allocation_method']}, "
                      f"imp={_h['importance_method']}, "
                      f"ev={_h['internal_evaluator']}, "
                      f"budget={_h['budget']} "
                      f"({_h['metric_name']}={_h['score_other_folds']:.4f} "
                      f"on {_h['n_other_folds']} other folds)")
            results['pool_first_fold'][_obj] = {
                'metric_name': _fam.get('metric_name'),
                'lower_is_better': _fam.get('lower_is_better'),
                'n_hybrid_candidates': _fam.get('n_hybrid_candidates'),
                'n_base_reps': _fam.get('n_base_reps'),
                'base_reps_top3': [(b['rep'], b['score_other_folds'])
                                   for b in _ranked],
                'hybrids': [{'rank': h['rank'], 'slot': h['slot'],
                             'tier': h['tier'], 'name': h['name'],
                             'recipe': h['recipe'],
                             'score_other_folds': h['score_other_folds']}
                            for h in _hyb],
            }
        if not ((_fold_rule(0, _base_budget) or {}).get('hybrids') or []):
            results['search_status'] = {
                'ok': False,
                'error_type': 'NoConfig',
                'error': ('the pool held no kNN hybrid configuration for fold 0 '
                          f'under subset {POOL_SUBSET}'),
            }

    # The distance the kNN models here find neighbours under (2026-10-05).
    # Until this date they were built with no metric argument, so sklearn's
    # default, Minkowski p=2, which is Euclidean, while config['distance_metric']
    # in every result file said manhattan and s06 measures manhattan. These
    # models are not tuned (this file runs its own kNN cross-validation), so the
    # process-wide setting get_config() applies from --distance-metric is the
    # one they use now: manhattan unless that flag names another.
    _knn_distance_kw = _sklearn_distance_kwargs()
    results['knn_distance_metric'] = dict(_knn_distance_kw)

    def _make_knn(k_val, n_train):
        nn = min(k_val, n_train - 1)
        if is_clf:
            return KNeighborsClassifier(n_neighbors=nn, weights=knn_vote,
                                        **_knn_distance_kw)
        return KNeighborsRegressor(n_neighbors=nn, weights=knn_vote,
                                   **_knn_distance_kw)

    # Honest leave-one-fold-out re-score arms (persisted below). This
    # experiment runs its OWN kNN CV (not find_best_hybrid), so we collect the
    # per-fold scores of every hybrid cell (budget x condition x k) and every
    # single-rep baseline cell (rep x k) here, and assemble the standard
    # fold_reselection block by hand after the loops. Single downstream model by
    # design ('knn'), so both arms live under the 'knn' key.
    hybrid_candidates = []  # [{'name': str, 'fold_scores': [...]}]

    # F18: one entry per budget arm saying whether its hybrid kNN cells were
    # computed and, if not, why. Kept OUT of results['per_budget'] so the summary
    # loop below still iterates over scored cells only.
    results['per_budget_status'] = {}

    # One build per (budget, family, rank, k, fold): the unweighted and weighted
    # conditions read the same matrix, so building it twice cost a second
    # select-and-build pass for an identical result.
    _fold_build_cache = {}

    def _build(cache_key, hybrid, k, train_idx, val_idx, budget):
        """(X_train, X_val, feat_info, blueprint) for one pool hybrid on one fold.

        `hybrid` is one entry of the pool's 'hybrids' list. Its blueprint
        already carries the augmentation condition the search was loaded under
        (get_fold_pool_recipes sets augmentation_strategy on a deep copy).

        `budget` is the number of features the row is labelled with, and it is
        passed to build_fold_hybrid as budget_override so the matrix is built
        at it. Without that the matrix is built at blueprint.budget, and the
        two differ whenever the pool found no configuration scored at this
        budget and relaxed its filter to every budget the search holds — so a
        row labelled 10 features could hold a hybrid built at 500.
        """
        if cache_key in _fold_build_cache:
            return _fold_build_cache[cache_key]
        blueprint = hybrid.get('blueprint')
        aug_cond = hybrid.get('aug_condition')
        if blueprint is None:
            out = (None, None, None, None)
            _fold_build_cache[cache_key] = out
            return out
        y_train = labels[train_idx]
        train_reps = {name: rep[train_idx] for name, rep in reps.items()}
        val_reps = {name: rep[val_idx] for name, rep in reps.items()}
        aug_tr = ({k2: v[train_idx] for k2, v in augmentations.items()}
                  if augmentations and aug_cond != 'none' else None)
        aug_va = ({k2: v[val_idx] for k2, v in augmentations.items()}
                  if augmentations and aug_cond != 'none' else None)
        X_tr_h, X_va_h, feat_info = build_fold_hybrid(
            train_reps, val_reps, y_train, blueprint,
            augmentations_train=aug_tr,
            augmentations_val=aug_va,
            budget_override=budget,
            groups=(groups[train_idx] if groups is not None else None),
            **fold_hybrid_scoring_kwargs(
                blueprint, model=_make_knn(k, len(y_train)),
                metric_fn=metric_fn, lower_is_better=lower_is_better))
        out = (X_tr_h, X_va_h, feat_info, blueprint)
        _fold_build_cache[cache_key] = out
        return out

    # Each budget's cells, and then each representation's baseline, are written
    # to s05_knn_prediction/<dataset>.partial.jsonl when they finish and read
    # back on the next attempt (2026-09-25; _checkpoint.py). The stored unit
    # holds everything the loop body adds to the results and to the
    # per-fold-reselection lists (hybrid_candidates, baseline_candidates), so
    # the code after the loops sees the same objects. Nothing in these loops
    # draws from a shared random state. A unit in which any fold raised (a
    # build, a weighting, an importance fit) is not stored, so it is retried.
    from experiments._checkpoint import (
        Checkpoint, experiment_sidecar, run_identity)
    _ck = Checkpoint(
        experiment_sidecar(output_dir, 's05_knn_prediction', dataset_name),
        run_identity('s05_knn_prediction', dataset_name, data, config,
                     kwargs=kwargs, k_values=list(k_values),
                     knn_vote=knn_vote, budgets=list(budgets),
                     # 2026-10-02: arms are the owner's set now; a sidecar
                     # written under the old top-3 distance arms is not read.
                     hybrid_set='top2_general_top1_distance',
                     # 2026-10-05: the kNN models find neighbours under the
                     # recorded distance (manhattan by default), not
                     # Euclidean, so a sidecar written before then holds
                     # numbers from the other distance.
                     knn_distance=dict(_sklearn_distance_kwargs())))

    for budget in budgets:
        print(f"\n  Budget={budget}:")
        if _ck.done(budget=budget):
            _saved = _ck.get(budget=budget)
            results['per_budget'][budget] = _saved['budget_results']
            results['per_budget_status'][budget] = _saved['status']
            hybrid_candidates.extend(_saved['candidates'])
            continue
        _n_cand_before = len(hybrid_candidates)
        _threw = []

        budget_results = {}
        # F18: cells whose every fold threw, so the reason for a missing
        # condition x k score travels with the arm.
        failed_cells = []

        # 2026-09-12: the arms at this budget are the pool's top three hybrids
        # per family, read for each fold from the other folds' scores. An arm is
        # (family, rank): fold i contributes ITS rank-r hybrid in that family,
        # which can be a different recipe from fold j's. Every arm is kept and
        # scored — nothing is merged across families and nothing is dropped.
        # 2026-10-02: an arm is (family, slot) of the owner's set -- general1,
        # general2, distance1 -- read per fold from _fold_rule. `rank` below
        # holds the slot name.
        arms = {}            # (family, slot) -> {fold_i: pool hybrid entry}
        for fold_i, (train_idx, val_idx) in enumerate(folds_b):
            pool = _fold_rule(fold_i, budget)
            if pool is None:
                continue
            for h in (pool.get('hybrids') or []):
                arms.setdefault((h['objective'], h['slot']), {})[fold_i] = h
        if not arms:
            print("    no pool hybrid for kNN at this budget")

        _slot_order = {sl: i for i, sl in enumerate(RULE_SLOTS)}
        for (family, rank) in sorted(
                arms, key=lambda a: (a[0], _slot_order.get(a[1], 99))):
            fold_hybrids = arms[(family, rank)]
            # Which recipe each fold's rank-r hybrid is, so the file records
            # that they can differ instead of naming one evaluator for all five.
            fold_recipes = {}
            for condition in ['hybrid_unweighted', 'hybrid_weighted']:
                for k in k_values:
                    # Fold-index-aligned: None where this fold produced no
                    # score, so a missing middle fold cannot shift the paired
                    # comparison the per-fold reselection makes below.
                    fold_scores = [None] * len(folds_b)

                    for fold_i, (train_idx, val_idx) in enumerate(folds_b):
                        hybrid = fold_hybrids.get(fold_i)
                        if hybrid is None:
                            print(f"  [warn] fold {fold_i}: the pool held no "
                                  f"{family} hybrid in slot {rank}, "
                                  f"budget={budget}")
                            continue
                        y_train, y_val = labels[train_idx], labels[val_idx]

                        try:
                            X_tr_h, X_va_h, feat_info, blueprint = _build(
                                (budget, family, rank, k, fold_i),
                                hybrid, k, train_idx, val_idx, budget)
                        except Exception as e:
                            print(f"  [warn] fold {fold_i} failed: "
                                  f"{type(e).__name__}: {str(e)[:80]}")
                            _threw.append(fold_i)
                            continue
                        if blueprint is None:
                            print(f"  [warn] fold {fold_i}: the pool's "
                                  f"{family} {rank} entry at "
                                  f"budget={budget} carries no blueprint")
                            continue
                        fold_recipes[fold_i] = {
                            # The number of features the matrix was actually
                            # built with, beside the budget it was asked for
                            # (2026-10-05, owner's decision). A recipe whose
                            # representations hold fewer usable features than
                            # the budget builds a narrower matrix.
                            'width_built': int(np.asarray(X_tr_h).shape[1]),
                            'allocation': blueprint.allocation_method,
                            'importance': blueprint.importance_method,
                            'evaluator': blueprint.internal_evaluator,
                            'augmentation_strategy': blueprint.augmentation_strategy,
                            'recipe': hybrid.get('recipe'),
                            'pool_name': hybrid.get('name'),
                            # The number of features the pool's own row was
                            # scored at. It differs from `budget` only when the
                            # pool relaxed its budget filter; the matrix is
                            # built at `budget` either way (budget_override).
                            'pool_budget': hybrid.get('budget'),
                            'pool_budget_relaxed': (
                                hybrid.get('budget') is not None
                                and int(hybrid.get('budget')) != int(budget)),
                            # The pool's own score of this recipe on the other
                            # folds, in the family's own metric. It is NOT the
                            # dataset's metric for the similarity family, and it
                            # is never compared with the numbers below.
                            'pool_metric': hybrid.get('metric_name'),
                            'pool_score_other_folds': hybrid.get(
                                'score_other_folds'),
                            'pool_n_other_folds': hybrid.get('n_other_folds'),
                        }

                        scaler = StandardScaler()
                        X_tr = scaler.fit_transform(X_tr_h)
                        X_va = scaler.transform(X_va_h)

                        # Importance weighting for weighted condition.
                        # Use flatten_fold_weights to handle augmentation tail
                        # correctly (manual concat dropped aug columns and
                        # shape-failed silently under bare except).
                        if condition == 'hybrid_weighted':
                            try:
                                w_raw = flatten_fold_weights(feat_info)
                                w = importance_feature_weights(w_raw)
                                assert w.shape[0] == X_tr.shape[1], (
                                    f"weight/feature mismatch: "
                                    f"w={w.shape[0]} vs X={X_tr.shape[1]}")
                                X_tr = X_tr * w
                                X_va = X_va * w
                            except Exception as e:
                                print(f"  [warn] fold {fold_i} failed: "
                                      f"{type(e).__name__}: {str(e)[:80]}")
                                _threw.append(fold_i)
                                continue

                        model = _make_knn(k, len(y_train))
                        model.fit(X_tr, y_train)
                        y_pred = model.predict(X_va)
                        fold_scores[fold_i] = float(metric_fn(y_val, y_pred))

                    scored = [s for s in fold_scores if s is not None]
                    if not scored:
                        failed_cells.append(
                            f"{family}_{rank}_{condition}_k{k}")
                        continue

                    score_mean = float(np.mean(scored))
                    score_std = float(np.std(scored))
                    # Per-fold reselection arm: keep the raw per-fold scores for this
                    # (budget, family, rank, condition, k) cell so held-out-fold
                    # selection can re-pick the winning cell per fold (not the
                    # same-fold argmax).
                    hybrid_candidates.append({
                        'name': f"b{budget}_{family}_{rank}_{condition}_k{k}",
                        'fold_scores': list(fold_scores),
                    })
                    # The key keeps the condition name in it because s09 reads
                    # every per_budget cell whose key contains
                    # 'hybrid_weighted'. It now also names the family and the
                    # pool rank the arm came from.
                    key = f"{family}_{rank}_{condition}_k{k}"
                    budget_results[key] = {
                        'score_mean': score_mean,
                        'score_std': score_std,
                        'metric': metric_name,
                        'k': k,
                        'condition': condition,
                        'budget': budget,
                        # Which pool family and which rank within it.
                        'pool_family': family,
                        # The slot of the owner's set: general1, general2 or
                        # distance1 (2026-10-02; was a rank 1-3).
                        'pool_rank': rank,
                        'pool_slot': rank,
                        'pool_tier': next(
                            (h.get('tier') for h in fold_hybrids.values()),
                            None),
                        'n_folds_scored': len(scored),
                        # One recipe per fold, not one evaluator for the cell.
                        'fold_recipes': {int(fi): dict(r)
                                         for fi, r in sorted(fold_recipes.items())},
                        # Which folds the pool had no configuration scored at
                        # this budget for, so it ranked over every budget the
                        # search holds instead. Those folds' recipes were
                        # chosen at another budget even though the matrix is
                        # built at this one.
                        'pool_budget_relaxed_folds': [
                            int(fi) for fi, r in sorted(fold_recipes.items())
                            if r.get('pool_budget_relaxed')],
                        'budget_override_applied': True,
                    }
                    print(f"    {family} {rank} {condition} k={k}: "
                          f"{metric_name}={score_mean:.4f}+/-{score_std:.3f}"
                          f" ({len(scored)} folds)")

        # Which family of stored search gave no hybrid at all at this budget,
        # and why (2026-10-05). The subset asks for two families, predictive
        # and similarity. A family the pool had nothing for got no arm, and
        # until this date the only trace was a printed line and a per-fold
        # sentence in rule_missing; the scored cells simply lacked it. The
        # reasons are get_rule_hybrids' own sentences for that family, one
        # per distinct sentence over the folds.
        _families = []
        _fam_reasons = {}
        for fold_i in range(len(folds_b)):
            _r = _rule_cache.get((fold_i, budget))
            if _r is None:
                continue
            for _f in (_r.get('objectives') or []):
                if _f not in _families:
                    _families.append(_f)
                for _m in (((_r.get('by_objective') or {}).get(_f) or {})
                           .get('missing') or []):
                    _fam_reasons.setdefault(_f, [])
                    if _m not in _fam_reasons[_f]:
                        _fam_reasons[_f].append(_m)
        _families_absent = {
            _f: (_fam_reasons.get(_f)
                 or ['no fold returned a pool read for this budget'])
            for _f in (_families or ['predictive', 'similarity'])
            if not any(a[0] == _f for a in arms)}
        if _families_absent:
            print(f"    no hybrid from the {sorted(_families_absent)} "
                  f"family at this budget; reasons in per_budget_status")

        results['per_budget'][budget] = budget_results
        results['per_budget_status'][budget] = {
            'computed': bool(budget_results),
            'n_cells_scored': len(budget_results),
            'n_arms': len(arms),
            'arms': [f"{f}_{r}" for (f, r) in sorted(arms)],
            'families_absent': _families_absent,
            'failed_cells': failed_cells,
            'reason': (None if budget_results
                       else ('the pool held no kNN hybrid at this budget'
                             if not arms
                             else 'every arm x condition x k cell lost all folds')),
        }
        if not _threw:
            _ck.put({'budget_results': budget_results,
                     'status': results['per_budget_status'][budget],
                     'candidates': hybrid_candidates[_n_cand_before:]},
                    budget=budget)
    # Run-level: per family with no hybrid at some budget, which budgets and
    # why (2026-10-05). Read off per_budget_status, so a budget read back from
    # a sidecar written before this date (which has no 'families_absent')
    # counts as not known rather than present.
    results['hybrid_families_absent'] = {}
    for _b, _st in results['per_budget_status'].items():
        if 'families_absent' not in _st:
            results['hybrid_families_absent'].setdefault(
                '_not_recorded_for_budgets', []).append(_b)
            continue
        for _f, _why in (_st.get('families_absent') or {}).items():
            _e = results['hybrid_families_absent'].setdefault(
                _f, {'budgets': [], 'reasons': []})
            _e['budgets'].append(_b)
            for _w in _why:
                if _w not in _e['reasons']:
                    _e['reasons'].append(_w)

    # ── Per-rep kNN baselines (individual representations) ──
    # Try all k values and pick the best per metric direction.
    print(f"\n  Individual representation kNN baselines (metric={metric_name}):")
    # Reuse the SAME folds as the hybrid conditions (folds_b) so the
    # hybrid-vs-baseline comparison is on identical splits. (Previously a
    # separate, deterministic splitter rebuilt identical folds — SIMA-18.)
    folds_baseline = folds_b

    # Per-fold reselection baseline arm: per-(rep, k) unweighted kNN per-fold scores.
    # Keying each (rep, k) as its own baseline cell lets the held-out-fold
    # selection replace BOTH the same-fold best-rep argmax AND the same-fold
    # best-k argmax in one pass. These are "the real baselines" (unweighted).
    baseline_candidates = {}  # {f"{rep}|k{k}": [per-fold score, ...]}

    for rep_name, X_rep in reps.items():
        if _ck.done(baseline_rep=rep_name):
            _saved = _ck.get(baseline_rep=rep_name)
            results['per_rep_baselines'][rep_name] = _saved['baseline']
            baseline_candidates.update(_saved['candidates'])
            continue
        _threw = []
        _cand_before = set(baseline_candidates)
        best_k_score = None
        best_k_used = k_values[0]
        best_k_scores_uw = None
        best_k_scores_w = None
        best_k_std = 0.0
        best_k_w_mean = float('nan')

        # Per-fold importance computed ONCE per rep (independent of k).
        fold_importance = []
        for train_idx, _val_idx in folds_baseline:
            try:
                imp = compute_feature_importance(
                    {rep_name: X_rep[train_idx]}, labels[train_idx],
                    method='lightgbm_gain')
                fold_importance.append(
                    importance_feature_weights(imp[rep_name]))
            except Exception as exc:
                # N9: log so a None-importance fold reads as "compute failed"
                # rather than silently falling back to unweighted kNN.
                print(f"      importance fold failed (rep={rep_name}): "
                      f"{type(exc).__name__}: {exc}")
                _threw.append('importance')
                fold_importance.append(None)

        for k_candidate in k_values:
            fold_scores_uw = []
            fold_scores_w = []
            for fold_i, (train_idx, val_idx) in enumerate(folds_baseline):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_rep[train_idx])
                X_va = scaler.transform(X_rep[val_idx])

                # Unweighted
                model = _make_knn(k_candidate, len(train_idx))
                model.fit(X_tr, labels[train_idx])
                fold_scores_uw.append(metric_fn(labels[val_idx], model.predict(X_va)))

                # Weighted by importance (reuses pre-computed weights)
                w = fold_importance[fold_i]
                if w is None:
                    fold_scores_w.append(float('nan'))
                else:
                    try:
                        X_tr_w = X_tr * w
                        X_va_w = X_va * w
                        model_w = _make_knn(k_candidate, len(train_idx))
                        model_w.fit(X_tr_w, labels[train_idx])
                        fold_scores_w.append(metric_fn(labels[val_idx],
                                                       model_w.predict(X_va_w)))
                    except Exception as exc:
                        # N9: log so a NaN weighted-kNN fold reads as "fit/predict
                        # threw" rather than a genuine score.
                        print(f"      weighted kNN fold failed "
                              f"(rep={rep_name}, k={k_candidate}): "
                              f"{type(exc).__name__}: {exc}")
                        _threw.append(('weighted', fold_i))
                        fold_scores_w.append(float('nan'))

            if not fold_scores_uw:
                continue
            # Persist this (rep, k) cell's per-fold unweighted scores for the
            # honest leave-one-fold-out re-score (held-out selection over rep+k).
            baseline_candidates[f"{rep_name}|k{k_candidate}"] = [
                float(s) for s in fold_scores_uw]
            sc_mean = float(np.mean(fold_scores_uw))
            if best_k_score is None or (sc_mean < best_k_score if lower_is_better else sc_mean > best_k_score):
                best_k_score = sc_mean
                best_k_used = k_candidate
                best_k_std = float(np.std(fold_scores_uw))
                best_k_scores_uw = fold_scores_uw
                best_k_w_mean = float(np.nanmean(fold_scores_w)) if fold_scores_w else float('nan')

        sc_mean = best_k_score if best_k_score is not None else float('nan')
        sc_std = best_k_std
        sc_w_mean = best_k_w_mean
        results['per_rep_baselines'][rep_name] = {
            'score_mean': sc_mean, 'score_std': sc_std,
            'score_weighted_mean': sc_w_mean,
            'metric': metric_name,
            'k': best_k_used, 'dim': X_rep.shape[1],
        }
        # Sign-correct delta: positive = improvement regardless of metric
        # direction (negated for lower-is-better metrics like RMSE).
        sign = -1 if lower_is_better else 1
        delta = sign * (sc_w_mean - sc_mean)
        print(f"    {rep_name} (dim={X_rep.shape[1]}, k={best_k_used}): "
              f"kNN {metric_name}={sc_mean:.4f}+/-{sc_std:.3f}, "
              f"weighted {metric_name}={sc_w_mean:.4f} "
              f"(delta={delta:+.4f}, positive=improvement)")
        if not _threw:
            _ck.put({'baseline': results['per_rep_baselines'][rep_name],
                     'candidates': {_c: v for _c, v in baseline_candidates.items()
                                    if _c not in _cand_before}},
                    baseline_rep=rep_name)

    # ── The pool's best single representation, per fold ──
    # 2026-09-12 decision: the pool gives the best single representation as well
    # as the top three hybrids, chosen for each fold from the other folds'
    # scores. Fold i's representation can differ from fold j's, so this is one
    # arm across the folds, not one representation.
    #
    # It reuses the per-(representation, neighbour count) fold scores the loop
    # above already computed — same folds, same scaler, same neighbour model, no
    # importance weighting — so it costs no extra fits and is the same
    # arithmetic, not a second recipe.
    #
    # It is NOT added to baseline_candidates. The per-fold reselection already picks
    # a single representation per held-out fold from those cells; a per-fold
    # selection cannot also be one of the cells it selects among.
    print(f"\n  Pool's best single representation per fold "
          f"(chosen on the other folds' scores):")
    # The owner's set (2026-10-02) puts each model's top 2 single
    # representations on the base side: rank 1 under 'pool_base_rep' as
    # before, rank 2 under 'pool_base_rep2', both re-picked per fold.
    for _bidx, _bkey in ((0, 'pool_base_rep'), (1, 'pool_base_rep2')):
        pool_base_rep = {'per_fold': {}, 'per_k': {}, 'status': {'computed': False,
                                                                'reason': None}}
        _base_by_fold = {}
        for fold_i in range(len(folds_b)):
            _p = _fold_pool(fold_i, _base_budget)
            if _p is None:
                continue
            # Rank 1 is base_rep; rank 2 is the second of base_reps_ranked in the
            # same family base_rep came from.
            if _bidx == 0:
                _b = _p.get('base_rep')
            else:
                _b = None
                for _f in (_p.get('by_objective') or {}).values():
                    _rk = _f.get('base_reps_ranked') or []
                    if _rk and _rk[0] is _p.get('base_rep'):
                        _b = _rk[1] if len(_rk) > 1 else None
                        break
            if _b is None:
                continue
            _base_by_fold[fold_i] = _b
            pool_base_rep['per_fold'][fold_i] = {
                'rep': _b.get('rep'),
                'objective': _b.get('objective'),
                'pool_metric': _b.get('metric_name'),
                'pool_score_other_folds': _b.get('score_other_folds'),
                'pool_n_other_folds': _b.get('n_other_folds'),
            }
        if not _base_by_fold:
            pool_base_rep['status']['reason'] = (
                'the pool returned no base representation with per-fold scores for '
                'model=knn (a search pickled before baseline_fold_scores existed '
                'carries none)')
            print(f"    not computed ({pool_base_rep['status']['reason']})")
        else:
            for k in k_values:
                _fold_scores = [None] * len(folds_b)
                _missing = []
                for fold_i, _b in _base_by_fold.items():
                    _rep = _b.get('rep')
                    _cell = baseline_candidates.get(f"{_rep}|k{k}")
                    if _cell is None or fold_i >= len(_cell):
                        _missing.append(
                            f"fold {fold_i}: '{_rep}' is not one of the "
                            f"representations scored here at {k} neighbours")
                        continue
                    _fold_scores[fold_i] = float(_cell[fold_i])
                _scored = [s for s in _fold_scores if s is not None]
                pool_base_rep['per_k'][k] = {
                    'score_mean': (float(np.mean(_scored)) if _scored
                                   else float('nan')),
                    'score_std': (float(np.std(_scored)) if _scored else 0.0),
                    'metric': metric_name,
                    'k': k,
                    'fold_scores': list(_fold_scores),
                    'n_folds_scored': len(_scored),
                    'reps_per_fold': {int(f): _b.get('rep')
                                      for f, _b in _base_by_fold.items()},
                    'missing': _missing,
                }
                if _scored:
                    pool_base_rep['status']['computed'] = True
                    print(f"    k={k}: {metric_name}="
                          f"{pool_base_rep['per_k'][k]['score_mean']:.4f}"
                          f"+/-{pool_base_rep['per_k'][k]['score_std']:.3f} over "
                          f"{len(_scored)} folds "
                          f"(reps: {', '.join(sorted({str(b.get('rep')) for b in _base_by_fold.values()}))})")
                for _m in _missing:
                    print(f"    [warn] {_m}")
            if not pool_base_rep['status']['computed']:
                pool_base_rep['status']['reason'] = (
                    'no neighbour count scored the pool-selected representations on '
                    'any fold')
        results[_bkey] = pool_base_rep

    # ── Summary ──
    print(f"\n  Summary (best per budget):")
    for b in sorted(results['per_budget'].keys()):
        budget_res = results['per_budget'][b]
        if budget_res:
            best_key = (min if lower_is_better else max)(
                budget_res, key=lambda k: budget_res[k]['score_mean'])
            best = budget_res[best_key]
            _evs = sorted({r['evaluator']
                           for r in best.get('fold_recipes', {}).values()})
            print(f"    budget={b}: best={best.get('pool_family')} "
                  f"{best.get('pool_rank')} {best['condition']} "
                  f"k={best['k']}, "
                  f"{metric_name}={best['score_mean']:.4f} "
                  f"(evaluators across folds: {', '.join(_evs) or '?'})")

    # ── The reference: one (representation, downstream model) pair per model ──
    # 2026-09-12 decision, see the block at the top of this file. For each model
    # in DISTANCE_MODELS the reference is the best single representation FOR
    # THAT MODEL, paired with it, and its per-fold scores are the stored
    # search's own baseline scores for that pairing — nothing new is fitted.
    # The representation is ranked for each fold on the other folds' scores, so
    # a fold never picks the representation it is then scored on, and it can
    # differ from fold to fold.
    #
    # Only a pool family scoring in this dataset's own metric is read: the
    # similarity family stores a correlation between each held-out molecule's
    # activity and what its five nearest neighbours predict, which is not the
    # metric any other number in this file is in, so it cannot be subtracted
    # from one.
    def _model_reference(model):
        """The (representation, model) reference pair, one score per fold."""
        fold_scores = [None] * len(folds_b)
        reps_by_fold = {}
        score_source = {}
        pool_metrics = set()
        missing = []
        for fold_i in range(len(folds_b)):
            pool = _fold_pool(fold_i, _base_budget, model=model)
            if pool is None:
                missing.append(f"fold {fold_i}: the pool read failed")
                continue
            _fam = None
            for _obj, _f in (pool.get('by_objective') or {}).items():
                if (_f.get('metric_name') == metric_name
                        and _f.get('base_reps_ranked')):
                    _fam = _f
                    break
            if _fam is None:
                missing.append(
                    f"fold {fold_i}: no pool family holds {model} baseline "
                    f"scores in {metric_name}, the metric every number here is "
                    f"in")
                continue
            _ranked = _fam['base_reps_ranked']
            if reference_override is not None:
                _entry = next((b for b in _ranked
                               if b.get('rep') == reference_override), None)
                if _entry is None:
                    missing.append(
                        f"fold {fold_i}: the override representation "
                        f"'{reference_override}' carries no stored {model} "
                        f"score")
                    continue
            else:
                _entry = _ranked[0]
            _rep = _entry.get('rep')
            reps_by_fold[fold_i] = _rep
            pool_metrics.add(_fam.get('metric_name'))
            # kNN: prefer the score this file computed for that representation
            # at REFERENCE_N_NEIGHBOURS neighbours on this fold, so the
            # reference and the hybrid cells come out of the same fits.
            _local = (baseline_candidates.get(f"{_rep}|k{REFERENCE_N_NEIGHBOURS}")
                      if model == 'knn' else None)
            if _local is not None and fold_i < len(_local):
                fold_scores[fold_i] = float(_local[fold_i])
                score_source[fold_i] = (
                    f'scored here, {REFERENCE_N_NEIGHBOURS} neighbours')
                continue
            _sc = _entry.get('score_this_fold')
            if _sc is None:
                missing.append(
                    f"fold {fold_i}: '{_rep}' has no stored {model} score on "
                    f"this fold")
                continue
            fold_scores[fold_i] = float(_sc)
            score_source[fold_i] = 'stored search baseline score'

        _scored = [s for s in fold_scores if s is not None]
        _names = sorted({r for r in reps_by_fold.values() if r})
        return {
            'model': model,
            'representation': (_names[0] if len(_names) == 1 else None),
            'reps_by_fold': {int(f): r for f, r in sorted(reps_by_fold.items())},
            'n_neighbours': (REFERENCE_N_NEIGHBOURS if model == 'knn' else None),
            'selection': ('override named before the run'
                          if reference_override is not None
                          else "best for this model on the pool search's other "
                               "folds, re-picked per fold"),
            'source': reference_source,
            'score_mean': (float(np.mean(_scored)) if _scored else None),
            'score_std': (float(np.std(_scored)) if _scored else None),
            'fold_scores': list(fold_scores),
            'n_folds': len(_scored),
            'metric': metric_name,
            'pool_metric': (sorted(m for m in pool_metrics if m) or [None])[0],
            'score_source_by_fold': {int(f): s
                                     for f, s in sorted(score_source.items())},
            'missing': missing,
        }

    reference_by_model = {}
    print(f"\n  Reference per downstream model "
          f"(best single representation for that model, {metric_name}):")
    for _model in DISTANCE_MODELS:
        _ref = _model_reference(_model)
        reference_by_model[_model] = _ref
        if _ref['score_mean'] is None:
            print(f"    {_model}: not computed "
                  f"({_ref['missing'][0] if _ref['missing'] else 'no folds'})")
            continue
        _names = sorted({r for r in _ref['reps_by_fold'].values() if r})
        print(f"    {_model}: {'/'.join(_names)} "
              f"{metric_name}={_ref['score_mean']:.4f}"
              f"+/-{_ref['score_std']:.3f} over {_ref['n_folds']} folds")
        for _m in _ref['missing']:
            print(f"      [warn] {_m}")

    # The headline reference is the kNN pair, because every hybrid cell in this
    # file is a kNN score. The old top-level keys are kept so the readers that
    # already print results['reference'] keep working; every model is under
    # 'by_model'.
    _knn_ref = reference_by_model.get('knn') or {}
    reference_rep = _knn_ref.get('representation')
    if reference_rep is None:
        # The per-fold ranking can land on different representations. Name them
        # all rather than leaving the field empty.
        _rnames = sorted({r for r in (_knn_ref.get('reps_by_fold') or {}).values()
                          if r})
        reference_rep = '/'.join(_rnames) if _rnames else None
    reference_score = _knn_ref.get('score_mean')
    reference_error = None
    results['reference'] = dict(_knn_ref)
    results['reference']['by_model'] = reference_by_model
    results['reference']['headline_model'] = 'knn'
    results['reference']['note'] = (
        'One (representation, downstream model) pair per model. The headline '
        'deltas below subtract the kNN pair; every other model\'s delta is in '
        "summary.reference_by_model. No fit is new: a pair's per-fold scores "
        'are the stored search\'s own baseline scores, except the kNN pair, '
        'which reuses the neighbour scores this file already computed for that '
        'representation on that fold.')
    if reference_score is None:
        reference_error = ((_knn_ref.get('missing') or ['no fold scored'])[0])
        results['reference_status'] = {
            'computed': False,
            'reason': reference_error,
        }
        print(f"\n  Reference (knn pair): not computed ({reference_error})")

    results['summary'] = {
        'reference_rep': reference_rep,
        'reference_model': 'knn',
        'reference_reps_by_fold': _knn_ref.get('reps_by_fold'),
        'reference_override': reference_override,
        'reference_n_neighbours': REFERENCE_N_NEIGHBOURS,
        'reference_score': reference_score,
        # One row per downstream model: which representation it pairs with and
        # what that pair scores. The deltas against each are filled in below.
        'reference_by_model': {
            m: {'model': m,
                'representation': r.get('representation'),
                'reps_by_fold': r.get('reps_by_fold'),
                'score_mean': r.get('score_mean'),
                'n_folds': r.get('n_folds'),
                'metric': r.get('metric')}
            for m, r in reference_by_model.items()},
    }

    # One delta per downstream model: the hybrid cell's score, which is always a
    # kNN score here, minus that model's own (representation, model) reference.
    # This is the model column the decision asks for — one hybrid cell now
    # reports one row per model instead of one row against one representation.
    def _deltas_by_model(score_mean):
        _sign = -1 if lower_is_better else 1
        out = {}
        for m, r in reference_by_model.items():
            if r.get('score_mean') is None:
                out[m] = {
                    'model': m,
                    'delta_vs_reference': None,
                    'reason': (r.get('missing') or ['no fold scored'])[0],
                }
                continue
            out[m] = {
                'model': m,
                'representation': r.get('representation'),
                'reps_by_fold': r.get('reps_by_fold'),
                'reference_score': r.get('score_mean'),
                'delta_vs_reference': _sign * (score_mean - r['score_mean']),
                'n_folds': r.get('n_folds'),
                'metric': r.get('metric'),
            }
        return out

    all_hybrid_scores = [v for b_res in results['per_budget'].values()
                         for v in b_res.values()]
    if reference_score is not None and all_hybrid_scores:
        sign = -1 if lower_is_better else 1
        best_hybrid = (min if lower_is_better else max)(
            all_hybrid_scores, key=lambda v: v['score_mean'])
        delta = sign * (best_hybrid['score_mean'] - reference_score)
        # MAX over budgets x conditions x neighbour counts on the HYBRID side,
        # scored on the folds it was picked on = post-hoc optimistic. Recorded
        # but NOT the headline number. The thing it is subtracted from is the
        # named reference, one representation at five neighbours.
        results['summary']['posthoc_max'] = {
            'budget': best_hybrid['budget'],
            'condition': best_hybrid['condition'],
            'k': best_hybrid['k'],
            # 2026-09-12: the max now also ranges over the pool family and the
            # pool rank the arm came from, so the cell names both.
            'pool_family': best_hybrid.get('pool_family'),
            'pool_rank': best_hybrid.get('pool_rank'),
            'score_mean': best_hybrid['score_mean'],
            'delta_vs_reference': delta,
            # Same number under the name s09 reads
            # (summary.posthoc_max.delta_vs_best_rep).
            'delta_vs_best_rep': delta,
            'reference_rep': reference_rep,
            'reference_model': 'knn',
            'reference_n_neighbours': REFERENCE_N_NEIGHBOURS,
            # One row per downstream model, each against that model's own
            # best single representation.
            'delta_by_model': _deltas_by_model(best_hybrid['score_mean']),
            'note': ('MAX over budgets x pool families x pool ranks x '
                     'conditions x neighbour counts on the hybrid side, scored '
                     'on the folds it was picked on = post-hoc optimistic; do '
                     'not report as the headline. Every arm in that maximum was '
                     "scored here in the dataset's own metric, so the two pool "
                     'families are comparable in it. It subtracts the named '
                     'reference.'),
        }
        print(f"  Best hybrid kNN [posthoc-max, optimistic]: "
              f"budget={best_hybrid['budget']}, "
              f"{best_hybrid.get('pool_family')} "
              f"{best_hybrid.get('pool_rank')} "
              f"{best_hybrid['condition']} k={best_hybrid['k']}, "
              f"{metric_name}={best_hybrid['score_mean']:.4f} "
              f"(delta vs {reference_rep} at {REFERENCE_N_NEIGHBOURS} "
              f"neighbours: {delta:+.4f})")

        # The pre-registered hybrid cell: 100 features (nearest budget scored),
        # unweighted, at the reference's five neighbours, so both sides of this
        # subtraction use the one neighbour count.
        #
        # 2026-09-12: the cell now also names which pool arm it is — the pool's
        # rank-1 hybrid in the family FIXED_REF_FAMILY names at the top of this
        # file. Both are fixed before the run. The pool does not rank the two
        # families against each other, so taking the better of the two here
        # would be a maximum, not a pre-registration; each family's own rank-1
        # cell is written under 'fixed_ref_by_family' instead.
        FIXED_BUDGET, FIXED_COND, FIXED_RANK = 100, 'hybrid_unweighted', FIXED_SLOT
        avail = sorted(results['per_budget'].keys())
        fb = min(avail, key=lambda b: abs(b - FIXED_BUDGET)) if avail else None
        fixed_cell = None
        if fb is not None:
            _fixed_key = (f'{FIXED_REF_FAMILY}_{FIXED_RANK}_{FIXED_COND}'
                          f'_k{REFERENCE_N_NEIGHBOURS}')
            fixed_cell = results['per_budget'][fb].get(_fixed_key)
            # Every family's own rank-1 cell at the same budget, condition and
            # neighbour count, so the one that is not the pre-registered family
            # is still on the record.
            _by_family = {}
            for _key, _cell in results['per_budget'][fb].items():
                if (_cell.get('pool_rank') == FIXED_RANK
                        and _cell.get('condition') == FIXED_COND
                        and _cell.get('k') == REFERENCE_N_NEIGHBOURS):
                    _fdelta = sign * (_cell['score_mean'] - reference_score)
                    _by_family[_cell.get('pool_family')] = {
                        'budget': _cell['budget'],
                        'score_mean': _cell['score_mean'],
                        'delta_vs_reference': _fdelta,
                        'n_folds_scored': _cell.get('n_folds_scored'),
                        'delta_by_model': _deltas_by_model(_cell['score_mean']),
                    }
            if _by_family:
                results['summary']['fixed_ref_by_family'] = _by_family
                results['summary']['fixed_ref_family'] = FIXED_REF_FAMILY
        if fixed_cell is not None:
            fdelta = sign * (fixed_cell['score_mean'] - reference_score)
            results['summary']['fixed_ref'] = {
                'budget': fixed_cell['budget'],
                'condition': fixed_cell['condition'],
                'k': fixed_cell['k'],
                'pool_family': fixed_cell.get('pool_family'),
                'pool_rank': fixed_cell.get('pool_rank'),
                'score_mean': fixed_cell['score_mean'],
                'delta_vs_reference': fdelta,
                # Same number under the name s09 reads
                # (summary.fixed_ref.delta_vs_best_rep).
                'delta_vs_best_rep': fdelta,
                'reference_rep': reference_rep,
                'reference_model': 'knn',
                'reference_n_neighbours': REFERENCE_N_NEIGHBOURS,
                # One row per downstream model, each against that model's own
                # best single representation.
                'delta_by_model': _deltas_by_model(fixed_cell['score_mean']),
                'note': ('Both sides fixed before the run: the pool\'s '
                         f'{FIXED_RANK} {FIXED_REF_FAMILY} hybrid at this '
                         'budget, '
                         'unweighted, at five neighbours, against the named '
                         'reference representation at five neighbours. Nothing '
                         'here is picked on the folds it is scored on — the '
                         'rank comes from the pool search\'s other folds. The '
                         'held-out number, knn_delta_heldout, is still the one '
                         'to report.'),
            }
            print(f"  Best hybrid kNN [fixed cell vs named reference]: "
                  f"budget={fixed_cell['budget']}, "
                  f"{fixed_cell.get('pool_family')} "
                  f"{fixed_cell.get('pool_rank')} "
                  f"{fixed_cell['condition']} "
                  f"k={fixed_cell['k']}, "
                  f"{metric_name}={fixed_cell['score_mean']:.4f} "
                  f"(delta vs {reference_rep} at {REFERENCE_N_NEIGHBOURS} "
                  f"neighbours: {fdelta:+.4f}, positive=improvement)")
            print(f"    per downstream model (positive=hybrid better):")
            for _m, _row in results['summary']['fixed_ref']['delta_by_model'].items():
                if _row.get('delta_vs_reference') is None:
                    print(f"      {_m:<9} not computed ({_row.get('reason')})")
                else:
                    print(f"      {_m:<9} vs {_row.get('representation') or 'per-fold'}: "
                          f"{metric_name}={_row['reference_score']:.4f}, "
                          f"delta={_row['delta_vs_reference']:+.4f}")
        else:
            # F18: 'posthoc_max' (selection-optimistic) exists but the
            # pre-registered comparison does not. Say so, so the optimistic
            # number is not left as the only summary quantity.
            results['summary']['fixed_ref_status'] = {
                'computed': False,
                'reason': (f"no {FIXED_RANK} {FIXED_REF_FAMILY} "
                           f"{FIXED_COND} cell at {REFERENCE_N_NEIGHBOURS} "
                           f"neighbours and the nearest budget scored "
                           f"(asked for {FIXED_BUDGET} features, nearest "
                           f"{fb}; neighbour counts scored this run: "
                           f"{k_values})"),
            }
            print(f"  Best hybrid kNN [fixed cell]: not computed "
                  f"({results['summary']['fixed_ref_status']['reason']})")
    else:
        # Neither comparison can be taken. Say which half is missing.
        _why = (reference_error if reference_score is None
                else 'no budget x condition x neighbour-count cell was scored')
        results['summary']['fixed_ref_status'] = {'computed': False,
                                                  'reason': _why}
        results['summary']['posthoc_max_status'] = {'computed': False,
                                                    'reason': _why}
        print(f"  No hybrid-versus-reference comparison this run ({_why})")

    # ── Honest leave-one-fold-out re-score block (selection-optimism fix) ──
    # build_fold_reselection_input() reads a SearchResult; this experiment runs
    # its own kNN CV, so we assemble the standard block by hand from the
    # per-fold scores collected above (single 'knn' model by design). A reader
    # loads results['fold_reselection'] and calls the shared fold_reselection(...).
    #
    # 2026-09-12: the candidates are no longer recipes this file searched for
    # itself. They are the pool's top three hybrids per family, ranked for each
    # fold on the other folds' scores, so two things change here.
    #
    # 'recipe_selection' now says which rows chose the recipe: the pool search's
    # other folds. The search itself ran over every row, so fold i's SCORES were
    # held out of the ranking, not fold i's rows.
    #
    # 'exclude_folds' is [0] when any pool member filtered its own candidate
    # list on fold 0's scores. Fold 0's hybrid score would otherwise be read off
    # recipes fold 0 helped keep, while the single representations passed
    # through no such filter — the two arms would not pay the same held-out tax.
    # pool_fold0_pruned reads None (a search pickled before the flag existed) as
    # pruned, the same reading hr_exclude_folds takes.
    # The hybrid arms come from get_rule_hybrids (2026-10-02), which does not
    # pass through whether the general searches pruned on fold 0. So fold 0 is
    # excluded whenever any arm was read from it -- the conservative reading.
    _rule_used = any((r or {}).get('hybrids') for r in _rule_cache.values())
    _exclude = ([0] if (pool_fold0_pruned is None or pool_fold0_pruned
                        or _rule_used) else [])
    results['fold_reselection'] = {
        'lower_is_better': bool(lower_is_better),
        'recipe_selection': 'other_folds_of_pool_search',
        'exclude_folds': _exclude,
        'fold0_pruned': pool_fold0_pruned,
        'candidates_by_model': (
            {'knn': hybrid_candidates} if hybrid_candidates else {}),
        'baselines_by_model': (
            {'knn': baseline_candidates} if baseline_candidates else {}),
    }
    # F18: an empty hybrid arm makes the block above unusable. Say which arm is
    # empty and why, so the block is not read as "no hybrid cell won".
    if not hybrid_candidates:
        _st = results.get('search_status', {})
        results['fold_reselection_status'] = {
            'computed': False,
            'n_hybrid_cells': 0,
            'n_baseline_cells': len(baseline_candidates),
            'reason': ((f"the pool read raised "
                        f"{_st.get('error_type', 'Exception')}: {_st.get('error', '')}")
                       if not _st.get('ok', False)
                       else ('no budget x pool arm x condition x k cell '
                             'produced fold scores')),
        }
        print(f"\n  Honest LOFO re-score (knn): not computed "
              f"({results['fold_reselection_status']['reason']})")
    else:
        results['fold_reselection_status'] = {
            'computed': True,
            'n_hybrid_cells': len(hybrid_candidates),
            'n_baseline_cells': len(baseline_candidates),
            'pool_subset': POOL_SUBSET,
            'n_hybrids_per_family': N_HYBRIDS_PER_FAMILY,
            'exclude_folds': _exclude,
            'reason': None,
        }
    # Reuse the shared pure-arithmetic helper (no ML) so the honest hybrid-vs-
    # best-single-rep delta is available inline alongside the optimistic
    # posthoc_max / pre-registered fixed_ref summaries.
    try:
        hr = fold_reselection(
            results['fold_reselection']['candidates_by_model'],
            results['fold_reselection']['baselines_by_model'],
            bool(lower_is_better),
            exclude_folds=hr_exclude_folds(results['fold_reselection']),
            # 2026-09-12: carried onto every model entry so a reader is told
            # what chose the recipe — the pool search's other folds.
            recipe_selection=results['fold_reselection']['recipe_selection'])
        results['fold_reselection_result'] = hr
        knn_hr = hr.get('knn')
        if knn_hr is not None:
            flag = ' [single-fold, NOT honest]' if knn_hr.get('single_fold') else ''
            print(f"\n  Honest LOFO re-score (knn){flag}: "
                  f"hybrid={knn_hr['hybrid_reselected']:.4f}, "
                  f"best_single={knn_hr['best_single_reselected']:.4f}, "
                  f"delta={knn_hr['delta']:+.4f} (positive=hybrid better)")
    except Exception as e:
        print(f"  [warn] fold_reselection failed: {type(e).__name__}: {e}")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('S05: kNN Prediction with Hybrid Distance',
                                     has_delta=True, has_augment=True)
    parser.add_argument(
        '--reference-rep', default=None,
        help=('Override the reference representation for every downstream '
              "model. It must be one of the representations in this dataset's "
              'pool. Leave it unset and each model is paired with the best '
              'single representation for it, chosen per fold on the other '
              "folds' scores; REFERENCE_REP_BY_DATASET overrides that for the "
              'two datasets it names.'))
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset,
                  reference_rep=args.reference_rep)
    save_experiment_results(results, args.output, 's05_knn_prediction', args.dataset)
