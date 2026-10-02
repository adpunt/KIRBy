#!/usr/bin/env python3
"""S02: Hybrid Property-Aware Similarity -- the fix via precomputed pool best + weighting.

Paper: Similarity
Conclusion: Hybrid distance improves SAC by 100%+.
Dependencies: None.

Computes per-rep SAC baselines, finds best individual rep, retrieves the
precomputed pool best hybrid via get_fitted_pool_hybrid, then for each weight
method computes importance-weighted distance matrix and compares SAC: hybrid
weighted vs best individual rep.
Saves blueprint and search details for downstream experiments (S03, S04, S05).

2026-09-12, owner's decision: the subset of the precomputed pool this file may
pick its hybrid from now INCLUDES hybrids built under objective='similarity' —
searches whose features were chosen to make the distance geometry predict each
molecule's activity from its five nearest neighbours. Until now every hybrid
this file could pick had been built for prediction, while the question it asks
is a distance-geometry one. The result file records which of the two families
the winning hybrid came from (see 'hybrid_family' below).

2026-09-12, two more of the owner's decisions changed the shape of the result
file. The budget is swept over all eight — 10, 25, 50, 100, 250, 500, 1,000 and
2,000 features — instead of the single 100 this file used to read; and both
distance lineages are asked, the general pool and the distance pool, the way s04
asks its own. Every hybrid statistic is therefore one per (lineage, budget)
cell, under results['lineages'][lineage]['budgets'][budget]. No search runs for
any of them: the finished searches on disk already hold every budget. The keys
that used to sit at the top level still do, mirroring one named cell — the
general lineage at budget 100, which is the only cell this file scored before —
and results['primary_cell'] says which cell that is.

2026-10-02, the owner's set: top 2 general + top 1 distance hybrids per model
per family. For each (budget, family) get_rule_hybrids returns knn's top 2
recipes from the general searches and top 1 from the distance searches; each is
built and scored, under results['rule_slots'][slot]['budgets'][budget] with
slot 'general1', 'general2' or 'distance1'. results['lineages'] still holds
general1 (as 'general') and distance1 (as 'general_distance'). Empty slots and
why: results['rule_missing'].
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, get_rule_hybrids, build_hybrid_from_entry,
    get_best_search_for_downstream,
    flatten_fold_weights, extract_top3_base_reps,
    cv_split, fold_reselection, build_fold_hybrid,
    hr_exclude_folds, MAIN_MODELS_WITH_KNN, FULL_BUDGETS,
    top_base_reps, top_hybrids, select_model_rep_pairs, hybrid_deltas,
    resolve_comparison_models,
)
from experiments.similarity.similarity_helpers import (
    similarity_activity_correlation, compute_sali, violation_rate,
    run_antibody_cliff_analysis,
)

from kirby.hybrid import compute_feature_importance
from kirby.utils.noise_mitigation_methods import compute_distance_matrix
from experiments.similarity._summary_keys import (
    SELECTION_RULE as _SELECTION_RULE)


# =============================================================================
# CROSS-VALIDATED, SELECTION-HONEST SAC (redesign — decision B, user-approved)
# =============================================================================
# ORIGINAL (whole-dataset) design: the best individual rep was argmax'd over the
# SAC of every rep on the FULL dataset (run() best_rep loop), the hybrid was
# built on the FULL dataset via get_fitted_pool_hybrid, and its weighted SAC was
# computed on the FULL dataset too. Both arms therefore SELECTED and SCORED on
# the same samples — pure selection optimism, exactly the leak the fold-reselection
# fix targets.
#
# REDESIGN: split by the dataset groups (group-aware CV). SELECT the winning rep
# and the winning hybrid recipe on TRAIN folds only, then COMPUTE the SAC
# statistic on the HELD-OUT fold, and average held-out SAC across folds. Because
# SAC is an UNSUPERVISED distance-metric statistic (not a model fold-score from
# find_best_hybrid), the SearchResult-based build_fold_reselection_input helper
# does not fit here; instead this file computes per-fold HELD-OUT SAC for every
# baseline rep and every hybrid recipe locally (below) and writes them into the
# CONTRACT's "fold_reselection" JSON block. The shared fold_reselection() helper then
# supplies the identical held-out arithmetic used everywhere else: for each
# held-out fold it re-picks the winner by the mean over the OTHER folds and reads
# that winner's held-out score. Any feature SELECTION that depends on labels
# (the importance WEIGHTING) is fit on the TRAIN rows of each fold only; the
# unweighted arms need no fitting so their held-out SAC is leak-free by
# construction and the honesty comes purely from the per-fold winner re-pick.
#
# WHAT IS AND IS NOT REFIT PER FOLD. The hybrid's FEATURES are refit per fold
# (2026-09-04): each held-out fold rebuilds the hybrid with build_fold_hybrid on
# its own training rows, so no feature is chosen using a label the fold then
# scores against. The importance WEIGHTS are refit per fold as well. What is
# still chosen over every row is the RECIPE — allocation method, budget,
# importance method, internal evaluator — which comes from the whole-dataset
# pool search (get_fitted_pool_hybrid). That is a separate open finding against
# get_best_search_for_downstream, not something this file decides. The
# recipe-level choice s02 itself makes (which weight method, weighted vs
# unweighted) is made per held-out fold by fold_reselection().
#
# SAC is a single-model statistic (the distance metric; downstream_model='knn'),
# so everything is keyed under one model name, 'knn'. lower_is_better is False
# (higher SAC = distance better preserves the structure-activity relationship).

_SAC_MODEL_KEY = 'knn'  # SAC is a distance-metric statistic; single "model".


def _hybrid_weights_from_fi(fi, wm_importance, n_cols):
    """Build the per-column hybrid weight vector from a feature_info layout and
    a (train-computed) per-rep importance dict.

    Mirrors run()'s full-data weight construction exactly, but the importance
    source is passed in (so callers can supply TRAIN-only importance). Raises if
    the flattened length does not match the hybrid column count.
    """
    feat_info = {}
    for rep_name, info in fi.items():
        if rep_name == 'augmentation_info' or not isinstance(info, dict):
            continue
        idx = info.get('selected_indices')
        if idx is None:
            continue
        entry = {'selected_indices': idx,
                 'n_features': info.get('n_features', len(idx))}
        if rep_name in wm_importance:
            entry['importance_scores'] = wm_importance[rep_name][idx]
        feat_info[rep_name] = entry
    if fi.get('augmentation_info'):
        # Augmentation importance stays as the global blueprint's (see LIMITATION
        # above); it keeps weights_flat aligned with the hybrid columns.
        feat_info['augmentation_info'] = fi['augmentation_info']
    weights_flat = flatten_fold_weights(feat_info)
    if len(weights_flat) != n_cols:
        raise ValueError(
            f"weights_flat length {len(weights_flat)} != hybrid columns {n_cols}")
    return weights_flat


def _cv_reselected_sac(reps, labels, groups, blueprint, aug_condition,
                   augmentations, weight_methods,
                   n_splits, n_sample, is_clf, task_type, seed=42):
    """Cross-validated, selection-honest SAC (the redesign described above).

    Computes, for every held-out CV fold, the SAC of each BASELINE rep
    (unweighted + train-weighted) and each HYBRID recipe (unweighted +
    train-weighted per weight method) on that held-out fold's samples. Label-
    dependent importance weighting is fit on TRAIN rows only; unweighted arms
    use no labels for the distance.

    The hybrid's FEATURES are now selected per fold too (2026-09-04): each fold
    rebuilds the hybrid from `blueprint` with build_fold_hybrid on that fold's
    training rows. Before that the whole-dataset matrix was sliced, so the
    features had been chosen with the held-out rows' labels in hand, while the
    single representations it is compared against had no selection at all.

    Returns the CONTRACT "fold_reselection" block ready to json.dump:
        {"lower_is_better": False,
         "candidates_by_model": {"knn": [{"name", "fold_scores"}, ...]},
         "baselines_by_model":  {"knn": {rep_key: [fold_scores], ...}}}
    A reader loads it and calls fold_reselection(...) to get the honest deltas.
    """
    labels = np.asarray(labels)
    baseline_folds = {}   # rep_key   -> [per-fold held-out SAC]
    cand_folds = {}       # recipe id -> [per-fold held-out SAC]

    def _push(store, key, val):
        store.setdefault(key, []).append(float(val))

    def _sac(X_sub, y_sub, weights=None):
        try:
            if weights is None:
                D = compute_distance_matrix(X_sub, standardize=True)
            else:
                D = compute_distance_matrix(
                    X_sub, feature_weights=np.maximum(weights, 0),
                    standardize=True)
            rho, _ = similarity_activity_correlation(
                D, y_sub, n_sample, is_clf=is_clf)
            return rho
        except Exception:
            return float('nan')

    _sample_X = next(iter(reps.values()))
    _augs = augmentations if (augmentations and aug_condition != 'none') else None
    for train_idx, test_idx in cv_split(groups, n_splits, _sample_X, labels,
                                        seed=seed, task_type=task_type):
        y_test = labels[test_idx]
        y_train = labels[train_idx]

        # This fold's hybrid: features chosen from the training rows only, then
        # applied to the held-out rows. A build that throws drops the hybrid arm
        # for this fold; the helper tolerates a ragged fold list.
        try:
            _, Xh_test, fi = build_fold_hybrid(
                {k: X[train_idx] for k, X in reps.items()},
                {k: X[test_idx] for k, X in reps.items()},
                y_train, blueprint,
                augmentations_train=({k: v[train_idx] for k, v in _augs.items()}
                                     if _augs else None),
                augmentations_val=({k: v[test_idx] for k, v in _augs.items()}
                                   if _augs else None),
                groups=(np.asarray(groups)[train_idx]
                        if groups is not None else None))
        except Exception:
            Xh_test, fi = None, None

        # Train-only importance for the WEIGHTED arms (lightgbm_gain matches the
        # full-data weighted baselines). Failure -> weighted arms simply absent
        # this fold (helper tolerates ragged/missing folds).
        reps_train = {k: X[train_idx] for k, X in reps.items()}
        try:
            base_imp_train = compute_feature_importance(
                reps_train, y_train, method='lightgbm_gain')
        except Exception:
            base_imp_train = None

        # --- baseline single-rep arm ---
        for rep_name, X in reps.items():
            Xt = X[test_idx]
            _push(baseline_folds, rep_name, _sac(Xt, y_test))
            if base_imp_train is not None and rep_name in base_imp_train:
                _push(baseline_folds, f'{rep_name}|weighted',
                      _sac(Xt, y_test, weights=base_imp_train[rep_name]))

        # --- hybrid candidate arm ---
        if Xh_test is None:
            continue
        _push(cand_folds, 'hybrid_unweighted', _sac(Xh_test, y_test))
        for wm in weight_methods:
            try:
                if wm == 'lightgbm_gain' and base_imp_train is not None:
                    wm_imp = base_imp_train
                else:
                    wm_imp = compute_feature_importance(
                        reps_train, y_train, method=wm)
                weights_flat = _hybrid_weights_from_fi(
                    fi, wm_imp, Xh_test.shape[1])
                sac_hw = _sac(Xh_test, y_test, weights=weights_flat)
            except Exception:
                sac_hw = float('nan')
            _push(cand_folds, f'hybrid_weighted|{wm}', sac_hw)

    return {
        'lower_is_better': False,  # SAC higher = better (see redesign note).
        # Every fold counts. The candidates here are the arms this file built itself,
        # not a search's recipe list, so no fold filtered them.
        'exclude_folds': [],
        'candidates_by_model': {
            _SAC_MODEL_KEY: [{'name': k, 'fold_scores': v}
                             for k, v in cand_folds.items()],
        },
        'baselines_by_model': {
            _SAC_MODEL_KEY: dict(baseline_folds),
        },
    }


# =============================================================================
# ONE POOL CELL — one distance lineage at one budget
# =============================================================================
# 2026-09-12, owner's decisions. Three things changed shape at once, so the
# per-cell scoring moved out of run() and into this function:
#   - the budget is swept over all eight (10, 25, 50, 100, 250, 500, 1000 and
#     2000 features) instead of the single 100 the file used to read, so every
#     hybrid statistic below is now one per budget;
#   - both distance lineages are asked — the general pool (distance=False) and
#     the distance pool (distance=True) — the way s04 asks its own, so every
#     statistic is also one per lineage;
#   - the recipe recorded, and the recipe each held-out fold rebuilds from, is
#     the one the scored matrix was actually built from.
# A cell whose pool lookup raises records hybrid_status and returns. It never
# falls back to the other lineage, and never to another budget.

def _search_report_rows(search, *, dataset_name, distance, budget):
    """The rows read off the stored search the pool picks for one lineage.

    These used to be written inside every scored cell, from the search
    get_fitted_pool_hybrid returned. Since 2026-10-02 the scored hybrids come
    from get_rule_hybrids, which hands back recipes rather than a search, so
    these rows are read here once per (lineage, budget, family) from the search
    get_best_search_for_downstream picks for knn. They describe that search;
    the hybrids this file scores are named in each cell's 'slot' and 'recipe'.
    0 new model fits.
    """
    rows = {}
    top3 = extract_top3_base_reps(search)
    if top3:
        rows['top3_base_reps'] = [(n, s) for n, s in top3]
    _sel_ctx = (f"s02/{dataset_name}/"
                f"{'general_distance' if distance else 'general'}/"
                f"budget={budget} features")
    try:
        rows['top2_base_reps'] = top_base_reps(
            search, _SAC_MODEL_KEY, 2, require_full=False, context=_sel_ctx)
    except Exception as e:
        print(f"    top2_base_reps not added: {type(e).__name__}: {e}")
    try:
        # No budget= filter (2026-09-28): a base representation has no budget,
        # so both halves rank over every stored row. Each entry carries its own
        # 'budget'.
        rows['top2_hybrids'] = top_hybrids(
            search, _SAC_MODEL_KEY, 2, require_full=False, context=_sel_ctx)
    except Exception as e:
        print(f"    top2_hybrids not added: {type(e).__name__}: {e}")
    # The six models this file asks for; a name the search never scored is
    # dropped and named (resolve_comparison_models), never swapped.
    _cmp_models, _cmp_dropped = resolve_comparison_models(
        search, MAIN_MODELS_WITH_KNN)
    rows['top2_models_compared'] = list(_cmp_models)
    rows['top2_models_dropped'] = {_n: _w for _n, _w in _cmp_dropped}
    try:
        rows['top2_pairs_by_model'] = select_model_rep_pairs(
            search, models=_cmp_models, n_base=2, n_hybrids=2,
            require_full=False, context=_sel_ctx)
    except Exception as e:
        print(f"    top2_pairs_by_model not added: {type(e).__name__}: {e}")
        rows['top2_pairs_by_model_absent_reason'] = f"{type(e).__name__}: {e}"
    # hybrid= is passed: hybrid_deltas with hybrid=None reads a 'row' key the
    # top_hybrids entry does not carry.
    _deltas = {}
    for _m in _cmp_models:
        try:
            _top1 = top_hybrids(search, _m, 1,
                                require_full=False, context=_sel_ctx)
            if not _top1['selected']:
                rows['top2_models_dropped'][_m] = (
                    'no hybrid delta: %s' % _top1['shortfall_reason'])
                continue
            _deltas[_m] = hybrid_deltas(search, _m,
                                        hybrid=_top1['selected'][0],
                                        context=_sel_ctx)
        except Exception as e:
            rows['top2_models_dropped'][_m] = (
                'no hybrid delta: %s: %s' % (type(e).__name__, e))
    if _deltas:
        rows['hybrid_deltas_by_model'] = _deltas
    return rows


def _rule_summary(rule):
    """A JSON-safe copy of one get_rule_hybrids return: the recipes and their
    stored scores, without the Blueprint objects."""
    def _e(h):
        return {k: h.get(k) for k in (
            'slot', 'lineage', 'objective', 'tier', 'aug_condition', 'recipe',
            'budget', 'budget_relaxed', 'metric_name', 'lower_is_better',
            'score_other_folds', 'n_other_folds', 'fold_scores')}

    def _b(b):
        return {k: b.get(k) for k in (
            'rep', 'objective', 'tier', 'aug_condition', 'metric_name',
            'score_other_folds', 'fold_scores')}
    return {
        'rule': rule.get('rule'), 'pool_subset': rule.get('pool_subset'),
        'hybrids': [_e(h) for h in rule.get('hybrids', [])],
        'base_reps': [_b(b) for b in rule.get('base_reps', [])],
        'missing': list(rule.get('missing', [])),
        'warnings': list(rule.get('warnings', [])),
    }


def _score_pool_cell(reps, labels, groups, *, dataset_name, task_type,
                     is_clf, augmentations, config, n_sample, weight_methods,
                     importance, budget, distance,
                     best_rep_sac, best_rep_name,
                     best_rep_sac_weighted, best_rep_name_weighted,
                     family=None, entry=None, slot=None, pool_subset=None,
                     missing_reason=None):
    """Every hybrid number in this file, for ONE (slot, budget, family) cell.

    The owner's set (top 2 general + top 1 distance per model per family,
    2026-10-02): `entry` is one of the three recipes get_rule_hybrids returned
    for knn — slot 'general1', 'general2' or 'distance1' — and this cell builds
    and scores it. Before that date each cell read only the single best recipe
    of one lineage through get_fitted_pool_hybrid.

    Returns the cell dict: the recipe that was built, one block per weight
    method, and the held-out fold_reselection block with its resolved summary.
    entry=None writes the reason the slot is empty and returns.

    family: 'predictive' or 'similarity', the search family the recipe comes
        from. The two families store their scores in different metrics, so
        they are compared here on this file's own statistic, one family per
        cell.
    """
    cell = {'budget': budget, 'distance_lineage': bool(distance),
            'requested_family': family, 'slot': slot, 'weight_methods': {}}

    # best_sac is the best SAC any HYBRID arm reached, and nothing else. Seeded
    # with None so a losing hybrid reports a negative delta and a hybrid that
    # never ran reports nothing at all.
    best_sac = None
    best_method = None

    # 2026-10-02: the pool subset written is the one the read actually used.
    # It used to be written as 'predictive_and_similarity' on every cell,
    # including cells that had asked for one family only.
    cell['pool_subset'] = pool_subset
    if entry is None:
        cell['best_weight_method'] = None
        cell['best_sac'] = None
        cell['hybrid_status'] = f'slot empty: {missing_reason}'
        cell['hybrid_family'] = None
        return cell

    t0 = time.time()
    try:
        # Built on every row from the recipe's full settings and its own
        # family's objective (build_hybrid_from_entry), as
        # get_fitted_pool_hybrid built the single winner before.
        X_hybrid, fi = build_hybrid_from_entry(
            reps, labels, groups, entry, augmentations=augmentations)
    except Exception as e:
        print(f"    Build FAILED: {e}")
        cell['best_weight_method'] = None
        cell['best_sac'] = None
        cell['hybrid_status'] = f'build failed: {type(e).__name__}: {e}'
        cell['hybrid_family'] = entry.get('objective')
        return cell

    search_time = time.time() - t0
    hybrid_dim = X_hybrid.shape[1]

    # Which recipe this cell is. hybrid_search_score is the recipe's stored
    # score averaged over every fold of the pool search, in hybrid_search_metric
    # (the dataset's own metric for a predictive recipe, the five-nearest-
    # neighbour prediction score for a similarity one).
    cell['hybrid_family'] = entry.get('objective')
    cell['hybrid_family_choice'] = 'named by the caller'
    cell['hybrid_search_metric'] = entry.get('metric_name')
    cell['hybrid_search_score'] = entry.get('score_other_folds')
    cell['tier'] = entry.get('tier')
    cell['aug_condition'] = entry.get('aug_condition')
    cell['recipe'] = entry.get('recipe')
    # The number of features the recipe was searched at. Differs from `budget`
    # only when budget_relaxed is True: the distance searches stop at 500
    # features, and a larger budget is ranked over every budget they hold.
    cell['recipe_budget'] = entry.get('budget')
    cell['budget_relaxed'] = bool(entry.get('budget_relaxed'))
    print(f"    {slot}: {cell['hybrid_family']} {cell['tier']}/"
          f"{cell['aug_condition']} {cell['recipe']} "
          f"(stored {cell['hybrid_search_metric']}="
          f"{cell['hybrid_search_score']})"
          + ("  [budget relaxed]" if cell['budget_relaxed'] else ""))

    _blueprint_for_folds = entry.get('blueprint')
    _aug_condition = entry.get('aug_condition') or 'none'
    _recipe_selection = (
        "whole-dataset pool ranking: the recipe's mean over every fold of "
        "the stored pool search, which ran over every row")

    # Hybrid unweighted (same for all weight methods -- only compute once)
    D_hybrid = compute_distance_matrix(X_hybrid, standardize=True)
    sac_hybrid, _ = similarity_activity_correlation(D_hybrid, labels, n_sample, is_clf=is_clf)
    viol_hybrid = violation_rate(D_hybrid, labels, k=5, is_clf=is_clf)
    sali_hybrid = compute_sali(D_hybrid, labels, is_clf=is_clf)

    print(f"    Search done ({search_time:.1f}s), dim={hybrid_dim}")
    print(f"    Hybrid unweighted: SAC={sac_hybrid:.4f}, viol={viol_hybrid:.3f}")

    # The comparator for the delta below. Both sides weighted: the delta used to
    # subtract the best UNWEIGHTED single representation from the WEIGHTED
    # hybrid, so it carried the weighting as well as the hybrid, while the best
    # weighted single representation was computed, stored and never subtracted.
    # When importance could not be computed at all there is no weighted
    # comparator, so it falls back to the unweighted one and 'comparator' says
    # which of the two it used.
    if best_rep_name_weighted is not None:
        _cmp_sac, _cmp_name, _cmp_kind = (
            best_rep_sac_weighted, best_rep_name_weighted, 'weighted')
    else:
        _cmp_sac, _cmp_name, _cmp_kind = (
            best_rep_sac, best_rep_name, 'unweighted_importance_unavailable')
    cell['sac_delta_comparator'] = {
        'name': _cmp_name, 'sac_rho': _cmp_sac, 'kind': _cmp_kind}

    for wm in weight_methods:
        print(f"\n  Weight method: {wm}")

        # Compute importance with this weight method, apply to selected features.
        # Use `fi` (feature_info dict from get_fitted_pool_hybrid) as the column
        # layout source — it matches X_hybrid exactly. Override per-rep
        # importance_scores with the current weight method's scores.
        try:
            if wm == 'lightgbm_gain' and importance is not None:
                wm_importance = importance
            else:
                wm_importance = compute_feature_importance(reps, labels, method=wm)
            weights_flat = _hybrid_weights_from_fi(
                fi, wm_importance, X_hybrid.shape[1])
        except Exception as e:
            print(f"    Weight computation failed: {e}")
            cell['weight_methods'][wm] = {'error': str(e)}
            continue

        # Hybrid weighted. Clamp to >=0 (sqrt-weighting in the helper has no
        # guard; wm may be a signed importance method).
        D_weighted = compute_distance_matrix(
            X_hybrid, feature_weights=np.maximum(weights_flat, 0), standardize=True)
        sac_weighted, _ = similarity_activity_correlation(D_weighted, labels, n_sample, is_clf=is_clf)
        viol_weighted = violation_rate(D_weighted, labels, k=5, is_clf=is_clf)
        sali_weighted = compute_sali(D_weighted, labels, is_clf=is_clf)

        wm_result = {
            'hybrid_unweighted': {
                'dim': hybrid_dim,
                'sac_rho': sac_hybrid,
                'violation_rate': viol_hybrid,
                'sali_mean': sali_hybrid['mean'],
            },
            'hybrid_weighted': {
                'dim': hybrid_dim,
                'sac_rho': sac_weighted,
                'violation_rate': viol_weighted,
                'sali_mean': sali_weighted['mean'],
            },
            # Weighted hybrid minus weighted best single representation. Named
            # for both sides so no reader has to guess; the comparator it used
            # is in cell['sac_delta_comparator'].
            'sac_delta_weighted_vs_best_rep_weighted': sac_weighted - _cmp_sac,
            'sac_delta_comparator_kind': _cmp_kind,
            'search_time_seconds': search_time,
            'blueprint': (_blueprint_for_folds.to_dict()
                          if _blueprint_for_folds is not None else None),
            # 2026-09-12: the blueprint alone does not say which family it came
            # from, and s03/s04/s05 read this block. Name it beside it.
            'hybrid_family': cell.get('hybrid_family'),
        }

        cell['weight_methods'][wm] = wm_result

        # higher = better distance-activity correlation
        if best_sac is None or sac_weighted > best_sac:
            best_sac = sac_weighted
            best_method = wm

        print(f"    Hybrid weighted:   SAC={sac_weighted:.4f}, viol={viol_weighted:.3f}")
        print(f"    Delta SAC from best rep, both weighted ({_cmp_name}): "
              f"{sac_weighted - _cmp_sac:+.4f}")

    cell['best_weight_method'] = best_method
    cell['best_sac'] = best_sac
    if best_sac is None:
        cell['hybrid_status'] = 'no weight method produced a hybrid SAC'
        print(f"    Best hybrid:        NONE (every weight method failed)")
    else:
        print(f"    Best hybrid:        {best_method} (SAC={best_sac:.4f}, "
              f"dim={hybrid_dim})")
        print(f"    Delta vs best rep, both weighted: "
              f"{best_sac - _cmp_sac:+.4f}")

    # -------------------------------------------------------------------------
    # Cross-validated, selection-honest SAC (redesign — decision B). The block
    # above is the ORIGINAL whole-dataset statistic (kept for back-compat /
    # reference); the numbers reported as the honest result come from held-out
    # CV folds with per-fold winner re-selection. See _cv_reselected_sac docstring.
    # Persisted under the CONTRACT key "fold_reselection" (per-fold scores for BOTH
    # arms) so any reader can call fold_reselection(...) without re-running.
    # -------------------------------------------------------------------------
    n_splits = int(config.get('n_splits', 5))
    if _blueprint_for_folds is None:
        cell['fold_reselection_error'] = (
            'the pool returned no blueprint for this budget, so no per-fold '
            'hybrid could be rebuilt')
        print("  Honest CV SAC skipped: no blueprint from the pool search")
    try:
        if _blueprint_for_folds is None:
            raise RuntimeError('no blueprint for this budget')
        hr_block = _cv_reselected_sac(
            reps, labels, groups, _blueprint_for_folds,
            _aug_condition,
            augmentations, weight_methods,
            n_splits=n_splits, n_sample=n_sample, is_clf=is_clf,
            task_type=task_type)
        hr_block['recipe_selection'] = _recipe_selection
        hr_block['slot'] = slot
        # 2026-09-12: the per-fold hybrid arms below are rebuilt from the pool
        # winner's recipe, so the held-out numbers belong to whichever family
        # won. Name it inside the block a reader loads on its own.
        hr_block['hybrid_family'] = cell.get('hybrid_family')
        hr_block['pool_subset'] = cell.get('pool_subset')
        hr_block['budget'] = budget
        hr_block['distance_lineage'] = bool(distance)
        cell['fold_reselection'] = hr_block
        # Reuse the shared honest arithmetic to also store the resolved summary
        # (held-out hybrid vs best-single, direction-corrected delta) inline.
        hr_summary = fold_reselection(
            hr_block['candidates_by_model'], hr_block['baselines_by_model'],
            hr_block['lower_is_better'],
            exclude_folds=hr_exclude_folds(hr_block),
            # The fold loop re-selects the FEATURES on each fold's training
            # rows, but the allocation method, budget, importance method and
            # evaluator all came from the pool search over every row. The
            # number below is held out for the weights and not for the recipe;
            # s09 prints this label next to it.
            recipe_selection=_recipe_selection)
        cell['fold_reselection_result'] = hr_summary
        knn = hr_summary.get(_SAC_MODEL_KEY, {})
        if knn:
            print(f"\n  Honest CV SAC ({n_splits}-fold held-out): "
                  f"hybrid={knn.get('hybrid_reselected'):+.4f}, "
                  f"best_single={knn.get('best_single_reselected'):+.4f}, "
                  f"delta={knn.get('delta'):+.4f}"
                  + ("  [single-fold: OPTIMISTIC, not honest]"
                     if knn.get('single_fold') else ""))
    except Exception as e:
        # Group-aware CV can legitimately fail (too few distinct groups for
        # n_splits); never let it crash the whole experiment. Record and move on.
        print(f"  Honest CV SAC skipped: {e}")
        cell['fold_reselection_error'] = str(e)

    return cell


# =============================================================================
# EXPERIMENT
# =============================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """The fix: find_best_hybrid + importance-weighted distance improves SAC."""
    print("\n" + "=" * 70)
    print("EXPERIMENT S02: HYBRID PROPERTY-AWARE SIMILARITY")
    print("=" * 70)

    weight_methods = kwargs.get('weight_methods', ['lightgbm_gain'])
    n_sample = kwargs.get('n_sample', 5000)
    task_type = data['task_type']
    is_classification = task_type == 'classification'

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    # 'weight_methods' stays at the top level for the readers that already
    # expect it; it is filled at the end from the primary cell named below.
    # 'lineages' is the new level: one entry per distance lineage, each holding
    # one cell per budget.
    results = {'weight_methods': {}, 'per_rep': {}, 'lineages': {},
               'selection_rule': _SELECTION_RULE}

    # Compute importance for weighted baselines (once, used for all baselines)
    print("  Computing feature importance (lgbm_gain) for weighted baselines...")
    try:
        importance = compute_feature_importance(reps, labels, method='lightgbm_gain')
    except Exception as e:
        print(f"  WARNING: importance failed ({e}), weighted baselines unavailable")
        importance = None

    # Best individual rep baseline (the real-world comparison)
    best_rep_sac = -np.inf  # SAC > 0 is good; we want highest
    best_rep_name = None
    best_rep_sac_weighted = -np.inf
    best_rep_name_weighted = None
    for rep_name, X in reps.items():
        D_rep = compute_distance_matrix(X, standardize=True)
        sac_rep, _ = similarity_activity_correlation(D_rep, labels, n_sample, is_clf=is_classification)
        viol_rep = violation_rate(D_rep, labels, k=5, is_clf=is_classification)
        rep_result = {
            'dim': X.shape[1], 'sac_rho': sac_rep, 'violation_rate': viol_rep,
        }

        # Weighted version. Clamp importance to >=0 before passing: the helper
        # applies sqrt(feature_weights) with no guard, so any signed importance
        # (permutation / SHAP, which weight_methods may request) would inject
        # NaN. lightgbm_gain is non-negative so this is a no-op there; the
        # clamp matches every other similarity experiment (s04/s05/s06).
        if importance is not None and rep_name in importance:
            D_rep_w = compute_distance_matrix(
                X, feature_weights=np.maximum(importance[rep_name], 0),
                standardize=True)
            sac_rep_w, _ = similarity_activity_correlation(D_rep_w, labels, n_sample, is_clf=is_classification)
            viol_rep_w = violation_rate(D_rep_w, labels, k=5, is_clf=is_classification)
            rep_result['sac_rho_weighted'] = sac_rep_w
            rep_result['violation_rate_weighted'] = viol_rep_w
            if sac_rep_w > best_rep_sac_weighted:
                best_rep_sac_weighted = sac_rep_w
                best_rep_name_weighted = rep_name
            print(f"  {rep_name}: SAC={sac_rep:.4f}, SAC_w={sac_rep_w:.4f}, dim={X.shape[1]}")
        else:
            print(f"  {rep_name}: SAC={sac_rep:.4f}, viol={viol_rep:.3f}, dim={X.shape[1]}")

        results['per_rep'][rep_name] = rep_result
        if best_rep_name is None or sac_rep > best_rep_sac:
            best_rep_sac = sac_rep
            best_rep_name = rep_name

    results['best_individual_rep'] = {
        'name': best_rep_name, 'sac_rho': best_rep_sac,
        'dim': reps[best_rep_name].shape[1],
    }
    if best_rep_name_weighted is not None:
        results['best_individual_rep_weighted'] = {
            'name': best_rep_name_weighted, 'sac_rho': best_rep_sac_weighted,
            'dim': reps[best_rep_name_weighted].shape[1],
        }
    print(f"\n  Best individual rep: {best_rep_name} (SAC={best_rep_sac:.4f})")
    if best_rep_name_weighted:
        print(f"  Best individual rep (weighted): {best_rep_name_weighted} "
              f"(SAC_w={best_rep_sac_weighted:.4f})")

    # Concat baseline (unweighted + weighted)
    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
    D_concat = compute_distance_matrix(X_concat, standardize=True)
    sac_concat, _ = similarity_activity_correlation(D_concat, labels, n_sample, is_clf=is_classification)
    viol_concat = violation_rate(D_concat, labels, k=5, is_clf=is_classification)

    concat_result = {
        'dim': X_concat.shape[1],
        'sac_rho': sac_concat,
        'violation_rate': viol_concat,
    }

    if importance is not None:
        w_concat = np.concatenate([importance[k] for k in sorted(reps.keys())])
        # Clamp to >=0 (sqrt-weighting in the helper has no guard; see above).
        D_concat_w = compute_distance_matrix(
            X_concat, feature_weights=np.maximum(w_concat, 0), standardize=True)
        sac_concat_w, _ = similarity_activity_correlation(D_concat_w, labels, n_sample, is_clf=is_classification)
        viol_concat_w = violation_rate(D_concat_w, labels, k=5, is_clf=is_classification)
        concat_result['sac_rho_weighted'] = sac_concat_w
        concat_result['violation_rate_weighted'] = viol_concat_w
        print(f"  Concat: SAC={sac_concat:.4f}, SAC_w={sac_concat_w:.4f}, "
              f"dim={X_concat.shape[1]}")
    else:
        print(f"  Concat (reference only): SAC={sac_concat:.4f}, viol={viol_concat:.3f}, "
              f"dim={X_concat.shape[1]}")

    results['concat'] = concat_result

    # 2026-09-12, owner's decisions: sweep all eight budgets and ask both
    # distance lineages. Every hybrid statistic below is one per (lineage,
    # budget) cell, under results['lineages'][lineage]['budgets'][budget], so
    # this file now makes 2 x len(budgets) pool lookups where it made one. No
    # search runs for any of them: the finished searches on disk already hold
    # every budget, and the added work is the per-fold rebuild in each cell.
    budgets = list(kwargs.get('budgets', config.get('budgets', FULL_BUDGETS)))
    augmentations = data.get('augmentations')
    results['budgets'] = budgets

    # Which search families to score, one cell each. None is the old
    # behaviour: ask the pool for the subset and take whichever family it
    # returns, which is always 'predictive' because the two families' stored
    # scores are in different metrics and the tie-break is subset order.
    # Naming them scores each one separately and compares them on THIS file's
    # statistic, which is the only ruler both are defined on.
    families = kwargs.get('families', config.get('s02_families',
                                                 ['predictive', 'similarity']))
    # Which distance lineages to ask. Both by default.
    lineages = kwargs.get('lineages', config.get('s02_lineages',
                                                 ['general', 'general_distance']))
    results['families'] = list(families) if families else [None]
    results['lineages_asked'] = list(lineages)

    # Each (lineage, budget, family) cell is written to
    # s02_hybrid_similarity/<dataset>.partial.jsonl when it finishes and read
    # back on the next attempt (2026-09-25; _checkpoint.py). A cell reads a
    # finished search and rebuilds the hybrid on each fold; nothing in it draws
    # from a random state shared with another cell (every sampler here is
    # seeded), so a cell read back holds the numbers a fresh computation gives.
    # A cell that recorded fold_reselection_error is not stored: it is retried.
    # The per-representation baselines above are recomputed every attempt,
    # because best_rep_sac and the weighted comparator feed every cell.
    from experiments._checkpoint import (
        Checkpoint, experiment_sidecar, run_identity)
    _ck = Checkpoint(
        experiment_sidecar(output_dir, 's02_hybrid_similarity', dataset_name),
        run_identity('s02_hybrid_similarity', dataset_name, data, config,
                     kwargs=kwargs, weight_methods=weight_methods,
                     budgets=budgets, families=results['families'],
                     lineages=list(lineages),
                     # 2026-10-02: cells are per slot now (owner's set), so a
                     # sidecar written under one recipe per lineage is not read.
                     hybrid_set='top2_general_top1_distance'))

    # THE OWNER'S SET (top 2 general + top 1 distance per model per family,
    # 2026-10-02). For each budget and family, get_rule_hybrids returns knn's
    # top 2 recipes from the general searches and its top 1 from the distance
    # searches, and each of the three is built and scored as its own cell:
    # results['rule_slots'][slot]['budgets'][budget]. Before that date each
    # lineage scored only its single best recipe. A slot the pool could not
    # fill is written as a cell whose hybrid_status says why, and every reason
    # is also listed in results['rule_missing'].
    # results['lineages'] keeps the shape existing readers expect: 'general'
    # holds slot general1 and 'general_distance' holds slot distance1.
    _slot_lineage = {'general1': 'general', 'general2': 'general',
                     'distance1': 'general_distance'}
    _slots = [sl for sl, lin in _slot_lineage.items() if lin in lineages]
    results['rule_slots'] = {sl: {'distance': _slot_lineage[sl] ==
                                  'general_distance', 'budgets': {}}
                             for sl in _slots}
    results['rule_missing'] = []
    results['rule_by_budget'] = {}
    results['search_report_rows'] = {}
    for budget in budgets:
        slot_blocks = {sl: {} for sl in _slots}
        rule_b = results['rule_by_budget'].setdefault(str(budget), {})
        for _fam in (families or [None]):
            _famkey = _fam or 'pool_choice'
            _fam_kw = ({'objective': _fam} if _fam
                       else {'pool_subset': 'predictive_and_similarity'})
            try:
                rule = get_rule_hybrids(
                    reps, labels, groups, downstream_model=_SAC_MODEL_KEY,
                    budget=budget, dataset_name=dataset_name,
                    augmentations=augmentations, task_type=task_type,
                    **_fam_kw)
            except Exception as e:
                rule = {'by_objective': {}, 'hybrids': [], 'base_reps': [],
                        'missing': [f'pool read raised {type(e).__name__}: '
                                    f'{e}'],
                        'warnings': [], 'pool_subset': None}
            rule_b[_famkey] = _rule_summary(rule)
            results['rule_missing'] += [
                f'budget={budget} features, {_famkey}: {m}'
                for m in rule.get('missing', [])]
            _by_slot = {h['slot']: h for h in rule.get('hybrids', [])
                        if _fam is None or h.get('objective') == _fam}
            for sl in _slots:
                _label = f"{sl}, budget={budget} features"
                if _fam:
                    _label += f", {_fam} family"
                print(f"\n  -- {_label} --")
                _ckey = dict(slot=sl, budget=budget, family=_fam)
                if _ck.done(**_ckey):
                    slot_blocks[sl][_famkey] = _ck.get(**_ckey)
                    continue
                _entry = _by_slot.get(sl)
                _why = None
                if _entry is None:
                    _why = ('; '.join(rule.get('missing', []))
                            or 'get_rule_hybrids returned no recipe for '
                               'this slot')
                slot_blocks[sl][_famkey] = _score_pool_cell(
                    reps, labels, groups,
                    dataset_name=dataset_name, task_type=task_type,
                    is_clf=is_classification, augmentations=augmentations,
                    config=config, n_sample=n_sample,
                    weight_methods=weight_methods, importance=importance,
                    budget=budget,
                    distance=_slot_lineage[sl] == 'general_distance',
                    best_rep_sac=best_rep_sac, best_rep_name=best_rep_name,
                    best_rep_sac_weighted=best_rep_sac_weighted,
                    best_rep_name_weighted=best_rep_name_weighted,
                    family=_fam, entry=_entry, slot=sl,
                    pool_subset=rule.get('pool_subset'),
                    missing_reason=_why)
                if ('fold_reselection_error' not in slot_blocks[sl][_famkey]
                        and _entry is not None):
                    _ck.put(slot_blocks[sl][_famkey], **_ckey)
            # Rows read off the search the pool picks per lineage (top 2 base
            # representations, top 2 hybrids and hybrid deltas for six
            # models). Kept from before 2026-10-02; 0 model fits.
            for _lineage, _distance in (('general', False),
                                        ('general_distance', True)):
                if _lineage not in lineages:
                    continue
                try:
                    _srch, _ = get_best_search_for_downstream(
                        reps, labels, groups,
                        downstream_model=_SAC_MODEL_KEY, budget=budget,
                        augmentations=augmentations,
                        models=MAIN_MODELS_WITH_KNN,
                        distance=_distance, dataset_name=dataset_name,
                        task_type=task_type, **_fam_kw)
                    _rows = _search_report_rows(
                        _srch, dataset_name=dataset_name,
                        distance=_distance, budget=budget)
                except Exception as e:
                    _rows = {'absent_reason': f'{type(e).__name__}: {e}'}
                (results['search_report_rows'].setdefault(_lineage, {})
                 .setdefault(str(budget), {}))[_famkey] = _rows

        for sl in _slots:
            budget_block = slot_blocks[sl]
            # The cell kept under the budget key is the family with the better
            # HELD-OUT delta, where the winning arm and the winning single
            # representation are each re-picked on the other folds. Not
            # best_sac, the whole-dataset statistic: measured on caco2, the
            # similarity family reads +0.0263 on the whole dataset at 100
            # features and -0.0083 held out, while the predictive family reads
            # +0.0244 and +0.0260.
            def _heldout(fc):
                r = ((fc.get('fold_reselection_result') or {}).get('knn') or {})
                v = r.get('delta_mean_of_folds')
                return v if isinstance(v, (int, float)) else None

            _heldout_scored = {k: _heldout(v) for k, v in budget_block.items()}
            _heldout_scored = {k: v for k, v in _heldout_scored.items()
                               if v is not None}
            _sac_scored = {k: v['best_sac'] for k, v in budget_block.items()
                           if isinstance(v.get('best_sac'), (int, float))}
            if _heldout_scored:
                _win = max(_heldout_scored, key=_heldout_scored.get)
                _rule = ('higher held-out delta, both arms re-picked on the '
                         'other folds')
            elif _sac_scored:
                _win = max(_sac_scored, key=_sac_scored.get)
                _rule = ('higher whole-dataset sac_rho — no family produced a '
                         'held-out delta')
            else:
                _win = next(iter(budget_block))
                _rule = 'only candidate'
            cell = dict(budget_block[_win])
            cell['by_family'] = budget_block
            cell['family_chosen'] = _win
            cell['family_choice_rule'] = (
                _rule if len(budget_block) > 1 else 'only one family scored')
            cell['family_heldout_deltas'] = _heldout_scored
            cell['family_wholedata_sac'] = _sac_scored
            if len(budget_block) > 1:
                print(f"    {sl} b={budget}: family chosen: {_win} ({_rule})")
            results['rule_slots'][sl]['budgets'][str(budget)] = cell

    # results['lineages'] in its old shape: general = slot general1,
    # general_distance = slot distance1.
    for _lineage, _sl in (('general', 'general1'),
                          ('general_distance', 'distance1')):
        if _sl in results['rule_slots']:
            results['lineages'][_lineage] = {
                'distance': _lineage == 'general_distance',
                'slot': _sl,
                'budgets': results['rule_slots'][_sl]['budgets']}

    # Back-compat top level. s09, firstpass_notready.py, quick_hybrid_summary.py
    # and make_supervisor_plots.py all read results['best_sac'],
    # results['fold_reselection_result'] and results['weight_methods'] off the top
    # level, and every s02 file already on disk has them there. They now mirror
    # ONE named cell — the general lineage at budget 100, the only cell this
    # file scored before today — and results['primary_cell'] names it. Every
    # other cell lives only under results['lineages'].
    _primary_budget = 100 if 100 in budgets else budgets[0]
    results['primary_cell'] = {'lineage': 'general', 'budget': _primary_budget}
    _primary = (results['lineages'].get('general') or {}).get(
        'budgets', {}).get(str(_primary_budget), {})
    for _key in ('weight_methods', 'best_weight_method', 'best_sac',
                 'hybrid_status', 'hybrid_family', 'hybrid_family_choice',
                 'hybrid_search_metric', 'hybrid_search_score', 'pool_subset',
                 'pool_by_objective', 'top3_base_reps',
                 'top3_base_reps_metric', 'top3_base_reps_family',
                 'sac_delta_comparator', 'fold_reselection',
                 'fold_reselection_result', 'fold_reselection_error'):
        if _key in _primary:
            results[_key] = _primary[_key]

    # Summary
    print(f"\n  Summary:")
    print(f"    Best individual rep: {best_rep_name} (SAC={best_rep_sac:.4f}, "
          f"dim={reps[best_rep_name].shape[1]})")
    if best_rep_name_weighted:
        print(f"    Best individual rep, weighted: {best_rep_name_weighted} "
              f"(SAC_w={best_rep_sac_weighted:.4f})")
    for _lineage, _lin in results['rule_slots'].items():
        for _b in budgets:
            _cell = _lin['budgets'].get(str(_b), {})
            _bs = _cell.get('best_sac')
            print(f"    {_lineage} b={_b}: best hybrid SAC="
                  f"{'none' if _bs is None else f'{_bs:.4f}'}"
                  f" ({_cell.get('hybrid_status', 'ok')})")

    # Sequence-based cliff sweep (antibody datasets + nucleotide) — independent
    # of any representation choice. Same sweep as s01 for cross-experiment
    # comparability; CDR indices cached per-dataset for repeat runs.
    cliff_results = run_antibody_cliff_analysis(
        data, is_clf=is_classification, dataset_name=dataset_name)
    if cliff_results is not None:
        results['_sequence_cliffs'] = cliff_results

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('S02: Hybrid Property-Aware Similarity',
                                     has_delta=True, has_augment=True)
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 's02_hybrid_similarity', args.dataset)
