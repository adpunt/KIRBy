#!/usr/bin/env python3
"""C03: Cross-dataset comparison of optimal hybrid configurations.

Paper: Complexity
Conclusion: Optimal allocation/importance/budget choices CO-VARY across
datasets in interpretable ways (correlations with complexity metrics).
This does NOT directly demonstrate transfer — see c24 for true source→target
blueprint transfer.  c03 finds the native-optimal configuration on each
dataset and analyses cross-dataset similarities (which configs cluster
together; which complexity dimensions predict optimal budget).

Dependencies: c02 (reads JSON from multiple datasets, or runs own searches).

Loads multiple datasets, computes complexity profiles (Lorena et al. 2018),
runs find_best_hybrid on each, and compares optimal configurations.  Also
evaluates a small set of novel regression models (SVR, ElasticNet) on the
native-optimal hybrid for regression datasets only (skipped for classification
— SVR/ElasticNet produce meaningless AUC otherwise).  Includes budget
variance decomposition and complexity↔budget correlation.

2026-09-12, decided (EXPERIMENTS_FINALIZED, c03): the complexity-descriptor
correlation is computed per downstream model — one Spearman coefficient per
(descriptor, model), not one overall.  The winning budget for a
(dataset, model) is the budget whose best feature set scores highest for that
model, which is the definition the model-by-dataset table already used, so both
questions now use one definition.  c03 read the pool once, for the support
vector model with a random forest fallback, so four of the five downstream
models had no data behind their correlation; it now reads every downstream
model's search.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
from copy import deepcopy

import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
# Ranking metrics need graded scores: an AUROC read off hard 0/1 labels
# collapses to the mean of sensitivity and specificity (hybrid.py:2661).
from kirby.find_best_hybrid import _predict_for_metric

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy, cv_split, _cv_split,
    compute_dataset_complexity, get_domain_metric,
    find_best_hybrid, create_hybrid, apply_feature_selection,
    get_best_for_downstream, get_best_search_for_downstream,
    build_fold_reselection_input, fold_reselection, hr_exclude_folds,
    top_base_reps, top_hybrids, resolve_comparison_models,
    DISTANCE_BUDGETS, DISTANCE_EVALUATORS, DOWNSTREAM_MODELS,
    get_rule_hybrids, build_hybrid_from_entry,
)
from kirby.hybrid import apply_augmentation_selection


def _winner_per_fold(all_results, model_name, lower_is_better, n_folds):
    """Recipe winner for each held-out fold, picked on the other folds only.

    Every row in a search's all_results carries its own per-fold scores. For
    held-out fold i the winner is the row with the best mean over the other
    folds; that row's features are then rebuilt on fold i's training rows. A row
    is considered only when it scored all n_folds folds, so every row is ranked
    on the same folds.

    Returns {fold index: row}. Empty when no row of this model scored every
    fold — the caller then keeps the whole-dataset winner and says so.
    """
    rows = []
    for r in (all_results or []):
        if r.get('model') != model_name:
            continue
        fs = r.get('fold_scores')
        if not fs or len(fs) != n_folds:
            continue
        if not all(s is not None and np.isfinite(float(s)) for s in fs):
            continue
        rows.append(r)
    if not rows:
        return {}
    pick = min if lower_is_better else max
    winners = {}
    for i in range(n_folds):
        others = [j for j in range(n_folds) if j != i]
        winners[i] = pick(
            rows,
            key=lambda r: float(np.mean([r['fold_scores'][j] for j in others])))
    return winners


def _rule_set_per_fold(data, ds_name, model_name, metric_name):
    """The owner's set for one downstream model, re-picked per held-out fold.

    Owner's set (2026-10-02): top 2 general + top 1 distance hybrids per
    model, per family, and the model's top 2 single representations. c03
    until then read the distance searches only and compared one hybrid.

    Each fold k's hybrids and base representations are ranked on the other
    folds' mean (get_rule_hybrids fold_i=k); the held-out number is the score
    the stored search recorded on fold k. Reads only, no fits -- the same
    arithmetic as the per-fold re-pick above, applied to every slot.

    Returns (block, missing, whole, raw_per_fold): block has 'whole_dataset' (the set ranked on
    every fold), 'per_fold' (one entry per fold), 'by_slot' (held-out score
    per fold for each slot and each base rank) and 'budget_relaxed'.
    """
    common = dict(downstream_model=model_name, budget=None,
                  dataset_name=ds_name,
                  augmentations=data.get('augmentations'),
                  task_type=data['task_type'], metric=metric_name, data=data)
    whole = get_rule_hybrids(data['reps'], data['labels'], data['groups'],
                             fold_i=None, **common)
    missing = list(whole.get('missing') or [])

    def _hyb_row(h):
        return {'slot': h.get('slot'), 'lineage': h.get('lineage'),
                'objective': h.get('objective'), 'tier': h.get('tier'),
                'aug_condition': h.get('aug_condition'),
                'recipe': h.get('recipe'), 'budget': h.get('budget'),
                'budget_relaxed': h.get('budget_relaxed'),
                'score_other_folds': h.get('score_other_folds'),
                'score_this_fold': h.get('score_this_fold')}

    def _base_row(b, rank):
        return {'rank': rank, 'rep': b.get('rep'),
                'objective': b.get('objective'),
                'score_other_folds': b.get('score_other_folds'),
                'score_this_fold': b.get('score_this_fold')}

    n_pool_folds = max([len(h.get('fold_scores') or [])
                        for h in whole['hybrids']] or [0])
    per_fold, by_slot, raw_per_fold = [], {}, []
    budget_relaxed = any(h.get('budget_relaxed') for h in whole['hybrids'])
    for k in range(n_pool_folds):
        rk = get_rule_hybrids(data['reps'], data['labels'], data['groups'],
                              fold_i=k, n_outer_folds=n_pool_folds, **common)
        budget_relaxed = budget_relaxed or any(
            h.get('budget_relaxed') for h in rk['hybrids'])
        raw_per_fold.append(list(rk['hybrids']))
        hy = [_hyb_row(h) for h in rk['hybrids']]
        ba = [_base_row(b, i + 1) for i, b in enumerate(rk['base_reps'])]
        per_fold.append({'fold': k, 'hybrids': hy, 'base_reps': ba,
                         'missing': rk.get('missing') or []})
        for h in hy:
            key = f"{h['objective']}|{h['slot']}"
            by_slot.setdefault(key, [None] * n_pool_folds)[k] = \
                h['score_this_fold']
        for b in ba:
            key = f"base{b['rank']}"
            by_slot.setdefault(key, [None] * n_pool_folds)[k] = \
                b['score_this_fold']
    return {
        'rule': whole.get('rule'),
        'whole_dataset': {
            'hybrids': [_hyb_row(h) for h in whole['hybrids']],
            'base_reps': [_base_row(b, i + 1)
                          for i, b in enumerate(whole['base_reps'])]},
        'per_fold': per_fold,
        'by_slot': by_slot,
        'n_pool_folds': n_pool_folds,
        'budget_relaxed': budget_relaxed,
    }, missing, whole, raw_per_fold

def run(datasets, config, output_dir, dataset_name=None, **kwargs):
    """Cross-context analysis across multiple datasets.

    Parameters
    ----------
    datasets : dict
        {dataset_name: data_dict} for each dataset.
    config : dict
        Config dict (fast or full).
    output_dir : str
        Output directory path.
    dataset_name : str, optional
        Ignored (we use datasets dict keys instead).
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT C03: CROSS-CONTEXT ANALYSIS")
    print("=" * 70)

    n_folds = config['n_splits']
    mode = 'intermediary' if config['n_splits'] == 1 else 'full'

    results = {}

    for ds_name, data in datasets.items():
        print(f"\n  --- {ds_name} ---")
        metric_name_ds, _, _ = get_domain_metric(data)

        # Compute dataset complexity profile
        print(f"  Computing complexity metrics...")
        complexity = compute_dataset_complexity(data['reps'], data['labels'])
        mle_str = f", MLE_ID={complexity['mle_id']:.1f}" if complexity.get('mle_id') else ""
        print(f"    erank={complexity['erank']:.1f}, PR={complexity['participation_ratio']:.1f}, "
              f"n/d={complexity['n_over_d']:.2f}, n/erank={complexity['n_over_erank']:.1f}{mle_str}")
        print(f"    C1={complexity['C1']:.3f}, C2={complexity['C2']:.3f}, "
              f"L1={complexity['L1']:.3f}, S3={complexity['S3']:.3f}")

        # Run hybrid search
        print(f"  Running find_best_hybrid...")
        start = time.time()

        try:
            from sklearn.svm import SVR
            from sklearn.linear_model import ElasticNet

            novel_models = {
                'svr': SVR(kernel='rbf', C=1.0),
                'elasticnet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
            }

            # 2026-09-12 — DECIDED. The complexity-descriptor correlation is
            # computed per downstream model: one Spearman coefficient per
            # (descriptor, model), not one overall. The winning budget for a
            # (dataset, model) is the budget whose best feature set scores
            # highest for that model.
            # What changed: c03 read the distance-tier pool ONCE, asking for
            # downstream_model='svm' and falling back to 'rf', so four of the
            # five downstream models had no data behind their correlation. It
            # now reads every downstream model's search — one pool read per
            # model, budget=None so each model's own best budget is returned.
            per_model = {}
            searches_by_model = {}
            # A downstream model this dataset has no pool search for, and any
            # model whose top-2 selection raised, are written into the result
            # below instead of only being printed.
            no_pool_search = []
            selection_failures = []
            for model_name in DOWNSTREAM_MODELS:
                search_m, meta_m = get_best_search_for_downstream(
                    data['reps'], data['labels'], data['groups'],
                    downstream_model=model_name, budget=None,
                    augmentations=data.get('augmentations'),
                    dataset_name=ds_name,
                    task_type=data['task_type'], metric=metric_name_ds,
                    distance=True,
                )
                if meta_m is None:
                    no_pool_search.append(
                        {'downstream_model': model_name,
                         'reason': f'no pool search on {ds_name} for this '
                                   f'downstream model, so it has no row in '
                                   f'best_by_model and no top-2 comparison'})
                    print(f"    {model_name}: no pool search on {ds_name}")
                    continue
                cfg_m = meta_m['best_config']
                bp_m = cfg_m['blueprint']
                searches_by_model[model_name] = (search_m, meta_m)
                per_model[model_name] = {
                    'budget': cfg_m.get('budget', getattr(bp_m, 'budget', None)),
                    'score': cfg_m.get('score_mean'),
                    'dim': cfg_m.get('n_features'),
                    'allocation': bp_m.allocation_method,
                    'importance': bp_m.importance_method,
                    'internal_evaluator': bp_m.internal_evaluator,
                    'tier': meta_m.get('tier'),
                    'aug_condition': meta_m.get('aug_condition'),
                    'metric_name': meta_m.get('metric_name'),
                    'lower_is_better': meta_m.get('lower_is_better'),
                    # 'selection_scope' is no longer recorded: the recipe is
                    # re-picked for each held-out fold below, so the scope of
                    # every pick is one fold and there is nothing left for the
                    # pool helper's whole-dataset label to describe.
                }
                # ADDED 2026-09-18. c03 named no base representation at all:
                # every row above describes the winning hybrid, and the
                # `baselines` dump at c03:362 is unranked and covers one
                # reference model. These two keys name the top 2 base
                # representations and the top 2 hybrids for THIS downstream
                # model, read out of the search that is already in hand.
                #
                # 0 new model fits. Both read rows the search already stored.
                #
                # 'budget', 'score' and 'dim' above are untouched, so the
                # correlation at c03:472 keeps reading the same rank-1 row it
                # reads today. Nothing already in a c03 result file moves.
                # Two changes, 2026-09-28.
                #
                # First: the models compared. The loop above asks for the five
                # names in DOWNSTREAM_MODELS. The pool read passes
                # distance=True, so the search it returns scored six downstream
                # models -- the five plus k-nearest-neighbours. The comparison
                # is now built from the models the search scored, so
                # k-nearest-neighbours gets its top 2 base representations and
                # its top 2 hybrids like every other model. No model's rows are
                # ever read under another model's name.
                #
                # Second: the two halves were in ONE try block, so an exception
                # raised while ranking the base representations threw away the
                # hybrid ranking as well, and the result file then held neither
                # half. Each half has its own try now, and whichever one fails
                # is written into 'top2_selection_failures' with its reason.
                _searched, _ = resolve_comparison_models(search_m, None)
                _top2_by_model = {}
                for _cm in _searched:
                    _entry = {}
                    try:
                        _base = top_base_reps(
                            search_m, _cm, 2, require_full=False,
                            context=f'c03 {ds_name}/{_cm}')
                        _entry['base_reps'] = [
                            (e['rep'], e['score']) for e in _base['selected']]
                    except Exception as _e:
                        selection_failures.append(
                            {'pool_read_for': model_name,
                             'downstream_model': _cm, 'half': 'base_reps',
                             'reason': f'{type(_e).__name__}: {_e}'})
                        print(f"    {_cm}: top-2 base representations "
                              f"unavailable ({type(_e).__name__}: {_e})")
                    try:
                        _hyb = top_hybrids(
                            search_m, _cm, 2, require_full=False,
                            context=f'c03 {ds_name}/{_cm}')
                        _entry['hybrids'] = [
                            {'label': e['label'], 'score_mean': e['score_mean'],
                             'budget': e['budget'],
                             'n_features': e['n_features']}
                            for e in _hyb['selected']]
                    except Exception as _e:
                        selection_failures.append(
                            {'pool_read_for': model_name,
                             'downstream_model': _cm, 'half': 'hybrids',
                             'reason': f'{type(_e).__name__}: {_e}'})
                        print(f"    {_cm}: top-2 hybrid configurations "
                              f"unavailable ({type(_e).__name__}: {_e})")
                    _top2_by_model[_cm] = _entry
                per_model[model_name]['top2_by_searched_model'] = _top2_by_model
                # The two keys readers already fetch keep holding this
                # downstream model's own halves.
                _own = _top2_by_model.get(model_name) or {}
                if 'base_reps' in _own:
                    per_model[model_name]['top2_base_reps'] = _own['base_reps']
                if 'hybrids' in _own:
                    per_model[model_name]['top2_hybrids'] = _own['hybrids']
                if model_name not in _searched:
                    selection_failures.append(
                        {'pool_read_for': model_name,
                         'downstream_model': model_name, 'half': 'both',
                         'reason': f'the search returned for {model_name} '
                                   f'scored {_searched} and not {model_name!r}'})
                _pm = per_model[model_name]
                print(f"    {model_name:<10} budget={_pm['budget']} "
                      f"{metric_name_ds}={_pm['score']:.4f} "
                      f"features={_pm['dim']} "
                      f"({_pm['tier']}/{_pm['aug_condition']})")

            # 2026-09-12 — DECIDED: the recipe is re-picked per held-out fold.
            # For each fold the search's own per-fold candidate scores are
            # ranked on the other four folds, that fold's winner is taken, and
            # its score on the held-out fold is read. The single
            # representations go through the identical rule, so both sides pay
            # the same tax for being a maximum. Pure arithmetic on per-fold
            # scores the search already stored — no fits.
            reselected_by_model = {}
            for _mname, (_search_m, _) in searches_by_model.items():
                try:
                    _blk = build_fold_reselection_input(_search_m)
                    _hr = fold_reselection(
                        _blk.get('candidates_by_model', {}),
                        _blk.get('baselines_by_model', {}),
                        _blk.get('lower_is_better'),
                        metric_name=_blk.get('metric_name'),
                        exclude_folds=hr_exclude_folds(_blk))
                except Exception as _e:
                    print(f"    {_mname}: per-fold re-pick unavailable - {_e}")
                    continue
                if _mname not in _hr:
                    continue
                reselected_by_model[_mname] = _hr[_mname]
                _d = _hr[_mname].get('delta_mean_of_folds')
                if _d is None or not np.isfinite(_d):
                    print(f"    {_mname:<10} per-fold re-pick: no fold scored "
                          f"on both sides")
                else:
                    print(f"    {_mname:<10} per-fold re-pick: hybrid minus "
                          f"best single rep = {_d:+.4f} over "
                          f"{_hr[_mname].get('n_paired_folds', 0)} folds")

            # Owner's set (2026-10-02): top 2 general + top 1 distance
            # hybrids per model, per family, plus the model's top 2 single
            # representations. Everything above reads the distance searches
            # only and keeps its keys; this block adds the general searches'
            # top 2. Held-out scores per fold, picked on the other folds.
            rule_by_model = {}
            rule_missing = []
            rule_whole_by_model = {}
            for _mname in DOWNSTREAM_MODELS:
                try:
                    _blk_r, _miss_r, _whole_r, _raw_r = _rule_set_per_fold(
                        data, ds_name, _mname, metric_name_ds)
                except Exception as _e:
                    rule_missing.append(
                        f'{_mname}: the owner\'s set could not be read '
                        f'({type(_e).__name__}: {_e})')
                    continue
                rule_by_model[_mname] = _blk_r
                rule_whole_by_model[_mname] = (_whole_r, _raw_r)
                rule_missing += [f'{_mname}: {m}' for m in _miss_r]
                for _key, _vals in _blk_r['by_slot'].items():
                    _v = [v for v in _vals if v is not None]
                    if _v:
                        print(f"    {_mname:<10} {_key:<28} held-out "
                              f"{metric_name_ds} mean over {len(_v)} folds "
                              f"= {np.mean(_v):.4f}")

            # Best model on this dataset is now the argmax over the per-model
            # reads above, in the dataset's own metric and its own direction.
            _lower_flags = [v['lower_is_better'] for v in per_model.values()
                            if v['lower_is_better'] is not None]
            lower_is_better = bool(_lower_flags[0]) if _lower_flags else False
            _scored = {m: v for m, v in per_model.items()
                       if v['score'] is not None and np.isfinite(v['score'])}
            if _scored:
                best_model_name = (min if lower_is_better else max)(
                    _scored, key=lambda m: _scored[m]['score'])
            else:
                best_model_name = None

            # The two novel regressors (SVR + ElasticNet) are evaluated on ONE
            # blueprint. The proxy is unchanged: the support-vector winner,
            # falling back to random forest — both are linear/kernel models
            # whose feature-budget preference is closest to the SVM evaluator's,
            # and neither is a precompute downstream model in its own right. It
            # is now taken from the per-model reads above instead of costing a
            # second pair of pool calls. The same search supplies `baselines`
            # and `performance_matrix`, recorded under 'search_source_model'.
            ref_model = next((m for m in ('svm', 'rf')
                              if m in searches_by_model), None)
            if ref_model is None and searches_by_model:
                ref_model = next(iter(searches_by_model))
            search_result, meta = searches_by_model.get(ref_model, (None, None))
            best = meta['best_config'] if meta else None
            metric_name, metric_fn, _ = get_domain_metric(data)
            augs = data.get('augmentations')
            novel_results = {}
            novel_fold_recipes = []
            if best:
                bp = best['blueprint']
                # Guard: novel models (SVR, ElasticNet) are regressors only;
                # silently fitting them on classification labels produces
                # meaningless AUC scores or hard crashes that the exception
                # handler used to swallow. Skip novel-model eval for clf tasks.
                if data['task_type'] == 'classification':
                    novel_results = {'_skipped': 'classification task; novel models are regressors'}
                    bp_eval = None
                else:
                    novel_results = {}
                    bp_eval = bp
                # Trust meta['aug_condition'] (the actual pool-winning condition)
                # not bp.augmentation_strategy (can be stale per the 2026-04-21
                # cache-collision bug).
                aug_strat = meta.get('aug_condition', 'none') if meta else 'none'
                if bp_eval is None:
                    novel_iter = []
                else:
                    novel_iter = list(novel_models.items())

                # The recipe the two novel regressors are fitted on is re-picked
                # for every held-out fold, from the reference search's own
                # per-fold scores: rank the candidates on the other folds, take
                # this fold's winner, rebuild its features on this fold's
                # training rows. bp_eval — the winner over every row — is the
                # fallback for a fold with no candidate that scored all folds,
                # and 'novel_model_fold_recipes' below says which was used.
                sample_X = next(iter(data['reps'].values()))
                novel_splits = list(cv_split(
                    data['groups'], n_folds, sample_X, data['labels'], mode=mode))
                fold_winners = _winner_per_fold(
                    getattr(search_result, 'all_results', None),
                    ref_model, lower_is_better, len(novel_splits))
                novel_fold_recipes = []
                for _i in range(len(novel_splits)):
                    _row = fold_winners.get(_i)
                    _bp_i = (_row.get('blueprint') if _row else None) or bp_eval
                    novel_fold_recipes.append({
                        'fold': _i,
                        'repicked_on_other_folds': _row is not None,
                        'allocation': getattr(_bp_i, 'allocation_method', None),
                        'importance': getattr(_bp_i, 'importance_method', None),
                        'internal_evaluator': getattr(_bp_i, 'internal_evaluator', None),
                        'budget': getattr(_bp_i, 'budget', None),
                    })

                for novel_name, novel_model in novel_iter:
                    try:
                        novel_scores = []

                        for _fold_i, (train_idx, val_idx) in enumerate(novel_splits):
                            _row = fold_winners.get(_fold_i)
                            bp_fold = (_row.get('blueprint') if _row else None) or bp_eval
                            # Restrict to the reps the blueprint actually
                            # selected — create_hybrid otherwise re-scores all
                            # reps and may not pick bp.reps for top_k alloc.
                            _bp_reps = set(getattr(bp_fold, 'reps', []) or data['reps'].keys())
                            train_reps = {k: v[train_idx] for k, v in data['reps'].items() if k in _bp_reps}
                            val_reps = {k: v[val_idx] for k, v in data['reps'].items() if k in _bp_reps}
                            y_train = data['labels'][train_idx]
                            y_val = data['labels'][val_idx]

                            train_augs = (
                                {k: v[train_idx] for k, v in augs.items()}
                                if augs and aug_strat != 'none' else None)
                            X_train, fi = create_hybrid(
                                train_reps, y_train,
                                groups=data['groups'][train_idx],
                                allocation_method=bp_fold.allocation_method,
                                budget=bp_fold.budget,
                                importance_method=bp_fold.importance_method,
                                internal_evaluator=bp_fold.internal_evaluator,
                                augmentations=train_augs,
                                augmentation_strategy=aug_strat,
                            )
                            X_val = apply_feature_selection(val_reps, fi)
                            aug_info = fi.get('augmentation_info')
                            if aug_info and augs and aug_strat != 'none':
                                val_augs = {k: v[val_idx] for k, v in augs.items()}
                                X_aug_val = apply_augmentation_selection(val_augs, aug_info)
                                if X_aug_val is not None:
                                    X_val = np.hstack([X_val, X_aug_val])

                            scaler = StandardScaler()
                            X_train_s = scaler.fit_transform(X_train)
                            X_val_s = scaler.transform(X_val)

                            m = deepcopy(novel_model)
                            m.fit(X_train_s, y_train)
                            novel_scores.append(metric_fn(y_val, _predict_for_metric(m, X_val_s, metric_fn)))

                        novel_results[novel_name] = float(np.mean(novel_scores))
                        print(f"    {novel_name} on {ds_name}: {metric_name}={np.mean(novel_scores):.4f}")
                    except Exception as e:
                        print(f"    {novel_name} on {ds_name}: FAILED - {e}")

            # Owner's set (2026-10-02): the two novel regressors are also
            # fitted on each of the reference model's three rule hybrids
            # (general1, general2, distance1, per family), re-picked per
            # held-out fold on the other folds. 'novel_models' above keeps the
            # earlier distance-search per-fold winner.
            novel_results_rule = {}
            if (best and data['task_type'] != 'classification'
                    and ref_model in rule_whole_by_model):
                _whole_ref, _raw_ref = rule_whole_by_model[ref_model]
                if len(_raw_ref) != len(novel_splits):
                    rule_missing.append(
                        f'novel regressors: the pool holds {len(_raw_ref)} '
                        f'folds, c03 splits into {len(novel_splits)}; folds '
                        f'without a per-fold pick use the whole-dataset set')
                for _h0 in _whole_ref['hybrids']:
                    _skey = f"{_h0.get('objective')}|{_h0.get('slot')}"
                    for novel_name, novel_model in novel_models.items():
                        try:
                            _sc = []
                            for _fold_i, (train_idx, val_idx) in enumerate(novel_splits):
                                _ent = _h0
                                if _fold_i < len(_raw_ref):
                                    _ent = next(
                                        (h for h in _raw_ref[_fold_i]
                                         if h.get('slot') == _h0.get('slot')
                                         and h.get('objective') == _h0.get('objective')),
                                        _h0)
                                _bp_reps = set(getattr(_ent['blueprint'], 'reps', [])
                                               or data['reps'].keys())
                                _tr = {k: v[train_idx] for k, v in data['reps'].items() if k in _bp_reps}
                                _va = {k: v[val_idx] for k, v in data['reps'].items() if k in _bp_reps}
                                _use_aug = bool(augs) and _ent.get('aug_condition', 'none') != 'none'
                                X_train, fi = build_hybrid_from_entry(
                                    _tr, data['labels'][train_idx],
                                    data['groups'][train_idx], _ent,
                                    augmentations=({k: v[train_idx] for k, v in augs.items()}
                                                   if _use_aug else None))
                                X_val = apply_feature_selection(_va, fi)
                                aug_info = fi.get('augmentation_info')
                                if aug_info and _use_aug:
                                    X_aug_val = apply_augmentation_selection(
                                        {k: v[val_idx] for k, v in augs.items()}, aug_info)
                                    if X_aug_val is not None:
                                        X_val = np.hstack([X_val, X_aug_val])
                                scaler = StandardScaler()
                                X_train_s = scaler.fit_transform(X_train)
                                X_val_s = scaler.transform(X_val)
                                m = deepcopy(novel_model)
                                m.fit(X_train_s, data['labels'][train_idx])
                                _sc.append(metric_fn(
                                    data['labels'][val_idx],
                                    _predict_for_metric(m, X_val_s, metric_fn)))
                            novel_results_rule.setdefault(_skey, {})[novel_name] = {
                                'score_mean': float(np.mean(_sc)),
                                'fold_scores': [float(x) for x in _sc],
                                'slot': _h0.get('slot'),
                                'lineage': _h0.get('lineage'),
                                'objective': _h0.get('objective'),
                                'tier': _h0.get('tier'),
                                'budget_relaxed': _h0.get('budget_relaxed'),
                            }
                            print(f"    {novel_name} on {ds_name} [{_skey}]: "
                                  f"{metric_name}={np.mean(_sc):.4f}")
                        except Exception as e:
                            rule_missing.append(
                                f'novel regressor {novel_name} on {_skey}: '
                                f'{type(e).__name__}: {e}')
                            print(f"    {novel_name} on {ds_name} [{_skey}]: FAILED - {e}")

            elapsed = time.time() - start

            _bm = per_model.get(best_model_name) or {}
            ds_result = {
                'n_samples': len(data['labels']),
                'elapsed_seconds': elapsed,
                'complexity': complexity,
                'metric': metric_name_ds,
                'lower_is_better': lower_is_better,
                'best_model': best_model_name,
                'best_score': _bm.get('score'),
                'best_allocation': _bm.get('allocation'),
                'best_importance': _bm.get('importance'),
                'best_budget': _bm.get('budget'),
                'best_dim': _bm.get('dim'),
                'baselines': search_result.baselines if search_result is not None else None,
                # 2026-09-12 — DECIDED: one entry per downstream model, each
                # from that model's own pool read, each holding the budget
                # whose best feature set scored highest for that model. This is
                # what both the correlation and the model-by-dataset table read.
                'best_by_model': per_model,
                # A downstream model with no pool search on this dataset, so no
                # row in 'best_by_model' at all.
                'models_dropped': no_pool_search,
                # A (pool read, downstream model, half) whose top-2 ranking
                # raised, with the exception text.
                'top2_selection_failures': selection_failures,
                # Hybrid minus best single representation with the recipe and
                # the representation both re-picked on the other folds, one
                # entry per downstream model. Read 'delta_mean_of_folds'.
                'reselected_by_model': reselected_by_model,
                'search_source_model': ref_model,
                # Which recipe each held-out fold's novel-model score was
                # built from, and whether that fold got its own re-pick.
                'novel_model_fold_recipes': novel_fold_recipes,
                'performance_matrix': (search_result.performance_matrix
                                       if search_result is not None else None),
                'novel_models': novel_results if best else {},
                # Owner's set (2026-10-02): per downstream model, top 2
                # general + top 1 distance hybrids per family and the top 2
                # single representations, held-out score per fold.
                'rule_by_model': rule_by_model,
                'novel_models_rule': novel_results_rule,
                'rule_missing': rule_missing,
            }
            results[ds_name] = ds_result

            if best_model_name is not None:
                print(f"    Best: {best_model_name}, "
                      f"{metric_name_ds}={ds_result['best_score']:.4f}, "
                      f"alloc={ds_result['best_allocation']}, "
                      f"imp={ds_result['best_importance']}, "
                      f"budget={ds_result['best_budget']}")
            else:
                print(f"    No downstream model had a pool search on {ds_name}.")

        except Exception as e:
            print(f"    {ds_name}: FAILED - {e}")
            results[ds_name] = {'error': str(e)}

    # Cross-dataset comparison table
    print("\n  CROSS-CONTEXT COMPARISON:")
    print(f"  {'Dataset':<10} {'N':<6} {'Best Model':<12} {'Importance':<15} {'Budget':<8} {'Score':<8}")
    print("  " + "-" * 65)
    for ds_name, r in results.items():
        if 'error' in r:
            continue
        if r.get('best_model') is None:
            print(f"  {ds_name:<10} {r['n_samples']:<6} no pool search for any "
                  f"downstream model")
            continue
        print(f"  {ds_name:<10} {r['n_samples']:<6} {r['best_model']:<12} "
              f"{r['best_importance']:<15} {r['best_budget']:<8} {r['best_score']:.4f}")

    # --- Dataset complexity vs optimal hybrid dimension ---
    valid = {ds: r for ds, r in results.items()
             if 'error' not in r and r.get('best_budget') is not None}
    if len(valid) >= 2:
        print("\n  DATASET COMPLEXITY vs OPTIMAL BUDGET:")
        print(f"  {'Dataset':<10} {'erank':>7} {'PR':>7} {'n/erank':>8} {'C1':>6} {'C2':>6} "
              f"{'L1':>6} {'S3':>6} {'Budget':>7} {'Dim':>5}")
        print("  " + "-" * 82)

        complexity_keys = ['erank', 'participation_ratio', 'n_over_erank', 'C1', 'C2', 'L1', 'S3']
        ds_names = sorted(valid.keys())

        for ds in ds_names:
            c = valid[ds]['complexity']
            print(f"  {ds:<10} {c['erank']:>7.1f} {c['participation_ratio']:>7.1f} "
                  f"{c['n_over_erank']:>8.1f} {c['C1']:>6.3f} {c['C2']:>6.3f} "
                  f"{c['L1']:>6.3f} {c['S3']:>6.3f} "
                  f"{valid[ds]['best_budget']:>7} {valid[ds]['best_dim']:>5}")

        # 2026-09-12 — DECIDED: one Spearman coefficient per (descriptor,
        # downstream model). One overall coefficient per descriptor, computed
        # from whichever model won the dataset, is gone; so is the sentence
        # that read a direction off the single erank coefficient, because there
        # is no longer one coefficient for it to come from. The per-model erank
        # coefficients are printed below with the number of datasets behind
        # each. Every coefficient lands under '_complexity_correlation_by_model'.
        corr_models = sorted({m for ds in ds_names
                              for m in (valid[ds].get('best_by_model') or {})})
        correlations_by_model = {}
        print("\n  Spearman correlation, complexity descriptor -> winning budget.")
        print("  One coefficient per (descriptor, downstream model). One point is "
              "one dataset; n counts the datasets that have a winning budget for "
              "that model.")
        for model in corr_models:
            model_ds = [
                ds for ds in ds_names
                if (valid[ds].get('best_by_model') or {}).get(model, {})
                .get('budget') is not None
            ]
            if len(model_ds) < 2:
                print(f"    {model:<12} {len(model_ds)} dataset(s) with a winning "
                      f"budget — no coefficient")
                correlations_by_model[model] = {
                    'n_datasets': len(model_ds), 'datasets': model_ds,
                    'descriptors': {},
                }
                continue

            m_budgets = np.array(
                [float(valid[ds]['best_by_model'][model]['budget'])
                 for ds in model_ds])
            _dim_vals = [valid[ds]['best_by_model'][model].get('dim')
                         for ds in model_ds]
            m_dims = (np.array([float(d) for d in _dim_vals])
                      if all(d is not None for d in _dim_vals) else None)

            per_desc = {}
            print(f"    {model}  (n={len(model_ds)} datasets: "
                  f"{', '.join(model_ds)})")
            for key in complexity_keys:
                values = np.array([float(valid[ds]['complexity'][key])
                                   for ds in model_ds])
                if np.std(values) < 1e-10:
                    continue
                rho, p = spearmanr(values, m_budgets)
                entry = {'rho_vs_budget': float(rho), 'p_vs_budget': float(p),
                         'n_datasets': len(model_ds)}
                # The same descriptor against the feature count the winning
                # budget actually reached.
                if m_dims is not None:
                    rho_dim, p_dim = spearmanr(values, m_dims)
                    entry['rho_vs_dim'] = float(rho_dim)
                    entry['p_vs_dim'] = float(p_dim)
                per_desc[key] = entry
                _dim_str = (f", features: rho={entry['rho_vs_dim']:+.3f} "
                            f"(p={entry['p_vs_dim']:.3f})"
                            if 'rho_vs_dim' in entry else "")
                print(f"      {key:<20} -> budget: rho={rho:+.3f} (p={p:.3f})"
                      f"{_dim_str}")
            correlations_by_model[model] = {
                'n_datasets': len(model_ds), 'datasets': model_ds,
                'descriptors': per_desc,
            }

        results['_complexity_correlation_by_model'] = correlations_by_model

        # erank against the winning budget, one coefficient per downstream model.
        print("\n  erank -> winning budget, one coefficient per downstream model:")
        for model in corr_models:
            entry = (correlations_by_model.get(model, {})
                     .get('descriptors', {}).get('erank'))
            n_ds = correlations_by_model.get(model, {}).get('n_datasets', 0)
            if entry is None:
                print(f"    {model:<12} no coefficient (n={n_ds} datasets)")
            else:
                print(f"    {model:<12} rho={entry['rho_vs_budget']:+.3f} "
                      f"(p={entry['p_vs_budget']:.3f}), n={n_ds} datasets")

        # Per-model optimal budget comparison across datasets.
        # 2026-09-12 — DECIDED: the winning budget for a (dataset, model) is the
        # budget whose best feature set scores highest for that model. That is
        # what these rows already showed; they now read the per-model pool reads
        # above, the same numbers the correlations use, rather than one search
        # read for a single downstream model.
        model_names = set()
        for ds in ds_names:
            model_names.update(valid[ds].get('best_by_model', {}).keys())
        if model_names:
            print("\n  PER-MODEL OPTIMAL BUDGET (cross-dataset):")
            print(f"  {'Model':<12}" + "".join(f"{ds:>14}" for ds in ds_names))
            print("  " + "-" * (12 + 14 * len(ds_names)))
            for model in sorted(model_names):
                row = f"  {model:<12}"
                for ds in ds_names:
                    bm = valid[ds].get('best_by_model', {}).get(model)
                    if bm:
                        row += f"b={bm['budget']:>3} score={bm['score']:.3f} "
                    else:
                        row += f"{'--':>14}"
                print(row)

            # MODEL x DATASET INTERACTION ANALYSIS
            model_list = sorted(model_names)
            budget_matrix = []
            score_matrix = []
            for model in model_list:
                model_budgets = []
                model_scores = []
                for ds in ds_names:
                    bm = valid[ds].get('best_by_model', {}).get(model)
                    if bm:
                        model_budgets.append(bm['budget'])
                        model_scores.append(bm['score'])
                    else:
                        model_budgets.append(np.nan)
                        model_scores.append(np.nan)
                budget_matrix.append(model_budgets)
                score_matrix.append(model_scores)
            budget_matrix = np.array(budget_matrix, dtype=float)
            score_matrix = np.array(score_matrix, dtype=float)

            # Budget variance decomposition
            if budget_matrix.shape[0] >= 2 and budget_matrix.shape[1] >= 2:
                valid_mask = ~np.isnan(budget_matrix)
                if valid_mask.all():
                    grand_mean = np.nanmean(budget_matrix)
                    row_means = np.nanmean(budget_matrix, axis=1)
                    col_means = np.nanmean(budget_matrix, axis=0)
                    ss_model = len(ds_names) * np.sum((row_means - grand_mean) ** 2)
                    ss_dataset = len(model_list) * np.sum((col_means - grand_mean) ** 2)
                    ss_total = np.sum((budget_matrix - grand_mean) ** 2)
                    ss_interaction = ss_total - ss_model - ss_dataset

                    if ss_total > 0:
                        print(f"\n  BUDGET VARIANCE DECOMPOSITION:")
                        print(f"    Model effect:       {ss_model/ss_total*100:5.1f}% of variance")
                        print(f"    Dataset effect:     {ss_dataset/ss_total*100:5.1f}% of variance")
                        print(f"    Interaction effect: {ss_interaction/ss_total*100:5.1f}% of variance")

                    results['_budget_variance'] = {
                        'ss_model': float(ss_model),
                        'ss_dataset': float(ss_dataset),
                        'ss_interaction': float(ss_interaction),
                        'ss_total': float(ss_total),
                        'pct_model': float(ss_model / ss_total * 100) if ss_total > 0 else 0,
                        'pct_dataset': float(ss_dataset / ss_total * 100) if ss_total > 0 else 0,
                        'pct_interaction': float(ss_interaction / ss_total * 100) if ss_total > 0 else 0,
                    }

                    # Does budget range differ across models?
                    print(f"\n  MODEL DIMENSION SENSITIVITY:")
                    print(f"  {'Model':<12} {'Budget Range':<15} {'Budget Std':<12} {'Score Range':<12}")
                    print("  " + "-" * 51)
                    for i, model in enumerate(model_list):
                        b_row = budget_matrix[i]
                        score_row = score_matrix[i]
                        if not np.any(np.isnan(b_row)):
                            print(f"  {model:<12} {int(b_row.min())}-{int(b_row.max()):<10} "
                                  f"{np.std(b_row):>8.1f}    {np.ptp(score_row):>8.4f}")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('C03: Cross-Dataset Transfer',
                                     has_delta=True, has_augment=True)
    parser.add_argument('--datasets', type=str, nargs='+', default=['logd', 'caco2', 'herg_ki'],
                        help='Dataset names to compare (at least 2)')
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Datasets: {args.datasets}")
    print(f"Reps: {args.reps}")

    # Load all datasets
    all_datasets = {}
    for ds_name in args.datasets:
        print(f"\nLoading {ds_name}...")
        try:
            all_datasets[ds_name] = load_data(
                ds_name, args.reps,
                delta=getattr(args, 'delta', False),
                augment=getattr(args, 'augment', False))
        except Exception as e:
            print(f"  FAILED: {e}")

    if len(all_datasets) < 2:
        print("Need at least 2 datasets for cross-context analysis.")
        sys.exit(1)

    results = run(all_datasets, config, args.output)
    save_experiment_results(results, args.output, 'c03_cross_dataset_transfer', 'multi')
