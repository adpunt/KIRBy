#!/usr/bin/env python3
"""C14: Cross-Domain Summary — hybrid improvement over best single rep.

Paper: Complexity
Conclusion: KIRBy hybrid consistently improves over best individual rep
    across domains (molecular, antibody, TCGA).
Dependencies: None (self-contained, runs find_best_hybrid per task).

For each task:
  1. Load data + reps via load_data (supports molecular, antibody, TCGA)
  2. Evaluate each individual rep with best model (domain-aware metric)
  3. Run nested_cv_evaluate: a recipe search inside each outer training fold
  4. Record: best_single_rep_name, best_single_score, hybrid_score, delta, pct_improvement
  5. Track per-fold scores for statistical testing

Aggregate across all tasks:
  - Wilcoxon signed-rank test (paired: hybrid vs best_single per task)
  - Win rate: fraction of tasks where hybrid > best single
  - Mean/median delta and pct improvement
  - Per-domain breakdown

Usage:
    python c14_cross_domain_summary.py --tasks logd,caco2,herg --reps all --mode full
    python c14_cross_domain_summary.py --tasks logd,flab_thermo_tresanco,tcga_luad_vs_lusc --mode intermediary
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import datetime
import numpy as np
from scipy.stats import wilcoxon

from experiments._shared import (
    load_data, get_domain_metric, get_model_factories,
    cv_split,
    save_experiment_results, convert_numpy,
    get_config,
    _local_aug_conditions,
    extract_top3_base_reps,
    top_base_reps, hybrid_deltas, resolve_comparison_models,
    DOWNSTREAM_MODELS,
    INTERNAL_EVALUATORS,
)
from kirby.find_best_hybrid import _predict_for_metric, nested_cv_evaluate
from experiments.complexity.complexity_helpers import DATASET_REPS

EXPERIMENT_ID = 'c14_cross_domain_summary'


# =========================================================================
# Per-task evaluation
# =========================================================================

def _evaluate_task(task_name, rep_preset, config):
    """Single-representation baselines plus nested CV for one task.

    Returns dict with:
        task, domain, metric_name, lower_is_better,
        best_single_rep, best_single_score, best_single_fold_scores,
        hybrid_score, hybrid_fold_scores, reselected_best_single_score,
        delta, pct_improvement, n_reps, n_samples

    `delta` is delta_mean_of_folds from nested_cv_evaluate: each outer fold's
    recipe and single representation are picked on that fold's training rows,
    then both are scored on the fold neither saw. The best_single_* keys are
    the older same-fold numbers and are kept for the per-representation table
    only; they are a maximum read on the folds that picked it.
    """
    print(f"\n  --- Task: {task_name} ---")

    # Load data
    data = load_data(task_name, rep_preset)
    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    domain = data.get('domain', 'molecular')
    task_type = data['task_type']

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    model_names = config['models']
    n_samples = len(labels)

    print(f"    Domain: {domain}, Metric: {metric_name} "
          f"({'lower better' if lower_is_better else 'higher better'})")
    print(f"    {n_samples} samples, {len(reps)} reps, {n_folds}-fold CV")

    # -----------------------------------------------------------------
    # Load augmentation reps from DATASET_REPS (if configured)
    # -----------------------------------------------------------------
    aug_rep_names = DATASET_REPS.get(task_name, {}).get('augmentation', [])
    # KIRBY_LOCAL_AUG_CONDITIONS (dirty local run 2026-09-28): the nested CV
    # below hard-coded 'uncapped' whenever augmentation representations exist,
    # so a local run cut to 'none' still searched 'uncapped' here and loaded
    # augmentation representations that may not be cached. Routed through the
    # same filter the pool uses; unset on the cluster, where it returns
    # ['uncapped'] unchanged.
    _c14_aug_strategy = (_local_aug_conditions(['uncapped'])[0]
                         if aug_rep_names else 'none')
    if _c14_aug_strategy == 'none':
        aug_rep_names = []
    aug_dict = None
    if aug_rep_names:
        missing = [r for r in aug_rep_names if r not in reps]
        if missing:
            print(f"    Loading {len(missing)} augmentation reps: {missing}")
            aug_data = load_data(task_name, rep_preset='all', rep_names=missing)
            aug_dict = {}
            for rn, arr in aug_data['reps'].items():
                if int(np.isnan(arr).sum()) > 0:
                    arr = np.nan_to_num(arr, nan=0.0)
                aug_dict[rn] = arr
        else:
            aug_dict = {r: reps[r] for r in aug_rep_names}
        print(f"    Augmentations ({len(aug_dict)}): "
              f"{', '.join(f'{k}={v.shape[1]}' for k, v in aug_dict.items())}")

    # -----------------------------------------------------------------
    # 1. Evaluate each individual rep with each model, track fold scores
    # -----------------------------------------------------------------
    # Restrict single-rep baseline to DOWNSTREAM_MODELS so the comparison
    # against the hybrid pool (which also iterates DOWNSTREAM_MODELS below) is
    # symmetric — config['models'] is 11 models, the hybrid pool was only 5
    # (missing xgboost+ridge), which biased best_single in favour of models
    # the hybrid could not match.
    factories = get_model_factories(DOWNSTREAM_MODELS, task_type=task_type,
                                    n_samples=n_samples)

    best_single_score = None
    best_single_rep = None
    best_single_model = None
    best_single_fold_scores = None
    best_single_fold_ids = None
    all_rep_results = {}

    for rep_name, X in sorted(reps.items()):
        rep_best_score = None
        rep_best_model = None
        rep_best_folds = None

        rep_best_fold_ids = None

        for model_name, factory in factories.items():
            # Collect per-fold scores (not just mean/std). fold_ids records
            # which folds actually produced a score — a fold that raises is
            # skipped, so fold_scores can be shorter than n_folds and the
            # surviving fold set can differ between reps and between models.
            fold_scores = []
            fold_ids = []
            for fold_id, (train_idx, val_idx) in enumerate(
                    cv_split(groups, n_folds, X, labels, mode=mode)):
                try:
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X[train_idx])
                    X_val = scaler.transform(X[val_idx])

                    model = factory()
                    model.fit(X_train, labels[train_idx])
                    score = metric_fn(labels[val_idx], _predict_for_metric(model, X_val, metric_fn))
                    fold_scores.append(float(score))
                    fold_ids.append(fold_id)
                except Exception:
                    continue

            if not fold_scores:
                continue

            mean_score = float(np.mean(fold_scores))

            # Track best model for this rep
            if rep_best_score is None or _is_better(mean_score, rep_best_score,
                                                     lower_is_better):
                rep_best_score = mean_score
                rep_best_model = model_name
                rep_best_folds = fold_scores
                rep_best_fold_ids = fold_ids

        if rep_best_score is not None:
            all_rep_results[rep_name] = {
                'best_model': rep_best_model,
                'score': rep_best_score,
                'fold_scores': rep_best_folds,
                'fold_ids': rep_best_fold_ids,
                'n_folds_used': len(rep_best_folds),
            }

            # Track overall best single rep
            if best_single_score is None or _is_better(
                    rep_best_score, best_single_score, lower_is_better):
                best_single_score = rep_best_score
                best_single_rep = rep_name
                best_single_model = rep_best_model
                best_single_fold_scores = rep_best_folds
                best_single_fold_ids = rep_best_fold_ids

        print(f"      {rep_name:<25} best_model={rep_best_model:<10} "
              f"score={rep_best_score:.4f}" if rep_best_score is not None
              else f"      {rep_name:<25} FAILED")

    if best_single_score is None:
        print(f"    WARNING: no single rep succeeded for {task_name}")
        return None

    print(f"    Best single rep: {best_single_rep} ({best_single_model}) "
          f"= {best_single_score:.4f}")

    # -----------------------------------------------------------------
    # 1b. Paired re-scoring across reps on the folds every rep survived.
    # The means above average over whatever folds survived for each rep
    # individually, so reps with different surviving fold sets are compared
    # on different data. common_fold_ids is the intersection of the fold ids
    # of every rep's selected model; the *_common_folds quantities below
    # average only those folds. The unpaired quantities are kept unchanged
    # under their existing keys.
    # -----------------------------------------------------------------
    fold_counts = {k: v['n_folds_used'] for k, v in all_rep_results.items()}
    common_fold_ids = None
    for v in all_rep_results.values():
        ids = set(v['fold_ids'])
        common_fold_ids = ids if common_fold_ids is None else (common_fold_ids & ids)
    common_fold_ids = sorted(common_fold_ids) if common_fold_ids else []

    all_rep_scores_common_folds = {}
    best_single_rep_cf = None
    best_single_model_cf = None
    best_single_score_cf = None
    best_single_fold_scores_cf = None

    for rep_name, v in (sorted(all_rep_results.items())
                        if common_fold_ids else []):
        by_fold = dict(zip(v['fold_ids'], v['fold_scores']))
        folds_cf = [by_fold[f] for f in common_fold_ids]
        mean_cf = float(np.mean(folds_cf))
        v['score_common_folds'] = mean_cf
        all_rep_scores_common_folds[rep_name] = mean_cf
        if best_single_score_cf is None or _is_better(mean_cf,
                                                      best_single_score_cf,
                                                      lower_is_better):
            best_single_score_cf = mean_cf
            best_single_rep_cf = rep_name
            best_single_model_cf = v['best_model']
            best_single_fold_scores_cf = folds_cf

    n_distinct_counts = len(set(fold_counts.values()))
    print(f"    Surviving folds per rep: min={min(fold_counts.values())}, "
          f"max={max(fold_counts.values())}, "
          f"{n_distinct_counts} distinct count(s), "
          f"{len(common_fold_ids)} fold(s) common to all {len(fold_counts)} reps")
    if best_single_rep_cf is not None:
        print(f"    Best single rep on common folds only: {best_single_rep_cf} "
              f"({best_single_model_cf}) = {best_single_score_cf:.4f}")

    # -----------------------------------------------------------------
    # 2. Hybrid vs best single representation, nested.
    # -----------------------------------------------------------------
    # Each outer fold runs its own recipe search on that fold's training rows,
    # then both arms are scored on the fold neither search saw. The previous
    # version read the winner's own mean off a search that ranked ~600 recipes
    # on the folds it then reported, against a single-representation side that
    # was a maximum over about ten — so the two sides were not comparable.
    print(f"    Nested CV: a recipe search inside each outer training fold...")

    try:
        ncv = nested_cv_evaluate(
            reps, labels, groups,
            n_outer_folds=n_folds,
            dataset_name=task_name,
            models=DOWNSTREAM_MODELS,
            budgets=config['budgets'],
            metric=metric_name,
            task_type=task_type,
            augmentations=aug_dict if aug_dict else None,
            augmentation_strategy=_c14_aug_strategy if aug_dict else 'none',
            internal_evaluators=INTERNAL_EVALUATORS,
            verbose=False,
        )
    except Exception as e:
        print(f"    WARNING: nested_cv_evaluate failed for {task_name}: {e}")
        return None

    if not ncv.by_model:
        print(f"    WARNING: nested_cv_evaluate produced no model for {task_name}")
        return None

    # Headline model. Fixed 2026-09-28 (dirty local run): it used to be the
    # model whose HELD-OUT hybrid mean was best, i.e. picked on the outer folds
    # the delta is then reported on. Only the hybrid side gets that maximum
    # (the single side is whatever that model scores), so the pick favoured
    # the hybrid by up to a maximum over three models. Now the pick reads
    # search_mean: the inner search's own score, computed on outer-training
    # rows only. A model is eligible only if its hybrid scored on some outer
    # fold; that is a check that a number exists, not a read of its value.
    # The old pick is kept under 'hybrid_model_by_outer_score'.
    def _finite(v):
        return v is not None and np.isfinite(v)

    _worst = np.inf if lower_is_better else -np.inf

    def _key_on(field):
        def _k(m):
            v = ncv.by_model[m].get(field)
            return float(v) if _finite(v) else _worst
        return _k

    def _pick(cands, field):
        return (min(cands, key=_key_on(field)) if lower_is_better
                else max(cands, key=_key_on(field)))

    _eligible = [m for m in ncv.by_model
                 if _finite(ncv.by_model[m].get('hybrid_mean'))]
    if not _eligible:
        print(f"    WARNING: no outer-fold hybrid score for {task_name}")
        return None
    model_by_outer_score = _pick(_eligible, 'hybrid_mean')
    if any(_finite(ncv.by_model[m].get('search_mean')) for m in _eligible):
        best_model_name = _pick(_eligible, 'search_mean')
        model_selected_on = ('search_mean: the inner search score on '
                             'outer-training rows')
    else:
        best_model_name = model_by_outer_score
        model_selected_on = ('hybrid_mean on the outer folds (no model had a '
                             'finite search_mean); optimistic for the hybrid')
    ncv_best = ncv.by_model[best_model_name]

    top3 = extract_top3_base_reps(ncv)
    if top3:
        print(f"    Top-3 base reps (by RF): {', '.join(f'{n}={sc:.4f}' for n, sc in top3)}")

    hybrid_score = ncv_best.get('hybrid_mean')
    if hybrid_score is None or np.isnan(hybrid_score):
        print(f"    WARNING: no outer-fold hybrid score for {task_name}")
        return None
    hybrid_fold_scores = list(ncv_best.get('hybrid_fold_scores') or [])
    # Both sides of this come from outer folds neither arm's search saw.
    # delta_mean_of_folds is the paired quantity: the fold-by-fold difference,
    # averaged. Do NOT subtract the two means instead — they diverge as soon as
    # either arm drops a fold.
    reselected_best_single = ncv_best.get('best_single_mean')
    if reselected_best_single is None:
        reselected_best_single = float('nan')
    reselected_best_single_fold_scores = list(
        ncv_best.get('best_single_fold_scores') or [])
    delta = ncv_best.get('delta_mean_of_folds')
    if delta is None:
        delta = float('nan')
    hybrid_wins = bool(not np.isnan(delta) and delta > 0)

    # pct improvement against the held-out single-representation score. Floor
    # the denominator so near-zero baselines (e.g. MCC ~ 0.02 on hard genomic
    # tasks) don't inflate pct to hundreds of percent for tiny improvements.
    if np.isnan(reselected_best_single) or np.isnan(delta):
        pct = float('nan')
    else:
        pct = delta / max(abs(reselected_best_single), 0.05) * 100

    # The same-fold single-representation numbers from section 1 stay under
    # their own keys; they are a maximum taken on the folds they are read on,
    # so they are not what the delta above is computed against.
    if best_single_score_cf is None:
        delta_cf = None
        pct_cf = None
    else:
        _denom_cf = max(abs(best_single_score_cf), 0.05)
        if lower_is_better:
            delta_cf = best_single_score_cf - hybrid_score
        else:
            delta_cf = hybrid_score - best_single_score_cf
        pct_cf = delta_cf / _denom_cf * 100

    # -----------------------------------------------------------------
    # Added 2026-09-18, beside everything above; no number above is re-derived.
    #
    # This file named ONE winning representation per task (best_single_rep) and
    # printed the top 3 by RandomForest only. Nothing recorded the runner-up
    # under the other 4 downstream models, so a task where two representations
    # were within rounding of each other read the same as one where the winner
    # was clear. top_base_reps ranks on ncv.baselines, which
    # NestedCVResult builds from baselines_by_model
    # (find_best_hybrid.py:9734-9741): the mean over the outer folds, one number
    # per representation per downstream model, the same held-out rows the
    # hybrid arm is scored on.
    #
    # hybrid_deltas puts two reference numbers under the hybrid instead of one:
    #   delta_vs_source_mean -- the hybrid against the mean of the
    #     representations its recipe actually gave features to.
    #   delta_vs_best_rep -- the hybrid against the single best representation
    #     for that downstream model.
    # 'delta' above is neither: it is delta_mean_of_folds, the fold-by-fold
    # difference against the representation re-picked inside each outer fold.
    # All three stay in the file under their own keys.
    #
    # hybrid= is passed explicitly, and it is ncv.best_by_model[m], whose
    # score_mean is the outer-fold hybrid mean.
    #
    # Both read stored rows: 0 new model fits, 0 new folds.
    # require_full=False so a downstream model the nested CV never scored
    # returns a short list instead of raising and losing the whole task.
    #
    # ⛔ top_hybrids AND select_model_rep_pairs ARE DELIBERATELY NOT CALLED
    # HERE, AND ADDING THEM WAS TRIED AND REVERTED ON 2026-09-28.
    #
    # The two halves would not be scored on the same rows. top_base_reps reads
    # ncv.baselines: the mean over the outer folds, each representation scored on
    # rows no search saw. top_hybrids reads ncv.all_results, and on a
    # NestedCVResult those are the INNER searches' own CV scores
    # (find_best_hybrid.py:10186-10196), which are selection-side numbers pooled
    # over the outer folds. A base number from the first against a hybrid number
    # from the second is a delta between two different things.
    #
    # So c14 carries no ranked pair of hybrids on purpose. The held-out hybrid
    # number per downstream model is ncv.best_by_model[m]['score_mean'], which is
    # what 'hybrid_deltas_by_model' and 'delta' above are built from, and it is
    # one configuration per model rather than two.
    #
    # WHAT DID CHANGE ON 2026-09-28: the model list. It comes from the nested CV
    # now, not from DOWNSTREAM_MODELS. A model the nested CV never scored -- its
    # own skip policy, or a sample-count limit -- used to fall out of this loop
    # with nothing written down; it is recorded in 'models_dropped', and no
    # other model's rows stand in for it.
    _ncv_best_by_model = ncv.best_by_model
    top2_base_reps_by_model = {}
    hybrid_deltas_by_model = {}
    comparison_failures = []
    comparison_models, models_dropped = resolve_comparison_models(
        ncv, list(DOWNSTREAM_MODELS))
    for _name, _why in models_dropped:
        print(f"    {_name} is not in the top-2 comparison: {_why}")
    for _m in comparison_models:
        try:
            top2_base_reps_by_model[_m] = top_base_reps(
                ncv, _m, 2, require_full=False,
                context=f'c14 {task_name} outer-fold means')
        except Exception as e:
            comparison_failures.append(
                {'downstream_model': _m, 'half': 'base_reps',
                 'reason': f'{type(e).__name__}: {e}'})
            print(f"    WARNING: top 2 base representations for {_m} failed: {e}")
        _bp_row = _ncv_best_by_model.get(_m)
        if _bp_row is None:
            comparison_failures.append(
                {'downstream_model': _m, 'half': 'hybrid_deltas',
                 'reason': 'the nested CV recorded no hybrid row for this '
                           'downstream model on best_by_model'})
            continue
        try:
            hybrid_deltas_by_model[_m] = hybrid_deltas(
                ncv, _m, hybrid=_bp_row,
                context=f'c14 {task_name} outer-fold means')
        except Exception as e:
            comparison_failures.append(
                {'downstream_model': _m, 'half': 'hybrid_deltas',
                 'reason': f'{type(e).__name__}: {e}'})
            print(f"    WARNING: the two hybrid deltas for {_m} failed: {e}")

    print(f"    Hybrid (held out): {hybrid_score:.4f} vs best single "
          f"{reselected_best_single:.4f}  "
          f"(delta_mean_of_folds={delta:+.4f}, {pct:+.1f}%, "
          f"{'WIN' if hybrid_wins else 'LOSS'}, model={best_model_name})")

    return {
        'task': task_name,
        'domain': domain,
        'task_type': task_type,
        'metric_name': metric_name,
        'lower_is_better': lower_is_better,
        'n_reps': len(reps),
        'n_samples': n_samples,
        # n_folds is the fold count requested in the config, not the count that
        # produced any given score. n_folds_used_per_rep holds the counts that
        # were actually averaged.
        'n_folds': n_folds,
        'n_folds_configured': n_folds,
        'n_folds_used_per_rep': fold_counts,
        'n_common_folds': len(common_fold_ids),
        'common_fold_ids': common_fold_ids,
        'best_single_rep': best_single_rep,
        'best_single_model': best_single_model,
        'best_single_score': best_single_score,
        'best_single_fold_scores': best_single_fold_scores,
        'best_single_fold_ids': best_single_fold_ids,
        'best_single_score_basis': (
            'mean over the folds that survived for that representation; '
            'surviving fold sets may differ between representations'),
        # Paired counterpart of the four keys above.
        'best_single_rep_common_folds': best_single_rep_cf,
        'best_single_model_common_folds': best_single_model_cf,
        'best_single_score_common_folds': best_single_score_cf,
        'best_single_fold_scores_common_folds': best_single_fold_scores_cf,
        'best_single_score_common_folds_basis': (
            f'mean over the {len(common_fold_ids)} fold(s) present for every '
            'representation; the model per representation is the one picked on '
            "that representation's own surviving folds, and the winning "
            'representation is picked on the folds it is scored on'),
        # Outer-fold hybrid score: every outer fold's recipe was searched on
        # that fold's training rows only.
        'hybrid_score': hybrid_score,
        'hybrid_fold_scores': hybrid_fold_scores,
        'hybrid_model': best_model_name,
        'hybrid_model_selected_on': model_selected_on,
        # The model the old rule picked: best hybrid mean on the outer folds
        # the delta is reported on. Kept for comparison only.
        'hybrid_model_by_outer_score': model_by_outer_score,
        # The single-representation arm of the delta, also scored on outer
        # folds, with its winner picked on each fold's training rows.
        'reselected_best_single_score': reselected_best_single,
        'reselected_best_single_fold_scores': reselected_best_single_fold_scores,
        'reselected_best_single_reps_per_fold': ncv_best.get('best_single_reps_per_fold'),
        'n_paired_outer_folds': ncv_best.get('n_paired_folds'),
        'per_fold_delta': ncv_best.get('per_fold_delta'),
        # How much the inner search's own mean exceeded the outer-fold score:
        # what picking the winner on the folds it is reported on was worth.
        'search_mean': ncv_best.get('search_mean'),
        'selection_bias': ncv_best.get('bias'),
        'nested_cv_by_model': {
            m: {k: v for k, v in e.items() if k != 'blueprints_per_fold'}
            for m, e in ncv.by_model.items()
        },
        'hybrid_config': {
            'model': best_model_name or '',
            # Recipe from the first outer fold that produced one — nested CV
            # picks a recipe per outer fold, so this names one of them, while
            # hybrid_score is the mean across all of them.
            'recipe_is_from_fold': 'first outer fold with a winner',
            'n_features': (ncv.best_by_model.get(best_model_name, {})
                           .get('n_features')),
            'allocation_method': (ncv.best_by_model.get(best_model_name, {})
                                  .get('allocation_method') or ''),
            'importance_method': (ncv.best_by_model.get(best_model_name, {})
                                  .get('importance_method') or ''),
            'budget': (ncv.best_by_model.get(best_model_name, {})
                       .get('budget')),
            'internal_evaluator': (ncv.best_by_model.get(best_model_name, {})
                                   .get('internal_evaluator') or ''),
        },
        # The paired fold-by-fold difference, averaged. Report this one.
        'delta': delta,
        'delta_basis': 'delta_mean_of_folds from nested_cv_evaluate',
        'pct_improvement': pct,
        'hybrid_wins': hybrid_wins,
        'delta_vs_best_single_common_folds': delta_cf,
        'pct_improvement_vs_best_single_common_folds': pct_cf,
        'all_rep_scores': {
            k: v['score'] for k, v in all_rep_results.items()
        },
        'all_rep_scores_common_folds': all_rep_scores_common_folds,
        'all_rep_n_folds_used': fold_counts,
        'top3_base_reps': [(n, s) for n, s in top3] if top3 else [],
        # See the block above the print. {downstream model: top_base_reps(...)},
        # ranked on the outer-fold mean. top3_base_reps stays as it was: it is
        # RandomForest only and carries no tie or drop record.
        'top2_base_reps_by_model': top2_base_reps_by_model,
        'top2_base_reps_score_basis': (
            'the mean over the outer folds, each representation scored on rows '
            'no search saw (NestedCVResult.baselines)'),
        # Which downstream models the three keys above cover, which of
        # DOWNSTREAM_MODELS the nested CV never scored, and which half of which
        # model's comparison raised.
        'comparison_models': comparison_models,
        'models_dropped': [{'downstream_model': _n, 'reason': _w}
                           for _n, _w in models_dropped],
        'comparison_failures': comparison_failures,
        # {downstream model: {'delta_vs_source_mean', 'delta_vs_best_rep', ...}},
        # both positive when the hybrid is ahead. Not the same quantity as
        # 'delta' above, which is delta_mean_of_folds.
        'hybrid_deltas_by_model': hybrid_deltas_by_model,
    }


def _is_better(a, b, lower_is_better):
    """Return True if a is better than b under the given metric direction."""
    if lower_is_better:
        return a < b
    return a > b


# =========================================================================
# Aggregation
# =========================================================================

def _aggregate(task_results):
    """Aggregate per-task results into summary statistics + Wilcoxon test.

    Returns dict with:
        per_task, summary, per_domain, wilcoxon_test
    """
    print("\n" + "=" * 70)
    print("  AGGREGATE SUMMARY")
    print("=" * 70)

    deltas = []
    pcts = []
    wins = 0
    total = len(task_results)

    # Per-domain tracking
    domain_results = {}

    for r in task_results:
        deltas.append(r['delta'])
        pcts.append(r['pct_improvement'])
        if r['hybrid_wins']:
            wins += 1

        domain = r['domain']
        if domain not in domain_results:
            domain_results[domain] = {'deltas': [], 'pcts': [], 'wins': 0,
                                       'tasks': []}
        domain_results[domain]['deltas'].append(r['delta'])
        domain_results[domain]['pcts'].append(r['pct_improvement'])
        domain_results[domain]['tasks'].append(r['task'])
        if r['hybrid_wins']:
            domain_results[domain]['wins'] += 1

    deltas_arr = np.array(deltas)
    pcts_arr = np.array(pcts)

    summary = {
        'n_tasks': total,
        'win_rate': wins / total if total > 0 else 0,
        'wins': wins,
        'losses': total - wins,
        'mean_delta': float(np.mean(deltas_arr)),
        'median_delta': float(np.median(deltas_arr)),
        'std_delta': float(np.std(deltas_arr)),
        'mean_pct_improvement': float(np.mean(pcts_arr)),
        'median_pct_improvement': float(np.median(pcts_arr)),
        'std_pct_improvement': float(np.std(pcts_arr)),
        'delta_basis': ('per-task delta against the best single rep scored on '
                        'its own surviving folds'),
    }

    # Same statistics over the tasks whose single-rep side was re-scored on the
    # folds common to every rep. Tasks with no common fold are left out, so
    # n_tasks_common_folds can be below n_tasks.
    deltas_cf = [r['delta_vs_best_single_common_folds'] for r in task_results
                 if r.get('delta_vs_best_single_common_folds') is not None]
    pcts_cf = [r['pct_improvement_vs_best_single_common_folds']
               for r in task_results
               if r.get('pct_improvement_vs_best_single_common_folds') is not None]
    if deltas_cf:
        d_cf_arr = np.array(deltas_cf)
        wins_cf = int(np.sum(d_cf_arr > 0))
        summary.update({
            'n_tasks_common_folds': len(deltas_cf),
            'wins_common_folds': wins_cf,
            'win_rate_common_folds': wins_cf / len(deltas_cf),
            'mean_delta_common_folds': float(np.mean(d_cf_arr)),
            'median_delta_common_folds': float(np.median(d_cf_arr)),
            'std_delta_common_folds': float(np.std(d_cf_arr)),
            'mean_pct_improvement_common_folds': (
                float(np.mean(pcts_cf)) if pcts_cf else None),
            'delta_common_folds_basis': (
                'per-task delta against the best single rep scored only on the '
                'folds present for every rep; the hybrid side is not restricted'),
        })

    print(f"\n  Win rate: {wins}/{total} ({summary['win_rate']*100:.1f}%)")
    print(f"  Mean delta:   {summary['mean_delta']:+.4f} "
          f"(+/- {summary['std_delta']:.4f})")
    print(f"  Median delta: {summary['median_delta']:+.4f}")
    print(f"  Mean %improvement:   {summary['mean_pct_improvement']:+.1f}%")
    print(f"  Median %improvement: {summary['median_pct_improvement']:+.1f}%")
    print(f"  (single-rep side scored on each rep's own surviving folds)")
    if 'n_tasks_common_folds' in summary:
        print(f"\n  Single-rep side scored on folds common to every rep:")
        print(f"    {summary['wins_common_folds']}/"
              f"{summary['n_tasks_common_folds']} tasks with delta > 0, "
              f"mean_delta={summary['mean_delta_common_folds']:+.4f}, "
              f"median_delta={summary['median_delta_common_folds']:+.4f}")

    # -----------------------------------------------------------------
    # Wilcoxon signed-rank test
    # -----------------------------------------------------------------
    wilcoxon_result = {}
    if total >= 3:
        # Test: are the deltas significantly > 0?
        # (i.e., does hybrid consistently improve over best single rep?)
        try:
            stat, p = wilcoxon(deltas_arr, alternative='greater')
            wilcoxon_result = {
                'statistic': float(stat),
                'p_value': float(p),
                'significant_005': p < 0.05,
                'significant_001': p < 0.01,
                'n': total,
            }
            print(f"\n  Wilcoxon signed-rank (one-sided, hybrid > single):")
            print(f"    statistic={stat:.4f}, p={p:.4f} "
                  f"({'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'})")
        except Exception as e:
            wilcoxon_result = {'error': str(e)}
            print(f"\n  Wilcoxon test failed: {e}")
    else:
        wilcoxon_result = {'error': f'Need >= 3 tasks, got {total}'}
        print(f"\n  Wilcoxon test skipped (need >= 3 tasks, got {total})")

    # -----------------------------------------------------------------
    # Per-domain breakdown
    # -----------------------------------------------------------------
    per_domain = {}
    print(f"\n  Per-domain breakdown:")
    for domain, dr in sorted(domain_results.items()):
        d_arr = np.array(dr['deltas'])
        p_arr = np.array(dr['pcts'])
        n_d = len(d_arr)
        per_domain[domain] = {
            'n_tasks': n_d,
            'win_rate': dr['wins'] / n_d if n_d > 0 else 0,
            'wins': dr['wins'],
            'mean_delta': float(np.mean(d_arr)),
            'median_delta': float(np.median(d_arr)),
            'mean_pct_improvement': float(np.mean(p_arr)),
            'tasks': dr['tasks'],
        }
        # Domain-level Wilcoxon if enough tasks
        if n_d >= 3:
            try:
                stat_d, p_d = wilcoxon(d_arr, alternative='greater')
                per_domain[domain]['wilcoxon_p'] = float(p_d)
            except Exception:
                pass

        # Complexity matching strength: how much does representation choice
        # matter in this domain? High std of deltas = some tasks benefit a lot,
        # some don't = representation matters differently by task.
        # High mean |delta| = representation consistently matters.
        per_domain[domain]['complexity_matching_strength'] = float(np.mean(np.abs(d_arr)))
        per_domain[domain]['delta_std'] = float(np.std(d_arr))

        print(f"    {domain}: {dr['wins']}/{n_d} wins, "
              f"mean_delta={np.mean(d_arr):+.4f}, "
              f"mean_%imp={np.mean(p_arr):+.1f}%, "
              f"matching_strength={np.mean(np.abs(d_arr)):.4f}")

    return {
        'summary': summary,
        'wilcoxon_test': wilcoxon_result,
        'per_domain': per_domain,
    }


# =========================================================================
# Main entry
# =========================================================================

def run(tasks, rep_preset, config, output_dir):
    """Run c14 across all tasks and aggregate.

    Args:
        tasks: list of dataset name strings.
        rep_preset: representation preset (e.g. 'all', 'fast').
        config: fast or full dict.
        output_dir: base output directory.

    Returns:
        Full results dict.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT C14: CROSS-DOMAIN SUMMARY")
    print(f"  Tasks: {tasks}")
    print(f"  Reps: {rep_preset}, Mode: {'full' if config['n_splits'] > 1 else 'intermediary'}")
    print("=" * 70)

    task_results = []
    failed_tasks = []  # {'task', 'reason'} for every task with no result
    for task_name in tasks:
        try:
            result = _evaluate_task(task_name, rep_preset, config)
            if result is not None:
                task_results.append(result)
            else:
                failed_tasks.append({'task': task_name,
                                     'reason': 'no single rep or no hybrid '
                                               'result for this task'})
        except Exception as e:
            failed_tasks.append({'task': task_name,
                                 'reason': f'{type(e).__name__}: {e}'})
            print(f"\n  FAILED: {task_name} — {e}")
            import traceback
            traceback.print_exc()

    if not task_results:
        # Every task failed. Carry an error key so a saved file from this run
        # cannot be read as a completed run with zero improvement.
        print("\nNo tasks completed successfully.")
        return {
            'experiment': EXPERIMENT_ID,
            'timestamp': datetime.datetime.now().isoformat(),
            'config': {
                'tasks': tasks,
                'rep_preset': rep_preset,
                'n_splits': config['n_splits'],
                'models': config['models'],
            },
            'error': f'all {len(tasks)} requested task(s) failed; '
                     f'no per-task result was produced',
            'failed_tasks': failed_tasks,
            'per_task': [],
            'aggregate': {},
        }

    # Print per-task summary table
    print(f"\n{'='*70}")
    print(f"  {'Task':<30} {'Domain':<10} {'Metric':<8} "
          f"{'BestSingle':>10} {'Hybrid':>10} {'Delta':>10} {'%Imp':>8}")
    print(f"  {'-'*30} {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for r in task_results:
        _bs = r.get('reselected_best_single_score')
        _bs_txt = f"{_bs:.4f}" if _bs is not None and not np.isnan(_bs) else 'n/a'
        print(f"  {r['task']:<30} {r['domain']:<10} {r['metric_name']:<8} "
              f"{_bs_txt:>10} {r['hybrid_score']:>10.4f} "
              f"{r['delta']:>+10.4f} {r['pct_improvement']:>+7.1f}%")
    print("  Every column above is an outer-fold score: each fold's hybrid "
          "recipe and each fold's single representation were picked on that "
          "fold's training rows only. Delta is the fold-by-fold difference, "
          "averaged.")

    # Same tasks with the single-rep side restricted to the folds every rep
    # survived. delta_vs_best_single_common_folds pairs the reps with each
    # other only; the hybrid side is unchanged.
    print(f"\n  {'Task':<30} {'ConfigFolds':>11} {'CommonFolds':>11} "
          f"{'BestSingleCF':>13} {'DeltaCF':>10} {'%ImpCF':>8}")
    print(f"  {'-'*30} {'-'*11} {'-'*11} {'-'*13} {'-'*10} {'-'*8}")
    for r in task_results:
        s_cf = r.get('best_single_score_common_folds')
        d_cf = r.get('delta_vs_best_single_common_folds')
        p_cf = r.get('pct_improvement_vs_best_single_common_folds')
        print(f"  {r['task']:<30} {r['n_folds_configured']:>11} "
              f"{r['n_common_folds']:>11} "
              f"{(f'{s_cf:.4f}' if s_cf is not None else 'n/a'):>13} "
              f"{(f'{d_cf:+.4f}' if d_cf is not None else 'n/a'):>10} "
              f"{(f'{p_cf:+.1f}%' if p_cf is not None else 'n/a'):>8}")

    # Aggregate
    aggregate = _aggregate(task_results)

    results = {
        'experiment': EXPERIMENT_ID,
        'timestamp': datetime.datetime.now().isoformat(),
        'config': {
            'tasks': tasks,
            'rep_preset': rep_preset,
            'n_splits': config['n_splits'],
            'models': config['models'],
        },
        'failed_tasks': failed_tasks,
        'per_task': convert_numpy(task_results),
        'aggregate': convert_numpy(aggregate),
    }

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='C14: Cross-domain summary — hybrid vs best single rep')
    parser.add_argument('--tasks', type=str, required=True,
                        help='Comma-separated list of dataset names '
                             '(e.g., "logd,caco2,herg,flab_thermo_tresanco,'
                             'tcga_luad_vs_lusc")')
    parser.add_argument('--reps', type=str, default='all',
                        help='Representation preset (fast, all)')
    parser.add_argument('--mode', choices=['intermediary', 'full'],
                        default='intermediary',
                        help='intermediary (1 split) or full (5-fold CV)')
    parser.add_argument('--output', type=str, default='tests/results',
                        help='Output directory for results JSON')
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(',') if t.strip()]
    config = get_config(args)

    if not tasks:
        print("ERROR: --tasks must specify at least one dataset name")
        sys.exit(1)

    results = run(tasks, args.reps, config, args.output)
    save_experiment_results(results, args.output, EXPERIMENT_ID)
