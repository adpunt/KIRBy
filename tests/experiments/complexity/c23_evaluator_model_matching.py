#!/usr/bin/env python3
"""C23: Internal Evaluator–Downstream Model Matching.

Paper: Complexity
Conclusion: Matched internal evaluators (where the evaluator model family
    matches the downstream model) produce better hybrid representations than
    mismatched configurations. The gap is especially pronounced for NNs.

Dependencies: None (standalone).

Design: For each downstream model and each internal evaluator setting, run
    nested_cv_evaluate: the allocation/importance/budget search runs inside
    each outer training fold, and the winner is scored on the fold it never
    saw. Deltas between evaluator settings are then paired fold by fold.

Conditions:
    A_matched:      evaluator = downstream model type (rf->rf, xgb->xgb, etc.)
    B_slight_off:   evaluator from a similar model family
                    (rf->lightgbm, lightgbm->rf, dnn->mlp).
    C_strong_off:   evaluator from a very different family
                    (rf->svm, lightgbm->svm, dnn->knn).
    D_nn_size:      NN size mismatches (mlp->dnn). Does not run while mlp is
                    outside the study's downstream set; the loop skips any
                    pair whose downstream model is not in MATCH_TABLE.
    E_quick_set:    evaluator always from {rf, lightgbm, xgboost}

Reports the held-out score delta (matched - mismatched), paired per outer
fold, per model x dataset x condition.

Usage:
    python c23_evaluator_model_matching.py --dataset herg --mode intermediary
    python c23_evaluator_model_matching.py --dataset caco2 --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_model_factories, get_domain_metric,
    DOWNSTREAM_MODELS, FULL_BUDGETS)
from kirby.find_best_hybrid import nested_cv_evaluate


# ── Evaluator mismatch configurations ──
# For each downstream model, define which internal evaluator to use per condition.
# Note: dnn uses itself as its A_matched evaluator (_evaluate_allocation in
# hybrid.py has an explicit branch for it).
#
# Cut to the study's five downstream models on 2026-09-10 (owner decision).
# The table used to carry eleven rows — ridge, knn, mlp, gp, catboost and cnn
# as well. Those six are tier 'extra' or 'off' in DEFAULT_MODELS, so none of
# them is a downstream model of this study any more. Re-adding one means
# writing its own row here with a stated near and far family.
#
# ridge is gone from BOTH sides. As a downstream model it is tier 'off'
# (find_best_hybrid.py, 2026-09-09). As the model scoring candidates inside a
# search there is no ridge branch at all: hybrid.py's _evaluate_allocation
# falls to its else and raises "Unknown internal_evaluator: ridge". It named
# the far family for rf, lightgbm and catboost and the near family for mlp and
# svm, so six cells threw, were caught, printed FAILED and stored nothing.
# svm now stands in as the far family for rf and lightgbm, which is the choice
# xgboost's row already made. Nothing else moved.
MATCH_TABLE = {
    # model:     A_matched   B_slight_off  C_strong_off
    'rf':        ('rf',      'lightgbm',   'svm'),
    'lightgbm':  ('lightgbm','rf',         'svm'),
    'xgboost':   ('xgboost', 'lightgbm',   'svm'),
    'svm':       ('svm',     'gp',         'lightgbm'),
    'dnn':       ('dnn',     'mlp',        'knn'),
}

# NN size mismatch pairs — Condition D.
# ('dnn', 'mlp') removed: it would produce the exact cache key already covered
# by ABC B_slight_off for dnn (MATCH_TABLE['dnn'][1] == 'mlp'), giving a
# duplicate score column with no extra information.
# mlp left the downstream set on 2026-09-04 (tier 'extra'), so this pair does
# not run — the condition D loop skips any downstream model not in
# MATCH_TABLE. Put mlp back in MATCH_TABLE and it runs again.
NN_PAIRS = [
    ('mlp', 'dnn'),   # small NN downstream, large NN evaluator
]

# Quick default set — Condition E
QUICK_EVALUATORS = ['rf', 'lightgbm', 'xgboost']

# The study's downstream models, one list shared with every other experiment.
TEST_MODELS = list(DOWNSTREAM_MODELS)


def _run_single_config(reps, labels, groups, model_name, model_factory,
                       evaluator, budgets,  metric_name, task_type,
                       lower_is_better, augmentations, aug_strategy,
                       dataset_name=None, n_outer_folds=5):
    """Held-out score for one (model, evaluator) pair, via nested CV.

    Each outer fold runs its own allocation/importance/budget search on that
    fold's training rows under this one internal evaluator, then scores the
    winner on the fold the search never saw. `score_mean` below is the mean of
    those outer-fold scores.

    Before, this returned the winning configuration's own inner-CV mean —
    maximised over the same folds it was then reported on. Every
    matched-minus-mismatched delta was a difference of two such maxima, so it
    was inflated on both sides before the subtraction.
    """
    models = {model_name: model_factory()}
    try:
        ncv = nested_cv_evaluate(
            reps, labels, groups,
            n_outer_folds=n_outer_folds,
            # The real dataset name, so the per-dataset budget cap matches.
            # This used to be f"{dataset_name}_{model_name}_{evaluator}", and
            # the cap and the per-dataset model skips are looked up BY NAME, so
            # neither ever matched: on the 22 capped datasets c23 searched
            # budgets the dataset is capped away from. The namespacing those
            # made-up names were doing now rides on condition_suffix, which is
            # appended to `tier` and so reaches every cache layer and the stored
            # row. Belt and braces in any case: `model` and `internal_evaluator`
            # are already per-row columns, so two of these calls could not have
            # shared a row. Dict-form `models` skips the policy's model list, so
            # no arm of c23 can be dropped by this change.
            dataset_name=dataset_name,
            condition_suffix=f"c23_{model_name}_{evaluator}",
            budgets=budgets,
            models=models,
            internal_evaluators=[evaluator],
            metric=metric_name,
            task_type=task_type,
            cache=True,
            verbose=False,
            augmentations=augmentations,
            augmentation_strategy=aug_strategy,
            tier='general')
        entry = (ncv.by_model or {}).get(model_name)
        if entry:
            rep_row = ncv.best_by_model.get(model_name, {})
            return {
                # Mean over the outer folds. Named score_mean so existing
                # readers keep working; it is no longer a search-side number.
                'score_mean': entry.get('hybrid_mean'),
                'score_std': entry.get('hybrid_std'),
                'fold_scores': list(entry.get('hybrid_fold_scores') or []),
                'score_basis': 'mean over outer folds (nested CV)',
                # What the winner's own inner mean was, and how far above the
                # outer-fold score it sat: picking on the folds you report on.
                'search_mean': entry.get('search_mean'),
                'selection_bias': entry.get('bias'),
                'n_features': rep_row.get('n_features') or 0,
                'budget': rep_row.get('budget') or 0,
                'allocation_method': rep_row.get('allocation_method') or '',
                'importance_method': rep_row.get('importance_method') or '',
                'recipe_is_from_fold': 'first outer fold with a winner',
            }
    except Exception as e:
        print(f"      FAILED: {e}")
    return None


def _paired_fold_delta(a_scores, other_scores, sign):
    """Fold-by-fold (matched - other), averaged over folds both scored.

    Both sides come from the same outer folds, so the difference is paired.
    Returns (delta, n_paired_folds) — (nan, 0) when nothing lines up.
    """
    a = (a_scores or {}).get('fold_scores') or []
    b = (other_scores or {}).get('fold_scores') or []
    paired = [sign * (x - y) for x, y in zip(a, b)
              if x is not None and y is not None
              and not np.isnan(x) and not np.isnan(y)]
    if not paired:
        return float('nan'), 0
    return float(np.mean(paired)), len(paired)


def run(data, config, output_dir, dataset_name, **kwargs):
    """C23: Evaluator-Model Matching."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C23: EVALUATOR-MODEL MATCHING")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data.get('groups')
    task_type = data['task_type']
    n_samples = len(labels)
    augmentations = data.get('augmentations') or None
    # CXA-15: `data` never carries an 'augmentation_strategy' key (no loader sets
    # it), so the old `data.get('augmentation_strategy', 'uncapped')` was a dead
    # lookup that always returned 'uncapped' — even when there are NO
    # augmentations. Passing augmentation_strategy='uncapped' with
    # augmentations=None writes a DIFFERENT find_best_hybrid cache key than the
    # canonical 'none' used by precompute / the other experiments, so c23 would
    # silently miss the cache and recompute. Derive it from whether
    # augmentations actually exist (mirrors c24:165 and get_precomputed_search).
    aug_strategy = 'uncapped' if augmentations else 'none'

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_outer_folds = config['n_splits']
    print(f"  Metric: {metric_name} (lower_is_better={lower_is_better})")
    print(f"  Outer folds: {n_outer_folds} — each runs its own recipe search "
          f"on its training rows, so every score below is held out.")

    all_models = get_model_factories(TEST_MODELS, task_type=task_type,
                                     n_samples=n_samples)
    available_models = {k: v for k, v in all_models.items() if k in MATCH_TABLE}
    print(f"  Models: {list(available_models.keys())}")

    budgets = FULL_BUDGETS
    sign = -1 if lower_is_better else 1

    results = []

    # ── Conditions A, B, C: matched / slight off / strong off ──
    for model_name, model_factory in available_models.items():
        matched_ev, slight_ev, strong_ev = MATCH_TABLE[model_name]
        print(f"\n  {model_name}: matched={matched_ev}, slight={slight_ev}, strong={strong_ev}")

        scores = {}
        for cond_name, evaluator in [('A_matched', matched_ev),
                                      ('B_slight_off', slight_ev),
                                      ('C_strong_off', strong_ev)]:
            print(f"    {cond_name} (eval={evaluator})...", end=' ', flush=True)
            r = _run_single_config(
                reps, labels, groups, model_name, model_factory,
                evaluator, budgets, metric_name, task_type,
                lower_is_better, augmentations, aug_strategy,
                dataset_name=dataset_name, n_outer_folds=n_outer_folds)
            if r:
                scores[cond_name] = r
                print(f"score={r['score_mean']:.4f}")
            else:
                print("no result")

        # Deltas, paired fold by fold. Both conditions ran on the same outer
        # folds, so subtracting per fold and averaging is the paired quantity;
        # the difference of the two means diverges from it as soon as either
        # side drops a fold.
        if 'A_matched' in scores:
            a_score = scores['A_matched']['score_mean']
            for cond in ['B_slight_off', 'C_strong_off']:
                if cond in scores:
                    d, n_pair = _paired_fold_delta(
                        scores['A_matched'], scores[cond], sign)
                    scores[cond]['delta_vs_matched'] = d
                    scores[cond]['n_paired_folds'] = n_pair
                    scores[cond]['delta_of_means'] = (
                        sign * (a_score - scores[cond]['score_mean']))

        results.append({
            'model': model_name,
            'condition_type': 'ABC',
            'evaluators': {
                'A_matched': matched_ev,
                'B_slight_off': slight_ev,
                'C_strong_off': strong_ev,
            },
            'scores': scores,
        })

    # ── Condition D: NN size mismatch ──
    print(f"\n  NN size mismatch pairs:")
    for downstream, evaluator in NN_PAIRS:
        if downstream not in available_models:
            continue
        model_factory = available_models[downstream]
        print(f"    {downstream} -> eval={evaluator}...", end=' ', flush=True)
        r = _run_single_config(
            reps, labels, groups, downstream, model_factory,
            evaluator, budgets, metric_name, task_type,
            lower_is_better, augmentations, aug_strategy,
            dataset_name=dataset_name, n_outer_folds=n_outer_folds)
        if r:
            print(f"score={r['score_mean']:.4f}")
        else:
            print("no result")

        # Find the matched result for comparison
        matched_entry = None
        for prev in results:
            if prev['model'] == downstream and prev['condition_type'] == 'ABC':
                s = prev['scores'].get('A_matched')
                if s:
                    matched_entry = s

        delta = None
        n_pair = 0
        if matched_entry is not None and r is not None:
            delta, n_pair = _paired_fold_delta(matched_entry, r, sign)

        results.append({
            'model': downstream,
            'condition_type': 'D_nn_size',
            'evaluators': {'downstream': downstream, 'evaluator': evaluator},
            'scores': {'D_nn_size': r} if r else {},
            'delta_vs_matched': delta,
            'n_paired_folds': n_pair,
        })

    # ── Condition E: Quick default set ──
    # For rf, lightgbm and xgboost the quick set CONTAINS that model's own
    # matched evaluator, so E searches a superset of A. On the old same-fold
    # numbers that alone forced its delta to be zero or negative. Each outer
    # fold now picks its evaluator on its own training rows and is scored on
    # rows it never saw, so the sign is no longer forced — the containment is
    # still recorded per row as evaluator_set_contains_matched.
    print(f"\n  Quick default set: {QUICK_EVALUATORS}")
    for model_name, model_factory in available_models.items():
        # Run with the quick evaluator set (all 3 evaluated, best wins)
        models = {model_name: model_factory()}
        print(f"    {model_name}...", end=' ', flush=True)
        try:
            ncv = nested_cv_evaluate(
                reps, labels, groups,
                n_outer_folds=n_outer_folds,
                dataset_name=dataset_name,
                condition_suffix=f"c23quick_{model_name}",
                budgets=budgets,
                models=models,
                internal_evaluators=QUICK_EVALUATORS,
                metric=metric_name,
                task_type=task_type,
                cache=True,
                verbose=False,
                augmentations=augmentations,
                augmentation_strategy=aug_strategy,
                tier='general')
            entry = (ncv.by_model or {}).get(model_name)
            if entry:
                rep_row = ncv.best_by_model.get(model_name, {})
                score_info = {
                    'score_mean': entry.get('hybrid_mean'),
                    'score_std': entry.get('hybrid_std'),
                    'fold_scores': list(entry.get('hybrid_fold_scores') or []),
                    'score_basis': 'mean over outer folds (nested CV)',
                    'search_mean': entry.get('search_mean'),
                    'selection_bias': entry.get('bias'),
                    'n_features': rep_row.get('n_features') or 0,
                    'winning_evaluator': rep_row.get('internal_evaluator') or '',
                    'winning_evaluator_is_from_fold':
                        'first outer fold with a winner',
                    'evaluator_set_contains_matched': bool(
                        MATCH_TABLE[model_name][0] in QUICK_EVALUATORS),
                }
                print(f"score={score_info['score_mean']:.4f} "
                      f"(winner={score_info['winning_evaluator']})")

                # Delta vs matched, paired fold by fold.
                matched_entry = None
                for prev in results:
                    if prev['model'] == model_name and prev['condition_type'] == 'ABC':
                        s = prev['scores'].get('A_matched')
                        if s:
                            matched_entry = s
                if matched_entry is not None:
                    d, n_pair = _paired_fold_delta(matched_entry, score_info, sign)
                    score_info['delta_vs_matched'] = d
                    score_info['n_paired_folds'] = n_pair

                results.append({
                    'model': model_name,
                    'condition_type': 'E_quick_set',
                    'evaluators': QUICK_EVALUATORS,
                    'scores': {'E_quick_set': score_info},
                })
            else:
                print("no result")
        except Exception as e:
            print(f"FAILED: {e}")

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    abc_results = [r for r in results if r['condition_type'] == 'ABC']
    if abc_results:
        print(f"\n  {'model':<12s} {'A_matched':>10s} {'B_slight':>10s} {'C_strong':>10s} {'delta_B':>8s} {'delta_C':>8s}")
        print(f"  {'-'*60}")
        for r in abc_results:
            s = r['scores']
            a = s.get('A_matched', {}).get('score_mean', float('nan'))
            b = s.get('B_slight_off', {}).get('score_mean', float('nan'))
            c = s.get('C_strong_off', {}).get('score_mean', float('nan'))
            db = s.get('B_slight_off', {}).get('delta_vs_matched', float('nan'))
            dc = s.get('C_strong_off', {}).get('delta_vs_matched', float('nan'))
            print(f"  {r['model']:<12s} {a:>10.4f} {b:>10.4f} {c:>10.4f} {db:>+8.4f} {dc:>+8.4f}")

        # Aggregate
        all_b_deltas = [r['scores'].get('B_slight_off', {}).get('delta_vs_matched')
                        for r in abc_results]
        all_c_deltas = [r['scores'].get('C_strong_off', {}).get('delta_vs_matched')
                        for r in abc_results]
        b_deltas = [d for d in all_b_deltas if d is not None]
        c_deltas = [d for d in all_c_deltas if d is not None]
        if b_deltas:
            print(f"\n  Mean delta (matched - slight_off):  {np.mean(b_deltas):+.4f}")
        if c_deltas:
            print(f"  Mean delta (matched - strong_off):  {np.mean(c_deltas):+.4f}")

    return convert_numpy({
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'n_samples': n_samples,
        'models_tested': list(available_models.keys()),
        'budgets': budgets,
        'n_outer_folds': n_outer_folds,
        # Every score_mean in results is a mean over outer folds whose recipe
        # search never saw them, and every delta_vs_matched is the fold-by-fold
        # difference of two such scores, averaged.
        'score_basis': 'mean over outer folds (nested CV)',
        'delta_basis': 'paired per outer fold, then averaged',
        'results': results,
    })


if __name__ == '__main__':
    parser = create_experiment_parser('C23: Evaluator-Model Matching',
                                     has_delta=False, has_augment=False)
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")

    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'c23_evaluator_model_matching',
                            args.dataset)
