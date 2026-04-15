#!/usr/bin/env python3
"""C23: Internal Evaluator–Downstream Model Matching.

Paper: Complexity
Conclusion: Matched internal evaluators (where the evaluator model family
    matches the downstream model) produce better hybrid representations than
    mismatched configurations. The gap is especially pronounced for NNs.

Dependencies: None (standalone).

Design: For each downstream model, run find_best_hybrid with different
    internal evaluator configurations and compare scores.

Conditions:
    A_matched:      evaluator = downstream model type (rf->rf, xgb->xgb, etc.)
    B_slight_off:   evaluator from a similar model family
                    (rf->lightgbm, xgb->rf, mlp->dnn, etc.)
    C_strong_off:   evaluator from a very different family
                    (rf->ridge, gp->xgb, mlp->knn, etc.)
    D_nn_size:      NN size mismatches (mlp->dnn, dnn->mlp)
    E_quick_set:    evaluator always from {rf, lightgbm, xgboost}

Reports score delta (matched - mismatched) per model x dataset x condition.

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
    FULL_BUDGETS,
)
from kirby.find_best_hybrid import find_best_hybrid


# ── Evaluator mismatch configurations ──
# For each downstream model, define which internal evaluator to use per condition.
# Models without a natural evaluator equivalent (catboost, dnn, cnn) use ridge
# for matched (A) since there's no evaluator for them — they serve as controls.
MATCH_TABLE = {
    # model:     A_matched   B_slight_off  C_strong_off
    'rf':        ('rf',      'lightgbm',   'ridge'),
    'lightgbm':  ('lightgbm','rf',         'ridge'),
    'xgboost':   ('xgboost', 'lightgbm',   'svm'),
    'ridge':     ('ridge',   'svm',        'rf'),
    'knn':       ('knn',     'svm',        'lightgbm'),
    'mlp':       ('mlp',     'ridge',      'knn'),
    'svm':       ('svm',     'ridge',      'lightgbm'),
    'gp':        ('gp',      'svm',        'xgboost'),
    'catboost':  ('catboost', 'lightgbm',   'ridge'),
    'dnn':       ('dnn',     'mlp',        'knn'),
    'cnn':       ('cnn',     'mlp',        'knn'),
}

# NN size mismatch pairs — Condition D
NN_PAIRS = [
    ('mlp', 'dnn'),   # small NN downstream, large NN evaluator
    ('dnn', 'mlp'),   # large NN downstream, small NN evaluator
]

# Quick default set — Condition E
QUICK_EVALUATORS = ['rf', 'lightgbm', 'xgboost']

# All 11 downstream models
TEST_MODELS = ['rf', 'lightgbm', 'xgboost', 'ridge', 'knn', 'mlp', 'svm', 'gp',
               'catboost', 'dnn', 'cnn']


def _run_single_config(reps, labels, groups, model_name, model_factory,
                       evaluator, budgets, speed, metric_name, task_type,
                       lower_is_better, augmentations, aug_strategy,
                       dataset_name=None):
    """Run find_best_hybrid for one (model, evaluator) pair."""
    models = {model_name: model_factory()}
    try:
        search = find_best_hybrid(
            base_reps=reps,
            labels=labels,
            groups=groups,
            budgets=budgets,
            models=models,
            internal_evaluators=[evaluator],
            speed=speed,
            metric=metric_name,
            task_type=task_type,
            cache=True,
            verbose=False,
            augmentations=augmentations,
            augmentation_strategy=aug_strategy,
            dataset_name=f"{dataset_name}_{model_name}_{evaluator}",
            tier='general',
        )
        best = search.best_by_model.get(model_name)
        if best:
            return {
                'score_mean': best['score_mean'],
                'score_std': best['score_std'],
                'n_features': best.get('n_features', 0),
                'budget': best.get('budget', 0),
                'allocation_method': best.get('allocation_method', ''),
                'importance_method': best.get('importance_method', ''),
            }
    except Exception as e:
        print(f"      FAILED: {e}")
    return None


def run(data, config, output_dir, dataset_name, **kwargs):
    """C23: Evaluator-Model Matching."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C23: EVALUATOR-MODEL MATCHING")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data.get('groups')
    task_type = data.get('task_type', 'regression')
    n_samples = len(labels)
    augmentations = data.get('augmentations')
    aug_strategy = data.get('augmentation_strategy', 'uncapped')

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    print(f"  Metric: {metric_name} (lower_is_better={lower_is_better})")

    all_models = get_model_factories(TEST_MODELS, task_type=task_type,
                                     n_samples=n_samples)
    available_models = {k: v for k, v in all_models.items() if k in MATCH_TABLE}
    print(f"  Models: {list(available_models.keys())}")

    budgets = FULL_BUDGETS
    search_speed = config.get('search_speed', 'full')
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
                evaluator, budgets, search_speed, metric_name, task_type,
                lower_is_better, augmentations, aug_strategy,
                dataset_name=dataset_name)
            if r:
                scores[cond_name] = r
                print(f"score={r['score_mean']:.4f}")
            else:
                print("no result")

        # Compute deltas
        if 'A_matched' in scores:
            a_score = scores['A_matched']['score_mean']
            for cond in ['B_slight_off', 'C_strong_off']:
                if cond in scores:
                    delta = sign * (a_score - scores[cond]['score_mean'])
                    scores[cond]['delta_vs_matched'] = delta

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
            evaluator, budgets, search_speed, metric_name, task_type,
            lower_is_better, augmentations, aug_strategy,
            dataset_name=dataset_name)
        if r:
            print(f"score={r['score_mean']:.4f}")
        else:
            print("no result")

        # Find the matched result for comparison
        matched_score = None
        for prev in results:
            if prev['model'] == downstream and prev['condition_type'] == 'ABC':
                s = prev['scores'].get('A_matched')
                if s:
                    matched_score = s['score_mean']

        delta = None
        if matched_score is not None and r is not None:
            delta = sign * (matched_score - r['score_mean'])

        results.append({
            'model': downstream,
            'condition_type': 'D_nn_size',
            'evaluators': {'downstream': downstream, 'evaluator': evaluator},
            'scores': {'D_nn_size': r} if r else {},
            'delta_vs_matched': delta,
        })

    # ── Condition E: Quick default set ──
    print(f"\n  Quick default set: {QUICK_EVALUATORS}")
    for model_name, model_factory in available_models.items():
        # Run with the quick evaluator set (all 3 evaluated, best wins)
        models = {model_name: model_factory()}
        print(f"    {model_name}...", end=' ', flush=True)
        try:
            search = find_best_hybrid(
                base_reps=reps,
                labels=labels,
                groups=groups,
                budgets=budgets,
                models=models,
                internal_evaluators=QUICK_EVALUATORS,
                speed=search_speed,
                metric=metric_name,
                task_type=task_type,
                cache=True,
                verbose=False,
                augmentations=augmentations,
                augmentation_strategy=aug_strategy,
                dataset_name=f"{dataset_name}_quick_{model_name}",
                tier='general',
            )
            best = search.best_by_model.get(model_name)
            if best:
                score_info = {
                    'score_mean': best['score_mean'],
                    'score_std': best['score_std'],
                    'n_features': best.get('n_features', 0),
                    'winning_evaluator': best.get('internal_evaluator', ''),
                }
                print(f"score={best['score_mean']:.4f} "
                      f"(winner={score_info['winning_evaluator']})")

                # Delta vs matched
                matched_score = None
                for prev in results:
                    if prev['model'] == model_name and prev['condition_type'] == 'ABC':
                        s = prev['scores'].get('A_matched')
                        if s:
                            matched_score = s['score_mean']
                if matched_score is not None:
                    score_info['delta_vs_matched'] = sign * (matched_score - score_info['score_mean'])

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
