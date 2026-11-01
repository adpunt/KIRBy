#!/usr/bin/env python3
"""Hybrid features for the MODNet feature scores experiment (Matbench v0.1)

Builds c02's hybrids into feature files that
opig_modnet_feature_scores.py reads. Runs in KIRBy's environment; MODNet
0.1.12 runs in its own (b04's), so the features pass between them as files.

Per outer fold, the same as c02 (complexity/c02_complexity_grid.py):
  folds      nested_cv_outer_folds(reps, labels, groups, n_splits, 42,
             task_type), c02's outer folds
  recipes    fold_standard_selection on the fold's training rows, with c02's
             arguments: the dataset's decided models (knn left out), its
             augmentations, metric and seed 42. After c02 has run, this reads
             its cached search; it does not search again.
  features   build_fold_hybrid from each recipe, with c02's arguments
             (_build_and_score), so the features are the ones c02 scored.
Run with the same --dataset, --reps and fold arguments as c02.

The features are written unscaled, as built: MODNet scales them itself (its
settings search tries minmax and standard scaling).

One file per dataset, fold, model and hybrid,
<KIRBY_MATBENCH_ROOT>/benchmarks_protocol/matbench/hybrid_features/
<dataset>_fold<k>_<model>_hybrid<i>.npz, holding X_train, X_test,
feature_names (h0, h1, ...), train_index, test_index (positions in the
dataset's row order), ids (the dataset's row ids, when it has them) and
recipe (the recipe's fields, as JSON). opig_modnet_feature_scores.py checks
the indexes against Matbench's own folds, so it stops if the two differ.

    python tests/experiments/opig_modnet_hybrid_features.py \\
        --dataset matbench_jdft2d <c02's other arguments>
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
from pathlib import Path

import numpy as np

from experiments._shared import (
    load_data, get_config, create_experiment_parser, get_model_factories,
    get_domain_metric, fold_standard_selection, build_fold_hybrid,
    fold_hybrid_scoring_kwargs, parse_rep_args, convert_numpy)
from kirby.find_best_hybrid import nested_cv_outer_folds

# The same root as b04 (KIRBY_MATBENCH_ROOT; /Volumes/seagate on the laptop).
OUT_DIR = (Path(os.environ.get('KIRBY_MATBENCH_ROOT', '/Volumes/seagate'))
           / 'benchmarks_protocol' / 'matbench' / 'hybrid_features')
SEED = 42   # c02's SEED
RECIPE_FIELDS = ('label', 'allocation_method', 'importance_method', 'budget',
                 'internal_evaluator', 'rep_count', 'step_size', 'n_features',
                 'aug_condition', 'score_mean')   # c02's HYBRID_FIELDS + two


def build(data, config, dataset_name):
    from experiments.decided_grid import decided_models
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    task_type = data['task_type']
    reps = data['reps']
    labels = np.asarray(data['labels'])
    groups = data['groups']
    augs = data.get('augmentations') or None
    models = [m for m in decided_models(dataset_name, 'general') if m != 'knn']
    factories = get_model_factories(models, task_type=task_type,
                                    n_samples=len(labels))
    folds = nested_cv_outer_folds(reps, labels, groups,
                                  int(config['n_splits']), SEED, task_type)
    ids = data.get('ids')
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for k, (tr, va) in enumerate(folds):
        sel = fold_standard_selection(
            reps, labels, groups, train_idx=tr, fold_i=k,
            n_outer_folds=len(folds), dataset_name=dataset_name,
            models=list(models), augmentations=augs, task_type=task_type,
            metric=metric_name, seed=SEED)
        y_tr = labels[tr]
        g_tr = None if groups is None else np.asarray(groups)[tr]
        tr_reps = {r: np.asarray(X)[tr] for r, X in reps.items()}
        va_reps = {r: np.asarray(X)[va] for r, X in reps.items()}
        for m, s in sel.items():
            if s is None or m not in factories:
                print(f'  fold {k} {m}: no search result, nothing written')
                continue
            for i, e in enumerate(s['hybrids'], 1):
                bp = e['blueprint']
                cond = e.get('aug_condition') or 'none'
                aug_tr = aug_va = None
                if cond != 'none' and augs:
                    aug_tr = {a: np.asarray(v)[tr] for a, v in augs.items()}
                    aug_va = {a: np.asarray(v)[va] for a, v in augs.items()}
                X_tr, X_va, _ = build_fold_hybrid(
                    tr_reps, va_reps, y_tr, bp,
                    augmentations_train=aug_tr, augmentations_val=aug_va,
                    groups=g_tr,
                    precomputed_alloc_scores=getattr(bp, 'rep_scores', None),
                    **fold_hybrid_scoring_kwargs(
                        bp, model=factories[m](), metric_fn=metric_fn,
                        lower_is_better=lower_is_better))
                X_tr, X_va = np.asarray(X_tr), np.asarray(X_va)
                recipe = convert_numpy({f: e.get(f) for f in RECIPE_FIELDS})
                path = OUT_DIR / f'{dataset_name}_fold{k}_{m}_hybrid{i}.npz'
                np.savez(path, X_train=X_tr, X_test=X_va,
                         feature_names=np.array(
                             [f'h{j}' for j in range(X_tr.shape[1])]),
                         train_index=np.asarray(tr), test_index=np.asarray(va),
                         ids=np.array(ids if ids is not None else [], dtype=str),
                         recipe=json.dumps(recipe))
                print(f'  fold {k} {m} hybrid{i}: {X_tr.shape[1]} features, '
                      f'{len(tr)} training, {len(va)} test -> {path}',
                      flush=True)


if __name__ == '__main__':
    parser = create_experiment_parser(
        'Hybrid features for opig_modnet_feature_scores', has_delta=True,
        has_augment=True)
    args = parser.parse_args()
    config = get_config(args)
    rep_preset, rep_names = parse_rep_args(args)
    data = load_data(args.dataset, rep_preset, rep_names=rep_names,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    build(data, config, args.dataset)
