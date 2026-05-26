#!/usr/bin/env python3
"""C18: Degrees of Freedom — Diversity vs Depth.

Paper: Complexity
Core question: Given a fixed total budget, is it better to draw features from many
    reps (diversity) or concentrate on few reps (depth)? Tests whether the hybrid's
    allocation across rep families adds value beyond selecting the best single rep.

Conditions at fixed budget:
  - depth_1: all features from top-1 rep
  - depth_2: all features split between top-2 reps
  - depth_3: all features from top-3 reps
  - uniform_all: equal allocation across all reps
  - natural: let the hybrid allocate freely (greedy)

Usage:
    python c18_diversity_vs_depth.py --dataset logd --mode intermediary
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric,
    compute_feature_importance,
    build_fold_hybrid, get_precomputed_search, get_best_for_downstream,
    get_best_search_for_downstream,
)


# =========================================================================
# PER-FOLD HELPERS
# =========================================================================

def _rank_reps_by_importance(reps, labels):
    """Rank reps by aggregate importance. Returns sorted list of (name, score)."""
    imp_dict = compute_feature_importance(reps, labels, method='lightgbm_gain')
    rep_scores = {name: float(np.sum(scores)) for name, scores in imp_dict.items()}
    return sorted(rep_scores.items(), key=lambda x: x[1], reverse=True)


def _select_top_features(train_reps, val_reps, y_train, rep_subset, budget):
    """Per-fold: select top features by train-only importance, apply to both."""
    sub_tr = {k: train_reps[k] for k in rep_subset if k in train_reps}
    sub_va = {k: val_reps[k] for k in rep_subset if k in val_reps}
    if not sub_tr:
        return None, None

    rep_names = sorted(sub_tr.keys())
    imp_dict = compute_feature_importance(sub_tr, y_train, method='lightgbm_gain')
    imp_concat = np.concatenate([imp_dict[n] for n in rep_names])

    X_tr = np.hstack([sub_tr[n] for n in rep_names])
    X_va = np.hstack([sub_va[n] for n in rep_names])

    n_select = min(budget, X_tr.shape[1])
    top_idx = np.argsort(imp_concat)[::-1][:n_select]
    return X_tr[:, top_idx], X_va[:, top_idx]


def _uniform_allocation(train_reps, val_reps, y_train, rep_subset, budget):
    """Per-fold: equal features per rep, top by train-only importance."""
    sub_names = sorted([k for k in rep_subset if k in train_reps])
    if not sub_names:
        return None, None

    per_rep = max(1, budget // len(sub_names))
    tr_parts, va_parts = [], []

    for name in sub_names:
        imp_dict = compute_feature_importance(
            {name: train_reps[name]}, y_train, method='lightgbm_gain')
        imp = imp_dict[name]
        n_sel = min(per_rep, train_reps[name].shape[1])
        top_idx = np.argsort(imp)[::-1][:n_sel]
        tr_parts.append(train_reps[name][:, top_idx])
        va_parts.append(val_reps[name][:, top_idx])

    return np.hstack(tr_parts), np.hstack(va_parts)


# =========================================================================
# RUN
# =========================================================================

def _score_prediction(model, X_val, y_val, metric_fn, task_type):
    """Score model predictions appropriately for the metric.

    Classification + binary: AUROC needs class-1 probabilities, not labels.
    Classification + multi-class: F1 needs class labels, not probabilities.
    Regression: anything that has .predict() works.
    """
    if task_type == 'classification' and hasattr(model, 'predict_proba'):
        try:
            probs = model.predict_proba(X_val)
            if probs.shape[1] == 2:
                return metric_fn(y_val, probs[:, 1])
        except Exception:
            pass
    return metric_fn(y_val, model.predict(X_val))


def run(data, config, output_dir, dataset_name, **kwargs):
    """C18: Diversity vs Depth."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C18: DIVERSITY VS DEPTH")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    metric_name, metric_fn, _ = get_domain_metric(data)
    task_type = data['task_type']
    _RF = RandomForestClassifier if task_type == 'classification' else RandomForestRegressor

    budgets = kwargs.get('budgets', [50, 100])
    all_rep_names = sorted(reps.keys())
    n_reps = len(all_rep_names)
    sample_X = next(iter(reps.values()))

    print(f"  {n_reps} reps, budgets: {budgets}")

    # Full-data ranking for reporting only (per-fold ranking used in evaluation)
    ranked_full = _rank_reps_by_importance(reps, labels)
    print(f"  Rep ranking (full data, reference only): "
          f"{', '.join(f'{n}({s:.2f})' for n, s in ranked_full[:5])}...")

    results = {
        'metric': metric_name,
        'budgets': budgets,
        'n_reps': n_reps,
        'rep_ranking_full_data': [(n, float(s)) for n, s in ranked_full],
        'per_budget': {},
    }

    # Per-budget blueprint discovery via pool helper (best (tier, aug_cond)
    # for downstream_model='rf' at each budget; may differ across budgets).
    augmentations = data.get('augmentations')

    for budget in budgets:
        print(f"\n  Budget={budget}:")
        try:
            search, meta = get_best_search_for_downstream(
                reps, labels, groups,
                downstream_model='rf', budget=budget,
                augmentations=augmentations,
                speed=config.get('search_speed', 'full'),
                dataset_name=dataset_name, task_type=task_type,
                data=data,
            )
        except Exception as e:
            print(f"  WARNING: pool search failed: {e}")
            search, meta = None, None
        fold_scores = {}  # {condition_name: [per_fold_scores]}

        for train_idx, val_idx in cv_split(groups, n_folds, sample_X,
                                           labels, mode=mode):
            train_reps = {name: X[train_idx] for name, X in reps.items()}
            val_reps_fold = {name: X[val_idx] for name, X in reps.items()}
            y_train = labels[train_idx]
            y_val = labels[val_idx]

            # Rank reps on training data only
            fold_ranked = _rank_reps_by_importance(train_reps, y_train)

            # Depth conditions (top-1, top-2, top-3)
            for n_top in [1, 2, 3]:
                if n_top > n_reps:
                    continue
                cond_name = f'depth_{n_top}'
                top_rep_names = [name for name, _ in fold_ranked[:n_top]]
                X_tr, X_va = _select_top_features(
                    train_reps, val_reps_fold, y_train, top_rep_names, budget)
                if X_tr is not None:
                    scaler = StandardScaler()
                    X_tr_s = scaler.fit_transform(X_tr)
                    X_va_s = scaler.transform(X_va)
                    rf = _RF(n_estimators=100, random_state=42, n_jobs=-1)
                    rf.fit(X_tr_s, y_train)
                    fold_scores.setdefault(cond_name, []).append(
                        _score_prediction(rf, X_va_s, y_val, metric_fn, task_type))

            # Uniform allocation across all reps
            X_tr, X_va = _uniform_allocation(
                train_reps, val_reps_fold, y_train, all_rep_names, budget)
            if X_tr is not None:
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_va_s = scaler.transform(X_va)
                rf = _RF(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(X_tr_s, y_train)
                fold_scores.setdefault('uniform_all', []).append(
                    _score_prediction(rf, X_va_s, y_val, metric_fn, task_type))

            # Natural hybrid (per-fold via search blueprint).
            # depth_N and uniform_all operate on base reps only, so the natural
            # arm must also be base-reps-only to keep the comparison apples-to-
            # apples. Augmentations were inflating natural's feature pool.
            if meta is not None:
                try:
                    best_config = meta['best_config']
                    if 'blueprint' in best_config:
                        bp = best_config['blueprint']
                        X_tr_h, X_va_h, _fi = build_fold_hybrid(
                            train_reps, val_reps_fold, y_train, bp,
                            augmentations_train=None,
                            augmentations_val=None)
                        scaler = StandardScaler()
                        X_tr_s = scaler.fit_transform(X_tr_h)
                        X_va_s = scaler.transform(X_va_h)
                        rf = _RF(n_estimators=100, random_state=42, n_jobs=-1)
                        rf.fit(X_tr_s, y_train)
                        fold_scores.setdefault('natural', []).append(
                            _score_prediction(rf, X_va_s, y_val, metric_fn, task_type))
                except Exception as e:
                    print(f"    natural fold FAILED: {e}")

        # Aggregate per-fold scores
        conditions = {}
        for cond_name, scores in fold_scores.items():
            score = float(np.mean(scores)) if scores else float('nan')
            conditions[cond_name] = {
                'score': score,
                'n_features': budget,
            }
            print(f"    {cond_name}: {metric_name}={score:.4f}")

        results['per_budget'][budget] = convert_numpy(conditions)

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('C18: Diversity vs Depth')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'c18_diversity_vs_depth',
                            args.dataset)
