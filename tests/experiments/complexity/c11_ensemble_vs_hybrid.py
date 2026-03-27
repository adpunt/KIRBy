#!/usr/bin/env python3
"""C11: Ensemble vs hybrid — is feature-level fusion better than model-level fusion?

Paper: Complexity
Conclusion: Whether a single model on a hybrid representation outperforms an
ensemble of models each trained on individual base representations, and how
both compare to the best single base rep.
Dependencies: None (standalone).

Three conditions per model type:
  1. Best single base rep (train one model on each rep, report best)
  2. Ensemble: train one model per base rep, average predictions
  3. Hybrid: train one model on the fused hybrid representation

Each condition is tested with and without augmented features.
Reports R2, total dimensionality, model count, training time, and inference time.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_model_factories, cv_split,
    apply_feature_selection, get_domain_metric,
)
from experiments.complexity.complexity_helpers import cached_create_hybrid as create_hybrid
from kirby.hybrid import apply_augmentation_selection
# TODO: create_hybrid API not finalised — allocation_method / importance_method
#       semantics may change after framework experiments. Revisit call sites below.


def _time_fit_predict(model_factory, X_train, y_train, X_val):
    """Fit a model and predict, returning (predictions, fit_time_s, predict_time_s)."""
    model = model_factory()
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0
    t0 = time.time()
    preds = model.predict(X_val)
    predict_time = time.time() - t0
    return preds, fit_time, predict_time


def run(data, config, output_dir, dataset_name, **kwargs):
    """Run ensemble vs hybrid comparison."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C11: ENSEMBLE VS HYBRID")
    print("=" * 70)

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    models = get_model_factories(config['models'])
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    importance_method = 'lightgbm_gain'

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    rep_names = sorted(reps.keys())
    n_reps = len(rep_names)
    total_dim = sum(v.shape[1] for v in reps.values())

    print(f"  {n_reps} base reps, total dim={total_dim}, {len(models)} models, "
          f"{n_folds} folds")
    print(f"  Reps: {', '.join(f'{k}({v.shape[1]})' for k, v in sorted(reps.items()))}\n")

    sample_X = next(iter(reps.values()))
    budgets = [b for b in [50, 100, 200, 500, 1024] if b <= total_dim]
    results = []

    for model_name, model_factory in models.items():
        print(f"\n  --- {model_name} ---")

        # =====================================================================
        # 1. Best single base rep
        # =====================================================================
        best_single_r2 = -np.inf
        best_single_rep = None
        single_rep_results = {}

        for rep_name in rep_names:
            X_rep = reps[rep_name]
            fold_scores = []
            fold_fit_times = []
            fold_pred_times = []

            for train_idx, val_idx in cv_split(groups, n_folds, sample_X,
                                               labels, mode=mode):
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_rep[train_idx])
                X_val = scaler.transform(X_rep[val_idx])
                y_train, y_val = labels[train_idx], labels[val_idx]

                preds, ft, pt = _time_fit_predict(model_factory, X_train, y_train, X_val)
                fold_scores.append(metric_fn(y_val, preds))
                fold_fit_times.append(ft)
                fold_pred_times.append(pt)

            if not fold_scores:
                r2_mean = float('nan')
                single_rep_results[rep_name] = {
                    'r2_mean': r2_mean, 'r2_std': float('nan'),
                    'dim': X_rep.shape[1],
                    'fit_time_s': 0.0, 'predict_time_s': 0.0,
                }
                continue

            r2_mean = float(np.mean(fold_scores))
            single_rep_results[rep_name] = {
                'r2_mean': r2_mean,
                'r2_std': float(np.std(fold_scores)),
                'dim': X_rep.shape[1],
                'fit_time_s': float(np.mean(fold_fit_times)),
                'predict_time_s': float(np.mean(fold_pred_times)),
            }
            if r2_mean > best_single_r2:
                best_single_r2 = r2_mean
                best_single_rep = rep_name

        if best_single_rep is None:
            print(f"    Best single: NONE (all reps failed)")
            continue

        best_single = single_rep_results[best_single_rep]
        print(f"    Best single: {best_single_rep} "
              f"R2={best_single['r2_mean']:.4f} dim={best_single['dim']}")

        results.append({
            'model': model_name,
            'method': 'best_single',
            'augmented': False,
            'r2_mean': best_single['r2_mean'],
            'r2_std': best_single['r2_std'],
            'total_dim': best_single['dim'],
            'n_models_trained': 1,
            'total_fit_time_s': best_single['fit_time_s'],
            'total_predict_time_s': best_single['predict_time_s'],
            'detail': {'best_rep': best_single_rep, 'all_reps': single_rep_results},
        })

        # =====================================================================
        # 2. Ensemble: one model per base rep, average predictions
        # =====================================================================
        fold_ensemble_scores = []
        fold_ensemble_fit_times = []
        fold_ensemble_pred_times = []

        for train_idx, val_idx in cv_split(groups, n_folds, sample_X,
                                           labels, mode=mode):
            y_train, y_val = labels[train_idx], labels[val_idx]
            all_preds = []
            total_fit = 0.0
            total_pred = 0.0

            for rep_name in rep_names:
                X_rep = reps[rep_name]
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_rep[train_idx])
                X_val = scaler.transform(X_rep[val_idx])

                preds, ft, pt = _time_fit_predict(model_factory, X_train, y_train, X_val)
                all_preds.append(preds)
                total_fit += ft
                total_pred += pt

            if not all_preds:
                continue
            ensemble_preds = np.mean(all_preds, axis=0)
            fold_ensemble_scores.append(metric_fn(y_val, ensemble_preds))
            fold_ensemble_fit_times.append(total_fit)
            fold_ensemble_pred_times.append(total_pred)

        if not fold_ensemble_scores:
            print(f"    Ensemble:    FAILED (no valid folds)")
        else:
            ens_r2 = float(np.mean(fold_ensemble_scores))
            max_single_dim = max(v.shape[1] for v in reps.values())
            print(f"    Ensemble:    R2={ens_r2:.4f} "
                  f"max_rep_dim={max_single_dim} ({n_reps} models)")

            results.append({
                'model': model_name,
                'method': 'ensemble',
                'augmented': False,
                'r2_mean': ens_r2,
                'r2_std': float(np.std(fold_ensemble_scores)),
                'total_dim': total_dim,
                'max_single_dim': max_single_dim,
                'n_models_trained': n_reps,
                'total_fit_time_s': float(np.mean(fold_ensemble_fit_times)),
                'total_predict_time_s': float(np.mean(fold_ensemble_pred_times)),
            })

        # =====================================================================
        # 2b. PCA-matched: concat all reps, PCA-compress to budget dims, one model
        # Isolates: is the hybrid advantage from intelligent selection or just
        # dimensionality reduction?
        # =====================================================================
        for budget in budgets:
            fold_pca_scores = []
            fold_pca_fit_times = []
            fold_pca_pred_times = []

            for train_idx, val_idx in cv_split(groups, n_folds, sample_X,
                                               labels, mode=mode):
                y_train, y_val = labels[train_idx], labels[val_idx]

                # Concat all reps
                X_train_cat = np.hstack([v[train_idx] for v in reps.values()])
                X_val_cat = np.hstack([v[val_idx] for v in reps.values()])

                n_comp = min(budget, X_train_cat.shape[1], len(y_train) - 1)
                if n_comp < 2:
                    continue

                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_train_cat)
                X_va_s = scaler.transform(X_val_cat)

                pca = PCA(n_components=n_comp, random_state=42)
                X_tr_pca = pca.fit_transform(X_tr_s)
                X_va_pca = pca.transform(X_va_s)

                try:
                    t0 = time.time()
                    model = model_factory()
                    model.fit(X_tr_pca, y_train)
                    fit_time = time.time() - t0
                    t0 = time.time()
                    preds = model.predict(X_va_pca)
                    pred_time = time.time() - t0
                    fold_pca_scores.append(metric_fn(y_val, preds))
                    fold_pca_fit_times.append(fit_time)
                    fold_pca_pred_times.append(pred_time)
                except Exception:
                    continue

            if fold_pca_scores:
                results.append({
                    'model': model_name,
                    'method': 'pca_matched',
                    'budget': budget,
                    'r2_mean': float(np.mean(fold_pca_scores)),
                    'r2_std': float(np.std(fold_pca_scores)),
                    'total_dim': n_comp,
                    'n_models_trained': 1,
                    'total_fit_time_s': float(np.mean(fold_pca_fit_times)),
                    'total_predict_time_s': float(np.mean(fold_pca_pred_times)),
                })

        # =====================================================================
        # 3. Hybrid: single model on fused representation
        # =====================================================================
        for budget in budgets:
            fold_hybrid_scores = []
            fold_hybrid_fit_times = []
            fold_hybrid_pred_times = []
            last_n_features = 0

            for train_idx, val_idx in cv_split(groups, n_folds, sample_X,
                                               labels, mode=mode):
                train_reps = {k: v[train_idx] for k, v in reps.items()}
                val_reps = {k: v[val_idx] for k, v in reps.items()}
                y_train, y_val = labels[train_idx], labels[val_idx]

                augs_full = data.get('augmentations') or {}
                train_augs = {k: v[train_idx] for k, v in augs_full.items()} or None
                val_augs = {k: v[val_idx] for k, v in augs_full.items()} or None

                try:
                    t0 = time.time()
                    X_train_h, feature_info = create_hybrid(
                        train_reps, y_train,
                        allocation_method='greedy',
                        budget=budget,
                        importance_method=importance_method,
                        augmentations=train_augs,
                        augmentation_strategy='capped',
                    )
                    X_val_h = apply_feature_selection(val_reps, feature_info)
                    aug_info = feature_info.get('augmentation_info', {})
                    if aug_info and val_augs:
                        aug_val = apply_augmentation_selection(val_augs, aug_info)
                        if aug_val is not None:
                            X_val_h = np.hstack([X_val_h, aug_val])
                    hybrid_build_time = time.time() - t0
                except Exception:
                    continue

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train_h)
                X_val_s = scaler.transform(X_val_h)
                last_n_features = X_train_h.shape[1]

                preds, ft, pt = _time_fit_predict(
                    model_factory, X_train_s, y_train, X_val_s)
                fold_hybrid_scores.append(metric_fn(y_val, preds))
                fold_hybrid_fit_times.append(ft + hybrid_build_time)
                fold_hybrid_pred_times.append(pt)

            if not fold_hybrid_scores:
                continue

            hyb_r2 = float(np.mean(fold_hybrid_scores))
            budget_label = 'full' if budget == total_dim else str(budget)
            print(f"    Hybrid b={budget_label:<5}: R2={hyb_r2:.4f} "
                  f"dim={last_n_features}")

            results.append({
                'model': model_name,
                'method': 'hybrid',
                'augmented': False,
                'budget': budget,
                'r2_mean': hyb_r2,
                'r2_std': float(np.std(fold_hybrid_scores)),
                'total_dim': last_n_features,
                'n_models_trained': 1,
                'total_fit_time_s': float(np.mean(fold_hybrid_fit_times)),
                'total_predict_time_s': float(np.mean(fold_hybrid_pred_times)),
            })

    # =====================================================================
    # Summary tables
    # =====================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    for model_name in models:
        model_results = [r for r in results if r['model'] == model_name]
        if not model_results:
            continue

        print(f"\n  {model_name}:")
        header = (f"    {'Method':<22} {'R2':>8} {'Dim':>6} "
                  f"{'Models':>7} {'Fit(s)':>8} {'Pred(s)':>8}")
        print(header)
        print("    " + "-" * 62)

        for r in model_results:
            method_label = r['method']
            if r['method'] == 'hybrid':
                b = r.get('budget', '?')
                b_label = 'full' if b == total_dim else str(b)
                method_label = f"hybrid b={b_label}"
            if r['augmented']:
                method_label += ' +aug'

            print(f"    {method_label:<22} {r['r2_mean']:>8.4f} "
                  f"{r['total_dim']:>6} {r['n_models_trained']:>7} "
                  f"{r['total_fit_time_s']:>8.3f} {r['total_predict_time_s']:>8.4f}")

    # Efficiency summary: hybrid vs ensemble
    print(f"\n  EFFICIENCY COMPARISON (best hybrid vs ensemble per model):")
    print(f"    {'Model':<12} {'Ens R2':>8} {'Hyb R2':>8} {'Delta':>8} "
          f"{'Ens #M':>7} {'Hyb Dim':>8} "
          f"{'Ens Fit':>8} {'Hyb Fit':>8} {'Speedup':>8}")
    print("    " + "-" * 82)

    for model_name in models:
        ens = [r for r in results if r['model'] == model_name
               and r['method'] == 'ensemble' and not r['augmented']]
        hybs = [r for r in results if r['model'] == model_name
                and r['method'] == 'hybrid' and not r['augmented']]

        if not ens or not hybs:
            continue

        ens_r = ens[0]
        best_hyb = max(hybs, key=lambda x: x['r2_mean'])

        delta = best_hyb['r2_mean'] - ens_r['r2_mean']
        speedup = ens_r['total_fit_time_s'] / best_hyb['total_fit_time_s'] if best_hyb['total_fit_time_s'] > 0 else 0

        print(f"    {model_name:<12} {ens_r['r2_mean']:>8.4f} {best_hyb['r2_mean']:>8.4f} "
              f"{delta:>+8.4f} {ens_r['n_models_trained']:>7} {best_hyb['total_dim']:>8} "
              f"{ens_r['total_fit_time_s']:>8.3f} "
              f"{best_hyb['total_fit_time_s']:>8.3f} {speedup:>7.1f}x")

    return {
        'results': results,
        'n_reps': n_reps,
        'total_dim': total_dim,
        'rep_names': rep_names,
    }


if __name__ == '__main__':
    parser = create_experiment_parser('C11: Ensemble vs Hybrid',
                                     has_delta=True, has_augment=True)
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")
    print(f"Models: {config['models']}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'c11_ensemble_vs_hybrid', args.dataset)
