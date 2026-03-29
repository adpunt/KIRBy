#!/usr/bin/env python3
"""C16: Feature Selection as Regularization — the hybrid's budget subsumes model-level regularization.

Paper: Complexity
Core thesis: Selecting features via the hybrid is a form of inductive bias equivalent
    to explicit regularization (L2 penalty, max_depth, C). At the optimal hybrid budget,
    model regularization becomes irrelevant because feature selection already controls
    the bias-variance tradeoff.

Three analyses:
  1. Heatmap: R² over (regularization_strength × hybrid_budget) per model family
  2. Sensitivity: std(R² across regularization) at each budget — should → 0 at optimal budget
  3. Equivalence: for Ridge, compute what L2 alpha is equivalent to hybrid budget=k

Usage:
    python c16_feature_selection_as_regularization.py --dataset logd --reps all --mode intermediary
    python c16_feature_selection_as_regularization.py --dataset flab_thermo_tresanco --mode intermediary
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric,
)
from kirby.hybrid import create_hybrid
from kirby.hybrid import apply_augmentation_selection


# =========================================================================
# MODEL FAMILIES WITH REGULARIZATION AXES
# =========================================================================

def _get_regularization_families():
    """Return {family: [(param_name, param_value, factory_fn), ...]}."""
    families = {
        'Ridge': [
            ('alpha', 0.001, lambda: Ridge(alpha=0.001)),
            ('alpha', 0.01, lambda: Ridge(alpha=0.01)),
            ('alpha', 0.1, lambda: Ridge(alpha=0.1)),
            ('alpha', 1.0, lambda: Ridge(alpha=1.0)),
            ('alpha', 10.0, lambda: Ridge(alpha=10.0)),
            ('alpha', 100.0, lambda: Ridge(alpha=100.0)),
        ],
        'RF': [
            ('max_depth', 3, lambda: RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42, n_jobs=-1)),
            ('max_depth', 5, lambda: RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)),
            ('max_depth', 10, lambda: RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
            ('max_depth', None, lambda: RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)),
        ],
        'SVM': [
            ('C', 0.01, lambda: SVR(kernel='rbf', C=0.01)),
            ('C', 0.1, lambda: SVR(kernel='rbf', C=0.1)),
            ('C', 1.0, lambda: SVR(kernel='rbf', C=1.0)),
            ('C', 10.0, lambda: SVR(kernel='rbf', C=10.0)),
        ],
    }
    return families


# =========================================================================
# CORE EVALUATION
# =========================================================================

def _evaluate_cell(reps, labels, groups, model_factory, budget, n_folds, mode,
                   metric_fn, augmentations=None):
    """Evaluate one (model, budget) cell. Returns (score, train_score)."""
    sample_X = next(iter(reps.values()))
    val_scores = []
    train_scores = []

    for train_idx, val_idx in cv_split(groups, n_folds, sample_X, labels, mode=mode):
        train_reps = {k: v[train_idx] for k, v in reps.items()}
        val_reps = {k: v[val_idx] for k, v in reps.items()}
        y_train, y_val = labels[train_idx], labels[val_idx]

        augs_full = augmentations or {}
        train_augs = {k: v[train_idx] for k, v in augs_full.items()} or None
        val_augs = {k: v[val_idx] for k, v in augs_full.items()} or None

        if budget is not None and budget < sum(v.shape[1] for v in reps.values()):
            try:
                X_train, fi = create_hybrid(
                    train_reps, y_train,
                    allocation_method='entropy_1.0',
                    budget=budget,
                    importance_method='lightgbm_gain',
                    augmentations=train_augs,
                    augmentation_strategy='capped',
                )
                from kirby.hybrid import apply_feature_selection
                X_val = apply_feature_selection(val_reps, fi)
                aug_info = fi.get('augmentation_info', {})
                if aug_info and val_augs:
                    aug_val = apply_augmentation_selection(val_augs, aug_info)
                    if aug_val is not None:
                        X_val = np.hstack([X_val, aug_val])
            except Exception:
                continue
        else:
            # Full dimensionality — just concat
            X_train = np.hstack([v[train_idx] for v in reps.values()])
            X_val = np.hstack([v[val_idx] for v in reps.values()])

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_va_s = scaler.transform(X_val)

        model = model_factory()
        model.fit(X_tr_s, y_train)
        val_scores.append(metric_fn(y_val, model.predict(X_va_s)))
        train_scores.append(metric_fn(y_train, model.predict(X_tr_s)))

    if not val_scores:
        return float('nan'), float('nan')
    return float(np.mean(val_scores)), float(np.mean(train_scores))


# =========================================================================
# RUN
# =========================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """C16: Feature selection as regularization."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C16: FEATURE SELECTION AS REGULARIZATION")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    total_dim = sum(v.shape[1] for v in reps.values())
    # Cap full_dim at 2048 to prevent MLP/SVM blowup on large concat spaces
    capped_full = min(total_dim, 2048)
    budgets = [b for b in [25, 50, 100, 200, 500, 1024, capped_full] if b <= total_dim]
    budgets = sorted(set(budgets))  # deduplicate

    families = _get_regularization_families()
    n_samples = len(labels)

    # SVM is slow on large datasets
    if n_samples > 5000 and 'SVM' in families:
        print(f"  Skipping SVM: n_samples={n_samples} > 5000")
        del families['SVM']

    augmentations = data.get('augmentations')

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'budgets': budgets,
        'total_dim': total_dim,
        'families': {},
    }

    for fam_name, variants in families.items():
        print(f"\n  --- {fam_name} ({len(variants)} regularization levels × {len(budgets)} budgets) ---")
        param_name = variants[0][0]

        heatmap = {}  # {budget: {param_value: {'val': score, 'train': score}}}
        for budget in budgets:
            heatmap[budget] = {}
            for pname, pval, factory in variants:
                val_score, train_score = _evaluate_cell(
                    reps, labels, groups, factory, budget, n_folds, mode, metric_fn,
                    augmentations=augmentations)
                heatmap[budget][str(pval)] = {
                    'val': val_score,
                    'train': train_score,
                    'gap': train_score - val_score,
                }
            # Print row
            scores_str = ', '.join(f'{pval}={heatmap[budget][str(pval)]["val"]:.4f}'
                                   for _, pval, _ in variants
                                   if str(pval) in heatmap[budget])
            print(f"    budget={budget}: {scores_str}")

        # --- Analysis 1: Regularization sensitivity per budget ---
        sensitivity = {}
        for budget in budgets:
            vals = [heatmap[budget][str(pval)]['val']
                    for _, pval, _ in variants if str(pval) in heatmap[budget]]
            vals = [v for v in vals if not np.isnan(v)]
            sensitivity[budget] = float(np.std(vals)) if len(vals) >= 2 else float('nan')

        # --- Analysis 2: Best-tuned-full-dim vs simple-hybrid ---
        full_dim_budget = max(budgets)
        full_vals = [(pval, heatmap[full_dim_budget].get(str(pval), {}).get('val', float('nan')))
                     for _, pval, _ in variants]
        best_tuned_full = max(full_vals, key=lambda x: x[1] if not np.isnan(x[1]) else -1e10)

        # Simple hybrid: default regularization, best non-full budget
        default_pval = variants[len(variants) // 2][1]  # middle regularization
        hybrid_scores = [(b, heatmap[b].get(str(default_pval), {}).get('val', float('nan')))
                         for b in budgets if b < full_dim_budget]
        best_simple_hybrid = max(hybrid_scores, key=lambda x: x[1] if not np.isnan(x[1]) else -1e10) if hybrid_scores else (None, float('nan'))

        comparison = {
            'best_tuned_full_dim': {
                'param_value': best_tuned_full[0],
                'score': best_tuned_full[1],
            },
            'simple_hybrid': {
                'budget': best_simple_hybrid[0],
                'default_param': default_pval,
                'score': best_simple_hybrid[1],
            },
            'hybrid_wins': best_simple_hybrid[1] > best_tuned_full[1] if not any(
                np.isnan(x) for x in [best_simple_hybrid[1], best_tuned_full[1]]) else None,
        }

        fam_result = {
            'param_name': param_name,
            'heatmap': convert_numpy(heatmap),
            'sensitivity_by_budget': convert_numpy(sensitivity),
            'comparison': convert_numpy(comparison),
        }

        # --- Analysis 3: L2 equivalence (Ridge only) ---
        if fam_name == 'Ridge':
            l2_equivalence = {}
            for budget in budgets:
                if budget >= full_dim_budget:
                    continue
                hybrid_gap = heatmap[budget].get('1.0', {}).get('gap', float('nan'))
                if np.isnan(hybrid_gap):
                    continue
                # Find alpha at full_dim that gives the same gap
                full_gaps = [(pval, heatmap[full_dim_budget].get(str(pval), {}).get('gap', float('nan')))
                             for _, pval, _ in variants]
                full_gaps = [(p, g) for p, g in full_gaps if not np.isnan(g)]
                if len(full_gaps) >= 2:
                    # Simple interpolation: find alpha where gap ≈ hybrid_gap
                    alphas = np.array([p for p, _ in full_gaps])
                    gaps = np.array([g for _, g in full_gaps])
                    # Find nearest gap
                    idx = np.argmin(np.abs(gaps - hybrid_gap))
                    l2_equivalence[budget] = {
                        'hybrid_gap': hybrid_gap,
                        'equivalent_alpha': float(alphas[idx]),
                        'alpha_gap': float(gaps[idx]),
                    }
                    print(f"    L2 equiv: budget={budget} ≈ alpha={alphas[idx]:.3f} "
                          f"(gap diff={abs(hybrid_gap - gaps[idx]):.4f})")

            fam_result['l2_equivalence'] = convert_numpy(l2_equivalence)

        results['families'][fam_name] = fam_result
        print(f"    Sensitivity: {', '.join(f'b={b}:{s:.4f}' for b, s in sensitivity.items())}")
        if comparison.get('hybrid_wins') is not None:
            winner = 'simple_hybrid' if comparison['hybrid_wins'] else 'tuned_full_dim'
            print(f"    Winner: {winner}")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser(
        'C16: Feature Selection as Regularization')
    args = parser.parse_args()
    config = get_config(args)
    print(f"Dataset: {args.dataset}")
    print(f"Mode:    {args.mode}")
    print(f"Reps:    {args.reps}")

    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'c16_feature_selection_as_regularization',
                            args.dataset)
