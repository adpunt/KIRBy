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
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric,
    get_precomputed_search, build_fold_hybrid,
    get_best_for_downstream,
)


# =========================================================================
# MODEL FAMILIES WITH REGULARIZATION AXES
# =========================================================================

def _get_regularization_families(task_type='regression'):
    """Return {family: [(param_name, param_value, factory_fn), ...]}.

    Regression families use Ridge/RFR/SVR. Classification families use
    LogisticRegression/RFClassifier/SVC (the latter with probability=True so
    AUROC can use class-1 probabilities downstream). LogReg's C parameter is
    inverse regularization: smaller C → stronger L2 penalty (opposite
    convention to Ridge's alpha). The sweep values are chosen so that
    _best_fn picks the most-regularized model that still fits well.
    """
    if task_type == 'classification':
        return {
            'LogReg': [
                # C is inverse reg: small C = strong L2 penalty.
                ('C', 0.01, lambda: LogisticRegression(C=0.01, max_iter=2000)),
                ('C', 0.1, lambda: LogisticRegression(C=0.1, max_iter=2000)),
                ('C', 1.0, lambda: LogisticRegression(C=1.0, max_iter=2000)),
                ('C', 10.0, lambda: LogisticRegression(C=10.0, max_iter=2000)),
                ('C', 100.0, lambda: LogisticRegression(C=100.0, max_iter=2000)),
                ('C', 1000.0, lambda: LogisticRegression(C=1000.0, max_iter=2000)),
            ],
            'RF': [
                ('max_depth', 3, lambda: RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42, n_jobs=-1)),
                ('max_depth', 5, lambda: RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)),
                ('max_depth', 10, lambda: RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
                ('max_depth', None, lambda: RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)),
            ],
            'SVM': [
                ('C', 0.01, lambda: SVC(kernel='rbf', C=0.01, probability=True)),
                ('C', 0.1, lambda: SVC(kernel='rbf', C=0.1, probability=True)),
                ('C', 1.0, lambda: SVC(kernel='rbf', C=1.0, probability=True)),
                ('C', 10.0, lambda: SVC(kernel='rbf', C=10.0, probability=True)),
            ],
        }
    return {
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


# =========================================================================
# CORE EVALUATION
# =========================================================================

def _score_prediction(model, X_val, y_val, metric_fn, task_type):
    """Score predictions with metric-appropriate format (see c18 for rationale)."""
    if task_type == 'classification' and hasattr(model, 'predict_proba'):
        try:
            probs = model.predict_proba(X_val)
            if probs.shape[1] == 2:
                return metric_fn(y_val, probs[:, 1])
        except Exception:
            pass
    return metric_fn(y_val, model.predict(X_val))


def _evaluate_cell(reps, labels, groups, model_factory, budget, n_folds, mode,
                   metric_fn, augmentations=None, blueprint=None,
                   task_type='regression'):
    """Evaluate one (model, budget) cell. Returns (score, train_score)."""
    sample_X = next(iter(reps.values()))
    val_scores = []
    train_scores = []

    for train_idx, val_idx in cv_split(groups, n_folds, sample_X, labels, mode=mode):
        train_reps = {k: v[train_idx] for k, v in reps.items()}
        val_reps = {k: v[val_idx] for k, v in reps.items()}
        y_train, y_val = labels[train_idx], labels[val_idx]

        augs_full = augmentations or {}
        train_augs = {k: v[train_idx] for k, v in augs_full.items()} if augs_full else None
        val_augs = {k: v[val_idx] for k, v in augs_full.items()} if augs_full else None

        if budget is not None and budget < sum(v.shape[1] for v in reps.values()):
            if blueprint is None:
                continue
            try:
                X_train, X_val, fi = build_fold_hybrid(
                    train_reps, val_reps, y_train, blueprint,
                    augmentations_train=train_augs, augmentations_val=val_augs)
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
        val_scores.append(_score_prediction(model, X_va_s, y_val, metric_fn, task_type))
        train_scores.append(_score_prediction(model, X_tr_s, y_train, metric_fn, task_type))

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
    task_type = data.get('task_type', 'regression')
    total_dim = sum(v.shape[1] for v in reps.values())
    # Cap full_dim at 2048 to prevent MLP/SVM blowup on large concat spaces
    capped_full = min(total_dim, 2048)
    budgets = [b for b in [25, 50, 100, 200, 500, 1024, capped_full] if b <= total_dim]
    budgets = sorted(set(budgets))  # deduplicate

    families = _get_regularization_families(task_type=task_type)
    n_samples = len(labels)

    # SVM is slow on large datasets
    if n_samples > 5000 and 'SVM' in families:
        print(f"  Skipping SVM: n_samples={n_samples} > 5000")
        del families['SVM']

    augmentations = data.get('augmentations')

    # Pre-compute search blueprints per budget (cached)
    # NOTE: get_precomputed_search returns results across ALL budgets (budget
    # param is deprecated). Use get_best_for_downstream to select per-budget.
    blueprints = {}
    try:
        search = get_precomputed_search(
            reps, labels, groups,
            aug_condition='capped', augmentations=augmentations,
            speed=config.get('search_speed', 'fast'),
            precomputed=data.get('precomputed_searches'))
        for budget in budgets:
            best = get_best_for_downstream(search, 'rf', budget=budget)
            if best and 'blueprint' in best:
                blueprints[budget] = best['blueprint']
    except Exception:
        pass  # will skip all budgets

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
                    augmentations=augmentations,
                    blueprint=blueprints.get(budget),
                    task_type=task_type)
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
        _sentinel = -1e10 if not lower_is_better else 1e10
        _best_fn = min if lower_is_better else max
        best_tuned_full = _best_fn(full_vals, key=lambda x: x[1] if not np.isnan(x[1]) else _sentinel)

        # Simple hybrid: default regularization, best non-full budget
        default_pval = variants[len(variants) // 2][1]  # middle regularization
        hybrid_scores = [(b, heatmap[b].get(str(default_pval), {}).get('val', float('nan')))
                         for b in budgets if b < full_dim_budget]
        best_simple_hybrid = _best_fn(hybrid_scores, key=lambda x: x[1] if not np.isnan(x[1]) else _sentinel) if hybrid_scores else (None, float('nan'))

        if not any(np.isnan(x) for x in [best_simple_hybrid[1], best_tuned_full[1]]):
            if lower_is_better:
                hybrid_wins = best_simple_hybrid[1] < best_tuned_full[1]
            else:
                hybrid_wins = best_simple_hybrid[1] > best_tuned_full[1]
        else:
            hybrid_wins = None

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
            'hybrid_wins': hybrid_wins,
        }

        fam_result = {
            'param_name': param_name,
            'heatmap': convert_numpy(heatmap),
            'sensitivity_by_budget': convert_numpy(sensitivity),
            'comparison': convert_numpy(comparison),
        }

        # --- Analysis 3: L2 equivalence (linear family — Ridge or LogReg) ---
        if fam_name in ('Ridge', 'LogReg'):
            # For Ridge, the default regularization pval is 1.0 (alpha=1.0).
            # For LogReg, the default is 1.0 (C=1.0 = moderate regularization).
            # Both live at index len(variants)//2 in the variants list.
            default_pval = variants[len(variants) // 2][1]
            l2_equivalence = {}
            for budget in budgets:
                if budget >= full_dim_budget:
                    continue
                hybrid_gap = heatmap[budget].get(str(default_pval), {}).get('gap', float('nan'))
                if np.isnan(hybrid_gap):
                    continue
                # Find the regularization pval at full_dim that gives the same gap
                full_gaps = [(pval, heatmap[full_dim_budget].get(str(pval), {}).get('gap', float('nan')))
                             for _, pval, _ in variants]
                full_gaps = [(p, g) for p, g in full_gaps if not np.isnan(g)]
                if len(full_gaps) >= 2:
                    # Simple interpolation: find pval where gap ≈ hybrid_gap
                    pvals = np.array([p for p, _ in full_gaps])
                    gaps = np.array([g for _, g in full_gaps])
                    # Find nearest gap
                    idx = np.argmin(np.abs(gaps - hybrid_gap))
                    l2_equivalence[budget] = {
                        'hybrid_gap': hybrid_gap,
                        'equivalent_param': float(pvals[idx]),
                        'param_gap': float(gaps[idx]),
                    }
                    print(f"    L2 equiv: budget={budget} ≈ {param_name}={pvals[idx]:.3f} "
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
