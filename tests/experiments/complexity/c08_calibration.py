#!/usr/bin/env python3
"""C08: Uncertainty Calibration at Optimal vs Suboptimal Complexity.

Paper: Complexity
Conclusion: Complexity matching improves calibration.
Dependencies: c02 (optional, reads JSON for optimal config).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, load_experiment_results, convert_numpy,
    get_model_factories, _cv_split,
    create_hybrid, apply_feature_selection,
    find_best_hybrid, get_domain_metric,
    extract_top3_base_reps,
    INTERNAL_EVALUATORS, SCREENING_BUDGETS, FULL_BUDGETS,
)
from kirby.hybrid import apply_augmentation_selection


def run(data, config, output_dir, dataset_name, conditions=None, c02_results_dir=None):
    """Exp 8: Complexity matching produces better-calibrated uncertainty estimates.

    The 'optimal' condition uses find_best_hybrid() (or Exp 2 results) to
    discover the actual best config. Suboptimal conditions are deliberately
    constructed to show the contrast.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 8: UNCERTAINTY CALIBRATION AT OPTIMAL VS SUBOPTIMAL COMPLEXITY")
    print("=" * 70)

    # Domain-aware metric
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    task_type = data['task_type']
    is_classification = (task_type == 'classification')

    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.linear_model import Ridge, LogisticRegression

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    n_folds = config['n_splits']
    search_speed = config['search_speed']
    total_dim = sum(v.shape[1] for v in reps.values())

    all_conditions = conditions or ['optimal', 'underfit', 'overfit', 'mismatched']

    # Try to load Exp 2 results if a directory was provided
    exp2_results = None
    if c02_results_dir:
        exp2_results = load_experiment_results(c02_results_dir, 'c02_complexity_matching_grid',
                                               dataset_name)
        if exp2_results:
            print(f"  Loaded Exp 2 results from {c02_results_dir}")
        else:
            print(f"  No Exp 2 results found at {c02_results_dir}")

    # Discover optimal config via find_best_hybrid or Exp 2 results
    optimal_model_factory = None
    optimal_budget = 100
    optimal_alloc = 'greedy'
    optimal_imp = 'treeshap'
    optimal_internal_eval = 'ridge'
    optimal_aug_strategy = 'uncapped'
    optimal_desc = 'RF at budget=100 (default)'
    augs = data.get('augmentations')

    if exp2_results and exp2_results.get('best_by_model'):
        # Use best config from Exp 2
        all_best = exp2_results['best_by_model']
        overall_best = (min if lower_is_better else max)(all_best.values(), key=lambda c: c['score_mean'])
        optimal_budget = overall_best['budget']
        optimal_alloc = overall_best['allocation']
        optimal_imp = overall_best['importance']
        best_model = overall_best['model']
        optimal_model_factory = get_model_factories(
            [best_model], task_type=data['task_type'],
            n_samples=len(data['labels'])).get(best_model)
        optimal_desc = (f"{best_model} at budget={optimal_budget} "
                        f"[{optimal_imp}/{optimal_alloc}] (from Exp 2)")
        print(f"  Using Exp 2 optimal: {optimal_desc}")
    else:
        # Run find_best_hybrid to discover optimal
        print(f"  Running find_best_hybrid(speed={search_speed}) to find optimal...")
        try:
            search = find_best_hybrid(
                reps, labels, groups=groups,
                speed=search_speed,
                budgets=FULL_BUDGETS,
                augmentations=augs,
                augmentation_strategy='uncapped' if augs else 'none',
                internal_evaluators=INTERNAL_EVALUATORS,
                metric=metric_name,
                verbose=False,
                cache=True,
                dataset_name=dataset_name,
                tier='general',
            )
            # Pick the best result across all models
            best = None
            best_model = None
            for m_name, m_best in search.best_by_model.items():
                if best is None or (m_best['score_mean'] < best['score_mean'] if lower_is_better else m_best['score_mean'] > best['score_mean']):
                    best = m_best
                    best_model = m_name
            if best:
                optimal_budget = best['blueprint'].budget
                optimal_alloc = best['blueprint'].allocation_method
                optimal_imp = best['blueprint'].importance_method
                optimal_internal_eval = best['blueprint'].internal_evaluator
                optimal_aug_strategy = best['blueprint'].augmentation_strategy
                best_model = best.get('model', best_model)
                optimal_model_factory = get_model_factories(
                    [best_model], task_type=data['task_type'],
                    n_samples=len(data['labels'])).get(best_model)
                optimal_desc = (f"{best_model} at budget={optimal_budget} "
                                f"[{optimal_imp}/{optimal_alloc}] (from find_best_hybrid)")
                print(f"  Discovered optimal: {optimal_desc}")
                top3 = extract_top3_base_reps(search)
                if top3:
                    print(f"  Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")
        except Exception as e:
            print(f"  find_best_hybrid failed ({e}), using RF/budget=100 default")

    if optimal_model_factory is None:
        if is_classification:
            optimal_model_factory = lambda: RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1)
        else:
            optimal_model_factory = lambda: RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1)

    # Define conditions: (model_factory, budget, alloc, imp, description)
    if is_classification:
        mismatched_factory = lambda: LogisticRegression(C=100, max_iter=1000, random_state=42)
        mismatched_desc = 'LogReg at full dim (low-capacity model, high-dim input)'
    else:
        mismatched_factory = lambda: Ridge(alpha=0.01)
        mismatched_desc = 'Ridge at full dim (low-capacity model, high-dim input)'

    condition_specs = {
        'optimal': (optimal_model_factory, optimal_budget, optimal_alloc, optimal_imp,
                    optimal_desc),
        'underfit': (optimal_model_factory, 25, optimal_alloc, optimal_imp,
                     f'Same model at budget=25 (too compressed)'),
        'overfit': (optimal_model_factory, total_dim, optimal_alloc, optimal_imp,
                    f'Same model at full dim={total_dim} (too many features)'),
        'mismatched': (mismatched_factory, total_dim, 'greedy', 'treeshap',
                       mismatched_desc),
    }

    results = {'conditions': {}, 'task_type': task_type}

    for cond_name in all_conditions:
        if cond_name not in condition_specs:
            continue
        model_factory, budget, alloc, imp, desc = condition_specs[cond_name]
        print(f"\n  {cond_name}: {desc}")

        sample_X = next(iter(reps.values()))

        fold_metrics = []
        for fold_i, (train_idx, val_idx) in enumerate(_cv_split(groups, n_folds, sample_X, labels)):
            train_reps = {k: v[train_idx] for k, v in reps.items()}
            val_reps = {k: v[val_idx] for k, v in reps.items()}
            y_train, y_val = labels[train_idx], labels[val_idx]
            train_augs = {k: v[train_idx] for k, v in augs.items()} if augs else None
            val_augs = {k: v[val_idx] for k, v in augs.items()} if augs else None

            try:
                X_train, feature_info = create_hybrid(
                    train_reps, y_train,
                    allocation_method=alloc,
                    budget=budget,
                    importance_method=imp,
                    internal_evaluator=optimal_internal_eval,
                    augmentations=train_augs,
                    augmentation_strategy=optimal_aug_strategy,
                )
                X_val = apply_feature_selection(val_reps, feature_info)
                aug_info = feature_info.get('augmentation_info')
                if aug_info and val_augs:
                    X_val_aug = apply_augmentation_selection(val_augs, aug_info)
                    if X_val_aug is not None:
                        X_val = np.hstack([X_val, X_val_aug])
            except Exception as e:
                print(f"    fold {fold_i}: hybrid creation failed - {e}")
                continue

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)

            # Train base model
            model = model_factory()
            model.fit(X_train_s, y_train)

            if is_classification:
                # Classification calibration via Platt scaling / prediction sets
                val_score = metric_fn(y_val, model.predict(X_val_s))

                # Conformal classification: split train into proper-train + cal
                n_train = len(y_train)
                n_cal = max(20, int(n_train * 0.2))
                cal_perm = np.random.RandomState(42 + fold_i).permutation(n_train)
                cal_idx = cal_perm[:n_cal]
                ptrain_idx = cal_perm[n_cal:]

                X_ptrain, y_ptrain = X_train_s[ptrain_idx], y_train[ptrain_idx]
                X_cal, y_cal = X_train_s[cal_idx], y_train[cal_idx]

                model_conf = model_factory()
                model_conf.fit(X_ptrain, y_ptrain)

                # Conformal classification: nonconformity = 1 - P(true class)
                if hasattr(model_conf, 'predict_proba'):
                    cal_probs = model_conf.predict_proba(X_cal)
                    cal_scores = np.array([1 - cal_probs[i, int(y_cal[i])]
                                           for i in range(len(y_cal))
                                           if int(y_cal[i]) < cal_probs.shape[1]])
                    if len(cal_scores) == 0:
                        print(f"    fold {fold_i}: no valid calibration scores")
                        continue

                    alpha = 0.1
                    conformal_level = min(1.0, np.ceil((len(cal_scores) + 1) * (1 - alpha)) / len(cal_scores))
                    q = np.quantile(cal_scores, conformal_level)

                    # Prediction sets on val
                    val_probs = model_conf.predict_proba(X_val_s)
                    classes = model_conf.classes_
                    pred_sets = []
                    for i in range(len(y_val)):
                        pset = [classes[j] for j in range(len(classes))
                                if 1 - val_probs[i, j] <= q]
                        pred_sets.append(pset)

                    # Coverage: true label in prediction set
                    covered = np.array([y_val[i] in pred_sets[i] for i in range(len(y_val))])
                    picp = float(np.mean(covered))
                    avg_set_size = float(np.mean([len(ps) for ps in pred_sets]))
                    n_classes = len(classes)
                    # Sharpness: avg set size / n_classes (1/n_classes = trivial, 1 = perfect)
                    sharpness_ratio = avg_set_size / n_classes if n_classes > 0 else float('nan')
                    cal_error = abs(picp - (1 - alpha))

                    # Uncertainty-error: entropy of predicted probabilities vs correctness
                    pred_labels = model_conf.predict(X_val_s)
                    correct = (pred_labels == y_val).astype(float)
                    entropy = -np.sum(val_probs * np.log(val_probs + 1e-12), axis=1)
                    ue_corr = float('nan')
                    if len(entropy) >= 3:
                        ue_corr_val, _ = spearmanr(entropy, 1 - correct)
                        ue_corr = float(ue_corr_val)

                    fold_result = {
                        'val_score': val_score, 'picp': picp,
                        'avg_set_size': avg_set_size,
                        'sharpness_ratio': sharpness_ratio, 'cal_error': cal_error,
                        'conformal_q': float(q), 'n_val': len(y_val),
                        'n_classes': n_classes,
                        'ue_corr': ue_corr,
                    }
                else:
                    # No predict_proba — report score only
                    fold_result = {
                        'val_score': val_score, 'picp': float('nan'),
                        'avg_set_size': float('nan'),
                        'sharpness_ratio': float('nan'), 'cal_error': float('nan'),
                        'conformal_q': float('nan'), 'n_val': len(y_val),
                        'ue_corr': float('nan'),
                    }
                fold_metrics.append(fold_result)
                print(f"    fold {fold_i}: {metric_name}={val_score:.4f}, PICP={picp:.3f}, "
                      f"set_size={avg_set_size:.2f}, cal_error={cal_error:.3f}")
            else:
                # Regression: conformal prediction intervals (existing logic)
                y_pred_val = model.predict(X_val_s)
                val_r2 = metric_fn(y_val, y_pred_val)

                n_train = len(y_train)
                n_cal = max(20, int(n_train * 0.2))
                cal_perm = np.random.RandomState(42 + fold_i).permutation(n_train)
                cal_idx = cal_perm[:n_cal]
                ptrain_idx = cal_perm[n_cal:]

                X_ptrain, y_ptrain = X_train_s[ptrain_idx], y_train[ptrain_idx]
                X_cal, y_cal = X_train_s[cal_idx], y_train[cal_idx]

                model_conf = model_factory()
                model_conf.fit(X_ptrain, y_ptrain)

                y_pred_cal = model_conf.predict(X_cal)
                cal_residuals = np.abs(y_cal - y_pred_cal)

                # Conformal quantile with finite-sample correction (Romano et al. 2019)
                alpha = 0.1
                n_cal = len(cal_residuals)
                conformal_level = min(1.0, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal)
                q = np.quantile(cal_residuals, conformal_level)

                y_pred_conf = model_conf.predict(X_val_s)
                interval_lo = y_pred_conf - q
                interval_hi = y_pred_conf + q

                covered = (y_val >= interval_lo) & (y_val <= interval_hi)
                picp = float(np.mean(covered))
                mpiw = float(np.mean(interval_hi - interval_lo))
                target_range = float(np.ptp(y_val))
                sharpness_ratio = mpiw / target_range if target_range > 0 else float('nan')
                cal_error = abs(picp - (1 - alpha))

                errors = np.abs(y_val - y_pred_conf)
                ue_corr = float('nan')
                if hasattr(model_conf, 'estimators_') and len(errors) >= 3:
                    tree_preds = np.array([t.predict(X_val_s) for t in model_conf.estimators_])
                    per_sample_unc = tree_preds.std(axis=0)
                    ue_corr, _ = spearmanr(per_sample_unc, errors)
                    ue_corr = float(ue_corr)

                fold_result = {
                    'val_r2': val_r2, 'picp': picp, 'mpiw': mpiw,
                    'sharpness_ratio': sharpness_ratio, 'cal_error': cal_error,
                    'conformal_q': float(q), 'n_val': len(y_val),
                    'ue_corr': ue_corr,
                }
                fold_metrics.append(fold_result)
                print(f"    fold {fold_i}: {metric_name}={val_r2:.4f}, PICP={picp:.3f}, "
                      f"MPIW={mpiw:.3f}, cal_error={cal_error:.3f}")

        if fold_metrics:
            avg = {}
            for key in fold_metrics[0]:
                vals = [fm[key] for fm in fold_metrics if not np.isnan(fm[key])]
                avg[key] = float(np.mean(vals)) if vals else float('nan')
            results['conditions'][cond_name] = {
                'description': desc,
                'budget': budget,
                'avg': avg,
                'folds': fold_metrics,
            }
            score_key = 'val_score' if is_classification else 'val_r2'
            picp_val = avg.get('picp', float('nan'))
            if is_classification:
                set_key = avg.get('avg_set_size', float('nan'))
                print(f"  >> {cond_name} avg: {metric_name}={avg.get(score_key, float('nan')):.4f}, "
                      f"PICP={picp_val:.3f}, set_size={set_key:.2f}, "
                      f"cal_error={avg.get('cal_error', float('nan')):.3f}")
            else:
                print(f"  >> {cond_name} avg: {metric_name}={avg.get(score_key, float('nan')):.4f}, "
                      f"PICP={picp_val:.3f}, MPIW={avg.get('mpiw', float('nan')):.3f}, "
                      f"cal_error={avg.get('cal_error', float('nan')):.3f}")

    # Summary table
    print("\n  CALIBRATION SUMMARY:")
    if is_classification:
        print(f"  {'Condition':<14} {metric_name:<8} {'PICP':<8} {'SetSize':<8} {'Cal Error':<10} {'Budget':<8}")
        print("  " + "-" * 56)
        for cond_name, cond_data in results['conditions'].items():
            a = cond_data['avg']
            print(f"  {cond_name:<14} {a.get('val_score', float('nan')):.4f}  "
                  f"{a.get('picp', float('nan')):.3f}  "
                  f"{a.get('avg_set_size', float('nan')):.2f}    "
                  f"{a.get('cal_error', float('nan')):.4f}    {cond_data['budget']}")
    else:
        print(f"  {'Condition':<14} {metric_name:<8} {'PICP':<8} {'MPIW':<8} {'Cal Error':<10} {'Budget':<8}")
        print("  " + "-" * 56)
        for cond_name, cond_data in results['conditions'].items():
            a = cond_data['avg']
            print(f"  {cond_name:<14} {a['val_r2']:.4f}  {a['picp']:.3f}  "
                  f"{a['mpiw']:.3f}  {a['cal_error']:.4f}    {cond_data['budget']}")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('C08: Uncertainty Calibration',
                                     has_delta=True, has_augment=True)
    parser.add_argument('--c02-results', type=str, default=None,
                        help='Path to c02 output directory (reads optimal config from JSON)')
    parser.add_argument('--conditions', type=str, nargs='+', default=None,
                        choices=['optimal', 'underfit', 'overfit', 'mismatched'],
                        help='Which conditions to test (default: all)')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset,
                  conditions=args.conditions,
                  c02_results_dir=args.c02_results)
    save_experiment_results(results, args.output, 'c08_calibration', args.dataset)
