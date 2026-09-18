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
    find_best_hybrid, get_domain_metric, make_cv_splitter,
    get_best_search_for_downstream,
    extract_top3_base_reps,
    INTERNAL_EVALUATORS, FULL_BUDGETS,
)
from kirby.hybrid import apply_augmentation_selection
from kirby.find_best_hybrid import _predict_for_metric


def _cal_split(train_groups, n_train, n_cal, fold_i):
    """Cut the conformal calibration rows out of a fold's training rows.

    Group-aware: a scaffold that appears in the proper-training half never
    appears in the calibration half, so the calibrated interval width comes
    from residuals as hard as the held-out fold's. Falls back to a permutation
    only when the fold has fewer than two distinct groups, where no group-aware
    cut exists.
    """
    if train_groups is not None and len(np.unique(train_groups)) >= 2:
        splitter, _ = make_cv_splitter(
            1, groups=train_groups, seed=42 + fold_i)
        dummy = np.zeros((n_train, 1))
        ptrain_idx, cal_idx = next(
            splitter.split(dummy, groups=train_groups))
        return ptrain_idx, cal_idx
    perm = np.random.RandomState(42 + fold_i).permutation(n_train)
    return perm[n_cal:], perm[:n_cal]


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
    augs = data.get('augmentations')
    optimal_model_factory = None
    optimal_budget = 100
    # Fallback when neither find_best_hybrid nor a c02 result supplies a
    # recipe: the engine's own current defaults. optimal_alloc was 'greedy',
    # which is cut.
    optimal_alloc = 'performance_weighted'
    optimal_imp = 'treeshap'
    # 'match' is the canonical Blueprint default (downstream==evaluator);
    # 'ridge' would force ridge as internal evaluator regardless of task and
    # would not match precomputed cache keys.
    optimal_internal_eval = 'match'
    # Match the actual augmentation presence (collision-safe default).
    optimal_aug_strategy = 'uncapped' if augs else 'none'
    optimal_desc = 'RF at budget=100 (default)'

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
        # Pool helper picks the best (tier, aug_condition) cache file per
        # downstream model. We try across all common models and pick the
        # overall winner (preserves the prior "best across all models" semantics).
        print(f"  Using pool helper to find optimal ()...")
        try:
            best = None
            best_model = None
            best_search = None
            best_meta = None
            for candidate_model in ['rf', 'svm', 'lightgbm', 'dnn', 'knn']:
                try:
                    s, m = get_best_search_for_downstream(
                        reps, labels, groups,
                        downstream_model=candidate_model, budget=None,
                        augmentations=augs,
                        dataset_name=dataset_name,
                        task_type=data['task_type'],
                        data=data,
                    )
                except Exception:
                    continue
                if not m:
                    continue
                bc = m['best_config']
                score = bc.get('score_mean')
                if score is None:
                    continue
                is_better = (best is None) or (
                    score < best['score_mean'] if lower_is_better
                    else score > best['score_mean'])
                if is_better:
                    best = bc
                    best_model = candidate_model
                    best_search = s
                    best_meta = m
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
                                f"[{optimal_imp}/{optimal_alloc}] "
                                f"(pool {best_meta['tier']}/{best_meta['aug_condition']})")
                print(f"  Discovered optimal: {optimal_desc}")
                top3 = extract_top3_base_reps(best_search)
                if top3:
                    print(f"  Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")
        except Exception as e:
            print(f"  pool helper failed ({e}), using RF/budget=100 default")

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

    # Underfit budget scales with total_dim so tiny rep sets aren't compared
    # to an "underfit" condition that happens to be essentially the same as
    # "overfit" (e.g. total_dim=40 → budget=25 is not really an underfit).
    underfit_budget = min(25, max(5, total_dim // 4))
    condition_specs = {
        'optimal': (optimal_model_factory, optimal_budget, optimal_alloc, optimal_imp,
                    optimal_desc),
        'underfit': (optimal_model_factory, underfit_budget, optimal_alloc, optimal_imp,
                     f'Same model at budget={underfit_budget} (too compressed)'),
        'overfit': (optimal_model_factory, total_dim, optimal_alloc, optimal_imp,
                    f'Same model at full dim={total_dim} (too many features)'),
        # The mismatch under test is the MODEL at full width, not the
        # allocation. Was 'greedy', which is cut and costs 20-100x for nothing.
        'mismatched': (mismatched_factory, total_dim, 'performance_weighted',
                       'treeshap', mismatched_desc),
    }

    # Condition-distinctness diagnostic. The calibration comparison is only
    # meaningful if optimal / underfit / overfit use genuinely different budgets.
    # (Earlier analysis mis-stated that "optimal == full dim" on EGFR — in fact
    # optimal=659 vs total_dim=68948, so they ARE separated and the "underfit
    # calibrates best" result is a real finding, not a collapse. This field lets
    # the rerun confirm separation per dataset.) See
    # COMPLEXITY_SIMILARITY_BROKEN_2026_06_03.md.
    _cond_budgets = {'optimal': optimal_budget, 'underfit': underfit_budget,
                     'overfit': total_dim}
    results = {
        'conditions': {},
        'task_type': task_type,
        'total_dim': total_dim,
        'condition_budgets': _cond_budgets,
        'conditions_distinct': len(set(_cond_budgets.values())) == len(_cond_budgets),
    }
    if not results['conditions_distinct']:
        print(f"  WARNING: calibration conditions collapse (budgets "
              f"{_cond_budgets}) — optimal/underfit/overfit not separated.")

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
                # model/metric_fn/lower_is_better/is_clf are passed so the
                # budget split is decided by the estimator and the metric this
                # fold is actually scored with. Without model, internal
                # evaluator 'match' resolves to lightgbm for every condition,
                # including the random-forest ones; without metric_fn the
                # allocation ranking falls back to r2_score or accuracy instead
                # of the dataset's own metric.
                X_train, feature_info = create_hybrid(
                    train_reps, y_train, groups=groups[train_idx],
                    allocation_method=alloc,
                    budget=budget,
                    importance_method=imp,
                    internal_evaluator=optimal_internal_eval,
                    augmentations=train_augs,
                    augmentation_strategy=optimal_aug_strategy,
                    model=model_factory(),
                    metric_fn=metric_fn,
                    lower_is_better=lower_is_better,
                    is_clf=is_classification,
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
                # Conformal classification: split train into proper-train + cal
                n_train = len(y_train)
                n_cal = max(20, int(n_train * 0.2))
                # The calibration rows are cut on scaffold groups, not at
                # random. A random 20% puts molecules from the same scaffold on
                # both sides, so the residuals the interval is calibrated from
                # are easier than the held-out fold the coverage is reported on.
                ptrain_idx, cal_idx = _cal_split(
                    groups[train_idx], n_train, n_cal, fold_i)

                X_ptrain, y_ptrain = X_train_s[ptrain_idx], y_train[ptrain_idx]
                X_cal, y_cal = X_train_s[cal_idx], y_train[cal_idx]

                model_conf = model_factory()
                model_conf.fit(X_ptrain, y_ptrain)

                # Compute val_score from model_conf so the performance number
                # and the calibration metrics come from the SAME model (was a
                # full-train vs proper-train mismatch).  For AUC metrics, use
                # predict_proba (hard labels would give a meaningless AUROC).
                _from_kfh = _predict_for_metric(model_conf, X_val_s, metric_fn)
                val_score = float(metric_fn(y_val, _from_kfh))

                # Conformal classification: nonconformity = 1 - P(true class)
                if hasattr(model_conf, 'predict_proba'):
                    cal_probs = model_conf.predict_proba(X_cal)
                    # Map y_cal labels to column indices in model_conf.classes_.
                    # Calibration points whose true label was never seen in proper-train
                    # are reported and assigned worst-case nonconformity (1.0) rather
                    # than silently dropped (which would bias coverage).
                    class_to_col = {c: j for j, c in enumerate(model_conf.classes_)}
                    cal_scores = np.empty(len(y_cal), dtype=float)
                    n_unseen = 0
                    for i in range(len(y_cal)):
                        col = class_to_col.get(int(y_cal[i]))
                        if col is None:
                            cal_scores[i] = 1.0
                            n_unseen += 1
                        else:
                            cal_scores[i] = 1 - cal_probs[i, col]
                    if n_unseen:
                        print(f"    fold {fold_i}: {n_unseen}/{len(y_cal)} "
                              f"calibration points have classes unseen in proper-train "
                              f"(assigned worst-case nonconformity 1.0)")
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
                    # Sharpness: avg set size / n_classes (≈ 1/n_classes ⇒ sharp/decisive;
                    # 1 ⇒ trivial full-prediction-set). LOWER is better.
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
                # Read from fold_result (defined in both branches) — the locals
                # picp/avg_set_size/cal_error are only assigned inside the
                # predict_proba branch, so a classification model WITHOUT
                # predict_proba (e.g. SVC without probability=True) used to crash
                # this print with NameError. (OTHER-22)
                print(f"    fold {fold_i}: {metric_name}="
                      f"{fold_result['val_score']:.4f}, "
                      f"PICP={fold_result['picp']:.3f}, "
                      f"set_size={fold_result['avg_set_size']:.2f}, "
                      f"cal_error={fold_result['cal_error']:.3f}")
            else:
                # Regression: conformal prediction intervals.
                # val_r2 must come from model_conf (not `model` trained on full
                # y_train) so the reported performance number and the
                # calibration metrics (PICP/MPIW) are from the same model.
                n_train = len(y_train)
                n_cal = max(20, int(n_train * 0.2))
                # The calibration rows are cut on scaffold groups, not at
                # random. A random 20% puts molecules from the same scaffold on
                # both sides, so the residuals the interval is calibrated from
                # are easier than the held-out fold the coverage is reported on.
                ptrain_idx, cal_idx = _cal_split(
                    groups[train_idx], n_train, n_cal, fold_i)

                X_ptrain, y_ptrain = X_train_s[ptrain_idx], y_train[ptrain_idx]
                X_cal, y_cal = X_train_s[cal_idx], y_train[cal_idx]

                model_conf = model_factory()
                model_conf.fit(X_ptrain, y_ptrain)

                y_pred_val = model_conf.predict(X_val_s)
                val_r2 = metric_fn(y_val, y_pred_val)

                y_pred_cal = model_conf.predict(X_cal)
                cal_residuals = np.abs(y_cal - y_pred_cal)

                # Conformal quantile with finite-sample correction (Romano et al. 2019)
                alpha = 0.1
                n_cal = len(cal_residuals)
                conformal_level = min(1.0, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal)
                q = np.quantile(cal_residuals, conformal_level)

                # Reuse y_pred_val above (same model, same X_val_s).
                interval_lo = y_pred_val - q
                interval_hi = y_pred_val + q

                covered = (y_val >= interval_lo) & (y_val <= interval_hi)
                picp = float(np.mean(covered))
                mpiw = float(np.mean(interval_hi - interval_lo))
                target_range = float(np.ptp(y_val))
                sharpness_ratio = mpiw / target_range if target_range > 0 else float('nan')
                cal_error = abs(picp - (1 - alpha))

                errors = np.abs(y_val - y_pred_val)  # 2026-06-04 fix: y_pred_conf was undefined (NameError crashed c08 on every regression dataset); same model/X_val per L376
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
            # Union of keys across folds, missing-key-tolerant: a classification
            # model without predict_proba emits fewer keys (no 'n_classes') than
            # one with it. If folds ever disagree on key sets, indexing
            # fold_metrics[0] keys with fm[key] would KeyError. (OTHER-22)
            all_keys = set()
            for fm in fold_metrics:
                all_keys.update(fm.keys())
            for key in all_keys:
                vals = [fm[key] for fm in fold_metrics
                        if key in fm and not np.isnan(fm[key])]
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
            print(f"  {cond_name:<14} {a.get('val_r2', float('nan')):.4f}  "
                  f"{a.get('picp', float('nan')):.3f}  "
                  f"{a.get('mpiw', float('nan')):.3f}  "
                  f"{a.get('cal_error', float('nan')):.4f}    {cond_data['budget']}")

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
