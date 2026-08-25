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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric,
    build_fold_hybrid,
    get_best_search_for_downstream,
)
from kirby.find_best_hybrid import _resolve_models, DEFAULT_MODELS


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
    """Evaluate one (model, budget) cell. Returns (score, train_score, fold_scores).

    fold_scores is the list of per-fold VALIDATION scores (same values that are
    averaged into `score`). It is threaded out so the honest leave-one-fold-out
    re-score can re-pick the winning budget / regularization per held-out fold
    (see the honest_rescore block built in run()). Empty list when no fold ran.
    """
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

        # Hybrid blueprint is the ONLY path for getting X_train/X_val here.
        # The previous else-branch concatenated every rep at full dim and
        # trained on it — that's a concat-train baseline, removed 2026-05-12
        # per user directive (no concat-all baselines). Cells where the
        # blueprint is missing OR budget >= total_dim are now skipped.
        if blueprint is None:
            continue
        if budget is None or budget >= sum(v.shape[1] for v in reps.values()):
            continue
        try:
            X_train, X_val, fi = build_fold_hybrid(
                train_reps, val_reps, y_train, blueprint,
                augmentations_train=train_augs, augmentations_val=val_augs,
                groups=(groups[train_idx] if groups is not None else None),
                model=model_factory(),
                metric_fn=metric_fn)
        except Exception:
            continue

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_va_s = scaler.transform(X_val)

        model = model_factory()
        model.fit(X_tr_s, y_train)
        val_scores.append(_score_prediction(model, X_va_s, y_val, metric_fn, task_type))
        train_scores.append(_score_prediction(model, X_tr_s, y_train, metric_fn, task_type))

    if not val_scores:
        return float('nan'), float('nan'), []
    return (float(np.mean(val_scores)), float(np.mean(train_scores)),
            [float(s) for s in val_scores])


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
    task_type = data['task_type']
    augmentations = data.get('augmentations')
    total_dim = sum(v.shape[1] for v in reps.values())
    n_samples = len(labels)
    # capped_full = "full-dim baseline" budget. Bounded by 2n to keep MLP/SVM
    # tractable, plus a 2048 hard cap on top of that. Searched via fast
    # find_best_hybrid (not the pool helper — precompute never searched at
    # this non-canonical value, so a pool lookup would silently return None).
    capped_full = min(total_dim, 2 * n_samples, 2048)
    # Canonical pool budgets: FULL_BUDGETS filtered by both total_dim and
    # n_samples (matches find_best_hybrid's internal filter).
    from experiments._shared import FULL_BUDGETS
    pool_budgets = sorted(set(
        b for b in FULL_BUDGETS
        if b <= total_dim and b <= n_samples
    ))
    # capped_full is treated separately (fast search), so add it to the
    # full budget list but track it as a non-pool entry.
    budgets = sorted(set(pool_budgets + [capped_full]))

    families = _get_regularization_families(task_type=task_type)

    # Apply canonical model-size gate via _resolve_models (list form triggers
    # _max_samples and _DOMAIN_MODEL_SKIPS / _DATASET_STRATEGY_SKIPS lookups).
    # 'LogReg' has no entry in DEFAULT_MODELS; route it through 'ridge' for
    # _max_samples and skip gating purposes (same family for gating intent).
    _family_to_canonical = {'Ridge': 'ridge', 'LogReg': 'ridge',
                            'RF': 'rf', 'SVM': 'svm'}
    # c16 evaluates on a fixed blueprint pulled via aug-aware pool helper;
    # 'uncapped' when augs present, 'none' otherwise.
    _c16_strategy = 'uncapped' if augmentations else 'none'
    for fam_name in list(families.keys()):
        canonical = _family_to_canonical.get(fam_name)
        if canonical is None or canonical not in DEFAULT_MODELS:
            continue  # no canonical entry → library has no opinion, keep family
        resolved = _resolve_models(
            [canonical], task_type=task_type, 
            n_samples=n_samples, dataset_name=dataset_name,
            strategy=_c16_strategy)
        if not resolved:
            print(f"  Skipping {fam_name}: gated by _resolve_models "
                  f"(canonical={canonical}, n_samples={n_samples}, "
                  f"strategy={_c16_strategy})")
            del families[fam_name]

    # Pre-compute search blueprints per budget.
    # - pool_budgets: precompute has results, use the pool helper (free).
    # - capped_full: precompute didn't search this value, run a fast
    #   find_best_hybrid with rf as downstream model and cache it under
    #   a dedicated tier so it doesn't collide with general/top precompute.
    blueprints = {}
    try:
        for budget in pool_budgets:
            search, meta = get_best_search_for_downstream(
                reps, labels, groups,
                downstream_model='rf', budget=budget,
                augmentations=augmentations,
                dataset_name=dataset_name, task_type=task_type,
                data=data,
            )
            if meta:
                blueprints[budget] = meta['best_config']['blueprint']
    except Exception:
        pass  # will skip pool-side budgets

    # Fast search for the capped_full slot (not in FULL_BUDGETS).
    capped_full_error = None  # surfaced in results so a NaN full-dim baseline
    # (blueprint never built -> _evaluate_cell skips all folds) is diagnosable
    # without the slurm log. This was the cause of full-dim NaN on VEGF/Ang2.
    if capped_full not in blueprints:
        try:
            from kirby.find_best_hybrid import find_best_hybrid as _fbh16
            from sklearn.ensemble import RandomForestRegressor as _RF16
            from sklearn.ensemble import RandomForestClassifier as _RFC16
            _aug_strat_c16 = 'uncapped' if augmentations else 'none'
            _model_for_search = (_RFC16(n_estimators=100, random_state=42, n_jobs=-1)
                                 if task_type == 'classification'
                                 else _RF16(n_estimators=100, max_depth=None,
                                            random_state=42, n_jobs=-1))
            cf_search = _fbh16(
                base_reps=reps, labels=labels, groups=groups,
                models={'rf': _model_for_search},
                budgets=[capped_full],
                task_type=task_type, metric=metric_name,
                cache=True,
                dataset_name=f"{dataset_name}__c16_capped{capped_full}",
                tier='c16_fast',
                augmentations=augmentations,
                augmentation_strategy=_aug_strat_c16,
                verbose=False,
            )
            cf_best = (cf_search.best_by_model_budget.get('rf', {}).get(capped_full)
                       or cf_search.best_by_model.get('rf'))
            if cf_best:
                blueprints[capped_full] = cf_best['blueprint']
            else:
                # Search ran without raising but produced no usable blueprint.
                # Record it so the resulting NaN full-dim baseline (all cells at
                # the top budget get skipped in _evaluate_cell) is diagnosable
                # instead of a silent blank; capped_full_error was staying None
                # on this path, which is exactly the unexplained-dash state.
                capped_full_error = (
                    f'fast search at capped_full={capped_full} returned no '
                    f'blueprint (no eligible rf result)')
                print(f"  c16 fast search at capped_full={capped_full}: "
                      "no usable blueprint returned")
        except Exception as e:
            capped_full_error = f'{type(e).__name__}: {e}'
            print(f"  c16 fast search at capped_full={capped_full} failed: {e}")

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'budgets': budgets,
        'total_dim': total_dim,
        'capped_full': capped_full,
        'capped_full_blueprint_built': capped_full in blueprints,
        'capped_full_error': capped_full_error,
        'families': {},
    }

    # Honest leave-one-fold-out re-score inputs (per-family = per-"model").
    # Two arms, mirroring the two selection axes this experiment argmaxes over:
    #   hybrid arm    = the simple-hybrid budget sweep at DEFAULT regularization
    #                   (best_simple_hybrid = max over budgets < full);
    #   baseline arm  = the tuned regularization sweep at the full-dim budget
    #                   (best_tuned_full = max over reg).
    # Persisting per-fold scores for BOTH lets a reader call honest_rescore()
    # and get a selection-optimism-corrected simple-hybrid vs tuned-full delta,
    # re-picking the winning budget / reg per held-out fold instead of on the
    # same folds it is scored on. Per-family structure is preserved (family name
    # is the model key). See experiments._shared.honest_rescore.
    honest_candidates = {}  # {family: [{"name": <budget id>, "fold_scores": [...]}, ...]}
    honest_baselines = {}   # {family: {<reg id>: [fold_scores], ...}}

    for fam_name, variants in families.items():
        print(f"\n  --- {fam_name} ({len(variants)} regularization levels × {len(budgets)} budgets) ---")
        param_name = variants[0][0]

        heatmap = {}  # {budget: {param_value: {'val': score, 'train': score}}}
        for budget in budgets:
            heatmap[budget] = {}
            for pname, pval, factory in variants:
                val_score, train_score, fold_scores = _evaluate_cell(
                    reps, labels, groups, factory, budget, n_folds, mode, metric_fn,
                    augmentations=augmentations,
                    blueprint=blueprints.get(budget),
                    task_type=task_type)
                heatmap[budget][str(pval)] = {
                    'val': val_score,
                    'train': train_score,
                    'gap': train_score - val_score,
                    # Per-fold validation scores kept for the honest re-score
                    # (leave-one-fold-out budget / reg selection). Additive.
                    'fold_scores': list(fold_scores),
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
                # OTHER-27: this baseline is evaluated at full_dim_budget =
                # max(budgets) = capped_full = min(total_dim, 2n, 2048), which
                # is NOT necessarily the true full dimensionality. Record the
                # actual budget + a flag so the "full_dim" label can't be
                # misread when total_dim > the cap.
                'budget': int(full_dim_budget),
                'is_capped': bool(full_dim_budget < total_dim),
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

        # --- Honest re-score inputs for this family (per-fold, both arms) ---
        # Hybrid arm: simple-hybrid budgets (b < full) at default regularization.
        _hybrid_default = variants[len(variants) // 2][1]
        _cand_list = []
        for b in budgets:
            if b >= full_dim_budget:
                continue
            _fs = heatmap[b].get(str(_hybrid_default), {}).get('fold_scores', [])
            _cand_list.append({'name': f'b{b}', 'fold_scores': list(_fs)})
        honest_candidates[fam_name] = _cand_list
        # Baseline arm: tuned regularization settings at the full-dim budget.
        _base_map = {}
        for _, pval, _f in variants:
            _fs = heatmap[full_dim_budget].get(str(pval), {}).get('fold_scores', [])
            _base_map[f'{param_name}={pval}'] = list(_fs)
        honest_baselines[fam_name] = _base_map

        results['families'][fam_name] = fam_result
        print(f"    Sensitivity: {', '.join(f'b={b}:{s:.4f}' for b, s in sensitivity.items())}")
        if comparison.get('hybrid_wins') is not None:
            winner = 'simple_hybrid' if comparison['hybrid_wins'] else 'tuned_full_dim'
            print(f"    Winner: {winner}")

    # Standard honest_rescore block (exact CONTRACT key names). Keys of
    # candidates_by_model / baselines_by_model are the FAMILY names (each family
    # is one "model" for the re-score). A reader loads this block and calls
    # experiments._shared.honest_rescore(candidates_by_model, baselines_by_model,
    # lower_is_better) to get the selection-optimism-corrected delta per family.
    results['honest_rescore'] = {
        'lower_is_better': lower_is_better,
        'candidates_by_model': honest_candidates,
        'baselines_by_model': honest_baselines,
    }

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
