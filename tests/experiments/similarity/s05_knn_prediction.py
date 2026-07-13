#!/usr/bin/env python3
"""S05: kNN Prediction with Hybrid Distance.

Paper: Similarity
Question: does kNN in the hybrid feature space beat kNN in each single rep's space?
Dependencies: None (runs its own distance-aware hybrid search).

CV evaluation with per-budget hybrid conditions: hybrid_unweighted, hybrid_weighted.
Per-rep kNN baselines (the real baselines). Budget sweep over DISTANCE_BUDGETS
to show how dimensionality affects kNN performance.

Reporting honesty (2026-06-03): the "best hybrid" taken as the MAX over
budgets x conditions x k is post-hoc optimistic (multiple-comparison inflation,
scored on the same CV used to report). `results['summary']` therefore carries
BOTH that `posthoc_max` AND a pre-registered `fixed_ref` (a fixed budget /
condition / k chosen a priori, no selection over the grid). Report `fixed_ref`
as the honest hybrid-vs-rep delta; `posthoc_max` only as an upper bound.
The importance-WEIGHTED condition is genuinely applied (sqrt-importance scaling
of the standardized features, lines ~161-169) but empirically does NOT beat the
unweighted hybrid downstream — that is a real result, not a wiring bug.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, get_fitted_hybrid, get_fitted_pool_hybrid, get_best_for_downstream,
    flatten_weights, flatten_fold_weights, make_cv_splitter,
    get_domain_metric, build_fold_hybrid, extract_top3_base_reps,
    honest_rescore,
    DISTANCE_BUDGETS, DISTANCE_EVALUATORS,
)

from kirby.hybrid import apply_feature_selection, compute_feature_importance


# =============================================================================
# EXPERIMENT
# =============================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """Downstream task validation: kNN with distance-aware hybrid selection
    across multiple budgets."""
    print("\n" + "=" * 70)
    print("EXPERIMENT S05: kNN PREDICTION WITH HYBRID DISTANCE")
    print("=" * 70)

    k_values = kwargs.get('k_values', [5, 10])
    budgets = kwargs.get('budgets', DISTANCE_BUDGETS)
    n_folds = config['n_splits']

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    augmentations = data.get('augmentations')
    search_speed = config.get('search_speed', 'full')

    task_type = data['task_type']
    is_clf = task_type == 'classification'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    results = {'per_budget': {}, 'per_rep_baselines': {}}

    # ── Distance-aware hybrid search (on-demand, disk-cached) ──
    # Uses kNN + tree evaluators competing, with lower budgets.
    print(f"  Running distance-aware hybrid search "
          f"(evaluators={DISTANCE_EVALUATORS}, budgets={budgets})...")
    try:
        # We only need the SearchResult, not the full-data hybrid matrix.
        # Per-budget blueprints are used with build_fold_hybrid in the CV loop.
        _X_unused, _fi_unused, search, _meta = get_fitted_pool_hybrid(
            reps, labels, groups,
            downstream_model='knn', budget=None,
            augmentations=augmentations,
            speed=search_speed,
            metric=metric_name, task_type=task_type,
            domain=data.get('domain', 'molecular'),
            dataset_name=dataset_name, distance=True)

        # Report which evaluator won
        best_knn = get_best_for_downstream(search, 'knn')
        if best_knn and 'blueprint' in best_knn:
            bp = best_knn['blueprint']
            print(f"  Best kNN config: alloc={bp.allocation_method}, "
                  f"imp={bp.importance_method}, ev={bp.internal_evaluator}, "
                  f"budget={bp.budget}, score={best_knn.get('score_mean', 0):.4f}")
        top3 = extract_top3_base_reps(search)
        if top3:
            print(f"  Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")
            results['top3_base_reps'] = [(n, s) for n, s in top3]
    except Exception as e:
        print(f"  Distance-aware search failed: {e}")
        search = None

    # ── Per-budget hybrid kNN evaluation ──
    sample_X = next(iter(reps.values()))
    # Build the CV folds ONCE (make_cv_splitter is deterministic: random_state
    # is fixed, so the folds are identical for every budget/condition/k). All
    # per-budget conditions MUST share the same folds for a fair comparison;
    # hoisting also avoids reconstructing the splitter inside the budget loop
    # (SIMA-18, result-neutral).
    splitter_b, use_groups_b = make_cv_splitter(n_folds, groups)
    folds_b = list(
        splitter_b.split(sample_X, labels, groups=groups)
        if use_groups_b
        else splitter_b.split(sample_X, labels)
    )

    def _make_knn(k_val, n_train):
        nn = min(k_val, n_train - 1)
        if is_clf:
            return KNeighborsClassifier(n_neighbors=nn)
        return KNeighborsRegressor(n_neighbors=nn)

    # Honest leave-one-fold-out re-score arms (persisted below). This
    # experiment runs its OWN kNN CV (not find_best_hybrid), so we collect the
    # per-fold scores of every hybrid cell (budget x condition x k) and every
    # single-rep baseline cell (rep x k) here, and assemble the standard
    # honest_rescore block by hand after the loops. Single downstream model by
    # design ('knn'), so both arms live under the 'knn' key.
    hybrid_candidates = []  # [{'name': str, 'fold_scores': [...]}]

    if search is not None:
        for budget in budgets:
            best_at_budget = get_best_for_downstream(search, 'knn', budget=budget)
            if best_at_budget is None or 'blueprint' not in best_at_budget:
                print(f"\n  Budget={budget}: no config found, skipping")
                continue

            blueprint = best_at_budget['blueprint']
            # Override blueprint augmentation_strategy with the pool-helper's
            # aug_condition (2026-04-21 cache-collision fix). Set BEFORE the
            # fold loop so build_fold_hybrid sees the correct strategy.
            if _meta is not None and 'aug_condition' in _meta:
                blueprint.augmentation_strategy = _meta['aug_condition']
            print(f"\n  Budget={budget} (ev={blueprint.internal_evaluator}, "
                  f"alloc={blueprint.allocation_method}):")

            budget_results = {}

            for condition in ['hybrid_unweighted', 'hybrid_weighted']:
                for k in k_values:
                    fold_scores = []

                    for train_idx, val_idx in folds_b:
                        y_train, y_val = labels[train_idx], labels[val_idx]

                        # Build per-fold hybrid from blueprint
                        train_reps = {name: rep[train_idx] for name, rep in reps.items()}
                        val_reps = {name: rep[val_idx] for name, rep in reps.items()}
                        aug_tr = ({k2: v[train_idx] for k2, v in augmentations.items()}
                                  if augmentations else None)
                        aug_va = ({k2: v[val_idx] for k2, v in augmentations.items()}
                                  if augmentations else None)
                        try:
                            X_tr_h, X_va_h, feat_info = build_fold_hybrid(
                                train_reps, val_reps, y_train, blueprint,
                                augmentations_train=aug_tr,
                                augmentations_val=aug_va)
                        except Exception as e:
                            fold_i = len(fold_scores)
                            print(f"  [warn] fold {fold_i} failed: "
                                  f"{type(e).__name__}: {str(e)[:80]}")
                            continue

                        scaler = StandardScaler()
                        X_tr = scaler.fit_transform(X_tr_h)
                        X_va = scaler.transform(X_va_h)

                        # Importance weighting for weighted condition.
                        # Use flatten_fold_weights to handle augmentation tail
                        # correctly (manual concat dropped aug columns and
                        # shape-failed silently under bare except).
                        if condition == 'hybrid_weighted':
                            try:
                                w_raw = flatten_fold_weights(feat_info)
                                w = np.sqrt(np.maximum(w_raw, 0))
                                assert w.shape[0] == X_tr.shape[1], (
                                    f"weight/feature mismatch: "
                                    f"w={w.shape[0]} vs X={X_tr.shape[1]}")
                                X_tr = X_tr * w
                                X_va = X_va * w
                            except Exception as e:
                                fold_i = len(fold_scores)
                                print(f"  [warn] fold {fold_i} failed: "
                                      f"{type(e).__name__}: {str(e)[:80]}")
                                continue

                        model = _make_knn(k, len(y_train))
                        model.fit(X_tr, y_train)
                        y_pred = model.predict(X_va)
                        fold_scores.append(metric_fn(y_val, y_pred))

                    if not fold_scores:
                        continue

                    score_mean = float(np.mean(fold_scores))
                    score_std = float(np.std(fold_scores))
                    # Honest-rescore arm: keep the raw per-fold scores for this
                    # (budget, condition, k) cell so held-out-fold selection can
                    # re-pick the winning cell per fold (not the same-fold argmax).
                    hybrid_candidates.append({
                        'name': f"b{budget}_{condition}_k{k}",
                        'fold_scores': [float(s) for s in fold_scores],
                    })
                    key = f"{condition}_k{k}"
                    budget_results[key] = {
                        'score_mean': score_mean,
                        'score_std': score_std,
                        'metric': metric_name,
                        'k': k,
                        'condition': condition,
                        'budget': budget,
                        'evaluator': blueprint.internal_evaluator,
                        'allocation': blueprint.allocation_method,
                    }
                    print(f"    {condition} k={k}: "
                          f"{metric_name}={score_mean:.4f}+/-{score_std:.3f}")

            results['per_budget'][budget] = budget_results

    # ── Per-rep kNN baselines (individual representations) ──
    # Try all k values and pick the best per metric direction.
    print(f"\n  Individual representation kNN baselines (metric={metric_name}):")
    # Reuse the SAME folds as the hybrid conditions (folds_b) so the
    # hybrid-vs-baseline comparison is on identical splits. (Previously a
    # separate, deterministic splitter rebuilt identical folds — SIMA-18.)
    folds_baseline = folds_b

    # Honest-rescore baseline arm: per-(rep, k) unweighted kNN per-fold scores.
    # Keying each (rep, k) as its own baseline cell lets the held-out-fold
    # selection replace BOTH the same-fold best-rep argmax AND the same-fold
    # best-k argmax in one pass. These are "the real baselines" (unweighted).
    baseline_candidates = {}  # {f"{rep}|k{k}": [per-fold score, ...]}

    for rep_name, X_rep in reps.items():
        best_k_score = None
        best_k_used = k_values[0]
        best_k_scores_uw = None
        best_k_scores_w = None
        best_k_std = 0.0
        best_k_w_mean = float('nan')

        # Per-fold importance computed ONCE per rep (independent of k).
        fold_importance = []
        for train_idx, _val_idx in folds_baseline:
            try:
                imp = compute_feature_importance(
                    {rep_name: X_rep[train_idx]}, labels[train_idx],
                    method='lightgbm_gain')
                fold_importance.append(
                    np.sqrt(np.maximum(imp[rep_name], 0)))
            except Exception as exc:
                # N9: log so a None-importance fold reads as "compute failed"
                # rather than silently falling back to unweighted kNN.
                print(f"      importance fold failed (rep={rep_name}): "
                      f"{type(exc).__name__}: {exc}")
                fold_importance.append(None)

        for k_candidate in k_values:
            fold_scores_uw = []
            fold_scores_w = []
            for fold_i, (train_idx, val_idx) in enumerate(folds_baseline):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_rep[train_idx])
                X_va = scaler.transform(X_rep[val_idx])

                # Unweighted
                model = _make_knn(k_candidate, len(train_idx))
                model.fit(X_tr, labels[train_idx])
                fold_scores_uw.append(metric_fn(labels[val_idx], model.predict(X_va)))

                # Weighted by importance (reuses pre-computed weights)
                w = fold_importance[fold_i]
                if w is None:
                    fold_scores_w.append(float('nan'))
                else:
                    try:
                        X_tr_w = X_tr * w
                        X_va_w = X_va * w
                        model_w = _make_knn(k_candidate, len(train_idx))
                        model_w.fit(X_tr_w, labels[train_idx])
                        fold_scores_w.append(metric_fn(labels[val_idx],
                                                       model_w.predict(X_va_w)))
                    except Exception as exc:
                        # N9: log so a NaN weighted-kNN fold reads as "fit/predict
                        # threw" rather than a genuine score.
                        print(f"      weighted kNN fold failed "
                              f"(rep={rep_name}, k={k_candidate}): "
                              f"{type(exc).__name__}: {exc}")
                        fold_scores_w.append(float('nan'))

            if not fold_scores_uw:
                continue
            # Persist this (rep, k) cell's per-fold unweighted scores for the
            # honest leave-one-fold-out re-score (held-out selection over rep+k).
            baseline_candidates[f"{rep_name}|k{k_candidate}"] = [
                float(s) for s in fold_scores_uw]
            sc_mean = float(np.mean(fold_scores_uw))
            if best_k_score is None or (sc_mean < best_k_score if lower_is_better else sc_mean > best_k_score):
                best_k_score = sc_mean
                best_k_used = k_candidate
                best_k_std = float(np.std(fold_scores_uw))
                best_k_scores_uw = fold_scores_uw
                best_k_w_mean = float(np.nanmean(fold_scores_w)) if fold_scores_w else float('nan')

        sc_mean = best_k_score if best_k_score is not None else float('nan')
        sc_std = best_k_std
        sc_w_mean = best_k_w_mean
        results['per_rep_baselines'][rep_name] = {
            'score_mean': sc_mean, 'score_std': sc_std,
            'score_weighted_mean': sc_w_mean,
            'metric': metric_name,
            'k': best_k_used, 'dim': X_rep.shape[1],
        }
        # Sign-correct delta: positive = improvement regardless of metric
        # direction (negated for lower-is-better metrics like RMSE).
        sign = -1 if lower_is_better else 1
        delta = sign * (sc_w_mean - sc_mean)
        print(f"    {rep_name} (dim={X_rep.shape[1]}, k={best_k_used}): "
              f"kNN {metric_name}={sc_mean:.4f}+/-{sc_std:.3f}, "
              f"weighted {metric_name}={sc_w_mean:.4f} "
              f"(delta={delta:+.4f}, positive=improvement)")

    # ── Summary ──
    print(f"\n  Summary (best per budget):")
    for b in sorted(results['per_budget'].keys()):
        budget_res = results['per_budget'][b]
        if budget_res:
            best_key = (min if lower_is_better else max)(
                budget_res, key=lambda k: budget_res[k]['score_mean'])
            best = budget_res[best_key]
            print(f"    budget={b}: best={best['condition']} k={best['k']}, "
                  f"{metric_name}={best['score_mean']:.4f} "
                  f"(ev={best.get('evaluator', '?')})")

    if results['per_rep_baselines']:
        best_rep = (min if lower_is_better else max)(
            results['per_rep_baselines'],
            key=lambda r: results['per_rep_baselines'][r]['score_mean'])
        best_rep_score = results['per_rep_baselines'][best_rep]['score_mean']
        print(f"\n  Best individual rep baseline: {best_rep} "
              f"(kNN {metric_name}={best_rep_score:.4f})")

        # Compare best hybrid budget to best individual rep
        all_hybrid_scores = []
        for b_res in results['per_budget'].values():
            for v in b_res.values():
                all_hybrid_scores.append(v)
        results['summary'] = {
            'best_rep_baseline': best_rep,
            'best_rep_score': best_rep_score,
        }
        if all_hybrid_scores:
            sign = -1 if lower_is_better else 1
            best_hybrid = (min if lower_is_better else max)(
                all_hybrid_scores, key=lambda v: v['score_mean'])
            delta = sign * (best_hybrid['score_mean'] - best_rep_score)
            # post-hoc MAX over budgets x conditions x k = optimistic (in-sample
            # multiple comparison). Recorded but NOT the headline number.
            results['summary']['posthoc_max'] = {
                'budget': best_hybrid['budget'], 'condition': best_hybrid['condition'],
                'k': best_hybrid['k'], 'score_mean': best_hybrid['score_mean'],
                'delta_vs_best_rep': delta,
                'note': ('MAX over budgets x conditions x k, scored in-sample = '
                         'post-hoc optimistic; do not report as the headline.'),
            }
            print(f"  Best hybrid kNN [posthoc-max, optimistic]: "
                  f"budget={best_hybrid['budget']}, "
                  f"{best_hybrid['condition']} k={best_hybrid['k']}, "
                  f"{metric_name}={best_hybrid['score_mean']:.4f} "
                  f"(delta vs {best_rep}: {delta:+.4f})")

            # Pre-registered fixed cell (no cherry-picking over the grid): the
            # honest hybrid-vs-rep comparator. budget=100 (nearest available),
            # hybrid_unweighted, k=max(k_values).
            FIXED_BUDGET, FIXED_COND = 100, 'hybrid_unweighted'
            avail = sorted(results['per_budget'].keys())
            fb = min(avail, key=lambda b: abs(b - FIXED_BUDGET)) if avail else None
            fixed_cell = None
            if fb is not None:
                cells = results['per_budget'][fb]
                fixed_cell = (cells.get(f'{FIXED_COND}_k{max(k_values)}')
                              or cells.get(f'{FIXED_COND}_k{k_values[0]}'))
            if fixed_cell is not None:
                fdelta = sign * (fixed_cell['score_mean'] - best_rep_score)
                results['summary']['fixed_ref'] = {
                    'budget': fixed_cell['budget'], 'condition': fixed_cell['condition'],
                    'k': fixed_cell['k'], 'score_mean': fixed_cell['score_mean'],
                    'delta_vs_best_rep': fdelta,
                    'note': ('pre-registered budget/condition/k (no selection over '
                             'the grid); report THIS as the honest delta.'),
                }
                print(f"  Best hybrid kNN [fixed-ref, HONEST]: "
                      f"budget={fixed_cell['budget']}, {fixed_cell['condition']} "
                      f"k={fixed_cell['k']}, {metric_name}={fixed_cell['score_mean']:.4f} "
                      f"(delta vs {best_rep}: {fdelta:+.4f}, positive=improvement)")

    # ── Honest leave-one-fold-out re-score block (selection-optimism fix) ──
    # build_honest_rescore_input() reads a SearchResult; this experiment runs
    # its own kNN CV, so we assemble the standard block by hand from the
    # per-fold scores collected above (single 'knn' model by design). A reader
    # loads results['honest_rescore'] and calls the shared honest_rescore(...).
    results['honest_rescore'] = {
        'lower_is_better': bool(lower_is_better),
        'candidates_by_model': (
            {'knn': hybrid_candidates} if hybrid_candidates else {}),
        'baselines_by_model': (
            {'knn': baseline_candidates} if baseline_candidates else {}),
    }
    # Reuse the shared pure-arithmetic helper (no ML) so the honest hybrid-vs-
    # best-single-rep delta is available inline alongside the optimistic
    # posthoc_max / pre-registered fixed_ref summaries.
    try:
        hr = honest_rescore(
            results['honest_rescore']['candidates_by_model'],
            results['honest_rescore']['baselines_by_model'],
            bool(lower_is_better))
        results['honest_rescore_result'] = hr
        knn_hr = hr.get('knn')
        if knn_hr is not None:
            flag = ' [single-fold, NOT honest]' if knn_hr.get('single_fold') else ''
            print(f"\n  Honest LOFO re-score (knn){flag}: "
                  f"hybrid={knn_hr['hybrid_honest']:.4f}, "
                  f"best_single={knn_hr['best_single_honest']:.4f}, "
                  f"delta={knn_hr['delta']:+.4f} (positive=hybrid better)")
    except Exception as e:
        print(f"  [warn] honest_rescore failed: {type(e).__name__}: {e}")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('S05: kNN Prediction with Hybrid Distance',
                                     has_delta=True, has_augment=True)
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 's05_knn_prediction', args.dataset)
