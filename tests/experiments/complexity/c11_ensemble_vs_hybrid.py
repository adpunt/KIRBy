#!/usr/bin/env python3
"""C11: Feature-level fusion (hybrid) vs best single rep vs PCA-matched control.

Paper: Complexity
Conclusion: Whether a single model on a fused hybrid representation outperforms
the best single base rep and a dimension-matched PCA control.
Dependencies: None (standalone).

Conditions per model type:
  1. Best single base rep (train one model on each rep, report best)
  2. PCA-matched: concat all reps, PCA-compress to the budget dim, one model
  3. Hybrid: train one model on the fused hybrid representation

NOTE (design): the original "ensemble" arm (train one model per base rep and
average predictions = model-level fusion) was REMOVED 2026-05-12 per user
directive — averaged baselines are not a valid comparator (report individual
reps only). The experiment name is kept for continuity, but model-level fusion
is intentionally absent; the comparison is hybrid vs best-single vs PCA control.

Each condition is tested with and without augmented features.
Reports the domain metric, total dimensionality, model count, training time,
and inference time.
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
    save_experiment_results,
    get_model_factories, cv_split,
    get_domain_metric, cap_dimensions,
    build_fold_hybrid, extract_top3_base_reps,
    get_best_search_for_downstream,
)


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
    task_type = data['task_type']

    models = get_model_factories(config['models'],
                                    task_type=task_type,
                                    n_samples=len(data['labels']))
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'

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
    n_samples = len(labels)
    # Canonical budget list (matches FULL_BUDGETS used by precompute).
    # Hardcoded [50, 100, 200, 500, 1024] was divergent — 200 and 1024 were never
    # searched by precompute, so blueprint lookups at those budgets returned None
    # and the loop silently skipped them.
    from experiments._shared import FULL_BUDGETS
    budgets = [b for b in FULL_BUDGETS if b <= total_dim and b <= n_samples]

    augmentations = data.get('augmentations')
    last_search = None

    def _blueprints_for(model_name):
        """Best hybrid blueprint per budget, optimised FOR THIS downstream model.

        The blueprint must match the model it will be evaluated with. The search
        already scored every candidate against every downstream model, so this is
        a cached lookup (get_best_for_downstream -> best_by_model_budget), not a
        new search. The pool helper can pick a different (tier, aug_condition)
        file per (model, budget) — that is intended.
        """
        nonlocal last_search
        bps = {}
        for budget in budgets:
            try:
                search, meta = get_best_search_for_downstream(
                    reps, labels, groups,
                    downstream_model=model_name, budget=budget,
                    augmentations=augmentations,
                    speed=config.get('search_speed', 'full'),
                    dataset_name=dataset_name, task_type=task_type,
                    data=data,
                )
            except Exception as e:
                print(f"    Blueprint lookup failed ({model_name}, b={budget}): {e}")
                continue
            if meta:
                bps[budget] = meta['best_config']['blueprint']
                last_search = search
        return bps

    results = []

    # Per-fold scores for the honest held-out rescore (contract json_convention).
    # Built manually (not via build_honest_rescore_input) because C11 computes its
    # OWN CV scores in-file — StandardScaler + cap_dimensions per rep/budget — rather
    # than reading them off a SearchResult. Shapes match honest_rescore()'s inputs:
    #   honest_candidates[model] = [ {"name": <recipe id>, "fold_scores": [...]}, ... ]
    #   honest_baselines[model]  = { rep_name: [...per-fold...] }
    honest_candidates = {}
    honest_baselines = {}

    for model_name, model_factory in models.items():
        print(f"\n  --- {model_name} ---")

        # Blueprints for THIS model — all three arms (single / PCA / hybrid) are
        # then compared under the same model, which is the whole point of C11.
        blueprints = _blueprints_for(model_name)
        print(f"    Blueprints ({model_name}): {sorted(blueprints.keys())}")
        if not blueprints:
            print(f"    No blueprint for {model_name} — hybrid arm will be empty.")

        # =====================================================================
        # 1. Best single base rep
        # =====================================================================
        best_single_score = np.inf if lower_is_better else -np.inf
        best_single_rep = None
        single_rep_results = {}
        model_baseline_folds = {}   # rep_name -> per-fold scores (honest helper input)
        model_candidate_folds = []  # [{name, fold_scores}] hybrid budgets (honest helper input)

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
                X_train, X_val = cap_dimensions(X_train, X_val)
                y_train, y_val = labels[train_idx], labels[val_idx]

                preds, ft, pt = _time_fit_predict(model_factory, X_train, y_train, X_val)
                fold_scores.append(metric_fn(y_val, preds))
                fold_fit_times.append(ft)
                fold_pred_times.append(pt)

            fold_scores_f = [float(s) for s in fold_scores]
            model_baseline_folds[rep_name] = fold_scores_f

            if not fold_scores:
                score_mean = float('nan')
                single_rep_results[rep_name] = {
                    'score_mean': score_mean, 'score_std': float('nan'),
                    'dim': X_rep.shape[1],
                    'fit_time_s': 0.0, 'predict_time_s': 0.0,
                    'fold_scores': fold_scores_f,
                }
                continue

            score_mean = float(np.mean(fold_scores))
            single_rep_results[rep_name] = {
                'score_mean': score_mean,
                'score_std': float(np.std(fold_scores)),
                'dim': X_rep.shape[1],
                'fit_time_s': float(np.mean(fold_fit_times)),
                'predict_time_s': float(np.mean(fold_pred_times)),
                'fold_scores': fold_scores_f,
            }
            if (lower_is_better and score_mean < best_single_score) or \
               (not lower_is_better and score_mean > best_single_score):
                best_single_score = score_mean
                best_single_rep = rep_name

        if best_single_rep is None:
            print(f"    Best single: NONE (all reps failed)")
            continue

        best_single = single_rep_results[best_single_rep]
        print(f"    Best single: {best_single_rep} "
              f"{metric_name}={best_single['score_mean']:.4f} dim={best_single['dim']}")

        results.append({
            'model': model_name,
            'method': 'best_single',
            'augmented': False,
            'score_mean': best_single['score_mean'],
            'score_std': best_single['score_std'],
            'total_dim': best_single['dim'],
            'n_models_trained': 1,
            'total_fit_time_s': best_single['fit_time_s'],
            'total_predict_time_s': best_single['predict_time_s'],
            'fold_scores': best_single.get('fold_scores', []),
            'detail': {'best_rep': best_single_rep, 'all_reps': single_rep_results},
        })

        # =====================================================================
        # 2. PCA-matched: concat all reps, PCA-compress to budget dims, one model
        # PCA-from-concat is allowed (per design directive: PCA reducing
        # high-dim concat to budget dim is meaningful; full-concat as raw
        # model input is not).
        #
        # The previous "ensemble" method (predict-then-average across 26
        # per-rep models) was removed 2026-05-12 per user directive: averaged
        # baselines are not a valid comparator; report individual reps only.
        # =====================================================================
        for budget in budgets:
            fold_pca_scores = []
            fold_pca_fit_times = []
            fold_pca_pred_times = []

            for train_idx, val_idx in cv_split(groups, n_folds, sample_X,
                                               labels, mode=mode):
                y_train, y_val = labels[train_idx], labels[val_idx]

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
                    'score_mean': float(np.mean(fold_pca_scores)),
                    'score_std': float(np.std(fold_pca_scores)),
                    'total_dim': n_comp,
                    'n_models_trained': 1,
                    'total_fit_time_s': float(np.mean(fold_pca_fit_times)),
                    'total_predict_time_s': float(np.mean(fold_pca_pred_times)),
                    'augmented': False,
                    'fold_scores': [float(s) for s in fold_pca_scores],
                })

        # =====================================================================
        # 3. Hybrid: single model on fused representation
        # =====================================================================
        for budget in budgets:
            if budget not in blueprints:
                continue

            fold_hybrid_scores = []
            fold_hybrid_fit_times = []
            fold_hybrid_pred_times = []
            last_n_features = 0

            for train_idx, val_idx in cv_split(groups, n_folds, sample_X,
                                               labels, mode=mode):
                train_reps_fold = {k: v[train_idx] for k, v in reps.items()}
                val_reps_fold = {k: v[val_idx] for k, v in reps.items()}
                y_train, y_val = labels[train_idx], labels[val_idx]

                augs_full = augmentations or {}
                train_augs = {k: v[train_idx] for k, v in augs_full.items()} if augs_full else None
                val_augs = {k: v[val_idx] for k, v in augs_full.items()} if augs_full else None

                try:
                    t0 = time.time()
                    X_train_h, X_val_h, feature_info = build_fold_hybrid(
                        train_reps_fold, val_reps_fold, y_train,
                        blueprints[budget],
                        augmentations_train=train_augs,
                        augmentations_val=val_augs,
                    )
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
            print(f"    Hybrid b={budget_label:<5}: {metric_name}={hyb_r2:.4f} "
                  f"dim={last_n_features}")

            fold_hybrid_scores_f = [float(s) for s in fold_hybrid_scores]
            # Each budget is a candidate recipe for held-out reselection by the helper.
            model_candidate_folds.append({
                'name': f'hybrid_b{budget_label}',
                'fold_scores': fold_hybrid_scores_f,
            })

            results.append({
                'model': model_name,
                'method': 'hybrid',
                'augmented': False,
                'budget': budget,
                'score_mean': hyb_r2,
                'score_std': float(np.std(fold_hybrid_scores)),
                'total_dim': last_n_features,
                'n_models_trained': 1,
                'total_fit_time_s': float(np.mean(fold_hybrid_fit_times)),
                'total_predict_time_s': float(np.mean(fold_hybrid_pred_times)),
                'fold_scores': fold_hybrid_scores_f,
            })

        # Record this model's per-fold arms for the honest held-out rescore.
        # Placed after all budgets so candidates cover every hybrid recipe tried.
        honest_baselines[model_name] = model_baseline_folds
        honest_candidates[model_name] = model_candidate_folds

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
        header = (f"    {'Method':<22} {metric_name:>8} {'Dim':>6} "
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

            print(f"    {method_label:<22} {r['score_mean']:>8.4f} "
                  f"{r['total_dim']:>6} {r['n_models_trained']:>7} "
                  f"{r['total_fit_time_s']:>8.3f} {r['total_predict_time_s']:>8.4f}")

    # Efficiency summary: hybrid vs PCA-matched (same-budget, all-reps PCA reduction)
    print(f"\n  EFFICIENCY COMPARISON (best hybrid vs best PCA-matched per model):")
    print(f"    {'Model':<12} {'PCA ' + metric_name:>8} {'Hyb ' + metric_name:>8} "
          f"{'Delta':>8} {'PCA Dim':>8} {'Hyb Dim':>8} "
          f"{'PCA Fit':>8} {'Hyb Fit':>8} {'Speedup':>8}")
    print("    " + "-" * 82)

    for model_name in models:
        pcas = [r for r in results if r['model'] == model_name
                and r['method'] == 'pca_matched' and not r['augmented']]
        hybs = [r for r in results if r['model'] == model_name
                and r['method'] == 'hybrid' and not r['augmented']]

        if not pcas or not hybs:
            continue

        _best = min if lower_is_better else max
        best_pca = _best(pcas, key=lambda x: x['score_mean'])
        best_hyb = _best(hybs, key=lambda x: x['score_mean'])

        delta = best_hyb['score_mean'] - best_pca['score_mean']
        speedup = (best_pca['total_fit_time_s'] / best_hyb['total_fit_time_s']
                   if best_hyb['total_fit_time_s'] > 0 else 0)

        print(f"    {model_name:<12} {best_pca['score_mean']:>8.4f} {best_hyb['score_mean']:>8.4f} "
              f"{delta:>+8.4f} {best_pca['total_dim']:>8} {best_hyb['total_dim']:>8} "
              f"{best_pca['total_fit_time_s']:>8.3f} "
              f"{best_hyb['total_fit_time_s']:>8.3f} {speedup:>7.1f}x")

    top3 = extract_top3_base_reps(last_search) if last_search is not None else []
    return {
        'results': results,
        'n_reps': n_reps,
        'total_dim': total_dim,
        'rep_names': rep_names,
        'top3_base_reps': [(n, s) for n, s in top3] if top3 else [],
        # Honest held-out rescore inputs (contract json_convention). A reader loads
        # this block and calls honest_rescore(candidates_by_model, baselines_by_model,
        # lower_is_better) to re-pick the winning hybrid budget / baseline rep per
        # held-out fold instead of trusting the same-fold argmax above.
        'honest_rescore': {
            'lower_is_better': bool(lower_is_better),
            'candidates_by_model': honest_candidates,
            'baselines_by_model': honest_baselines,
        },
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
