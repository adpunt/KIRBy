#!/usr/bin/env python3
"""S04: Activity Cliffs -- RMSE_cliff improvement.

Paper: Similarity
Conclusion: Hybrid reduces RMSE on activity cliffs.
Dependencies: S02 (reads JSON, optional).

Identifies cliff compounds using ECFP4 Tanimoto (representation-independent
definition based on Bajorath standard). CV evaluation of kNN on cliff vs
non-cliff compounds for: per-rep kNN baselines, hybrid kNN unweighted,
hybrid kNN weighted, RF baseline.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, load_experiment_results,
    get_fitted_pool_hybrid, flatten_fold_weights, make_cv_splitter,
    get_domain_metric, get_model_factories, build_fold_hybrid,
    get_best_for_downstream, extract_top3_base_reps,
    DISTANCE_BUDGETS, DISTANCE_EVALUATORS,
)
from experiments.similarity.similarity_helpers import identify_cliff_compounds

from kirby.hybrid import apply_feature_selection, compute_feature_importance


# =============================================================================
# EXPERIMENT
# =============================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """Prediction performance on activity cliff compounds.

    Uses representation-independent cliff definition (Bajorath ECFP4 standard):
      - ECFP4 Tanimoto >= tanimoto_threshold
      - |delta_y| > activity_k * MAD(y)

    Reports RMSE_cliff, RMSE_noncliff for:
      - Per-rep kNN baselines
      - Hybrid kNN (unweighted and weighted)
      - RF baseline
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT S04: PREDICTION PERFORMANCE ON ACTIVITY CLIFFS")
    print("=" * 70)

    tanimoto_threshold = kwargs.get('tanimoto_threshold', 0.55)
    activity_k = kwargs.get('activity_k', 2.0)
    n_folds = config['n_splits']
    mode = 'intermediary' if config['n_splits'] == 1 else 'full'

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    # --- Identify cliff compounds using ECFP4 Tanimoto ---
    # S04 is molecular-only (Bajorath ECFP4 standard). Non-molecular domains
    # should use S12 (cross-domain Euclidean cliffs) instead.
    domain = data.get('domain', 'molecular')
    if domain != 'molecular':
        print(f"  SKIP: S04 requires molecular ECFP4 (domain={domain}). Use S12 for cross-domain cliffs.")
        # F21: the S12 pointer is a CLAIM about another experiment. Nothing here
        # loads, runs or checks an S12 result, so the stub records the pointer as
        # unverified — "handled elsewhere" must stay distinguishable from
        # "answered nowhere" for a reader who only has this file's output.
        return {
            'skipped': True,
            'reason': f'molecular-only experiment, domain={domain}',
            'replacement_experiment': 's12',
            'replacement_verified': False,
            'replacement_note': (
                'S12 (cross-domain Euclidean cliffs) is named as the replacement; '
                'this stub does not read or check for any S12 cliff result for '
                'this dataset. Verify one exists before treating cliffs as covered.'),
        }
    if 'ecfp4' not in reps:
        print("  ERROR: ecfp4 not in reps, cannot identify cliffs")
        return {'error': 'ecfp4 not available'}

    # Derive task_type BEFORE cliff identification so classification cliffs
    # use the class-mismatch indicator, not the regression |delta_y| rule.
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    task_type = data['task_type']
    is_clf = task_type == 'classification'

    cliff_mask, cliff_info = identify_cliff_compounds(
        reps['ecfp4'], labels,
        tanimoto_threshold=tanimoto_threshold,
        activity_k=activity_k,
        is_clf=is_clf,
    )
    cliff_indices = set(np.where(cliff_mask)[0].tolist())

    if is_clf:
        print(f"  Cliff definition (clf): Tanimoto>={tanimoto_threshold}, "
              f"different class")
    else:
        print(f"  Cliff definition: Tanimoto>={tanimoto_threshold}, "
              f"|delta_y|>{activity_k}*MAD(y) = {cliff_info['activity_threshold']:.3f}")
    print(f"  Cliff compounds: {cliff_info['n_cliff_compounds']} / {len(labels)} "
          f"({cliff_info['pct_cliff_compounds']:.1f}%)")
    print(f"  Cliff pairs: {cliff_info['n_cliff_pairs']}")

    results = {
        'cliff_definition': {
            'tanimoto_threshold': tanimoto_threshold,
            'activity_k': activity_k,
            'activity_threshold': cliff_info['activity_threshold'],
            'mad_y': cliff_info['mad_y'],
            'n_cliff_compounds': cliff_info['n_cliff_compounds'],
            'n_cliff_pairs': cliff_info['n_cliff_pairs'],
            'pct_cliff_compounds': cliff_info['pct_cliff_compounds'],
        },
        'per_rep': {},
        'hybrid': {},
    }

    # --- Honest-rescore accumulators (per this file's own cliff-CV folds) ---
    # The honest hybrid-vs-best-single comparison is done by the reader via the
    # shared honest_rescore() helper. We only persist the per-fold inputs here,
    # keyed by downstream MODEL, using the CONTRACT json_convention names. This
    # file runs its OWN CV (kNN + RF on the shared splitter), so we build the
    # block MANUALLY rather than build_honest_rescore_input(search) — the search's
    # internal baselines are on different folds/metric. Fold scores are POSITIONAL
    # (one slot per split, NaN where a fold was skipped) so both arms stay aligned
    # across the identical split order; the helper skips NaN slots.
    honest_candidates = {}  # model -> [ {name, fold_scores}, ... ]
    honest_baselines = {}   # model -> { rep_name: [fold_scores], ... }

    # --- Get hybrid blueprint for per-fold hybrid construction ---
    hybrid_blueprint = None
    search = None
    try:
        budget = kwargs.get('budget', 100)
        augmentations = data.get('augmentations')
        _X_hybrid, _fi, search, _meta = get_fitted_pool_hybrid(
            reps, labels, groups,
            downstream_model='knn', budget=budget,
            augmentations=augmentations,
            metric=metric_name, task_type=task_type, domain=domain,
            dataset_name=dataset_name, distance=True,
            data=data)
        best_knn = get_best_for_downstream(search, 'knn')
        if best_knn and 'blueprint' in best_knn:
            hybrid_blueprint = best_knn['blueprint']
        top3 = extract_top3_base_reps(search)
        if top3:
            print(f"  Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")
            results['top3_base_reps'] = [(n, s) for n, s in top3]
        results['hybrid_search_status'] = {'ok': True}
    except Exception as e:
        print(f"  Hybrid search failed: {e}")
        search = None
        # F17: persist the throw. Without this the 'hybrid' block stays the empty
        # dict it was initialised to while 'per_rep' fills normally, so the file
        # reads as a complete run with no hybrid arm.
        results['hybrid_search_status'] = {
            'ok': False,
            'error_type': type(e).__name__,
            'error': str(e),
        }

    # --- CV evaluation of prediction on cliff vs non-cliff ---
    splitter, use_groups = make_cv_splitter(n_folds, groups,
                                            task_type=task_type)
    sample_X = next(iter(reps.values()))

    def _make_knn(n_train):
        k = min(5, n_train - 1)
        if is_clf:
            return KNeighborsClassifier(n_neighbors=k)
        return KNeighborsRegressor(n_neighbors=k)

    def _evaluate_condition(get_Xy_fn, pre_scaled=False):
        """Run CV for a condition, return cliff/noncliff metrics."""
        fold_all, fold_cliff, fold_noncliff = [], [], []
        fold_rmse_all, fold_rmse_cliff, fold_rmse_noncliff = [], [], []
        # Positional per-fold overall metric for honest_rescore (NaN = skipped
        # fold), one slot per split so arms stay index-aligned.
        fold_scores_pos = []

        for train_idx, val_idx in splitter.split(sample_X, labels, groups=groups) if use_groups else splitter.split(sample_X, labels):
            y_train, y_val = labels[train_idx], labels[val_idx]
            try:
                X_tr, X_va = get_Xy_fn(train_idx, val_idx, y_train)
            except Exception as exc:
                # N9: log the dropped fold so a thrown build is distinguishable
                # from a genuinely empty fold (X_tr is None below).
                print(f"      cliff CV fold build failed: "
                      f"{type(exc).__name__}: {exc}")
                fold_scores_pos.append(float('nan'))
                continue
            if X_tr is None:
                fold_scores_pos.append(float('nan'))
                continue

            if pre_scaled:
                X_tr_s, X_va_s = X_tr, X_va
            else:
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_va_s = scaler.transform(X_va)

            model = _make_knn(len(y_train))
            model.fit(X_tr_s, y_train)
            y_pred = model.predict(X_va_s)

            # Overall
            _score_all = float(metric_fn(y_val, y_pred))
            fold_all.append(_score_all)
            fold_scores_pos.append(_score_all)
            if not is_clf:
                fold_rmse_all.append(float(np.sqrt(mean_squared_error(y_val, y_pred))))

            # Stratify by cliff membership
            val_cliff = np.array([i for i, vi in enumerate(val_idx) if vi in cliff_indices])
            val_noncliff = np.array([i for i, vi in enumerate(val_idx) if vi not in cliff_indices])

            if len(val_cliff) > 1:
                fold_cliff.append(metric_fn(y_val[val_cliff], y_pred[val_cliff]))
                if not is_clf:
                    fold_rmse_cliff.append(float(np.sqrt(mean_squared_error(
                        y_val[val_cliff], y_pred[val_cliff]))))
            if len(val_noncliff) > 1:
                fold_noncliff.append(metric_fn(y_val[val_noncliff], y_pred[val_noncliff]))
                if not is_clf:
                    fold_rmse_noncliff.append(float(np.sqrt(mean_squared_error(
                        y_val[val_noncliff], y_pred[val_noncliff]))))

        if not fold_all:
            return None
        out = {
            'score_all': float(np.mean(fold_all)),
            'score_cliff': float(np.mean(fold_cliff)) if fold_cliff else float('nan'),
            'score_noncliff': float(np.mean(fold_noncliff)) if fold_noncliff else float('nan'),
            'metric': metric_name,
            # Positional overall per-fold scores for the honest_rescore block.
            'fold_scores': [float(s) for s in fold_scores_pos],
        }
        if not is_clf:
            out['rmse_all'] = float(np.mean(fold_rmse_all))
            out['rmse_cliff'] = float(np.mean(fold_rmse_cliff)) if fold_rmse_cliff else float('nan')
            out['rmse_noncliff'] = float(np.mean(fold_rmse_noncliff)) if fold_rmse_noncliff else float('nan')
        return out

    # --- Precompute per-fold feature importance per rep (hoisted from lambda) ---
    # Cache fold train indices so we compute importance once per (fold, rep).
    fold_splits = list(
        splitter.split(sample_X, labels, groups=groups) if use_groups
        else splitter.split(sample_X, labels)
    )
    def _fold_key(tr):
        return (int(tr[0]), int(tr[-1]), int(len(tr)))
    fold_importance = {}  # {rep_name: {fold_key: imp_array}}
    for rep_name, X_rep in reps.items():
        fold_importance[rep_name] = {}
        for tr, _va in fold_splits:
            try:
                imp = compute_feature_importance(
                    {rep_name: X_rep[tr]}, labels[tr], method='lightgbm_gain')
                fold_importance[rep_name][_fold_key(tr)] = imp[rep_name]
            except Exception as exc:
                # N9: log so a None-importance fold reads as "compute failed"
                # rather than silently weakening the weighted kNN.
                print(f"      importance fold failed (rep={rep_name}): "
                      f"{type(exc).__name__}: {exc}")
                fold_importance[rep_name][_fold_key(tr)] = None

    # --- Per-rep kNN baselines (unweighted + weighted) ---
    print(f"\n  Per-rep kNN baselines (k=5, metric={metric_name}):")
    for rep_name, X_rep in reps.items():
        # Unweighted
        def _get_rep(tr, va, yt, _X=X_rep):
            return _X[tr], _X[va]
        res = _evaluate_condition(_get_rep)
        if res:
            results['per_rep'][rep_name] = res
            # Plain single-rep kNN is a baseline arm for honest_rescore (model=knn).
            honest_baselines.setdefault('knn', {})[rep_name] = res['fold_scores']

            # Weighted: importance-weighted distance on this rep (precomputed per fold)
            def _get_rep_w(tr, va, yt, _X=X_rep, _rname=rep_name):
                imp_arr = fold_importance[_rname].get(_fold_key(tr))
                if imp_arr is None:
                    return None, None
                w = np.sqrt(np.maximum(imp_arr, 0))
                scaler = StandardScaler()
                return scaler.fit_transform(_X[tr]) * w, scaler.transform(_X[va]) * w
            res_w = _evaluate_condition(_get_rep_w, pre_scaled=True)
            if res_w:
                results['per_rep'][rep_name]['score_cliff_weighted'] = res_w['score_cliff']
                results['per_rep'][rep_name]['score_all_weighted'] = res_w['score_all']
                if not is_clf:
                    results['per_rep'][rep_name]['rmse_cliff_weighted'] = res_w.get('rmse_cliff', float('nan'))
                # Importance-weighted single-rep kNN is still a single-rep baseline
                # (no cross-rep mixing) — record as a distinct baseline entry.
                honest_baselines.setdefault('knn', {})[f'{rep_name}|weighted'] = res_w['fold_scores']

            rmse_part = f"  RMSE_cliff={res.get('rmse_cliff', float('nan')):.4f}" if not is_clf else ""
            w_part = ""
            if res_w and not is_clf:
                w_part = f"  | weighted: RMSE_cliff={res_w.get('rmse_cliff', float('nan')):.4f}"
            print(f"    {rep_name:<15} {metric_name}_all={res['score_all']:.4f}  "
                  f"{metric_name}_cliff={res['score_cliff']:.4f}{rmse_part}  "
                  f"{metric_name}_noncliff={res['score_noncliff']:.4f}"
                  + w_part)

    # --- Hybrid kNN (unweighted) ---
    if hybrid_blueprint is not None:
        print("\n  Hybrid kNN (unweighted, k=5):")

        augmentations = data.get('augmentations')

        def _get_hybrid_uw(tr, va, yt):
            train_reps_h = {name: rep[tr] for name, rep in reps.items()}
            val_reps_h = {name: rep[va] for name, rep in reps.items()}
            aug_tr = {k: v[tr] for k, v in augmentations.items()} if augmentations else None
            aug_va = {k: v[va] for k, v in augmentations.items()} if augmentations else None
            X_tr_h, X_va_h, feat_info = build_fold_hybrid(
                train_reps_h, val_reps_h, yt, hybrid_blueprint,
                augmentations_train=aug_tr, augmentations_val=aug_va,
                groups=(groups[tr] if groups is not None else None),
                model=_make_knn(len(yt)),
                metric_fn=metric_fn,
                lower_is_better=lower_is_better)
            return X_tr_h, X_va_h

        res = _evaluate_condition(_get_hybrid_uw)
        if res:
            results['hybrid']['unweighted'] = res
            # Hybrid kNN is a candidate arm for honest_rescore (model=knn).
            honest_candidates.setdefault('knn', []).append(
                {'name': 'hybrid_unweighted', 'fold_scores': res['fold_scores']})
            rmse_part = f"  RMSE_cliff={res.get('rmse_cliff', float('nan')):.4f}" if not is_clf else ""
            print(f"    hybrid_uw      {metric_name}_all={res['score_all']:.4f}  "
                  f"{metric_name}_cliff={res['score_cliff']:.4f}{rmse_part}  "
                  f"{metric_name}_noncliff={res['score_noncliff']:.4f}")
        else:
            # F17: every fold of this condition was dropped, so 'unweighted' is
            # absent. Name that instead of leaving the key simply missing.
            results['hybrid']['unweighted_status'] = {
                'computed': False,
                'reason': 'no cliff-CV fold produced a score (all folds skipped or threw)',
            }
            print("    hybrid_uw      not computed: all cliff-CV folds dropped")

        # --- Hybrid kNN (importance-weighted) ---
        print("  Hybrid kNN (weighted, k=5):")

        def _get_hybrid_w(tr, va, yt):
            train_reps_h = {name: rep[tr] for name, rep in reps.items()}
            val_reps_h = {name: rep[va] for name, rep in reps.items()}
            aug_tr = {k: v[tr] for k, v in augmentations.items()} if augmentations else None
            aug_va = {k: v[va] for k, v in augmentations.items()} if augmentations else None
            X_tr_h, X_va_h, feat_info = build_fold_hybrid(
                train_reps_h, val_reps_h, yt, hybrid_blueprint,
                augmentations_train=aug_tr, augmentations_val=aug_va,
                groups=(groups[tr] if groups is not None else None),
                model=_make_knn(len(yt)),
                metric_fn=metric_fn,
                lower_is_better=lower_is_better)
            imp = compute_feature_importance(train_reps_h, yt, method='lightgbm_gain')
            w_parts = []
            for rname in feat_info:
                if not isinstance(feat_info[rname], dict):
                    continue
                if 'selected_indices' not in feat_info[rname]:
                    continue
                idx = feat_info[rname]['selected_indices']
                if rname in imp:
                    w_parts.append(imp[rname][idx])
                else:
                    w_parts.append(np.ones(len(idx)))
            w = np.sqrt(np.maximum(np.concatenate(w_parts), 0))
            # Scale-then-weight (matches Exp 5 and compute_distance_matrix convention)
            scaler_w = StandardScaler()
            return scaler_w.fit_transform(X_tr_h) * w, scaler_w.transform(X_va_h) * w

        res = _evaluate_condition(_get_hybrid_w, pre_scaled=True)
        if res:
            results['hybrid']['weighted'] = res
            honest_candidates.setdefault('knn', []).append(
                {'name': 'hybrid_weighted', 'fold_scores': res['fold_scores']})
            rmse_part = f"  RMSE_cliff={res.get('rmse_cliff', float('nan')):.4f}" if not is_clf else ""
            print(f"    hybrid_w       {metric_name}_all={res['score_all']:.4f}  "
                  f"{metric_name}_cliff={res['score_cliff']:.4f}{rmse_part}  "
                  f"{metric_name}_noncliff={res['score_noncliff']:.4f}")
        else:
            # F17: same as above for the importance-weighted condition.
            results['hybrid']['weighted_status'] = {
                'computed': False,
                'reason': 'no cliff-CV fold produced a score (all folds skipped or threw)',
            }
            print("    hybrid_w       not computed: all cliff-CV folds dropped")
    else:
        # F17: no blueprint means NEITHER hybrid cliff score is attempted. Record
        # both absences and which of the two causes applies — the search threw, or
        # it returned without a kNN blueprint.
        _st = results.get('hybrid_search_status', {})
        if not _st.get('ok', False):
            _reason = (f"hybrid search raised "
                       f"{_st.get('error_type', 'Exception')}: {_st.get('error', '')}")
        else:
            _reason = 'hybrid search returned no kNN blueprint'
        for _cond in ('unweighted', 'weighted'):
            results['hybrid'][f'{_cond}_status'] = {
                'computed': False,
                'reason': _reason,
            }
        print(f"\n  Hybrid kNN: not computed ({_reason})")

    # --- RF baseline (on best individual rep for RF, from cached search) ---
    print("\n  RF baseline (per-rep, best rep for RF):")
    best_rep_name = None
    if search is not None:
        rf_top = extract_top3_base_reps(search, model='rf')
        if rf_top:
            best_rep_name = rf_top[0][0]
    # Fallback: if no cached RF ranking available, use best-kNN rep from per_rep.
    if best_rep_name is None and results['per_rep']:
        best_rep_name = (min if lower_is_better else max)(
            results['per_rep'], key=lambda r: results['per_rep'][r]['score_all']
        )
    if best_rep_name:
        X_best = reps[best_rep_name]
        rf_factory = get_model_factories(['rf'], task_type=task_type,
                                          n_samples=len(labels))['rf']

        fold_all, fold_cliff, fold_noncliff = [], [], []
        fold_rmse_cliff = []
        for train_idx, val_idx in splitter.split(sample_X, labels, groups=groups) if use_groups else splitter.split(sample_X, labels):
            y_train, y_val = labels[train_idx], labels[val_idx]
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_best[train_idx])
            X_va = scaler.transform(X_best[val_idx])
            rf = rf_factory()
            rf.fit(X_tr, y_train)
            y_pred = rf.predict(X_va)

            fold_all.append(metric_fn(y_val, y_pred))
            val_cliff = np.array([i for i, vi in enumerate(val_idx) if vi in cliff_indices])
            val_noncliff = np.array([i for i, vi in enumerate(val_idx) if vi not in cliff_indices])
            if len(val_cliff) > 1:
                fold_cliff.append(metric_fn(y_val[val_cliff], y_pred[val_cliff]))
                if not is_clf:
                    fold_rmse_cliff.append(float(np.sqrt(mean_squared_error(
                        y_val[val_cliff], y_pred[val_cliff]))))
            if len(val_noncliff) > 1:
                fold_noncliff.append(metric_fn(y_val[val_noncliff], y_pred[val_noncliff]))

        results['rf_baseline'] = {
            'rep': best_rep_name,
            'score_all': float(np.mean(fold_all)),
            'score_cliff': float(np.mean(fold_cliff)) if fold_cliff else float('nan'),
            'score_noncliff': float(np.mean(fold_noncliff)) if fold_noncliff else float('nan'),
            'metric': metric_name,
        }
        # RF is a separate downstream model — record its per-fold single-rep
        # baseline under model=rf. No RF hybrid candidate exists here, so the
        # honest delta for 'rf' is NaN (hybrid arm empty); the helper tolerates it.
        honest_baselines.setdefault('rf', {})[best_rep_name] = [float(s) for s in fold_all]
        if not is_clf:
            results['rf_baseline']['rmse_cliff'] = float(np.mean(fold_rmse_cliff)) if fold_rmse_cliff else float('nan')
        rmse_part = f"  RMSE_cliff={results['rf_baseline'].get('rmse_cliff', float('nan')):.4f}" if not is_clf else ""
        print(f"    RF({best_rep_name}) {metric_name}_all={results['rf_baseline']['score_all']:.4f}  "
              f"{metric_name}_cliff={results['rf_baseline']['score_cliff']:.4f}"
              f"{rmse_part}")

    # --- Summary ---
    print(f"\n  --- Summary ({metric_name}_cliff comparison) ---")
    all_methods = []
    for rname, rv in results['per_rep'].items():
        all_methods.append((f"kNN({rname})", rv.get('score_cliff', float('nan'))))
        if 'score_cliff_weighted' in rv:
            all_methods.append((f"kNN({rname})_w", rv.get('score_cliff_weighted', float('nan'))))
    if 'unweighted' in results.get('hybrid', {}):
        rv = results['hybrid']['unweighted']
        all_methods.append(("hybrid_kNN_uw", rv.get('score_cliff', float('nan'))))
    if 'weighted' in results.get('hybrid', {}):
        rv = results['hybrid']['weighted']
        all_methods.append(("hybrid_kNN_w", rv.get('score_cliff', float('nan'))))
    if 'rf_baseline' in results:
        rv = results['rf_baseline']
        all_methods.append((f"RF({rv['rep']})", rv.get('score_cliff', float('nan'))))

    _nan_sentinel = np.inf if lower_is_better else -np.inf
    all_methods.sort(key=lambda x: x[1] if not np.isnan(x[1]) else _nan_sentinel,
                     reverse=not lower_is_better)
    print(f"  {'Method':<25} {metric_name + '_cliff':<12}")
    print("  " + "-" * 37)
    for name, score in all_methods:
        print(f"  {name:<25} {score:<12.4f}")

    # --- Honest-rescore block (CONTRACT json_convention) ---
    # Reader: honest_rescore(block['candidates_by_model'],
    #                        block['baselines_by_model'], block['lower_is_better']).
    results['honest_rescore'] = {
        'lower_is_better': bool(lower_is_better),
        'candidates_by_model': honest_candidates,
        'baselines_by_model': honest_baselines,
    }

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('S04: Activity Cliffs',
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
    save_experiment_results(results, args.output, 's04_activity_cliffs', args.dataset)
