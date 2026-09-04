#!/usr/bin/env python3
"""S08: Optimal Dimension for Similarity vs Prediction.

Paper: Similarity
Hypothesis: Optimal dimensionality may differ for similarity vs prediction.
Dependencies: None (standalone).

At each budget, runs get_hybrid_features_and_weights with specific budget and
computes: MLE intrinsic dim, SAC, violation rate, distance correlation with
full concat, RF R2 and kNN R2. Also computes per-rep baselines and concat
reference. Finds optimal budget for prediction (RF) vs similarity (SAC).

Includes Pareto frontier analysis to quantify the SAC-vs-prediction trade-off
and determine whether the optima genuinely differ or coincide.

Self-diagnosis (2026-06-03): an "optimal budget" is only meaningful if several
DISTINCT feature dimensions were evaluated. Budgets cap at the feature pool, so
many requested budgets collapse to the same actual_dim, giving tied SAC and a
noise-driven argmax. The output now carries `n_distinct_dims` /
`insufficient_budgets`, and `pareto.tradeoff.optima_differ` is suppressed (None,
`optima_differ_trustworthy=False`) when fewer than 4 distinct dims were seen.
The budget grid is also deduped/sorted to the full DISTANCE_BUDGETS sweep.

Honest-SAC redesign (2026-07-11, decision B, user-approved): the SAC arm
(`best_sac_budget` = argmax SAC over budgets) used to be computed on the FULL
dataset — feature selection and SAC scoring shared every sample, the same leak
the prediction arm already avoids. SAC is now ALSO measured under CV: for each
budget (hybrid) and each rep (single-rep baseline) we rebuild the hybrid on the
TRAIN fold and score SAC on the HELD-OUT fold (`sac_fold_scores`), then feed
those per-fold scores to the shared `honest_rescore` helper, which re-picks the
best budget/rep per held-out fold from the OTHER folds. The honest result is in
`results['sac_honest']` and the reader-ready block is `results['honest_rescore']`
(keyed by a pseudo-model 'sac'; lower_is_better=False because SAC is
higher-is-better regardless of the dataset's prediction metric). The old
full-data `best_sac_budget` / Pareto fields are kept unchanged for back-compat
but are the optimistic version. The prediction arm is unchanged.

Added 2026-08-26. No existing key changes value:
- `budgets[b]['sac_fold_scores_unweighted']`, `results['sac_honest_unweighted']`
  and `results['honest_rescore_unweighted']`: the hybrid held-out SAC recomputed
  WITHOUT importance weights. The single-rep baselines are scored unweighted, so
  this arm differs from them in feature selection only. `results['sac_honest']`
  keeps the weighted hybrid arm against those same unweighted baselines, i.e. it
  differs in feature selection AND weighting at once.
  `results['sac_weighting_effect']` carries, per budget, the mean of each arm's
  held-out fold scores and the weighted-minus-unweighted difference.
- `results['optimal']['best_model_budgets_nan_skipped']` and
  `['best_sac_budget_nan_skipped']`: the same argmin/argmax with NaN-scored
  budgets dropped, None when no budget had a non-NaN score. The dropped budgets
  are listed in `budgets_without_score_by_model` / `budgets_without_sac`. The
  original `best_model_budgets` / `best_sac_budget` are unchanged: they compare
  NaN scores, so they can name a budget whose score was NaN.
  `best_rf_budget_nan_skipped` / `best_knn_budget_nan_skipped` are flat aliases
  of the same two entries, beside `best_rf_budget` / `best_knn_budget`.
- `results['pareto']['tradeoff']` now carries each quantity twice. The existing
  keys (`sac_loss_at_rf_optimum`, `rf_loss_at_sac_optimum`, `optima_differ`,
  `optima_differ_trustworthy`, `budget_gap`) keep their old values, read at the
  NaN-compared optima now also stored as `rf_budget_used` / `sac_budget_used`.
  The `_nan_skipped` twin of each key is read at the NaN-skipped optima
  (`rf_budget_used_nan_skipped` / `sac_budget_used_nan_skipped`). When either
  NaN-skipped optimum is None, the `_nan_skipped` numbers are NaN/None and
  `nan_skipped_unavailable_reason` names the objective with no scored budget;
  the NaN-compared pick is not substituted.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_fitted_hybrid, get_fitted_pool_hybrid, flatten_fold_weights,
    estimate_intrinsic_dim_mle, make_cv_splitter,
    get_domain_metric, get_model_factories,
    build_fold_hybrid, extract_top3_base_reps, honest_rescore,
    DISTANCE_BUDGETS, DISTANCE_EVALUATORS, DISTANCE_MODELS,
)
from experiments.similarity.similarity_helpers import (
    similarity_activity_correlation, violation_rate,
)
from kirby.utils.noise_mitigation_methods import compute_distance_matrix


def run(data, config, output_dir, dataset_name, **kwargs):
    """The optimal hybrid dimension may differ for prediction vs similarity.
    Similarity may peak at lower dim than prediction (curse of dimensionality)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT S08: OPTIMAL DIMENSION FOR SIMILARITY VS PREDICTION")
    print("=" * 70)

    n_folds = config['n_splits']
    # knn is a distance-tier evaluator, so budgets must come from
    # DISTANCE_BUDGETS=[10,25,50,100,250,500] to match the cached lineage.
    budgets = kwargs.get('budgets', config.get('budgets', list(DISTANCE_BUDGETS)))
    # Keep only cache-lineage budgets, dedupe, and sort so the full
    # DISTANCE_BUDGETS=[10,25,50,100,250,500] grid is swept (older runs that
    # passed [50, 500] are the reason "optima" looked meaningless).
    budgets = sorted({b for b in budgets if b in DISTANCE_BUDGETS})
    n_sample = kwargs.get('n_sample', 5000)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    augmentations = data.get('augmentations')

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    task_type = data['task_type']
    domain = data.get('domain', 'molecular')
    is_clf = task_type == 'classification'
    # DISTANCE_MODELS, not DOWNSTREAM_MODELS: it is the same five plus knn, and
    # it is what every distance-lineage search already scores. s09's
    # optimal_dim_ratio divides the kNN-optimal budget by the RF-optimal one and
    # never ran, because best_model_budgets carried no 'knn' entry.
    model_factories = get_model_factories(DISTANCE_MODELS, task_type=task_type,
                                           n_samples=len(labels))
    print(f"  Models: {list(model_factories.keys())}")

    results = {'budgets': {}}
    _top3_reported = False

    # Full-dim reference
    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
    full_dim = X_concat.shape[1]

    splitter, use_groups = make_cv_splitter(n_folds, groups,
                                            task_type=task_type)
    sample_X = next(iter(reps.values()))

    # Pre-compute full-dim distance matrix + sampled pair indices (reused per budget)
    D_full = compute_distance_matrix(X_concat, standardize=True)
    n = len(labels)
    triu_i, triu_j = np.triu_indices(n, k=1)
    n_pairs = len(triu_i)
    if n_pairs > 5000:
        rng = np.random.RandomState(42)
        _idx = rng.choice(n_pairs, 5000, replace=False)
        triu_i_s, triu_j_s = triu_i[_idx], triu_j[_idx]
    else:
        triu_i_s, triu_j_s = triu_i, triu_j
    d_full_sampled = D_full[triu_i_s, triu_j_s]

    for budget in budgets:
        print(f"\n  Budget={budget}:")
        t0 = time.time()

        try:
            # Dimension is the study variable: pool helper picks best
            # (tier, aug_cond) cache file for (knn, budget). 'capped' enters
            # the comparison naturally via the pool (top tier carries it).
            X_hybrid, fi, search, _meta = get_fitted_pool_hybrid(
                reps, labels, groups,
                downstream_model='knn', budget=budget,
                augmentations=data.get('augmentations'),
                metric=metric_name, task_type=task_type, domain=domain,
                dataset_name=dataset_name, distance=True)
            # Clamp to >=0: weights feed compute_distance_matrix (X*sqrt(w),
            # no internal guard). lightgbm_gain is non-negative today, so this
            # is a no-op now but prevents silent NaN if importance ever becomes
            # signed. Mirrors the per-rep clamp below.
            weights_flat = (np.maximum(flatten_fold_weights(fi), 0) if fi
                            else np.ones(X_hybrid.shape[1]))
            if not _top3_reported:
                top3 = extract_top3_base_reps(search)
                if top3:
                    # Use distinct loop vars (rep_n/rep_s) to avoid shadowing the
                    # scalar `n = len(labels)` bound above (OTHER-32).
                    print(f"  Top-3 base reps (by RF): {', '.join(f'{rep_n}={rep_s:.4f}' for rep_n, rep_s in top3)}")
                    results['top3_base_reps'] = [(rep_n, rep_s) for rep_n, rep_s in top3]
                _top3_reported = True
        except Exception as e:
            print(f"    Failed: {e}")
            results['budgets'][budget] = {'error': str(e)}
            continue

        actual_dim = X_hybrid.shape[1]

        # MLE intrinsic dimensionality of the hybrid
        if len(X_hybrid) <= 11:
            print(f"    Skipping MLE_ID: n={len(X_hybrid)} too small for k=10")
            mle_id = float('nan')
        else:
            try:
                X_hybrid_scaled = StandardScaler().fit_transform(X_hybrid)
                mle_id = float(estimate_intrinsic_dim_mle(X_hybrid_scaled, k=10))
            except Exception:
                mle_id = float('nan')

        # Similarity quality
        D_weighted = compute_distance_matrix(X_hybrid, feature_weights=weights_flat,
                                             standardize=True)
        sac_rho, _ = similarity_activity_correlation(D_weighted, labels, n_sample, is_clf=is_clf)
        viol = violation_rate(D_weighted, labels, k=5, is_clf=is_clf)

        # Distance correlation with full concat (using pre-computed pairs)
        d_hybrid_sampled = D_weighted[triu_i_s, triu_j_s]
        dist_corr, _ = spearmanr(d_full_sampled, d_hybrid_sampled)

        # Prediction quality: per-model CV -- use search blueprint's method
        bp = search.best_by_model['knn']['blueprint'] if search.best_by_model.get('knn') else None
        if bp is None:
            print(f"    No blueprint at budget={budget}, skipping prediction")
            continue

        # 2026-04-21 cache-collision fix: align blueprint.augmentation_strategy
        # with the pool helper's winning aug_condition BEFORE fold loop so
        # build_fold_hybrid sees the correct strategy.
        aug_cond = _meta.get('aug_condition', 'uncapped') if _meta else 'uncapped'
        bp.augmentation_strategy = aug_cond

        # Every model is fitted here, at THIS loop's budget. The cache shortcut
        # that used to sit here copied search.best_by_model[model]['fold_scores']
        # in whenever that entry's budget matched, and those scores came from the
        # search's own winning configuration, not from a rebuild at this budget.
        # With the shortcut in place five of six budgets scored identically and
        # the sixth was the search's own number, so the sweep measured nothing.
        model_scores = {mname: [] for mname in model_factories}
        refit_models = dict(model_factories)

        # ── Honest-SAC redesign: the fold loop rebuilds a hybrid on each
        # TRAIN fold, and both the held-out SAC and every model's prediction
        # score are measured on it at THIS loop's budget.
        # sac_fold_scores[f] = SAC measured on held-out fold f using a hybrid
        # whose feature selection + weights were fit on that fold's TRAIN rows
        # only, so no sample is used to both select and score. This replaces the
        # old full-data best_sac_budget leak with a cross-validated statistic.
        sac_fold_scores = []
        sac_fold_scores_unweighted = []  # same folds, no importance weights
        for train_idx, val_idx in splitter.split(sample_X, labels, groups=groups) if use_groups else splitter.split(sample_X, labels):
            y_train, y_val = labels[train_idx], labels[val_idx]

            train_reps_fold = {name: rep[train_idx] for name, rep in reps.items()}
            val_reps_fold = {name: rep[val_idx] for name, rep in reps.items()}
            # Skip augmentations when winning condition is 'none' — the
            # pool-fitted X_hybrid was built without them, mismatch otherwise.
            if aug_cond == 'none' or not augmentations:
                aug_tr, aug_va = None, None
            else:
                aug_tr = {k: v[train_idx] for k, v in augmentations.items()}
                aug_va = {k: v[val_idx] for k, v in augmentations.items()}

            try:
                X_tr_h, X_va_h, feat_info = build_fold_hybrid(
                    train_reps_fold, val_reps_fold, y_train, bp,
                    augmentations_train=aug_tr, augmentations_val=aug_va,
                    groups=(groups[train_idx] if groups is not None else None),
                    # The swept axis. Without this every step of the sweep
                    # rebuilds at the blueprint's own budget, so the curve is
                    # the same hybrid measured over and over.
                    budget_override=budget)
            except Exception:
                continue

            # Held-out SAC: distance quality on the val fold using train-fold
            # weights. Clamp weights >=0 (sqrt-weighting has no guard) and fall
            # back to unweighted if the flattened length ever disagrees with the
            # rebuilt column count.
            try:
                _w_fold = np.maximum(flatten_fold_weights(feat_info), 0)
                _w_fold = _w_fold if _w_fold.size == X_va_h.shape[1] else None
                D_va = compute_distance_matrix(X_va_h, feature_weights=_w_fold,
                                               standardize=True)
                sac_va, _ = similarity_activity_correlation(
                    D_va, y_val, n_sample, is_clf=is_clf)
                sac_fold_scores.append(float(sac_va))
            except Exception:
                pass

            # Second hybrid arm (2026-08-26): the SAME held-out fold and the same
            # train-fold feature selection, scored with UNWEIGHTED distances --
            # the identical statistic the single-rep baselines use below. The
            # weighted arm above differs from those baselines in two ways at
            # once (which features are kept AND how they are weighted); this arm
            # differs only in which features are kept.
            try:
                D_va_unw = compute_distance_matrix(X_va_h, standardize=True)
                sac_va_unw, _ = similarity_activity_correlation(
                    D_va_unw, y_val, n_sample, is_clf=is_clf)
                sac_fold_scores_unweighted.append(float(sac_va_unw))
            except Exception:
                pass

            # Prediction arm (unchanged): only fit models missing from cache.
            if refit_models:
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr_h)
                X_va_s = scaler.transform(X_va_h)
                for mname, mfactory in refit_models.items():
                    try:
                        m = mfactory()
                        m.fit(X_tr_s, y_train)
                        model_scores[mname].append(metric_fn(y_val, m.predict(X_va_s)))
                    except Exception:
                        pass

        model_means = {mname: float(np.mean(s)) if s else float('nan')
                       for mname, s in model_scores.items()}

        elapsed = time.time() - t0

        budget_result = {
            'actual_dim': actual_dim,
            'mle_id': mle_id,
            'metric': metric_name,
            'sac_rho': sac_rho,  # FULL-data SAC (optimistic; kept for back-compat)
            'sac_fold_scores': sac_fold_scores,  # held-out SAC per fold, importance-weighted
            # held-out SAC per fold, unweighted (same statistic as per_rep)
            'sac_fold_scores_unweighted': sac_fold_scores_unweighted,
            'violation_rate': viol,
            'dist_corr_with_full': dist_corr,
            'time_seconds': elapsed,
            'model_scores': model_means,
        }
        # Keep rf_score/knn_score for backwards compat with Pareto/optimal analysis
        budget_result['rf_score'] = model_means.get('rf', float('nan'))
        budget_result['knn_score'] = model_means.get('knn', float('nan'))
        results['budgets'][budget] = budget_result

        scores_str = ', '.join(f"{mn}={model_means[mn]:.4f}" for mn in model_means
                               if not np.isnan(model_means[mn]))
        print(f"    dim={actual_dim} (MLE_ID={mle_id:.1f}): {scores_str}, "
              f"SAC={sac_rho:.4f}, viol={viol:.3f}, dist_corr={dist_corr:.3f}")

    # Per-rep baselines (unweighted + weighted)
    print("\n  Individual rep baselines:")
    results['per_rep'] = {}
    best_rep_sac = -1
    best_rep_name = None

    # Compute importance once for weighted baselines
    from kirby.hybrid import compute_feature_importance as _cfi
    try:
        importance = _cfi(reps, labels, method='lightgbm_gain', task_type=task_type)
    except Exception:
        importance = None

    for rep_name, X in reps.items():
        D_rep = compute_distance_matrix(X, standardize=True)
        sac_rep, _ = similarity_activity_correlation(D_rep, labels, n_sample, is_clf=is_clf)
        viol_rep = violation_rate(D_rep, labels, k=5, is_clf=is_clf)

        # Weighted SAC
        sac_rep_w = float('nan')
        if importance is not None and rep_name in importance:
            # Clamp importance to >=0 before passing: compute_distance_matrix
            # does X * sqrt(feature_weights) with NO internal guard, so any
            # negative weight (future signed importance) would inject NaN into
            # the distance matrix and silently corrupt SAC. lightgbm_gain is
            # non-negative today; this makes the contract explicit and matches
            # the np.sqrt(np.maximum(.,0)) pattern used everywhere else in s02-s07.
            _w_rep = np.maximum(importance[rep_name], 0)
            D_rep_w = compute_distance_matrix(X, feature_weights=_w_rep,
                                              standardize=True)
            sac_rep_w, _ = similarity_activity_correlation(D_rep_w, labels, n_sample, is_clf=is_clf)

        rep_model_scores = {mname: [] for mname in model_factories}
        rep_sac_fold_scores = []  # held-out SAC per fold (unweighted, honest)
        for train_idx, val_idx in splitter.split(sample_X, labels, groups=groups) if use_groups else splitter.split(sample_X, labels):
            y_train, y_val = labels[train_idx], labels[val_idx]
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_va = scaler.transform(X[val_idx])

            for mname, mfactory in model_factories.items():
                try:
                    m = mfactory()
                    m.fit(X_tr, y_train)
                    rep_model_scores[mname].append(metric_fn(y_val, m.predict(X_va)))
                except Exception:
                    pass

            # Held-out SAC for this single rep on the val fold. Unweighted: the
            # per-rep importance above is computed on the FULL dataset (leak), so
            # weighting the held-out distance would contaminate it; the unweighted
            # distance is the honest single-rep similarity baseline (same
            # definition as sac_rho, just measured out-of-fold).
            try:
                D_va_rep = compute_distance_matrix(X[val_idx], standardize=True)
                sac_va_rep, _ = similarity_activity_correlation(
                    D_va_rep, y_val, n_sample, is_clf=is_clf)
                rep_sac_fold_scores.append(float(sac_va_rep))
            except Exception:
                pass

        rep_model_means = {mname: float(np.mean(s)) if s else float('nan')
                           for mname, s in rep_model_scores.items()}

        # MLE ID for individual rep
        if len(X) <= 11:
            rep_mle = float('nan')
        else:
            try:
                rep_mle = float(estimate_intrinsic_dim_mle(
                    StandardScaler().fit_transform(X), k=10))
            except Exception:
                rep_mle = float('nan')

        rep_result = {
            'dim': X.shape[1],
            'mle_id': rep_mle,
            'metric': metric_name,
            'sac_rho': sac_rep,  # FULL-data SAC (optimistic; back-compat)
            'sac_rho_weighted': sac_rep_w,
            'sac_fold_scores': rep_sac_fold_scores,  # held-out SAC per fold (honest)
            'violation_rate': viol_rep,
            'model_scores': rep_model_means,
            'rf_score': rep_model_means.get('rf', float('nan')),
            'knn_score': rep_model_means.get('knn', float('nan')),
        }
        results['per_rep'][rep_name] = rep_result
        if best_rep_name is None or sac_rep > best_rep_sac:
            best_rep_sac = sac_rep
            best_rep_name = rep_name
        scores_str = ', '.join(f"{mn}={rep_model_means[mn]:.4f}" for mn in rep_model_means
                               if not np.isnan(rep_model_means[mn]))
        print(f"    {rep_name} (dim={X.shape[1]}): {scores_str}, "
              f"SAC={sac_rep:.4f}, SAC_w={sac_rep_w:.4f}")

    results['best_individual_rep'] = best_rep_name
    print(f"  Best individual rep: {best_rep_name} (SAC={best_rep_sac:.4f})")

    # Top-3 baselines per model
    _best_fn = min if lower_is_better else max
    _default = 999 if lower_is_better else -999
    results['top3_by_model'] = {}
    for mname in model_factories:
        top3 = sorted(results['per_rep'].items(),
                       key=lambda x: x[1].get('model_scores', {}).get(mname, _default),
                       reverse=not lower_is_better)[:3]
        results['top3_by_model'][mname] = [
            (name, r['model_scores'].get(mname, float('nan'))) for name, r in top3]
    # Backwards compat
    results['top3_rf'] = results['top3_by_model'].get('rf', [])
    results['top3_knn'] = results['top3_by_model'].get('knn', [])
    for mname in list(model_factories.keys())[:3]:
        print(f"\n  Top-3 baselines by {mname} {metric_name}:")
        for name, score in results['top3_by_model'][mname]:
            print(f"    {name}: {score:.4f}")

    # Find optimal budget for prediction (per model) vs similarity
    budget_list = sorted(results['budgets'].keys())

    # ── Self-diagnosis: an "optimal budget" is only meaningful if several
    # DISTINCT feature dimensions were actually evaluated. Many budgets collapse
    # to the same actual_dim (capped at the feature pool), so 6 budget requests
    # can yield only 1-2 distinct dims and SAC/argmax becomes floating-point
    # noise. Record the distinct-dim count and flag when it is too small. ──
    _evaluated = [b for b in budget_list if 'error' not in results['budgets'][b]]
    _distinct_dims = sorted({results['budgets'][b].get('actual_dim')
                             for b in _evaluated
                             if results['budgets'][b].get('actual_dim') is not None})
    results['n_budgets_evaluated'] = len(_evaluated)
    results['n_distinct_dims'] = len(_distinct_dims)
    results['distinct_dims'] = _distinct_dims
    insufficient = len(_distinct_dims) < 4
    results['insufficient_budgets'] = bool(insufficient)
    if insufficient:
        print(f"\n  [diag] only {len(_distinct_dims)} distinct feature dim(s) "
              f"evaluated ({_distinct_dims}) across {len(_evaluated)} budget(s) "
              f"-> 'optima_differ' is NOT trustworthy and is suppressed.")

    if budget_list:
        _best_fn = min if lower_is_better else max
        _default = 999 if lower_is_better else -999

        # Per-model optimal budget
        best_model_budgets = {}
        for mname in model_factories:
            best_model_budgets[mname] = _best_fn(
                budget_list,
                key=lambda b: results['budgets'][b].get('model_scores', {}).get(mname, _default))

        # Highest SAC = best distance-activity correlation
        best_sac_budget = max(budget_list,
                              key=lambda b: results['budgets'][b].get('sac_rho', -999))

        # ── NaN-skipped optima (2026-08-26), reported ALONGSIDE the two above.
        # A model with no successful fold is stored as float('nan') (see
        # model_means), not as a missing entry, so the `.get(..., _default)`
        # fallback never fires. max()/min() seed on the first item and every
        # comparison against NaN is False, so a NaN-scored budget survives and is
        # returned as the optimum. The keys below drop every budget whose score
        # is NaN or absent, are None when NO budget had a usable score, and list
        # the dropped budgets. The originals keep their old values so already
        # stored results still load. ──
        def _scored_budgets(score_of):
            """[(budget, score)] over budget_list, NaN and missing dropped."""
            out = []
            for b in budget_list:
                v = score_of(b)
                if v is None:
                    continue
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                if not np.isnan(v):
                    out.append((b, v))
            return out

        best_model_budgets_ns = {}
        budgets_without_score = {}
        for mname in model_factories:
            _scored = _scored_budgets(
                lambda b: results['budgets'][b].get('model_scores', {}).get(mname))
            _have = {b for b, _ in _scored}
            budgets_without_score[mname] = [b for b in budget_list if b not in _have]
            best_model_budgets_ns[mname] = (_best_fn(_scored, key=lambda t: t[1])[0]
                                            if _scored else None)

        _sac_scored = _scored_budgets(lambda b: results['budgets'][b].get('sac_rho'))
        _sac_have = {b for b, _ in _sac_scored}
        budgets_without_sac = [b for b in budget_list if b not in _sac_have]
        best_sac_budget_ns = (max(_sac_scored, key=lambda t: t[1])[0]
                              if _sac_scored else None)

        results['optimal'] = {
            # Original selection: NaN scores take part in the comparison, so the
            # named budget may be one whose score was NaN.
            'best_model_budgets': best_model_budgets,
            'best_sac_budget': best_sac_budget,
            # NaN-skipped selection; None means no budget had a non-NaN score.
            'best_model_budgets_nan_skipped': best_model_budgets_ns,
            'best_sac_budget_nan_skipped': best_sac_budget_ns,
            'budgets_without_score_by_model': budgets_without_score,
            'budgets_without_sac': budgets_without_sac,
            'models_with_no_scored_budget': [mn for mn, b in best_model_budgets_ns.items()
                                             if b is None],
            'sac_has_no_scored_budget': bool(best_sac_budget_ns is None),
            # Backwards compat
            'best_rf_budget': best_model_budgets.get('rf'),
            'best_knn_budget': best_model_budgets.get('knn'),
            # Flat aliases of the NaN-skipped picks, beside the two above.
            'best_rf_budget_nan_skipped': best_model_budgets_ns.get('rf'),
            'best_knn_budget_nan_skipped': best_model_budgets_ns.get('knn'),
        }
        # Pareto below reports the trade-off at BOTH selections: the original
        # argmin/argmax keeps its existing keys, the NaN-skipped optima are
        # added beside it under '_nan_skipped' keys.
        best_rf_budget = best_model_budgets.get('rf', budget_list[0])
        best_rf_budget_ns = best_model_budgets_ns.get('rf')
        budget_strs = ', '.join(f"{mn}={b}" for mn, b in best_model_budgets.items())
        print(f"\n  Optimal budgets [NaN compared]: {budget_strs}, SAC={best_sac_budget}")
        budget_strs_ns = ', '.join(
            f"{mn}=" + ("none scored" if b is None else str(b))
            for mn, b in best_model_budgets_ns.items())
        print(f"  Optimal budgets [NaN skipped]: {budget_strs_ns}, SAC="
              + ("none scored" if best_sac_budget_ns is None else str(best_sac_budget_ns)))
        for mn, skipped in budgets_without_score.items():
            if skipped:
                print(f"    {mn}: no usable score at budget(s) {skipped}")
        if budgets_without_sac:
            print(f"    sac_rho: no usable score at budget(s) {budgets_without_sac}")

        # ── Pareto frontier: SAC (higher=better) vs RF (direction depends on metric) ──
        # Both objectives should be "higher is better" for Pareto.
        # RF score is already in the right direction (metric_fn handles it).
        # For lower-is-better metrics (RMSE), negate so higher = better.
        pareto_points = []
        for b in budget_list:
            bd = results['budgets'][b]
            sac = bd.get('sac_rho', float('nan'))
            rf = bd.get('rf_score', float('nan'))
            if np.isnan(sac) or np.isnan(rf):
                continue
            rf_max = -rf if lower_is_better else rf  # convert to higher=better
            pareto_points.append((b, sac, rf_max, rf))

        pareto_front = []
        if pareto_points:
            # Sort by SAC descending; sweep for Pareto-optimal
            pareto_points.sort(key=lambda x: x[1], reverse=True)
            best_rf_seen = float('-inf')
            for budget_val, sac, rf_max, rf_orig in pareto_points:
                if rf_max > best_rf_seen:
                    pareto_front.append({
                        'budget': budget_val,
                        'sac_rho': sac,
                        f'rf_{metric_name}': rf_orig,
                    })
                    best_rf_seen = rf_max

            # Quantify trade-off: how much SAC do you lose going from
            # SAC-optimal to RF-optimal budget, and vice versa?
            def _tradeoff_pair(rf_b, sac_b):
                """SAC loss at the RF budget, RF loss at the SAC budget, gap."""
                _sac_at_rf = results['budgets'][rf_b].get('sac_rho', float('nan'))
                _rf_at_sac = results['budgets'][sac_b].get('rf_score', float('nan'))
                _sac_at_sac = results['budgets'][sac_b].get('sac_rho', float('nan'))
                _rf_at_rf = results['budgets'][rf_b].get('rf_score', float('nan'))
                _rf_loss = (float(_rf_at_sac - _rf_at_rf) if lower_is_better
                            else float(_rf_at_rf - _rf_at_sac))
                return (float(_sac_at_sac - _sac_at_rf), _rf_loss, abs(rf_b - sac_b))

            # Original selection: NaN scores took part in the argmin/argmax, so
            # best_rf_budget / best_sac_budget can name a budget whose own score
            # was NaN. These four numbers keep the keys they have always had.
            sac_loss, rf_loss, budget_gap = _tradeoff_pair(best_rf_budget, best_sac_budget)
            # optima_differ is only trustworthy with >=4 distinct dims; otherwise
            # the argmax is decided by SAC ties / float noise -> report None.
            optima_differ = (None if insufficient
                             else best_rf_budget != best_sac_budget)

            # ── Same arithmetic at the NaN-skipped optima (2026-08-26). With
            # budgets 50 (NaN), 100 (0.3) and 200 (no score) the original pick is
            # 50 and this pick is 100, so the two sets of numbers come from
            # different budgets. When a NaN-skipped optimum is None -- every
            # budget scored NaN for that objective -- the '_nan_skipped' numbers
            # are NaN/None and the reason is recorded; the original pick is NOT
            # substituted. ──
            _ns_missing = [_n for _n, _b in (('rf', best_rf_budget_ns),
                                             ('sac', best_sac_budget_ns))
                           if _b is None]
            if _ns_missing:
                ns_sac_loss = float('nan')
                ns_rf_loss = float('nan')
                ns_budget_gap = None
                ns_optima_differ = None
                ns_reason = ('no budget had a non-NaN score for: '
                             + ', '.join(_ns_missing))
            else:
                ns_sac_loss, ns_rf_loss, ns_budget_gap = _tradeoff_pair(
                    best_rf_budget_ns, best_sac_budget_ns)
                ns_optima_differ = (None if insufficient
                                    else best_rf_budget_ns != best_sac_budget_ns)
                ns_reason = None

            tradeoff = {
                # Read at the NaN-compared optima.
                'sac_loss_at_rf_optimum': sac_loss,
                'rf_loss_at_sac_optimum': rf_loss,
                'optima_differ': optima_differ,
                'optima_differ_trustworthy': not insufficient,
                'budget_gap': budget_gap,
                'rf_budget_used': best_rf_budget,
                'sac_budget_used': best_sac_budget,
                # Read at the NaN-skipped optima.
                'sac_loss_at_rf_optimum_nan_skipped': ns_sac_loss,
                'rf_loss_at_sac_optimum_nan_skipped': ns_rf_loss,
                'optima_differ_nan_skipped': ns_optima_differ,
                'optima_differ_trustworthy_nan_skipped': bool(
                    not insufficient and not _ns_missing),
                'budget_gap_nan_skipped': ns_budget_gap,
                'rf_budget_used_nan_skipped': best_rf_budget_ns,
                'sac_budget_used_nan_skipped': best_sac_budget_ns,
                'nan_skipped_unavailable_reason': ns_reason,
            }

            results['pareto'] = {
                'frontier': pareto_front,
                'n_pareto_points': len(pareto_front),
                'tradeoff': tradeoff,
            }

            print(f"\n  Pareto frontier ({len(pareto_front)} points):")
            for pt in pareto_front:
                print(f"    budget={pt['budget']:4d}  "
                      f"SAC={pt['sac_rho']:.4f}  "
                      f"RF_{metric_name}={pt[f'rf_{metric_name}']:.4f}")
            for (_label, _differ, _sac_loss, _rf_loss,
                 _rf_b, _sac_b, _reason) in (
                    ('NaN skipped', ns_optima_differ, ns_sac_loss, ns_rf_loss,
                     best_rf_budget_ns, best_sac_budget_ns, ns_reason),
                    ('NaN compared', optima_differ, sac_loss, rf_loss,
                     best_rf_budget, best_sac_budget, None)):
                if _reason:
                    print(f"  Trade-off [{_label}] not computed: {_reason}")
                elif _differ is None:
                    print(f"  Optima comparison [{_label}] suppressed "
                          f"(too few distinct dims).")
                elif _differ:
                    print(f"  Trade-off [{_label}]: "
                          f"SAC-opt@{_sac_b} vs RF-opt@{_rf_b}")
                    print(f"    SAC loss at RF-optimum: {_sac_loss:.4f}")
                    print(f"    RF loss at SAC-optimum: {_rf_loss:.4f}")
                else:
                    print(f"  No trade-off [{_label}]: both objectives peak "
                          f"at budget={_rf_b}")

    # ── Honest, cross-validated SAC comparison (the redesigned "optimal
    # similarity dimension" claim). best_sac_budget above is an argmax of SAC
    # computed on the FULL dataset — selection and scoring share every sample,
    # so it is optimistic in exactly the way the prediction arm avoids. Here we
    # instead reuse the per-fold HELD-OUT SAC collected per budget (hybrid
    # candidates) and per rep (single-rep baselines) and hand them to the shared
    # honest_rescore helper, which for each held-out fold re-picks the budget/rep
    # that was best on the OTHER folds and reads its held-out SAC. SAC is
    # higher-is-better regardless of the dataset's prediction metric, so
    # lower_is_better is hard-coded False for this arm (NOT get_domain_metric's
    # direction). Keyed by a single pseudo-model 'sac' because SAC is a property
    # of the distance, not of a downstream regressor.
    sac_candidates = {'sac': [
        {'name': f'budget_{b}',
         'fold_scores': results['budgets'][b].get('sac_fold_scores', [])}
        for b in budget_list
        if 'error' not in results['budgets'][b]
        and results['budgets'][b].get('sac_fold_scores')
    ]}
    sac_baselines = {'sac': {
        rep_name: r['sac_fold_scores']
        for rep_name, r in results['per_rep'].items()
        if r.get('sac_fold_scores')
    }}
    # Reader-ready block in the shared json_convention so analysis code can call
    # honest_rescore(block['candidates_by_model'], block['baselines_by_model'],
    # block['lower_is_better']) directly. NOTE: for s08 this block carries the
    # unsupervised SAC statistic (higher=better), not a prediction metric; the
    # '_arm' marker and lower_is_better=False make that explicit.
    results['honest_rescore'] = {
        'lower_is_better': False,
        # Every fold counts. The candidates here are the arms this file built itself,
        # not a search's recipe list, so no fold filtered them.
        'exclude_folds': [],
        'candidates_by_model': sac_candidates,
        'baselines_by_model': sac_baselines,
        '_arm': 'sac',
        # The two arms are not the same statistic: the hybrid candidates carry
        # importance-weighted held-out distances, the single-rep baselines carry
        # unweighted ones. The '_unweighted' block below holds the like-for-like
        # version (both arms unweighted).
        '_hybrid_distance': 'importance-weighted',
        '_baseline_distance': 'unweighted',
    }
    # ── Like-for-like SAC arm (2026-08-26). Same held-out folds, same candidate
    # budgets, same single-rep baselines; the hybrid fold scores here come from
    # sac_fold_scores_unweighted, so both arms use unweighted distances and the
    # comparison isolates feature selection. ──
    sac_candidates_unw = {'sac': [
        {'name': f'budget_{b}',
         'fold_scores': results['budgets'][b].get('sac_fold_scores_unweighted', [])}
        for b in budget_list
        if 'error' not in results['budgets'][b]
        and results['budgets'][b].get('sac_fold_scores_unweighted')
    ]}
    results['honest_rescore_unweighted'] = {
        'lower_is_better': False,
        # Every fold counts. The candidates here are the arms this file built itself,
        # not a search's recipe list, so no fold filtered them.
        'exclude_folds': [],
        'candidates_by_model': sac_candidates_unw,
        'baselines_by_model': sac_baselines,
        '_arm': 'sac_unweighted',
        '_hybrid_distance': 'unweighted',
        '_baseline_distance': 'unweighted',
    }
    # Per-budget mean of each hybrid arm's held-out fold scores and their
    # difference. Both are held out; they differ only in whether the held-out
    # distance used the train-fold importance weights.
    results['sac_weighting_effect'] = {}
    for b in budget_list:
        bd = results['budgets'][b]
        if 'error' in bd:
            continue
        _w = bd.get('sac_fold_scores') or []
        _u = bd.get('sac_fold_scores_unweighted') or []
        results['sac_weighting_effect'][b] = {
            'weighted_mean': float(np.mean(_w)) if _w else float('nan'),
            'unweighted_mean': float(np.mean(_u)) if _u else float('nan'),
            'weighted_minus_unweighted': (float(np.mean(_w) - np.mean(_u))
                                          if _w and _u else float('nan')),
            'n_folds_weighted': len(_w),
            'n_folds_unweighted': len(_u),
        }
    try:
        # Each budget's hybrid was rebuilt per fold, but the recipe at that
        # budget came from the pool search over every row.
        _sac_hr = honest_rescore(sac_candidates, sac_baselines,
                                 lower_is_better=False,
                                 recipe_selection='whole_dataset')
        results['sac_honest'] = _sac_hr.get('sac', {})
        _sh = results['sac_honest']
        if _sh:
            print(f"\n  Honest CV SAC (held-out) "
                  f"[hybrid weighted vs baselines unweighted]: "
                  f"hybrid={_sh.get('hybrid_honest', float('nan')):.4f}, "
                  f"best single={_sh.get('best_single_honest', float('nan')):.4f}, "
                  f"delta={_sh.get('delta', float('nan')):+.4f}"
                  + ("  [SINGLE-FOLD: optimistic, NOT honest]"
                     if _sh.get('single_fold') else ""))
    except Exception as e:
        print(f"  Honest CV SAC failed: {e}")
        results['sac_honest'] = {'error': str(e)}
    try:
        _sac_hr_u = honest_rescore(sac_candidates_unw, sac_baselines,
                                   lower_is_better=False,
                                   recipe_selection='whole_dataset')
        results['sac_honest_unweighted'] = _sac_hr_u.get('sac', {})
        _shu = results['sac_honest_unweighted']
        if _shu:
            print(f"  Honest CV SAC (held-out) [both arms unweighted]: "
                  f"hybrid={_shu.get('hybrid_honest', float('nan')):.4f}, "
                  f"best single={_shu.get('best_single_honest', float('nan')):.4f}, "
                  f"delta={_shu.get('delta', float('nan')):+.4f}"
                  + ("  [SINGLE-FOLD: optimistic, NOT honest]"
                     if _shu.get('single_fold') else ""))
    except Exception as e:
        print(f"  Honest CV SAC (unweighted) failed: {e}")
        results['sac_honest_unweighted'] = {'error': str(e)}

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('S08: Optimal Dimension for Similarity vs Prediction',
                                     has_delta=True, has_augment=True)
    # create_experiment_parser already defines --budgets (comma-separated), so
    # re-adding it here raised ArgumentError and s08 could not be launched from
    # its own command line at all. Read the shared flag instead.
    parser.add_argument('--n-sample', type=int, default=5000,
                        help='Number of pairs to sample for SAC (default: 5000)')
    args = parser.parse_args()
    config = get_config(args)

    # get_config turns --budgets into config['budgets']; with no flag it is
    # FULL_BUDGETS, which is not the distance lineage s08 runs on.
    _budgets = ([int(b.strip()) for b in args.budgets.split(',')]
                if getattr(args, 'budgets', None) else list(DISTANCE_BUDGETS))

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")
    print(f"Budgets: {_budgets}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset,
                  budgets=_budgets, n_sample=args.n_sample)
    save_experiment_results(results, args.output, 's08_optimal_dimension', args.dataset)
