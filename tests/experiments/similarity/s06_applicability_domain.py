#!/usr/bin/env python3
"""S06: Applicability Domain -- does hybrid distance track prediction error?

Paper: Similarity
Question: does distance-to-training in the hybrid space correlate with the
          model's prediction error (i.e. is it a usable applicability domain)?
Dependencies: s02 (reads JSON, optional -- falls back to fresh search).

Per-fold CV with per-fold hybrid construction:
  - kNN distances in the best single base representation, the hybrid, and the
    weighted hybrid
  - error = |residual| (regression) or 1 - P(true class) (classification;
    graded, NOT a degenerate 0/1 indicator)
  - Spearman correlation between distance and error, reported two ways that
    differ in how folds are combined:
      * POOLED -- `distance_error_correlation['k<K>'][<comparator rep> |
        'hybrid' | '<comparator rep>_weighted' | 'hybrid_weighted']`: the
        comparator representation's name is recorded at `results['comparator_rep']`
        (it was the fixed string 'concat' until 2026-09-04; a concatenation is
        not a baseline in this project). Distances and errors from
        every fold are concatenated and ONE Spearman is taken over the pooled
        arrays. Every fold refits its own StandardScaler, so fold-to-fold
        differences in distance scale enter this single rank ordering.
      * PER-FOLD -- `distance_error_correlation['k<K>']['per_fold_mean']`:
        one Spearman inside each fold, then the mean over folds. Ranks never
        cross a fold boundary.
    Both blocks are split the same way by which model produced the errors:
      * top level -- errors from the RF fitted on the per-fold HYBRID
        features, i.e. the same feature space the `hybrid` and
        `hybrid_weighted` distances are measured in (tautological pair).
      * `['neutral_model']` -- errors from an RF fitted on every representation
        glued together, which never sees the hybrid feature space
        (de-tautologised pair). That model is an error generator, not an arm;
        nothing is ranked against it.
  - Reliability diagram (distance decile -> mean error)
  - Conformal calibration comparison: comparator rep vs hybrid (PICP, MPIW, AND
    undercoverage + `hybrid_genuinely_tighter` so a tighter-but-undercovered
    interval cannot be misread as an improvement)

Empirically (2026-06-03) the regression distance-error correlation is near
zero and sign-inconsistent across reps -> for these representations the
applicability-domain signal does NOT hold. This is a genuine negative finding,
not a measurement bug (error is the absolute residual; per-fold standardization
is train-only and consistent). That entry was recorded against the pooled
correlation, which was the only one stored at the time; the per-fold mean
added on 2026-08-26 is a separate number and was not part of it.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from experiments._shared import make_nearest_neighbors  # honours --distance-metric
from sklearn.model_selection import GroupShuffleSplit
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, load_experiment_results, convert_numpy,
    get_fitted_hybrid, get_fitted_pool_hybrid, flatten_weights, flatten_fold_weights, make_cv_splitter,
    apply_feature_selection, compute_feature_importance,
    get_domain_metric, get_model_factories, build_fold_hybrid,
    get_best_for_downstream, extract_top3_base_reps,
    DISTANCE_BUDGETS, DISTANCE_EVALUATORS,
)
# Default kNN budget for AD hybrid search (kept modest; AD is a diagnostic, not
# a tuned downstream model).
_DEFAULT_AD_BUDGET = 100


def _graded_clf_error(clf, X, y_true):
    """Continuous classification error = 1 - P(true class).

    Robust to classes present in y_true but unseen in training (those get
    error 1.0). Far higher power than a 0/1 misclassification indicator for
    Spearman(distance, error) applicability-domain correlations.
    """
    proba = clf.predict_proba(X)
    col = {c: i for i, c in enumerate(clf.classes_)}
    return np.array([
        1.0 - (proba[i, col[c]] if c in col else 0.0)
        for i, c in enumerate(y_true)
    ], dtype=float)


def _fold_spearman(dists, errors):
    """Spearman(distance, error) inside a single fold.

    Returns NaN when the fold has fewer than 3 points or either array is
    constant -- spearmanr is undefined there, and a NaN is dropped by
    `_mean_over_folds` rather than propagating into the fold mean.
    """
    if len(dists) < 3:
        return float('nan')
    rho, _ = spearmanr(dists, errors)
    return float(rho)


def _mean_over_folds(rhos):
    """Mean of the finite per-fold rhos plus how many folds contributed.

    Returns (None, 0) when no fold produced a finite rho.
    """
    finite = [r for r in rhos if np.isfinite(r)]
    if not finite:
        return None, 0
    return float(np.mean(finite)), len(finite)


def _fmt_rho(value):
    """4-dp rho for the printed lines; 'n/a' when the fold mean is None."""
    return 'n/a' if value is None else f'{value:.4f}'


def run(data, config, output_dir, dataset_name, **kwargs):
    """Hybrid distance tracks prediction uncertainty better than standard distance."""
    print("\n" + "=" * 70)
    print("EXPERIMENT S06: APPLICABILITY DOMAIN")
    print("=" * 70)

    n_folds = config['n_splits']
    k_values = kwargs.get('k_values', [5])

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    task_type = data['task_type']
    domain = data.get('domain', 'molecular')
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    _rf_factory = get_model_factories(['rf'], task_type=task_type,
                                       n_samples=len(labels))['rf']

    results = {'distance_error_correlation': {}, 'reliability': {}, 'conformal': {}}

    augmentations = data.get('augmentations')
    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])

    # Get hybrid + blueprint for per-fold CV
    try:
        budget = kwargs.get('budget', _DEFAULT_AD_BUDGET)
        augmentations_search = data.get('augmentations')
        X_hybrid, fi, search, _meta = get_fitted_pool_hybrid(
            reps, labels, groups,
            downstream_model='knn', budget=budget,
            augmentations=augmentations_search,
            metric=metric_name, task_type=task_type, domain=domain,
            dataset_name=dataset_name, distance=True)
        best_config = get_best_for_downstream(search, 'knn')
        bp = best_config['blueprint'] if best_config and 'blueprint' in best_config else None
        # Pool helper overrides blueprint.augmentation_strategy to dodge the
        # 2026-04-21 cache-collision bug; mirror that override before per-fold
        # rebuilds so they match the pooled meta.
        if bp is not None and _meta and 'aug_condition' in _meta:
            bp.augmentation_strategy = _meta['aug_condition']
        top3 = extract_top3_base_reps(search)
        if top3:
            print(f"  Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")
            results['top3_base_reps'] = [(n, s) for n, s in top3]
    except Exception as e:
        print(f"  Hybrid search failed: {e}")
        return {'error': str(e)}

    # The arm the hybrid is compared against is ONE base representation, never
    # every representation glued together. Until 2026-09-04 'concat' was the
    # only comparator in the reliability deciles and in the conformal block, and
    # it was one of the four printed arms of the distance-error correlation. A
    # concatenation is not a baseline in this project.
    #
    # Ranked by knn, which is what this experiment searched on; rf is the
    # fallback because extract_top3_base_reps defaults to it and some searches
    # carry no knn baseline.
    _cmp_ranked = (extract_top3_base_reps(search, model='knn')
                   or extract_top3_base_reps(search, model='rf'))
    if not _cmp_ranked:
        return {'error': "s06 has no ranked base representation to compare the "
                         "hybrid against; the search carried no knn or rf "
                         "baselines."}
    cmp_rep = _cmp_ranked[0][0]
    results['comparator_rep'] = cmp_rep
    print(f"  Comparator: {cmp_rep} (best single rep by "
          f"{'knn' if extract_top3_base_reps(search, model='knn') else 'rf'})")

    splitter, use_groups = make_cv_splitter(n_folds, groups,
                                            task_type=task_type)
    sample_X = next(iter(reps.values()))

    for k in k_values:
        print(f"\n  k={k}:")
        all_d_cmp = []
        all_d_cmp_w = []
        all_d_hybrid = []
        all_d_weighted = []
        all_d_per_rep = {rname: [] for rname in reps}
        all_d_per_rep_w = {rname: [] for rname in reps}
        all_errors = []
        all_errors_neutral = []  # all-features RF (removes tautology)
        # F05: per-fold Spearman, one value per fold, kept separately from the
        # pooled lists above. Every fold refits its own StandardScaler, so a
        # correlation taken inside a fold never ranks one fold's distances
        # against another's.
        fold_rhos = {cmp_rep: [], f'{cmp_rep}_weighted': [],
                     'hybrid': [], 'hybrid_weighted': []}
        fold_rhos_neutral = {cmp_rep: [], 'hybrid': [], 'hybrid_weighted': []}

        for train_idx, val_idx in splitter.split(sample_X, labels, groups=groups) if use_groups else splitter.split(sample_X, labels):
            y_train, y_val = labels[train_idx], labels[val_idx]

            # Per-fold hybrid using search blueprint.
            # NOTE: use a distinct comprehension variable (aug_k) — NOT `k` —
            # so the outer `for k in k_values:` loop variable is never at risk of
            # being shadowed (Python-3 comprehensions are scoped, but a bare
            # `for k,...` would leak; keep this unambiguous so every downstream
            # min(k, len(y_train)-1) uses the integer neighbour count, never a key).
            train_reps_h = {name: rep[train_idx] for name, rep in reps.items()}
            val_reps_h = {name: rep[val_idx] for name, rep in reps.items()}
            aug_tr = {aug_k: v[train_idx] for aug_k, v in augmentations.items()} if augmentations else None
            aug_va = {aug_k: v[val_idx] for aug_k, v in augmentations.items()} if augmentations else None
            try:
                X_tr_h, X_va_h, feat_info = build_fold_hybrid(
                    train_reps_h, val_reps_h, y_train, bp,
                    augmentations_train=aug_tr, augmentations_val=aug_va,
                    groups=(groups[train_idx] if groups is not None else None),
                    model=_rf_factory(),
                    metric_fn=metric_fn,
                    lower_is_better=lower_is_better)

                # Importance weights for this fold (also weight the comparator)
                imp = compute_feature_importance(
                    train_reps_h, y_train, method='lightgbm_gain')
                # Use the shared helper: handles the augmentation tail so the
                # weight vector matches X_tr_h/X_va_h column count exactly.
                # NaN-guard: importance helpers can emit NaN for constant
                # features; np.maximum(NaN, 0) is NaN which would corrupt the
                # weighted kNN distance.
                w = np.nan_to_num(flatten_fold_weights(feat_info), nan=0.0)
                w = np.sqrt(np.maximum(w, 0))
            except Exception as exc:
                # N9: log the dropped fold so a thrown hybrid/importance build
                # is distinguishable from a genuinely skipped fold.
                print(f"      AD hybrid weight fold failed: "
                      f"{type(exc).__name__}: {exc}")
                continue

            # Train RF on hybrid features (existing — may have tautological AD advantage)
            scaler_h = StandardScaler()
            X_tr_s = scaler_h.fit_transform(X_tr_h)
            X_va_s = scaler_h.transform(X_va_h)
            rf = _rf_factory()
            rf.fit(X_tr_s, y_train)
            y_pred = rf.predict(X_va_s)
            # Classification: graded error = 1 - P(true class). A 0/1
            # misclassification indicator makes Spearman(dist, error) a
            # degenerate, low-power statistic; the predicted-probability of the
            # true label is the continuous AD analogue. Regression: |residual|.
            if task_type == 'classification':
                errors = _graded_clf_error(rf, X_va_s, y_val)
            else:
                errors = np.abs(y_val - y_pred)

            # Neutral error model: an RF fitted on every representation glued
            # together. This is NOT an arm and nothing is ranked against it — it
            # exists only to produce prediction errors that neither the hybrid
            # nor the comparator representation supplied the features for, so
            # neither one is being scored against its own model. Fitting it on a
            # single representation would just move the tautology onto that arm.
            scaler_c = StandardScaler()
            Xtr_c = scaler_c.fit_transform(X_concat[train_idx])
            Xva_c = scaler_c.transform(X_concat[val_idx])
            rf_neutral = _rf_factory()
            rf_neutral.fit(Xtr_c, y_train)
            y_pred_neutral = rf_neutral.predict(Xva_c)
            if task_type == 'classification':
                errors_neutral = _graded_clf_error(rf_neutral, Xva_c, y_val)
            else:
                errors_neutral = np.abs(y_val - y_pred_neutral)

            # Comparator arm: the best single base representation, scaled on
            # this fold's training rows. Its importance weights fall back to
            # ones when the fold produced none, which matches what the old
            # concat arm did for a representation the search never scored.
            scaler_cmp = StandardScaler()
            Xtr_cmp = scaler_cmp.fit_transform(reps[cmp_rep][train_idx])
            Xva_cmp = scaler_cmp.transform(reps[cmp_rep][val_idx])
            w_cmp = (imp[cmp_rep] if cmp_rep in imp
                     else np.ones(reps[cmp_rep].shape[1]))
            w_cmp = np.sqrt(np.maximum(np.nan_to_num(w_cmp, nan=0.0), 0))

            # Distance to k-NN in training set for each val molecule
            # Comparator representation (unweighted)
            nn_c = make_nearest_neighbors(n_neighbors=min(k, len(y_train) - 1))
            nn_c.fit(Xtr_cmp)
            dists_c, _ = nn_c.kneighbors(Xva_cmp)
            d_cmp = dists_c.mean(axis=1)

            # Comparator representation (weighted)
            nn_cw = make_nearest_neighbors(n_neighbors=min(k, len(y_train) - 1))
            nn_cw.fit(Xtr_cmp * w_cmp)
            dists_cw, _ = nn_cw.kneighbors(Xva_cmp * w_cmp)
            d_cmp_w = dists_cw.mean(axis=1)

            # Per-rep distances (unweighted + weighted)
            for rname, X_rep in reps.items():
                scaler_r = StandardScaler()
                Xtr_r = scaler_r.fit_transform(X_rep[train_idx])
                Xva_r = scaler_r.transform(X_rep[val_idx])
                nn_r = make_nearest_neighbors(n_neighbors=min(k, len(y_train) - 1))
                nn_r.fit(Xtr_r)
                dr, _ = nn_r.kneighbors(Xva_r)
                all_d_per_rep[rname].extend(dr.mean(axis=1).tolist())

                if rname in imp:
                    w_r = np.sqrt(np.maximum(imp[rname], 0))
                    nn_rw = make_nearest_neighbors(n_neighbors=min(k, len(y_train) - 1))
                    nn_rw.fit(Xtr_r * w_r)
                    drw, _ = nn_rw.kneighbors(Xva_r * w_r)
                    all_d_per_rep_w[rname].extend(drw.mean(axis=1).tolist())

            # Hybrid space (unweighted)
            nn_h = make_nearest_neighbors(n_neighbors=min(k, len(y_train) - 1))
            nn_h.fit(X_tr_s)
            dists_h, _ = nn_h.kneighbors(X_va_s)
            d_hybrid = dists_h.mean(axis=1)

            # Weighted hybrid space
            nn_w = make_nearest_neighbors(n_neighbors=min(k, len(y_train) - 1))
            nn_w.fit(X_tr_s * w)
            dists_w, _ = nn_w.kneighbors(X_va_s * w)
            d_weighted = dists_w.mean(axis=1)

            all_d_cmp.extend(d_cmp.tolist())
            all_d_cmp_w.extend(d_cmp_w.tolist())
            all_d_hybrid.extend(d_hybrid.tolist())
            all_d_weighted.extend(d_weighted.tolist())
            all_errors.extend(errors.tolist())
            all_errors_neutral.extend(errors_neutral.tolist())

            # Per-fold correlations, computed before the fold's distances are
            # pooled. Top block scores against `errors` (RF fitted on this
            # fold's hybrid features); neutral block scores the same distances
            # against `errors_neutral` (RF fitted on concat features).
            for _name, _d in ((cmp_rep, d_cmp),
                              (f'{cmp_rep}_weighted', d_cmp_w),
                              ('hybrid', d_hybrid),
                              ('hybrid_weighted', d_weighted)):
                fold_rhos[_name].append(_fold_spearman(_d, errors))
            for _name, _d in ((cmp_rep, d_cmp),
                              ('hybrid', d_hybrid),
                              ('hybrid_weighted', d_weighted)):
                fold_rhos_neutral[_name].append(
                    _fold_spearman(_d, errors_neutral))

        if not all_errors:
            continue

        all_d_cmp = np.array(all_d_cmp)
        all_d_cmp_w = np.array(all_d_cmp_w)
        all_d_hybrid = np.array(all_d_hybrid)
        all_d_weighted = np.array(all_d_weighted)
        all_errors = np.array(all_errors)

        # Spearman correlations: distance vs |error| (hybrid-model errors),
        # over the arrays pooled across all folds.
        rho_c, _ = spearmanr(all_d_cmp, all_errors)
        rho_cw, _ = spearmanr(all_d_cmp_w, all_errors)
        rho_h, _ = spearmanr(all_d_hybrid, all_errors)
        rho_w, _ = spearmanr(all_d_weighted, all_errors)

        corr_result = {
            # Unchanged keys: pooled-over-folds Spearman against the errors of
            # the RF fitted on the per-fold hybrid features.
            cmp_rep: float(rho_c),
            f'{cmp_rep}_weighted': float(rho_cw),
            'hybrid': float(rho_h),
            'hybrid_weighted': float(rho_w),
            'error_source': 'rf_fitted_on_hybrid_features',
            'error_source_note': (
                "the 'hybrid' and 'hybrid_weighted' distances here are measured "
                "in the same feature space this error model was fitted on "
                "(tautological pair); ['neutral_model'] holds the "
                "de-tautologised pair"),
            'fold_handling': 'pooled_across_folds',
            'fold_handling_note': (
                "distances and errors from every fold concatenated, one Spearman "
                "over the pooled arrays; each fold refit its own StandardScaler, "
                "so cross-fold scale differences enter this rank ordering; "
                "['per_fold_mean'] holds the within-fold version"),
        }

        # F05: per-fold means alongside the pooled values above.
        per_fold_mean = {
            'error_source': 'rf_fitted_on_hybrid_features',
            'fold_handling': 'spearman_within_fold_then_mean_over_folds',
        }
        for _name, _vals in fold_rhos.items():
            _m, _n = _mean_over_folds(_vals)
            per_fold_mean[_name] = _m
            per_fold_mean[f'{_name}_n_folds'] = _n
            per_fold_mean[f'{_name}_per_fold'] = [float(v) for v in _vals]
        corr_result['per_fold_mean'] = per_fold_mean

        # Neutral-model correlations (errors from RF on concat — removes tautology)
        all_errors_neutral = np.array(all_errors_neutral)
        if len(all_errors_neutral) == len(all_d_cmp):
            rho_c_n, _ = spearmanr(all_d_cmp, all_errors_neutral)
            rho_h_n, _ = spearmanr(all_d_hybrid, all_errors_neutral)
            rho_w_n, _ = spearmanr(all_d_weighted, all_errors_neutral)
            neutral_per_fold = {
                'error_source': 'rf_fitted_on_concat_features',
                'fold_handling': 'spearman_within_fold_then_mean_over_folds',
            }
            for _name, _vals in fold_rhos_neutral.items():
                _m, _n = _mean_over_folds(_vals)
                neutral_per_fold[_name] = _m
                neutral_per_fold[f'{_name}_n_folds'] = _n
                neutral_per_fold[f'{_name}_per_fold'] = [float(v) for v in _vals]
            corr_result['neutral_model'] = {
                cmp_rep: float(rho_c_n),
                'hybrid': float(rho_h_n),
                'hybrid_weighted': float(rho_w_n),
                'error_source': 'rf_fitted_on_concat_features',
                'error_source_note': (
                    'errors come from a model that never sees the hybrid feature '
                    'space, so these distances and these errors do not share a '
                    'feature space (de-tautologised pair)'),
                'fold_handling': 'pooled_across_folds',
                'per_fold_mean': neutral_per_fold,
            }
            print(f"    Neutral-model AD [errors from RF on concat features; "
                  f"de-tautologised pair], pooled over folds: "
                  f"{cmp_rep}={rho_c_n:.4f}, hybrid={rho_h_n:.4f}, "
                  f"hybrid_w={rho_w_n:.4f}")
            print(f"      same, per-fold mean ({neutral_per_fold['hybrid_n_folds']} "
                  f"folds): {cmp_rep}={_fmt_rho(neutral_per_fold[cmp_rep])}, "
                  f"hybrid={_fmt_rho(neutral_per_fold['hybrid'])}, "
                  f"hybrid_w={_fmt_rho(neutral_per_fold['hybrid_weighted'])}")

        # Per-rep correlations
        for rname in reps:
            if all_d_per_rep[rname]:
                rho_r, _ = spearmanr(np.array(all_d_per_rep[rname]), all_errors)
                corr_result[rname] = float(rho_r)
            if all_d_per_rep_w[rname]:
                rho_rw, _ = spearmanr(np.array(all_d_per_rep_w[rname]), all_errors)
                corr_result[f'{rname}_weighted'] = float(rho_rw)

        results['distance_error_correlation'][f'k{k}'] = corr_result
        print(f"    Same-space AD [errors from RF fitted on the hybrid features, "
              f"the space hybrid/hybrid_w distances are measured in; "
              f"tautological pair], pooled over folds: "
              f"{cmp_rep}={rho_c:.4f}, {cmp_rep}_w={rho_cw:.4f}, "
              f"hybrid={rho_h:.4f}, hybrid_w={rho_w:.4f}")
        print(f"      same, per-fold mean ({per_fold_mean['hybrid_n_folds']} folds): "
              f"{cmp_rep}={_fmt_rho(per_fold_mean[cmp_rep])}, "
              f"{cmp_rep}_w={_fmt_rho(per_fold_mean[f'{cmp_rep}_weighted'])}, "
              f"hybrid={_fmt_rho(per_fold_mean['hybrid'])}, "
              f"hybrid_w={_fmt_rho(per_fold_mean['hybrid_weighted'])}")
        print(f"      keys: {cmp_rep!r}/'hybrid'/... = pooled; "
              f"['per_fold_mean'] = within-fold mean; ['neutral_model'] = "
              f"errors from the all-features model")

        # Reliability diagram: bin by distance decile
        for dist_name, dists in [(cmp_rep, all_d_cmp),
                                  ('hybrid', all_d_hybrid),
                                  ('hybrid_weighted', all_d_weighted)]:
            deciles = np.percentile(dists, np.arange(0, 110, 10))
            decile_errors = []
            decile_counts = []
            for di in range(10):
                mask = (dists >= deciles[di]) & (dists < deciles[di + 1])
                if di == 9:  # include upper bound in last bin
                    mask = (dists >= deciles[di]) & (dists <= deciles[di + 1])
                if mask.sum() > 0:
                    decile_errors.append(float(np.mean(all_errors[mask])))
                    decile_counts.append(int(mask.sum()))
                else:
                    decile_errors.append(float('nan'))
                    decile_counts.append(0)

            results['reliability'].setdefault(f'k{k}', {})[dist_name] = {
                'decile_errors': decile_errors,
                'decile_counts': decile_counts,
            }

    # Conformal calibration comparison (regression only — residual-based intervals)
    if task_type == 'classification':
        print("\n  Conformal calibration: SKIP (regression-only, not applicable to classification)")
        results['conformal'] = {'skipped': True, 'reason': 'classification'}
    else:
        print("\n  Conformal calibration (CV):")
        try:
            splitter_conf, use_groups_conf = make_cv_splitter(
                n_folds, groups, task_type=task_type)
            conf_picp = {cmp_rep: [], 'hybrid': []}
            conf_mpiw = {cmp_rep: [], 'hybrid': []}
            alpha = 0.1

            for train_idx, val_idx in (splitter_conf.split(sample_X, labels, groups=groups) if use_groups_conf else splitter_conf.split(sample_X, labels)):
                y_train, y_val = labels[train_idx], labels[val_idx]

                # Per-fold hybrid using search blueprint (avoids data leakage)
                train_reps_fold = {k: v[train_idx] for k, v in reps.items()}
                val_reps_fold = {k: v[val_idx] for k, v in reps.items()}
                aug_tr_c = {k: v[train_idx] for k, v in augmentations.items()} if augmentations else None
                aug_va_c = {k: v[val_idx] for k, v in augmentations.items()} if augmentations else None
                try:
                    X_hybrid_tr, X_hybrid_va, fi = build_fold_hybrid(
                        train_reps_fold, val_reps_fold, y_train, bp,
                        augmentations_train=aug_tr_c, augmentations_val=aug_va_c,
                        groups=(groups[train_idx] if groups is not None else None),
                        model=_rf_factory(),
                        metric_fn=metric_fn,
                        lower_is_better=lower_is_better)
                except Exception as exc:
                    # N9: log the dropped fold so a thrown hybrid build is
                    # distinguishable from a genuinely skipped fold.
                    print(f"      AD calibration fold hybrid build failed: "
                          f"{type(exc).__name__}: {exc}")
                    continue

                # Split train into proper-train and calibration (group-aware)
                train_groups = groups[train_idx] if groups is not None else None
                if train_groups is not None and len(np.unique(train_groups)) >= 2:
                    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                    tr_local, cal_local = next(gss.split(
                        np.arange(len(y_train)), y_train, groups=train_groups))
                else:
                    n_cal = max(int(len(y_train) * 0.2), 10)
                    rng = np.random.RandomState(42)
                    perm = rng.permutation(len(y_train))
                    cal_local, tr_local = perm[:n_cal], perm[n_cal:]

                for feat_name, X_tr_all, X_va in [
                    (cmp_rep, reps[cmp_rep][train_idx], reps[cmp_rep][val_idx]),
                    ('hybrid', X_hybrid_tr, X_hybrid_va),
                ]:
                    scaler = StandardScaler()
                    X_tr_all_s = scaler.fit_transform(X_tr_all)
                    X_va_s = scaler.transform(X_va)

                    X_tr = X_tr_all_s[tr_local]
                    X_cal = X_tr_all_s[cal_local]
                    y_tr = y_train[tr_local]
                    y_cal = y_train[cal_local]

                    # Train RF, compute conformal quantile on calibration set
                    rf = _rf_factory()
                    rf.fit(X_tr, y_tr)

                    # Calibration: residuals normalized by tree variance
                    y_pred_cal = rf.predict(X_cal)
                    tree_preds_cal = np.array([t.predict(X_cal) for t in rf.estimators_])
                    cal_std = tree_preds_cal.std(axis=0) + 1e-10
                    conformity = np.abs(y_cal - y_pred_cal) / cal_std
                    q = np.quantile(conformity, 1 - alpha)

                    # Val: predict + interval
                    y_pred_va = rf.predict(X_va_s)
                    tree_preds_va = np.array([t.predict(X_va_s) for t in rf.estimators_])
                    va_std = tree_preds_va.std(axis=0)
                    interval_width = 2 * q * va_std

                    picp = float(np.mean(np.abs(y_val - y_pred_va) <= interval_width / 2))
                    mpiw = float(np.mean(interval_width))
                    conf_picp[feat_name].append(picp)
                    conf_mpiw[feat_name].append(mpiw)

            results['conformal'] = {'nominal_coverage': 1 - alpha}
            for feat_name in [cmp_rep, 'hybrid']:
                if conf_picp[feat_name]:
                    picp_mean = float(np.mean(conf_picp[feat_name]))
                    mpiw_mean = float(np.mean(conf_mpiw[feat_name]))
                    # undercoverage = how far PICP falls below the 90% target.
                    # A smaller MPIW (tighter interval) is ONLY an improvement
                    # if undercoverage stays ~0; tighter-but-undercovered is not.
                    undercov = float((1 - alpha) - picp_mean)
                    results['conformal'][feat_name] = {
                        'picp': picp_mean, 'mpiw': mpiw_mean,
                        'undercoverage': undercov,
                    }
                    print(f"    {feat_name.capitalize()}:  PICP={picp_mean:.3f}, "
                          f"MPIW={mpiw_mean:.3f}, undercoverage={undercov:+.3f}")
            # Honest comparison: hybrid's tighter interval only counts as better
            # if it does not lose coverage relative to the single-rep
            # comparator (tolerance 0.02).
            if cmp_rep in results['conformal'] and 'hybrid' in results['conformal']:
                c, h = results['conformal'][cmp_rep], results['conformal']['hybrid']
                results['conformal']['hybrid_genuinely_tighter'] = bool(
                    h['mpiw'] < c['mpiw'] and h['picp'] >= c['picp'] - 0.02)
        except Exception as e:
            print(f"    Conformal failed: {e}")
            results['conformal'] = {'error': str(e)}

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('S06: Applicability Domain',
                                     has_delta=True, has_augment=True)
    parser.add_argument('--k-values', type=int, nargs='+', default=[5],
                        help='k values for kNN distance (default: [5])')
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")
    print(f"k values: {args.k_values}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset,
                  k_values=args.k_values)
    save_experiment_results(results, args.output, 's06_applicability_domain', args.dataset)
