#!/usr/bin/env python3
"""Fed02: Representation Diversity as Federated Silos.

Tests whether KIRBy's feature-level hybrid outperforms FLuID's prediction-level
ensemble when silos contribute different molecular representations (not different
data). Each silo sees the same molecules but through a different computational
lens (e.g., ECFP4, Mordred, MolFormer, PDV, ChemBERTa).

6 conditions:
    1. Best single silo — best individual rep baseline
    2. FLuID similarity ensemble — 8-NN weighted mean of regressor predictions
    3. FLuID quality ensemble — inverse-RMSE weighted mean
    4. KIRBy hybrid — feature-level fusion across all 5 silo reps
    5. KIRBy top-tier hybrid — feature-level fusion on top 3 reps only
    6. FLuID→KIRBy pipeline — intra-silo clustering + cross-silo fusion (novel)

Datasets: herg (classification), logd, caco2 (regression).

Usage:
    python fed02_rep_diversity_silos.py --dataset herg --mode intermediary
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import traceback
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy, cv_split,
    get_precomputed_search, get_best_search_for_downstream, get_domain_metric,
)
from experiments.federated.federated_helpers import (
    compute_reg_metrics,
    compute_clf_metrics,
    predict_fluid_similarity_ensemble_regression,
    predict_fluid_quality_ensemble_regression,
    predict_fluid_similarity_ensemble_classification,
    predict_fluid_quality_ensemble_classification,
    fluid_within_kirby_across,
    smoke_n_folds,
    fit_fed_model,
    fed_to_kirby_model,
    make_fed_classifier,
    make_fed_regressor,
)


# ── Constants ────────────────────────────────────────────────────────────────

# 5 silo representations
SILO_REPS = {
    'A': 'ecfp4',        # structural fingerprints, 2048D
    'B': 'mordred',       # physicochemical descriptors, ~1600D
    'C': 'molformer',     # foundation model embeddings, 768D
    'D': 'pdv',           # RDKit descriptors, 200D
    'E': 'chemberta',     # learned SMILES embeddings, 768D
}

# Fallback for datasets where chemberta is unavailable
SILO_REPS_FALLBACK = {
    'caco2': {**SILO_REPS, 'E': 'avalon'},
}

K_INTRA = 3   # sub-clusters per silo for Condition 6
BUDGET = 100  # feature budget for KIRBy hybrid


def _get_silo_reps(dataset_name, available_reps):
    """Resolve silo reps for a dataset, using fallbacks if needed."""
    import warnings
    base = SILO_REPS_FALLBACK.get(dataset_name, SILO_REPS)
    resolved = {}
    for silo_letter, rep_name in base.items():
        if rep_name in available_reps:
            resolved[silo_letter] = rep_name
        else:
            # Try common fallbacks
            for fallback in ['chembert', 'avalon', 'graph_kernel']:
                if fallback in available_reps and fallback not in resolved.values():
                    resolved[silo_letter] = fallback
                    warnings.warn(
                        f"Silo {silo_letter}: '{rep_name}' unavailable, "
                        f"using fallback '{fallback}'")
                    break
            else:
                warnings.warn(
                    f"Silo {silo_letter}: '{rep_name}' unavailable and "
                    f"no fallback found — silo dropped")
    return resolved


def run(data, config, output_dir, dataset_name, **kwargs):
    """Fed02: Representation diversity as federated silos."""
    print("\n" + "=" * 70)
    print("EXPERIMENT FED02: REPRESENTATION DIVERSITY AS FEDERATED SILOS")
    print("=" * 70)

    fed_model = config.get('fed_model', 'rf')
    fed_seed = config.get('fed_seed', 0)

    reps = data['reps']
    labels = data['labels']
    groups = data.get('groups')
    task_type = data['task_type']
    is_clf = task_type == 'classification'

    # ── Get optimal hybrid config from precomputed search ──
    augs = data.get('augmentations')
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    try:
        _search, meta = get_best_search_for_downstream(
            reps, labels, groups,
            downstream_model=fed_to_kirby_model(fed_model), budget=BUDGET,
            augmentations=augs,
            metric=metric_name,
            task_type=data['task_type'],
            domain=data.get('domain', 'molecular'),
            dataset_name=dataset_name,
            data=data,
        )
        bp = meta['best_config']['blueprint'] if meta else None
    except Exception:
        meta = None
        bp = None
    # Pull from meta['aug_condition'] (set at _shared.py:2747) — the
    # authoritative winning condition. bp.augmentation_strategy is a
    # coincidence-of-construction and can be stale (2026-04-21 cache bug).
    # Same for alloc/imp/int_eval: prefer meta['best_config']['blueprint']
    # when present, fall back to bp attributes, then hard defaults.
    if meta is not None:
        aug_cond = meta.get('aug_condition',
                            bp.augmentation_strategy if bp else
                            ('uncapped' if augs else 'none'))
        _bp_meta = meta.get('best_config', {}).get('blueprint') if meta else None
        _alloc = _bp_meta.allocation_method if _bp_meta else (
            bp.allocation_method if bp else 'performance_weighted')
        _imp = _bp_meta.importance_method if _bp_meta else (
            bp.importance_method if bp else 'treeshap')
        _int_eval = _bp_meta.internal_evaluator if _bp_meta else (
            bp.internal_evaluator if bp else 'match')
    else:
        aug_cond = bp.augmentation_strategy if bp else ('uncapped' if augs else 'none')
        _alloc = bp.allocation_method if bp else 'performance_weighted'
        _imp = bp.importance_method if bp else 'treeshap'
        _int_eval = bp.internal_evaluator if bp else 'match'

    # ── Phase 1: Setup ───────────────────────────────────────────────────
    print("\n--- Phase 1: Setup ---")

    # Resolve silo reps
    silo_reps = _get_silo_reps(dataset_name, set(reps.keys()))
    silo_letters = sorted(silo_reps.keys())
    K = len(silo_reps)

    if K < 3:
        raise ValueError(
            f"Need at least 3 silo reps, got {K}. "
            f"Available: {sorted(reps.keys())}")

    print(f"  {dataset_name}: {len(labels)} samples, "
          f"{K} silos assigned:")
    for s in silo_letters:
        rep_name = silo_reps[s]
        dim = reps[rep_name].shape[1]
        print(f"    Silo {s}: {rep_name} ({dim}D)")

    # P0 fix: 5-fold scaffold-group CV (was single 80/20 → all 6 conditions
    # evaluated on one fold). Hard-fail when groups unavailable (per
    # feedback_no_random_splits.md — no random splits anywhere).
    n_samples = len(labels)
    N_FOLDS = smoke_n_folds(5)   # 2 under KIRBY_SMOKE_N, else 5
    if groups is None or len(np.unique(groups)) < N_FOLDS:
        n_groups = 0 if groups is None else len(np.unique(groups))
        raise ValueError(
            f"fed02 requires scaffold groups for {N_FOLDS}-fold CV. "
            f"Got n_groups={n_groups} (need >= {N_FOLDS}). "
            f"Per project rule (no random splits), refusing to fall back "
            f"to KFold.")

    fold_splits = list(cv_split(
        groups, N_FOLDS, np.zeros(n_samples), labels, seed=fed_seed))
    print(f"  Using {N_FOLDS}-fold GroupKFold CV")

    def _make_model():
        # Unfitted template of the swept FLuID model family (per-run fed_model
        # + fed_seed). Used only as create_hybrid's model= for 'match' → model
        # family resolution; all *fitted* models go through fit_fed_model.
        if is_clf:
            return make_fed_classifier(fed_model, seed=fed_seed)
        return make_fed_regressor(fed_model, seed=fed_seed)

    _compute_metrics = compute_clf_metrics if is_clf else compute_reg_metrics

    # Per-fold accumulators
    fold_single_silo = []   # list of {silo: metrics_dict} per fold
    fold_c2 = []
    fold_c3 = []
    fold_c4 = []
    fold_c5 = []
    fold_c6 = []
    fold_best_silo = []     # best silo letter per fold
    fold_top_3_silos = []   # top-3 silo letters per fold
    fold_top_3_reps = []
    fold_n_train = []
    fold_n_test = []

    from kirby.hybrid import (
        create_hybrid, apply_feature_selection, apply_augmentation_selection)

    for fold_i, (train_idx, test_idx) in enumerate(fold_splits):
        print(f"\n{'=' * 70}\nFOLD {fold_i + 1}/{N_FOLDS}\n{'=' * 70}")

        y_train = labels[train_idx]
        y_test = labels[test_idx]
        print(f"  Split: {len(train_idx)} train / {len(test_idx)} test")
        fold_n_train.append(len(train_idx))
        fold_n_test.append(len(test_idx))

        # Pre-split reps (per fold)
        train_reps = {r: arr[train_idx] for r, arr in reps.items()}
        test_reps = {r: arr[test_idx] for r, arr in reps.items()}

        # P1 fix: compute train_augs/test_augs ONCE per fold (was computed
        # three times — phase 4 base + C4 + C5).
        train_augs = ({k: v[train_idx] for k, v in augs.items()}
                      if augs else None)
        test_augs = ({k: v[test_idx] for k, v in augs.items()}
                     if augs else None)

        def _append_test_augs(X_test, feature_info, _test_augs=test_augs):
            if not _test_augs:
                return X_test
            aug_info = feature_info.get('augmentation_info') if isinstance(feature_info, dict) else None
            if not aug_info:
                return X_test
            X_aug = apply_augmentation_selection(_test_augs, aug_info)
            return np.hstack([X_test, X_aug]) if X_aug is not None else X_test

        # P0/P1 fix: 'match' → 'rf' resolution. create_hybrid resolves
        # 'match' via type(model).__name__ — passing an instantiated RF
        # ensures 'rf' (not 'ridge' silent fallback).
        _match_model = _make_model()

        # ── Phase 2: Single-silo baselines (C1) ──────────────────────
        print(f"\n--- Phase 2: Single-silo baselines (C1) [task={task_type}] ---")
        single_silo = {}

        for s in silo_letters:
            rep_name = silo_reps[s]
            X_tr = train_reps[rep_name]
            X_te = test_reps[rep_name]

            mdl = fit_fed_model(fed_model, X_tr, y_train, seed=fed_seed,
                                task_type=task_type)
            y_pred = mdl.predict(X_te)

            if is_clf:
                y_prob = mdl.predict_proba(X_te)[:, 1] if hasattr(mdl, 'predict_proba') else None
                metrics = _compute_metrics(y_test, y_pred, y_prob=y_prob)
            else:
                metrics = _compute_metrics(y_test, y_pred)
            single_silo[s] = {
                'rep_name': rep_name,
                'n_features': int(X_tr.shape[1]),
                **metrics,
            }
            if is_clf:
                print(f"  Silo {s} ({rep_name}): "
                      f"MCC={metrics['mcc']:.4f}  "
                      f"BAC={metrics['bac']:.4f}  "
                      f"AUROC={metrics.get('auroc', float('nan')):.4f}")
            else:
                print(f"  Silo {s} ({rep_name}): "
                      f"RMSE={metrics['rmse']:.4f}  "
                      f"rho={metrics['spearman_rho']:.4f}  "
                      f"R2={metrics['r2']:.4f}")

        # Best silo (per fold)
        if is_clf:
            best_silo_f = max(single_silo, key=lambda s: single_silo[s]['mcc'])
        else:
            best_silo_f = min(single_silo, key=lambda s: single_silo[s]['rmse'])
        print(f"  Best: Silo {best_silo_f} ({silo_reps[best_silo_f]})")
        fold_single_silo.append(single_silo)
        fold_best_silo.append(best_silo_f)

        # ── Phase 3: FLuID ensemble (C2, C3) ─────────────────────────
        print("\n--- Phase 3: FLuID ensemble conditions ---")
        teacher_models = [None]
        silo_train_features_sim = [None]
        silo_test_features_sim = [None]
        silo_train_features_qual = [None]
        silo_test_features_qual = [None]

        for s in silo_letters:
            rep_name = silo_reps[s]
            X_tr = train_reps[rep_name]
            X_te = test_reps[rep_name]

            mdl = fit_fed_model(fed_model, X_tr, y_train, seed=fed_seed,
                                task_type=task_type)

            teacher_models.append((mdl, len(X_tr), X_tr))
            silo_train_features_sim.append(X_tr)
            silo_test_features_sim.append(X_te)
            silo_train_features_qual.append((X_tr, y_train))
            silo_test_features_qual.append(X_te)

        print("  C2: FLuID similarity...", end=' ')
        try:
            if is_clf:
                y_pred_c2, y_prob_c2, tw_c2 = predict_fluid_similarity_ensemble_classification(
                    teacher_models, silo_train_features_sim,
                    silo_test_features_sim, K)
                c2 = _compute_metrics(y_test, y_pred_c2, y_prob=y_prob_c2)
                print(f"MCC={c2['mcc']:.4f}")
            else:
                y_pred_c2, tw_c2 = predict_fluid_similarity_ensemble_regression(
                    teacher_models, silo_train_features_sim,
                    silo_test_features_sim, K)
                c2 = _compute_metrics(y_test, y_pred_c2)
                print(f"RMSE={c2['rmse']:.4f}")
            c2['teacher_weights_mean'] = convert_numpy(
                {silo_letters[i]: float(tw_c2[:, i].mean())
                 for i in range(K)})
        except Exception as e:
            c2 = {'error': str(e)}
            print(f"FAILED: {e}")
        fold_c2.append(c2)

        print("  C3: FLuID quality...", end=' ')
        try:
            if is_clf:
                y_pred_c3, y_prob_c3, mcc_w_c3 = predict_fluid_quality_ensemble_classification(
                    teacher_models, silo_train_features_qual,
                    silo_test_features_qual, K)
                c3 = _compute_metrics(y_test, y_pred_c3, y_prob=y_prob_c3)
                c3['mcc_weights'] = convert_numpy(mcc_w_c3)
                print(f"MCC={c3['mcc']:.4f}")
            else:
                y_pred_c3, rmse_w_c3 = predict_fluid_quality_ensemble_regression(
                    teacher_models, silo_train_features_qual,
                    silo_test_features_qual, K)
                c3 = _compute_metrics(y_test, y_pred_c3)
                c3['rmse_weights'] = convert_numpy(rmse_w_c3)
                print(f"RMSE={c3['rmse']:.4f}")
        except Exception as e:
            c3 = {'error': str(e)}
            print(f"FAILED: {e}")
        fold_c3.append(c3)

        # ── Phase 4: KIRBy hybrid (C4, C5) ───────────────────────────
        print("\n--- Phase 4: KIRBy hybrid conditions ---")

        # Build silo-only rep dicts
        kirby_train = {silo_reps[s]: train_reps[silo_reps[s]]
                       for s in silo_letters}
        kirby_test = {silo_reps[s]: test_reps[silo_reps[s]]
                      for s in silo_letters}

        # Condition 4: KIRBy hybrid (all K reps)
        print("  C4: KIRBy hybrid (all)...", end=' ')
        try:
            X_h_train, fi_c4 = create_hybrid(
                kirby_train, y_train,
                groups=(groups[train_idx] if groups is not None else None),
                model=_match_model,  # P0/P1: enables 'match' → 'rf' resolution
                allocation_method=_alloc,
                importance_method=_imp,
                budget=BUDGET,
                internal_evaluator=_int_eval,
                augmentations=train_augs,
                augmentation_strategy=aug_cond)
            X_h_test = _append_test_augs(
                apply_feature_selection(kirby_test, fi_c4), fi_c4)

            mdl_c4 = fit_fed_model(fed_model, X_h_train, y_train, seed=fed_seed,
                                   task_type=task_type)
            y_pred_c4 = mdl_c4.predict(X_h_test)

            if is_clf:
                y_prob_c4 = mdl_c4.predict_proba(X_h_test)[:, 1] if hasattr(mdl_c4, 'predict_proba') else None
                c4 = _compute_metrics(y_test, y_pred_c4, y_prob=y_prob_c4)
                print(f"MCC={c4['mcc']:.4f}  ({X_h_train.shape[1]}D)")
            else:
                c4 = _compute_metrics(y_test, y_pred_c4)
                print(f"RMSE={c4['rmse']:.4f}  ({X_h_train.shape[1]}D)")
            c4['n_features'] = int(X_h_train.shape[1])
            c4['allocation'] = convert_numpy(fi_c4.get('allocation', {}))
        except Exception as e:
            # Record traceback (not just str(e)) so a shape/import/NameError in
            # the create_hybrid path surfaces during the expensive run instead
            # of being reduced to a one-line message (FED-21).
            c4 = {'error': str(e), 'traceback': traceback.format_exc()}
            print(f"FAILED: {e}")
            traceback.print_exc()
        fold_c4.append(c4)

        # Condition 5: KIRBy top-tier hybrid (top 3 reps by best silo metric)
        print("  C5: KIRBy top-tier (3)...", end=' ')
        try:
            if is_clf:
                silo_ranked = sorted(single_silo.items(),
                                     key=lambda x: x[1]['mcc'], reverse=True)
            else:
                silo_ranked = sorted(single_silo.items(),
                                     key=lambda x: x[1]['rmse'])
            top_3_silos = [s for s, _ in silo_ranked[:3]]
            top_3_names = [silo_reps[s] for s in top_3_silos]

            kirby_top3_train = {silo_reps[s]: train_reps[silo_reps[s]]
                                for s in top_3_silos}
            kirby_top3_test = {silo_reps[s]: test_reps[silo_reps[s]]
                               for s in top_3_silos}

            X_t3_train, fi_c5 = create_hybrid(
                kirby_top3_train, y_train,
                groups=(groups[train_idx] if groups is not None else None),
                model=_match_model,  # P0/P1: 'match' → 'rf' resolution
                allocation_method=_alloc,
                importance_method=_imp,
                budget=BUDGET,
                internal_evaluator=_int_eval,
                augmentations=train_augs,
                augmentation_strategy=aug_cond)
            X_t3_test = _append_test_augs(
                apply_feature_selection(kirby_top3_test, fi_c5), fi_c5)

            mdl_c5 = fit_fed_model(fed_model, X_t3_train, y_train, seed=fed_seed,
                                   task_type=task_type)
            y_pred_c5 = mdl_c5.predict(X_t3_test)

            if is_clf:
                y_prob_c5 = mdl_c5.predict_proba(X_t3_test)[:, 1] if hasattr(mdl_c5, 'predict_proba') else None
                c5 = _compute_metrics(y_test, y_pred_c5, y_prob=y_prob_c5)
                print(f"MCC={c5['mcc']:.4f}  ({X_t3_train.shape[1]}D, "
                      f"reps={top_3_names})")
            else:
                c5 = _compute_metrics(y_test, y_pred_c5)
                print(f"RMSE={c5['rmse']:.4f}  ({X_t3_train.shape[1]}D, "
                      f"reps={top_3_names})")
            c5['n_features'] = int(X_t3_train.shape[1])
            c5['top_3_silos'] = top_3_silos
            c5['top_3_reps'] = top_3_names
            c5['allocation'] = convert_numpy(fi_c5.get('allocation', {}))
            fold_top_3_silos.append(top_3_silos)
            fold_top_3_reps.append(top_3_names)
        except Exception as e:
            c5 = {'error': str(e), 'traceback': traceback.format_exc()}
            print(f"FAILED: {e}")
            traceback.print_exc()
            fold_top_3_silos.append(None)
            fold_top_3_reps.append(None)
        fold_c5.append(c5)

        # ── Phase 5: FLuID→KIRBy pipeline (C6) ───────────────────────
        print("\n--- Phase 5: FLuID-within-KIRBy-across pipeline (C6) ---")
        print("  C6: FLuID→KIRBy...", end=' ')
        try:
            train_groups_c6 = (groups[train_idx]
                               if groups is not None else None)
            X_c6_train, X_c6_test, budgets_c6, scores_c6, meta_c6 = \
                fluid_within_kirby_across(
                    kirby_train, kirby_test, y_train,
                    k_intra=K_INTRA,
                    allocation_method=_alloc, importance_method=_imp,
                    internal_evaluator=_int_eval,
                    downstream_model=fed_to_kirby_model(fed_model),  # 'match' resolution
                    groups=train_groups_c6,
                    seed=fed_seed,
                    task_type=task_type)

            mdl_c6 = fit_fed_model(fed_model, X_c6_train, y_train, seed=fed_seed,
                                   task_type=task_type)
            y_pred_c6 = mdl_c6.predict(X_c6_test)

            if is_clf:
                y_prob_c6 = mdl_c6.predict_proba(X_c6_test)[:, 1] if hasattr(mdl_c6, 'predict_proba') else None
                c6 = _compute_metrics(y_test, y_pred_c6, y_prob=y_prob_c6)
                _c6_label = f"MCC={c6['mcc']:.4f}"
            else:
                c6 = _compute_metrics(y_test, y_pred_c6)
                _c6_label = f"RMSE={c6['rmse']:.4f}"
            c6['n_features'] = int(X_c6_train.shape[1])
            c6['silo_budgets'] = convert_numpy(budgets_c6)
            c6['silo_scores'] = convert_numpy(scores_c6)
            c6['n_virtual_silos'] = len(meta_c6)
            c6['virtual_silo_metadata'] = convert_numpy(meta_c6)
            print(f"{_c6_label}  "
                  f"({c6['n_features']}D, {c6['n_virtual_silos']} virtual silos)")
        except Exception as e:
            c6 = {'error': str(e), 'traceback': traceback.format_exc()}
            print(f"FAILED: {e}")
            traceback.print_exc()
        fold_c6.append(c6)

    # ── Phase 6: Aggregate per-fold metrics → mean ± std ─────────────────
    print(f"\n{'=' * 70}\nAGGREGATING ACROSS {N_FOLDS} FOLDS\n{'=' * 70}")

    _scalar_keys_reg = ['rmse', 'spearman_rho', 'spearman_p',
                        'pearson_r', 'pearson_p', 'r2']
    _scalar_keys_clf = ['mcc', 'bac', 'f1', 'auroc', 'accuracy', 'f1_weighted']
    _scalar_keys = _scalar_keys_clf if is_clf else _scalar_keys_reg

    def _aggregate_folds(fold_list):
        """Mean±std of scalar metrics across folds; preserve last-fold extras."""
        out = {}
        # Collect scalars
        for k in _scalar_keys:
            vals = [f.get(k) for f in fold_list
                    if isinstance(f, dict) and isinstance(f.get(k), (int, float))
                    and not (isinstance(f.get(k), float) and np.isnan(f.get(k)))]
            if vals:
                out[k] = float(np.mean(vals))
                out[f'{k}_std'] = float(np.std(vals))
                out[f'{k}_per_fold'] = [float(v) for v in vals]
            else:
                out[k] = float('nan')
                out[f'{k}_std'] = float('nan')
                out[f'{k}_per_fold'] = []
        # Preserve non-scalar extras from the last successful fold (e.g.
        # allocation, n_features, teacher_weights_mean, silo_budgets).
        for f in reversed(fold_list):
            if isinstance(f, dict) and 'error' not in f:
                for k, v in f.items():
                    if k in _scalar_keys:
                        continue
                    if k.endswith('_std') or k.endswith('_per_fold'):
                        continue
                    out.setdefault(k, v)
                break
        # Record any per-fold errors
        errors = [f.get('error') for f in fold_list
                  if isinstance(f, dict) and 'error' in f]
        if errors:
            out['errors_per_fold'] = errors
        return out

    c2 = _aggregate_folds(fold_c2)
    c3 = _aggregate_folds(fold_c3)
    c4 = _aggregate_folds(fold_c4)
    c5 = _aggregate_folds(fold_c5)
    c6 = _aggregate_folds(fold_c6)

    # Aggregate single-silo per-letter, then pick overall best by mean primary
    _primary_key = 'mcc' if is_clf else 'rmse'
    silo_agg = {}
    for s in silo_letters:
        vals_by_metric = {k: [] for k in _scalar_keys}
        rep_name = silo_reps[s]
        n_feat = None
        for fold_ss in fold_single_silo:
            entry = fold_ss.get(s)
            if entry is None:
                continue
            for k in _scalar_keys:
                v = entry.get(k)
                if isinstance(v, (int, float)) and not (
                        isinstance(v, float) and np.isnan(v)):
                    vals_by_metric[k].append(float(v))
            if n_feat is None:
                n_feat = entry.get('n_features')
        agg = {'rep_name': rep_name, 'n_features': n_feat}
        for k, vals in vals_by_metric.items():
            if vals:
                agg[k] = float(np.mean(vals))
                agg[f'{k}_std'] = float(np.std(vals))
            else:
                agg[k] = float('nan')
                agg[f'{k}_std'] = float('nan')
        silo_agg[s] = agg

    if is_clf:
        best_silo = max(silo_agg, key=lambda s: silo_agg[s].get('mcc', -np.inf))
    else:
        best_silo = min(silo_agg, key=lambda s: silo_agg[s].get('rmse', np.inf))
    print(f"  Overall best silo (mean across folds): "
          f"{best_silo} ({silo_reps[best_silo]})")

    # ── Phase 6: Assemble results ─────────────────────────────────────────
    results = {
        'experiment': 'fed02_rep_diversity_silos',
        'dataset': dataset_name,
        'task_type': task_type,
        'config': {
            'silo_reps': {s: silo_reps[s] for s in silo_letters},
            'n_silos': K,
            'n_samples': len(labels),
            'n_folds': N_FOLDS,
            'n_train_per_fold': fold_n_train,
            'n_test_per_fold': fold_n_test,
            'k_intra_c6': K_INTRA,
            'budget': BUDGET,
            'best_silo_per_fold': fold_best_silo,
            'top_3_silos_per_fold': fold_top_3_silos,
        },
        'c1_best_single_silo': convert_numpy({
            'best_silo': best_silo,
            'best_rep': silo_reps[best_silo],
            f'best_{_primary_key}': silo_agg[best_silo][_primary_key],
            f'best_{_primary_key}_std': silo_agg[best_silo].get(
                f'{_primary_key}_std', float('nan')),
            'per_silo': silo_agg,
        }),
        'c2_fluid_similarity': convert_numpy(c2),
        'c3_fluid_quality': convert_numpy(c3),
        'c4_kirby_hybrid': convert_numpy(c4),
        'c5_kirby_top_tier': convert_numpy(c5),
        'c6_fluid_within_kirby_across': convert_numpy(c6),
        'privacy_notes': {
            'c1': 'each silo trains independently, no sharing',
            'c2': 'silos share predictions only',
            'c3': f'silos share predictions + one {"MCC" if is_clf else "RMSE"} scalar',
            'c4': 'centralized: all reps must be co-located',
            'c5': 'centralized: all reps must be co-located',
            'c6': 'intra-silo clustering local, cross-silo via federated FS',
        },
    }

    # Summary table
    print(f"\n{'─' * 70}")
    summary_items = [
        ('c1_best_single_silo', f'C1: Best silo ({silo_reps[best_silo]})'),
        ('c2_fluid_similarity', 'C2: FLuID similarity'),
        ('c3_fluid_quality', 'C3: FLuID quality'),
        ('c4_kirby_hybrid', 'C4: KIRBy hybrid (all)'),
        ('c5_kirby_top_tier', 'C5: KIRBy top-tier (3)'),
        ('c6_fluid_within_kirby_across', 'C6: FLuID→KIRBy'),
    ]

    if is_clf:
        print(f"  {'Condition':<35} {'MCC':>8} {'BAC':>8} {'AUROC':>8}")
        print(f"  {'─' * 61}")
        for key, name in summary_items:
            r = results[key]
            if key == 'c1_best_single_silo':
                r = r.get('per_silo', {}).get(best_silo, {})
            mcc = r.get('mcc', float('nan'))
            bac = r.get('bac', float('nan'))
            auroc = r.get('auroc', float('nan'))
            if np.isnan(mcc):
                print(f"  {name:<35} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
            else:
                print(f"  {name:<35} {mcc:>8.4f} {bac:>8.4f} {auroc:>8.4f}")
    else:
        print(f"  {'Condition':<35} {'RMSE':>8} {'Spearman':>10} {'R2':>8}")
        print(f"  {'─' * 61}")
        for key, name in summary_items:
            r = results[key]
            if key == 'c1_best_single_silo':
                r = r.get('per_silo', {}).get(best_silo, {})
            rmse = r.get('rmse', float('nan'))
            rho = r.get('spearman_rho', float('nan'))
            r2 = r.get('r2', float('nan'))
            if np.isnan(rmse):
                print(f"  {name:<35} {'N/A':>8} {'N/A':>10} {'N/A':>8}")
            else:
                print(f"  {name:<35} {rmse:>8.4f} {rho:>10.4f} {r2:>8.4f}")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('Fed02: Rep Diversity Silos')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(
        results, args.output, 'fed02_rep_diversity_silos', args.dataset)
