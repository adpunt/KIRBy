#!/usr/bin/env python3
"""Fed04: Cluster-Aware Importance.

Does the optimal KIRBy allocation vary across chemical space regions?
Uses FLuID's K-Means clustering to partition molecules, then runs KIRBy
importance independently per cluster. Compares whether structural FPs
dominate near activity cliffs while foundation model embeddings dominate
in smooth SAR regions.

Extends C07 (data complexity partitions) with FLuID's clustering.
Adds smooth_factor sweep to test cluster boundary regularization.

Usage:
    python fed04_cluster_aware_importance.py --dataset herg
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import traceback
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy, cv_split,
    get_precomputed_search, get_best_search_for_downstream, get_domain_metric,
)
from experiments.federated.federated_helpers import (
    smiles_to_fluid_df, cluster_training_only, compute_clf_metrics,
    extract_importance_profile, compare_importance_profiles,
    _resolve_match, fit_fed_classifier, fed_to_kirby_model,
    smoke_grid, smoke_n_folds, smoke_k_silos,
)
from kirby.hybrid import create_hybrid, apply_feature_selection


# ── Constants ────────────────────────────────────────────────────────────────
# smoke_* helpers are no-ops unless KIRBY_SMOKE_N is set.

K = smoke_k_silos(5)
PKI_THRESHOLD = 6.0
SMOOTH_FACTORS = smoke_grid([0, 0.01, 0.05, 0.1, 0.2], [0, 0.05])
BUDGET = 100
N_FOLDS = smoke_n_folds(5)


def _compute_cluster_importance(train_reps, train_labels, silo_indices, k,
                                budget=BUDGET,
                                alloc='performance_weighted',
                                imp='treeshap',
                                int_eval='match',
                                augs=None, aug_cond='none', train_groups=None):
    """Run create_hybrid per cluster and extract importance profiles.

    Returns:
        (profiles, feature_infos, cluster_stats)
        - profiles: {cluster_id: {rep_name: importance_score}}
        - feature_infos: {cluster_id: feature_info_dict}
        - cluster_stats: {cluster_id: {n_samples, n_active, top_rep}}
    """
    # Defensive: resolve 'match' to a concrete evaluator ('rf') so the
    # no-model create_hybrid calls below never fall through to 'ridge'
    # (hybrid.py:3354-3355, forbidden by feedback_no_ridge.md). run() already
    # passes a resolved int_eval; this guards future callers.
    int_eval = _resolve_match(int_eval, 'rf')

    profiles = {}
    feature_infos = {}
    cluster_stats = {}

    for s in range(1, k + 1):
        idx = silo_indices[s]
        if len(idx) < 20:
            cluster_stats[s] = {
                'n_samples': len(idx),
                'error': 'too few samples',
            }
            continue

        cluster_reps = {r: arr[idx] for r, arr in train_reps.items()}
        cluster_labels = train_labels[idx]
        cluster_augs = ({k: v[idx] for k, v in augs.items()}
                        if augs else None)

        try:
            _, fi = create_hybrid(
                cluster_reps, cluster_labels,
                groups=(train_groups[idx] if train_groups is not None else None),
                allocation_method=alloc,
                importance_method=imp,
                budget=budget,
                internal_evaluator=int_eval,
                augmentations=cluster_augs,
                augmentation_strategy=aug_cond,
            )
            feature_infos[s] = fi
            profile = extract_importance_profile(fi)
            profiles[s] = profile

            top_rep = max(profile, key=profile.get) if profile else 'none'
            n_active = int(cluster_labels.sum())
            cluster_stats[s] = {
                'n_samples': len(idx),
                'n_active': n_active,
                'n_inactive': len(idx) - n_active,
                'top_rep': top_rep,
                'n_reps_selected': len(profile),
            }
        except Exception as e:
            # Surface the traceback so a real bug (shape/import/NameError) in
            # the per-cluster create_hybrid path is debuggable from the
            # expensive run's logs rather than reduced to a string (FED-21).
            cluster_stats[s] = {
                'n_samples': len(idx),
                'error': str(e),
                'traceback': traceback.format_exc(),
            }
            traceback.print_exc()

    return profiles, feature_infos, cluster_stats


def run(data, config, output_dir, dataset_name, **kwargs):
    """Fed04: Cluster-aware importance."""
    print("\n" + "=" * 70)
    print("EXPERIMENT FED04: CLUSTER-AWARE IMPORTANCE")
    print("=" * 70)

    fed_model = config.get('fed_model', 'rf')
    fed_seed = config.get('fed_seed', 0)

    reps = data['reps']
    labels = data['labels']
    smiles = data['smiles']

    # ── Phase 1: Setup ───────────────────────────────────────────────────
    print("\n--- Phase 1: Setup ---")

    # Binarize labels (skip if already binary). The Lhasa loader sets
    # data['is_binary'], but the regular load_data('herg', ...) path returns
    # 0/1 labels WITHOUT that flag — so also detect already-binary labels by
    # value (unique values ⊆ {0,1}). Without this, the pKi>=6.0 threshold on
    # already-0/1 labels collapses every sample to class 0 (degenerate).
    _uniq = set(np.unique(labels).tolist())
    if data.get('is_binary') or _uniq <= {0, 1, 0.0, 1.0}:
        labels_binary = labels.astype(int)
    else:
        labels_binary = (labels >= PKI_THRESHOLD).astype(int)
    n_active = int(labels_binary.sum())
    print(f"  {dataset_name}: {len(labels)} samples, "
          f"{n_active} active, {len(labels) - n_active} inactive")

    # Compute Morgan FPs (for valid_mask)
    fluid_df_all, valid_mask = smiles_to_fluid_df(
        smiles, labels_binary, pki_threshold=0.5)

    # Filter to valid molecules
    groups = data.get('groups')
    if not valid_mask.all():
        valid_idx = np.where(valid_mask)[0]
        reps = {r: arr[valid_idx] for r, arr in reps.items()}
        labels_binary = labels_binary[valid_idx]
        smiles = [smiles[i] for i in valid_idx]
        labels = labels[valid_idx]
        if groups is not None:
            groups = groups[valid_idx]
        # P0 fix: reindex augmentations too — otherwise subsequent indexing
        # with post-filter indices corrupts non-Lhasa runs.
        if data.get('augmentations'):
            data = {**data, 'augmentations': {
                k: v[valid_idx] for k, v in data['augmentations'].items()}}

    n_samples = len(labels_binary)

    # ── Get optimal hybrid config from precomputed search ──
    augs = data.get('augmentations')
    metric_name, _metric_fn, _lower_is_better = get_domain_metric(data)  # metric_name used below; fn/dir unused here
    # P1 fix: pool blueprint must be computed on the labels that create_hybrid
    # will actually see downstream (labels_binary). Labels were binarized at
    # line 127, so force task_type='classification' even when the original
    # dataset is regression (e.g. caco2/logd) — labels_binary is always binary.
    try:
        _search, _meta = get_best_search_for_downstream(
            reps, labels_binary, groups,
            downstream_model=fed_to_kirby_model(fed_model), budget=BUDGET,
            augmentations=augs,
            metric=metric_name,
            task_type='classification',
            domain=data.get('domain', 'molecular'),
            dataset_name=dataset_name,
            data=data,
        )
        _bp = _meta['best_config']['blueprint'] if _meta else None
    except Exception as e:
        print(f"  WARN: pool selection failed: {e}, using defaults")
        _bp = None
    aug_cond = _bp.augmentation_strategy if _bp else ('uncapped' if augs else 'none')
    _alloc = _bp.allocation_method if _bp else 'performance_weighted'
    _imp = _bp.importance_method if _bp else 'treeshap'
    _int_eval = _bp.internal_evaluator if _bp else 'match'
    _int_eval = _resolve_match(_int_eval, 'rf')

    # SPLIT — scaffold-group-aware, 5-fold per-fold accumulation.
    # P1 fix: hard-fail when groups unavailable. federated_helpers.py:107-110
    # silently sets groups=None, which would trigger the cv_split random
    # fallback (forbidden by feedback_no_random_splits.md).
    if groups is None or len(np.unique(groups)) < N_FOLDS:
        raise RuntimeError(
            f"fed04 requires scaffold groups for {N_FOLDS}-fold CV "
            f"(got groups={'None' if groups is None else f'{len(np.unique(groups))} unique'}). "
            "Random splits are forbidden — fix the loader to provide groups.")

    fold_records = []
    for fold_id, (train_idx, test_idx) in enumerate(cv_split(
            groups, N_FOLDS, np.zeros(n_samples), labels_binary,
            seed=fed_seed)):
        print(f"\n=== Fold {fold_id + 1}/{N_FOLDS}: "
              f"{len(train_idx)} train / {len(test_idx)} test ===")

        y_train = labels_binary[train_idx]
        y_test = labels_binary[test_idx]

        train_reps = {r: arr[train_idx] for r, arr in reps.items()}
        test_reps = {r: arr[test_idx] for r, arr in reps.items()}
        train_augs = ({k: v[train_idx] for k, v in augs.items()}
                      if augs else None)

        # CLUSTER TRAINING DATA ONLY — matching FLuID's pattern
        t0 = time.time()
        train_fluid_df = fluid_df_all.iloc[train_idx]
        train_silo_indices, cluster_data, kmeans_model = cluster_training_only(
            train_fluid_df, k=K, smooth_factor=0.05, seed=fed_seed)
        print(f"  Clustering (train only): K={K}, {time.time() - t0:.1f}s")

        for s in range(1, K + 1):
            idx = train_silo_indices[s]
            n_act = int(y_train[idx].sum())
            print(f"    Silo {s}: {len(idx)} samples "
                  f"({n_act} active, {len(idx) - n_act} inactive)")

        # Assign test molecules to nearest training cluster centroid
        test_fps = np.asarray(list(fluid_df_all.iloc[test_idx].FP))
        test_silo = kmeans_model.predict(test_fps) + 1  # 1-indexed to match silos

        # ── Phase 2: Per-cluster importance ──────────────────────────────
        print("  --- Phase 2: Per-cluster importance (smooth=0.05) ---")
        profiles, feature_infos, cluster_stats = _compute_cluster_importance(
            train_reps, y_train, train_silo_indices, K,
            alloc=_alloc, imp=_imp, int_eval=_int_eval,
            augs=train_augs, aug_cond=aug_cond,
            train_groups=(groups[train_idx] if groups is not None else None))

        # Global baseline (full training set)
        _, fi_global = create_hybrid(
            train_reps, y_train,
            groups=(groups[train_idx] if groups is not None else None),
            allocation_method=_alloc,
            importance_method=_imp,
            budget=BUDGET,
            internal_evaluator=_int_eval,
            augmentations=train_augs,
            augmentation_strategy=aug_cond)
        global_profile = extract_importance_profile(fi_global)
        global_top = max(global_profile, key=global_profile.get) if global_profile else 'none'
        print(f"  Global top rep: {global_top}")

        # ── Phase 3: Compare importance profiles ─────────────────────────
        mean_spearman, pairwise = compare_importance_profiles(profiles)
        _ms_str = 'n/a' if mean_spearman is None else f"{mean_spearman:.3f}"
        print(f"  Phase 3 mean pairwise Spearman: {_ms_str}")
        rep_rankings = {}
        for s, profile in profiles.items():
            ranked = sorted(profile.items(), key=lambda x: -x[1])
            rep_rankings[s] = [r for r, _ in ranked]

        # ── Phase 4: Cluster-conditional hybrid ──────────────────────────
        per_cluster_models = {}
        per_cluster_fis = {}
        for s in range(1, K + 1):
            fi = feature_infos.get(s)
            if fi is None:
                continue
            idx = train_silo_indices[s]
            X_s = apply_feature_selection(train_reps, fi)
            clf = fit_fed_classifier(
                fed_model, X_s[idx], y_train[idx], seed=fed_seed)
            per_cluster_models[s] = clf
            per_cluster_fis[s] = fi

        # Predict: use ORIGINAL FLuID cluster assignments for test molecules
        # (test_silo was computed earlier in this fold from the original
        # clustering — do NOT re-fit KMeans, which would produce misaligned
        # cluster IDs)
        per_cluster_test_X = {}
        for s, fi in per_cluster_fis.items():
            per_cluster_test_X[s] = apply_feature_selection(test_reps, fi)

        # Global hybrid baseline (compute first so fallback is available)
        X_global_train = apply_feature_selection(train_reps, fi_global)
        X_global_test = apply_feature_selection(test_reps, fi_global)
        clf_global = fit_fed_classifier(
            fed_model, X_global_train, y_train, seed=fed_seed)
        y_pred_global = clf_global.predict(X_global_test)
        y_prob_global = clf_global.predict_proba(X_global_test)[:, 1]

        # Per-cluster prediction using original cluster assignments
        y_pred_conditional = y_pred_global.copy()
        y_prob_conditional = y_prob_global.copy()
        fallback_mask = np.ones(len(test_idx), dtype=bool)
        for i in range(len(test_idx)):
            cluster = test_silo[i]
            if cluster in per_cluster_models and cluster in per_cluster_test_X:
                model = per_cluster_models[cluster]
                X_i = per_cluster_test_X[cluster][[i]]
                y_pred_conditional[i] = model.predict(X_i)[0]
                proba = model.predict_proba(X_i)
                y_prob_conditional[i] = proba[0, 1] if proba.shape[1] == 2 else 0.5
                fallback_mask[i] = False
        n_fallback = int(fallback_mask.sum())
        if n_fallback > 0:
            print(f"  {n_fallback} test samples used global fallback")

        conditional_metrics = compute_clf_metrics(
            y_test, y_pred_conditional, y_prob_conditional)
        global_metrics = compute_clf_metrics(
            y_test, y_pred_global, y_prob_global)
        print(f"  Cluster-conditional MCC={conditional_metrics['mcc']:.3f} | "
              f"Global hybrid MCC={global_metrics['mcc']:.3f}")

        # ── Phase 5: Smooth factor sweep ─────────────────────────────────
        smooth_sweep = {}
        for sf in SMOOTH_FACTORS:
            try:
                train_silo_idx_sf, _, _ = cluster_training_only(
                    train_fluid_df, k=K, smooth_factor=sf, seed=fed_seed)
                # P1 fix: pass augs/aug_cond — Phase 2 does, so Phase 5 must too.
                prof_sf, _, stats_sf = _compute_cluster_importance(
                    train_reps, y_train, train_silo_idx_sf, K,
                    alloc=_alloc, imp=_imp, int_eval=_int_eval,
                    augs=train_augs, aug_cond=aug_cond,
                    train_groups=(groups[train_idx] if groups is not None else None))
                mean_sp, pw = compare_importance_profiles(prof_sf)
                smooth_sweep[str(sf)] = convert_numpy({
                    'stability': mean_sp,
                    'pairwise': pw,
                    'cluster_stats': stats_sf,
                    'profiles': prof_sf,
                })
            except Exception as e:
                smooth_sweep[str(sf)] = {'error': str(e)}

        fold_records.append({
            'fold': fold_id,
            'n_train': len(train_idx),
            'n_test': len(test_idx),
            'global_importance': convert_numpy({
                'profile': global_profile,
                'top_rep': global_top,
            }),
            'per_cluster_importance': convert_numpy({
                'profiles': profiles,
                'cluster_stats': cluster_stats,
                'rep_rankings': rep_rankings,
            }),
            'profile_stability': convert_numpy({
                'mean_spearman': mean_spearman,
                'pairwise': pairwise,
            }),
            'cluster_conditional_hybrid': convert_numpy(conditional_metrics),
            'global_hybrid': convert_numpy(global_metrics),
            'smooth_factor_sweep': smooth_sweep,
            'n_fallback': n_fallback,
        })

    # ── Aggregate across folds ───────────────────────────────────────────
    def _mean_metric(records, key, metric):
        vals = [r[key].get(metric) for r in records
                if isinstance(r[key].get(metric), (int, float))
                and not np.isnan(r[key].get(metric))]
        return float(np.mean(vals)) if vals else float('nan')

    def _std_metric(records, key, metric):
        vals = [r[key].get(metric) for r in records
                if isinstance(r[key].get(metric), (int, float))
                and not np.isnan(r[key].get(metric))]
        return float(np.std(vals)) if vals else float('nan')

    metric_keys = ['mcc', 'bac', 'f1', 'auroc', 'accuracy', 'f1_weighted']
    cond_agg = {m: {'mean': _mean_metric(fold_records, 'cluster_conditional_hybrid', m),
                    'std': _std_metric(fold_records, 'cluster_conditional_hybrid', m)}
                for m in metric_keys}
    glob_agg = {m: {'mean': _mean_metric(fold_records, 'global_hybrid', m),
                    'std': _std_metric(fold_records, 'global_hybrid', m)}
                for m in metric_keys}

    stability_vals = [r['profile_stability'].get('mean_spearman') for r in fold_records]
    stability_vals = [v for v in stability_vals
                      if isinstance(v, (int, float)) and not np.isnan(v)]
    # None (JSON null) when no fold produced a defined stability — not NaN.
    mean_stability = float(np.mean(stability_vals)) if stability_vals else None

    # ── Phase 6: Results ──────────────────────────────────────────────────
    results = {
        'experiment': 'fed04_cluster_aware_importance',
        'dataset': dataset_name,
        'config': {
            'k': K,
            'pki_threshold': PKI_THRESHOLD,
            'budget': BUDGET,
            'smooth_factors': SMOOTH_FACTORS,
            'n_samples': n_samples,
            'n_folds': N_FOLDS,
        },
        'folds': fold_records,
        'aggregate': {
            'cluster_conditional_hybrid': cond_agg,
            'global_hybrid': glob_agg,
            'mean_profile_stability': mean_stability,
        },
    }

    # Summary
    print(f"\n{'─' * 70}")
    print(f"  Aggregated over {N_FOLDS} folds:")
    # mean_stability is None (JSON null) when no fold produced a defined
    # stability (<3 reps or all-constant importance) — guard the format spec.
    _stab_str = 'n/a' if mean_stability is None else f"{mean_stability:.3f}"
    print(f"  Profile stability (mean): {_stab_str}")
    print(f"  Conditional MCC: {cond_agg['mcc']['mean']:.3f} ± {cond_agg['mcc']['std']:.3f}")
    print(f"  Global MCC:      {glob_agg['mcc']['mean']:.3f} ± {glob_agg['mcc']['std']:.3f}")
    print(f"  Per-fold smooth factor sweep recorded under folds[].smooth_factor_sweep")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('Fed04: Cluster-Aware Importance')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(
        results, args.output, 'fed04_cluster_aware_importance', args.dataset)
