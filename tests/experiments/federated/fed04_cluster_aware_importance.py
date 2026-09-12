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
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy, cv_split,
    get_best_search_for_downstream,
)
from experiments.federated.federated_helpers import (
    smiles_to_fluid_df, cluster_training_only, compute_clf_metrics,
    extract_importance_profile, compare_importance_profiles,
    extract_selected_features, compare_selected_features,
    _resolve_match, fit_fed_classifier, fed_to_kirby_model,
    binarise_labels,
    smoke_grid, smoke_n_folds, smoke_k_silos,
)
from kirby.hybrid import create_hybrid, apply_feature_selection


# ── Constants ────────────────────────────────────────────────────────────────
# smoke_* helpers are no-ops unless KIRBY_SMOKE_N is set.

K = smoke_k_silos(5)
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
    # value (unique values ⊆ {0,1}). Without this, a threshold on already-0/1
    # labels collapses every sample to class 0 (degenerate).
    labels_binary, cutoff_note = binarise_labels(
        dataset_name, labels, is_binary=bool(data.get('is_binary')))
    n_active = int(labels_binary.sum())
    print(f"  {dataset_name}: {len(labels)} samples, "
          f"{n_active} active, {len(labels) - n_active} inactive "
          f"({cutoff_note})")

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
    # The pool blueprint has to come from a search over the labels create_hybrid
    # will actually see downstream — labels_binary — so task_type is
    # 'classification' even where the dataset is registered as regression
    # (caco2, logd).
    #
    # Two consequences, both fixed 2026-09-04.
    #
    # data= is NOT passed. data['precomputed_searches'] is keyed by
    # augmentation strategy alone and holds the CLEAN, continuous-label
    # searches, so a hit there answered this classification request with a
    # regression search — same allocation, importance and evaluator names, but
    # ranked on the continuous labels. Dropping data= sends the request to the
    # on-disk cache, whose identity carries labels_hash.
    #
    # The metric is 'auc', not the registered one. get_domain_metric returns
    # 'rmse' for caco2 and logd, and scoring a classifier with RMSE is not a
    # thing this file can report. herg already carries 'auc'.
    try:
        _search, _meta = get_best_search_for_downstream(
            reps, labels_binary, groups,
            downstream_model=fed_to_kirby_model(fed_model), budget=BUDGET,
            augmentations=augs,
            metric='auc',
            task_type='classification',
            domain=data.get('domain', 'molecular'),
            dataset_name=dataset_name,
            data=None,
        )
        _bp = _meta['best_config']['blueprint'] if _meta else None
        if _bp is None:
            print("  WARN: the search returned no recipe for "
                  f"{fed_to_kirby_model(fed_model)} at budget {BUDGET} — "
                  "falling back to performance_weighted / treeshap / match.")
    except Exception as e:
        print(f"  WARN: pool selection failed: {e}, using defaults")
        _bp = None
    # Whether every condition ran on a searched recipe or on the fallback is
    # not something a reader should have to infer from the log, so it is
    # written into the result file below as config.recipe_source.
    _recipe_source = 'searched' if _bp is not None else 'fallback_defaults'
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

        # ── Phase 3b: Compare the selected features themselves ───────────
        # The Spearman above ranks representations by how much of the budget
        # each one won. Two clusters can split 100 features across the same
        # representations in the same proportions and still take 100 different
        # descriptors and fingerprint bits, and that comparison says so.
        cluster_selections = {
            s: extract_selected_features(fi)
            for s, fi in feature_infos.items() if fi is not None}
        feature_overlap = compare_selected_features(cluster_selections)
        global_selection = extract_selected_features(fi_global)
        _global_set = set(global_selection['labels'])
        # How much of each cluster's own selection the whole-training-fold
        # selection also took. A cluster at 1.0 is asking for nothing the
        # global selection does not already give it.
        cluster_vs_global = {}
        for s, sel in cluster_selections.items():
            own = set(sel['labels'])
            cluster_vs_global[s] = {
                'n_selected': len(own),
                'n_also_in_global': len(own & _global_set),
                'fraction_also_in_global': (
                    float(len(own & _global_set) / len(own)) if own else None),
                'not_in_global': sorted(own - _global_set),
            }
        _mj = feature_overlap.get('mean_jaccard')
        _nc = feature_overlap.get('n_clusters', 0)
        if _mj is None:
            print(f"  Phase 3b feature overlap: undefined — {_nc} cluster(s) "
                  f"produced a selection, so there is no pair to compare")
        else:
            print(f"  Phase 3b mean pairwise feature overlap: {_mj:.3f} "
                  f"across {_nc} clusters  "
                  f"({feature_overlap['n_shared_by_all']} of the "
                  f"{BUDGET} features taken by every one of them)")
        for s in sorted(cluster_selections):
            _top = cluster_selections[s]['labels'][:5]
            print(f"    Cluster {s} top 5: {', '.join(_top)}")

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
            # Which features, not just which representations. 'per_cluster'
            # lists every feature each cluster took, most important first;
            # 'overlap' says how far any two clusters agree; 'vs_global' says
            # how much each cluster wanted that the whole-training-fold
            # selection did not give it.
            'selected_features': convert_numpy({
                'per_cluster': cluster_selections,
                'overlap': feature_overlap,
                'global': global_selection,
                'vs_global': cluster_vs_global,
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

    # ── Which features this dataset prefers, across every fold ───────────
    # Each fold x cluster produced one selection of BUDGET features. Counting
    # how many of those selections took each feature says what the dataset
    # wants everywhere (a high count) against what one region of chemical
    # space wants and the rest does not (a count of 1). The same counts from
    # another dataset's file are directly comparable, because a feature label
    # is the representation and the descriptor name or index, nothing else.
    _cluster_sel_counts = {}
    _n_cluster_selections = 0
    _global_sel_counts = {}
    _n_global_selections = 0
    _jaccards = []
    for _r in fold_records:
        _sf = _r.get('selected_features') or {}
        for _s, _sel in (_sf.get('per_cluster') or {}).items():
            _n_cluster_selections += 1
            for _lab in _sel.get('labels', []):
                _cluster_sel_counts[_lab] = _cluster_sel_counts.get(_lab, 0) + 1
        _g = _sf.get('global') or {}
        if _g.get('labels'):
            _n_global_selections += 1
            for _lab in _g['labels']:
                _global_sel_counts[_lab] = _global_sel_counts.get(_lab, 0) + 1
        _j = (_sf.get('overlap') or {}).get('mean_jaccard')
        if isinstance(_j, (int, float)) and not np.isnan(_j):
            _jaccards.append(float(_j))

    _always = sorted(l for l, n in _cluster_sel_counts.items()
                     if n == _n_cluster_selections) if _n_cluster_selections else []
    _once = sorted(l for l, n in _cluster_sel_counts.items() if n == 1)
    _ranked_feats = sorted(_cluster_sel_counts.items(), key=lambda x: (-x[1], x[0]))

    dataset_feature_preference = {
        'n_cluster_selections': _n_cluster_selections,
        'n_global_selections': _n_global_selections,
        'budget_per_selection': BUDGET,
        # Every feature any cluster took, with how many of the
        # n_cluster_selections selections took it. Full list, not a top slice.
        'cluster_selection_counts': dict(_ranked_feats),
        'global_selection_counts': _global_sel_counts,
        'taken_by_every_cluster_in_every_fold': _always,
        'n_taken_by_every_cluster_in_every_fold': len(_always),
        'taken_exactly_once': _once,
        'n_taken_exactly_once': len(_once),
        'n_distinct_features_ever_taken': len(_cluster_sel_counts),
        'mean_pairwise_feature_overlap': (
            float(np.mean(_jaccards)) if _jaccards else None),
    }

    # ── Phase 6: Results ──────────────────────────────────────────────────
    results = {
        'experiment': 'fed04_cluster_aware_importance',
        'dataset': dataset_name,
        'config': {
            'k': K,
            'classification_cutoff': cutoff_note,
            'recipe_source': _recipe_source,
            'allocation_method': _alloc,
            'importance_method': _imp,
            'internal_evaluator': _int_eval,
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
            'feature_preference': convert_numpy(dataset_feature_preference),
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
    _fp = dataset_feature_preference
    _ov = _fp['mean_pairwise_feature_overlap']
    _ov_str = 'n/a' if _ov is None else f"{_ov:.3f}"
    print(f"\n  Features: {_fp['n_distinct_features_ever_taken']} distinct "
          f"features taken across {_fp['n_cluster_selections']} selections of "
          f"{BUDGET} features each")
    print(f"  Mean pairwise overlap between two clusters' selections: {_ov_str}")
    print(f"  Taken by every cluster in every fold: "
          f"{_fp['n_taken_by_every_cluster_in_every_fold']} features")
    print(f"  Taken by exactly one cluster in one fold: "
          f"{_fp['n_taken_exactly_once']} features")
    _top = list(_fp['cluster_selection_counts'].items())[:10]
    if _top:
        print(f"  Most-taken 10 (count out of {_fp['n_cluster_selections']}):")
        for _lab, _n in _top:
            print(f"    {_n:>3}  {_lab}")
    print(f"  Full counts under aggregate.feature_preference")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('Fed04: Cluster-Aware Importance')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(
        results, args.output, 'fed04_cluster_aware_importance', args.dataset)
