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
    smoke_grid, smoke_n_folds, smoke_k_silos, smoke_search_kwargs,
    federated_budgets,
)
from kirby.hybrid import create_hybrid, apply_feature_selection


# ── Constants ────────────────────────────────────────────────────────────────
# smoke_* helpers are no-ops unless KIRBY_SMOKE_N is set.

K = smoke_k_silos(5)
SMOOTH_FACTORS = smoke_grid([0, 0.01, 0.05, 0.1, 0.2], [0, 0.05])
# Every number this file reported was a number about 100 features. The
# three arms are compared at each width in BUDGETS; the headline tables
# below are the headline width so they read as they always did.
BUDGETS = federated_budgets()
HEADLINE_BUDGET = 100 if 100 in BUDGETS else BUDGETS[0]
N_FOLDS = smoke_n_folds(5)


def _compute_cluster_importance(train_reps, train_labels, silo_indices, k,
                                budget=HEADLINE_BUDGET,
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


def _run_one(data, config, output_dir, dataset_name, **kwargs):
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
            downstream_model=fed_to_kirby_model(fed_model),
            budget=HEADLINE_BUDGET,
            augmentations=augs,
            metric='auc',
            task_type='classification',
            domain=data.get('domain', 'molecular'),
            dataset_name=dataset_name,
            data=None,
            **smoke_search_kwargs(config),
        )
        _bp = _meta['best_config']['blueprint'] if _meta else None
        if _bp is None:
            print("  WARN: the search returned no recipe for "
                  f"{fed_to_kirby_model(fed_model)} at budget "
                  f"{HEADLINE_BUDGET} — "
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
    # The folds are drawn once and reused at every width, so a difference
    # between two budgets is a difference in width and nothing else.
    _folds = list(cv_split(groups, N_FOLDS, np.zeros(n_samples),
                           labels_binary, seed=fed_seed))
    for BUDGET in BUDGETS:
        print(f"\n########## budget {BUDGET} features "
              f"({BUDGETS.index(BUDGET) + 1} of {len(BUDGETS)}) ##########")
        for fold_id, (train_idx, test_idx) in enumerate(_folds):
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
                train_reps, y_train, train_silo_indices, K, budget=BUDGET,
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

            # ── Phase 2b: Null control — a RANDOM partition of the same sizes ─
            # Phase 3 asks whether two regions of chemical space split the budget
            # across the representations differently. Two subsamples of a few
            # hundred rows already split it differently from subsampling alone,
            # so the number Phase 3 reports has no scale without something to
            # read it against. This partitions the SAME training rows into the
            # SAME cluster sizes at random — no chemistry in the assignment — and
            # runs the identical per-cluster selection over it. The chemical
            # clusters are doing something only if their profiles agree LESS with
            # each other than the random parts of the same sizes do.
            _sizes = [len(train_silo_indices[s]) for s in range(1, K + 1)]
            _perm = np.random.RandomState(1000 + fed_seed + fold_id).permutation(
                len(train_idx))
            _null_indices, _cut = {}, 0
            for _s, _n in enumerate(_sizes, 1):
                _null_indices[_s] = _perm[_cut:_cut + _n]
                _cut += _n
            null_profiles, _, null_cluster_stats = _compute_cluster_importance(
                train_reps, y_train, _null_indices, K, budget=BUDGET,
                alloc=_alloc, imp=_imp, int_eval=_int_eval,
                augs=train_augs, aug_cond=aug_cond,
                train_groups=(groups[train_idx] if groups is not None else None))
            null_spearman, null_pairwise = compare_importance_profiles(
                null_profiles)

            # ── Phase 3: Compare importance profiles ─────────────────────────
            mean_spearman, pairwise = compare_importance_profiles(profiles)
            _ms_str = 'n/a' if mean_spearman is None else f"{mean_spearman:.3f}"
            _ns_str = 'n/a' if null_spearman is None else f"{null_spearman:.3f}"
            print(f"  Phase 3 mean pairwise Spearman: {_ms_str}  "
                  f"(random partition of the same sizes: {_ns_str})")
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
            # THREE arms, not two. The two that were here answer a question with
            # two things moving at once:
            #   global            — all training rows, ONE feature set for all.
            #   cluster_model     — this cluster's rows only, its OWN features.
            # A cluster's rows are a fifth of the fold and FLuID's classifier puts
            # every fit through RandomUnderSampler first, so on caco2 the
            # per-cluster forests trained on 58 to 596 rows against roughly a
            # thousand for the global one (measured 2026-09-18, all five folds).
            # Whatever the gap between those two arms is, most of it is the row
            # count, and the experiment reads it as the features.
            #   cluster_features  — ALL training rows, this cluster's features.
            # The third arm holds the rows fixed and varies only the feature set,
            # which is the question fed04 exists to ask. cluster_features minus
            # global is the value of choosing features per region; cluster_model
            # minus cluster_features is what fitting on one region's rows costs.
            per_cluster_models = {}
            per_cluster_models_allrows = {}
            per_cluster_fis = {}
            per_cluster_fit_rows = {}
            for s in range(1, K + 1):
                fi = feature_infos.get(s)
                if fi is None:
                    continue
                idx = train_silo_indices[s]
                X_s = apply_feature_selection(train_reps, fi)
                clf = fit_fed_classifier(
                    fed_model, X_s[idx], y_train[idx], seed=fed_seed)
                per_cluster_models[s] = clf
                # Same features, every training row.
                per_cluster_models_allrows[s] = fit_fed_classifier(
                    fed_model, X_s, y_train, seed=fed_seed)
                per_cluster_fis[s] = fi
                # How many rows each arm actually fitted on, AFTER the
                # under-sampler, so the row-count confound is in the file rather
                # than only in this comment.
                _na = int(y_train[idx].sum())
                per_cluster_fit_rows[s] = {
                    'n_cluster_rows': int(len(idx)),
                    'n_cluster_rows_after_undersample': 2 * min(
                        _na, int(len(idx)) - _na),
                    'n_all_train_rows': int(len(y_train)),
                    'n_all_train_rows_after_undersample': 2 * min(
                        int(y_train.sum()), int(len(y_train) - y_train.sum())),
                }

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

            # Route each held-out molecule to its own cluster, twice: once to the
            # model fitted on that cluster's rows, once to the model fitted on
            # every training row. Same routing, same features, different training
            # set — so the two differ by the row count and nothing else.
            def _route(models):
                y_p = y_pred_global.copy()
                y_q = y_prob_global.copy()
                miss = np.ones(len(test_idx), dtype=bool)
                for i in range(len(test_idx)):
                    cluster = test_silo[i]
                    if cluster in models and cluster in per_cluster_test_X:
                        mdl = models[cluster]
                        X_i = per_cluster_test_X[cluster][[i]]
                        y_p[i] = mdl.predict(X_i)[0]
                        proba = mdl.predict_proba(X_i)
                        y_q[i] = proba[0, 1] if proba.shape[1] == 2 else 0.5
                        miss[i] = False
                return y_p, y_q, miss

            y_pred_conditional, y_prob_conditional, fallback_mask = _route(
                per_cluster_models)
            y_pred_featonly, y_prob_featonly, _ = _route(
                per_cluster_models_allrows)
            n_fallback = int(fallback_mask.sum())
            if n_fallback > 0:
                print(f"  {n_fallback} test samples used global fallback")

            conditional_metrics = compute_clf_metrics(
                y_test, y_pred_conditional, y_prob_conditional)
            featonly_metrics = compute_clf_metrics(
                y_test, y_pred_featonly, y_prob_featonly)
            global_metrics = compute_clf_metrics(
                y_test, y_pred_global, y_prob_global)
            print(f"  MCC — per-cluster features + per-cluster rows "
                  f"{conditional_metrics['mcc']:.3f} | per-cluster features + ALL "
                  f"rows {featonly_metrics['mcc']:.3f} | global features + all "
                  f"rows {global_metrics['mcc']:.3f}")

            # ── Phase 5: Smooth factor sweep ─────────────────────────────────
            smooth_sweep = {}
            for sf in SMOOTH_FACTORS:
                try:
                    train_silo_idx_sf, _, _ = cluster_training_only(
                        train_fluid_df, k=K, smooth_factor=sf, seed=fed_seed)
                    # P1 fix: pass augs/aug_cond — Phase 2 does, so Phase 5 must too.
                    prof_sf, _, stats_sf = _compute_cluster_importance(
                        train_reps, y_train, train_silo_idx_sf, K, budget=BUDGET,
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
                'budget': BUDGET,
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
                'profile_stability_null': convert_numpy({
                    'mean_spearman': null_spearman,
                    'pairwise': null_pairwise,
                    'cluster_stats': null_cluster_stats,
                    'partition': 'random, same cluster sizes as K-Means',
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
                'cluster_features_all_rows': convert_numpy(featonly_metrics),
                'global_hybrid': convert_numpy(global_metrics),
                'per_cluster_fit_rows': convert_numpy(per_cluster_fit_rows),
                'smooth_factor_sweep': smooth_sweep,
                'n_fallback': n_fallback,
            })

    # ── Aggregate across folds ───────────────────────────────────────────
    # Every table below this line reads as it did before the sweep: it is
    # the headline width only. The other widths are in 'by_budget' and in
    # 'folds', where each fold record carries the width it was measured at.
    headline_records = [r for r in fold_records
                        if r.get('budget') == HEADLINE_BUDGET]
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
    cond_agg = {m: {'mean': _mean_metric(headline_records, 'cluster_conditional_hybrid', m),
                    'std': _std_metric(headline_records, 'cluster_conditional_hybrid', m)}
                for m in metric_keys}
    glob_agg = {m: {'mean': _mean_metric(headline_records, 'global_hybrid', m),
                    'std': _std_metric(headline_records, 'global_hybrid', m)}
                for m in metric_keys}
    feat_agg = {m: {'mean': _mean_metric(headline_records,
                                         'cluster_features_all_rows', m),
                    'std': _std_metric(headline_records,
                                       'cluster_features_all_rows', m)}
                for m in metric_keys}

    stability_vals = [r['profile_stability'].get('mean_spearman') for r in headline_records]
    stability_vals = [v for v in stability_vals
                      if isinstance(v, (int, float)) and not np.isnan(v)]
    # None (JSON null) when no fold produced a defined stability — not NaN.
    mean_stability = float(np.mean(stability_vals)) if stability_vals else None

    null_vals = [r.get('profile_stability_null', {}).get('mean_spearman')
                 for r in headline_records]
    null_vals = [v for v in null_vals
                 if isinstance(v, (int, float)) and not np.isnan(v)]
    mean_stability_null = float(np.mean(null_vals)) if null_vals else None

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
    for _r in headline_records:
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
        'budget_per_selection': HEADLINE_BUDGET,
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
    # The smooth-factor sweep, averaged over folds. Every value below is
    # already in folds[].smooth_factor_sweep; nothing aggregated them, so the
    # question this sweep exists to answer — does the K-Means smooth_factor
    # change how stable the per-cluster profiles are — could only be answered
    # by averaging five per-fold numbers by hand for each of five factors.
    smooth_factor_stability = {}
    for _sf in SMOOTH_FACTORS:
        _key = str(_sf)
        _vals = []
        for _r in headline_records:
            _e = (_r.get('smooth_factor_sweep') or {}).get(_key) or {}
            _v = _e.get('stability')
            if isinstance(_v, (int, float)) and not np.isnan(_v):
                _vals.append(float(_v))
        smooth_factor_stability[_key] = {
            'mean': float(np.mean(_vals)) if _vals else None,
            'std': float(np.std(_vals)) if _vals else None,
            'n_folds_defined': len(_vals),
        }

    # How many held-out molecules landed in a cluster with no per-cluster
    # model and were scored by the whole-fold model instead. A large count
    # makes the two arms of the comparison partly the same arm, so it belongs
    # beside the two scores rather than only inside each fold.
    _fb = [int(_r.get('n_fallback', 0)) for _r in headline_records]
    _nte = [int(_r.get('n_test', 0)) for _r in headline_records]
    n_fallback_agg = {
        'mean': float(np.mean(_fb)) if _fb else None,
        'std': float(np.std(_fb)) if _fb else None,
        'per_fold': _fb,
        'mean_fraction_of_test': (
            float(np.mean([a / b for a, b in zip(_fb, _nte) if b]))
            if any(_nte) else None),
    }

    # The three arms at every width. One row is one width; the numbers are
    # the mean over the folds of that width, in MCC.
    by_budget = {}
    for _b in BUDGETS:
        _recs = [r for r in fold_records if r.get('budget') == _b]
        if not _recs:
            continue
        by_budget[str(_b)] = {
            'n_folds': len(_recs),
            'cluster_conditional_hybrid_mcc':
                _mean_metric(_recs, 'cluster_conditional_hybrid', 'mcc'),
            'cluster_features_all_rows_mcc':
                _mean_metric(_recs, 'cluster_features_all_rows', 'mcc'),
            'global_hybrid_mcc': _mean_metric(_recs, 'global_hybrid', 'mcc'),
        }
    print(f"\n  {'budget':>8} {'conditional':>12} {'feat_allrows':>13} "
          f"{'global':>10}   (mean MCC over folds)")
    for _b, _row in by_budget.items():
        print(f"  {_b:>8} {_row['cluster_conditional_hybrid_mcc']:>12.4f} "
              f"{_row['cluster_features_all_rows_mcc']:>13.4f} "
              f"{_row['global_hybrid_mcc']:>10.4f}")

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
            'budgets_swept': BUDGETS,
            'headline_budget': HEADLINE_BUDGET,
            'smooth_factors': SMOOTH_FACTORS,
            'n_samples': n_samples,
            'n_folds': N_FOLDS,
        },
        'by_budget': convert_numpy(by_budget),
        'folds': fold_records,
        'aggregate': {
            'cluster_conditional_hybrid': cond_agg,
            'cluster_features_all_rows': feat_agg,
            'global_hybrid': glob_agg,
            'mean_profile_stability': mean_stability,
            'mean_profile_stability_null': mean_stability_null,
            'smooth_factor_stability': convert_numpy(smooth_factor_stability),
            'n_fallback': convert_numpy(n_fallback_agg),
            'feature_preference': convert_numpy(dataset_feature_preference),
        },
    }

    # Summary
    print(f"\n{'─' * 70}")
    print(f"  Aggregated over {N_FOLDS} folds:")
    # mean_stability is None (JSON null) when no fold produced a defined
    # stability (<3 reps or all-constant importance) — guard the format spec.
    _stab_str = 'n/a' if mean_stability is None else f"{mean_stability:.3f}"
    _null_str = ('n/a' if mean_stability_null is None
                 else f"{mean_stability_null:.3f}")
    print(f"  Profile stability (mean): {_stab_str}")
    print(f"  Same number over a RANDOM partition of the same cluster sizes: "
          f"{_null_str}")
    print(f"  MCC, per-cluster features + per-cluster rows: "
          f"{cond_agg['mcc']['mean']:.3f} ± {cond_agg['mcc']['std']:.3f}")
    print(f"  MCC, per-cluster features + ALL training rows: "
          f"{feat_agg['mcc']['mean']:.3f} ± {feat_agg['mcc']['std']:.3f}")
    print(f"  MCC, global features + all training rows:      "
          f"{glob_agg['mcc']['mean']:.3f} ± {glob_agg['mcc']['std']:.3f}")
    print(f"  choosing features per region is worth "
          f"{feat_agg['mcc']['mean'] - glob_agg['mcc']['mean']:+.4f} MCC; "
          f"fitting on one region's rows costs "
          f"{cond_agg['mcc']['mean'] - feat_agg['mcc']['mean']:+.4f}")
    print(f"  Smooth factor -> mean pairwise Spearman over {N_FOLDS} folds:")
    for _k, _v in smooth_factor_stability.items():
        _m = 'n/a' if _v['mean'] is None else f"{_v['mean']:.3f}"
        _sd = '' if _v['std'] is None else f" ± {_v['std']:.3f}"
        print(f"    smooth_factor={_k}: {_m}{_sd} "
              f"({_v['n_folds_defined']} of {N_FOLDS} folds defined)")
    _fbm = n_fallback_agg['mean']
    print(f"  Held-out molecules scored by the whole-fold model instead: "
          f"{'n/a' if _fbm is None else f'{_fbm:.1f}'} per fold")
    _fp = dataset_feature_preference
    _ov = _fp['mean_pairwise_feature_overlap']
    _ov_str = 'n/a' if _ov is None else f"{_ov:.3f}"
    print(f"\n  Features: {_fp['n_distinct_features_ever_taken']} distinct "
          f"features taken across {_fp['n_cluster_selections']} selections of "
          f"{HEADLINE_BUDGET} features each")
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


# ── One (FLuID model, seed) call saved as it finishes (2026-09-25) ──────────
# federated_helpers.run_all calls run() once per (fed_model, fed_seed) -- nine
# calls -- and wrote fed04_cluster_aware_importance/<dataset>.json only after the ninth. A kill or
# an exception in any call lost every call before it. Each call's return value
# is now written to fed04_cluster_aware_importance/<dataset>.partial.jsonl the moment it returns
# and read back on the next attempt (_checkpoint.resumable_run), keyed on the
# model, the seed and any keyword argument, under an identity that includes the
# input fingerprint, the plain config and the KIRBY_* settings. The returned
# dict is decoded to the same objects, so aggregate_seed_runs and the final
# JSON are unchanged. --force-rerun / KIRBY_FORCE_RERUN=1 discards the file.
def run(data, config, output_dir, dataset_name, **kwargs):
    from experiments._checkpoint import resumable_run
    _key = {'fed_model': str(config.get('fed_model', 'rf')),
            'fed_seed': int(config.get('fed_seed', 0)),
            **{k: v for k, v in kwargs.items()}}
    return resumable_run(
        'fed04_cluster_aware_importance',
        lambda: _run_one(data, config, output_dir, dataset_name, **kwargs),
        data, config, output_dir, dataset_name, key=_key)



if __name__ == '__main__':
    parser = create_experiment_parser('Fed04: Cluster-Aware Importance')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(
        results, args.output, 'fed04_cluster_aware_importance', args.dataset)
