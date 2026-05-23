#!/usr/bin/env python3
"""Fed01: Noise Heterogeneity Across Silos + Federated Noise Detection.

Tests whether KIRBy's feature-level fusion handles heterogeneous noise across
federated silos better than FLuID's prediction-level ensemble. Uses hERG
binary classification (same endpoint as Lhasa's published hackathon, MCC 0.55).

7 conditions compared at each noise level:
    0. Centralized KIRBy hybrid (privacy-violating upper bound)
    1. Naive pool (privacy-violating baseline)
    2. FLuID similarity-weighted ensemble (privacy-preserving)
    3. FLuID quality-weighted ensemble (privacy-preserving)
    4. FLuID published student pipeline (privacy-preserving)
    5. KIRBy federated feature selection (privacy-preserving)
    6. KIRBy + per-silo noise detection (privacy-preserving)

Plus: noise detection analysis, importance-noise correlation, smooth factor sweep.

Usage:
    python fed01_noise_heterogeneity.py --dataset herg --mode intermediary
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy, cv_split,
    get_precomputed_search, get_best_search_for_downstream, get_domain_metric,
)
from experiments.federated.federated_helpers import (
    smiles_to_fluid_df, cluster_training_only, inject_noise_per_silo,
    train_teachers_per_silo, predict_fluid_similarity_ensemble,
    predict_fluid_quality_ensemble, federate_labels_and_train_student,
    federated_feature_selection, federated_noise_detection,
    compute_clf_metrics,
)
from kirby.hybrid import create_hybrid, apply_feature_selection, apply_augmentation_selection


# ── Constants ────────────────────────────────────────────────────────────────

K = 5                              # number of silos/clusters
PKI_THRESHOLD = 6.0                # pKi >= 6.0 → Active (Ki <= 1 uM)
SIGMA_FRACS = [0, 0.05, 0.1, 0.15, 0.2, 0.3]
SMOOTH_FACTORS = [0, 0.01, 0.05, 0.1, 0.2]

DETECTORS = [
    'knn_k5', 'knn_k10', 'lof', 'distance_weighted',
    'cv_disagreement', 'bootstrap_ensemble',
    'zscore', 'prediction_error',
]




def _condition_naive_pool(train_fps, train_labels_noisy, test_fps,
                          test_labels_clean):
    """Condition 1: Merge all silo data, train single model.

    Privacy: VIOLATES (all data pooled). Baseline only.
    """
    clf = RandomForestClassifier(
        n_estimators=150, max_depth=15, class_weight='balanced',
        random_state=0)
    clf.fit(train_fps, train_labels_noisy)
    pred_probs = clf.predict_proba(test_fps)
    pred_classes = clf.predict(test_fps)
    prob_positive = pred_probs[:, 1] if pred_probs.shape[1] == 2 else None
    return compute_clf_metrics(test_labels_clean, pred_classes, prob_positive)


def run(data, config, output_dir, dataset_name, **kwargs):
    """Fed01: Noise heterogeneity across silos."""
    print("\n" + "=" * 70)
    print("EXPERIMENT FED01: NOISE HETEROGENEITY ACROSS SILOS")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    smiles = data['smiles']

    # ── Phase 1: Setup ───────────────────────────────────────────────────
    print("\n--- Phase 1: Setup ---")

    # Binarize labels (skip if already binary from Lhasa loader)
    if data.get('is_binary'):
        labels_binary = labels.astype(int)
        print(f"  {dataset_name}: {len(labels)} samples (pre-binarized), "
              f"{int(labels_binary.sum())} active, "
              f"{len(labels_binary) - int(labels_binary.sum())} inactive")
    else:
        labels_binary = (labels >= PKI_THRESHOLD).astype(int)
        n_active = int(labels_binary.sum())
        n_inactive = len(labels_binary) - n_active
        print(f"  {dataset_name}: {len(labels)} samples, "
              f"{n_active} active, {n_inactive} inactive "
              f"(threshold pKi >= {PKI_THRESHOLD})")

    # Compute Morgan FPs for all molecules (to get valid_mask)
    fluid_df_all, valid_mask = smiles_to_fluid_df(
        smiles, labels_binary, pki_threshold=0.5)

    # Filter to valid molecules only
    groups = data.get('groups')
    if not valid_mask.all():
        n_invalid = int((~valid_mask).sum())
        print(f"  WARNING: {n_invalid} molecules failed RDKit parsing, "
              f"excluded from experiment")
        valid_idx = np.where(valid_mask)[0]
        reps = {r: arr[valid_idx] for r, arr in reps.items()}
        labels_binary = labels_binary[valid_idx]
        labels = labels[valid_idx]
        if groups is not None:
            groups = groups[valid_idx]
        smiles = [smiles[i] for i in valid_idx]

    n_samples = len(labels_binary)

    # ── Get optimal hybrid config from precomputed search ──
    augs = data.get('augmentations')
    aug_cond = 'uncapped' if augs else 'none'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    try:
        search, meta = get_best_search_for_downstream(
            reps, labels, groups,
            downstream_model='rf', budget=100,
            augmentations=augs,
            speed=config.get('search_speed', 'full'),
            metric=metric_name,
            task_type=data['task_type'],
            domain=data.get('domain', 'molecular'),
            dataset_name=dataset_name,
            data=data,
        )
        bp = meta['best_config']['blueprint'] if meta else None
    except Exception:
        bp = None
    _alloc = bp.allocation_method if bp else 'performance_weighted'
    _imp = bp.importance_method if bp else 'treeshap'
    _int_eval = bp.internal_evaluator if bp else 'match'

    # SPLIT FIRST — scaffold-group-aware, matching FLuID's pattern
    # (FLuID separates training/test before clustering)
    train_idx, test_idx = next(cv_split(
        groups, 1, np.zeros(n_samples), labels_binary))
    test_labels_clean = labels_binary[test_idx]
    print(f"  Split: {len(train_idx)} train / {len(test_idx)} test")

    # CLUSTER TRAINING DATA ONLY — matching FLuID's cluster_data_space(trainingData)
    # Test molecules never influence K-Means centroids.
    t0 = time.time()
    train_fluid_df = fluid_df_all.iloc[train_idx]
    train_silo_indices, cluster_data, _ = cluster_training_only(
        train_fluid_df, k=K, smooth_factor=0.05)

    print(f"  Clustering (train only): K={K}, {time.time() - t0:.1f}s")
    train_labels_binary = labels_binary[train_idx]
    for s in range(1, K + 1):
        idx = train_silo_indices[s]
        n_act = int(train_labels_binary[idx].sum())
        print(f"    Silo {s}: {len(idx)} samples "
              f"({n_act} active, {len(idx) - n_act} inactive)")

    # Morgan FPs for FLuID conditions
    all_fps = np.asarray(list(fluid_df_all.FP))
    train_fps = all_fps[train_idx]
    test_fps = all_fps[test_idx]

    # ── Phase 2: Primary noise sweep ─────────────────────────────────────
    print("\n--- Phase 2: Primary noise sweep ---")
    primary_sweep = {}

    for sigma_frac in SIGMA_FRACS:
        print(f"\n  === Sigma={sigma_frac} (noise on silo {K}) ===")

        # Inject noise on training labels only, noisiest silo = silo K
        noise_levels = {s: 0.0 for s in range(1, K + 1)}
        noise_levels[K] = sigma_frac

        train_labels_clean = labels_binary[train_idx]
        train_labels_noisy, noise_mask_global, noise_mask_per_silo = \
            inject_noise_per_silo(
                train_labels_clean, train_silo_indices,
                noise_levels, noise_type='label_flip')

        n_flipped = int(noise_mask_global.sum())
        print(f"    {n_flipped}/{len(train_idx)} labels flipped in training")

        sigma_results = {'n_flipped': n_flipped}

        # Rebuild training cluster_data with noisy labels.
        # FLuID's cluster_data[1..K] are smoothed views (k-NN bleed across
        # clusters when smooth_factor>0). Re-slicing by CLUSTER would drop
        # those smoothed rows — instead, copy each existing view and overwrite
        # CLASS/ACTIVITY for the rows whose index appears in our noise table
        # (P1-8 fix).
        train_df_noisy = cluster_data[0].copy()
        train_df_noisy['CLASS'] = train_labels_noisy
        train_df_noisy['ACTIVITY'] = np.where(
            train_labels_noisy == 1, 'Active', 'Inactive')
        cluster_data_noisy = [train_df_noisy]
        noisy_lookup = train_df_noisy[['CLASS', 'ACTIVITY']]
        for s in range(1, K + 1):
            smoothed_view = cluster_data[s].copy()
            shared = smoothed_view.index.intersection(noisy_lookup.index)
            if len(shared) > 0:
                smoothed_view.loc[shared, 'CLASS'] = noisy_lookup.loc[shared, 'CLASS']
                smoothed_view.loc[shared, 'ACTIVITY'] = noisy_lookup.loc[shared, 'ACTIVITY']
            cluster_data_noisy.append(smoothed_view)

        # Train FLuID teachers on training data per cluster
        # Matches FLuID's build_teacher_models: one RF per cluster, balance=False
        # (KIRBy uses class_weight='balanced' as a lighter alternative to FLuID's
        # RandomUnderSampler — documented deviation)
        from sklearn.ensemble import RandomForestClassifier as RFC
        teacher_models = []
        cluster_data_train = [None]  # Index 0 placeholder (Teacher 0 unused)

        for t in range(0, K + 1):
            cd = cluster_data_noisy[t]
            if len(cd) < 2:
                teacher_models.append((None, 0, cd))
                if t > 0:
                    cluster_data_train.append(cd)
                continue
            X_t = np.asarray(list(cd.FP))
            y_t = np.asarray(list(cd.CLASS))
            clf = RFC(n_estimators=150, max_depth=15,
                      class_weight='balanced', random_state=0)
            clf.fit(X_t, y_t)
            teacher_models.append((clf, len(cd), cd))
            if t > 0:
                cluster_data_train.append(cd)

        # -- Condition 0: Centralized KIRBy hybrid trained on NOISY labels --
        # (i.e. matched to the federated conditions; this is NOT a clean-label
        # oracle. Renamed in summary to clarify upper-bound semantics — P1-6.)
        print("    C0: Centralized KIRBy (noisy labels)...", end=' ')
        try:
            train_reps_all = {r: arr[train_idx] for r, arr in reps.items()}
            train_augs = ({k: v[train_idx] for k, v in augs.items()}
                          if augs else None)
            test_augs = ({k: v[test_idx] for k, v in augs.items()}
                         if augs else None)
            X_h_train, fi_central = create_hybrid(
                train_reps_all, train_labels_noisy,
                allocation_method=_alloc,
                importance_method=_imp,
                internal_evaluator=_int_eval,
                augmentations=train_augs,
                augmentation_strategy=aug_cond)
            X_h_test = apply_feature_selection(
                {r: arr[test_idx] for r, arr in reps.items()}, fi_central)
            _aug_info = fi_central.get('augmentation_info') if isinstance(fi_central, dict) else None
            if _aug_info and test_augs:
                _X_test_aug = apply_augmentation_selection(test_augs, _aug_info)
                if _X_test_aug is not None:
                    X_h_test = np.hstack([X_h_test, _X_test_aug])
            clf_c0 = RandomForestClassifier(
                n_estimators=150, max_depth=15,
                class_weight='balanced', random_state=0)
            clf_c0.fit(X_h_train, train_labels_noisy)
            prob_c0 = clf_c0.predict_proba(X_h_test)
            prob_pos_c0 = prob_c0[:, 1] if prob_c0.shape[1] == 2 else None
            c0 = compute_clf_metrics(
                test_labels_clean, clf_c0.predict(X_h_test), prob_pos_c0)
            c0['n_features'] = int(X_h_train.shape[1])
            sigma_results['c0_centralized_kirby'] = c0
            print(f"MCC={c0['mcc']:.3f}")
        except Exception as e:
            sigma_results['c0_centralized_kirby'] = {'error': str(e)}
            print(f"FAILED: {e}")

        # -- Condition 1: Naive pool --
        print("    C1: Naive pool...", end=' ')
        c1 = _condition_naive_pool(
            train_fps, train_labels_noisy, test_fps, test_labels_clean)
        sigma_results['c1_naive_pool'] = c1
        print(f"MCC={c1['mcc']:.3f}")

        # -- Condition 2: FLuID similarity ensemble --
        # Pass train-only cluster data to avoid test leakage in KNN
        print("    C2: FLuID similarity...", end=' ')
        try:
            pred_cls, pred_prob, tw = predict_fluid_similarity_ensemble(
                teacher_models, cluster_data_train, test_fps, K)
            c2 = compute_clf_metrics(test_labels_clean, pred_cls, pred_prob)
            c2['teacher_weights_mean'] = convert_numpy(
                {str(s): float(tw[:, s - 1].mean()) for s in range(1, K + 1)})
            sigma_results['c2_fluid_similarity'] = c2
            print(f"MCC={c2['mcc']:.3f}")
        except Exception as e:
            sigma_results['c2_fluid_similarity'] = {'error': str(e)}
            print(f"FAILED: {e}")

        # -- Condition 3: FLuID quality ensemble --
        # Pass train-only cluster data for cross-val MCC (no test leakage)
        print("    C3: FLuID quality...", end=' ')
        try:
            pred_cls, pred_prob, mcc_weights = \
                predict_fluid_quality_ensemble(
                    teacher_models, cluster_data_train, test_fps, K)
            c3 = compute_clf_metrics(test_labels_clean, pred_cls, pred_prob)
            c3['mcc_weights'] = convert_numpy(mcc_weights)
            sigma_results['c3_fluid_quality'] = c3
            print(f"MCC={c3['mcc']:.3f}")
        except Exception as e:
            sigma_results['c3_fluid_quality'] = {'error': str(e)}
            print(f"FAILED: {e}")

        # -- Condition 4: FLuID published student --
        # cluster_data_noisy[0] is the full training DataFrame with noisy labels
        # (no train_indices needed — it's already training-only)
        print("    C4: FLuID student...", end=' ')
        try:
            pred_cls, pred_prob, n_fed = \
                federate_labels_and_train_student(
                    teacher_models, cluster_data_noisy,
                    cluster_data_noisy[0], K, test_fps)
            c4 = compute_clf_metrics(test_labels_clean, pred_cls, pred_prob)
            c4['n_federated_samples'] = n_fed
            sigma_results['c4_fluid_student'] = c4
            print(f"MCC={c4['mcc']:.3f} (n_fed={n_fed})")
        except Exception as e:
            sigma_results['c4_fluid_student'] = {'error': str(e)}
            print(f"FAILED: {e}")

        # -- Condition 5: KIRBy federated feature selection --
        print("    C5: KIRBy federated FS...", end=' ')
        try:
            # Use training data reps with noisy labels
            train_reps = {r: arr[train_idx] for r, arr in reps.items()}
            test_reps = {r: arr[test_idx] for r, arr in reps.items()}

            train_groups_c5 = (groups[train_idx]
                               if groups is not None else None)
            X_hybrid_train, silo_budgets, silo_scores, silo_fis = \
                federated_feature_selection(
                    train_reps, train_labels_noisy, train_silo_indices,
                    K,
                    allocation_method=_alloc, importance_method=_imp,
                    internal_evaluator=_int_eval,
                    groups=train_groups_c5,
                    task_type=data.get('task_type'))

            # Reconstruct hybrid on test set using same feature selections
            from kirby.hybrid import apply_feature_selection
            hybrid_test_parts = []
            for s in range(1, K + 1):
                fi = silo_fis.get(s)
                if fi is None:
                    continue
                try:
                    X_part = apply_feature_selection(test_reps, fi)
                    hybrid_test_parts.append(X_part)
                except Exception:
                    continue

            if hybrid_test_parts:
                X_hybrid_test = np.hstack(hybrid_test_parts)
                clf = RandomForestClassifier(
                    n_estimators=150, max_depth=15,
                    class_weight='balanced', random_state=0)
                clf.fit(X_hybrid_train, train_labels_noisy)
                pred_cls = clf.predict(X_hybrid_test)
                pred_prob = clf.predict_proba(X_hybrid_test)
                prob_pos = pred_prob[:, 1] if pred_prob.shape[1] == 2 else None
                c5 = compute_clf_metrics(
                    test_labels_clean, pred_cls, prob_pos)
            else:
                c5 = {'error': 'no valid silo selections'}

            c5['silo_budgets'] = convert_numpy(silo_budgets)
            c5['silo_scores'] = convert_numpy(silo_scores)
            sigma_results['c5_kirby_federated'] = c5
            mcc_str = f"MCC={c5['mcc']:.3f}" if 'mcc' in c5 else 'N/A'
            print(f"{mcc_str} budgets={silo_budgets}")

        except Exception as e:
            sigma_results['c5_kirby_federated'] = {'error': str(e)}
            print(f"FAILED: {e}")

        # -- Condition 6: KIRBy + per-silo noise detection --
        # Each silo runs detectors locally, flags noisy samples, retrain
        print("    C6: KIRBy + noise detect...", end=' ')
        try:
            from sklearn.preprocessing import StandardScaler
            from kirby.utils.noise_mitigation_methods import (
                get_mitigation_method)

            train_reps_local = {r: arr[train_idx] for r, arr in reps.items()}

            # Use multiple detectors, take majority vote
            C6_DETECTORS = ['knn_k5', 'prediction_error', 'zscore']
            cleaned_train_mask = np.ones(len(train_idx), dtype=bool)
            detection_results = {}

            for s in range(1, K + 1):
                s_idx = train_silo_indices[s]
                if len(s_idx) < 20:
                    continue
                X_silo = np.hstack(
                    [train_reps_local[r][s_idx]
                     for r in sorted(train_reps_local.keys())])
                X_silo_s = StandardScaler().fit_transform(X_silo)

                # Majority vote across detectors
                vote_count = np.zeros(len(s_idx), dtype=int)
                for det_name in C6_DETECTORS:
                    try:
                        det = get_mitigation_method(det_name)
                        flagged = det.identify_noise(
                            X_silo_s, train_labels_noisy[s_idx])
                        vote_count[flagged] += 1
                    except Exception:
                        pass

                # Flag samples flagged by >= 2 detectors
                flagged_local = np.where(
                    vote_count >= 2)[0]
                flagged_train = s_idx[flagged_local]
                cleaned_train_mask[flagged_train] = False
                detection_results[f'silo_{s}'] = {
                    'n_flagged': int(len(flagged_local)),
                    'n_total': int(len(s_idx)),
                    'detectors_used': C6_DETECTORS,
                }

            # Retrain on cleaned data using KIRBy hybrid features
            # (consistent with C5's feature space for fair comparison)
            n_excluded = int((~cleaned_train_mask).sum())
            clean_idx = np.where(cleaned_train_mask)[0]

            if len(clean_idx) > 20:
                # C5's blueprint was selected on dirty labels; here we
                # rebuild the hybrid on the cleaned subset of training data.
                if 'c5_kirby_federated' in sigma_results and 'error' not in sigma_results['c5_kirby_federated']:
                    try:
                        clean_reps = {r: arr[train_idx][clean_idx]
                                      for r, arr in reps.items()}
                        clean_augs = ({k: v[train_idx][clean_idx]
                                       for k, v in augs.items()}
                                      if augs else None)
                        test_augs_c6 = ({k: v[test_idx] for k, v in augs.items()}
                                        if augs else None)
                        X_c6_tr, fi_c6 = create_hybrid(
                            clean_reps,
                            train_labels_noisy[clean_idx],
                            allocation_method=_alloc,
                            importance_method=_imp,
                            internal_evaluator=_int_eval,
                            augmentations=clean_augs,
                            augmentation_strategy=aug_cond)
                        X_c6_te = apply_feature_selection(
                            {r: arr[test_idx] for r, arr in reps.items()},
                            fi_c6)
                        _aug_info = fi_c6.get('augmentation_info') if isinstance(fi_c6, dict) else None
                        if _aug_info and test_augs_c6:
                            _X_c6_te_aug = apply_augmentation_selection(test_augs_c6, _aug_info)
                            if _X_c6_te_aug is not None:
                                X_c6_te = np.hstack([X_c6_te, _X_c6_te_aug])
                        clf = RandomForestClassifier(
                            n_estimators=150, max_depth=15,
                            class_weight='balanced', random_state=0)
                        clf.fit(X_c6_tr, train_labels_noisy[clean_idx])
                        pred_cls = clf.predict(X_c6_te)
                        pred_prob = clf.predict_proba(X_c6_te)
                        prob_pos = (pred_prob[:, 1]
                                    if pred_prob.shape[1] == 2 else None)
                        c6 = compute_clf_metrics(
                            test_labels_clean, pred_cls, prob_pos)
                    except Exception:
                        # Fallback to Morgan FPs
                        clf = RandomForestClassifier(
                            n_estimators=150, max_depth=15,
                            class_weight='balanced', random_state=0)
                        clf.fit(train_fps[clean_idx],
                                train_labels_noisy[clean_idx])
                        pred_cls = clf.predict(test_fps)
                        pred_prob = clf.predict_proba(test_fps)
                        prob_pos = (pred_prob[:, 1]
                                    if pred_prob.shape[1] == 2 else None)
                        c6 = compute_clf_metrics(
                            test_labels_clean, pred_cls, prob_pos)
                        c6['note'] = 'hybrid failed, used Morgan FPs'
                else:
                    clf = RandomForestClassifier(
                        n_estimators=150, max_depth=15,
                        class_weight='balanced', random_state=0)
                    clf.fit(train_fps[clean_idx],
                            train_labels_noisy[clean_idx])
                    pred_cls = clf.predict(test_fps)
                    pred_prob = clf.predict_proba(test_fps)
                    prob_pos = (pred_prob[:, 1]
                                if pred_prob.shape[1] == 2 else None)
                    c6 = compute_clf_metrics(
                        test_labels_clean, pred_cls, prob_pos)
                    c6['note'] = 'C5 not available, used Morgan FPs'
            else:
                c6 = {'error': 'too few samples after cleaning'}

            c6['n_excluded'] = n_excluded
            c6['detection_results'] = convert_numpy(detection_results)
            sigma_results['c6_kirby_noise_detect'] = c6
            mcc_str = f"MCC={c6['mcc']:.3f}" if 'mcc' in c6 else 'N/A'
            print(f"{mcc_str} (excluded {n_excluded})")

        except Exception as e:
            sigma_results['c6_kirby_noise_detect'] = {'error': str(e)}
            print(f"FAILED: {e}")

        primary_sweep[str(sigma_frac)] = convert_numpy(sigma_results)

    # ── Phase 3: Noise detection analysis ─────────────────────────────────
    print("\n--- Phase 3: Noise detection analysis ---")
    detection_analysis = {}

    # Run at sigma=0.2 (meaningful noise level)
    sigma_det = 0.2
    noise_levels_det = {s: 0.0 for s in range(1, K + 1)}
    noise_levels_det[K] = sigma_det

    train_labels_clean_det = labels_binary[train_idx]
    train_labels_noisy_det, noise_mask_det, _ = inject_noise_per_silo(
        train_labels_clean_det, train_silo_indices,
        noise_levels_det, noise_type='label_flip')

    train_reps_det = {r: arr[train_idx] for r, arr in reps.items()}

    try:
        # Pass NOISY labels so detectors can find the corrupted samples
        det_results = federated_noise_detection(
            train_reps_det, train_labels_noisy_det, train_silo_indices,
            noise_mask_det, K, detector_names=DETECTORS)
        detection_analysis = convert_numpy(det_results)
        print(f"  Centralized detectors: "
              f"{len(det_results.get('centralized', {}))} methods")
        print(f"  Per-silo detectors: "
              f"{len(det_results.get('per_silo', {}))} silos")
    except Exception as e:
        detection_analysis = {'error': str(e)}
        print(f"  FAILED: {e}")

    # ── Phase 4: Importance-noise correlation ──────────────────────────────
    print("\n--- Phase 4: Importance-noise correlation ---")
    importance_correlation = {}

    # Collect silo budgets from the sigma=0.2 run in primary_sweep
    sweep_02 = primary_sweep.get('0.2', {})
    c5_data = sweep_02.get('c5_kirby_federated', {})
    if 'silo_budgets' in c5_data and 'error' not in c5_data:
        silo_noise = [noise_levels_det.get(s, 0.0) for s in range(1, K + 1)]
        silo_budget_vals = [c5_data['silo_budgets'].get(str(s), 0)
                            for s in range(1, K + 1)]
        silo_score_vals = [c5_data.get('silo_scores', {}).get(str(s), 0)
                           for s in range(1, K + 1)]

        if len(set(silo_noise)) > 1:
            r_budget, p_budget = spearmanr(silo_noise, silo_budget_vals)
            r_score, p_score = spearmanr(silo_noise, silo_score_vals)
            importance_correlation = {
                'noise_levels': silo_noise,
                'budgets': silo_budget_vals,
                'scores': silo_score_vals,
                'spearman_budget_r': float(r_budget),
                'spearman_budget_p': float(p_budget),
                'spearman_score_r': float(r_score),
                'spearman_score_p': float(p_score),
            }
            print(f"  Budget-noise correlation: r={r_budget:.3f}, p={p_budget:.3f}")
            print(f"  Score-noise correlation:  r={r_score:.3f}, p={p_score:.3f}")
            if r_budget < 0:
                print("  --> Noisier silo gets LESS budget (implicit detection)")
            else:
                print("  --> No clear implicit noise detection")
        else:
            importance_correlation = {'note': 'all silos have same noise level'}
            print("  Skipped (all silos same noise)")
    else:
        importance_correlation = {'note': 'C5 data not available at sigma=0.2'}
        print("  Skipped (C5 not available)")

    # ── Phase 5: Smooth factor sweep ──────────────────────────────────────
    print("\n--- Phase 5: Smooth factor sweep (sigma=0.2) ---")
    smooth_sweep = {}

    for sf in SMOOTH_FACTORS:
        print(f"  smooth_factor={sf}...", end=' ')
        try:
            # Re-cluster training data with different smooth factor
            train_silo_sf, cd_sf, _ = cluster_training_only(
                train_fluid_df, k=K, smooth_factor=sf)

            # Noise injection (training-relative indices)
            noise_levels_sf = {s: 0.0 for s in range(1, K + 1)}
            noise_levels_sf[K] = 0.2
            tl_noisy, _, _ = inject_noise_per_silo(
                labels_binary[train_idx], train_silo_sf,
                noise_levels_sf, noise_type='label_flip')

            # Rebuild training cluster_data with noisy labels.
            # Preserve smoothed views as in P1-8 fix above.
            train_df_sf = cd_sf[0].copy()
            train_df_sf['CLASS'] = tl_noisy
            train_df_sf['ACTIVITY'] = np.where(
                tl_noisy == 1, 'Active', 'Inactive')
            cd_sf_noisy = [train_df_sf]
            noisy_lookup_sf = train_df_sf[['CLASS', 'ACTIVITY']]
            for s in range(1, K + 1):
                smoothed_view_sf = cd_sf[s].copy()
                shared_sf = smoothed_view_sf.index.intersection(noisy_lookup_sf.index)
                if len(shared_sf) > 0:
                    smoothed_view_sf.loc[shared_sf, 'CLASS'] = (
                        noisy_lookup_sf.loc[shared_sf, 'CLASS'])
                    smoothed_view_sf.loc[shared_sf, 'ACTIVITY'] = (
                        noisy_lookup_sf.loc[shared_sf, 'ACTIVITY'])
                cd_sf_noisy.append(smoothed_view_sf)

            # Train teachers (already training-only)
            from sklearn.ensemble import RandomForestClassifier as RFC
            teacher_models_sf = []
            cd_train_sf = [None]  # index 0 placeholder
            for t in range(0, K + 1):
                cd_t = cd_sf_noisy[t]
                if len(cd_t) < 2:
                    teacher_models_sf.append((None, 0, cd_t))
                    if t > 0:
                        cd_train_sf.append(cd_t)
                    continue
                X_t = np.asarray(list(cd_t.FP))
                y_t = np.asarray(list(cd_t.CLASS))
                clf_t = RFC(n_estimators=150, max_depth=15,
                            class_weight='balanced', random_state=0)
                clf_t.fit(X_t, y_t)
                teacher_models_sf.append((clf_t, len(cd_t), cd_t))
                if t > 0:
                    cd_train_sf.append(cd_t)

            # C2: FLuID similarity (using train-only cluster data)
            pred_cls_sf, pred_prob_sf, _ = predict_fluid_similarity_ensemble(
                teacher_models_sf, cd_train_sf, test_fps, K)
            c2_sf = compute_clf_metrics(
                test_labels_clean, pred_cls_sf, pred_prob_sf)

            # C5: KIRBy federated FS
            train_reps_sf = {r: arr[train_idx] for r, arr in reps.items()}
            test_reps_sf = {r: arr[test_idx] for r, arr in reps.items()}
            train_groups_sf = (groups[train_idx]
                               if groups is not None else None)
            X_h_train, sb, ss, sfis = federated_feature_selection(
                train_reps_sf, tl_noisy, train_silo_sf, K,
                allocation_method=_alloc, importance_method=_imp,
                internal_evaluator=_int_eval,
                groups=train_groups_sf,
                task_type=data.get('task_type'))
            h_test_parts = []
            for s in range(1, K + 1):
                fi = sfis.get(s)
                if fi is None:
                    continue
                try:
                    h_test_parts.append(
                        apply_feature_selection(test_reps_sf, fi))
                except Exception:
                    continue
            if h_test_parts:
                X_h_test = np.hstack(h_test_parts)
                clf_h = RFC(n_estimators=150, max_depth=15,
                            class_weight='balanced', random_state=0)
                clf_h.fit(X_h_train, tl_noisy)
                pred_cls_h = clf_h.predict(X_h_test)
                pred_prob_h = clf_h.predict_proba(X_h_test)
                c5_sf = compute_clf_metrics(
                    test_labels_clean, pred_cls_h,
                    pred_prob_h[:, 1] if pred_prob_h.shape[1] == 2 else None)
            else:
                c5_sf = {'error': 'no valid silo selections'}

            smooth_sweep[str(sf)] = convert_numpy({
                'c2_fluid_similarity': c2_sf,
                'c5_kirby_federated': c5_sf,
            })
            c2_mcc = c2_sf.get('mcc', float('nan'))
            c5_mcc = c5_sf.get('mcc', float('nan'))
            print(f"C2 MCC={c2_mcc:.3f}, C5 MCC={c5_mcc:.3f}")

        except Exception as e:
            smooth_sweep[str(sf)] = {'error': str(e)}
            print(f"FAILED: {e}")

    # ── Phase 6: Assemble results ─────────────────────────────────────────
    results = {
        'experiment': 'fed01_noise_heterogeneity',
        'dataset': dataset_name,
        'config': {
            'k': K,
            'pki_threshold': PKI_THRESHOLD,
            'sigma_fracs': SIGMA_FRACS,
            'smooth_factors': SMOOTH_FACTORS,
            'n_samples': n_samples,
            'n_train': len(train_idx),
            'n_test': len(test_idx),
            'detectors': DETECTORS,
        },
        'primary_sweep': primary_sweep,
        'noise_detection': detection_analysis,
        'importance_noise_correlation': convert_numpy(importance_correlation),
        'smooth_factor_sweep': smooth_sweep,
    }

    # Print summary
    print("\n--- Summary ---")
    print(f"  {'Sigma':<8} {'C1 Pool':<10} {'C2 Sim':<10} {'C3 Qual':<10} "
          f"{'C4 Stud':<10} {'C5 KIRBy':<10} {'C6 Detect':<10}")
    for sigma in SIGMA_FRACS:
        sr = primary_sweep.get(str(sigma), {})
        vals = []
        for cond in ['c1_naive_pool', 'c2_fluid_similarity',
                     'c3_fluid_quality', 'c4_fluid_student',
                     'c5_kirby_federated', 'c6_kirby_noise_detect']:
            c = sr.get(cond, {})
            mcc = c.get('mcc', float('nan'))
            vals.append(f"{mcc:>8.3f}" if not np.isnan(mcc) else f"{'N/A':>8}")
        print(f"  {sigma:<8} {'  '.join(vals)}")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('Fed01: Noise Heterogeneity')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(
        results, args.output, 'fed01_noise_heterogeneity', args.dataset)
