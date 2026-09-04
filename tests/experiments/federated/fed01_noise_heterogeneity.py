#!/usr/bin/env python3
"""Fed01: Noise Heterogeneity Across Silos + Federated Noise Detection.

Tests whether KIRBy's feature-level fusion handles heterogeneous noise across
federated silos better than FLuID's prediction-level ensemble. Uses hERG
binary classification (same endpoint as Lhasa's published hackathon, MCC 0.55).

7 conditions compared at each noise level (numbered 0..6):
    0. Centralized KIRBy hybrid trained on NOISY labels (privacy-violating; matched
       to the federated conditions, NOT a clean-label oracle)
    1. Naive pool (privacy-violating baseline)
    2. FLuID similarity-weighted ensemble (privacy-preserving)
    3. FLuID quality-weighted ensemble (privacy-preserving)
    4. FLuID published student pipeline (privacy-preserving)
    5. KIRBy federated feature selection (privacy-preserving)
    6. Condition 5 with each silo's flagged rows removed first
       (privacy-preserving). Identical call to condition 5 — per-silo feature
       selection, 100 features split across silos, no augmented features — so
       condition 6 minus condition 5 is per-silo detection alone.

Each noise level gets its own recipe search, trained on labels corrupted at
that level and scored on clean ones. Conditions 1-4 run fixed settings and see
no search.

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
import traceback
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
    predict_fluid_similarity_ensemble,
    predict_fluid_quality_ensemble, federate_labels_and_train_student,
    federated_feature_selection, federated_noise_detection,
    stack_reps_for_distance,
    compute_clf_metrics, _resolve_match,
    smoke_grid, smoke_n_folds, smoke_k_silos,
    fit_fed_classifier, make_fed_classifier, fed_to_kirby_model,
    binarise_labels,
)
from kirby.hybrid import create_hybrid, apply_feature_selection, apply_augmentation_selection
from kirby.noise_spec import NoiseSpec


# ── Constants ────────────────────────────────────────────────────────────────
# smoke_grid/smoke_n_folds/smoke_k_silos are no-ops unless KIRBY_SMOKE_N is set.

K = smoke_k_silos(5)               # number of silos/clusters
SIGMA_FRACS = smoke_grid([0, 0.05, 0.1, 0.15, 0.2, 0.3], [0, 0.2])
SMOOTH_FACTORS = smoke_grid([0, 0.01, 0.05, 0.1, 0.2], [0, 0.05])

DETECTORS = smoke_grid([
    'knn_k5', 'knn_k10', 'lof', 'distance_weighted',
    'cv_disagreement', 'bootstrap_ensemble',
    'zscore', 'prediction_error',
], ['knn_k5', 'zscore'])




def _condition_naive_pool(train_fps, train_labels_noisy, test_fps,
                          test_labels_clean, fed_model='rf', fed_seed=0):
    """Condition 1: Merge all silo data, train single model.

    Privacy: VIOLATES (all data pooled). Baseline only.
    """
    clf = fit_fed_classifier(
        fed_model, train_fps, train_labels_noisy, seed=fed_seed)
    pred_probs = clf.predict_proba(test_fps)
    pred_classes = clf.predict(test_fps)
    prob_positive = pred_probs[:, 1] if pred_probs.shape[1] == 2 else None
    return compute_clf_metrics(test_labels_clean, pred_classes, prob_positive)


def run(data, config, output_dir, dataset_name, **kwargs):
    """Fed01: Noise heterogeneity across silos."""
    print("\n" + "=" * 70)
    print("EXPERIMENT FED01: NOISE HETEROGENEITY ACROSS SILOS")
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
    # value (unique values ⊆ {0,1}). Continuous endpoints take their own
    # cutoff from CLASSIFICATION_CUTOFFS in federated_helpers.
    labels_binary, cutoff_note = binarise_labels(
        dataset_name, labels, is_binary=bool(data.get('is_binary')))
    n_active = int(labels_binary.sum())
    print(f"  {dataset_name}: {len(labels)} samples, "
          f"{n_active} active, {len(labels_binary) - n_active} inactive "
          f"({cutoff_note})")

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
    # fed01 is a two-class experiment on every dataset it runs on: it binarises
    # above and every condition trains on labels_binary. get_domain_metric
    # reads the registry, which has caco2 and logd as regression with 'rmse'.
    # That metric ranked the search, and metric_fn scored the silos inside
    # federated_feature_selection, on labels that are 0 and 1 by then.
    from kirby.find_best_hybrid import BUILTIN_METRICS
    metric_name = 'auc'
    metric_fn, lower_is_better = BUILTIN_METRICS[metric_name]

    # ONE SEARCH PER NOISE LEVEL, each trained on labels corrupted at that
    # level and scored on clean ones.
    #
    # Until 2026-09-04 there was a single search, run on the uncorrupted
    # labels, before any noise was injected. The four settings it picked —
    # allocation method, importance method, internal evaluator, augmentation
    # strategy — were then used at every noise level by C0, C5 and C6. So
    # those three conditions entered the sweep having chosen their recipe with
    # the ground truth in view, while C1 to C4 run fixed settings and never see
    # a search at all.
    #
    # find_best_hybrid's label_noise draws one deterministic corrupted copy of
    # the label column, trains its folds on that, and scores on the clean
    # labels. The condition is carried by `tier`, so each level gets its own
    # cache rows rather than serving the clean ones back.
    #
    # The level handed to NoiseSpec is the mean of the per-silo linspace, i.e.
    # sigma_frac / 2. fed01 corrupts silo s at linspace(0, sigma_frac, K)[s],
    # so the training set's overall flip fraction is that mean, weighted by
    # silo size. Silos are clustered per fold and this search runs before the
    # folds, so the unweighted mean is what is available here. The search sees
    # the right AMOUNT of corruption; the way it is spread across silos is a
    # fold-level fact that only C5 and C6 act on.
    def _search_settings(noise_spec):
        try:
            _search, _meta = get_best_search_for_downstream(
                # labels_binary, not labels: the search has to rank recipes on
                # the labels C0, C5 and C6 actually train on.
                reps, labels_binary, groups,
                downstream_model=fed_to_kirby_model(fed_model), budget=100,
                augmentations=augs,
                metric=metric_name,
                task_type='classification',
                domain=data.get('domain', 'molecular'),
                dataset_name=dataset_name,
                # data= is NOT passed. data['precomputed_searches'] is keyed by
                # augmentation strategy alone and holds the clean searches over
                # the registry's own labels — continuous on caco2 and logd — so
                # a hit there answers this request with a regression search.
                # Without data=, the request goes to the on-disk cache, whose
                # identity carries labels_hash and whose tier carries the noise
                # condition.
                data=None,
                label_noise=noise_spec,
            )
            _bp = _meta['best_config']['blueprint'] if _meta else None
        except Exception:
            traceback.print_exc()
            _bp = None
        # Trust the blueprint's actual augmentation_strategy — the pool picked
        # whichever (none/uncapped/capped) won, so any downstream create_hybrid
        # call MUST match that choice. Previously this was hardcoded to
        # 'uncapped' which silently mismatched the search recipe when pool
        # picked 'none' or 'capped'.
        _aug_cond = _bp.augmentation_strategy if _bp else (
            'uncapped' if augs else 'none')
        _a = _bp.allocation_method if _bp else 'performance_weighted'
        _i = _bp.importance_method if _bp else 'treeshap'
        _e = _bp.internal_evaluator if _bp else 'match'
        # Resolve 'match' once here so downstream create_hybrid calls (C0) do
        # NOT silently degrade to 'ridge' inside hybrid.py — those calls lack a
        # `model=` arg so the in-hybrid fallback would pick ridge, which is
        # forbidden (feedback_no_ridge.md). _resolve_match returns 'rf' when no
        # downstream model is known, which is the safe default for fed01's RF
        # final-layer classifiers.
        return _a, _i, _resolve_match(_e, 'rf'), _aug_cond

    settings_by_sigma = {}
    for _sigma in SIGMA_FRACS:
        _mean_flip = float(np.mean(np.linspace(0.0, float(_sigma), K)))
        if _mean_flip <= 0.0:
            _spec = None
        else:
            _spec = NoiseSpec(strategy='uniform',
                              target_effective_noise=_mean_flip,
                              noise_seed=fed_seed)
        settings_by_sigma[_sigma] = _search_settings(_spec)
        _a, _i, _e, _ac = settings_by_sigma[_sigma]
        print(f"  sigma={_sigma}: mean flip fraction {_mean_flip:.4f} — "
              f"alloc={_a} importance={_i} evaluator={_e} aug={_ac}")

    # 5-FOLD CV — scaffold-group-aware. Previous single 80/20 ran ALL
    # conditions × sigma × Phases 3/4/5 on one fold, so the noise contrast
    # was estimated from one draw. We now loop the entire per-fold pipeline
    # (clustering → noise injection → C0..C6 → detection → correlation →
    # smooth sweep) and aggregate mean ± std across folds (P1 fix).
    if groups is None:
        # Hard-fail per project rule (feedback_no_random_splits.md). Silent
        # random KFold fallback inside cv_split is forbidden for fed01.
        raise ValueError(
            "fed01: groups is None — refusing to run 5-fold CV without "
            "scaffold groups (no random splits)."
        )

    N_FOLDS = smoke_n_folds(5)   # 2 under KIRBY_SMOKE_N, else 5
    # Second guard: cv_split also falls back to random KFold when the
    # number of unique groups is below n_splits. That fallback violates
    # feedback_no_random_splits.md just as silently as `groups is None`.
    n_unique_groups = len(np.unique(groups))
    if n_unique_groups < N_FOLDS:
        raise ValueError(
            f"fed01: only {n_unique_groups} unique scaffold groups but "
            f"N_FOLDS={N_FOLDS} — refusing silent random KFold fallback."
        )
    per_fold_primary = []      # list of {sigma_str: {cond: metrics, ...}}
    per_fold_detection = []    # list of detection_analysis dicts
    per_fold_correlation = []  # list of importance_correlation dicts
    per_fold_smooth = []       # list of {sf_str: {...}}
    per_fold_sizes = []        # list of {'n_train': int, 'n_test': int}

    fold_iter = cv_split(groups, N_FOLDS, np.zeros(n_samples), labels_binary,
                         seed=fed_seed)
    for fold_idx, (train_idx, test_idx) in enumerate(fold_iter):
        print(f"\n========== FOLD {fold_idx + 1}/{N_FOLDS} ==========")
        test_labels_clean = labels_binary[test_idx]
        print(f"  Split: {len(train_idx)} train / {len(test_idx)} test")

        # CLUSTER TRAINING DATA ONLY — matching FLuID's cluster_data_space(trainingData)
        # Test molecules never influence K-Means centroids.
        t0 = time.time()
        train_fluid_df = fluid_df_all.iloc[train_idx]
        train_silo_indices, cluster_data, _ = cluster_training_only(
            train_fluid_df, k=K, smooth_factor=0.05, seed=fed_seed)

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
        print("\n  --- Phase 2: Primary noise sweep ---")
        primary_sweep = {}

        for sigma_frac in SIGMA_FRACS:
            print(f"\n  === Sigma={sigma_frac} (noise linspace 0..{sigma_frac} across K={K} silos) ===")

            # This level's own settings, from the search trained on labels
            # corrupted at this level.
            _alloc, _imp, _int_eval, aug_cond = settings_by_sigma[sigma_frac]

            # Inject noise on training labels only using a heterogeneous linspace
            # pattern: silo 1 = 0.0, silo K = sigma_frac, others interpolated.
            # NOTE: this noise vector is SHARED between Phase 2 (which produces
            # C5 silo_budgets) and Phase 4 (which reads `noise_levels_det` to
            # compute the importance-noise Spearman). Phase 4 below uses the
            # exact same linspace(0.0, sigma_det=0.2, K) — they must match for
            # the correlation to be meaningful.
            silo_noise_arr = np.linspace(0.0, sigma_frac, K).tolist()
            noise_levels = {s: float(silo_noise_arr[s - 1])
                            for s in range(1, K + 1)}

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
                clf = fit_fed_classifier(fed_model, X_t, y_t, seed=fed_seed)
                teacher_models.append((clf, len(cd), cd))
                if t > 0:
                    cluster_data_train.append(cd)

            # -- Condition 0: Centralized KIRBy hybrid trained on NOISY labels --
            # NOT a clean-label oracle / upper bound. C0 sees the SAME noisy labels
            # the federated conditions see — it just gets centralized access to all
            # silo features. Treat as "centralized noisy" baseline, not an upper
            # bound. (P1 fix: header docstring was misleading.)
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
                clf_c0 = fit_fed_classifier(
                    fed_model, X_h_train, train_labels_noisy, seed=fed_seed)
                prob_c0 = clf_c0.predict_proba(X_h_test)
                prob_pos_c0 = prob_c0[:, 1] if prob_c0.shape[1] == 2 else None
                c0 = compute_clf_metrics(
                    test_labels_clean, clf_c0.predict(X_h_test), prob_pos_c0)
                c0['n_features'] = int(X_h_train.shape[1])
                sigma_results['c0_centralized_kirby'] = c0
                print(f"MCC={c0['mcc']:.3f}")
            except Exception as e:
                # Record traceback (not just str(e)) so a shape/import/NameError
                # in the create_hybrid path surfaces during the expensive run
                # rather than being reduced to a one-line message (FED-21).
                sigma_results['c0_centralized_kirby'] = {
                    'error': str(e), 'traceback': traceback.format_exc()}
                print(f"FAILED: {e}")
                traceback.print_exc()

            # -- Condition 1: Naive pool --
            print("    C1: Naive pool...", end=' ')
            c1 = _condition_naive_pool(
                train_fps, train_labels_noisy, test_fps, test_labels_clean,
                fed_model=fed_model, fed_seed=fed_seed)
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
                        cluster_data_noisy[0], K, test_fps,
                        algorithm=fed_model, seed=fed_seed)
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
                        downstream_model=fed_to_kirby_model(fed_model),
                        groups=train_groups_c5,
                        task_type='classification',
                        metric_fn=metric_fn, lower_is_better=lower_is_better)

                # Reconstruct hybrid on test set using same feature selections
                # (apply_feature_selection is imported at module level; a
                # redundant local import here made Python treat the name as a
                # function-local everywhere in run(), raising "cannot access
                # local variable 'apply_feature_selection'" in the C0 path).
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
                    clf = fit_fed_classifier(
                        fed_model, X_hybrid_train, train_labels_noisy,
                        seed=fed_seed)
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
                sigma_results['c5_kirby_federated'] = {
                    'error': str(e), 'traceback': traceback.format_exc()}
                print(f"FAILED: {e}")
                traceback.print_exc()

            # -- Condition 6: KIRBy + per-silo noise detection --
            # Each silo runs detectors locally, flags noisy samples, retrain
            print("    C6: KIRBy + noise detect...", end=' ')
            try:
                from sklearn.preprocessing import StandardScaler
                from kirby.utils.noise_mitigation_methods import (
                    get_mitigation_method)

                train_reps_local = {r: arr[train_idx] for r, arr in reps.items()}
                # Sliced here rather than reused from C5: C5 builds test_reps
                # inside its own try, so a C5 that failed early would leave the
                # name unbound.
                test_reps = {r: arr[test_idx] for r, arr in reps.items()}

                # Use multiple detectors, take majority vote
                C6_DETECTORS = ['knn_k5', 'prediction_error', 'zscore']
                cleaned_train_mask = np.ones(len(train_idx), dtype=bool)
                detection_results = {}

                for s in range(1, K + 1):
                    s_idx = train_silo_indices[s]
                    if len(s_idx) < 20:
                        continue
                    # Each representation z-scored on this silo's own rows and
                    # then divided by sqrt(its width), so a 2560-feature protein
                    # embedding and a 200-feature descriptor set count the same
                    # in the distance knn_k5 measures. hstack + one
                    # StandardScaler gave every feature variance 1, which made
                    # the widest representation the one that picked the
                    # neighbours. Already z-scored — do not scale it again, that
                    # cancels the divisor exactly.
                    X_silo_s = stack_reps_for_distance(
                        train_reps_local, rows=s_idx)

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

                # C6 IS C5 WITH THE FLAGGED ROWS REMOVED — same call, same
                # per-silo feature selection, same 100 features split across
                # silos, no augmented features, same downstream model passed.
                # So C6 minus C5 is per-silo detection and nothing else.
                #
                # Until 2026-09-04 this ran ONE create_hybrid on every silo's
                # cleaned rows pooled together — the same call C0 makes — with
                # no budget argument (so create_hybrid's default 100 for all
                # silos, not 100 split across five), with augmented features
                # included, and with no model passed. Four differences at once,
                # so the C6-minus-C5 gap was not detection.
                n_excluded = int((~cleaned_train_mask).sum())
                clean_idx = np.where(cleaned_train_mask)[0]

                if len(clean_idx) > 20:
                    try:
                        # Silo membership, re-expressed as positions in the
                        # cleaned rows. A silo that loses every row drops out.
                        _pos = np.full(len(train_idx), -1, dtype=int)
                        _pos[clean_idx] = np.arange(len(clean_idx))
                        clean_silo_indices = {}
                        for s in range(1, K + 1):
                            mapped = _pos[train_silo_indices[s]]
                            clean_silo_indices[s] = mapped[mapped >= 0]

                        clean_reps = {r: arr[train_idx][clean_idx]
                                      for r, arr in reps.items()}
                        clean_groups_c6 = (groups[train_idx][clean_idx]
                                           if groups is not None else None)
                        X_c6_tr, sb_c6, ss_c6, sfis_c6 = \
                            federated_feature_selection(
                                clean_reps, train_labels_noisy[clean_idx],
                                clean_silo_indices, K,
                                allocation_method=_alloc,
                                importance_method=_imp,
                                internal_evaluator=_int_eval,
                                downstream_model=fed_to_kirby_model(fed_model),
                                groups=clean_groups_c6,
                                task_type='classification',
                                metric_fn=metric_fn,
                                lower_is_better=lower_is_better)

                        c6_test_parts = []
                        for s in range(1, K + 1):
                            fi = sfis_c6.get(s)
                            if fi is None:
                                continue
                            try:
                                c6_test_parts.append(
                                    apply_feature_selection(test_reps, fi))
                            except Exception:
                                continue

                        if c6_test_parts:
                            X_c6_te = np.hstack(c6_test_parts)
                            clf = fit_fed_classifier(
                                fed_model, X_c6_tr,
                                train_labels_noisy[clean_idx], seed=fed_seed)
                            pred_cls = clf.predict(X_c6_te)
                            pred_prob = clf.predict_proba(X_c6_te)
                            prob_pos = (pred_prob[:, 1]
                                        if pred_prob.shape[1] == 2 else None)
                            c6 = compute_clf_metrics(
                                test_labels_clean, pred_cls, prob_pos)
                        else:
                            c6 = {'error': 'no valid silo selections'}
                        c6['silo_budgets'] = convert_numpy(sb_c6)
                        c6['silo_scores'] = convert_numpy(ss_c6)
                    except Exception as e:
                        # No Morgan-fingerprint fallback. It used to write an
                        # MCC into this table under a 'note' key, so a fold
                        # where the hybrid never built read afterwards as a
                        # hybrid result.
                        c6 = {'error': str(e),
                              'traceback': traceback.format_exc()}
                        traceback.print_exc()
                else:
                    c6 = {'error': 'too few samples after cleaning'}

                c6['n_excluded'] = n_excluded
                c6['detection_results'] = convert_numpy(detection_results)
                sigma_results['c6_kirby_noise_detect'] = c6
                mcc_str = f"MCC={c6['mcc']:.3f}" if 'mcc' in c6 else 'N/A'
                print(f"{mcc_str} (excluded {n_excluded})")

            except Exception as e:
                sigma_results['c6_kirby_noise_detect'] = {
                    'error': str(e), 'traceback': traceback.format_exc()}
                print(f"FAILED: {e}")
                traceback.print_exc()

            primary_sweep[str(sigma_frac)] = convert_numpy(sigma_results)

        # ── Phase 3: Noise detection analysis ─────────────────────────────────
        print("\n--- Phase 3: Noise detection analysis ---")
        detection_analysis = {}

        # Spread noise across silos so the Phase 4 importance-noise correlation has
        # actual rank discrimination. Previous pattern was [0,0,0,0,0.2] which
        # produced a degenerate 4-way Spearman tie (P1 fix).
        sigma_det = 0.2
        silo_noise_levels = np.linspace(0.0, sigma_det, K).tolist()
        noise_levels_det = {s: float(silo_noise_levels[s - 1])
                            for s in range(1, K + 1)}

        train_labels_clean_det = labels_binary[train_idx]
        train_labels_noisy_det, noise_mask_det, _ = inject_noise_per_silo(
            train_labels_clean_det, train_silo_indices,
            noise_levels_det, noise_type='label_flip')

        train_reps_det = {r: arr[train_idx] for r, arr in reps.items()}

        try:
            # Pass NOISY labels so detectors can find the corrupted samples
            det_results = federated_noise_detection(
                train_reps_det, train_labels_noisy_det, train_silo_indices,
                noise_mask_det, K, detector_names=DETECTORS,
                groups=(groups[train_idx] if groups is not None else None),
                task_type='classification')
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
        # This phase fixes the noise at 0.2, so it takes 0.2's settings. Under
        # KIRBY_SMOKE_N the grid is [0, 0.2] and 0.2 is still in it; the
        # fallback is the largest level present.
        _sf_sigma = 0.2 if 0.2 in settings_by_sigma else max(settings_by_sigma)
        _alloc, _imp, _int_eval, aug_cond = settings_by_sigma[_sf_sigma]

        for sf in SMOOTH_FACTORS:
            print(f"  smooth_factor={sf}...", end=' ')
            try:
                # Re-cluster training data with different smooth factor
                train_silo_sf, cd_sf, _ = cluster_training_only(
                    train_fluid_df, k=K, smooth_factor=sf, seed=fed_seed)

                # Noise injection (training-relative indices). Use the same
                # linspace(0.0, 0.2, K) heterogeneous pattern as Phase 2/4 so
                # the smooth-factor sweep is comparable; the old
                # {1:0, 2:0, 3:0, 4:0, 5:0.2} pattern only noised silo K and
                # produced degenerate behaviour (P1 fix).
                silo_noise_sf = np.linspace(0.0, 0.2, K)
                noise_levels_sf = {s: float(silo_noise_sf[s - 1])
                                   for s in range(1, K + 1)}
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
                    clf_t = fit_fed_classifier(
                        fed_model, X_t, y_t, seed=fed_seed)
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
                    downstream_model=fed_to_kirby_model(fed_model),
                    groups=train_groups_sf,
                    task_type='classification',
                    metric_fn=metric_fn, lower_is_better=lower_is_better)
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
                    clf_h = fit_fed_classifier(
                        fed_model, X_h_train, tl_noisy, seed=fed_seed)
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

        # Accumulate this fold's results.
        per_fold_primary.append(primary_sweep)
        per_fold_detection.append(detection_analysis)
        per_fold_correlation.append(convert_numpy(importance_correlation))
        per_fold_smooth.append(smooth_sweep)
        per_fold_sizes.append({'n_train': int(len(train_idx)),
                               'n_test': int(len(test_idx))})

    # ── Aggregate across folds (mean ± std on numeric leaves) ─────────────
    def _aggregate(fold_dicts):
        """Recursively mean/std numeric leaves across a list of nested dicts.

        Non-numeric leaves: take the first fold's value as representative.
        Lists of numerics: aggregate element-wise. Mixed/string lists: pass
        through from fold 0. Missing keys in some folds are skipped.
        """
        if not fold_dicts:
            return {}
        # Strip out folds that errored out at this level
        first = fold_dicts[0]
        if isinstance(first, dict):
            keys = set()
            for fd in fold_dicts:
                if isinstance(fd, dict):
                    keys.update(fd.keys())
            out = {}
            for k in keys:
                sub = [fd[k] for fd in fold_dicts
                       if isinstance(fd, dict) and k in fd]
                out[k] = _aggregate(sub)
            return out
        # Numeric scalar?
        numeric = []
        for v in fold_dicts:
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)) and not (
                isinstance(v, float) and np.isnan(v)
            ):
                numeric.append(float(v))
        # Accept any non-NaN subset (previous strict-equality check silently
        # fell through to `return first` when even one fold was NaN, dropping
        # all other folds). Use nanmean/nanstd over the kept values.
        non_bool = [v for v in fold_dicts if not isinstance(v, bool)]
        if numeric and non_bool and all(
            isinstance(v, (int, float)) for v in non_bool
        ):
            arr = np.asarray(numeric, dtype=float)
            return {'mean': float(np.nanmean(arr)),
                    'std': float(np.nanstd(arr, ddof=0)),
                    'n_folds': int(len(arr)),
                    'per_fold': numeric}
        # List of numerics (e.g. silo budgets)?
        if isinstance(first, list) and all(
            isinstance(v, list) and len(v) == len(first) for v in fold_dicts
        ) and all(
            isinstance(x, (int, float)) for v in fold_dicts for x in v
        ):
            arr = np.asarray(fold_dicts, dtype=float)
            return {'mean': arr.mean(axis=0).tolist(),
                    'std': arr.std(axis=0, ddof=0).tolist(),
                    'n_folds': int(arr.shape[0])}
        # Fallback: keep first fold's value as representative.
        return first

    primary_agg = _aggregate(per_fold_primary)
    detection_agg = _aggregate(per_fold_detection)
    correlation_agg = _aggregate(per_fold_correlation)
    smooth_agg = _aggregate(per_fold_smooth)

    # ── Phase 6: Assemble results ─────────────────────────────────────────
    results = {
        'experiment': 'fed01_noise_heterogeneity',
        'dataset': dataset_name,
        'config': {
            'k': K,
            'classification_cutoff': cutoff_note,
            'sigma_fracs': SIGMA_FRACS,
            'smooth_factors': SMOOTH_FACTORS,
            'n_samples': n_samples,
            'n_folds': N_FOLDS,
            'fold_sizes': per_fold_sizes,
            'detectors': DETECTORS,
        },
        'primary_sweep': primary_agg,
        'noise_detection': detection_agg,
        'importance_noise_correlation': correlation_agg,
        'smooth_factor_sweep': smooth_agg,
        'per_fold': {
            'primary_sweep': per_fold_primary,
            'noise_detection': per_fold_detection,
            'importance_noise_correlation': per_fold_correlation,
            'smooth_factor_sweep': per_fold_smooth,
        },
    }

    # Print summary (mean ± std across folds, MCC only)
    print("\n--- Summary (mean ± std across folds) ---")
    print(f"  {'Sigma':<8} {'C1 Pool':<16} {'C2 Sim':<16} {'C3 Qual':<16} "
          f"{'C4 Stud':<16} {'C5 KIRBy':<16} {'C6 Detect':<16}")
    for sigma in SIGMA_FRACS:
        sr = primary_agg.get(str(sigma), {})
        vals = []
        for cond in ['c1_naive_pool', 'c2_fluid_similarity',
                     'c3_fluid_quality', 'c4_fluid_student',
                     'c5_kirby_federated', 'c6_kirby_noise_detect']:
            mcc_agg = sr.get(cond, {}).get('mcc', {}) if isinstance(
                sr.get(cond), dict) else {}
            if isinstance(mcc_agg, dict) and 'mean' in mcc_agg:
                vals.append(f"{mcc_agg['mean']:>6.3f}±{mcc_agg['std']:.3f}")
            else:
                vals.append(f"{'N/A':>14}")
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
