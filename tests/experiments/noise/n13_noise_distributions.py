#!/usr/bin/env python3
"""N13: Alternative Noise Distributions.

Paper: Noise
Tests whether methods that work for Gaussian noise generalize to other
distributions: uniform, multiplicative, heavy-tailed, and partial corruption.

Two evaluation tracks:
  Track A: ALL 12 LNL methods from the registry (not a pre-hybrid shortlist)
           (data-level, training-loop, GP) — retention metric
  Track B: Selected estimators covering each family tested in N11
           (kNN-based, GP-based, gradient-boosted) — recovery metric

Usage:
    python n13_noise_distributions.py --dataset logd --mode intermediary
    python n13_noise_distributions.py --dataset qm9 --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_model_factories,
    DOWNSTREAM_MODELS_NOISE,
    get_precomputed_search, get_best_search_for_downstream, extract_top3_base_reps,
)
from kirby.hybrid import create_hybrid, apply_feature_selection, apply_augmentation_selection
from kirby.utils.lnl_methods import (
    get_lnl_method, _standard_dnn_train_predict, LNL_REGISTRY)
from kirby.utils.noise_estimation import get_noise_estimator, weights_from_scores


NOISE_TYPES = ['gaussian', 'uniform', 'multiplicative', 'heavy_tailed', 'partial']
STRENGTHS = [0.1, 0.3, 0.6]
PARTIAL_FRACTIONS = [0.1, 0.3, 0.5]  # for partial noise type
# 'partial' noise sweeps PARTIAL_FRACTIONS at this FIXED strength (the other
# noise types sweep STRENGTHS instead). See the 'partial_axis' note in results.
PARTIAL_STRENGTH = 0.3

# Multi-seed noise injection: aggregate scores across NOISE_SEEDS to reduce
# dependence on a single noise realization. Per-fold injection still uses
# distinct seeds (combined with fold index below) so each (fold, seed) draw
# is unique.
NOISE_SEEDS = [0, 1, 2]

# Test the FULL LNL method registry (all 12), NOT a pre-selected shortlist.
# The previous 4-method list ('dividemix', 'small_loss', 'uncertainty_curriculum',
# 'heteroscedastic_gp') was the winner set from the OLD pre-hybrid confirmation
# report. But n09's whole question is whether hybrids RERANK the methods — so
# pre-filtering n13 to a pre-hybrid shortlist would bias the cross-distribution
# robustness result. Testing all 12 keeps n13 self-contained (no dependency on
# n09's output) at ~3x the track-A training cost. Regression-only methods are
# still gated out for classification datasets below (REGRESSION_ONLY_LNL).
LNL_METHODS = list(LNL_REGISTRY.keys())
ESTIMATORS = ['local_variance_ratio', 'gaussian_process']
# ngboost is optional
try:
    get_noise_estimator('ngboost')
    ESTIMATORS.append('ngboost')
except Exception:
    pass

LNL_EPOCHS = 50  # reduced for speed

# Downstream models whose fit() does NOT accept a `sample_weight` argument, so
# they cannot participate in the Track-B sample-weight remediation track. Seeded
# with 'mlp' (sklearn MLPRegressor/MLPClassifier reject sample_weight); the loop
# below auto-adds any other model that raises TypeError on a weighted fit. Rows
# for these models are intentionally absent from track_b (not recorded as a
# zero-recovery result) and the names are surfaced in results['track_b_unweighted_models'].
UNWEIGHTED_TRACK_B_MODELS = {'mlp'}


def _inject_noise(y, noise_type, strength, seed=42, partial_fraction=0.3,
                  is_clf=False, return_corrupt_mask=False):
    """Inject noise of the specified distribution.

    Args:
        y: Clean labels.
        noise_type: One of NOISE_TYPES.
        strength: Noise magnitude (sigma for Gaussian, width for uniform, etc.).
        seed: Random seed.
        partial_fraction: Fraction of samples to corrupt (for 'partial' type).
        is_clf: If True, use the DivideMix-style label flipping models
            described below — strength now has a meaning for classification.
        return_corrupt_mask: If True, also return a boolean mask of the samples
            whose label actually changed (the ground-truth corruption mask).
            For 'partial' noise this is the genuine corrupted subset; for the
            other distributions every sample is perturbed (continuous Gaussian/
            uniform/etc draws), so the mask is all-True for regression and the
            flip mask for classification. Previously this mask was computed
            locally and discarded, so downstream detection/recovery tracks had
            no ground truth for "did the estimator find the corrupted samples".

    Returns:
        y_noisy: Noisy labels (same shape as y), or
        (y_noisy, corrupt_mask) if return_corrupt_mask is True.
    """
    rng = np.random.RandomState(seed)
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    corrupt_mask = np.zeros(n, dtype=bool)

    if is_clf:
        # DivideMix-style label-noise models (Li et al. 2020). Each noise
        # distribution maps onto a different per-sample flip probability and
        # destination-class distribution:
        #   gaussian / uniform: symmetric noise — flip with prob `strength`,
        #       destination is a uniformly random OTHER class.
        #   multiplicative: same symmetric model — kept for backwards
        #       compatibility with the regression naming.
        #   heavy_tailed: pair-flip / asymmetric noise — flip with prob
        #       `strength`, destination is the class numerically adjacent
        #       (class i → class (i + 1) mod n_classes). This is DivideMix's
        #       'asymmetric' setting and represents systematic confusion
        #       between similar classes.
        #   partial: flip a `partial_fraction` subset of samples with
        #       per-sample probability `strength`. Models "only some samples
        #       are corrupted but those corruptions are confident."
        # Strength controls the off-diagonal mass of the implied confusion
        # matrix; the previous version silently ignored strength for clf
        # and uniformly flipped a fixed fraction of labels.
        classes = np.unique(y)
        n_classes = len(classes)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_idx = np.array([class_to_idx[c] for c in y], dtype=np.int64)
        y_noisy_idx = y_idx.copy()

        if noise_type == 'partial':
            # Pick the corrupted subset; flip each with prob `strength`
            subset_size = max(1, int(n * partial_fraction))
            subset = rng.choice(n, subset_size, replace=False)
            for i in subset:
                if rng.random() < strength:
                    if n_classes > 1:
                        # Pick a different class uniformly
                        choices = [c for c in range(n_classes) if c != y_idx[i]]
                        y_noisy_idx[i] = rng.choice(choices)
        elif noise_type == 'heavy_tailed':
            # Pair-flip / asymmetric: i -> (i + 1) mod n_classes
            flip_mask = rng.random(n) < strength
            y_noisy_idx[flip_mask] = (y_idx[flip_mask] + 1) % n_classes
        else:
            # gaussian / uniform / multiplicative: symmetric uniform-other
            flip_mask = rng.random(n) < strength
            for i in np.where(flip_mask)[0]:
                if n_classes > 1:
                    choices = [c for c in range(n_classes) if c != y_idx[i]]
                    y_noisy_idx[i] = rng.choice(choices)

        y_noisy = classes[y_noisy_idx]
        # Ground-truth corruption mask: labels that actually changed.
        corrupt_mask = (y_noisy_idx != y_idx)
        y_noisy = y_noisy.astype(y.dtype if y.dtype != np.float64 else np.float32)
        if return_corrupt_mask:
            return y_noisy, corrupt_mask
        return y_noisy

    y_range = float(np.max(y) - np.min(y))

    if noise_type == 'gaussian':
        noise = rng.normal(0, strength * y_range, n)
        corrupt_mask = np.ones(n, dtype=bool)  # all samples perturbed
        y_noisy = (y + noise).astype(np.float32)

    elif noise_type == 'uniform':
        half_width = strength * y_range
        noise = rng.uniform(-half_width, half_width, n)
        corrupt_mask = np.ones(n, dtype=bool)
        y_noisy = (y + noise).astype(np.float32)

    elif noise_type == 'multiplicative':
        # Relative noise: y_noisy = y * (1 + N(0, sigma))
        multiplier = 1 + rng.normal(0, strength, n)
        corrupt_mask = np.ones(n, dtype=bool)
        y_noisy = (y * multiplier).astype(np.float32)

    elif noise_type == 'heavy_tailed':
        # Student-t with df=3, scaled to match Gaussian sigma
        t_std = np.sqrt(3 / (3 - 2))  # theoretical std of t(df=3)
        raw = rng.standard_t(df=3, size=n)
        noise = raw * (strength * y_range / t_std)
        corrupt_mask = np.ones(n, dtype=bool)
        y_noisy = (y + noise).astype(np.float32)

    elif noise_type == 'partial':
        # Gaussian noise on only a fraction of samples
        n_corrupt = max(1, int(n * partial_fraction))
        corrupt_idx = rng.choice(n, n_corrupt, replace=False)
        noise = np.zeros(n)
        noise[corrupt_idx] = rng.normal(0, strength * y_range, n_corrupt)
        corrupt_mask[corrupt_idx] = True  # only the corrupted subset
        y_noisy = (y + noise).astype(np.float32)

    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    if return_corrupt_mask:
        return y_noisy, corrupt_mask
    return y_noisy


def _build_spaces(train_reps, y_train, augmentations, bp, base_rep_name):
    """Build 2 spaces: base rep + hybrid.

    Args:
        train_reps: {rep_name: X_train} already sliced to fold.
        y_train: Training labels for this fold.
        augmentations: {aug_name: X_train} already sliced, or None.
        bp: Blueprint from get_precomputed_search.
        base_rep_name: Name of base representation.

    Returns (spaces, feature_info_or_None, scaler_hybrid_or_None).
    feature_info can be used to apply hybrid selection to val reps without
    a second create_hybrid call.
    """
    spaces = {}
    feature_info = None
    scaler_h = None

    X_base = train_reps[base_rep_name]
    sc = StandardScaler()
    spaces[base_rep_name] = sc.fit_transform(X_base)

    try:
        X_h, feature_info = create_hybrid(
            train_reps, y_train,
            allocation_method=bp.allocation_method,
            importance_method=bp.importance_method,
            internal_evaluator=bp.internal_evaluator,
            budget=bp.budget,
            augmentations=augmentations,
            augmentation_strategy=bp.augmentation_strategy,
        )
        scaler_h = StandardScaler()
        spaces['hybrid'] = scaler_h.fit_transform(X_h)
    except Exception as e:
        print(f"      hybrid failed: {e}")

    return spaces, feature_info, scaler_h


def run(data, config, output_dir, dataset_name, **kwargs):
    """N13: Alternative noise distributions."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N13: ALTERNATIVE NOISE DISTRIBUTIONS")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    augmentations = data.get('augmentations')
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    model_factories = get_model_factories(
        DOWNSTREAM_MODELS_NOISE, task_type=task_type,
        n_samples=len(labels))
    print(f"  Models: {list(model_factories.keys())}")
    # SVM internally caps training data at _max_samples=10000. On big
    # datasets (e.g. qm9 at 10K, kiba at 30K+) this triggers SVM-only and
    # silently makes the SVM column an apples-to-oranges comparison against
    # the other models that see the full data. Surface the situation
    # explicitly when SVM is in the set and the dataset is at risk.
    if 'svm' in model_factories and len(labels) > 10000:
        print(f"  WARNING: dataset has {len(labels)} samples — SVM will "
              f"internally cap training data to 10000, so SVM rows below "
              f"are NOT comparable head-to-head with the other models. "
              f"Consider dropping 'svm' from DOWNSTREAM_MODELS_NOISE for "
              f"this dataset.")

    sample_X = next(iter(reps.values()))

    # Find base rep
    base_rep_name = None
    for candidate in ['ecfp4', 'pdv', 'blosum']:
        if candidate in reps:
            base_rep_name = candidate
            break
    if base_rep_name is None:
        base_rep_name = sorted(reps.keys())[0]

    print(f"  Dataset: {dataset_name} ({len(labels)} samples)")
    print(f"  Metric: {metric_name}")
    # Filter out regression-only methods for classification
    # Canonical regression-only set lives in lnl_methods.py; mirroring it
    # locally previously missed evidential_kernel and would silently crash on
    # clf datasets (NIG distribution is regression-only).
    from kirby.utils.lnl_methods import REGRESSION_ONLY_LNL
    lnl_methods = [m for m in LNL_METHODS if not (is_clf and m in REGRESSION_ONLY_LNL)]
    estimators = [e for e in ESTIMATORS if not (is_clf and e == 'gaussian_process')]

    print(f"  Base rep: {base_rep_name}")
    print(f"  Noise types: {NOISE_TYPES}")
    print(f"  LNL methods: {lnl_methods}")
    print(f"  Estimators: {estimators}")
    print(f"  Folds: {n_folds}")

    # Get blueprint ONCE from pool helper (uses clean labels)
    search, meta = get_best_search_for_downstream(
        reps, labels, groups,
        downstream_model='rf', budget=None,
        augmentations=augmentations,
        speed=config.get('search_speed', 'full'),
        dataset_name=dataset_name, task_type=task_type,
        data=data,
    )
    # Always define `top3` up front so the results dict below never trips a
    # NameError when search is None (the old code only set it inside the
    # success branch).
    top3 = None
    if search is not None:
        top3 = extract_top3_base_reps(search)
        if top3:
            print(f"  Top-3 base reps (ranked by RF baseline): "
                  f"{', '.join(f'{n}={s:.4f}' for n, s in top3)}")
    bp = meta['best_config']['blueprint'] if meta else None
    if bp is None:
        print("  WARNING: No blueprint found, skipping")
        return {'error': 'no_blueprint', 'metric': metric_name}

    # Align the blueprint's stored aug_strategy with the pool slot it actually
    # came from (cache-collision guard, see project_cache_collision_fix). Gate
    # the augmentations passed downstream so 'none' really means no aug cols.
    aug_condition = meta.get('aug_condition', 'uncapped') if meta else 'uncapped'
    if getattr(bp, 'augmentation_strategy', None) != aug_condition:
        try:
            bp.augmentation_strategy = aug_condition
        except Exception:
            pass
    use_augmentations = augmentations if aug_condition != 'none' else None

    # Reset the unweighted-model registry to its seed for this run (module-level
    # mutable set persists across run() calls in a single process).
    global UNWEIGHTED_TRACK_B_MODELS
    UNWEIGHTED_TRACK_B_MODELS = {'mlp'}

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'noise_types': NOISE_TYPES,
        'strengths': STRENGTHS,
        # AXIS NOTE: every noise type except 'partial' sweeps `strength`
        # (STRENGTHS). 'partial' instead holds strength fixed at
        # PARTIAL_STRENGTH and sweeps the corrupted-subset fraction
        # (partial_fractions). The two axes are NOT comparable cell-for-cell —
        # downstream readers must treat the partial track as a fraction sweep,
        # not a strength sweep. Strength keys for partial are encoded as
        # '<strength>_frac<fraction>' to make this explicit.
        'partial_fractions': PARTIAL_FRACTIONS,
        'partial_strength': PARTIAL_STRENGTH,
        'partial_axis': 'partial_fraction (strength fixed at PARTIAL_STRENGTH)',
        'lnl_methods': lnl_methods,
        'estimators': estimators,
        'top3_base_reps': [(n, s) for n, s in top3] if top3 else [],
        # Track-B sample-weight remediation excludes these models because their
        # fit() ignores sample_weight; their rows are absent from track_b
        # rather than masquerading as zero-recovery noisy-baseline ties.
        'track_b_unweighted_models': sorted(UNWEIGHTED_TRACK_B_MODELS),
        'per_noise_type': {},
    }

    # ── Cache fold splits and label-independent eval features ──
    clean_scores_by_model = {}
    # Track A is DNN-based (all LNL methods use _standard_dnn_train_predict
    # internally). Use a dedicated DNN clean baseline for Track A retention
    # so we're comparing DNN-with-LNL vs DNN-without-LNL, not against the
    # RF baseline used by Track B and noisy-baseline reporting.
    clean_scores_dnn = []
    fold_data = []

    for train_idx, val_idx in cv_split(groups, n_folds, sample_X,
                                        labels, mode=mode):
        y_train_clean = labels[train_idx]
        y_val = labels[val_idx]

        train_reps_fold = {k: v[train_idx] for k, v in reps.items()}
        val_reps_fold = {k: v[val_idx] for k, v in reps.items()}

        # Base rep eval features (independent of labels)
        X_eval_train = reps[base_rep_name][train_idx]
        X_eval_val = reps[base_rep_name][val_idx]
        sc_eval = StandardScaler()
        X_eval_train_s = sc_eval.fit_transform(X_eval_train)
        X_eval_val_s = sc_eval.transform(X_eval_val)

        # Clean baseline per model (RF/xgboost/etc — for Track B reference)
        for mname, mfactory in model_factories.items():
            if mname not in clean_scores_by_model:
                clean_scores_by_model[mname] = []
            try:
                m = mfactory()
                m.fit(X_eval_train_s, y_train_clean)
                clean_scores_by_model[mname].append(
                    float(metric_fn(y_val, m.predict(X_eval_val_s))))
            except Exception:
                pass

        # DNN clean baseline (matched-evaluator anchor for Track A)
        try:
            dnn_pred = _standard_dnn_train_predict(
                X_eval_train_s, y_train_clean, X_eval_val_s,
                epochs=LNL_EPOCHS, patience=10, random_state=42)
            clean_scores_dnn.append(float(metric_fn(y_val, dnn_pred)))
        except Exception:
            pass

        # Slice augmentations through the aug-condition-aware shim so 'none'
        # pool slots really propagate no aug columns.
        train_augs = {k: v[train_idx] for k, v in use_augmentations.items()} if use_augmentations else None
        # val_augs was missing — the Track A hybrid path below tries to read
        # fd.get('val_augs'), couldn't find it, dropped the augmentation
        # columns from the val feature matrix, and scaler_h.transform then
        # failed silently (column-count mismatch). Hybrid Track A was thus
        # skipped on every augmentation dataset.
        val_augs = {k: v[val_idx] for k, v in use_augmentations.items()} if use_augmentations else None

        fold_data.append({
            'train_idx': train_idx,
            'val_idx': val_idx,
            'y_train_clean': y_train_clean,
            'y_val': y_val,
            'train_reps_fold': train_reps_fold,
            'val_reps_fold': val_reps_fold,
            'train_augs': train_augs,
            'val_augs': val_augs,
            'X_eval_train_s': X_eval_train_s,
            'X_eval_val_s': X_eval_val_s,
        })

    clean_mean_by_model = {mn: float(np.mean(s)) if s else float('nan')
                           for mn, s in clean_scores_by_model.items()}
    # Primary clean score for top-level reporting still uses RF; Track A
    # uses the DNN-specific clean baseline below for matched-evaluator
    # retention math.
    clean_mean = clean_mean_by_model.get('rf', float('nan'))
    clean_mean_dnn = float(np.mean(clean_scores_dnn)) if clean_scores_dnn else float('nan')

    # ── Main loop: noise type × strength ──
    for noise_type in NOISE_TYPES:
        print(f"\n  === {noise_type} ===")
        t_nt = time.time()
        nt_result = {'per_strength': {}}

        # Determine strength variants
        if noise_type == 'partial':
            # Fixed strength, sweep the corrupted-subset fraction. NOT a
            # strength sweep — see 'partial_axis' note in results.
            strength_variants = [(PARTIAL_STRENGTH, f) for f in PARTIAL_FRACTIONS]
        else:
            strength_variants = [(s, None) for s in STRENGTHS]

        for strength, partial_frac in strength_variants:
            strength_key = str(strength) if partial_frac is None else f'{strength}_frac{partial_frac}'

            noisy_baseline_by_model = {mn: [] for mn in model_factories}
            # {space: {method: [scores]}} for track A
            track_a_acc = {}
            # {space: {estimator: {model: [scores]}}} for track B
            track_b_acc = {}

            for fi, fd in enumerate(fold_data):
                kw = {}
                if noise_type == 'partial' and partial_frac is not None:
                    kw['partial_fraction'] = partial_frac

                for noise_seed in NOISE_SEEDS:
                    # Per-(fold, seed) noise realization
                    _seed_combined = 42 + fi * 1000 + noise_seed
                    y_noisy = _inject_noise(fd['y_train_clean'], noise_type,
                                             strength, seed=_seed_combined,
                                             is_clf=is_clf, **kw)

                    # Noisy baseline per model
                    for mname, mfactory in model_factories.items():
                        try:
                            m = mfactory()
                            m.fit(fd['X_eval_train_s'], y_noisy)
                            noisy_baseline_by_model[mname].append(
                                float(metric_fn(fd['y_val'], m.predict(fd['X_eval_val_s']))))
                        except Exception:
                            pass

                    # Build spaces from NOISY labels (realistic — no oracle leak)
                    spaces, feature_info, scaler_h = _build_spaces(
                        fd['train_reps_fold'], y_noisy, fd['train_augs'],
                        bp, base_rep_name)

                    # Build hybrid val features using feature_info (no second create_hybrid)
                    hybrid_val_s = None
                    if 'hybrid' in spaces and feature_info is not None and scaler_h is not None:
                        try:
                            X_h_val = apply_feature_selection(fd['val_reps_fold'], feature_info)
                            aug_info = feature_info.get('augmentation_info')
                            if aug_info and fd.get('val_augs'):
                                X_aug_val = apply_augmentation_selection(fd['val_augs'], aug_info)
                                if X_aug_val is not None:
                                    X_h_val = np.hstack([X_h_val, X_aug_val])
                            hybrid_val_s = scaler_h.transform(X_h_val)
                        except Exception:
                            pass

                    # ── Track A: LNL methods ──
                    for space_name in [base_rep_name, 'hybrid']:
                        if space_name not in spaces:
                            continue
                        X_train_s = spaces[space_name]

                        if space_name == 'hybrid' and hybrid_val_s is not None:
                            X_val_s = hybrid_val_s
                        elif space_name == base_rep_name:
                            X_val_s = fd['X_eval_val_s']
                        else:
                            continue

                        if space_name not in track_a_acc:
                            track_a_acc[space_name] = {}

                        for method_name in lnl_methods:
                            if method_name not in track_a_acc[space_name]:
                                track_a_acc[space_name][method_name] = []
                            try:
                                m = get_lnl_method(
                                    method_name, epochs=LNL_EPOCHS,
                                    patience=10, random_state=_seed_combined,
                                    is_clf=is_clf,
                                    classes_=(np.unique(labels) if is_clf else None))
                                res = m.train_and_predict(X_train_s, y_noisy,
                                                          X_val_s, fd['y_val'])
                                score = float(metric_fn(fd['y_val'], res['y_pred']))
                                track_a_acc[space_name][method_name].append(score)
                            except Exception:
                                pass

                    # ── Track B: Estimation + remediation ──
                    # Asymmetry by design: the noise-detection step iterates
                    # over both base and hybrid spaces (so the estimator sees
                    # different geometry), but the model that consumes the
                    # resulting sample weights ALWAYS trains on the base rep
                    # `fd['X_eval_train_s']`. That makes Track B specifically
                    # ask "does hybrid-derived noise detection improve base-
                    # rep training?", distinct from Track A's "does hybrid
                    # feature space help LNL methods that train on hybrid?".
                    # Recovery denominator (clean/noisy baselines) is also
                    # base-rep, so the comparison is apples-to-apples.
                    for space_name in [base_rep_name, 'hybrid']:
                        if space_name not in spaces:
                            continue
                        X_space = spaces[space_name]

                        if space_name not in track_b_acc:
                            track_b_acc[space_name] = {}

                        for est_name in estimators:
                            if est_name not in track_b_acc[space_name]:
                                track_b_acc[space_name][est_name] = {}
                            try:
                                est = get_noise_estimator(est_name)
                                noise_scores = est.estimate(X_space, y_noisy)
                                weights = weights_from_scores(noise_scores, strength=0.1,
                                                              method='exponential')
                                for mname, mfactory in model_factories.items():
                                    if mname not in track_b_acc[space_name][est_name]:
                                        track_b_acc[space_name][est_name][mname] = []
                                    # Track B is a SAMPLE-WEIGHT remediation track:
                                    # the only thing distinguishing it from the
                                    # noisy baseline is the per-sample reliability
                                    # weight. Models whose fit() does NOT accept
                                    # `sample_weight` (sklearn MLPRegressor /
                                    # MLPClassifier) cannot be remediated this
                                    # way — a fallback unweighted fit would
                                    # reproduce the noisy baseline EXACTLY
                                    # (recovery == 0 by construction) and silently
                                    # inflate "no recovery" counts. Skip such
                                    # models entirely and record them so the
                                    # aggregation can mark them as N/A rather than
                                    # as a genuine zero-recovery result.
                                    if mname in UNWEIGHTED_TRACK_B_MODELS:
                                        continue
                                    try:
                                        m = mfactory()
                                        try:
                                            m.fit(fd['X_eval_train_s'], y_noisy,
                                                  sample_weight=weights)
                                        except TypeError:
                                            # Model does not support sample_weight:
                                            # it is not remediable in Track B.
                                            # Remember it and skip — do NOT record
                                            # an unweighted score masquerading as
                                            # a remediated one.
                                            UNWEIGHTED_TRACK_B_MODELS.add(mname)
                                            continue
                                        score = float(metric_fn(fd['y_val'],
                                                                 m.predict(fd['X_eval_val_s'])))
                                        track_b_acc[space_name][est_name][mname].append(score)
                                    except Exception:
                                        pass
                            except Exception:
                                pass

            # Aggregate per model
            noisy_mean_by_model = {mn: float(np.mean(s)) if s else float('nan')
                                   for mn, s in noisy_baseline_by_model.items()}
            noisy_mean = noisy_mean_by_model.get('rf', float('nan'))
            # Top-level gap is RF-anchored (Track B uses per-model gaps below;
            # Track A uses the DNN-specific gap). Kept here for the JSON
            # output's headline 'noise_gap' field.
            noise_gap_rf = clean_mean - noisy_mean

            strength_result = {
                'noisy_score_by_model': noisy_mean_by_model,
                'clean_score_by_model': clean_mean_by_model,
                'noisy_score': noisy_mean,
                'clean_score': clean_mean,
                'noise_gap_rf': noise_gap_rf,
                'track_a': {},
                'track_b': {},
            }

            # Track A aggregation (DNN-specific — retention is anchored to the
            # DNN clean baseline so it's a matched-evaluator number, not an
            # RF-anchored mismatch as it was previously).
            for space_name, methods in track_a_acc.items():
                space_a = {}
                for method_name, score_list in methods.items():
                    if score_list:
                        mean_s = float(np.mean(score_list))
                        # retention_pct only when both sides have the same sign
                        # and the denominator is bounded away from zero; same
                        # robustness guard as n09 (R² can be negative).
                        if (not np.isnan(clean_mean_dnn)
                                and clean_mean_dnn > 1e-10 and mean_s > 1e-10):
                            if lower_is_better:
                                retention = float(clean_mean_dnn / mean_s * 100)
                            else:
                                retention = float(mean_s / clean_mean_dnn * 100)
                        else:
                            retention = float('nan')
                        space_a[method_name] = {
                            'score': mean_s,
                            'retention': retention,
                            'delta_from_clean_dnn': float(mean_s - clean_mean_dnn)
                                if not np.isnan(clean_mean_dnn) else float('nan'),
                        }
                strength_result['track_a'][space_name] = space_a
            strength_result['clean_score_dnn'] = clean_mean_dnn

            # Track B aggregation (per estimator × per model)
            for space_name, est_dict in track_b_acc.items():
                space_b = {}
                for est_name, model_dict in est_dict.items():
                    est_results = {}
                    for mname, score_list in model_dict.items():
                        if score_list:
                            denoised = float(np.mean(score_list))
                            mn_noisy = noisy_mean_by_model.get(mname, noisy_mean)
                            mn_clean = clean_mean_by_model.get(mname, clean_mean)
                            mn_gap = mn_clean - mn_noisy
                            recovery = float((denoised - mn_noisy) / mn_gap * 100) if abs(mn_gap) > 1e-10 else float('nan')
                            est_results[mname] = {
                                'denoised': denoised,
                                'recovery': recovery,
                            }
                    space_b[est_name] = est_results
                strength_result['track_b'][space_name] = space_b

            nt_result['per_strength'][strength_key] = convert_numpy(strength_result)

            # Print compact
            print(f"    σ={strength_key}: noisy={noisy_mean:.4f}", end='')
            # Best track A
            best_a = None
            best_a_score = float('-inf') if not lower_is_better else float('inf')
            for sp_d in strength_result['track_a'].values():
                for mn, mv in sp_d.items():
                    s = mv.get('score', float('nan'))
                    if not np.isnan(s):
                        if (not lower_is_better and s > best_a_score) or \
                           (lower_is_better and s < best_a_score):
                            best_a_score = s
                            best_a = mn
            if best_a:
                print(f", best_LNL={best_a}({best_a_score:.4f})", end='')
            print()

        results['per_noise_type'][noise_type] = convert_numpy(nt_result)
        print(f"    [{time.time() - t_nt:.0f}s]")

    # Refresh after the loop in case extra models were auto-detected as
    # sample_weight-incompatible (TypeError on weighted fit).
    results['track_b_unweighted_models'] = sorted(UNWEIGHTED_TRACK_B_MODELS)

    # Summary
    results['summary'] = _compute_summary(results)
    s = results['summary']
    print(f"\n  === SUMMARY ===")
    print(f"  Gaussian rank transfers: {s.get('gaussian_rank_transfers')}")
    print(f"  Hybrid advantage persists: {s.get('hybrid_advantage_persists')}")
    if s.get('best_method_per_distribution'):
        for nt, m in s['best_method_per_distribution'].items():
            print(f"    {nt}: best={m}")

    return results


def _compute_summary(results):
    """Compute summary: does Gaussian ranking transfer? Does hybrid help across distributions?"""
    # Collect best Track A method per (noise_type, space)
    best_per_dist = {}  # noise_type -> best method name
    # Hybrid advantage tally: count by NOISE TYPE, not by (noise_type ×
    # strength). The previous tally was per-strength, so noise types like
    # 'partial' (which expands into PARTIAL_FRACTIONS strength variants)
    # contributed 3-5 entries while gaussian/uniform/etc contributed only
    # len(STRENGTHS) = 3. That inflation made 'partial' over-represented
    # in the win-rate. Now we tally one vote per (noise_type, strength)
    # but report results normalized per noise_type as well.
    hybrid_wins_by_type = {}
    hybrid_total_by_type = {}

    for nt, nt_data in results.get('per_noise_type', {}).items():
        best_method = None
        best_score = float('-inf')
        lower = results.get('lower_is_better', False)
        hybrid_wins_by_type[nt] = 0
        hybrid_total_by_type[nt] = 0

        for sk, sd in nt_data.get('per_strength', {}).items():
            for space_name, space_data in sd.get('track_a', {}).items():
                for mn, mv in space_data.items():
                    s = mv.get('score', float('nan'))
                    if np.isnan(s):
                        continue
                    if lower:
                        if best_method is None or s < best_score:
                            best_score = s
                            best_method = mn
                    else:
                        if s > best_score:
                            best_score = s
                            best_method = mn

            # Check hybrid advantage (Track A) at this (noise_type, strength)
            base_scores = []
            hybrid_scores = []
            for space_name, space_data in sd.get('track_a', {}).items():
                for mv in space_data.values():
                    s = mv.get('score', float('nan'))
                    if not np.isnan(s):
                        if space_name == 'hybrid':
                            hybrid_scores.append(s)
                        else:
                            base_scores.append(s)
            if base_scores and hybrid_scores:
                hybrid_total_by_type[nt] += 1
                b_avg = np.mean(base_scores)
                h_avg = np.mean(hybrid_scores)
                if (not lower and h_avg > b_avg) or (lower and h_avg < b_avg):
                    hybrid_wins_by_type[nt] += 1

        if best_method:
            best_per_dist[nt] = best_method

    # Aggregate two ways:
    #   - "raw" totals (sum of every (noise_type, strength) cell) — kept for
    #     backwards compat with downstream readers.
    #   - "per_type" win rate (mean of within-type win rates) — the
    #     inflation-resistant headline. Each noise type contributes equally
    #     regardless of how many strength variants it has.
    hybrid_wins = sum(hybrid_wins_by_type.values())
    hybrid_total = sum(hybrid_total_by_type.values())
    per_type_rates = {
        nt: (hybrid_wins_by_type[nt] / hybrid_total_by_type[nt])
        for nt in hybrid_total_by_type
        if hybrid_total_by_type[nt] > 0
    }
    persists_by_type = (np.mean(list(per_type_rates.values())) > 0.5
                       if per_type_rates else None)

    # Does Gaussian winner transfer?
    gaussian_best = best_per_dist.get('gaussian')
    if gaussian_best:
        transfers = all(best_per_dist.get(nt) == gaussian_best
                        for nt in NOISE_TYPES if nt in best_per_dist)
    else:
        transfers = None

    return convert_numpy({
        'best_method_per_distribution': best_per_dist,
        'gaussian_rank_transfers': transfers,
        # Headline flag: hybrid wins more than half of noise types on average.
        # Anti-inflation: 'partial' has multiple strength variants but only
        # counts as ONE type here.
        'hybrid_advantage_persists': persists_by_type,
        # Legacy raw totals (still per-(type, strength), for backwards
        # compatibility with readers that consumed the old fields).
        'hybrid_wins': hybrid_wins,
        'hybrid_total': hybrid_total,
        # Per-type breakdown — the supplementary view.
        'hybrid_wins_by_noise_type': hybrid_wins_by_type,
        'hybrid_total_by_noise_type': hybrid_total_by_type,
        'hybrid_win_rate_by_noise_type': per_type_rates,
    })


if __name__ == '__main__':
    parser = create_experiment_parser('N13: Noise Distributions')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'n13_noise_distributions',
                            args.dataset)
