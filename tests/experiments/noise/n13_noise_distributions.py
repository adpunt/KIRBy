#!/usr/bin/env python3
"""N13: Synthetic-Noise SENSITIVITY Study.

Paper: Noise

This is a SENSITIVITY study, not an absolute-robustness benchmark. It measures
how method rankings and hybrid advantage SHIFT when the injected synthetic-noise
DISTRIBUTION is changed, holding everything else fixed. All noise here is
artificially injected onto clean labels; the numbers describe sensitivity to the
choice of synthetic noise model, NOT a claim about robustness to real-world
label noise. Read every cell as "how much does the result move under this
injected distribution", never as "method X is robust to noise".

Conditions swept (regression): gaussian, student_t at nu = 10, 5 and 3, laplace,
outlier at p = 1%, 5% and 10%, grouped_wider and grouped_shifted. Every one of them
delivers the SAME amount of noise at a given level, so a difference between two cells
is a difference of shape. Censoring is not here: its level is the fraction of labels
clipped rather than a dose, so it is not dose-matched and belongs on its own axis.
  - 'multiplicative' was REMOVED: for mean-centred / arbitrarily-shifted targets
    y*(1+eps) is not a meaningful label-noise model (it scales by absolute value,
    so its magnitude depends on an arbitrary origin), so it was dropped rather
    than reported as a comparable distribution.
  - 'uniform' was REMOVED with the redesign. It assumed the error depends on the
    measured value, and that was tested directly on 16,844 repeat measurements and
    refuted.
  - 'partial' is gone as a separate axis. A corrupted subset is what outlier_p01,
    outlier_p05 and outlier_p10 are, with the fraction carried by the condition,
    so every condition sweeps one shared level grid and every cell is comparable.
  - Classification sweeps the six noiseInject flip models instead: uniform,
    class_imbalance, binary_asymmetric, instance_noise, class_dependent and
    confusion_directed. A flipped class label has no magnitude, so dose matching
    does not apply and the level is a flip rate.
  - grouped_wider and grouped_shifted need one scaffold group per molecule and are
    dropped on datasets with no SMILES rather than run over invented groups.

Two evaluation tracks:
  Track A: ALL 12 LNL methods from the registry (not a pre-hybrid shortlist)
           (data-level, training-loop, GP) — retention metric
  Track B: Selected estimators covering each family tested in N11
           (kNN-based, GP-based, gradient-boosted) — recovery metric.
           SAMPLE-WEIGHT remediation only: models whose fit() ignores
           sample_weight (e.g. MLP) are excluded, never fitted unweighted and
           mislabelled as remediated.

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
    pick_noise_base_rep,
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_model_factories,
    DOWNSTREAM_MODELS_NOISE,
    get_precomputed_search, get_best_search_for_downstream, extract_top3_base_reps,
    build_fold_hybrid, honest_rescore, build_honest_rescore_input,
)
from kirby.hybrid import create_hybrid, apply_feature_selection, apply_augmentation_selection
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _similarity_hybrid_space import build_similarity_hybrid
from experiments.dataset_policy import (
    noise_dose_units, noise_groups_for, reference_noise_level,
    noise_levels as dataset_noise_levels,
)
from experiments.noise.noise_helpers import (
    inject_train_labels, noise_spec_for,
)
from kirby.noise_spec import (
    CLASSIFICATION_STRATEGIES, DOSE_MATCHED_STRATEGIES, GROUPED_STRATEGIES,
)
from kirby.utils.lnl_methods import (
    get_lnl_method, _standard_dnn_train_predict, LNL_REGISTRY)
from kirby.utils.noise_estimation import get_noise_estimator, weights_from_scores


# The conditions. This experiment's whole subject is whether the SHAPE of the noise
# matters, so it runs the full set rather than a sample of it.
#
# It used to run four of its own: gaussian, uniform, heavy_tailed and partial, each at
# `strength * (max(y) - min(y))`. Two problems, both fatal to the comparison it was
# making. They were not dose-matched, so a difference between them was partly a
# difference of AMOUNT -- at one nominal setting the old six delivered between 0.49x
# and 2.00x the Gaussian amount, and their apparent severity ordering was entirely
# explained by that. And 'partial' needed a second axis of its own (a corrupted-subset
# fraction at a fixed strength) that was not comparable cell-for-cell with the others.
#
# Under the dose-matched set both problems go away. `uniform` is gone: the Uniform
# SHAPE was deleted, because its premise -- that error depends on the measured value --
# was tested on 16,844 repeat measurements and refuted. `partial` is what
# outlier_p01 / outlier_p05 / outlier_p10 now are, with the corrupted fraction folded
# into the condition, so every condition sweeps the same single level grid.
NOISE_TYPES = None       # resolved per task_type in run()

# Noise levels. `None` means the grid this dataset is anchored to in
# experiments.dataset_policy.
STRENGTHS = None

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


def _bp_sig(bp):
    """Stable recipe id for a blueprint.

    Doubles as (a) the honest_rescore candidate 'name' (per-fold winner label)
    and (b) the de-dup key so identical per-model blueprints only trigger one
    (expensive) create_hybrid build per fold.
    """
    return "{}|{}|b{}|{}|aug={}".format(
        getattr(bp, 'allocation_method', None),
        getattr(bp, 'importance_method', None),
        getattr(bp, 'budget', None),
        getattr(bp, 'internal_evaluator', None),
        getattr(bp, 'augmentation_strategy', None),
    )


# The private `_inject_noise` this module used to carry has been deleted, along
# with its four hand-rolled distributions and the separate 'partial' axis. Every
# condition now comes from noiseInject through `noise_helpers.inject_train_labels`,
# which solves in closed form for the scale that delivers the requested dose and
# records what it actually delivered.
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

    # Control arm = one plain base representation. Shared helper: it will
    # NOT fall back onto a structure-tuned sim_* recipe, which is a study
    # treatment and must never become the thing treatments are measured
    # against. See _shared.pick_noise_base_rep.
    base_rep_name = pick_noise_base_rep(reps, dataset_name)

    print(f"  Dataset: {dataset_name} ({len(labels)} samples)")
    print(f"  Metric: {metric_name}")
    # Filter out regression-only methods for classification
    # Canonical regression-only set lives in lnl_methods.py; mirroring it
    # locally previously missed evidential_kernel and would silently crash on
    # clf datasets (NIG distribution is regression-only).
    from kirby.utils.lnl_methods import REGRESSION_ONLY_LNL
    lnl_methods = [m for m in LNL_METHODS if not (is_clf and m in REGRESSION_ONLY_LNL)]
    estimators = [e for e in ESTIMATORS if not (is_clf and e == 'gaussian_process')]

    # Regression and classification take different condition sets, because a flipped
    # class label has no magnitude and so cannot be dose-matched.
    noise_types = kwargs.get('noise_types') or NOISE_TYPES
    if noise_types is None:
        noise_types = (list(CLASSIFICATION_STRATEGIES) if is_clf
                       else list(DOSE_MATCHED_STRATEGIES))
    noise_groups, n_noise_groups, largest_group_share = noise_groups_for(data)
    if noise_groups is None:
        _dropped = [c for c in noise_types if c in GROUPED_STRATEGIES]
        if _dropped:
            print(f"  {', '.join(_dropped)}: DROPPED, this dataset has no SMILES to "
                  f"build scaffold groups from. Grouped noise over invented groups is "
                  f"uniform noise wearing a grouped name.")
            noise_types = [c for c in noise_types if c not in _dropped]
    else:
        print(f"  noise groups: {n_noise_groups} scaffolds, largest holds "
              f"{largest_group_share:.1%} of molecules")

    strengths = kwargs.get('strengths') or kwargs.get('levels') or STRENGTHS
    if strengths is None:
        strengths = list(dataset_noise_levels(dataset_name))
    noise_provenance = {}

    print(f"  Base rep: {base_rep_name}")
    print(f"  Noise conditions: {noise_types}")
    print(f"  Levels ({noise_dose_units(dataset_name)}): {strengths}")
    print(f"  LNL methods: {lnl_methods}")
    print(f"  Estimators: {estimators}")
    print(f"  Folds: {n_folds}")

    # Get blueprint ONCE from pool helper (uses clean labels)
    search, meta = get_best_search_for_downstream(
        reps, labels, groups,
        downstream_model='rf', budget=None,
        augmentations=augmentations,
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

    # Extra arm: similarity-objective hybrid (label-blind winning recipe), reused
    # from the similarity study's precomputed MCS/3D caches and aligned by
    # canonical SMILES. Added only at FULL row coverage (else skipped, logged).
    # Selection never saw activity labels, so it is NOT oracle-biased.
    X_sim_full, sim_info = build_similarity_hybrid(dataset_name, data.get('smiles'))
    if X_sim_full is not None:
        print(f"  hybrid_sim arm ON: dim={sim_info['dim']} "
              f"(cache n={sim_info['n_cache']}, full coverage)")
    else:
        print(f"  hybrid_sim arm OFF: {sim_info.get('skip')}")

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

    # ── Per-model MATCHED blueprints for the honest hybrid-vs-baseline arm ──
    # The single RF clean-label blueprint above (`bp`) still drives Track A
    # (DNN-only) and the Track-B estimator geometry, UNCHANGED. The audit note
    # is that every model reused that one RF-selected blueprint. For the new
    # honest per-fold re-score arm below we additionally fetch a MATCHED-
    # evaluator blueprint per downstream model (n07 pattern; each pool call is
    # cached), so each model's hybrid space is built with a recipe tuned for
    # that model rather than reusing RF's. Distinct blueprints are de-duplicated
    # per fold (via _bp_sig) before the create_hybrid call to bound cost.
    bp_by_model = {}
    aug_by_model = {}
    for _mname in list(model_factories.keys()):
        try:
            _s_m, _m_m = get_best_search_for_downstream(
                reps, labels, groups,
                downstream_model=_mname, budget=None,
                augmentations=augmentations,
                dataset_name=dataset_name, task_type=task_type,
                data=data,
            )
        except Exception as _exc:
            print(f"  WARNING: matched blueprint fetch failed for {_mname} ({_exc})")
            continue
        if _m_m is None:
            continue
        _bp_m = _m_m['best_config']['blueprint']
        _augc_m = _m_m.get('aug_condition', 'uncapped')
        if getattr(_bp_m, 'augmentation_strategy', None) != _augc_m:
            try:
                _bp_m.augmentation_strategy = _augc_m
            except Exception:
                pass
        bp_by_model[_mname] = _bp_m
        aug_by_model[_mname] = _augc_m
    # Fall back to the RF blueprint for any gated model the pool lacked, so
    # every evaluated model still contributes an honest arm.
    for _mname in model_factories:
        if _mname not in bp_by_model and bp is not None:
            bp_by_model[_mname] = bp
            aug_by_model[_mname] = aug_condition

    # Reset the unweighted-model registry to its seed for this run (module-level
    # mutable set persists across run() calls in a single process).
    global UNWEIGHTED_TRACK_B_MODELS
    UNWEIGHTED_TRACK_B_MODELS = {'mlp'}

    results = {
        # This experiment is a SYNTHETIC-noise SENSITIVITY study: it reports how
        # rankings/hybrid-advantage SHIFT across injected noise distributions,
        # NOT an absolute robustness claim about real-world label noise.
        'study_type': 'synthetic_noise_sensitivity',
        'is_absolute_robustness_claim': False,
        'noise_model': 'synthetic (labels injected onto clean targets)',
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        # Every condition now sweeps ONE axis -- the level -- and delivers the same
        # amount of noise at a given level, so cells are comparable across the whole
        # grid. The old 'partial' track had an axis of its own that was not.
        'noise_types': noise_types,
        'noise_types_dropped': ['multiplicative', 'uniform'],
        'strengths': strengths,
        'dose_units': noise_dose_units(dataset_name),
        'noise_provenance': noise_provenance,
        'n_noise_groups': n_noise_groups,
        'largest_group_share': largest_group_share,
        'lnl_methods': lnl_methods,
        'estimators': estimators,
        # Downstream models participating in the honest per-fold re-score arm
        # (gated by n_samples via get_model_factories — never a fixed set).
        'models': list(model_factories.keys()),
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

    # A sensitivity study has no single "the" condition, so the top-level
    # contract block exposes a CANONICAL moderate-noise slice (gaussian
    # sigma=0.3 when present, else the first computed cell). The full per-
    # (noise_type, strength) set is also kept nested. See the assignment after
    # the main loop.
    canonical_honest = None
    canonical_honest_key = None
    # A sensitivity study has no single "the" condition, so the canonical slice is
    # Gaussian at one unit of real error for this dataset -- 0.54 log units on
    # herg_ki, 0.35 on caco2, half the label SD where nothing is published.
    _canonical_level = min(strengths, key=lambda v: abs(v - reference_noise_level(dataset_name))) \
        if strengths else 0.0

    # ── Main loop: noise type × strength ──
    for noise_type in noise_types:
        print(f"\n  === {noise_type} ===")
        t_nt = time.time()
        nt_result = {'per_strength': {}}

        # One axis for every condition. The corrupted fraction that used to need its
        # own axis is carried by the condition itself (outlier_p01/p05/p10).
        for strength in strengths:
            strength_key = str(strength)

            noisy_baseline_by_model = {mn: [] for mn in model_factories}
            # {space: {method: [scores]}} for track A
            track_a_acc = {}
            # {space: {estimator: {model: [scores]}}} for track B
            track_b_acc = {}

            # Honest per-fold arms for the shared honest_rescore helper:
            #   honest_base_folds[model][fold] = [per-seed base-rep scores]
            #   honest_hyb_folds[model][fold]  = [per-seed matched-hybrid scores]
            # Each fold entry is averaged over NOISE_SEEDS below, then laid out
            # fold-aligned so honest_rescore can leave-one-fold-out re-select
            # each arm's winner WITHOUT the held-out fold (the arithmetic
            # embodiment of the n09 nested-selection template). We do NOT run a
            # per-fold find_best_hybrid inner search here: that would multiply
            # the noise_type x strength x fold x seed matrix by a full search
            # and bust the 7-day wall; the LOFO re-selection over the cached
            # matched-blueprint candidates gives the held-out-fold-free
            # selection instead.
            honest_base_folds = {mn: {} for mn in model_factories}
            honest_hyb_folds = {mn: {} for mn in model_factories}
            honest_hyb_name = {}  # model -> its blueprint signature (candidate name)

            for fi, fd in enumerate(fold_data):
                for noise_seed in NOISE_SEEDS:
                    # Per-(condition, level, fold, seed) noise realization
                    _spec = noise_spec_for(dataset_name, noise_type, strength,
                                           fold_idx=fi, noise_seed=noise_seed,
                                           is_clf=is_clf)
                    _tr_idx = fd['train_idx']
                    y_noisy, _cmask, _prov = inject_train_labels(
                        fd['y_train_clean'], _spec, is_clf=is_clf,
                        groups=(noise_groups[_tr_idx]
                                if noise_groups is not None else None))
                    noise_provenance.setdefault(_spec.spec_id, _prov)

                    # Noisy baseline per model
                    for mname, mfactory in model_factories.items():
                        try:
                            m = mfactory()
                            m.fit(fd['X_eval_train_s'], y_noisy)
                            _sc_base = float(metric_fn(
                                fd['y_val'], m.predict(fd['X_eval_val_s'])))
                            noisy_baseline_by_model[mname].append(_sc_base)
                            # Per-fold base-rep arm for honest_rescore (reuses
                            # this same fit — no extra training cost).
                            honest_base_folds[mname].setdefault(fi, []).append(_sc_base)
                        except Exception:
                            pass

                    # Build spaces from NOISY labels (realistic — no oracle leak)
                    spaces, feature_info, scaler_h = _build_spaces(
                        fd['train_reps_fold'], y_noisy, fd['train_augs'],
                        bp, base_rep_name)

                    # similarity-objective hybrid arm (fixed representation, so
                    # its own train+val slices; no feature_info needed)
                    hybrid_sim_val_s = None
                    if X_sim_full is not None:
                        _sc_sim = StandardScaler()
                        spaces['hybrid_sim'] = _sc_sim.fit_transform(
                            X_sim_full[fd['train_idx']])
                        hybrid_sim_val_s = _sc_sim.transform(
                            X_sim_full[fd['val_idx']])

                    # ── Honest hybrid arm: train each model PLAINLY (no LNL,
                    # no sample weights) on its MATCHED hybrid space, built from
                    # that model's blueprint on the NOISY labels. This is the
                    # hybrid candidate the honest_rescore helper compares against
                    # the base-rep baseline arm. Identical blueprints across
                    # models share one create_hybrid build via _hyb_space_cache.
                    _hyb_space_cache = {}  # bp_sig -> (X_tr_s, X_va_s) or None
                    for mname, mfactory in model_factories.items():
                        _bp_m = bp_by_model.get(mname)
                        if _bp_m is None:
                            continue
                        _sig = _bp_sig(_bp_m)
                        honest_hyb_name[mname] = _sig
                        if _sig not in _hyb_space_cache:
                            try:
                                _augc_m = aug_by_model.get(mname, 'uncapped')
                                if _augc_m != 'none' and augmentations:
                                    _tr_aug = {k: v[fd['train_idx']]
                                               for k, v in augmentations.items()}
                                    _va_aug = {k: v[fd['val_idx']]
                                               for k, v in augmentations.items()}
                                else:
                                    _tr_aug = _va_aug = None
                                _Xh_tr, _Xh_va, _ = build_fold_hybrid(
                                    fd['train_reps_fold'], fd['val_reps_fold'],
                                    y_noisy, _bp_m,
                                    augmentations_train=_tr_aug,
                                    augmentations_val=_va_aug,
                                    groups=(groups[fd['train_idx']] if groups is not None else None),
                                    model=mfactory(),
                                    metric_fn=metric_fn,
                                    lower_is_better=lower_is_better)
                                _sc_h = StandardScaler()
                                _hyb_space_cache[_sig] = (
                                    _sc_h.fit_transform(_Xh_tr),
                                    _sc_h.transform(_Xh_va))
                            except Exception:
                                _hyb_space_cache[_sig] = None
                        _built = _hyb_space_cache[_sig]
                        if _built is None:
                            continue
                        _Xh_tr_s, _Xh_va_s = _built
                        try:
                            m = mfactory()
                            m.fit(_Xh_tr_s, y_noisy)
                            honest_hyb_folds[mname].setdefault(fi, []).append(
                                float(metric_fn(fd['y_val'], m.predict(_Xh_va_s))))
                        except Exception:
                            pass

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
                    for space_name in [base_rep_name, 'hybrid', 'hybrid_sim']:
                        if space_name not in spaces:
                            continue
                        X_train_s = spaces[space_name]

                        if space_name == 'hybrid' and hybrid_val_s is not None:
                            X_val_s = hybrid_val_s
                        elif space_name == 'hybrid_sim' and hybrid_sim_val_s is not None:
                            X_val_s = hybrid_sim_val_s
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
                    for space_name in [base_rep_name, 'hybrid', 'hybrid_sim']:
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

            # ── Honest re-score block for this (noise_type, strength) ──
            # Fold-aligned per-arm scores (mean over NOISE_SEEDS within each
            # fold; np.nan for folds a model failed on), in the exact shape the
            # shared honest_rescore helper consumes. Written under the contract
            # key names so a reader can call
            #   honest_rescore(block['candidates_by_model'],
            #                  block['baselines_by_model'],
            #                  block['lower_is_better']).
            hr_candidates = {}
            hr_baselines = {}
            for mn in model_factories:
                base_map = honest_base_folds.get(mn, {})
                hyb_map = honest_hyb_folds.get(mn, {})
                if any(base_map.get(fi) for fi in range(n_folds)):
                    hr_baselines[mn] = {base_rep_name: [
                        float(np.mean(base_map[fi]))
                        if base_map.get(fi) else float('nan')
                        for fi in range(n_folds)]}
                if any(hyb_map.get(fi) for fi in range(n_folds)):
                    hr_candidates[mn] = [{
                        'name': honest_hyb_name.get(mn, 'hybrid'),
                        'fold_scores': [
                            float(np.mean(hyb_map[fi]))
                            if hyb_map.get(fi) else float('nan')
                            for fi in range(n_folds)],
                    }]
            honest_block = {
                'lower_is_better': lower_is_better,
                'candidates_by_model': hr_candidates,
                'baselines_by_model': hr_baselines,
            }
            strength_result['honest_rescore'] = honest_block

            # Stash the canonical top-level slice (moderate gaussian noise) so a
            # reader can json.load(...)['honest_rescore'] directly.
            _is_canonical = (noise_type == 'gaussian'
                             and abs(strength - _canonical_level) < 1e-9)
            if _is_canonical or canonical_honest is None:
                canonical_honest = honest_block
                canonical_honest_key = f'{noise_type}:{strength_key}'

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

    # ── Top-level honest_rescore block (contract json_convention) ──
    # A sensitivity study has no single condition; expose a CANONICAL moderate-
    # noise slice here (gaussian sigma=0.3 when available, else the first
    # computed cell). The full per-(noise_type, strength) set is preserved under
    # per_noise_type[*]['per_strength'][*]['honest_rescore']. A reader loads this
    # block and calls honest_rescore(block['candidates_by_model'],
    # block['baselines_by_model'], block['lower_is_better']).
    if canonical_honest is not None:
        results['honest_rescore'] = convert_numpy(canonical_honest)
        results['honest_rescore_condition'] = canonical_honest_key
        # Sanity log via the shared helper (pure arithmetic, no model fitting).
        try:
            _hr = honest_rescore(canonical_honest['candidates_by_model'],
                                 canonical_honest['baselines_by_model'],
                                 lower_is_better)
            print(f"\n  === HONEST RE-SCORE ({canonical_honest_key}) ===")
            for _mn, _r in _hr.items():
                print(f"    {_mn}: hybrid={_r['hybrid_honest']:.4f} "
                      f"base={_r['best_single_honest']:.4f} "
                      f"delta={_r['delta']:+.4f} single_fold={_r['single_fold']}")
        except Exception as _exc:
            print(f"  honest_rescore summary skipped ({_exc})")

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
    # Hybrid advantage tally: count by NOISE TYPE, not by (noise_type x level).
    # Every condition now sweeps the SAME single level grid, so no condition
    # contributes more cells than any other -- the inflation the old 'partial'
    # track caused, with its own separate axis, is gone. The per-type
    # normalisation is kept anyway, so an unequal grid could not silently
    # reintroduce it.
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

    # Every condition sweeps the SAME axis now -- the noise level -- and delivers
    # the same amount at a given level, so every cell is comparable with every other
    # and they all share one x-axis. Censoring is the one condition that does not
    # belong on it, because its level is the fraction of labels clipped rather than
    # a dose; it is swept on its own axis and n13 does not include it.
    sweep_axis_by_noise_type = {
        nt: 'noise_level' for nt in results.get('per_noise_type', {})
    }

    return convert_numpy({
        'best_method_per_distribution': best_per_dist,
        'gaussian_rank_transfers': transfers,
        'sweep_axis_by_noise_type': sweep_axis_by_noise_type,
        # Headline flag: hybrid wins more than half of noise types on average.
        # One vote per condition, whatever its level grid.
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
