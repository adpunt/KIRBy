#!/usr/bin/env python3
"""N07: Synthetic-Noise Remediation Strategies — what to do after detecting noise.

Paper: Noise

SCOPE (read first): this is a SYNTHETIC-noise study. Noise is injected
artificially onto the TRAINING labels only. For regression that is the Outlier
condition — a randomly chosen `noise_fraction` of records get an error three
times wider than the rest, which is Huber contamination and is what real
database errors look like (Kalliokoski et al. 2013 inspected the extreme
disagreements and found nine or ten in ten beyond 2.5 log units were clear
errors). For classification it is random class flips at that rate. Injection
goes through `noise_helpers.inject_train_labels`, which solves for the scale
that delivers the requested amount and records what it delivered.
Validation labels are always clean. There is no
real-world / "fiction"-scored noise detector and no naturally noisy
dataset here: the noise is generated, so its statistics are known by
construction. Every claim below is therefore confined to the
synthetic-noise regime — it says which remediation strategy best recovers
performance when the corruption matches this injected model, and does NOT
establish behaviour on real label noise of unknown structure.

Compares 4 remediation strategies after noise detection: remove, downweight,
label_smooth, knn_smooth. Two baselines (no_noise, noisy_baseline) anchor
recovery. All strategies keep the iteration's downstream model fixed so the
comparison is apples-to-apples within a column.

(Earlier versions also included a robust_loss strategy that swapped in
HuberRegressor, but that conflated "different cleanup strategy" with
"different model entirely" — its number wasn't comparable to the other
rows. Removed in the 2026-05-26 audit. If you want to evaluate Huber as
a noise-robust model, run a separate experiment with Huber as the
baseline across all strategies.)

Evaluated across all five noise-suite models (random forest, xgboost,
support vector, DNN, lightgbm) for the hybrid feature space — each model
uses its own search-derived feature mix. The per-rep block is RF-only
(diagnostic; expanding it to 5 × n_reps × 4 strategies would balloon
runtime without adding to the core remediation claim).

WHICH LABELS CHOOSE THE FEATURES. create_hybrid ranks features and splits the
budget against whatever labels it is handed. Until 2026-09-04 every fold here
was handed the CLEAN ones, so the feature space knew which molecules the
experiment was about to corrupt before it corrupted them. Each fold now builds
its space from that fold and noise draw's NOISY training labels, which costs one
hybrid build per (fold, noise seed) instead of one per fold. The clean-label
space is kept as an explicitly named oracle arm,
`per_model_per_strategy_oracle_features`, so the size of that head start is
readable instead of assumed.

The RECIPE — allocation method, importance method, internal evaluator, budget —
still comes from the clean-label search pool, because the noisy search lineage
covers 0.05/0.10/0.20 gaussian only and this experiment runs the uniform
condition at p=0.1. Choosing the recipe therefore remains a head start; only
the per-fold ranking and budget split are now noise-aware.

Usage:
    python n07_remediation.py --dataset logd --mode intermediary
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.preprocessing import StandardScaler
from experiments._shared import make_nearest_neighbors  # honours --distance-metric
from experiments.dataset_policy import (
    noise_dose_units, reference_noise_level,
)
from experiments.noise.noise_helpers import inject_train_labels, noise_spec_for
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_model_factories,
    get_precomputed_search, get_best_search_for_downstream, build_fold_hybrid,
    extract_top3_base_reps, DOWNSTREAM_MODELS_NOISE,
    honest_rescore, build_honest_rescore_input,
)
from kirby.utils.noise_estimation import (
    ActivityCliffAwareEstimator, smooth_by_scores, weights_from_scores,
)


# Multi-seed noise averaging. Each fold runs noise injection at multiple
# seeds and aggregates — reduces dependence on a single noise realization.
NOISE_SEEDS = [0, 1, 2]


# The private `_inject_noise` this module used to carry has been deleted. It picked a
# `fraction` of rows at random and gave each of them Gaussian noise at 0.3 of the label
# RANGE -- a fixed magnitude with no level axis, and a fraction of the range rather than
# of the SD, so it ran at several times the level any other experiment meant by the same
# number. That IS the Outlier condition, so it becomes one: `outlier` with p set from
# `noise_fraction`, at a dose. Selection stays random, which is the point of the
# condition -- a mistyped unit is a property of the record, not of the value.


def _noise_spec(dataset_name, noise_fraction, level, fold_idx, noise_seed,
                is_clf=False):
    """The Outlier condition at this dataset's corrupted fraction.

    For classification the level is the flip rate and the corrupted fraction IS the
    level, because a flipped class label has no magnitude to dose-match.
    """
    from dataclasses import replace
    if is_clf:
        return noise_spec_for(dataset_name, 'uniform', noise_fraction,
                              fold_idx=fold_idx, noise_seed=noise_seed, is_clf=True)
    spec = noise_spec_for(dataset_name, 'outlier_p05', level,
                          fold_idx=fold_idx, noise_seed=noise_seed)
    return replace(spec, params=(('p', float(noise_fraction)),))


def _predict_for_metric(model, X, metric_name):
    """Return predictions in the format the given metric expects.

    AUROC needs continuous scores in [0, 1]; class labels make it a
    degenerate balanced-accuracy approximation. All other metrics in
    BUILTIN_METRICS take either class labels (f1/accuracy/mcc) or raw
    floats (rmse/r2/spearman/ci/pearson), which is what predict() returns.
    """
    if metric_name == 'auc' and hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)
        if getattr(proba, 'ndim', 1) == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        return proba
    return model.predict(X)


def _compute_noise_scores(X_tr_s, y_tr, is_clf=False):
    """Fit ActivityCliffAwareEstimator on (X_tr_s, y_tr) and return noise_scores_.

    Hoisted out of _evaluate_fold so the caller can compute once per (fold,
    noise_seed) and reuse across the 4 strategies that need it (remove,
    downweight, label_smooth, knn_smooth). Previously each strategy refit
    the same estimator on the same data — ~4× wasted work per fold.
    Returns None if the estimator fit fails.

    is_clf threads into the estimator so classification datasets use its
    class-disagreement noise-scoring branch instead of the regression
    neighbor-mean / local-variance path (which divides integer class labels
    by their range — meaningless on nominal labels). Mirrors the per-rep /
    hybrid is_clf threading audited as P0#1.
    """
    try:
        est = ActivityCliffAwareEstimator(k=10, metric=None, is_clf=is_clf)
        est.fit(X_tr_s, y_tr)
        return est.noise_scores_
    except Exception as exc:
        print(f"      noise_scores estimator failed: "
              f"{type(exc).__name__}: {exc}")
        return None


def _evaluate_fold(X_tr_s, X_va_s, y_train, y_val, strategy, noise_fraction,
                   model_factory, metric_fn, is_clf=False, metric_name=None,
                   noise_seed=42, precomputed_y_tr=None, noise_scores=None,
                   dataset_name=None, noise_level=0.5):
    """Evaluate a single remediation strategy on one fold.

    noise_seed: forwarded to the injector so the caller can average across
    multiple noise realizations. Default 42 preserves single-seed behavior.
    precomputed_y_tr / noise_scores: optional caller-supplied y_tr and
    noise_scores. When provided, the per-strategy call doesn't reinject
    noise or refit the estimator — the caller is expected to do that once
    per (fold, noise_seed) and pass them in. This is the recommended path;
    the no-cache path is kept for callers that haven't migrated.

    Returns the fold score (float) or None if the fold failed.
    """
    # Inject noise (except for clean baseline) — if caller didn't precompute.
    if strategy == 'no_noise':
        y_tr = y_train.copy()
    elif precomputed_y_tr is not None:
        y_tr = precomputed_y_tr
    else:
        _spec = _noise_spec(dataset_name, noise_fraction, noise_level,
                            fold_idx=0, noise_seed=noise_seed, is_clf=is_clf)
        y_tr, _noise_mask, _ = inject_train_labels(y_train, _spec, is_clf=is_clf)

    # If caller supplied noise_scores, use them; otherwise compute. Returns
    # None when the strategy needs scores and they're unavailable (loud).
    if strategy in ('remove', 'downweight', 'label_smooth', 'knn_smooth'):
        if noise_scores is None:
            noise_scores = _compute_noise_scores(X_tr_s, y_tr, is_clf=is_clf)
        if noise_scores is None:
            return None

    # Apply strategy
    try:
        if strategy in ('no_noise', 'noisy_baseline'):
            m = model_factory()
            m.fit(X_tr_s, y_tr)
            return metric_fn(y_val, _predict_for_metric(m, X_va_s, metric_name))

        elif strategy == 'remove':
            threshold = np.percentile(noise_scores, 80)
            keep = noise_scores < threshold
            if keep.sum() >= 10:
                m = model_factory()
                m.fit(X_tr_s[keep], y_tr[keep])
                return metric_fn(y_val, _predict_for_metric(m, X_va_s, metric_name))

        elif strategy == 'downweight':
            weights = weights_from_scores(noise_scores, strength=1.0)
            m = model_factory()
            m.fit(X_tr_s, y_tr, sample_weight=weights)
            return metric_fn(y_val, _predict_for_metric(m, X_va_s, metric_name))

        elif strategy == 'label_smooth':
            if is_clf:
                # Label smooth uses neighbor means — use neighbor mode for clf
                nn = make_nearest_neighbors(n_neighbors=min(10, len(y_tr) - 1))
                nn.fit(X_tr_s)
                _, nbr_idx = nn.kneighbors(X_tr_s)
                from scipy.stats import mode as scipy_mode
                nbr_modes = scipy_mode(y_tr[nbr_idx[:, 1:]], axis=1).mode.ravel()
                # Only correct high-noise-score samples
                threshold = np.percentile(noise_scores, 80)
                y_smoothed = y_tr.copy()
                y_smoothed[noise_scores >= threshold] = nbr_modes[noise_scores >= threshold]
                m = model_factory()
                m.fit(X_tr_s, y_smoothed)
                return metric_fn(y_val, _predict_for_metric(m, X_va_s, metric_name))
            else:
                nn = make_nearest_neighbors(n_neighbors=min(10, len(y_tr) - 1))
                nn.fit(X_tr_s)
                _, nbr_idx = nn.kneighbors(X_tr_s)
                nbr_means = np.mean(y_tr[nbr_idx[:, 1:]], axis=1)
                y_smoothed = smooth_by_scores(y_tr, noise_scores, nbr_means,
                                              alpha=0.5, method='proportional')
                m = model_factory()
                m.fit(X_tr_s, y_smoothed)
                return metric_fn(y_val, _predict_for_metric(m, X_va_s, metric_name))

        elif strategy == 'knn_smooth':
            threshold = np.percentile(noise_scores, 80)
            flagged = noise_scores >= threshold
            nn = make_nearest_neighbors(n_neighbors=min(10, len(y_tr) - 1))
            nn.fit(X_tr_s)
            _, nbr_idx = nn.kneighbors(X_tr_s)
            if is_clf:
                from scipy.stats import mode as scipy_mode
                nbr_modes = scipy_mode(y_tr[nbr_idx[:, 1:]], axis=1).mode.ravel()
                y_corrected = y_tr.copy()
                y_corrected[flagged] = nbr_modes[flagged]
            else:
                nbr_means = np.mean(y_tr[nbr_idx[:, 1:]], axis=1)
                y_corrected = y_tr.copy()
                y_corrected[flagged] = nbr_means[flagged]
            m = model_factory()
            m.fit(X_tr_s, y_corrected)
            return metric_fn(y_val, _predict_for_metric(m, X_va_s, metric_name))

    except Exception as exc:
        # Surface the failure rather than silently dropping it. Helps
        # distinguish real bugs from "strategy not applicable here" cases.
        print(f"      {strategy} fold failed: {type(exc).__name__}: {exc}")
        return None

    return None


def run(data, config, output_dir, dataset_name, **kwargs):
    """N07: Noise remediation strategies."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N07: NOISE REMEDIATION STRATEGIES")
    print("=" * 70)

    task_type = data['task_type']
    is_clf = task_type == 'classification'
    # Per noise convention, evaluate remediation across DOWNSTREAM_MODELS_NOISE
    # (rf, xgboost, svm, dnn, lightgbm) instead of just RF. The audit called the
    # RF-only setup a P1 because remediation behaviour is highly model-dependent
    # (e.g. label_smooth tends to help a neural net more than tree ensembles).
    # Each model gets a matched-evaluator blueprint via the cached pool helper
    # below.
    model_factories = get_model_factories(
        DOWNSTREAM_MODELS_NOISE, task_type=task_type,
        n_samples=len(data['labels']))

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    augmentations = data.get('augmentations')
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    budget = kwargs.get('budget', 100)
    noise_fraction = kwargs.get('noise_fraction', 0.1)
    # How much noise the corrupted rows receive. This experiment had no level axis
    # at all -- the magnitude was hardcoded at 0.3 of the label range -- so the
    # default is one unit of real error for this dataset: 0.54 log units on herg_ki
    # (Kramer et al. 2012), 0.35 on caco2 (Bentz et al. 2013 Table 6), half the label
    # SD where nothing is published.
    noise_level = kwargs.get('noise_level', reference_noise_level(dataset_name))
    noise_provenance = {}

    sample_X = next(iter(reps.values()))

    # Per-model blueprint fetch (matched evaluator). Each call is cached.
    # rf is the canonical "top-3 base reps" reference, so fetch it first
    # both for that print and as a fallback if the pool lacks a model's
    # blueprint entry.
    bp_by_model = {}
    aug_by_model = {}
    # Keep the winning per-model SearchResult so we can persist an honest
    # hybrid-vs-baseline re-scoring block (see honest_rescore block below).
    search_by_model = {}
    # Every model in DOWNSTREAM_MODELS_NOISE that was ASKED for, and what
    # happened to it. A model with no blueprint contributes no row to
    # per_model_per_strategy, so without this record the file looks like the
    # experiment was configured for the smaller set. 'ok' means a blueprint
    # was obtained; anything else is the reason it was not.
    models_requested = list(model_factories.keys())
    blueprint_status = {m: 'not_attempted' for m in models_requested}
    rf_search = None
    rf_meta = None
    for _mname in models_requested:
        try:
            _search, _meta = get_best_search_for_downstream(
                reps, labels, groups,
                downstream_model=_mname, budget=budget,
                augmentations=augmentations,
                dataset_name=dataset_name, task_type=task_type,
                data=data,
            )
            if _mname == 'rf':
                rf_search, rf_meta = _search, _meta
            if _meta is None:
                blueprint_status[_mname] = 'no_search_metadata'
                print(f"  WARNING: no search metadata for {_mname} — "
                      f"no remediation scores will be written for it")
                continue
            _bp = _meta['best_config']['blueprint']
            _aug_cond = _meta.get('aug_condition', 'uncapped')
            if getattr(_bp, 'augmentation_strategy', None) != _aug_cond:
                try:
                    _bp.augmentation_strategy = _aug_cond
                except Exception:
                    pass
            bp_by_model[_mname] = _bp
            aug_by_model[_mname] = _aug_cond
            search_by_model[_mname] = _search
            blueprint_status[_mname] = 'ok'
        except Exception as _exc:
            blueprint_status[_mname] = (
                f'fetch_raised: {type(_exc).__name__}: {_exc}')
            print(f"  WARNING: blueprint fetch failed for {_mname} ({_exc})")

    models_with_blueprint = [m for m in models_requested
                             if blueprint_status[m] == 'ok']
    models_missing_blueprint = {m: blueprint_status[m] for m in models_requested
                                if blueprint_status[m] != 'ok'}
    # DECISION (2026-08-26): a partial model set CONTINUES rather than stops.
    # The models that dropped out are named in the result under
    # 'models_missing_blueprint', with the reason for each, and the models that
    # were asked for are listed under 'models_requested'. A reader comparing
    # 'models_requested' against 'models' therefore sees the difference instead
    # of having to infer it from an absent row. Stopping would throw away the
    # folds already runnable for the models that DID get a blueprint; the
    # condition for continuing is that no model disappears without a record.
    if models_missing_blueprint:
        print(f"  Models requested: {len(models_requested)} "
              f"({', '.join(models_requested)})")
        print(f"  Models continuing with a blueprint: "
              f"{len(models_with_blueprint)} "
              f"({', '.join(models_with_blueprint) or 'none'})")
        for _m, _why in models_missing_blueprint.items():
            print(f"  DROPPED {_m}: {_why} — no remediation scores in output")

    if not bp_by_model:
        print("  WARNING: No blueprint found for any model, skipping")
        return {
            'error': 'no_blueprint',
            'metric': metric_name,
            'models_requested': models_requested,
            'blueprint_status': blueprint_status,
            'models_missing_blueprint': models_missing_blueprint,
        }

    # If RF specifically is missing, fall back to any model whose blueprint
    # we did get so we can still print the top-3 base reps and populate
    # the legacy top3_base_reps field.
    top3 = None
    if rf_search is not None:
        top3 = extract_top3_base_reps(rf_search)
    else:
        _any_name = next(iter(bp_by_model))
        try:
            _any_search, _ = get_best_search_for_downstream(
                reps, labels, groups,
                downstream_model=_any_name, budget=budget,
                augmentations=augmentations,
                dataset_name=dataset_name, task_type=task_type,
                data=data,
            )
            top3 = extract_top3_base_reps(_any_search) if _any_search else None
        except Exception:
            top3 = None
    if top3:
        print(f"  Top-3 base reps (ranked by RF baseline): "
              f"{', '.join(f'{n}={s:.4f}' for n, s in top3)}")

    strategies = [
        'no_noise',         # clean baseline (no injection)
        'noisy_baseline',   # inject noise, do nothing
        'remove',           # detect + remove flagged samples
        'downweight',       # detect + weight by 1/(1+score)
        'label_smooth',     # detect + smooth labels toward neighbor means
        'knn_smooth',       # replace flagged labels with kNN-mean in hybrid space
    ]
    # robust_loss removed (2026-05-26 audit decision): the strategy swapped
    # the iteration's downstream model for HuberRegressor, conflating "data
    # cleanup" with "different model entirely" — its recovery numbers were
    # not comparable to the four cleanup strategies above. If you want to
    # test Huber separately, run a dedicated experiment with Huber as the
    # baseline model across all strategies.

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'noise_fraction': noise_fraction,
        'noise_level': noise_level,
        'noise_condition': 'outlier',
        'dose_units': noise_dose_units(dataset_name),
        'noise_provenance': noise_provenance,
        # Scoping metadata: this experiment studies SYNTHETIC label noise
        # injected on the TRAINING labels only (Gaussian for regression /
        # random class flips for classification; validation labels stay
        # clean). No real/"fiction"-scored noise detector is used, so the
        # remediation claim is confined to the synthetic-noise regime.
        'noise_regime': 'synthetic',
        'noise_injection': (
            'gaussian_label_noise_train_only' if not is_clf
            else 'class_flip_label_noise_train_only'),
        'claim_scope': (
            'Remediation ranking holds only under synthetically injected '
            'training-label noise; not validated on real label noise of '
            'unknown structure.'),
        'strategies': strategies,
        # 'models' is unchanged: the models that actually produced remediation
        # scores. The three keys below say what the run was ASKED for, so an
        # absent model reads as a recorded drop-out rather than as a smaller
        # configured set. models_requested == models when nothing dropped.
        'models': list(bp_by_model.keys()),
        'models_requested': models_requested,
        'blueprint_status': blueprint_status,
        'models_missing_blueprint': models_missing_blueprint,
        'top3_base_reps': [(n, s) for n, s in top3] if top3 else [],
        # Legacy single-model section kept for backwards compatibility;
        # populated from RF if available, else first available model.
        'per_strategy': {},
        # Canonical per-model results — every reader should prefer this.
        'per_model_per_strategy': {m: {} for m in bp_by_model},
        # The same strategies measured in a feature space chosen from the CLEAN
        # labels, before the noise this experiment studies was injected. That
        # is what every score in this file used to be. It is kept as a named
        # oracle arm so the head start can be read off rather than assumed —
        # subtract it from per_model_per_strategy to see what choosing features
        # with the true labels was worth.
        'per_model_per_strategy_oracle_features': {},
        'feature_selection_labels': 'noisy_per_fold',
        'feature_selection_note': (
            'Each fold builds its hybrid from that fold and noise draw\'s NOISY '
            'training labels. The RECIPE (allocation method, importance method, '
            'internal evaluator, budget) still comes from the clean-label search '
            'pool, so the choice of recipe remains a head start; only the '
            'per-fold feature ranking and budget split are now noise-aware.'),
    }

    # --- Honest hybrid-vs-baseline re-scoring input (shared-helper contract) ---
    # Persist, per model, EVERY searched hybrid candidate recipe's per-fold
    # scores AND the per-fold baseline-rep scores. A reader feeds this straight
    # to experiments._shared.honest_rescore, which for each HELD-OUT fold
    # re-picks the winning recipe from the OTHER folds (leave-one-fold-out) and
    # applies the identical held-out tax to both arms. That is the n09
    # selection-honesty principle — "the recipe used at a fold is chosen without
    # that fold" — expressed as pure arithmetic instead of a per-fold re-search.
    # Built from the per-model precomputed searches (the clean-label search
    # space that also SELECTED each model's blueprint); the injected-noise
    # remediation numbers stay in per_model_per_strategy below and keep their
    # own recovery derivation.
    honest_block = {
        'lower_is_better': lower_is_better,
        # Every fold counts. The candidates here are the arms this file built itself,
        # not a search's recipe list, so no fold filtered them.
        'exclude_folds': [],
        'candidates_by_model': {},
        'baselines_by_model': {},
    }
    for _m, _s in search_by_model.items():
        try:
            _blk = build_honest_rescore_input(_s)
        except Exception as _exc:
            print(f"  WARNING: honest_rescore input build failed for {_m} "
                  f"({_exc})")
            continue
        # Keep only THIS model's slice — a pooled SearchResult's all_results
        # may carry other models' candidates, so stay model-gated.
        if _m in _blk.get('candidates_by_model', {}):
            honest_block['candidates_by_model'][_m] = \
                _blk['candidates_by_model'][_m]
        if _m in _blk.get('baselines_by_model', {}):
            honest_block['baselines_by_model'][_m] = \
                _blk['baselines_by_model'][_m]
    results['honest_rescore'] = honest_block

    # --- Hybrid conditions: per-model ---
    # Loop order is (model × fold × seed × strategy) so the per-(fold, seed)
    # noise injection + estimator fit are computed ONCE and reused across
    # all 7 strategies. The previous (model × strategy × fold) order
    # recomputed those 4× per (fold, seed) — once per strategy that needed
    # noise_scores. The audit P1 perf-recompute fix.
    for model_name, model_factory in model_factories.items():
        if model_name not in bp_by_model:
            continue
        model_bp = bp_by_model[model_name]
        model_aug_cond = aug_by_model[model_name]
        model_augs = augmentations if model_aug_cond != 'none' else None

        per_strategy_scores = {s: [] for s in strategies}
        # Same strategies again, measured in the CLEAN-label feature space.
        # That space is the head start: create_hybrid ranks features and splits
        # the budget against the labels it is given, so features chosen before
        # the noise was injected already know which molecules the experiment is
        # about to corrupt. The reported arm above no longer gets it; this one
        # keeps it so the size of the head start is readable rather than
        # assumed. Read it as an oracle, the way n15 labels its clean_bp arm.
        per_strategy_scores_oracle = {s: [] for s in strategies}

        for fold_i, (train_idx, val_idx) in enumerate(
                cv_split(groups, n_folds, sample_X, labels, mode=mode)):
            y_train_clean = labels[train_idx]
            y_val = labels[val_idx]

            train_reps_fold = {k: v[train_idx] for k, v in reps.items()}
            val_reps_fold = {k: v[val_idx] for k, v in reps.items()}
            train_augs = {k: v[train_idx] for k, v in model_augs.items()} if model_augs else None
            val_augs = {k: v[val_idx] for k, v in model_augs.items()} if model_augs else None

            def _fold_space(y_for_features, _tr=train_reps_fold, _va=val_reps_fold,
                            _ta=train_augs, _va_a=val_augs, _tri=train_idx,
                            _mf=model_factory, _bp=model_bp):
                """(X_train, X_val) standardised, features chosen from
                y_for_features. Returns None when the build fails."""
                try:
                    X_tr_h, X_va_h, _fi = build_fold_hybrid(
                        _tr, _va, y_for_features, _bp,
                        augmentations_train=_ta, augmentations_val=_va_a,
                        groups=(groups[_tri] if groups is not None else None),
                        model=_mf(),
                        metric_fn=metric_fn,
                        lower_is_better=lower_is_better)
                except Exception:
                    return None
                sc = StandardScaler()
                return sc.fit_transform(X_tr_h), sc.transform(X_va_h)

            # Clean-label space. Used for the no_noise ceiling, which is a
            # clean-label condition throughout, and for the oracle arm.
            clean_space = _fold_space(y_train_clean)
            if clean_space is None:
                continue
            X_tr_s, X_va_s = clean_space

            # no_noise: noise-independent, run once per fold (not per seed)
            no_noise_score = _evaluate_fold(
                X_tr_s, X_va_s, y_train_clean, y_val, 'no_noise',
                noise_fraction, model_factory, metric_fn,
                is_clf=is_clf, metric_name=metric_name)
            if no_noise_score is not None:
                per_strategy_scores['no_noise'].append(no_noise_score)
                per_strategy_scores_oracle['no_noise'].append(no_noise_score)

            # Noise-dependent strategies: average across NOISE_SEEDS
            for noise_seed in NOISE_SEEDS:
                actual_seed = 42 + fold_i * 1000 + noise_seed
                # Compute the noisy y and noise_scores ONCE per (fold, seed);
                # all 6 noise-dependent strategies reuse them.
                _spec = _noise_spec(dataset_name, noise_fraction, noise_level,
                                    fold_idx=fold_i, noise_seed=noise_seed,
                                    is_clf=is_clf)
                y_tr_noisy, _, _prov = inject_train_labels(
                    y_train_clean, _spec, is_clf=is_clf)
                noise_provenance.setdefault(_spec.spec_id, _prov)

                # The feature space every noisy condition is measured in, built
                # from THIS draw's noisy training labels. One build per (fold,
                # seed) — the recipe is fixed, so only the ranking and the
                # budget split are recomputed.
                noisy_space = _fold_space(y_tr_noisy)
                if noisy_space is None:
                    continue
                X_tr_n, X_va_n = noisy_space

                noise_scores_noisy = _compute_noise_scores(
                    X_tr_n, y_tr_noisy, is_clf=is_clf)
                noise_scores_clean_space = _compute_noise_scores(
                    X_tr_s, y_tr_noisy, is_clf=is_clf)

                for strategy in strategies:
                    if strategy == 'no_noise':
                        continue
                    needs_scores = strategy in (
                        'remove', 'downweight', 'label_smooth', 'knn_smooth')
                    for _acc, _Xtr, _Xva, _ns in (
                            (per_strategy_scores, X_tr_n, X_va_n,
                             noise_scores_noisy),
                            (per_strategy_scores_oracle, X_tr_s, X_va_s,
                             noise_scores_clean_space)):
                        fold_score = _evaluate_fold(
                            _Xtr, _Xva, y_train_clean, y_val, strategy,
                            noise_fraction, model_factory, metric_fn,
                            is_clf=is_clf, metric_name=metric_name,
                            noise_seed=actual_seed,
                            precomputed_y_tr=y_tr_noisy,
                            noise_scores=(_ns if needs_scores else None),
                            dataset_name=dataset_name, noise_level=noise_level)
                        if fold_score is not None:
                            _acc[strategy].append(fold_score)

        # Aggregate per strategy. All strategies now use the iteration's
        # model, so model_used == model_name throughout.
        print(f"\n  {model_name}/hybrid:")
        for strategy in strategies:
            fold_scores = per_strategy_scores[strategy]
            score = float(np.mean(fold_scores)) if fold_scores else float('nan')
            entry = {
                'score': score,
                'n_folds': len(fold_scores),
                'model_used': model_name,
                # Per-fold scores kept alongside the mean so honest
                # (held-out-aware) reads have the raw folds, not just the
                # selection-fold mean. Additive; 'score' type unchanged.
                'fold_scores': [float(x) for x in fold_scores],
            }
            results['per_model_per_strategy'][model_name][strategy] = entry

            # Same strategy in the clean-label feature space.
            o_scores = per_strategy_scores_oracle[strategy]
            o_mean = float(np.mean(o_scores)) if o_scores else float('nan')
            results['per_model_per_strategy_oracle_features'].setdefault(
                model_name, {})[strategy] = {
                    'score': o_mean,
                    'n_folds': len(o_scores),
                    'model_used': model_name,
                    'fold_scores': [float(x) for x in o_scores],
                }
            _gapstr = ('' if (np.isnan(o_mean) or np.isnan(score))
                       else f"  oracle_features={o_mean:.4f}")
            print(f"    {strategy:<18} {metric_name}={score:.4f} "
                  f"(n={len(fold_scores)}){_gapstr}")

    # Populate legacy per_strategy from RF (or first available model)
    _legacy_model = 'rf' if 'rf' in bp_by_model else next(iter(bp_by_model))
    for strategy in strategies:
        results['per_strategy'][strategy] = (
            results['per_model_per_strategy'][_legacy_model].get(strategy,
                {'score': float('nan'), 'n_folds': 0}))

    # --- Per-rep conditions (RF only, diagnostic). The headline claim is the
    # per-model hybrid recovery above; the per-rep section serves as a
    # baseline reference and would balloon to 6 × n_reps × 7 strategies if
    # iterated across DOWNSTREAM_MODELS_NOISE. Tag with the fixed model name
    # so downstream consumers don't misread these as multi-model.
    rep_model_factory = model_factories.get(
        'rf', next(iter(model_factories.values())))
    results['per_rep_model'] = 'rf' if 'rf' in model_factories else \
        next(iter(model_factories))
    # Restructured to (rep × fold × seed × strategy) so noise_scores and y_tr
    # are computed ONCE per (rep, fold, seed) and reused across all 6 noise-
    # dependent strategies. The previous (rep × strategy × fold × seed) order
    # recomputed them 4× per (fold, seed) — the same waste the hybrid block
    # avoids via `_compute_noise_scores` hoisting.
    for rep_name, X_rep in reps.items():
        per_strategy_scores_rep = {s: [] for s in strategies}

        for fold_i, (train_idx, val_idx) in enumerate(
                cv_split(groups, n_folds, sample_X, labels, mode=mode)):
            y_train_clean = labels[train_idx]
            y_val = labels[val_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_rep[train_idx])
            X_va_s = scaler.transform(X_rep[val_idx])

            # no_noise: noise-independent, run once per fold
            no_noise_score = _evaluate_fold(
                X_tr_s, X_va_s, y_train_clean, y_val, 'no_noise',
                noise_fraction, rep_model_factory, metric_fn,
                is_clf=is_clf, metric_name=metric_name)
            if no_noise_score is not None:
                per_strategy_scores_rep['no_noise'].append(no_noise_score)

            for noise_seed in NOISE_SEEDS:
                actual_seed = 42 + fold_i * 1000 + noise_seed
                _spec = _noise_spec(dataset_name, noise_fraction, noise_level,
                                    fold_idx=fold_i, noise_seed=noise_seed,
                                    is_clf=is_clf)
                y_tr_noisy, _, _prov = inject_train_labels(
                    y_train_clean, _spec, is_clf=is_clf)
                noise_provenance.setdefault(_spec.spec_id, _prov)
                noise_scores_cache = _compute_noise_scores(
                    X_tr_s, y_tr_noisy, is_clf=is_clf)

                for strategy in strategies:
                    if strategy == 'no_noise':
                        continue
                    needs_scores = strategy in (
                        'remove', 'downweight', 'label_smooth', 'knn_smooth')
                    fold_score = _evaluate_fold(
                        X_tr_s, X_va_s, y_train_clean, y_val, strategy,
                        noise_fraction, rep_model_factory, metric_fn,
                        is_clf=is_clf, metric_name=metric_name,
                        noise_seed=actual_seed,
                        precomputed_y_tr=y_tr_noisy,
                        noise_scores=(noise_scores_cache if needs_scores else None),
                        dataset_name=dataset_name, noise_level=noise_level)
                    if fold_score is not None:
                        per_strategy_scores_rep[strategy].append(fold_score)

        for strategy in strategies:
            key = f"{rep_name}/{strategy}"
            fold_scores = per_strategy_scores_rep[strategy]
            score = float(np.mean(fold_scores)) if fold_scores else float('nan')
            results['per_strategy'][key] = {
                'score': score,
                'n_folds': len(fold_scores),
                'fold_scores': [float(x) for x in fold_scores],
            }
            print(f"  {key}... {metric_name}={score:.4f}")

    # Summary: recovery = (strategy - noisy_baseline) / (no_noise - noisy_baseline)
    clean = results['per_strategy'].get('no_noise', {}).get('score', float('nan'))
    noisy = results['per_strategy'].get('noisy_baseline', {}).get('score', float('nan'))
    # Sign-normalised so a positive noise_gap always means the injected noise
    # cost the model something. Unguarded clean-minus-noisy is NEGATIVE on rmse
    # exactly when the noise did damage — the recovery denominator below already
    # knows this (see _hurt). Fixed 2026-09-10 (metric-direction audit).
    if any(np.isnan(x) for x in [clean, noisy]):
        gap = float('nan')
    else:
        gap = (noisy - clean) if lower_is_better else (clean - noisy)

    # Compute recovery for all conditions (hybrid and per-rep)
    # Recovery uses hybrid baselines (no_noise, noisy_baseline) as reference
    recovery = {}
    remediation_strategies = [s for s in strategies if s not in ('no_noise', 'noisy_baseline')]
    all_condition_keys = [k for k in results['per_strategy']
                          if not k.endswith('/no_noise') and not k.endswith('/noisy_baseline')
                          and k not in ('no_noise', 'noisy_baseline')]
    for key in all_condition_keys:
        s = results['per_strategy'].get(key, {}).get('score', float('nan'))
        # For per-rep conditions use the per-rep baselines. If a per-rep
        # baseline is missing (NaN), do NOT silently fall through to the
        # hybrid baselines — that would mix denominators and reintroduce
        # exactly the cross-denominator pollution that best_strategy_per_rep
        # was supposed to avoid. Mark recovery NaN instead so the reader
        # knows the per-rep number isn't trustworthy.
        if '/' in key:
            prefix = key.rsplit('/', 1)[0]
            ref_clean = results['per_strategy'].get(
                f'{prefix}/no_noise', {}).get('score', float('nan'))
            ref_noisy = results['per_strategy'].get(
                f'{prefix}/noisy_baseline', {}).get('score', float('nan'))
        else:
            ref_clean = clean
            ref_noisy = noisy
        ref_gap = (ref_clean - ref_noisy
                   if not any(np.isnan(x) for x in [ref_clean, ref_noisy])
                   else float('nan'))
        # A recovery percentage divides by clean-minus-noisy. That denominator
        # is only a "how much did the noise cost" number when the noise
        # actually cost something, and which sign that is depends on the
        # metric: for RMSE the clean score is the SMALLER one. When the sign
        # runs the other way the injected noise left the model no worse, and
        # dividing by it turns every strategy's number inside out — a strategy
        # that helped reads as negative recovery. buchwald_hartwig.json on disk
        # has exactly that. Report NaN and say why instead.
        _hurt = (ref_gap < 0) if lower_is_better else (ref_gap > 0)
        if (not np.isnan(s) and not np.isnan(ref_gap)
                and abs(ref_gap) > 1e-10 and _hurt):
            recovery[key] = float((s - ref_noisy) / ref_gap * 100)
        else:
            recovery[key] = float('nan')
            if not np.isnan(ref_gap) and not _hurt:
                results.setdefault('recovery_undefined', {})[key] = (
                    f'noise did not degrade this reference: clean={ref_clean:.6g}, '
                    f'noisy={ref_noisy:.6g}, lower_is_better={lower_is_better}')

    # Best strategy must be reported within a single denominator class —
    # hybrid recoveries use the hybrid (clean, noisy) baseline while per-rep
    # recoveries use their own per-rep baselines. Picking a global max across
    # both was meaningless: a per-rep 200% (tiny denominator) silently beat a
    # stable hybrid 60%. Report per-class winners instead.
    def _arg_max_finite(d):
        finite = {k: v for k, v in d.items() if v is not None and not np.isnan(v)}
        if not finite:
            return None
        return max(finite, key=finite.get)

    hybrid_recovery_only = {k: v for k, v in recovery.items() if '/' not in k}
    per_rep_recovery = {}
    for k, v in recovery.items():
        if '/' in k:
            rep_name = k.split('/', 1)[0]
            per_rep_recovery.setdefault(rep_name, {})[k.split('/', 1)[1]] = v

    best_strategy_per_rep = {
        rep: _arg_max_finite(rep_recoveries)
        for rep, rep_recoveries in per_rep_recovery.items()
    }

    # Per-model recovery: each model has its own (no_noise, noisy_baseline)
    # reference, computed against the same model. This is what the audit
    # actually wanted — "do these strategies recover lost performance" can
    # only be answered model-by-model.
    per_model_recovery = {}
    per_model_best = {}
    for m_name, m_strats in results['per_model_per_strategy'].items():
        m_clean = m_strats.get('no_noise', {}).get('score', float('nan'))
        m_noisy = m_strats.get('noisy_baseline', {}).get('score', float('nan'))
        m_gap = (m_clean - m_noisy
                 if not (np.isnan(m_clean) or np.isnan(m_noisy))
                 else float('nan'))
        # Same denominator rule as recovery_pct above: the sign of clean minus
        # noisy has to say the noise hurt, and which sign that is depends on
        # the metric's direction.
        m_hurt = (m_gap < 0) if lower_is_better else (m_gap > 0)
        m_recovery = {}
        for strat in remediation_strategies:
            s = m_strats.get(strat, {}).get('score', float('nan'))
            if (not np.isnan(s) and not np.isnan(m_gap)
                    and abs(m_gap) > 1e-10 and m_hurt):
                m_recovery[strat] = float((s - m_noisy) / m_gap * 100)
            else:
                m_recovery[strat] = float('nan')
        if not np.isnan(m_gap) and not m_hurt:
            results.setdefault('recovery_undefined', {})[m_name] = (
                f'noise did not degrade this model: clean={m_clean:.6g}, '
                f'noisy={m_noisy:.6g}, lower_is_better={lower_is_better}')
        per_model_recovery[m_name] = m_recovery
        per_model_best[m_name] = _arg_max_finite(m_recovery)

    results['summary'] = {
        'clean_score': clean,
        'noisy_score': noisy,
        'noise_gap': float(gap) if not np.isnan(gap) else None,
        'recovery_pct': recovery,
        # Hybrid (RF) and per-rep have different reference gaps; report separately.
        'best_hybrid_strategy': _arg_max_finite(hybrid_recovery_only),
        'best_strategy_per_rep': best_strategy_per_rep,
        # Canonical multi-model view: per-model recovery + per-model winners.
        'per_model_recovery_pct': per_model_recovery,
        'best_strategy_per_model': per_model_best,
        # per_model_recovery_pct and best_strategy_per_model cover only the
        # models that got a blueprint. Repeated here so the two keys are read
        # next to the list of models they do NOT cover, and why.
        'models_requested': models_requested,
        'models_scored': list(per_model_recovery.keys()),
        'models_missing_blueprint': models_missing_blueprint,
        # Conditions whose recovery could not be computed, and why. A recovery
        # percentage needs a reference where the injected noise actually cost
        # something; where it did not, the number is NaN rather than a
        # sign-flipped percentage.
        'recovery_undefined': results.get('recovery_undefined', {}),
    }
    print(f"\n  Clean: {clean:.4f}, Noisy: {noisy:.4f}, "
          f"Gap ({metric_name}, positive = the noise hurt): {gap:.4f}")
    hybrid_recovery = {k: round(v, 1) for k, v in recovery.items() if '/' not in k}
    print(f"  Recovery (hybrid): {hybrid_recovery}")
    for rep_name in reps:
        rep_recovery = {k.split('/')[-1]: v for k, v in recovery.items()
                        if k.startswith(f'{rep_name}/')}
        if rep_recovery:
            print(f"  Recovery ({rep_name}): {rep_recovery}")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('N07: Noise Remediation')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'n07_remediation', args.dataset)
