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

Three feature spaces are compared in both tracks: the base representation, a
hybrid built from the predictive-objective search, and a hybrid built from the
similarity-objective search (2026-09-12 decision: n13's pool subset includes
similarity hybrids, and the pool replaces the hand-built four-block stack this
file used to carry).

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
    build_fold_hybrid, fold_reselection, build_fold_reselection_input,
    hr_exclude_folds,
    # The top-2 selectors and the two hybrid deltas, added 2026-09-18. All four
    # read rows the search already stored and fit nothing.
    top_base_reps, top_hybrids, select_model_rep_pairs, hybrid_deltas,
)
from kirby.find_best_hybrid import OBJECTIVE_SIMILARITY
# 2026-09-12 decision: n13's pool subset includes similarity hybrids, and the
# pool replaces the hand-built stack. `_similarity_hybrid_space` (the stacked
# count + physchem-minus-saboteurs + 3D-shape + fine-2D matrix, with two of its
# four blocks chosen against MCS structural overlap) is no longer imported here.
# The file itself is untouched — n11 still imports it.
from experiments.dataset_policy import (
    noise_dose_units, noise_groups_for, reference_noise_level,
    noise_levels as dataset_noise_levels,
)
from experiments.noise.noise_helpers import (
    inject_train_labels, noise_spec_for, detection_scores,
)
from kirby.noise_spec import (
    CENSORING_STRATEGIES,
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

# 2026-09-12 decision: the part of the precomputed search pool n13 may take a
# hybrid from includes the searches run under objective='similarity' — the ones
# whose features were chosen so that the distance geometry predicts each
# molecule's activity from its five nearest neighbours — not only the
# prediction-built ones. This is NOT the distance=True flag (a different
# evaluator and budget lineage, still prediction-built); n13 does not pass it.
# The two families' stored scores are in different units, so the pool does not
# rank one against the other: it returns the first family named here as the
# winner and every family's own best under metadata['by_objective'], which is
# where the hybrid_sim arm below gets its recipe.
POOL_SUBSET = 'predictive_and_similarity'

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
# they cannot participate in the Track-B sample-weight remediation track. The
# loop below adds any model that raises TypeError on a weighted fit. The seed is
# empty: it used to hold 'dnn', because kirby/models/dnn.py was
# fit(self, X, y, groups=None), and on 2026-09-17 both DNNRegressor.fit and
# DNNClassifier.fit gained sample_weight. A seeded name is never tried, so
# leaving 'dnn' here would have kept it out of track_b after the fix.
# Rows for these models are intentionally absent from track_b (not recorded as a
# zero-recovery result) and the names are surfaced in
# results['track_b_unweighted_models'].
UNWEIGHTED_TRACK_B_SEED = frozenset()
UNWEIGHTED_TRACK_B_MODELS = set(UNWEIGHTED_TRACK_B_SEED)


def _bp_sig(bp):
    """Stable recipe id for a blueprint.

    Doubles as (a) the fold_reselection candidate 'name' (per-fold winner label)
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
def _build_spaces(train_reps, val_reps, y_train, augmentations_train,
                  augmentations_val, bp, base_rep_name, groups=None,
                  model=None, metric_fn=None, lower_is_better=None,
                  objective=None, is_clf=None):
    """Build 2 spaces: base rep + hybrid, with the hybrid's val features.

    The hybrid goes through build_fold_hybrid, not a bare create_hybrid call.
    The bare call passed only allocation_method / importance_method /
    internal_evaluator / budget / augmentations / augmentation_strategy, so
    accept-or-reject inside the build was scored by the internal evaluator's
    r2 rather than by fitting the real downstream model and calling the
    dataset's metric, and candidate_ranking, augmentation_margin,
    augmentation_fraction and aug_per_source_budget were dropped — the rebuild
    was not the recipe the search stored. build_fold_hybrid forwards all of
    them, and it returns the validation features too, so the separate replay
    of apply_feature_selection on the val reps is gone.

    Args:
        train_reps: {rep_name: X_train} already sliced to fold.
        val_reps: {rep_name: X_val} already sliced to fold.
        y_train: Training labels for this fold.
        augmentations_train: {aug_name: X_train} already sliced, or None.
        augmentations_val: {aug_name: X_val} already sliced, or None.
        bp: Blueprint from get_precomputed_search.
        base_rep_name: Name of base representation.
        model: The downstream estimator INSTANCE the accept-or-reject step
            scores with, or None to fall back on the internal evaluator.
        metric_fn / lower_is_better: The dataset's metric and its direction,
            used with `model`.
        objective: The search family this recipe came from. Under
            objective='similarity' create_hybrid fits no model, so the model
            and the dataset metric are refused there and the objective goes in
            instead.

    Returns (spaces, hybrid_val_scaled_or_None).
    """
    spaces = {}
    hybrid_val_s = None

    X_base = train_reps[base_rep_name]
    sc = StandardScaler()
    spaces[base_rep_name] = sc.fit_transform(X_base)

    # The pool subset offers both search families, so read which one this
    # recipe came from before deciding what may be passed. build_fold_hybrid
    # raises when a similarity-objective recipe is handed a model or a metric,
    # and it falls back on the recipe's own objective field the same way.
    _obj = objective if objective is not None else getattr(bp, 'objective', None)
    if _obj == OBJECTIVE_SIMILARITY:
        _obj_kw = {'objective': OBJECTIVE_SIMILARITY, 'is_clf': is_clf}
    else:
        _obj_kw = {'model': model, 'metric_fn': metric_fn,
                   'lower_is_better': lower_is_better}

    try:
        X_h, X_h_val, _ = build_fold_hybrid(
            train_reps, val_reps, y_train, bp,
            augmentations_train=augmentations_train,
            augmentations_val=augmentations_val,
            groups=groups, **_obj_kw)
        scaler_h = StandardScaler()
        spaces['hybrid'] = scaler_h.fit_transform(X_h)
        hybrid_val_s = scaler_h.transform(X_h_val)
    except Exception as e:
        print(f"      hybrid failed: {e}")

    return spaces, hybrid_val_s


def _mean_detection(rows):
    """Average the detection readings over the (fold, draw) passes."""
    if not rows:
        return None
    keys = ('detection_average_precision', 'detection_precision_at_k',
            'detection_recall_at_k', 'detection_n_corrupted', 'detection_n_training')
    out = {}
    for k in keys:
        vals = [float(r[k]) for r in rows if r.get(k) is not None
                and np.isfinite(float(r[k]))]
        out[k] = float(np.mean(vals)) if vals else float('nan')
    out['detection_condition'] = rows[0].get('detection_condition')
    out['detection_n_passes'] = len(rows)
    return out


def _top2_and_deltas(search, models, where):
    """The top 2 base representations, the top 2 hybrids and the two hybrid
    deltas, per downstream model. 0 new model fits.

    Added 2026-09-18 on the owner's instruction: the top 2 go in beside what
    this file already reports, and nothing already reported moves. The six keys
    returned here are new and nothing reads them -- not the summary, not the
    printed output, not any other experiment.

    Every selector reads rows the search already stored: baselines[rep][model]
    (find_best_hybrid.py:5524-5525) and all_results (:1536). Nothing is fitted,
    so the whole block costs 0 model fits and only dict passes in wall-clock
    time. select_model_rep_pairs walks those same two structures a second time,
    which is why it is called rather than reassembled from the per-model
    results above it.

    require_full=False on every call. A downstream model the search never
    scored -- svm past its 10,000-molecule cap (find_best_hybrid.py:2969-2978),
    or one a registered skip policy dropped (:2841-2848) -- then comes back as
    a short list carrying its own shortfall_reason. Under the default True it
    raises ValueError instead, which would end a run that has already spent
    hours of fits, for the sake of an extra.

    `hybrid=` is handed to hybrid_deltas rather than left to its default. The
    default path reads `_picked['selected'][0]['row']` (_shared.py:7432) and
    the rows top_hybrids builds carry no 'row' key (_shared.py:7043-7066), so
    the default raises KeyError. The row passed in is that same top-ranked one,
    and it carries the 'score_mean' and 'blueprint' hybrid_deltas reads.

    'blueprint' is dropped from every stored row. A Blueprint is the whole
    recipe (find_best_hybrid.py:1344) and save_experiment_results json.dumps
    this dict (_shared.py:8021), so leaving the object in would write a full
    recipe into each of the 4 rows per downstream model.
    """
    def _no_blueprint(rows):
        return [{k: v for k, v in r.items() if k != 'blueprint'} for r in rows]

    block = {
        'top2_extra_new_model_fits': 0,
        'top2_base_reps': {},
        'top2_hybrids': {},
        'top2_pairs_by_model': {},
        'hybrid_deltas_by_model': {},
        'top2_extra_failures': {},
    }

    for m in models:
        ctx = '{}/{}'.format(where, m)
        try:
            _b = top_base_reps(search, m, n=2, require_full=False, context=ctx)
            block['top2_base_reps'][m] = {
                'selected': _no_blueprint(_b['selected']),
                'n_returned': _b['n_returned'],
                'n_reps_in_baselines': _b['n_reps_in_baselines'],
                'n_reps_scored_for_model': _b['n_reps_scored_for_model'],
                'score_basis': _b['score_basis'],
                'tie_at_cut': _b['tie_at_cut'],
                'shortfall_reason': _b['shortfall_reason'],
            }
        except Exception as exc:
            block['top2_extra_failures']['top_base_reps/%s' % m] = \
                '%s: %s' % (type(exc).__name__, exc)
            print(f"    top-2 base representations skipped for {m}: "
                  f"{type(exc).__name__}: {exc}")

        _h = None
        try:
            _h = top_hybrids(search, m, n=2, require_full=False, context=ctx)
            block['top2_hybrids'][m] = {
                'selected': _no_blueprint(_h['selected']),
                'n_returned': _h['n_returned'],
                'score_basis': _h['score_basis'],
                'ranked_over': _h['ranked_over'],
                'tie_at_cut': _h['tie_at_cut'],
                'notes': _h['notes'],
                'shortfall_reason': _h['shortfall_reason'],
            }
        except Exception as exc:
            _h = None
            block['top2_extra_failures']['top_hybrids/%s' % m] = \
                '%s: %s' % (type(exc).__name__, exc)
            print(f"    top-2 hybrids skipped for {m}: "
                  f"{type(exc).__name__}: {exc}")

        # The two reference numbers for this model's top-ranked hybrid: against
        # the mean of the representations the recipe's per-fold budget split
        # actually gave features to, and against that model's single best base
        # representation. Positive means the hybrid is better, on either metric
        # direction. A representation listed in the recipe but given 0 features
        # on every fold is left out of the first one.
        try:
            if _h is not None and _h['selected']:
                block['hybrid_deltas_by_model'][m] = hybrid_deltas(
                    search, m, hybrid=_h['selected'][0], context=ctx)
        except Exception as exc:
            block['top2_extra_failures']['hybrid_deltas/%s' % m] = \
                '%s: %s' % (type(exc).__name__, exc)
            print(f"    hybrid deltas skipped for {m}: "
                  f"{type(exc).__name__}: {exc}")

    # The same two selectors again, across every downstream model at once, for
    # the counts that exist only across models: how many of the 4 rows per
    # model came back, and which representations or recipes two models both
    # picked. 'per_model' is deliberately not kept -- it is those same rows a
    # second time, with the Blueprint objects still in them.
    try:
        _p = select_model_rep_pairs(search, models=list(models), n_base=2,
                                    n_hybrids=2, require_full=False,
                                    context=where)
        block['top2_pairs_by_model'] = {
            'pairs': _p['pairs'],
            'n_pairs': _p['n_pairs'],
            'n_pairs_possible': _p['n_pairs_possible'],
            'n_distinct_units': _p['n_distinct_units'],
            'collapsed': _p['collapsed'],
            'shortfalls': _p['shortfalls'],
        }
    except Exception as exc:
        block['top2_extra_failures']['select_model_rep_pairs'] = \
            '%s: %s' % (type(exc).__name__, exc)
        print(f"    top-2 grid skipped: {type(exc).__name__}: {exc}")

    return block


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
        # CENSORING JOINED 2026-09-12, owner's decision. It is the one condition
        # that moves labels in ONE direction: every label above a cut-point is
        # replaced by that cut-point, so the corrupted molecules are chosen by
        # their own value. The sister study measured it at roughly twelve times
        # any difference between the zero-mean shapes. It was excluded here only
        # because this file swept DOSE_MATCHED_STRATEGIES, and censoring's level
        # is a fraction of labels clipped rather than an amount of error, so it
        # cannot share the level axis. It gets its own level grid below.
        noise_types = (list(CLASSIFICATION_STRATEGIES) if is_clf
                       else list(DOSE_MATCHED_STRATEGIES) + list(CENSORING_STRATEGIES))
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
        strengths = list(dataset_noise_levels(dataset_name, is_clf=is_clf))
    noise_provenance = {}

    print(f"  Base rep: {base_rep_name}")
    print(f"  Noise conditions: {noise_types}")
    censoring_levels = list(kwargs.get('censoring_levels')
                            or dataset_noise_levels(dataset_name,
                                                    strategy='censoring'))

    def _levels_for_condition(condition):
        """The level grid this condition sweeps.

        Ten conditions sweep an AMOUNT of error, as a fraction of the clean label
        spread. Censoring sweeps a FRACTION OF LABELS CLIPPED. They are different
        quantities and putting them on one axis was the defect the sister study
        called 'a knob that means different things in different conditions'.
        """
        return censoring_levels if condition in CENSORING_STRATEGIES else strengths

    print(f"  Levels ({noise_dose_units(dataset_name)}): {strengths}")
    if any(c in CENSORING_STRATEGIES for c in noise_types):
        print(f"  Censored fractions (censoring only): {censoring_levels}")
    print(f"  LNL methods: {lnl_methods}")
    print(f"  Estimators: {estimators}")
    print(f"  Folds: {n_folds}")

    # Get blueprint ONCE from pool helper (uses clean labels)
    # 2026-09-12 decision: ask the pool for the widened subset — searches run
    # under objective='similarity' are eligible here alongside the
    # prediction-built ones. The returned winner is still the predictive one
    # whenever that family is on disk, because the two families' scores are in
    # different units and the pool takes the first family the subset names.
    search, meta = get_best_search_for_downstream(
        reps, labels, groups,
        downstream_model='rf', budget=None,
        augmentations=augmentations,
        dataset_name=dataset_name, task_type=task_type,
        data=data,
        pool_subset=POOL_SUBSET,
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
    print(f"  Pool subset: {meta.get('pool_subset')} — 'hybrid' recipe from "
          f"the {meta.get('objective')} search "
          f"(chosen by {meta.get('objective_choice')}, stored "
          f"{meta.get('metric_name')})")

    # Extra arm: the similarity-objective hybrid, 'hybrid_sim'.
    #
    # 2026-09-12 decision: n13's pool subset includes similarity hybrids, and
    # the pool replaces the hand-built stack. Until today this arm was the
    # four-block matrix from _similarity_hybrid_space — count +
    # physchem-minus-saboteurs + 3D shape + fine-2D, with the physchem and
    # fine-2D blocks chosen against MCS structural overlap over sampled
    # molecule pairs, and no activity label anywhere in the choice. It now
    # comes from the stored search run under objective='similarity' for this
    # dataset, the same way the predictive hybrid comes from the search run
    # under objective='predictive'. Two things change with it: the recipe is a
    # searched allocation method / budget / importance method / internal
    # evaluator instead of a fixed four-block stack, and the features are
    # refit per fold on that fold's noisy training labels rather than chosen
    # label-blind.
    #
    # The arm is on only when that search family is on disk for this dataset;
    # tests/precompute_hybrids.py <dataset> --with-similarity writes it.
    _by_objective = (meta.get('by_objective') or {}) if meta else {}
    sim_entry = _by_objective.get(OBJECTIVE_SIMILARITY)
    bp_sim = sim_entry['best_config']['blueprint'] if sim_entry else None
    sim_aug_condition = sim_entry.get('aug_condition') if sim_entry else None
    sim_arm_on = bp_sim is not None
    sim_build_failures = {}
    if sim_arm_on:
        if getattr(bp_sim, 'augmentation_strategy', None) != sim_aug_condition:
            try:
                bp_sim.augmentation_strategy = sim_aug_condition
            except Exception:
                pass
        print(f"  hybrid_sim arm ON: recipe from the similarity-objective "
              f"search ({sim_entry['tier']}/{sim_entry['aug_condition']}, "
              f"stored {sim_entry['metric_name']}="
              f"{sim_entry['score_mean']:.4f}), rebuilt per fold on that "
              f"fold's noisy training labels")
    else:
        print("  hybrid_sim arm OFF: this dataset has no stored search under "
              "objective='similarity' (write one with "
              "tests/precompute_hybrids.py <dataset> --with-similarity)")

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

    # The estimator the 'hybrid' space's accept-or-reject step is scored with.
    # `bp` was fetched with downstream_model='rf', so the rebuild is scored by
    # fitting the same model the search fitted and calling the dataset's own
    # metric. None when this dataset's gating dropped rf, and then
    # build_fold_hybrid falls back to the internal evaluator the way the old
    # bare create_hybrid call always did.
    bp_objective = meta.get('objective') if meta else None
    hybrid_model_factory = model_factories.get('rf')

    # ── Per-model MATCHED blueprints for the honest hybrid-vs-baseline arm ──
    # The single RF clean-label blueprint above (`bp`) still drives Track A
    # (DNN-only) and the Track-B estimator geometry, UNCHANGED. The audit note
    # is that every model reused that one RF-selected blueprint. For the new
    # honest per-fold re-score arm below we additionally fetch a MATCHED-
    # evaluator blueprint per downstream model (n07 pattern; each pool call is
    # cached), so each model's hybrid space is built with a recipe tuned for
    # that model rather than reusing RF's. Distinct blueprints are de-duplicated
    # per fold (via _bp_sig) before the create_hybrid call to bound cost.
    #
    # 2026-09-12 decision: this resolution takes the widened pool subset too —
    # every place n13 picks a hybrid out of the pool offers the
    # similarity-built searches, not only the prediction-built ones. Which
    # family each model's recipe came from is recorded in
    # results['reselected_hybrid_objective_by_model'].
    bp_by_model = {}
    aug_by_model = {}
    objective_by_model = {}
    for _mname in list(model_factories.keys()):
        try:
            _s_m, _m_m = get_best_search_for_downstream(
                reps, labels, groups,
                downstream_model=_mname, budget=None,
                augmentations=augmentations,
                dataset_name=dataset_name, task_type=task_type,
                data=data,
                pool_subset=POOL_SUBSET,
            )
        except Exception as _exc:
            print(f"  WARNING: matched blueprint fetch failed for {_mname} ({_exc})")
            continue
        if _m_m is None:
            continue
        _bp_m = _m_m['best_config']['blueprint']
        objective_by_model[_mname] = _m_m.get('objective')
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
            objective_by_model[_mname] = (meta.get('objective') if meta
                                          else None)

    # Reset the unweighted-model registry to its seed for this run (module-level
    # mutable set persists across run() calls in a single process).
    global UNWEIGHTED_TRACK_B_MODELS
    # From the one seed above. This line held its own literal {'dnn'} until
    # 2026-09-17, so the module constant and the per-run reset could disagree.
    UNWEIGHTED_TRACK_B_MODELS = set(UNWEIGHTED_TRACK_B_SEED)

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
        # 2026-09-12 decision: the pool subset this file draws its hybrids from
        # includes the similarity-built searches. Which family each hybrid
        # actually came from is recorded here, so a reader can tell the two
        # apart. 'objective' is the family of the recipe driving the 'hybrid'
        # space (Track A and Track B) and 'objective_choice' says how the pool
        # picked it: 'only_family' when just one family was on disk,
        # 'subset_order' when both were and their scores are in different
        # units, 'score' when they share a metric.
        'pool_subset': (meta.get('pool_subset') if meta else POOL_SUBSET),
        'hybrid_objective': (meta.get('objective') if meta else None),
        'hybrid_objective_choice': (meta.get('objective_choice') if meta
                                    else None),
        'hybrid_metric_name': (meta.get('metric_name') if meta else None),
        'pool_by_objective': convert_numpy({
            _o: {'tier': _d.get('tier'),
                 'aug_condition': _d.get('aug_condition'),
                 'score_mean': _d.get('score_mean'),
                 'metric_name': _d.get('metric_name'),
                 'lower_is_better': _d.get('lower_is_better')}
            for _o, _d in _by_objective.items()}),
        # Family of the per-model matched recipe used by the honest per-fold
        # arm below, one entry per downstream model.
        'reselected_hybrid_objective_by_model': dict(objective_by_model),
        'per_noise_type': {},
    }

    # ── Top-2 rows and the two hybrid deltas, beside what is reported ──
    # Owner's instruction 2026-09-18: add the top 2 as an extra and change
    # nothing already reported. 'top3_base_reps' above is untouched and still
    # holds the three names extract_top3_base_reps ranked on the rf baseline;
    # these keys sit next to it.
    #
    # One search answers for every downstream model. It was fetched with
    # downstream_model='rf', but search.baselines and search.all_results carry
    # a score under every downstream model the search scored, so the selectors
    # read the other four out of the same object. 0 new model fits.
    if search is not None:
        results.update(_top2_and_deltas(
            search, list(model_factories.keys()), f'n13/{dataset_name}'))
    else:
        results['top2_extra_absent_reason'] = (
            'the pool returned a recipe but no SearchResult, so there are no '
            'stored rows to rank and no baseline scores to difference against')

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

        # 2026-09-12: nothing is precomputed per fold for hybrid_sim any more.
        # The hand-built four-block stack was label-blind, so one build per
        # fold served every noise condition. The pool recipe that replaced it
        # selects features from the labels, so it is rebuilt inside the seed
        # loop below from that fold's noisy training labels, exactly like the
        # predictive hybrid.
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
    canonical_reselected = None
    canonical_reselected_key = None
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

        # One axis for the ten dose-matched conditions; censoring has its own,
        # because its level counts labels clipped and theirs measures an amount of
        # error. The corrupted fraction that used to need a third axis is carried
        # by the condition itself (outlier_p01/p05/p10).
        for strength in _levels_for_condition(noise_type):
            strength_key = str(strength)

            noisy_baseline_by_model = {mn: [] for mn in model_factories}
            # {space: {method: [scores]}} for track A
            track_a_acc = {}
            # {space: {estimator: {model: [scores]}}} for track B
            track_b_acc = {}
            # {space: {estimator: [detection reading per (fold, draw)]}}. Stays
            # empty unless the condition is one with a corrupted subset.
            detection_acc = {}

            # Honest per-fold arms for the shared fold_reselection helper:
            #   reselected_base_folds[model][fold] = [per-seed base-rep scores]
            #   reselected_hyb_folds[model][fold]  = [per-seed matched-hybrid scores]
            # Each fold entry is averaged over NOISE_SEEDS below, then laid out
            # fold-aligned so fold_reselection can leave-one-fold-out re-select
            # each arm's winner WITHOUT the held-out fold (the arithmetic
            # embodiment of the n09 nested-selection template). We do NOT run a
            # per-fold find_best_hybrid inner search here: that would multiply
            # the noise_type x strength x fold x seed matrix by a full search
            # and bust the 7-day wall; the LOFO re-selection over the cached
            # matched-blueprint candidates gives the held-out-fold-free
            # selection instead.
            reselected_base_folds = {mn: {} for mn in model_factories}
            reselected_hyb_folds = {mn: {} for mn in model_factories}
            reselected_hyb_name = {}  # model -> its blueprint signature (candidate name)

            for fi, fd in enumerate(fold_data):
                for noise_seed in NOISE_SEEDS:
                    # One seed per (fold, noise draw) for the label-noise
                    # learners below. It was referenced there and never
                    # defined, so every Track A method raised NameError into
                    # a bare except and Track A produced nothing at all.
                    _seed_combined = 42 + 1000 * fi + noise_seed
                    # Per-(condition, level, fold, seed) noise realization
                    _spec = noise_spec_for(dataset_name, noise_type, strength,
                                           noise_seed=noise_seed,
                                           is_clf=is_clf)
                    _tr_idx = fd['train_idx']
                    y_noisy, _cmask, _prov = inject_train_labels(
                        labels, _spec, train_idx=_tr_idx, is_clf=is_clf,
                        groups=noise_groups, dataset_name=dataset_name)
                    noise_provenance.setdefault(_spec.spec_id, _prov)

                    # Noisy baseline per model
                    for mname, mfactory in model_factories.items():
                        try:
                            m = mfactory()
                            m.fit(fd['X_eval_train_s'], y_noisy)
                            _sc_base = float(metric_fn(
                                fd['y_val'], m.predict(fd['X_eval_val_s'])))
                            noisy_baseline_by_model[mname].append(_sc_base)
                            # Per-fold base-rep arm for fold_reselection (reuses
                            # this same fit — no extra training cost).
                            reselected_base_folds[mname].setdefault(fi, []).append(_sc_base)
                        except Exception:
                            pass

                    # Build spaces from NOISY labels (realistic — no oracle leak)
                    spaces, hybrid_val_s = _build_spaces(
                        fd['train_reps_fold'], fd['val_reps_fold'], y_noisy,
                        fd['train_augs'], fd['val_augs'],
                        bp, base_rep_name,
                        groups=(groups[fd['train_idx']]
                                if groups is not None else None),
                        model=(hybrid_model_factory()
                               if hybrid_model_factory is not None else None),
                        metric_fn=metric_fn,
                        lower_is_better=lower_is_better,
                        objective=bp_objective, is_clf=is_clf)

                    # similarity-objective hybrid arm. 2026-09-12 decision: the
                    # pool replaces the hand-built four-block stack, so this is
                    # the recipe the similarity-objective search picked, refit
                    # on this fold's training rows and the same noisy labels
                    # the predictive hybrid above sees.
                    #
                    # No downstream model and no dataset metric go into the
                    # build: under objective='similarity' create_hybrid scores
                    # its own neighbour geometry and fits no model, and
                    # build_fold_hybrid refuses a model/metric_fn there.
                    #
                    # The old stack was taken as built, not rescaled, because
                    # each of its four blocks carried a weight in the distance.
                    # A pool-built hybrid has no such blocks, so it is scaled
                    # the same way the predictive hybrid is.
                    hybrid_sim_val_s = None
                    if bp_sim is not None:
                        try:
                            if sim_aug_condition != 'none' and augmentations:
                                _tr_aug_s = {k: v[fd['train_idx']]
                                             for k, v in augmentations.items()}
                                _va_aug_s = {k: v[fd['val_idx']]
                                             for k, v in augmentations.items()}
                            else:
                                _tr_aug_s = _va_aug_s = None
                            _Xs_tr, _Xs_va, _ = build_fold_hybrid(
                                fd['train_reps_fold'], fd['val_reps_fold'],
                                y_noisy, bp_sim,
                                augmentations_train=_tr_aug_s,
                                augmentations_val=_va_aug_s,
                                groups=(groups[fd['train_idx']]
                                        if groups is not None else None),
                                objective=OBJECTIVE_SIMILARITY,
                                is_clf=is_clf)
                            _sc_sim = StandardScaler()
                            spaces['hybrid_sim'] = _sc_sim.fit_transform(_Xs_tr)
                            hybrid_sim_val_s = _sc_sim.transform(_Xs_va)
                        except Exception as _exc:
                            _key_f = f"{type(_exc).__name__}: {_exc}"[:200]
                            if _key_f not in sim_build_failures:
                                print(f"      hybrid_sim build failed: {_key_f}")
                            sim_build_failures[_key_f] = \
                                sim_build_failures.get(_key_f, 0) + 1

                    # ── Honest hybrid arm: train each model PLAINLY (no LNL,
                    # no sample weights) on its MATCHED hybrid space, built from
                    # that model's blueprint on the NOISY labels. This is the
                    # hybrid candidate the fold_reselection helper compares against
                    # the base-rep baseline arm. Identical blueprints across
                    # models share one create_hybrid build via _hyb_space_cache.
                    _hyb_space_cache = {}  # bp_sig -> (X_tr_s, X_va_s) or None
                    for mname, mfactory in model_factories.items():
                        _bp_m = bp_by_model.get(mname)
                        if _bp_m is None:
                            continue
                        # 2026-09-12: the recipe name carries the search family
                        # it came from, so two families that happen to share an
                        # allocation method / budget / importance method /
                        # internal evaluator are still told apart in the
                        # per-fold score block.
                        _obj_m = objective_by_model.get(mname)
                        _sig = f"{_obj_m}|{_bp_sig(_bp_m)}"
                        reselected_hyb_name[mname] = _sig
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
                                # A recipe from the similarity family is refit
                                # under its own objective, which fits no
                                # downstream model — passing the model and the
                                # dataset metric there is refused, so they go
                                # in only for a prediction-built recipe.
                                if _obj_m == OBJECTIVE_SIMILARITY:
                                    _obj_kw = {'objective': OBJECTIVE_SIMILARITY,
                                               'is_clf': is_clf}
                                else:
                                    _obj_kw = {'model': mfactory(),
                                               'metric_fn': metric_fn,
                                               'lower_is_better': lower_is_better}
                                _Xh_tr, _Xh_va, _ = build_fold_hybrid(
                                    fd['train_reps_fold'], fd['val_reps_fold'],
                                    y_noisy, _bp_m,
                                    augmentations_train=_tr_aug,
                                    augmentations_val=_va_aug,
                                    groups=(groups[fd['train_idx']] if groups is not None else None),
                                    **_obj_kw)
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
                            reselected_hyb_folds[mname].setdefault(fi, []).append(
                                float(metric_fn(fd['y_val'], m.predict(_Xh_va_s))))
                        except Exception:
                            pass

                    # The hybrid's val features come back from _build_spaces
                    # with the train side, so there is no second build and no
                    # replay of the feature selection on the val reps here.

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
                                # How well this estimator FOUND the corrupted
                                # molecules, before any repair. Owner's decision
                                # 2026-09-12. Absent under the six conditions where
                                # every training molecule moves. Costs no model fit.
                                _det = detection_scores(_cmask, noise_scores,
                                                        noise_type)
                                if _det is not None:
                                    (detection_acc
                                     .setdefault(space_name, {})
                                     .setdefault(est_name, [])
                                     .append(_det))
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
                            # Same difference, sign-normalised so a positive
                            # number always means this method beat the clean-label
                            # DNN. The raw one above runs the dataset's own way
                            # and reads backwards on rmse. Added 2026-09-10
                            # (metric-direction audit).
                            'delta_from_clean_dnn_oriented': float(
                                (clean_mean_dnn - mean_s) if lower_is_better
                                else (mean_s - clean_mean_dnn))
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

            # ── Per-fold scores for this (noise_type, strength) ──
            # Fold-aligned per-arm scores (mean over NOISE_SEEDS within each
            # fold; np.nan for folds a model failed on), in the shape the shared
            # fold_reselection helper consumes.
            #
            # This is NOT a re-selection. fold_reselection's job is to re-pick
            # each side's winner from the folds other than the held-out one, and
            # it needs a list of recipes to pick from. n13 gives it ONE hybrid
            # recipe per model and ONE base representation, so there is nothing
            # to re-pick and the arithmetic returns those same two numbers. The
            # key used to be called 'fold_reselection', which promised the
            # re-selection; it is now named for what it holds.
            #
            # That one recipe per model came from the clean-label search pool
            # (get_best_search_for_downstream above), like every recipe in this
            # suite. Making the re-selection real means feeding every searched
            # recipe's per-fold scores, which costs one refit per
            # noise_type x strength x fold x seed per recipe.
            hr_candidates = {}
            hr_baselines = {}
            for mn in model_factories:
                base_map = reselected_base_folds.get(mn, {})
                hyb_map = reselected_hyb_folds.get(mn, {})
                if any(base_map.get(fi) for fi in range(n_folds)):
                    hr_baselines[mn] = {base_rep_name: [
                        float(np.mean(base_map[fi]))
                        if base_map.get(fi) else float('nan')
                        for fi in range(n_folds)]}
                if any(hyb_map.get(fi) for fi in range(n_folds)):
                    hr_candidates[mn] = [{
                        'name': reselected_hyb_name.get(mn, 'hybrid'),
                        'fold_scores': [
                            float(np.mean(hyb_map[fi]))
                            if hyb_map.get(fi) else float('nan')
                            for fi in range(n_folds)],
                    }]
            reselection_block = {
                'lower_is_better': lower_is_better,
                # Every fold counts. The candidates here are the arms this file built itself,
                # not a search's recipe list, so no fold filtered them.
                'exclude_folds': [],
                'candidates_by_model': hr_candidates,
                'baselines_by_model': hr_baselines,
                # One hybrid recipe per model and one base representation, so
                # no leave-one-fold-out re-selection happens. Stated here so a
                # reader of the file does not have to infer it from the list
                # lengths.
                'n_candidate_recipes_per_model': {
                    mn: len(v) for mn, v in hr_candidates.items()},
                'reselection': 'none_single_recipe_per_model',
                'recipe_source': 'clean_label_search_pool',
            }
            strength_result['per_fold_scores_single_recipe'] = reselection_block

            # Stash the canonical top-level slice (moderate gaussian noise) so a
            # reader can json.load(...)['per_fold_scores_single_recipe'] directly.
            _is_canonical = (noise_type == 'gaussian'
                             and abs(strength - _canonical_level) < 1e-9)
            if _is_canonical or canonical_reselected is None:
                canonical_reselected = reselection_block
                canonical_reselected_key = f'{noise_type}:{strength_key}'

            if detection_acc:
                strength_result['detection'] = {
                    _sp: {_e: _mean_detection(_rows) for _e, _rows in _by_est.items()}
                    for _sp, _by_est in detection_acc.items()}

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

    # ── Top-level per-fold score block ──
    # A sensitivity study has no single condition; expose a CANONICAL moderate-
    # noise slice here (gaussian sigma=0.3 when available, else the first
    # computed cell). The full per-(noise_type, strength) set is preserved under
    # per_noise_type[*]['per_strength'][*]['per_fold_scores_single_recipe'].
    #
    # Deliberately NOT written under the key 'fold_reselection': that name says
    # the winning recipe was re-picked without the fold it is scored on, and
    # here there is only one recipe per model to pick. The delta printed below
    # is one hybrid recipe against one base representation, fold for fold.
    if canonical_reselected is not None:
        results['per_fold_scores_single_recipe'] = convert_numpy(canonical_reselected)
        results['per_fold_scores_condition'] = canonical_reselected_key
        # Sanity log via the shared helper (pure arithmetic, no model fitting).
        try:
            _hr = fold_reselection(canonical_reselected['candidates_by_model'],
                                 canonical_reselected['baselines_by_model'],
                                 lower_is_better,
                                 exclude_folds=hr_exclude_folds(canonical_reselected))
            print(f"\n  === PER-FOLD SCORES, ONE RECIPE PER MODEL "
                  f"({canonical_reselected_key}) ===")
            print("  No re-selection: one hybrid recipe and one base "
                  "representation, so the leave-one-fold-out re-pick is a "
                  "no-op.")
            for _mn, _r in _hr.items():
                print(f"    {_mn}: hybrid={_r['hybrid_reselected']:.4f} "
                      f"base={_r['best_single_reselected']:.4f} "
                      f"delta={_r['delta']:+.4f} single_fold={_r['single_fold']}")
        except Exception as _exc:
            print(f"  per-fold score summary skipped ({_exc})")

    # Where the 'hybrid_sim' space came from. 2026-09-12 decision: the pool
    # replaces the hand-built four-block stack, so this records the stored
    # similarity-objective search the recipe was read from, not a molecule
    # cache. 'recipe_refit_rows' = the rows the features were refit on, one
    # rebuild per (noise condition, level, fold, noise seed).
    results['hybrid_sim'] = convert_numpy({
        'arm_on': bool(sim_arm_on),
        'source': 'pool_search_objective_similarity',
        'objective': OBJECTIVE_SIMILARITY if sim_arm_on else None,
        'tier': sim_entry.get('tier') if sim_entry else None,
        'aug_condition': sim_aug_condition,
        'search_score_mean': sim_entry.get('score_mean') if sim_entry else None,
        'search_metric_name': (sim_entry.get('metric_name') if sim_entry
                               else None),
        'recipe_refit_rows': 'fold_train_rows_noisy_labels' if sim_arm_on else None,
        'skip': (None if sim_arm_on else
                 "no stored search under objective='similarity' for this "
                 "dataset"),
        'build_failures': dict(sim_build_failures),
    })

    # Summary
    results['summary'] = _compute_summary(results)
    s = results['summary']
    print(f"\n  === SUMMARY ===")
    print(f"  Gaussian rank transfers: {s.get('gaussian_rank_transfers')}")
    print(f"  Hybrid advantage persists: {s.get('hybrid_advantage_persists')}")
    print(f"  Similarity-objective hybrid vs base rep: "
          f"{s.get('hybrid_sim_advantage_persists')}")
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
    # 2026-09-12: 'hybrid_sim' — the space built from the similarity-objective
    # search — gets its own tally against the base representation alone, so the
    # two hybrid families can be read apart. The two buckets above no longer
    # pool it into the baseline: hybrid_wins_by_type compares the
    # predictive-objective space against the base representation alone, so
    # 'base' is one base representation and not a base representation averaged
    # with a second hybrid.
    hybrid_sim_wins_by_type = {}
    hybrid_sim_total_by_type = {}

    for nt, nt_data in results.get('per_noise_type', {}).items():
        best_method = None
        best_score = float('-inf')
        # results['lower_is_better'] is written unconditionally at the top of
        # run() (the dict literal that builds `results`), so a plain read is
        # right and a defaulted False would only hide a real absence.
        # Fixed 2026-09-10 (metric-direction audit).
        #
        # It STILL is not defaulted, and the reason is worth keeping. Which way a
        # metric runs decides which of two scores wins every comparison below. A
        # default would pick a direction for a caller who never said one, and on
        # half the datasets it would pick the wrong one and still return numbers.
        # What changed on 2026-09-16 is only the message: a bare KeyError here
        # says nothing about what to pass.
        if 'lower_is_better' not in results:
            raise KeyError(
                "n13._compute_summary needs results['lower_is_better']: which way "
                "the dataset's metric runs. run() writes it, so a results dict "
                "without it was built by hand. It is not defaulted, because a "
                "guessed direction reverses every win count below and still "
                "returns a full set of numbers.")
        lower = bool(results['lower_is_better'])
        hybrid_wins_by_type[nt] = 0
        hybrid_total_by_type[nt] = 0
        hybrid_sim_wins_by_type[nt] = 0
        hybrid_sim_total_by_type[nt] = 0

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
                        elif space_name != 'hybrid_sim':
                            base_scores.append(s)
            if base_scores and hybrid_scores:
                hybrid_total_by_type[nt] += 1
                b_avg = np.mean(base_scores)
                h_avg = np.mean(hybrid_scores)
                if (not lower and h_avg > b_avg) or (lower and h_avg < b_avg):
                    hybrid_wins_by_type[nt] += 1

            # 2026-09-12: the similarity-objective space against the base
            # representation only, counted separately from the tally above.
            base_only_scores = []
            hybrid_sim_scores = []
            for space_name, space_data in sd.get('track_a', {}).items():
                for mv in space_data.values():
                    s = mv.get('score', float('nan'))
                    if np.isnan(s):
                        continue
                    if space_name == 'hybrid_sim':
                        hybrid_sim_scores.append(s)
                    elif space_name != 'hybrid':
                        base_only_scores.append(s)
            if base_only_scores and hybrid_sim_scores:
                hybrid_sim_total_by_type[nt] += 1
                b_avg = np.mean(base_only_scores)
                hs_avg = np.mean(hybrid_sim_scores)
                if (not lower and hs_avg > b_avg) or (lower and hs_avg < b_avg):
                    hybrid_sim_wins_by_type[nt] += 1

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

    # Same arithmetic for the similarity-objective space, against the base
    # representation alone. Added 2026-09-12 with the pool subset change.
    sim_per_type_rates = {
        nt: (hybrid_sim_wins_by_type[nt] / hybrid_sim_total_by_type[nt])
        for nt in hybrid_sim_total_by_type
        if hybrid_sim_total_by_type[nt] > 0
    }
    sim_persists_by_type = (np.mean(list(sim_per_type_rates.values())) > 0.5
                            if sim_per_type_rates else None)

    # Does Gaussian winner transfer?
    gaussian_best = best_per_dist.get('gaussian')
    if gaussian_best:
        # The noise types this run actually produced. The module-level
        # NOISE_TYPES is None until run() resolves it per task_type, so
        # iterating it here raised TypeError and killed the whole summary.
        transfers = all(best_per_dist.get(nt) == gaussian_best
                        for nt in best_per_dist)
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
        # The similarity-objective space on its own, against the base
        # representation only. Empty when that search family was not on disk
        # for this dataset, so the arm never ran.
        'hybrid_sim_advantage_persists': sim_persists_by_type,
        'hybrid_sim_wins_by_noise_type': hybrid_sim_wins_by_type,
        'hybrid_sim_total_by_noise_type': hybrid_sim_total_by_type,
        'hybrid_sim_win_rate_by_noise_type': sim_per_type_rates,
    })


if __name__ == '__main__':
    parser = create_experiment_parser('N13: Noise Distributions')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'n13_noise_distributions',
                            args.dataset)
