#!/usr/bin/env python3
"""N11: Noise Estimation × Remediation Grid.

Paper: Noise

SCOPE: MOLECULAR datasets only, and SYNTHETIC noise only (a dose-matched
condition from noiseInject on regression labels, defaulting to gaussian /
label-flipping on classification labels — see
`noise_helpers.inject_train_labels`). Results do NOT speak to real-world label
noise or to non-molecular domains.

Full grid from the confirmation report: estimator × smoothing/weighting × alpha ×
distance metric × representation. Extends with hybrid representations.

PRIMARY REPORTED CLAIM: the held-out RECOVERY metric (multi-seed) — how much of
the clean→noisy performance gap a given estimator × remediation config recovers.
This is scientifically sound.

CAVEAT (not the headline claim): the hybrid-vs-base space comparison is
ORACLE-BIASED. The hybrid blueprint (allocation / importance / internal
evaluator) is selected once from a precomputed pool fit on CLEAN labels, then
reused per fold under noisy labels. The *choice* of hybrid construction has thus
seen clean labels, so every hybrid space is a BEST-CASE hybrid — it is NOT a
fair hybrid-vs-base comparison and must not be reported as one.

2026-09-12, owner's decision: n11's pool subset includes similarity hybrids.
The pool lookup asks for subset 'predictive_and_similarity', so it may hand back
a recipe chosen under objective='similarity' — features chosen to make the
distance geometry predict each molecule's activity from its five nearest
neighbours — as well as one chosen to predict the label with a downstream model.
One estimation space is built per family that the pool holds, both through
create_hybrid on the same budget, the same folds and the same importance
machinery. Space names carry the family ('hybrid_predictive_<budget>',
'hybrid_similarity_<budget>') and results['pool_selection'] records what each
one was built from.

That REPLACES the four-block stack this file used to build by hand for its
'hybrid_sim' arm (count + physchem-minus-saboteurs + 3D shape + fine-2D, from
_similarity_hybrid_space). Two things change with it. The hand-built stack chose
its blocks against MCS structural overlap, which never reads activity; the
similarity objective scores a configuration by the correlation between each
held-out molecule's activity and what its five nearest neighbours predict, which
does. So the distance-based arm is now selected on clean activity labels, the
same way the prediction arm is, and carries the same oracle caveat. The stack
also only existed where a sim_<dataset>.npz cache covered every molecule; the
pool arm exists wherever a similarity-objective search has been precomputed
(tests/precompute_hybrids.py <dataset> --with-similarity).

Each config: estimate noise scores, apply remediation (smoothing, weighting or
the quantile clip), fit every downstream model on the repaired training labels,
compute recovery rate.

THE SINGLE REPRESENTATIONS, owner's decision 2026-09-25. Each downstream
model's top 2 single representations, ranked by that model's own score in the
stored search (_shared.noise_base_reps_by_model), do two jobs:

  * EVALUATION. Every repaired fit of a model is done twice, once on its rank-1
    representation and once on its rank-2 one -- including the rows where the
    noise was detected in a hybrid space. Each (model, rank) pair is an
    evaluation unit keyed '<model>@base_<k>' ('rf@base_1'), and it carries its
    own clean score, noisy score and clean-to-noisy gap, so the recovery
    denominator is one per model per rank. Every '*_by_model' field and every
    remediation key's model half is keyed by that unit.
  * DETECTION. Each model's rank-k representation is an estimation space named
    'base_<k>_<rep>', uncapped and given the two added smoothing targets. Its
    rows are fitted only for the models that rank that representation k-th; a
    representation shared by several models at the same rank is built once. A
    representation that is not one of a model's top 2 stays an estimation space
    for that model under its plain name, capped as before.

Which representations those are is written to 'base_reps_by_model'.

WHERE A LABEL IS MOVED TO, added 2026-09-17. Until then every smoothing row
aimed at whatever target its own estimator carried — a neighbour mean for the
kNN estimators, the model's own prediction for the rest, which is the same
quantity that defined the suspicion score. Two targets are now swept alongside,
the same for every estimator: a group-aware out-of-fold LightGBM prediction and
a Gaussian-process posterior mean. Their remediation keys carry the target name
('oof+proportional_0.5'); a key without one is the estimator's own target and
has not changed. A third rule, 'quantile_clip_<alpha>', moves no label toward
anything: it appears under the quantile_forest estimator, which gives each
molecule the 10th-to-90th-percentile range of the labels carried by training
molecules sharing its leaves, leaves every label inside its range untouched and
pulls each label outside it onto the edge it crossed.

Usage:
    python n11_estimation_remediation_grid.py --dataset logd --mode intermediary
    python n11_estimation_remediation_grid.py --dataset caco2 --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import inspect
import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments.dataset_policy import (
    noise_groups_for, noise_levels as dataset_noise_levels,
)
from kirby.noise_spec import GROUPED_STRATEGIES
from experiments.noise.noise_helpers import (
    inject_train_labels, noise_spec_for, detection_scores, sweep_conditions, conditions_for_task)
from experiments._shared import (
    noise_base_reps_by_model, base_rank_groups, NOISE_BASE_REP_COUNT,
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_model_factories,
    get_precomputed_search, get_best_search_for_downstream, cap_dimensions,
    DOWNSTREAM_MODELS, get_rule_hybrids, POOL_SUBSETS,
)
from kirby.hybrid import create_hybrid
from kirby.similarity.distance_config import DISTANCE_METRICS
from kirby.utils.noise_estimation import (
    get_noise_estimator, smooth_by_scores, weights_from_scores,
    clip_to_range, GaussianProcessEstimator,
)
from kirby.utils.noisy_labels import oof_predict
# The similarity-sweep kit and the disk caches, added 2026-09-25 (plan
# sections 1-3). See n07_remediation's SIMILARITY-SWEEP KIT block.
from experiments.noise.n07_remediation import (
    sweep_settings, sweep_ks, RowSink, sweep_context, similarity_sweep,
    cleaned_from_weights, cleaned_from_labels, similarity_record,
)
from experiments.noise import noise_cache as _nc
from experiments.noise.noise_cache import NEIGHBOUR_DETECTORS

# 2026-09-12, owner's decision: n11's pool subset includes similarity hybrids,
# which replaces the four-block stack this file built by hand. The import of
# _similarity_hybrid_space (build_similarity_hybrid / similarity_hybrid_available)
# is gone with it. That module is still on disk and n13 still uses it.
#
# Which families of stored search the pool may draw a recipe from. Named here
# so the one decision is in one place; 'predictive_only' is what every other
# experiment reads and is what n11 read before today.
POOL_SUBSET = 'predictive_and_similarity'


# Noise levels. `None` means the grid this dataset is anchored to in
# experiments.dataset_policy.
SIGMAS = None

# The noise condition. Gaussian is what this experiment has always injected; the
# other ten are reachable by passing noise_condition=..., one run each.
NOISE_CONDITION = 'gaussian'
# ⛔ ONE LIST WAS DRIVING TWO KNOBS THAT MEAN DIFFERENT THINGS, and it put the
# whole weighting arm out of reach of the setting that works.
#
#   smooth_by_scores(..., alpha=a)      a is a FRACTION of the way from the
#                                       label to the target. [0, 1] is its
#                                       whole range and 0.1 to 0.7 covers it.
#   weights_from_scores(..., strength=a) a is a RATIO parameter with no upper
#                                       bound. Under 'inverse' the most suspect
#                                       molecule gets 1/(1+a) of the cleanest
#                                       one's weight, so 0.1 is 1.1 to 1 and
#                                       0.7 is 1.7 to 1 -- barely a weighting
#                                       at all.
#
# So every weighting cell in this grid was gentler than the fixed 1.0 (2 to 1)
# that n07 and n18 used before 2026-09-18, and n18 then MEASURED the best
# setting at 3.0 -- a 4-to-1 ratio, outside this grid entirely, winning 5 of 5
# folds on herg_ki (+0.0069 rmse) and 5 of 5 on flab_thermo_tresanco (+0.0447
# spearman). The grid could not have found it.
#
# Two lists now. The smoothing one is unchanged, because a fraction of the way
# to a target was always the right range for it.
SMOOTH_ALPHAS = [0.1, 0.3, 0.5, 0.7]
WEIGHT_STRENGTHS = [0.3, 1.0, 3.0, 9.0]   # 1.3:1, 2:1, 4:1, 10:1

# Kept as a name: the result file writes it, and the smoothing arm is what it
# always meant.
ALPHAS = SMOOTH_ALPHAS
WEIGHT_METHODS = ['inverse', 'exponential', 'soft']
SMOOTH_METHODS = ['proportional', 'threshold', 'soft_threshold', 'confidence']
# n11 is the one experiment that sweeps the distance INSIDE itself rather
# than across runs, so it gets the whole list. --distance-metric sets the
# metric for everything else n11 computes, not this sweep.
NEIGHBOR_METRICS = list(DISTANCE_METRICS)

# Estimators that use kNN internally — test with multiple distance metrics
NEIGHBOR_ESTIMATORS = {
    'neighbor_residual', 'local_variance_ratio',
    'distance_weighted', 'hybrid',
}

# Which estimators produce a meaningful noise score when the label is a CLASS.
#
# The rule: the estimator's constructor takes `is_clf`, and under it scores a
# molecule by how far its label is from what the other molecules say it should
# be in the only units a nominal label has — a class-mismatch rate among its
# neighbours, or one minus the probability an out-of-fold classifier gives the
# class it carries. Every other estimator here fits a regressor (or a GP, a
# network, a quantile model, ngboost) on the integer class codes. That does not
# crash, and it means a molecule labelled class 2 that a model calls class 0
# scores twice as suspect as one it calls class 1 — an ordering the class
# numbers do not carry. Those are dropped on a classification dataset, with the
# reason recorded (FIX 4f).
#
# ⛔ THIS WAS A HAND-WRITTEN LIST OF FOUR NAMES AND IT DRIFTED. On 2026-09-20
# `CrossValResidualEstimator` gained the flag, for n07; the list did not gain
# the name, so n11 and n06 went on dropping cross_val from every classification
# dataset with the reason 'regression-only', which had stopped being true of
# it. The list is now derived from the constructors it was meant to track, so
# there is nothing to keep in step. `_smoke_n08_n12.py` holds the names a human
# has actually read the branch of, and fails when this set stops matching them
# — which is how the drift was caught and is the only thing a signature cannot
# tell you: accepting the flag is not the same as doing something sensible
# under it.
#
# What stays out and why it is not an oversight. 'hybrid' is in
# NEIGHBOR_ESTIMATORS — it does use the kNN distance metrics — but
# HybridNoiseEstimator takes no is_clf and passes none to the three estimators
# it averages, so each of them falls back to its regression branch on the class
# codes whatever the task is. It is excluded because its own constructor lacks
# the flag, not because of anything about the estimators inside it.
def _is_clf_aware_estimators():
    """The registry names whose estimator class accepts is_clf."""
    import inspect as _inspect
    from kirby.utils.noise_estimation import ESTIMATOR_REGISTRY
    out = set()
    for _name, _cls in ESTIMATOR_REGISTRY.items():
        try:
            if 'is_clf' in _inspect.signature(_cls.__init__).parameters:
                out.add(_name)
        except (TypeError, ValueError):
            continue
    return out


IS_CLF_AWARE_ESTIMATORS = _is_clf_aware_estimators()

# All estimators to try (torch-dependent ones are try-imported).
# leave_one_out removed (2026-05-26 audit decision): it required N model
# fits per use, dominating runtime on datasets above ~500 samples while
# the other estimators ran in seconds. The library function still exists
# in kirby.utils.noise_estimation if any other experiment wants it.
#
# activity_cliff_aware CUT 2026-09-12, measured, user decision. It scores a
# molecule by its distance from its ten nearest neighbours' activity values,
# multiplied by how much those neighbours agree with each other;
# neighbor_residual is the first half of that on its own. On herg_ki in ecfp4,
# 1,412 molecules, the two rank the molecules the same way at Spearman 0.989 —
# one method in two slots. The plain neighbour residual is the quantity the
# literature names, so the derived one goes. The class is still in
# ESTIMATOR_REGISTRY; n11, n13 and n06 stop running it.
#
# quantile_forest ADDED 2026-09-17. It scores a molecule by how far its label
# lies outside the 10th-to-90th-percentile range of the labels carried by the
# training molecules sharing its leaves in a quantile regression forest, and is
# zero for every label inside its range. That is a different quantity from
# quantile_regression and conformal, which both score the WIDTH of an interval:
# a width is a property of the neighbourhood and says nothing about this row's
# label. It is also the only estimator here that exposes both edges of a range,
# which is what the clip remediation below needs.
BASE_ESTIMATORS = [
    'neighbor_residual', 'local_variance_ratio',
    'distance_weighted', 'hybrid',
    'cross_val', 'ensemble',
    'gaussian_process', 'conformal', 'bayesian_rf',
    'quantile_forest',
]

# The two smoothing targets added 2026-09-17, on top of whatever target each
# estimator carries itself. Both are computed once per (fold, draw, space) and
# handed to every estimator, so what varies across a row of this grid is the
# suspicion score deciding how far each label moves, not the place it moves to.
#
#   oof   what a LightGBM model trained on the OTHER training molecules
#         predicts for this one, on a scaffold-group-aware inner split
#         (kirby/utils/noisy_labels/core.py:127, the same call n19 makes)
#   gp    the posterior mean of a Gaussian process fitted the same way
#         (GaussianProcessEstimator.predictions_, which that class has always
#         computed and which only its own rows read before today)
#
# The estimator's own target keeps its existing key, '<model>|<method>_<alpha>',
# unchanged; these two prefix it, '<model>|oof+<method>_<alpha>'. So no
# remediation that this file used to report has moved or been renamed.
EXTRA_TARGETS = ('oof', 'gp')

# How many inner folds those two targets are built from. Five, matching the
# n_folds default of every out-of-fold estimator in kirby.utils.noise_estimation.
EXTRA_TARGET_SPLITS = 5
TORCH_ESTIMATORS = ['heteroscedastic_nn', 'deep_ensemble', 'quantile_regression']
OPTIONAL_ESTIMATORS = ['ngboost']

# GP max samples
GP_MAX = 1000

# Minimum clean->noisy gap, as a fraction of the clean score's magnitude,
# required before a recovery RATIO is reported. recovery = (denoised - noisy)
# / gap * 100 divides by the gap, so when the injected noise barely moved the
# model the gap is a tiny number and the ratio explodes into absurd values
# (e.g. 2827%). The old guard only caught a gap of almost exactly zero
# (abs(gap) > 1e-10), so any small-but-nonzero gap sailed through. Requiring
# the gap to be a meaningful fraction of the clean score turns those
# uninformative-noise cases into NaN (flagged) instead of garbage. Datasets
# with a real gap (e.g. buchwald_hartwig, gap ~20-60% of clean) are unaffected.
GAP_MIN_FRAC = 0.02


# The private `_inject_noise` this module used to carry has been deleted. It added
# Gaussian noise at `sigma_frac * (max(y) - min(y))` -- a fraction of the label RANGE,
# which is roughly 4-6x the label SD, so a level of 0.3 here delivered far more noise
# than a level of 0.3 anywhere else in the repo. Injection now goes through
# `noise_helpers.inject_train_labels`, which is the same definition the hybrid search,
# c10 and the Rust half of the study all use, and which records what was delivered.


def _get_correction_target(estimator):
    """Extract correction target from estimator for smoothing methods.

    Returns (target_array, source_name) or (None, None) if no target available.

    Look-up priority is fixed: neighbor-based correction targets win over
    model-based ones. This matters for ensemble estimators that expose both
    `neighbor_means_` AND e.g. `mu_` — without a stable priority the smoothing
    target would silently flip depending on which attribute was set first.
    The priority is intentional: neighbor-mean smoothing converges to local
    structure of the data, whereas a model's own prediction is the SAME
    quantity that defined the noise score, so smoothing toward it is
    self-fulfilling.
    """
    NEIGHBOR_ATTRS = ('neighbor_means_', 'weighted_means_')
    MODEL_ATTRS = ('mu_', 'cv_predictions_', 'prediction_means_',
                   'oob_predictions_', 'predictions_')

    for attr in NEIGHBOR_ATTRS:
        target = getattr(estimator, attr, None)
        if target is not None:
            return target, attr

    for attr in MODEL_ATTRS:
        target = getattr(estimator, attr, None)
        if target is not None:
            return target, attr

    return None, None


def _build_estimation_spaces(train_reps, y_train, augmentations,
                             hybrid_recipes, base_spaces,
                             dataset_name=None, groups=None,
                             build_errors=None, all_models=None):
    """Build estimation spaces: each model's top-2 single representations,
    one hybrid per pool family, and every other representation.

    Args:
        train_reps: {rep_name: X_train} already sliced to fold.
        y_train: Training labels for this fold.
        augmentations: {aug_name: X_train} already sliced, or None.
        hybrid_recipes: list of the pool's winning recipes, one per search
            family, as built in run(). Each entry carries 'space' (the name the
            space gets), 'objective' ('predictive' or 'similarity') and
            'blueprint'.
        base_spaces: {space name: (rep name, [models])}, one entry per
            (rank, representation) of the downstream models' top 2, as built in
            run(). Each is scaled on this fold's training rows and NOT capped.

    Returns (spaces_dict, models_by_space). models_by_space names, for every
    space that is not open to the whole roster, the models whose rows are
    fitted in it. A space absent from it is open to every model.

    2026-09-12, owner's decision: n11's pool subset includes similarity
    hybrids. This argument was a single blueprint `bp` and built one hybrid
    space; it is now a list and builds one space per family. Every entry goes
    through the same create_hybrid call with the same budget, the same fold's
    training rows and the same importance method the search picked — only the
    blueprint and the objective differ between them.
    """
    spaces = {}
    models_by_space = {}

    # Each model's own top-2 single representations, one space per (rank,
    # representation), shared by the models that rank it there. Uncapped.
    for _sp, (_rn, _ms) in base_spaces.items():
        scaler = StandardScaler()
        spaces[_sp] = scaler.fit_transform(train_reps[_rn])
        models_by_space[_sp] = list(_ms)

    # One hybrid space per family the pool holds. The space name carries the
    # family and the blueprint's own budget (the name was hardcoded 'hybrid_100'
    # before 2026-06 even when bp.budget differed, and carried no family before
    # 2026-09-12).
    #
    # 2026-09-12, owner's decision: n11's pool subset includes similarity
    # hybrids. `objective` is the family the recipe was searched under and is
    # passed to create_hybrid, so the allocators split the budget by the same
    # quantity the search ranked configurations by: the model's held-out score
    # under 'predictive', and under 'similarity' the correlation between each
    # held-out molecule's activity and what its five nearest neighbours predict.
    # Everything else in this call is identical between the families.
    for _recipe in (hybrid_recipes or []):
        bp = _recipe.get('blueprint')
        if bp is None:
            continue
        hybrid_key = _recipe['space']
        try:
            X_h, fi = create_hybrid(
                train_reps, y_train, groups=groups,
                allocation_method=bp.allocation_method,
                importance_method=bp.importance_method,
                internal_evaluator=bp.internal_evaluator,
                budget=bp.budget,
                augmentations=augmentations,
                augmentation_strategy=bp.augmentation_strategy,
                objective=_recipe.get('objective', 'predictive'),
            )
            scaler_h = StandardScaler()
            spaces[hybrid_key] = scaler_h.fit_transform(X_h)
        except Exception as e:
            print(f"      {hybrid_key} failed: {e}")
            # 2026-09-12: a family whose hybrid never builds would otherwise
            # leave no trace at all — the space is simply absent from
            # per_space. Count the failures per space and keep the first
            # message, so a reader can tell an arm that failed from an arm
            # that was never asked for.
            if build_errors is not None:
                _e = build_errors.setdefault(
                    hybrid_key, {'n_failures': 0, 'first_error': None,
                                 'objective': _recipe.get('objective')})
                _e['n_failures'] += 1
                if _e['first_error'] is None:
                    _e['first_error'] = f'{type(e).__name__}: {e}'

    # Every other representation, capped. A model whose top 2 includes a
    # representation already has it above as a base_<k> space, so its rows are
    # not fitted here a second time -- two copies with different scaling and
    # capping would inflate the "best space" win count and double-charge the
    # per-level config counts. The other models still get it here.
    _models_all = sorted({m for _, _ms in base_spaces.values() for m in _ms}
                         | set(all_models or []))
    for rep_name, X_rep in train_reps.items():
        _has_it = {m for _rn, _ms in base_spaces.values() if _rn == rep_name
                   for m in _ms}
        if _models_all and _has_it >= set(_models_all):
            continue
        scaler = StandardScaler()
        spaces[rep_name] = cap_dimensions(scaler.fit_transform(X_rep))[0]
        if _has_it:
            models_by_space[rep_name] = [m for m in _models_all
                                         if m not in _has_it]

    return spaces, models_by_space


def _resolve_estimators():
    """Determine which estimators are available (handle torch/ngboost imports)."""
    # Light local mode: neighbour-based estimators only. These are kNN-in-the-
    # space detectors — the on-point test for a SIMILARITY metric — and skip the
    # heavy model-based ones (GP is O(n^3), plus torch/ngboost), which the
    # out-of-fold refactor made 5x costlier. Set KIRBY_NOISE_LIGHT_ESTIMATORS=1.
    if os.environ.get('KIRBY_NOISE_LIGHT_ESTIMATORS') == '1':
        return [e for e in BASE_ESTIMATORS if e in NEIGHBOR_ESTIMATORS]

    available = list(BASE_ESTIMATORS)

    try:
        import torch  # noqa: F401
        available.extend(TORCH_ESTIMATORS)
    except ImportError:
        pass

    for opt in OPTIONAL_ESTIMATORS:
        try:
            get_noise_estimator(opt)
            available.append(opt)
        except Exception:
            pass

    return available


def _mean_detection(rows):
    """Average the detection readings over the (fold, draw) passes.

    Every reading is over the same condition and the same level, so they are the
    same quantity measured several times. The two counts are averaged too, and are
    written beside the fractions so no ratio appears without what it counts.
    """
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


def _gp_target(dataset_name, X, y, groups):
    """The Gaussian-process smoothing target, GaussianProcessEstimator's
    out-of-fold posterior mean, stored through noise_cache (added
    2026-09-25): keyed on the content of X, y and the groups and on the fold
    count, so the base-space target and the similarity sweep's target for the
    same (features, labels) are fitted once. Deterministic: the estimator is
    seeded (random_state=42)."""
    def _fit():
        _gp = GaussianProcessEstimator(n_folds=EXTRA_TARGET_SPLITS,
                                       groups=groups)
        _gp.estimate(X, y)
        return np.asarray(_gp.predictions_, dtype=float)
    return _nc.cached(dataset_name, 'n11_gp_target',
                      {'X': _nc.array_hash(X), 'y': _nc.array_hash(y),
                       'groups': _nc.array_hash(groups),
                       'n_folds': int(EXTRA_TARGET_SPLITS)}, _fit)


def _run_estimator_grid(est_name, metric_name, X_space, y_noisy,
                         X_eval_train, X_eval_val, y_val,
                         model_factories, metric_fn, n_samples,
                         is_clf=False, groups=None,
                         corrupted_mask=None, noise_condition=None,
                         extra_targets=None, eval_by_model=None,
                         precomputed_neighbors=None, k_override=None,
                         dprime_cb=None, fit_models=True):
    """Run full alpha × remediation grid for one estimator config.

    precomputed_neighbors / k_override / dprime_cb / fit_models, added
    2026-09-25 for the similarity sweep (plan sections 1-2). All four default
    to the behaviour before that date.
      precomputed_neighbors  a similarity_space.neighbour_lists dict handed to
                             a neighbour estimator, so it finds each
                             molecule's neighbours in the similarity setting's
                             features while X_space stays whatever it is;
                             metric_name is then not read
      k_override             how many neighbours (default 10, as below)
      dprime_cb              called as dprime_cb(remediation, cleaned) with
                             every D' this grid builds -- the weighted or
                             relabelled training set, in the dict shape
                             n07.cleaned_training_set returns -- keyed by the
                             same remediation string the legacy key carries
      fit_models             False skips this function's own model fits, for
                             a caller that fits D' itself (the D-versus-D'
                             rows)

    eval_by_model: {model key: (X_train, X_val)} -- the features each entry of
    model_factories is fitted and scored on. run() passes one pair per
    evaluation unit ('<model>@base_<k>'), so each unit is fitted on its own
    single representation. None fits every entry on X_eval_train/X_eval_val.

    extra_targets: {name: per-training-row array} — smoothing targets computed
    outside this function, once per (fold, draw, space), and reused by every
    estimator. Each one is swept over the same smoothing methods and alphas as
    the estimator's own target, under keys that carry the target name; see
    EXTRA_TARGETS. None or {} reproduces the grid exactly as it was.

    Returns: {remed_key: denoised_score} where remed_key =
    f'{model}|{method}_{alpha}'. The model is in the key because this experiment
    fits the whole roster as of 2026-09-12, and recovery is a fraction of that
    model's own clean-to-noisy gap. The bar is a separator that no model name and
    no remediation name contains, so the two halves split back apart exactly.

    Also returns `_detection`: how well this estimator's per-molecule suspicion
    score found the molecules whose label actually moved, before any repair.
    Owner's decision 2026-09-12. It is absent under the six conditions where every
    training molecule moves, because there it would be 1.0 by construction. It costs
    no extra model fit -- the suspicion scores and the list of corrupted molecules
    are both already in memory.

    For classification (is_clf=True) the smoothing branch is intentionally
    skipped: smooth_by_scores produces continuous floats which a
    RandomForestClassifier cannot consume — the old code's bare-except
    swallowed the resulting ValueError on every (estimator, alpha, method)
    triple, so half the grid silently produced no results for clf datasets.
    Smoothing results carry a `_smoothing_skipped_clf: True` marker so
    downstream readers don't confuse the empty cells with failed runs.
    """
    # Build estimator with appropriate kwargs
    kwargs = {}
    if est_name in NEIGHBOR_ESTIMATORS:
        kwargs['metric'] = metric_name
        kwargs['k'] = 10 if k_override is None else int(k_override)
    if precomputed_neighbors is not None:
        # Only the neighbour estimators read a list (noise_cache.
        # NEIGHBOUR_DETECTORS); handed to any other one it would be ignored
        # and the row would carry today's numbers under a setting's name.
        if est_name not in NEIGHBOUR_DETECTORS:
            raise ValueError(f"{est_name} reads no neighbour list")
        kwargs['precomputed_neighbors'] = precomputed_neighbors
        kwargs['k'] = 10 if k_override is None else int(k_override)
        kwargs['metric'] = None
    # Inspect the estimator constructor once so we only forward kwargs it
    # actually accepts (different estimator families expose different flags).
    try:
        est_params = inspect.signature(
            type(get_noise_estimator(est_name)).__init__).parameters
    except Exception:
        est_params = {}
    # On a classification task, pass is_clf to the kNN-based estimators that
    # accept it so the per-neighbour disagreement is a class-mismatch rate
    # rather than a regression value-residual on integer class labels. The
    # model-based estimators (cross_val, ensemble, gaussian_process, …) take
    # no is_clf flag, so we only set it when the constructor exposes it.
    # (Audit N08 "no clf branch", same defect applies to N11's kNN estimators.)
    if is_clf and 'is_clf' in est_params:
        kwargs['is_clf'] = True
    # Pass scaffold-group labels (this fold's training rows) to any estimator
    # that splits internally (e.g. conformal), so its calibration set stays
    # scaffold-disjoint from its training set instead of leaking near-copies.
    if groups is not None and 'groups' in est_params:
        kwargs['groups'] = groups
    # GP subsamples internally to GP_MAX. n_samples is passed in for future
    # heuristics; the dead `pass` block that used to guard it has been removed.

    try:
        estimator = get_noise_estimator(est_name, **kwargs)
        scores = estimator.estimate(X_space, y_noisy)
    except Exception as e:
        return {'_error': str(e)}

    target, target_source = _get_correction_target(estimator)
    results = {}

    def _xy(mn):
        if eval_by_model is not None:
            return eval_by_model[mn]
        return X_eval_train, X_eval_val
    # Grade the estimator on FINDING the corrupted molecules, not only on the score
    # after repair. Until 2026-09-12 every detector here was graded only after repair.
    if corrupted_mask is not None:
        _det = detection_scores(corrupted_mask, scores, noise_condition)
        if _det is not None:
            results['_detection'] = _det
    # Record per-(strategy, alpha) failure reasons so the grid's empty cells
    # can be distinguished from real failures during downstream analysis.
    failure_log = {}

    # Weighting: its own list, because `strength` is an unbounded ratio
    # parameter and not a fraction of the way to a target. See WEIGHT_STRENGTHS.
    for alpha in WEIGHT_STRENGTHS:
        for wm in WEIGHT_METHODS:
            try:
                weights = weights_from_scores(scores, strength=alpha, method=wm)
            except Exception as exc:
                for mn in model_factories:
                    failure_log[f'{mn}|{wm}_{alpha}'] = f'{type(exc).__name__}: {exc}'
                continue
            if dprime_cb is not None:
                dprime_cb(f'{wm}_{alpha}', cleaned_from_weights(
                    X_space, y_noisy, weights))
            for mn, mf in (model_factories.items() if fit_models else ()):
                key = f'{mn}|{wm}_{alpha}'
                try:
                    model = mf()
                    _xtr, _xva = _xy(mn)
                    model.fit(_xtr, y_noisy, sample_weight=weights)
                    y_pred = model.predict(_xva)
                    results[key] = float(metric_fn(y_val, y_pred))
                except Exception as exc:
                    failure_log[key] = f'{type(exc).__name__}: {exc}'

    for alpha in SMOOTH_ALPHAS:
        # Smoothing methods (need a correction target AND a regression task).
        # One pass per target: the estimator's own first, under the key it has
        # always had, then each target computed outside this function under a
        # key that names it. Everything else about the pass is identical, so a
        # row under 'oof+proportional_0.5' and a row under 'proportional_0.5'
        # differ in the place the labels moved to and in nothing else.
        if not is_clf:
            _targets = []
            if target is not None:
                _targets.append(('', target))
            for _tname, _tvec in (extra_targets or {}).items():
                if _tvec is not None:
                    _targets.append((f'{_tname}+', np.asarray(_tvec, dtype=float)))
            for _prefix, _tvec in _targets:
                for sm in SMOOTH_METHODS:
                    try:
                        y_smoothed = smooth_by_scores(
                            y_noisy, scores, _tvec,
                            alpha=alpha, method=sm,
                        )
                    except Exception as exc:
                        for mn in model_factories:
                            failure_log[f'{mn}|{_prefix}{sm}_{alpha}'] = \
                                f'{type(exc).__name__}: {exc}'
                        continue
                    if dprime_cb is not None:
                        dprime_cb(f'{_prefix}{sm}_{alpha}',
                                  cleaned_from_labels(X_space, y_smoothed))
                    for mn, mf in (model_factories.items() if fit_models
                                   else ()):
                        key = f'{mn}|{_prefix}{sm}_{alpha}'
                        try:
                            model = mf()
                            _xtr, _xva = _xy(mn)
                            model.fit(_xtr, y_smoothed)
                            y_pred = model.predict(_xva)
                            results[key] = float(metric_fn(y_val, y_pred))
                        except Exception as exc:
                            failure_log[key] = f'{type(exc).__name__}: {exc}'

        # The clip. Only an estimator that exposes both edges of a per-molecule
        # range gets this row, which today is quantile_forest: it reads the
        # 10th and 90th percentile of the labels that training molecules
        # sharing a leaf carry. A label inside its range is left exactly as it
        # is; a label outside is pulled alpha of the way onto the edge it
        # crossed. No suspicion score enters, which is why this is not one of
        # the smoothing methods above.
        _lo = getattr(estimator, 'quantile_lo_', None)
        _hi = getattr(estimator, 'quantile_hi_', None)
        if _lo is not None and _hi is not None and not is_clf:
            try:
                y_clipped = clip_to_range(y_noisy, _lo, _hi, alpha=alpha)
            except Exception as exc:
                for mn in model_factories:
                    failure_log[f'{mn}|quantile_clip_{alpha}'] = \
                        f'{type(exc).__name__}: {exc}'
                y_clipped = None
            if y_clipped is not None:
                results['_n_labels_outside_range'] = int(
                    np.count_nonzero(y_clipped != np.asarray(y_noisy,
                                                             dtype=float)))
                results['_n_training_labels'] = int(len(y_noisy))
                if dprime_cb is not None:
                    dprime_cb(f'quantile_clip_{alpha}',
                              cleaned_from_labels(X_space, y_clipped))
                for mn, mf in (model_factories.items() if fit_models else ()):
                    key = f'{mn}|quantile_clip_{alpha}'
                    try:
                        model = mf()
                        _xtr, _xva = _xy(mn)
                        model.fit(_xtr, y_clipped)
                        y_pred = model.predict(_xva)
                        results[key] = float(metric_fn(y_val, y_pred))
                    except Exception as exc:
                        failure_log[key] = f'{type(exc).__name__}: {exc}'

    if is_clf:
        results['_smoothing_skipped_clf'] = True
    if failure_log:
        results['_failures'] = failure_log

    return results


def run(data, config, output_dir, dataset_name, **kwargs):
    """n11_estimation_remediation_grid across every condition this dataset can be given.

    The condition is an axis this experiment sweeps, not something a caller
    runs it once per. The sweep, the per-condition checkpointing, the resume
    and the stride slice are noise_helpers.sweep_conditions.

    Its headline is a downstream score, which is defined under every condition,
    so it sweeps the whole list rather than only the ones where a corrupted
    SUBSET exists. It also records detection, and detection_scores already
    returns None where that is undefined.
    """
    return sweep_conditions(
        _run_one_condition, 'n11_estimation_remediation_grid',
        data, config, output_dir, dataset_name,
        default_conditions=conditions_for_task(data), kwargs=kwargs,
        groups_needed=GROUPED_STRATEGIES)


def _run_one_condition(data, config, output_dir, dataset_name, **kwargs):
    print("\n" + "=" * 70)
    print("EXPERIMENT N11: NOISE ESTIMATION × REMEDIATION GRID")
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
    n_samples = len(labels)

    # The whole roster, not random forest alone. Owner's decision 2026-09-12.
    # This experiment used to fit `rf` and nothing else, so every statement it made
    # about which detector and which repair works best was a statement about one
    # model. The estimators are unaffected: a suspicion score is computed once per
    # (space, estimator) and every model is then fitted on the same repaired labels,
    # so the added cost is the downstream fits only.
    # ⛔ config FIRST, then the kwarg. run_all passes the caller's model list in
    # `config['models']` and never as a `models` kwarg -- the only kwarg it ever
    # forwards is noise_condition (noise_helpers.py:1053). This file read
    # kwargs alone, so the list the runner was asked for was silently dropped
    # and every run fell back to the module default. Eight files in this
    # directory had it; the same files read config['n_splits'] correctly two
    # lines away. Fixed 2026-09-20. An explicit kwarg still wins, which is what
    # a smoke or a direct caller uses.
    downstream_models = list(
        kwargs.get('models') or config.get('models') or DOWNSTREAM_MODELS)
    model_factories = get_model_factories(
        downstream_models, task_type=task_type, n_samples=n_samples)
    downstream_models = [m for m in downstream_models if m in model_factories]
    if not downstream_models:
        raise RuntimeError(
            "n11: no downstream model could be built. Asked for "
            f"{list(kwargs.get('models') or DOWNSTREAM_MODELS)}, got "
            f"{sorted(model_factories)}.")

    noise_condition = kwargs.get('noise_condition', NOISE_CONDITION)
    sigmas = kwargs.get('sigmas') or kwargs.get('levels') or SIGMAS
    if sigmas is None:
        # detect_grid=True: on censoring this asks for the four censored fractions
        # this experiment runs (0.0, 0.10, 0.25, 0.40), not n13's full six. Owner's
        # decision 2026-09-12. It changes nothing for the other ten conditions.
        sigmas = list(dataset_noise_levels(dataset_name, strategy=noise_condition,
                                           detect_grid=True, is_clf=is_clf))
    noise_groups, _n_ng, _lg = noise_groups_for(data)
    noise_provenance = {}
    sample_X = next(iter(reps.values()))

    available_estimators = _resolve_estimators()
    # FIX 4f: on a classification dataset, the model-based regression estimators
    # (cross_val, ensemble, gaussian_process, conformal, bayesian_rf, hybrid,
    # ngboost, and the torch heteroscedastic_nn / deep_ensemble /
    # quantile_regression) fit a regressor on integer class labels — no crash
    # but a meaningless noise score. Keep ONLY the kNN/is_clf-aware estimators
    # (IS_CLF_AWARE_ESTIMATORS), which _run_estimator_grid passes is_clf=True so
    # they score class disagreement. Previously only gaussian_process was
    # dropped, leaving the rest to emit junk. Record the dropped names + reason.
    skipped_estimators = []
    if is_clf:
        skipped_estimators = [e for e in available_estimators
                              if e not in IS_CLF_AWARE_ESTIMATORS]
        available_estimators = [e for e in available_estimators
                                if e in IS_CLF_AWARE_ESTIMATORS]

    print(f"  Dataset: {dataset_name} ({n_samples} samples)")
    print(f"  Metric: {metric_name}")
    print(f"  Estimators: {len(available_estimators)}")
    if skipped_estimators:
        print(f"  Skipped (regression-only on clf): {skipped_estimators}")
    print(f"  Alphas: {ALPHAS}")
    print(f"  Sigmas: {sigmas}")
    print(f"  Folds: {n_folds}")

    # Recipes selected from the CLEAN-label pool, per-fold construction under
    # noise (see _build_estimation_spaces). These are BEST-CASE hybrids, NOT a
    # fair baseline comparison: the structural recipe choice (allocation /
    # importance / internal evaluator) has already seen clean labels, so it is
    # ORACLE-BIASED and biases every hybrid space toward higher retention than
    # a fair baseline. Only the per-fold feature geometry is noisy. A
    # leak-free hybrid-vs-base design would re-select the recipe per fold
    # under noise; we do not. The RECOVERY metric (per estimator × remediation)
    # remains the sound primary claim — the hybrid-vs-base space comparison
    # must be read as "best-case hybrid", not a fair result. (Audit P1#5.)
    #
    # 2026-09-12, owner's decision: n11's pool subset includes similarity
    # hybrids. One lookup asks the pool for subset 'predictive_and_similarity'
    # and every family it holds gets its own estimation space, built on the
    # same budget, the same folds and the same importance machinery as the
    # prediction arm it is compared against. This is NOT distance=True — n11
    # does not pass distance=True and that is unchanged.
    #
    # Local light-run mode, kept and re-pointed: it used to drop the predictive
    # hybrid and leave the hand-built four-block stack as the only hybrid arm.
    # That stack is gone, so the flag now asks the pool for 'similarity_only'
    # — the same intent (one hybrid arm, the distance-based one) through the
    # one mechanism. Both families are pool reads now, so neither runs a search
    # here; a family with no precomputed search simply has no arm.
    # Set KIRBY_SIMILARITY_HYBRID_ONLY=1.
    pool_subset = POOL_SUBSET
    if os.environ.get('KIRBY_SIMILARITY_HYBRID_ONLY') == '1':
        pool_subset = 'similarity_only'
        print("  similarity-hybrid-only mode (KIRBY_SIMILARITY_HYBRID_ONLY=1) "
              f"— pool subset {pool_subset!r}; the prediction-built hybrid is "
              "not asked for")

    search, meta = get_best_search_for_downstream(
        reps, labels, groups,
        downstream_model='rf', budget=None,
        augmentations=augmentations,
        dataset_name=dataset_name, task_type=task_type,
        data=data,
        pool_subset=pool_subset,
    )
    # ⛔ A POOL WITH NO RECIPE IN IT IS NOT A REASON TO ABANDON THE CROSS.
    # Until 2026-09-21 this check and the one below both returned
    # {'error': 'no_blueprint'} before the level loop started, so a dataset
    # whose stored search was absent produced no file at all — and n11 had
    # produced none since it was written.
    #
    # What the recipe buys is the hybrid estimation space, one arm of the space
    # axis, and the header above calls that arm oracle-biased and refuses to
    # headline it. The estimator × correction cross — what this file names as
    # its primary claim — needs no recipe: it scores every training molecule in
    # the control representation and repairs the labels there.
    #
    # So an absent recipe now drops the hybrid arm and records why, and the run
    # continues in the representation spaces. `hybrid_arm` is written into
    # results['pool_selection'], so a file with no hybrid space in it reads as
    # a recorded decision rather than as a silent gap.
    hybrid_arm = {'spaces_built': [], 'dropped_reason': None}
    if meta is None:
        print("  no recipe in the pool for this dataset: the hybrid "
              "estimation spaces are not built, and the estimator x "
              "correction cross runs in the representation spaces.")
        hybrid_arm['dropped_reason'] = (
            'the pool returned no entry for this dataset, so no hybrid '
            'estimation space was built; every space in this file is a plain '
            'representation')
        meta = {}

    # 2026-09-12: one hybrid space per family the pool holds. meta['objective']
    # names the family the pool returned as its winner; meta['by_objective']
    # carries EVERY eligible family's own best, which is what n11 builds from,
    # because the two families' stored scores are in different units and the
    # pool does not rank one against the other. Under 'predictive' the stored
    # score is the dataset's own metric; under 'similarity' it is the
    # correlation between each held-out molecule's activity and what its five
    # nearest neighbours predict.
    hybrid_recipes = []
    # THE OWNER'S SET (top 2 general + top 1 distance hybrids per model and
    # family, 2026-10-02). Each family now gives up to three estimation spaces:
    # general rank 1 keeps the name this file has always used
    # ('hybrid_<family>_<budget>'), general rank 2 and distance rank 1 add
    # '__general2' / '__distance1'. The recipe is still the random forest's,
    # as before: no downstream model is fitted on these spaces, they are where
    # the noise estimators score molecules. A family or slot that could not be
    # filled is written to hybrid_arm['unavailable'] with its reason.
    hybrid_arm['unavailable'] = {}
    _rule = None
    try:
        _rule = get_rule_hybrids(
            reps, labels, groups, downstream_model='rf', budget=None,
            dataset_name=dataset_name, augmentations=augmentations,
            pool_subset=pool_subset, task_type=task_type, data=data)
    except Exception as exc:
        from kirby.result_cache import StaleResultDB as _Stale
        if isinstance(exc, _Stale):
            raise
        hybrid_arm['unavailable']['_rule_read'] = (
            f'the read of the top 2 general + top 1 distance hybrids raised '
            f'{type(exc).__name__}: {exc}; no hybrid estimation space was built')
        print(f"  rule hybrid read failed: {type(exc).__name__}: {exc}")
    for _obj, _fo in ((_rule or {}).get('by_objective') or {}).items():
        for _h in _fo.get('hybrids') or []:
            _bp = _h.get('blueprint')
            _slot = _h.get('slot') or 'general1'
            if _bp is None:
                hybrid_arm['unavailable'][f'hybrid_{_obj}__{_slot}'] = (
                    f'the {_obj} {_slot} hybrid carried no blueprint')
                continue
            _budget = int(getattr(_bp, 'budget', 0) or 0)
            _name = (f'hybrid_{_obj}_{_budget}' if _budget
                     else f'hybrid_{_obj}')
            if _slot != 'general1':
                _name = f'{_name}__{_slot}'
            hybrid_recipes.append({
                'space': _name,
                'objective': _obj,
                'slot': _slot,
                'lineage': _h.get('lineage'),
                'blueprint': _bp,
                'tier': _h.get('tier'),
                'aug_condition': _h.get('aug_condition'),
                'budget': _budget,
                'budget_relaxed': _h.get('budget_relaxed'),
                'allocation_method': getattr(_bp, 'allocation_method', None),
                'importance_method': getattr(_bp, 'importance_method', None),
                'internal_evaluator': getattr(_bp, 'internal_evaluator', None),
                'augmentation_strategy': getattr(_bp, 'augmentation_strategy',
                                                 None),
                'search_score_mean': _h.get('score_other_folds'),
                'search_metric_name': _fo.get('metric_name'),
                'search_lower_is_better': _fo.get('lower_is_better'),
            })
        if _fo.get('missing'):
            hybrid_arm['unavailable'][f'hybrid_{_obj}'] = ' | '.join(
                _fo['missing'])
    # A family the pool subset asked for and the read returned nothing under.
    # Until 2026-10-02 such a family left no trace in the result file.
    for _obj in POOL_SUBSETS.get(pool_subset, ()):
        if (_rule is not None
                and _obj not in (_rule.get('by_objective') or {})):
            hybrid_arm['unavailable'][f'hybrid_{_obj}'] = (
                f'the pool held no search under objective={_obj} for this '
                f'dataset in either the general or the distance searches')

    # The rule read found no hybrid at all but the single-winner lookup
    # above did (a stand-in pool, or a pool the two reads see differently):
    # each family's single winner is built, as before 2026-10-02, and the
    # result file says so.
    if not hybrid_recipes and meta:
        _by_obj = meta.get('by_objective') or {
            meta.get('objective', 'predictive'): {
                'tier': meta.get('tier'),
                'aug_condition': meta.get('aug_condition'),
                'score_mean': meta.get('score_mean'),
                'metric_name': meta.get('metric_name'),
                'lower_is_better': meta.get('lower_is_better'),
                'best_config': meta.get('best_config'),
            }}
        for _obj, _entry in _by_obj.items():
            _bp = (_entry.get('best_config') or {}).get('blueprint')
            if _bp is None:
                hybrid_arm['unavailable'].setdefault(
                    f'hybrid_{_obj}',
                    'the family was in the pool but its winning configuration '
                    'carried no blueprint')
                continue
            _budget = int(getattr(_bp, 'budget', 0) or 0)
            hybrid_recipes.append({
                'space': (f'hybrid_{_obj}_{_budget}' if _budget
                          else f'hybrid_{_obj}'),
                'objective': _obj, 'slot': 'general1',
                'lineage': 'single_winner_fallback', 'blueprint': _bp,
                'tier': _entry.get('tier'),
                'aug_condition': _entry.get('aug_condition'),
                'budget': _budget, 'budget_relaxed': None,
                'allocation_method': getattr(_bp, 'allocation_method', None),
                'importance_method': getattr(_bp, 'importance_method', None),
                'internal_evaluator': getattr(_bp, 'internal_evaluator', None),
                'augmentation_strategy': getattr(_bp, 'augmentation_strategy',
                                                 None),
                'search_score_mean': _entry.get('score_mean'),
                'search_metric_name': _entry.get('metric_name'),
                'search_lower_is_better': _entry.get('lower_is_better'),
            })
        if hybrid_recipes:
            hybrid_arm['rule_fallback'] = (
                'the read of the top 2 general + top 1 distance hybrids '
                'returned none, so each family\'s single winner from the pool '
                'lookup was built instead')
            print("  rule set empty: built each family's single winner")
    if hybrid_recipes:
        # The rule read found hybrids even where the single-winner lookup
        # above returned nothing, so no arm was dropped.
        hybrid_arm['dropped_reason'] = None
    if not hybrid_recipes:
        print("  no blueprint on any family the pool holds: the hybrid "
              "estimation spaces are not built, and the estimator x "
              "correction cross runs in the representation spaces.")
        hybrid_arm['dropped_reason'] = hybrid_arm['dropped_reason'] or (
            'the pool answered but carried no blueprint on any family, so no '
            'hybrid estimation space was built; every space in this file is a '
            'plain representation')
    hybrid_arm['spaces_built'] = [_r['space'] for _r in hybrid_recipes]

    for _r in hybrid_recipes:
        print(f"  hybrid space {_r['space']}: family {_r['objective']!r}, "
              f"budget {_r['budget']} features, allocation "
              f"{_r['allocation_method']}, importance {_r['importance_method']}, "
              f"tier {_r['tier']}/{_r['aug_condition']}, search score "
              f"{_r['search_score_mean']} in {_r['search_metric_name']!r}")

    # ── Each downstream model's top-2 single representations ──
    # Ranked by that model's own score in the stored search fetched above (the
    # rf one: its baselines carry a score under every downstream model it
    # scored). A `base_reps` kwarg ({model: [reps]}) overrides the ranking, for
    # a smoke or a direct caller. A model with none has no evaluation features
    # and is left out of the run, with the reason in 'base_reps_missing'.
    _base_override = kwargs.get('base_reps')
    if _base_override is not None:
        base_by_model = {m: list(_base_override[m])[:NOISE_BASE_REP_COUNT]
                         for m in downstream_models if _base_override.get(m)}
        base_missing = {m: 'not given by the caller' for m in downstream_models
                        if m not in base_by_model}
    else:
        base_by_model, base_missing = noise_base_reps_by_model(
            {m: search for m in downstream_models}, downstream_models,
            reps=reps, dataset_name=dataset_name)
    eval_models = [m for m in downstream_models if m in base_by_model]
    # Evaluation units: one per (model, rank). '<model>@base_<k>'.
    eval_units = {f'{m}@base_{k}': (m, k, rn)
                  for m in eval_models
                  for k, rn in enumerate(base_by_model[m], start=1)}
    unit_factories = {u: model_factories[m] for u, (m, _k, _rn)
                      in eval_units.items()}
    # Detection spaces from the same ranking: {space: (rep, [models])}.
    base_spaces = {}
    for _k in range(1, NOISE_BASE_REP_COUNT + 1):
        for _rn, _ms in base_rank_groups(base_by_model, _k,
                                         eval_models).items():
            base_spaces[f'base_{_k}_{_rn}'] = (_rn, list(_ms))
    for _u, (_m, _k, _rn) in eval_units.items():
        print(f"  evaluation unit {_u}: {_m} fitted on {_rn}")
    for _m, _why in base_missing.items():
        print(f"  {_m}: no single representation, left out -- {_why}")
    if not eval_units:
        print("  no downstream model has a single representation to be "
              "fitted on, so nothing can be scored")
        return {'error': 'no_single_representation',
                'metric': metric_name,
                'base_reps_by_model': base_by_model,
                'base_reps_missing': base_missing}

    # 2026-09-12: per-space build failures, filled by _build_estimation_spaces.
    hybrid_build_errors = {}
    # 2026-09-17: the first failure per (space, target) for the two smoothing
    # targets that are computed outside the estimators. A target that never
    # builds leaves no row at all, so without this a reader cannot tell a target
    # that failed from one that was never asked for.
    extra_target_errors = {}
    # The same switch _resolve_estimators reads. Named here because the
    # Gaussian-process smoothing target is the same O(n^3) fit that flag exists
    # to drop, so it has to see the flag too.
    light_estimators = os.environ.get('KIRBY_NOISE_LIGHT_ESTIMATORS') == '1'
    # The spaces the two added targets are built in: every hybrid, plus each
    # model's top-2 single representations. See the note at the build site.
    extra_target_spaces = (set(base_spaces)
                           | {_r['space'] for _r in hybrid_recipes})

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        # Each downstream model's top 2 single representations, from its own
        # score in the stored search. Every repaired fit is done once on each;
        # see the module docstring.
        'base_reps_by_model': base_by_model,
        'base_reps_missing': base_missing,
        'base_reps_search_model': 'rf',
        'evaluation_units': {u: {'model': m, 'rank': k, 'representation': rn}
                             for u, (m, k, rn) in eval_units.items()},
        'evaluation_unit_key': (
            "'<model>@base_<k>': that model fitted on its rank-k single "
            "representation. Every *_by_model field and the model half of every "
            "remediation key is keyed by it."),
        'base_detection_spaces': {sp: {'representation': rn, 'models': ms}
                                  for sp, (rn, ms) in base_spaces.items()},
        'estimators': available_estimators,
        # 2026-09-17: the smoothing targets that do not come from an estimator.
        # Each one is swept over the same methods and alphas as the estimator's
        # own target, under a remediation key that names it — 'oof+proportional'
        # against the estimator's own 'proportional'.
        'extra_smoothing_targets': list(EXTRA_TARGETS),
        'extra_smoothing_targets_note': (
            "'oof' is what a LightGBM model trained on the other training "
            "molecules of this fold predicts for this one, on a "
            "scaffold-group-aware inner split. 'gp' is the posterior mean of a "
            "Gaussian process fitted the same way. Both are computed once per "
            "(fold, draw, space) and handed to every estimator, so two rows "
            "differing only in this prefix differ in where the labels moved to "
            "and in nothing else. A remediation key with no prefix is the "
            "estimator's own target, unchanged from before 2026-09-17. "
            "'quantile_clip_<alpha>' is not a smoothing row at all: it appears "
            "only under an estimator that exposes both edges of a per-molecule "
            "label range (quantile_forest), leaves every label inside its range "
            "byte-identical, and pulls each label outside it alpha of the way "
            "onto the edge it crossed."),
        'extra_target_errors': extra_target_errors,
        # USER DECISION 2026-09-17: which spaces the two added targets are built
        # in. Building one costs five model fits, so they are built in the
        # hybrid feature sets and in each model's top-2 single representations
        # only, not in every representation the dataset has. A smoothing row carrying a
        # target name appears under these spaces and nowhere else.
        'extra_target_spaces': sorted(extra_target_spaces),
        # FIX 4f: regression-only estimators dropped on classification, with the
        # explicit reason, so empty cells aren't confused with failed runs.
        'skipped_estimators': {e: 'skipped_regression_estimator_on_clf'
                               for e in skipped_estimators},
        'alphas': ALPHAS,
        'sigmas': sigmas,
        'noise_condition': noise_condition,
        'noise_provenance': noise_provenance,
        'per_sigma': {},
    }

    # Multi-seed noise injection: average across NOISE_SEEDS to reduce
    # dependence on a single noise realization. The clean baseline is
    # noise-independent so it's only computed once per fold; the noisy
    # baseline + estimation spaces + grid are recomputed per seed.
    NOISE_SEEDS = kwargs.get('noise_seeds', [0, 1, 2])

    # ── The similarity axis, added 2026-09-25 (plan sections 1-2) ──────────
    # n11's own estimation-space axis above is unchanged: every key it fills
    # is what it was. The sweep below is the suite-wide axis on top. Each
    # neighbour estimator (the ones noise_cache.NEIGHBOUR_DETECTORS names that
    # this run enables) reads its neighbours from every similarity setting at
    # every k in SWEPT_KS, and the full correction grid of
    # _run_estimator_grid -- every weighting, every smoothing rule, target and
    # alpha, the clip -- is applied to its scores. The model input is each
    # evaluation unit's own single representation, as in the legacy grid.
    # One D-versus-D' row per (estimator, correction, setting, k, model): RF,
    # SVM and the unit's own model.
    #
    # TODO(similarity-sweep 2026-09-25): 'hybrid' (HybridNoiseEstimator)
    # averages three neighbour estimators but takes no neighbour list, so it
    # is not in the sweep; giving it one is a change to noise_estimation.py.
    sim_settings = sweep_settings()
    sink = RowSink(output_dir, 'n11_estimation_remediation_grid',
                   dataset_name, kwargs.get('noise_condition'))
    # TODO(similarity-sweep 2026-09-25): the hybrid and hybrid_similarity
    # similarity spaces use the one recipe this file reads, rf's
    # (get_best_search_for_downstream(downstream_model='rf') above), for every
    # evaluation unit's model. The plan says each model's own prediction-built
    # hybrid; reading one recipe per model is a pool read per model. Owner to
    # say whether rf's recipe stands in for every model here.
    # The similarity settings know two hybrid spaces by name, 'hybrid' and
    # 'hybrid_similarity', so only each family's general rank 1 is read there
    # (2026-10-02); the other slots are estimation spaces above.
    sim_recipes = {}
    for _r in hybrid_recipes:
        if _r.get('slot', 'general1') != 'general1':
            continue
        _sp = ('hybrid' if _r['objective'] == 'predictive'
               else 'hybrid_similarity' if _r['objective'] == 'similarity'
               else None)
        if _sp is not None:
            sim_recipes[_sp] = (_r['blueprint'], _r['aug_condition'])
    sim_estimators = [e for e in available_estimators
                      if e in NEIGHBOUR_DETECTORS]
    sim_detection = {}
    sweep_ctx = sweep_context(
        experiment='n11_estimation_remediation_grid',
        dataset_name=dataset_name, sink=sink, settings=sim_settings,
        metric=(metric_name, metric_fn, lower_is_better), task_type=task_type,
        condition=noise_condition, level=None, data=data,
        factories=model_factories)
    print(f"  Similarity axis: {len(sim_settings)} settings x ks "
          f"{list(sweep_ks())} x estimators {sim_estimators}; rows -> "
          f"{sink.path}")

    # 2026-09-12, owner's decision: n11's pool subset includes similarity
    # hybrids. The per-fold cache the hand-built four-block stack needed
    # (sim_by_fold / sim_fold_info) is gone with it — both hybrid spaces are now
    # built inside the (fold, seed) loop by create_hybrid, on that seed's noisy
    # labels, exactly as the prediction arm already was.

    for sigma in sigmas:
        print(f"\n  === Sigma={sigma} ===")
        t_sigma = time.time()

        clean_scores = {u: [] for u in eval_units}
        # One list per evaluation unit, for both references. Recovery is a
        # fraction of THIS unit's clean-to-noisy gap -- one model on one of its
        # two single representations -- so the references cannot be shared.
        noisy_scores = {u: [] for u in eval_units}
        # {space: {est_key: {remed_key: [scores]}}}
        grid_acc = {}
        # The same numbers again, kept fold by fold, so the winning config can be
        # re-picked from the OTHER folds and read on the held-out one. Without
        # this the reported recovery is the largest of ~4,800 numbers measured on
        # the folds that produced it.
        #   clean_by_fold[fold] = score           (noise-independent, one per fold)
        #   noisy_by_fold[fold] = [score, ...]    (one per noise seed)
        #   denoised_by_fold[space][est][remed][fold] = [score, ...]
        clean_by_fold = {u: {} for u in eval_units}
        noisy_by_fold = {u: {} for u in eval_units}
        denoised_by_fold = {u: {} for u in eval_units}

        fold_idx = 0
        for train_idx, val_idx in cv_split(groups, n_folds, sample_X,
                                            labels, mode=mode):
            fold_idx += 1
            y_train_clean = labels[train_idx]
            y_val = labels[val_idx]

            train_reps_fold = {k: v[train_idx] for k, v in reps.items()}

            # The evaluation features, one scaled pair per single
            # representation any unit is fitted on, shared by those units.
            # Cached 2026-09-25 (noise_cache.cached_scaled): the same
            # StandardScaler arithmetic, stored once per (representation
            # content, fold) across every level and condition.
            _eval_by_rep = {}
            for _u, (_m, _k, _rn) in eval_units.items():
                if _rn not in _eval_by_rep:
                    _eval_by_rep[_rn] = _nc.cached_scaled(
                        dataset_name, reps[_rn], train_idx, val_idx)
            eval_by_unit = {_u: _eval_by_rep[_rn]
                            for _u, (_m, _k, _rn) in eval_units.items()}

            # Clean baseline (noise-independent — compute once per fold per unit)
            # Through noise_cache.cached_fit_predict since 2026-09-25: the same
            # fit and model.predict, stored once per (features, labels, model)
            # -- it was refitted at every level -- and read back by the
            # D-versus-D' rows.
            for _u in eval_units:
                _xtr, _xva = eval_by_unit[_u]
                _clean_f = float(metric_fn(y_val, _nc.cached_fit_predict(
                    dataset_name, eval_units[_u][0], unit_factories[_u],
                    _xtr, y_train_clean, _xva)))
                clean_scores[_u].append(_clean_f)
                clean_by_fold[_u][fold_idx] = _clean_f

            train_augs = {k: v[train_idx] for k, v in augmentations.items()} if augmentations else None

            for noise_seed in NOISE_SEEDS:
                # Per-(fold, seed) noise realization. Combining fold_idx and
                # noise_seed into the seed keeps every (fold, seed) draw
                # distinct without collisions.
                _spec = noise_spec_for(dataset_name, noise_condition, sigma,
                                       noise_seed=noise_seed,
                                       is_clf=is_clf)
                y_noisy, _cmask, _prov = inject_train_labels(
                    labels, _spec, train_idx=train_idx, is_clf=is_clf,
                    groups=noise_groups, dataset_name=dataset_name)
                noise_provenance.setdefault(_spec.spec_id, _prov)

                # 2026-09-12, owner's decision: n11's pool subset includes
                # similarity hybrids. `bp` (one blueprint) became
                # `hybrid_recipes` (one per family), and the hand-built
                # four-block stack that used to be pasted in below as
                # 'hybrid_sim' is gone — the distance-based arm is built here,
                # in this loop, on the same budget, the same fold and the same
                # importance machinery as the prediction arm.
                spaces, models_by_space = _build_estimation_spaces(
                    train_reps_fold, y_noisy, train_augs, hybrid_recipes,
                    base_spaces,
                    dataset_name=dataset_name,
                    groups=(groups[train_idx] if groups is not None else None),
                    build_errors=hybrid_build_errors,
                    all_models=eval_models,
                )

                for _u in eval_units:
                    _xtr, _xva = eval_by_unit[_u]
                    _noisy_fs = float(metric_fn(y_val, _nc.cached_fit_predict(
                        dataset_name, eval_units[_u][0], unit_factories[_u],
                        _xtr, y_noisy, _xva)))
                    noisy_scores[_u].append(_noisy_fs)
                    noisy_by_fold[_u].setdefault(fold_idx, []).append(_noisy_fs)

                for space_name, X_space in spaces.items():
                    if space_name not in grid_acc:
                        grid_acc[space_name] = {}
                    # The units fitted on this space's suspicion scores: every
                    # unit of every model the space is open to, on both of
                    # that model's single representations.
                    _open = models_by_space.get(space_name)
                    _space_units = {u: f for u, f in unit_factories.items()
                                    if _open is None
                                    or eval_units[u][0] in _open}
                    if not _space_units:
                        continue

                    # The two smoothing targets that do not belong to any one
                    # estimator, built once for this (fold, draw, space) and
                    # handed to every estimator below. Both are fitted on this
                    # fold's training molecules only and read off molecules the
                    # fit did not see, and both take the fold's scaffold groups
                    # so the inner split is group-disjoint. Cost: one LightGBM
                    # out-of-fold pass and one Gaussian-process out-of-fold pass
                    # per (fold, draw, space), whatever estimators are enabled.
                    # USER DECISION 2026-09-17: the two targets added today are
                    # NOT computed in every representation. Building one costs
                    # five model fits -- a LightGBM pass and a Gaussian-process
                    # pass -- and n11 holds one space per representation the
                    # dataset has, each at its full width. They are built in the
                    # hybrid feature sets and in each model's top-2 single
                    # representations, and nowhere else. Which spaces got them is
                    # written to results['extra_target_spaces'], so an absent row
                    # reads as a decision rather than as a failure. The clip is
                    # not affected: it costs no extra fit beyond the
                    # quantile_forest estimator, which sweeps every space as
                    # every other estimator does.
                    # ⛔ EXTRA_TARGETS is the list, and this loop is what reads
                    # it. Until 2026-09-20 the two names were written out here
                    # by hand and EXTRA_TARGETS was read in one place only --
                    # results['extra_smoothing_targets'], the field that TELLS a
                    # reader which targets the run used. So a harness that set
                    # EXTRA_TARGETS = ('oof',) got a file claiming one target
                    # and carrying rows under both, and paid for the Gaussian
                    # process it had asked to drop. Both names are still
                    # spelled out below because each is built by a different
                    # call; what changed is that each one is now gated on the
                    # list saying it was asked for.
                    extra_targets = {}
                    if not is_clf and space_name in extra_target_spaces:
                        _g_tr = (groups[train_idx] if groups is not None
                                 else None)
                        if 'oof' in EXTRA_TARGETS:
                            try:
                                # noise_cache.cached_oof_predict, 2026-09-25:
                                # the same oof_predict call, stored once per
                                # (features, labels, groups).
                                extra_targets['oof'] = _nc.cached_oof_predict(
                                    dataset_name, X_space, y_noisy, _g_tr,
                                    n_splits=min(EXTRA_TARGET_SPLITS,
                                                 len(y_noisy)))
                            except Exception as exc:
                                extra_target_errors.setdefault(
                                    f'{space_name}|oof',
                                    f'{type(exc).__name__}: {exc}')
                        # The Gaussian-process target is the same O(n^3) fit
                        # KIRBY_NOISE_LIGHT_ESTIMATORS was set up to drop, so
                        # the flag drops it here too rather than reintroducing
                        # it through a side door. Recorded, not silent.
                        if 'gp' in EXTRA_TARGETS:
                            if light_estimators:
                                extra_target_errors.setdefault(
                                    f'{space_name}|gp',
                                    'not computed: '
                                    'KIRBY_NOISE_LIGHT_ESTIMATORS=1')
                            else:
                                try:
                                    extra_targets['gp'] = _gp_target(
                                        dataset_name, X_space, y_noisy, _g_tr)
                                except Exception as exc:
                                    extra_target_errors.setdefault(
                                        f'{space_name}|gp',
                                        f'{type(exc).__name__}: {exc}')

                    for est_name in available_estimators:
                        if est_name in NEIGHBOR_ESTIMATORS:
                            metrics_to_test = NEIGHBOR_METRICS
                        else:
                            metrics_to_test = [None]

                        for metric_var in metrics_to_test:
                            est_key = est_name
                            if metric_var:
                                est_key = f'{est_name}_{metric_var}'

                            grid_result = _run_estimator_grid(
                                est_name, metric_var,
                                X_space, y_noisy,
                                None, None, y_val,
                                _space_units, metric_fn, n_samples,
                                is_clf=is_clf,
                                groups=groups[train_idx],
                                corrupted_mask=_cmask,
                                noise_condition=noise_condition,
                                extra_targets=extra_targets,
                                eval_by_model=eval_by_unit,
                            )

                            if est_key not in grid_acc[space_name]:
                                grid_acc[space_name][est_key] = {}

                            if '_error' in grid_result:
                                # Persist the per-(space, est) error so it's
                                # not silently dropped.
                                grid_acc[space_name][est_key].setdefault(
                                    '_errors', []).append(grid_result['_error'])
                                continue

                            for remed_key, score in grid_result.items():
                                if remed_key.startswith('_'):
                                    # Carry markers like `_failures` and
                                    # `_smoothing_skipped_clf` through to
                                    # the saved results so downstream
                                    # consumers can tell "skipped/failed"
                                    # apart from "no positive recoveries".
                                    if remed_key == '_failures':
                                        merged = grid_acc[space_name][est_key].setdefault(
                                            '_failures', {})
                                        merged.update(score)
                                    elif remed_key == '_detection':
                                        # One detection reading per (fold, draw).
                                        # Kept as a list and averaged below, like
                                        # every other per-fold number here, so a
                                        # later draw does not overwrite an earlier
                                        # one.
                                        (grid_acc[space_name][est_key]
                                         .setdefault('_detection_per_pass', [])
                                         .append(score))
                                    else:
                                        grid_acc[space_name][est_key][remed_key] = score
                                    continue
                                if remed_key not in grid_acc[space_name][est_key]:
                                    grid_acc[space_name][est_key][remed_key] = []
                                grid_acc[space_name][est_key][remed_key].append(score)
                                _mn_of_key = remed_key.split('|', 1)[0]
                                (denoised_by_fold[_mn_of_key]
                                 .setdefault(space_name, {})
                                 .setdefault(est_key, {})
                                 .setdefault(remed_key, {})
                                 .setdefault(fold_idx, []).append(score))

                # ── The similarity sweep for this (level, fold, draw). ──
                # One pass per evaluation unit. The extra smoothing targets are
                # the ones the legacy grid builds for the unit's own base
                # space (same features, same labels, same groups), read back
                # from noise_cache.
                _g_sw = groups[train_idx] if groups is not None else None
                for _u, (_m, _k_rank, _rn) in eval_units.items():
                    if not sim_estimators or not sink.enabled:
                        break
                    _xtr, _xva = eval_by_unit[_u]
                    _xt = {}
                    if not is_clf and 'oof' in EXTRA_TARGETS:
                        try:
                            _xt['oof'] = _nc.cached_oof_predict(
                                dataset_name, _xtr, y_noisy, _g_sw,
                                n_splits=min(EXTRA_TARGET_SPLITS,
                                             len(y_noisy)))
                        except Exception as exc:
                            extra_target_errors.setdefault(
                                f'sweep:{_rn}|oof',
                                f'{type(exc).__name__}: {exc}')
                    if not is_clf and 'gp' in EXTRA_TARGETS \
                            and not light_estimators:
                        try:
                            _xt['gp'] = _gp_target(dataset_name, _xtr,
                                                   y_noisy, _g_sw)
                        except Exception as exc:
                            extra_target_errors.setdefault(
                                f'sweep:{_rn}|gp',
                                f'{type(exc).__name__}: {exc}')

                    def _dprimes(setting, lists, k, _xtr=_xtr, _xt=_xt):
                        for est_name in sim_estimators:
                            got = []
                            res = _run_estimator_grid(
                                est_name, None, _xtr, y_noisy, None, None,
                                y_val, {}, metric_fn, n_samples,
                                is_clf=is_clf, groups=_g_sw,
                                corrupted_mask=_cmask,
                                noise_condition=noise_condition,
                                extra_targets=_xt,
                                precomputed_neighbors=lists, k_override=k,
                                dprime_cb=lambda r, c: got.append((r, c)),
                                fit_models=False)
                            if '_error' in res:
                                yield est_name, 'estimator_failed', None
                                continue
                            if '_detection' in res:
                                (sim_detection.setdefault(str(sigma), {})
                                 .setdefault(est_name, {})
                                 .setdefault(setting.name, {})
                                 .setdefault(str(k), [])
                                 .append(res['_detection']))
                            for remed, cleaned in got:
                                yield est_name, remed, cleaned
                    similarity_sweep(
                        sweep_ctx, model_name=_m,
                        model_input=f'{_u}:{_rn}', X_tr=_xtr,
                        X_va=_xva, y_tr_noisy=y_noisy,
                        y_tr_clean=y_train_clean, y_val=y_val,
                        train_idx=train_idx, val_idx=val_idx,
                        groups_tr=_g_sw, fold=fold_idx - 1, draw=noise_seed,  # rows count folds from 0
                        recipes=sim_recipes,
                        base_reps=base_by_model.get(_m, []),
                        make_dprimes=_dprimes, level=sigma)

        # Aggregate across folds. One clean mean, one noisy mean and one gap PER
        # MODEL: recovery is a fraction of that model's own clean-to-noisy gap, and
        # sharing one gap across five models would divide every model's recovery by
        # random forest's damage.
        clean_by_model = {m: float(np.mean(v)) for m, v in clean_scores.items() if v}
        noisy_by_model = {m: float(np.mean(v)) for m, v in noisy_scores.items() if v}
        _models_scored = [u for u in eval_units
                          if u in clean_by_model and u in noisy_by_model]
        if not _models_scored:
            raise RuntimeError(
                f"n11: no evaluation unit produced both a clean and a noisy "
                f"reference score at level {sigma}. Units: {list(eval_units)}.")
        gap_by_model = {}
        gap_floor_by_model = {}
        for _m in _models_scored:
            _c, _nz = clean_by_model[_m], noisy_by_model[_m]
            gap_by_model[_m] = (_nz - _c) if lower_is_better else (_c - _nz)
            gap_floor_by_model[_m] = max(1e-10, GAP_MIN_FRAC * abs(_c))
        # The first model in the roster still fills the three top-level fields, so a
        # reader written before the roster widened still finds a number where it
        # expects one -- and `reference_model` says which model that number is.
        _ref_model = _models_scored[0]
        clean_mean = clean_by_model[_ref_model]
        noisy_mean = noisy_by_model[_ref_model]
        # Sign-normalised so a positive gap always means the injected noise cost
        # the model something. Unguarded clean-minus-noisy is NEGATIVE on rmse
        # exactly when the noise did damage. n07 was fixed this way on 2026-09-10
        # (metric-direction audit); this file was not, so the same field name
        # meant opposite things on an rmse dataset. Fixed 2026-09-12.
        # The recovery RATIO below is unaffected either way: its numerator and
        # denominator are both differences in the same metric, so the sign cancels.
        gap = gap_by_model[_ref_model]
        # A recovery ratio is only meaningful when the noise actually moved the
        # model. Require the gap to exceed a small fraction of the clean score's
        # magnitude (not just a bare 1e-10) so a near-zero gap does not blow the
        # ratio up into absurd percentages.
        gap_floor = gap_floor_by_model[_ref_model]

        sigma_result = {
            'clean_score': clean_mean,
            'noisy_score': noisy_mean,
            'gap': float(gap),
            'reference_model': _ref_model,
            'models_scored': list(_models_scored),
            'clean_score_by_model': clean_by_model,
            'noisy_score_by_model': noisy_by_model,
            'gap_by_model': {k: float(v) for k, v in gap_by_model.items()},
            'per_space': {},
        }

        for space_name in sorted(grid_acc.keys()):
            space_agg = {}
            for est_key in sorted(grid_acc[space_name].keys()):
                est_agg = {}
                for remed_key, score_list in grid_acc[space_name][est_key].items():
                    # Skip marker fields (carried through for diagnostics);
                    # forward them as-is rather than treating them like
                    # score lists.
                    if remed_key == '_detection_per_pass':
                        est_agg['detection'] = _mean_detection(score_list)
                        continue
                    if remed_key.startswith('_'):
                        est_agg[remed_key] = score_list
                        continue
                    if score_list:
                        denoised = float(np.mean(score_list))
                        _mn = remed_key.split('|', 1)[0]
                        _c = clean_by_model.get(_mn)
                        _nz = noisy_by_model.get(_mn)
                        if (_c is not None and _nz is not None
                                and abs(gap_by_model[_mn]) > gap_floor_by_model[_mn]):
                            recovery = float((denoised - _nz) / (_c - _nz) * 100)
                        else:
                            recovery = float('nan')
                        est_agg[remed_key] = {
                            'denoised_score': denoised,
                            'lower_is_better': bool(lower_is_better),
                            'recovery': recovery,
                            # Named, not only encoded in the key, so no reader has
                            # to split a string to know which model this is.
                            # 'model' is the evaluation unit ('rf@base_1');
                            # the two fields after it split it.
                            'model': _mn,
                            'downstream_model': eval_units[_mn][0],
                            'fitted_on': f'base_{eval_units[_mn][1]}',
                            'fitted_on_representation': eval_units[_mn][2],
                            'remediation': remed_key.split('|', 1)[-1],
                        }
                if est_agg:
                    space_agg[est_key] = est_agg
            sigma_result['per_space'][space_name] = space_agg

        # Recovery for a config picked without the fold it is read on. Pure
        # arithmetic over the per-fold scores just collected — no extra fitting.
        # One held-out recovery per model, for the same reason the gap is per model:
        # the config is re-picked from the other folds within that model's own arm.
        sigma_result['heldout_recovery_by_model'] = {
            _m: _heldout_recovery(clean_by_fold[_m], noisy_by_fold[_m],
                                  denoised_by_fold[_m], GAP_MIN_FRAC)
            for _m in _models_scored}
        sigma_result['heldout_recovery'] = (
            sigma_result['heldout_recovery_by_model'][_ref_model])

        results['per_sigma'][str(sigma)] = convert_numpy(sigma_result)

        # Print compact summary
        n_total = 0
        n_positive = 0
        best_recovery = float('-inf')
        best_config = None
        for space_name, space_data in sigma_result['per_space'].items():
            for est_key, est_data in space_data.items():
                for rk, rv in est_data.items():
                    # Skip diagnostic markers (`_failures`, `_errors`,
                    # `_smoothing_skipped_clf`) so they don't inflate counts
                    # or trip the recovery-comparison logic.
                    if rk.startswith('_') or not isinstance(rv, dict):
                        continue
                    n_total += 1
                    r = rv.get('recovery', float('nan'))
                    if not np.isnan(r) and r > 0:
                        n_positive += 1
                    if not np.isnan(r) and r > best_recovery:
                        best_recovery = r
                        best_config = f'{space_name}/{est_key}/{rk}'

        print(f"    Clean={clean_mean:.4f}, Noisy={noisy_mean:.4f}")
        print(f"    Configs: {n_total} total, {n_positive} positive "
              f"({100*n_positive/max(n_total,1):.1f}%)")
        if best_config:
            print(f"    Best over all configs on their own folds: "
                  f"{best_config} ({best_recovery:+.1f}%)")
            _ho = sigma_result.get('heldout_recovery')
            if _ho and not np.isnan(_ho['mean_recovery']):
                print(f"    Picked without the fold it is read on: "
                      f"{_ho['mean_recovery']:+.1f}% over "
                      f"{_ho['n_folds_scored']} fold(s), same config every "
                      f"fold={_ho['same_config_every_fold']}")
        elif n_total > 0:
            # All recoveries were NaN. Usually this means the (clean, noisy)
            # gap was near-zero for this sigma (no detectable noise effect)
            # so dividing by it produced NaN everywhere. Surface that
            # explicitly rather than silently skipping the "Best" line.
            print(f"    Best: None — all {n_total} configs returned NaN "
                  f"recovery (gap={gap:.4g}; sigma may be uninformative).")
        print(f"    [{time.time() - t_sigma:.0f}s]")

    # =========================================================================
    # 2026-09-12, owner's decision: n11's pool subset includes similarity
    # hybrids. Which objective each hybrid space was built under, and which
    # numbers in this file change units when that objective is 'similarity'.
    # This replaces results['hybrid_sim'], which recorded the coverage of the
    # hand-built four-block stack.
    # =========================================================================
    # Same denominator caution as n02, in n11's own terms:
    #
    #   clean_score, noisy_score, gap, denoised_score and recovery are all
    #   scored by n11 itself with metric_fn, on the held-out rows of its own
    #   folds, in the dataset's own metric. Every one of them is in that metric
    #   whichever family built the space.
    #
    #   Every space shares ONE denominator per evaluation unit. The gap is
    #   clean minus noisy for one model fitted on one of its two single
    #   representations, computed once per sigma, and every space's recovery
    #   for that unit divides by that same number. So a recovery in a
    #   similarity-built space and a recovery in a prediction-built space are
    #   the same quantity over the same bottom, and the two can be read against
    #   each other. (The gap is refused below GAP_MIN_FRAC of the clean score's
    #   magnitude, which is unchanged.)
    #
    #   The search scores below are the one place the units move. They are read
    #   straight out of the stored search, not measured here. Under
    #   objective='predictive' that number is the dataset's own metric; under
    #   objective='similarity' find_best_hybrid stores a model-independent
    #   number under every model name — the correlation between each held-out
    #   molecule's activity and what its five nearest neighbours predict. Each
    #   family's own score sits beside its own metric name; nothing stored puts
    #   the two on one ruler and the pool does not rank them against each other.
    results['pool_selection'] = {
        # Which hybrid estimation spaces this run holds, and — when it holds
        # none — why. An empty list with a reason beside it is a run that kept
        # its estimator x correction cross and dropped the space axis's hybrid
        # arm; before 2026-09-21 that case wrote no file at all.
        'hybrid_arm': hybrid_arm,
        'pool_subset': meta.get('pool_subset'),
        'pool_winner_objective': meta.get('objective'),
        'pool_winner_objective_choice': meta.get('objective_choice'),
        'dataset_metric': metric_name,
        'hybrid_spaces': {
            _r['space']: {
                'objective': _r['objective'],
                'slot': _r.get('slot'),
                'lineage': _r.get('lineage'),
                'budget_relaxed': _r.get('budget_relaxed'),
                'tier': _r['tier'],
                'aug_condition': _r['aug_condition'],
                'budget_features': _r['budget'],
                'allocation_method': _r['allocation_method'],
                'importance_method': _r['importance_method'],
                'internal_evaluator': _r['internal_evaluator'],
                'augmentation_strategy': _r['augmentation_strategy'],
                'search_score_mean': _r['search_score_mean'],
                'search_metric_name': _r['search_metric_name'],
                'search_lower_is_better': _r['search_lower_is_better'],
            }
            for _r in hybrid_recipes
        },
        'hybrid_build_errors': convert_numpy(hybrid_build_errors),
        'scores_in_dataset_metric': [
            'per_sigma.*.clean_score', 'per_sigma.*.noisy_score',
            'per_sigma.*.gap', 'per_sigma.*.per_space.*.denoised_score',
            'per_sigma.*.per_space.*.recovery', 'heldout_recovery'],
        'scores_read_from_the_search': [
            'pool_selection.hybrid_spaces.*.search_score_mean'],
        'recipe_selection_rows': meta.get('selection_scope'),
        'recipe_selection_note': meta.get('selection_note'),
        'recipes_saw_clean_labels': (
            'Both families rank a configuration against activity labels, and '
            'the pool searched on the clean ones. The prediction family scores '
            "a model's held-out prediction; the similarity family scores the "
            "correlation between each held-out molecule's activity and what "
            'its five nearest neighbours predict. So every hybrid space here '
            'is a best-case hybrid, the caveat that already applied to the '
            'prediction arm.'),
    }

    # The similarity axis, 2026-09-25. Every key above is n11's own grid,
    # unchanged. The swept numbers are the D-versus-D' rows in
    # results['similarity']['rows_file'] and, for detection,
    # {level: {estimator: {setting: {k: mean detection reading}}}}.
    results['similarity'] = similarity_record(
        sim_settings, sink, estimators_swept=sim_estimators,
        spaces_unavailable=sweep_ctx['unavailable'])
    results['detection_by_similarity'] = {
        _lv: {_e: {_sn: {_k: _mean_detection(_rows)
                         for _k, _rows in _byk.items()}
                   for _sn, _byk in _bys.items()}
              for _e, _bys in _bye.items()}
        for _lv, _bye in sim_detection.items()}

    # Summary
    results['summary'] = _compute_summary(results)

    return results


def _heldout_recovery(clean_by_fold, noisy_by_fold, denoised_by_fold,
                      gap_min_frac):
    """Recovery for a config chosen WITHOUT the fold it is read on.

    n11 reports the largest recovery among about 4,800 space x estimator x
    remediation x alpha configs, and every one of them was scored on the same
    folds that recovery is read from. The largest of 4,800 numbers measured on
    their own folds is partly luck, so it is not an estimate of how well that
    config recovers on rows it has not seen.

    This does the arithmetic that removes it, using per-fold scores that already
    exist — no extra fitting. For each fold in turn: rank every config by its
    recovery over the OTHER folds, take the top one, and read that config's
    recovery on this fold. Averaging those gives what picking a config this way
    is worth out of sample. The winner is allowed to differ from fold to fold,
    which is itself the answer to "is one config reliably best".

    Returns a dict, or None when fewer than two folds carry a usable gap.
    """
    folds = sorted(f for f in clean_by_fold if f in noisy_by_fold)
    if len(folds) < 2:
        return None

    clean_f = {f: float(clean_by_fold[f]) for f in folds}
    noisy_f = {f: float(np.mean(noisy_by_fold[f])) for f in folds}

    # {(space, est, remed): {fold: denoised}}
    flat = {}
    for space, ed in denoised_by_fold.items():
        for est, rd in ed.items():
            for remed, fd in rd.items():
                per_fold = {f: float(np.mean(v)) for f, v in fd.items()
                            if f in clean_f and len(v)}
                if len(per_fold) == len(folds):
                    flat[(space, est, remed)] = per_fold
    if not flat:
        return None

    def _recovery(cfg_folds, use):
        """Recovery over a set of folds, pooled the way the reported number is."""
        c = float(np.mean([clean_f[f] for f in use]))
        n = float(np.mean([noisy_f[f] for f in use]))
        d = float(np.mean([cfg_folds[f] for f in use]))
        gap = c - n
        if abs(gap) <= max(1e-10, gap_min_frac * abs(c)):
            return float('nan')
        return float((d - n) / gap * 100)

    per_fold_rows = []
    for f in folds:
        others = [g for g in folds if g != f]
        best_key, best_val = None, float('-inf')
        for key, cfg_folds in flat.items():
            r = _recovery(cfg_folds, others)
            if not np.isnan(r) and r > best_val:
                best_val, best_key = r, key
        if best_key is None:
            continue
        held = _recovery(flat[best_key], [f])
        per_fold_rows.append({
            'fold': int(f),
            'config': '/'.join(best_key),
            'recovery_on_other_folds': float(best_val),
            'recovery_on_this_fold': float(held),
        })

    scored = [r['recovery_on_this_fold'] for r in per_fold_rows
              if not np.isnan(r['recovery_on_this_fold'])]
    picks = {r['config'] for r in per_fold_rows}
    return {
        'mean_recovery': float(np.mean(scored)) if scored else float('nan'),
        'n_folds_scored': len(scored),
        'n_configs_ranked': len(flat),
        'per_fold': per_fold_rows,
        'same_config_every_fold': bool(len(picks) == 1),
        'configs_picked': sorted(picks),
        'what_this_is': (
            'For each fold the winning config is chosen from the other folds '
            'only, then read on this one. Compare it with best_config_search_max, '
            'which is the largest recovery over every config on the folds that '
            'produced it.'),
    }


# What the number at the end of a remediation key counts, per family. Named
# here so no field of the summary reports a bare value.
_CORRECTION_UNITS = {
    'weighting': ('the row-weight ratio parameter a: the most suspect molecule '
                  'gets 1/(1+a) of the cleanest one\'s weight, so 1.0 is 2-to-1 '
                  'and 3.0 is 4-to-1'),
    'smoothing': ('the fraction of the way from the label to its target that '
                  'the most suspect molecule\'s label is moved'),
    'quantile_clip': ('the fraction of the way onto the edge it crossed that a '
                      'label outside its 10th-to-90th-percentile range is '
                      'pulled; a label inside its range is untouched'),
}


def _parse_remediation(remed_key):
    """Split a remediation key into the parts a summary groups by.

    A key is '<model>|<rule>_<setting>' or '<model>|<target>+<rule>_<setting>'.
    Returns the model, the correction FAMILY (weighting, smoothing or the
    clip), the rule inside that family, where the label moved to, and the
    number swept. The family matters because the number means a different
    quantity in each one — see _CORRECTION_UNITS.
    """
    model, _, remed = remed_key.partition('|')
    rule_and_target, _, num = remed.rpartition('_')
    try:
        setting = float(num)
    except ValueError:
        rule_and_target, setting = remed, float('nan')
    target, _, rule = rule_and_target.rpartition('+')
    if rule in WEIGHT_METHODS:
        family = 'weighting'
    elif rule in SMOOTH_METHODS:
        family = 'smoothing'
    elif rule == 'quantile_clip':
        family = 'quantile_clip'
    else:
        family = 'unrecognised'
    return {
        'model': model,
        'method': rule_and_target,
        'correction': family,
        'rule': rule,
        # '' is the estimator's own target, which is what a key with no prefix
        # has always meant.
        'target': target,
        'setting': setting,
    }


def _compute_summary(results):
    """Compute summary: best configs, the swept setting per correction family,
    and the space comparison."""
    summary = {}

    for sigma_key, sigma_data in results['per_sigma'].items():
        gap = sigma_data.get('gap', 0)
        all_configs = []

        for space_name, space_data in sigma_data.get('per_space', {}).items():
            for est_key, est_data in space_data.items():
                for remed_key, rv in est_data.items():
                    # Skip diagnostic markers, same guard as the print loop.
                    # `_smoothing_skipped_clf` is a bool, so rv.get raised
                    # AttributeError here on every classification dataset —
                    # after the whole grid had run, which threw the run away.
                    if remed_key.startswith('_') or not isinstance(rv, dict):
                        continue
                    r = rv.get('recovery', float('nan'))
                    if not np.isnan(r):
                        _p = _parse_remediation(remed_key)
                        all_configs.append({
                            'space': space_name,
                            'estimator': est_key,
                            'method': _p['method'],
                            'correction': _p['correction'],
                            'rule': _p['rule'],
                            'target': _p['target'],
                            'setting': _p['setting'],
                            # Kept under its old name as well: it is what
                            # top_10 and best_config have always carried.
                            'alpha': _p['setting'],
                            'recovery': r,
                        })

        if not all_configs:
            summary[f'sigma_{sigma_key}'] = {}
            continue

        # Sort by recovery (descending)
        all_configs.sort(key=lambda x: x['recovery'], reverse=True)

        # ⛔ ONE FIELD WAS POOLING THE TWO KNOBS THE SWEPT LISTS WERE SPLIT
        # APART TO SEPARATE. This was `alpha_effect`, keyed by the number at
        # the end of the remediation key alone. A smoothing row's number is a
        # FRACTION of the way from the label to its target; a weighting row's
        # number is an unbounded RATIO parameter, where the most suspect
        # molecule gets 1/(1+a) of the cleanest one's row weight. 0.3 is in
        # both swept lists, so the entry keyed '0.3' averaged a label moved
        # three tenths of the way to its target together with a 1.3-to-1 row
        # weighting — the exact confusion SMOOTH_ALPHAS and WEIGHT_STRENGTHS
        # were split apart on 2026-09-18 to end, still merged one field later.
        #
        # The effect is now reported per correction family, so the two are
        # never averaged together, and each family names the unit of its own
        # number. n07 measured that moving a label further helps and n18
        # measured the opposite on real labels; this is the field that answers
        # it here.
        setting_recoveries = {}
        for c in all_configs:
            setting_recoveries.setdefault(c['correction'], {}).setdefault(
                str(c['setting']), []).append(c['recovery'])
        setting_effect = {
            fam: {
                'unit': _CORRECTION_UNITS.get(fam, 'the value swept'),
                'by_setting': {
                    a: {
                        'mean_recovery': float(np.mean(rs)),
                        'n_cells': len(rs),
                        'n_cells_positive': int(sum(1 for r in rs if r > 0)),
                        'pct_positive': float(
                            sum(1 for r in rs if r > 0) / len(rs) * 100),
                    }
                    for a, rs in sorted(by_a.items(), key=lambda kv: float(kv[0]))
                },
            }
            for fam, by_a in setting_recoveries.items()
        }

        # Per-space recovery (single representations vs hybrid). This is the hybrid-vs-base
        # comparison. It is ORACLE-BIASED — the hybrid recipe was selected on
        # CLEAN labels (see run(): get_best_search_for_downstream on clean
        # labels, reused per fold under noise). Hybrid spaces are therefore
        # BEST-CASE and NOT a fair baseline comparison. The numbers are kept for
        # transparency but carry an explicit caveat so they are not read as a
        # fair result. Do NOT headline hybrid vs base from this field.
        #
        # 2026-09-12, owner's decision: n11's pool subset includes similarity
        # hybrids, so there can be two hybrid spaces here, one per family, named
        # hybrid_predictive_<budget> and hybrid_similarity_<budget>. Both
        # recipes were ranked against clean activity labels, so the caveat above
        # covers both. Every space's recovery divides by the same clean-minus-
        # noisy gap, so the rows are comparable with each other.
        space_recoveries = {}
        for c in all_configs:
            s = c['space']
            if s not in space_recoveries:
                space_recoveries[s] = []
            space_recoveries[s].append(c['recovery'])
        space_effect_oracle_biased = {
            s: {
                'mean_recovery': float(np.mean(rs)),
                'pct_positive': float(sum(1 for r in rs if r > 0) / len(rs) * 100),
            }
            for s, rs in space_recoveries.items()
        }

        n_positive = sum(1 for c in all_configs if c['recovery'] > 0)

        # best_config / top_10 are the top of ~4,800 configs, every one scored
        # on the same folds this recovery is read from. They say which config
        # came first in the search, NOT how much noise that config recovers on
        # rows it has not seen. heldout_recovery below is the second number:
        # for each fold the winner is picked from the other folds and read on
        # this one. Renamed so the two are not confused at the point of use;
        # the old names are kept alongside for readers written before this.
        s = {
            'n_total': len(all_configs),
            'n_positive': n_positive,
            'pct_positive': float(n_positive / len(all_configs) * 100),
            'best_config_search_max': all_configs[0],
            'top_10_search_max': all_configs[:10],
            'search_max_caveat': (
                'Largest recovery over every config, scored on the folds that '
                'chose it. A search result, not a measured recovery — read '
                'heldout_recovery for the out-of-fold number.'),
            'heldout_recovery': sigma_data.get('heldout_recovery'),
            'best_config': all_configs[0],
            'top_10': all_configs[:10],
            'setting_effect': setting_effect,
            # Renamed from 'space_effect' to make the bias explicit at the point
            # of consumption: hybrid blueprint selected on clean labels => this
            # is a best-case hybrid, NOT a fair hybrid-vs-base comparison.
            'space_effect_oracle_biased': space_effect_oracle_biased,
            'space_effect_caveat': (
                'ORACLE-BIASED: every hybrid recipe here was selected on clean '
                'labels — the prediction family against a model\'s held-out '
                'score, the similarity family against the correlation between '
                'each held-out molecule\'s activity and what its five nearest '
                'neighbours predict. NOT a fair hybrid-vs-base comparison — '
                'best-case hybrid. The recovery metric per estimator x '
                'remediation is the sound primary claim. Which family built '
                'each space is in results["pool_selection"]["hybrid_spaces"].'
            ),
        }

        # Robust configs (positive at BOTH sigma levels if we have multiple)
        # This is computed after all sigmas in the final summary merge
        summary[f'sigma_{sigma_key}'] = convert_numpy(s)

    # Cross-sigma robust configs
    sigma_keys = [k for k in results['per_sigma'] if k != '0.0']
    if len(sigma_keys) >= 2:
        # Collect all config keys that are positive at every sigma
        config_recovery = {}  # config_key -> {sigma: recovery}
        for sigma_key in sigma_keys:
            sigma_data = results['per_sigma'][sigma_key]
            for space_name, space_data in sigma_data.get('per_space', {}).items():
                for est_key, est_data in space_data.items():
                    for remed_key, rv in est_data.items():
                        # Same guard as the per-sigma loop above.
                        if remed_key.startswith('_') or not isinstance(rv, dict):
                            continue
                        r = rv.get('recovery', float('nan'))
                        if not np.isnan(r):
                            ck = f'{space_name}/{est_key}/{remed_key}'
                            if ck not in config_recovery:
                                config_recovery[ck] = {}
                            config_recovery[ck][sigma_key] = r

        robust = []
        for ck, sr in config_recovery.items():
            if len(sr) >= 2 and all(r > 0 for r in sr.values()):
                avg = float(np.mean(list(sr.values())))
                robust.append({'config': ck, 'avg_recovery': avg,
                               'per_sigma': sr})

        robust.sort(key=lambda x: x['avg_recovery'], reverse=True)
        # Same caveat as best_config: these 20 are the top of the same ~4,800
        # scored on the folds that ranked them. What they add is that each one
        # is positive at every sigma, which the single maximum does not check.
        summary['robust_configs_search_max'] = convert_numpy(robust[:20])
        summary['robust_configs'] = convert_numpy(robust[:20])
        summary['robust_configs_caveat'] = (
            'Ranked on the folds that chose them. Positive at every sigma is '
            'the claim; the size of the recovery is a search maximum.')
        summary['n_robust'] = len(robust)

    return summary


if __name__ == '__main__':
    parser = create_experiment_parser('N11: Estimation × Remediation Grid')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'n11_estimation_remediation_grid',
                            args.dataset)
