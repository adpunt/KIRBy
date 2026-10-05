#!/usr/bin/env python3
"""N11: Noise Estimation × Remediation Grid.

Paper: Noise

SCOPE: MOLECULAR datasets only, and SYNTHETIC noise only (the decided noise
kinds and levels on regression labels, label-flipping on classification
labels -- see `noise_helpers.inject_condition`). Results do NOT speak to
real-world label noise or to non-molecular domains.

Full grid from the confirmation report: estimator × smoothing/weighting × alpha ×
distance metric × representation. Extends with hybrid representations.

PRIMARY REPORTED CLAIM: the held-out RECOVERY metric (multi-seed) — how much of
the clean→noisy performance gap a given estimator × remediation config recovers.
This is scientifically sound.

CHAT 5 ALIGNMENT, 2026-10-05 (HANDOFF_EXPERIMENT_ALIGNMENT.md section 2.4;
group C of 2.3 'Still open': n11 switches to 2 + 2 with distance hybrids only
as detection and neighbour spaces). This replaces the whole-dataset design
described in earlier versions of this docstring (the pool lookup of
2026-09-12, every model's own top 2 general + top 1 distance hybrids of
2026-10-02, the top-2 single representations ranked on the stored search of
2026-09-25, the five-model roster of 2026-09-12). What a run now does:

  * Models: the dataset's two decided models (decided_grid.decided_models,
    DECISIONS.md Step 1). knn only inside the distance search.
  * Outer folds: noise_helpers.noise_outer_folds (count from
    config['n_splits'] or KIRBY_NOISE_N_FOLDS; any count, including 1).
  * Per fold, per model, from searches on that fold's training rows with clean
    labels (noise_helpers.fold_configurations): 2 base representations and 2
    general hybrids = the 4 FITTED configurations, keyed '<model>@base_1',
    '<model>@base_2', '<model>@hybrid', '<model>@hybrid__general2'. 2 models x
    4 = 8 (R5). Each is an evaluation unit with its own clean score, damaged
    score and gap.
  * Noise-estimation spaces: each fitted configuration; each model's top
    distance-search hybrid and top similarity-search hybrid, which are
    estimation spaces only and never fitted on; and every other pruned
    representation, capped, as before. Hybrids are rebuilt on the fold's
    training rows from each draw's damaged labels with the recipe's own
    representation count (noise_helpers.fold_model_inputs).
  * Noise: the decided kinds and levels (noise_helpers.inject_condition,
    decided_levels); 3 draws per non-zero level, level 0 once.
  * Neighbour settings: the similarity sweep reads the settings of the
    dataset's group (KNN_SEARCH_SPACE_RECORD), screened, top 3 refitted, its
    D' rows fitted on the unit's own model only. The estimator grid's own
    neighbour estimators run only in the axis 1 spaces among the estimation
    spaces, each screened over the group's distances, 5/10/20 neighbours and
    the two counting schemes, top 3 refitted, the label target being the
    kept scheme's neighbour label.
  * Results: per fold (folds.<i>), per level (per_sigma), each (fold, level,
    draw) checkpointed as it finishes.

The recipe of each hybrid is chosen on the fold's clean training labels and
never sees the held-out rows; its features are re-chosen from the damaged
labels at every draw.

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
pulls each label outside it onto the edge it crossed. Since 2026-10-05, in
the similarity sweep a neighbour estimator's own target is its neighbours'
label under the kept counting scheme ('rank' or 'inverse_square',
KNN_SEARCH_SPACE_RECORD axis 5), not their plain mean.

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

from experiments.dataset_policy import noise_groups_for
from kirby.noise_spec import GROUPED_STRATEGIES
from kirby.scaling import FeatureScaler
from experiments.noise.noise_helpers import (
    detection_scores, sweep_conditions, conditions_for_task,
    noise_outer_folds, noise_n_folds, fold_spread, fold_configurations,
    fold_model_inputs, configuration_record, inject_condition, decided_levels,
    dataset_knn_group, level_unit, CONFIGURATION_SPACES)
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_domain_metric, get_model_factories, cap_dimensions, as_rep_dtype,
)
from experiments.decided_grid import decided_models
from kirby.utils.noise_estimation import (
    get_noise_estimator, smooth_by_scores, weights_from_scores,
    clip_to_range, GaussianProcessEstimator,
)
# The similarity-sweep kit and the disk caches, added 2026-09-25 (plan
# sections 1-3). See n07_remediation's SIMILARITY-SWEEP KIT block.
from experiments.noise.n07_remediation import (
    sweep_settings, sweep_ks, sweep_schemes, screen_top, SCREEN_SCORED_ON,
    RowSink, sweep_context, FoldCheckpoint,
    cleaned_from_weights, cleaned_from_labels, similarity_record,
    _predict_for_metric, screen_settings, fold_neighbours,
    fold_similarity_spaces, dprime_rows, row_identity,
)
from experiments.noise import noise_cache as _nc
from experiments.noise import similarity_space as _ss
from experiments.noise.noise_cache import NEIGHBOUR_DETECTORS

# 2026-10-05 (chat 5): what this file writes. Stamped on every condition file,
# written into the condition files' fingerprint (run() below), the per-unit
# checkpoint's and the rows file's name, so nothing written by the
# whole-dataset design before this date is read back. The 'b' (also
# 2026-10-05) marks two changes: the estimator grid's neighbour estimators
# are screened (5/10/20 neighbours, two counting schemes, top 3 refitted) in
# place of a fixed 10 neighbours under every distance, and the D' rows are
# fitted on the unit's own model only (no rf/svm). Units, condition files and
# rows from before it are not read back.
RESULT_LAYOUT = 'fold_aligned_2026-10-05b'

# 2026-10-05 (chat 5): the spaces of one model that are noise-estimation
# spaces only and are never fitted (handoff 2.3 item 1 and group C of 'Still
# open': n11 switches to 2 + 2 with distance hybrids only as detection and
# neighbour spaces). 'hybrid__distance1' is the model's top hybrid of the
# per-fold distance search; 'hybrid_similarity' the top hybrid of the per-fold
# similarity search.
NEIGHBOUR_HYBRID_SPACES = ('hybrid__distance1', 'hybrid_similarity')
# The two of a model's four fitted configurations whose features are chosen
# from the labels, so they are rebuilt from each draw's damaged labels.
HYBRID_CONFIGURATIONS = ('hybrid', 'hybrid__general2')

# Noise draws per non-zero level. Level 0 (the labels as delivered) runs once.
NOISE_SEEDS = (0, 1, 2)

# The noise condition run when a caller names one and passes no list. run()
# sweeps every decided kind (conditions_for_task).
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
# than across runs. 2026-10-05 (chat 5): the list is the dataset group's
# distances (KNN_SEARCH_SPACE_RECORD axis 2: manhattan, cosine, minkowski
# p=0.5 on full_knn; manhattan, cosine on fast_knn; euclidean cut in both),
# read per run from similarity_space.GROUP_METRICS. It was every name in
# kirby.similarity.distance_config.DISTANCE_METRICS, euclidean included.
def neighbor_metrics(knn_group):
    return list(_ss.GROUP_METRICS[knn_group])

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
# `noise_helpers.inject_condition` (2026-10-05, chat 5: the decided kinds and
# levels), which records what was delivered.


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


def _build_estimation_spaces(unit_train, neighbour_train, base_by_model, reps,
                             train_idx, models):
    """The noise-estimation spaces of one (fold, level, draw), 2026-10-05.

    Args:
        unit_train: {unit: (model, configuration, X_train)} for every fitted
            configuration that was built (unit '<model>@<configuration>').
        neighbour_train: {model: {space: X_train}} for the estimation-only
            hybrid spaces (NEIGHBOUR_HYBRID_SPACES).
        base_by_model: {model: [its two base representation names]}.
        reps: data['reps'], every pruned representation, full rows.
        train_idx: this fold's training rows.
        models: the models with at least one fitted configuration.

    Returns (spaces, models_by_space, space_info):
        spaces           {space: X_train}
        models_by_space  {space: [models whose units are scored in it]}
        space_info       {space: {'kind', 'model', 'what', 'also'}}

    Three kinds of space:
      * each fitted configuration of each model ('<configuration>@<model>'),
        scored for that model's four units;
      * each model's distance hybrid and similarity hybrid
        ('hybrid__distance1@<model>', 'hybrid_similarity@<model>'), scored for
        that model's units and never fitted on;
      * every other pruned representation, capped (cap_dimensions) and scored
        for the models that do not have it as a base representation -- the
        space axis this file has always had.
    Two spaces holding the same matrix (same content hash) are built once and
    both models' units are scored in it; 'also' says what else it stands for.
    """
    spaces, models_by_space, info, by_hash = {}, {}, {}, {}

    def _add(name, X, ms, kind, model, what):
        h = _nc.array_hash(X)
        if h in by_hash:
            nm = by_hash[h]
            for m in ms:
                if m not in models_by_space[nm]:
                    models_by_space[nm].append(m)
            info[nm]['also'].append({'space': name, 'models': list(ms),
                                     'what': what})
            return
        by_hash[h] = name
        spaces[name] = X
        models_by_space[name] = list(ms)
        info[name] = {'kind': kind, 'model': model, 'what': what, 'also': []}

    for u, (m, cfg, X_tr) in unit_train.items():
        _add(f'{cfg}@{m}', X_tr, [m], 'fitted_configuration', m, cfg)
    for m, by_sp in neighbour_train.items():
        for sp, X_tr in by_sp.items():
            _add(f'{sp}@{m}', X_tr, [m], 'estimation_only_hybrid', m, sp)
    # These per-representation spaces are not on KNN_SEARCH_SPACE_RECORD axis
    # 1's list of neighbour spaces, so since 2026-10-05 (chat 5) the
    # neighbour estimators skip them (_neighbour_axis_space); the estimators
    # that read no neighbours still run in them, the space axis this
    # experiment has always had.
    tr = np.asarray(train_idx)
    for rep_name, X_rep in reps.items():
        ms = [m for m in models if rep_name not in (base_by_model.get(m) or [])]
        if not ms:
            continue
        X = cap_dimensions(FeatureScaler().fit_transform(
            as_rep_dtype(np.asarray(X_rep)[tr])))[0]
        _add(rep_name, X, ms, 'representation_capped', None, rep_name)
    return spaces, models_by_space, info


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
                         dprime_cb=None, fit_models=True,
                         score_metric_name=None, neighbour_target=None):
    """Run full alpha × remediation grid for one estimator config.

    2026-10-05 (chat 5), two arguments:
      score_metric_name  the dataset's metric name (get_domain_metric). On
                         'auc' every fit is scored on the positive-class
                         probability (n07._predict_for_metric), not on 0/1
                         calls, which made AUC a balanced-accuracy stand-in.
      neighbour_target   the neighbours' label under the swept counting scheme
                         (similarity_space.weighted_neighbour_label), handed in
                         by the similarity sweep. When the estimator's own
                         target is its neighbours' mean (neighbor_means_ or
                         weighted_means_) it is replaced by this, so where a
                         label moves to follows KNN_SEARCH_SPACE_RECORD axis 5
                         ('rank' or 'inverse_square'). The suspicion SCORE
                         still averages the neighbours inside the estimator
                         (src/kirby/utils/noise_estimation.py), which this
                         file does not own; listed as a shared change.

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
    evaluation unit ('<model>@<configuration>', configuration one of base_1,
    base_2, hybrid, hybrid__general2), so each unit is fitted on its own
    configuration's features. None fits every entry on X_eval_train/X_eval_val.

    extra_targets: {name: per-training-row array} — smoothing targets computed
    outside this function, once per (fold, draw, space), and reused by every
    estimator. Each one is swept over the same smoothing methods and alphas as
    the estimator's own target, under keys that carry the target name; see
    EXTRA_TARGETS. None or {} reproduces the grid exactly as it was.

    Returns: {remed_key: denoised_score} where remed_key =
    f'{model}|{method}_{alpha}'. The model half is the evaluation unit
    ('<model>@<configuration>', 2026-10-05), because recovery is a fraction of
    that unit's own clean-to-noisy gap. The bar is a separator that no model name and
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
    if (neighbour_target is not None
            and target_source in ('neighbor_means_', 'weighted_means_')):
        target = np.asarray(neighbour_target)
        target_source = 'scheme_weighted_neighbour_label'
    results = {}

    def _pred(model, X):
        return _predict_for_metric(model, X, score_metric_name)

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
                    y_pred = _pred(model, _xva)
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
                            y_pred = _pred(model, _xva)
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
                        y_pred = _pred(model, _xva)
                        results[key] = float(metric_fn(y_val, y_pred))
                    except Exception as exc:
                        failure_log[key] = f'{type(exc).__name__}: {exc}'

    if is_clf:
        results['_smoothing_skipped_clf'] = True
    if failure_log:
        results['_failures'] = failure_log

    return results


def _condition_levels(data, dataset_name, condition, override=None):
    """The levels one condition runs at, level 0 first: the decided levels for
    this dataset's group (noise_helpers.decided_levels, KNN_SEARCH_SPACE_RECORD
    "The noise grid", Adelaide 2026-10-05), unless a smoke or direct caller
    passes its own."""
    if override:
        return tuple(float(x) for x in override)
    return tuple(float(x) for x in decided_levels(data, dataset_name,
                                                  condition))


def run(data, config, output_dir, dataset_name, **kwargs):
    """n11_estimation_remediation_grid across every condition this dataset can be given.

    The condition is an axis this experiment sweeps, not something a caller
    runs it once per. The sweep, the per-condition checkpointing, the resume
    and the stride slice are noise_helpers.sweep_conditions.

    Its headline is a downstream score, which is defined under every condition,
    so it sweeps the whole list rather than only the ones where a corrupted
    SUBSET exists. It also records detection, and detection_scores already
    returns None where that is undefined.

    2026-10-05 (chat 5): the conditions are the decided kinds
    (conditions_for_task), each at its decided levels, and the condition files
    are fingerprinted with the levels, the draws, the outer fold count
    (KIRBY_NOISE_N_FOLDS included), the folds run (KIRBY_NOISE_ONLY_FOLDS) and
    RESULT_LAYOUT, so a file written by the whole-dataset design is recomputed
    instead of skipped.
    """
    conditions = list(conditions_for_task(data))
    _override = kwargs.get('sigmas') or kwargs.get('levels')
    fp_config = dict(config)
    fp_config['n_splits'] = noise_n_folds(config)
    fp_config['noise_levels'] = {
        c: list(_condition_levels(data, dataset_name, c, _override))
        for c in conditions}
    fp_config['noise_seeds'] = list(kwargs.get('noise_seeds') or NOISE_SEEDS)
    fp_config['noise_strategies'] = [
        RESULT_LAYOUT,
        f"only_folds={os.environ.get('KIRBY_NOISE_ONLY_FOLDS') or 'all'}"]
    return sweep_conditions(
        _run_one_condition, 'n11_estimation_remediation_grid',
        data, fp_config, output_dir, dataset_name,
        default_conditions=conditions, kwargs=kwargs,
        groups_needed=GROUPED_STRATEGIES)


def _merge_grid_result(acc, grid_result):
    """Fold one _run_estimator_grid result into the per-pass record
    {remed_key: score, '_failures': {...}, '_detection': {...}, ...}."""
    if '_error' in grid_result:
        acc.setdefault('_errors', []).append(grid_result['_error'])
        return
    for remed_key, score in grid_result.items():
        if remed_key == '_failures':
            acc.setdefault('_failures', {}).update(score)
        else:
            acc[remed_key] = score


def _neighbour_axis_space(space_name, info, group_spaces):
    """The KNN_SEARCH_SPACE_RECORD axis 1 name an estimation space stands for,
    or None. 2026-10-05 (chat 5). A model's fitted configuration
    ('base_1@rf' -> 'base_1') or estimation-only hybrid
    ('hybrid__distance1@rf' -> 'hybrid__distance1') is an axis 1 space when
    the dataset's group sweeps it (fast_knn sweeps no 'base_2'). The capped
    per-representation spaces are not on axis 1's list."""
    names = [info.get('what')] + [a.get('what') for a in info.get('also') or []]
    if info.get('kind') == 'representation_capped':
        names = [a.get('what') for a in info.get('also') or []]
    for nm in group_spaces:
        if nm in names:
            return nm
    return None


def _screened_neighbour_grid(est_names, axis_space, X_space, y_noisy, *,
                             dataset_name, nbr_metrics, metric, is_clf,
                             grid_kwargs):
    """The estimator grid's neighbour estimators in one estimation space,
    under the kNN record. 2026-10-05 (chat 5).

    Every (distance of the group, k in 5/10/20, counting scheme rank or
    inverse_square) is scored with no model fit (n07.screen_settings, on the
    training labels held); only the best 3 are run through the correction grid
    with model refits. Neighbours are found in this space's own matrix, with
    scaffold-mates kept in the lists (axis 6). Each kept setting's neighbour
    label under its counting scheme is the label target of the neighbour
    estimators (neighbour_target).

    Until this date each neighbour estimator ran here under every distance at
    a fixed 10 neighbours with equal weight, unscreened, refitted for each.

    'hybrid' (HybridNoiseEstimator) takes no neighbour list. It is run with the
    kept setting's distance and k, so it finds the same neighbours in the same
    matrix itself; the scheme reaches it only through the label target.

    Returns ({est_key: grid_result}, screen record). est_key is
    '<estimator>_<distance>|k=<k>|<scheme>'.
    """
    settings = [_ss.SimilaritySetting(axis_space, 'none', d)
                for d in nbr_metrics]
    nbrs = fold_neighbours(dataset_name, settings, {axis_space: X_space}, None,
                           y_noisy, exclude_groups=None, is_clf=is_clf)
    kept, table = screen_settings(settings, nbrs, y_noisy, metric=metric,
                                  is_clf=is_clf)
    out = {}
    for s, k, scheme in kept:
        L = dict(nbrs[s.name], scheme=scheme)
        try:
            _i, _d = _ss.neighbours_at(L, k)
            target = _ss.weighted_neighbour_label(y_noisy, _i, _d, scheme,
                                                  is_clf=is_clf)
        except Exception:
            target = None
        for est in est_names:
            if est in NEIGHBOUR_DETECTORS:
                res = _run_estimator_grid(
                    est, None, X_space, y_noisy, *grid_kwargs['args'],
                    precomputed_neighbors=L, k_override=k,
                    neighbour_target=target, **grid_kwargs['kw'])
            else:
                res = _run_estimator_grid(
                    est, s.metric, X_space, y_noisy, *grid_kwargs['args'],
                    k_override=k, neighbour_target=target,
                    **grid_kwargs['kw'])
            out[f'{est}_{s.metric}|k={k}|{scheme}'] = res
    rec = {'axis_space': axis_space,
           'kept': [f'{s.name}|k={k}|{sch}' for s, k, sch in kept],
           'unavailable': {n: L.get('reason') for n, L in nbrs.items()
                           if not L.get('available')},
           'table': table}
    return out, rec


def _similarity_sweep_own_model(ctx, *, model_name, model_input, X_tr, X_va,
                                y_tr_noisy, y_tr_clean, y_val, train_idx,
                                val_idx, groups_tr, fold, draw, recipes,
                                base_reps, make_dprimes, level):
    """n07.similarity_sweep with the D' rows fitted on the unit's own model
    only. 2026-10-05 (chat 5).

    n07.similarity_sweep fits every D' row with row_models([model_name]),
    which puts rf and svm (n07.SWEEP_ROW_MODELS) in front of the model, so
    each row n11 wrote also fitted two models the dataset may not have
    decided (R1). This is the same pass with models=(model_name,): the same
    spaces (fold_similarity_spaces), the same lists (fold_neighbours, no
    exclusion), the same screen (screen_settings, top 3 refitted), the same
    row identity. When n07.similarity_sweep stops adding rf and svm, n11 can
    call it again.
    """
    sink = ctx['sink']
    ds = ctx['dataset_name']
    settings = ctx['settings']
    spaces_tr, spaces_va, unavail = fold_similarity_spaces(
        ds, settings, model_input_tr=X_tr, model_input_va=X_va,
        reps=ctx['reps'], train_idx=train_idx, val_idx=val_idx,
        y_train=y_tr_noisy, groups_train=groups_tr, base_reps=base_reps,
        recipes=recipes, augmentations=ctx['augmentations'],
        model_name=model_name, model_factory=ctx['factories'].get(model_name),
        metric=ctx['metric'], task_type=ctx['task_type'])
    smi = ctx.get('smiles')
    smi_tr = None if smi is None else [smi[i] for i in np.asarray(train_idx)]
    smi_va = None if smi is None else [smi[i] for i in np.asarray(val_idx)]
    nbrs = fold_neighbours(ds, settings, spaces_tr, spaces_va, y_tr_noisy,
                           exclude_groups=None, smiles_train=smi_tr,
                           smiles_val=smi_va, is_clf=ctx['is_clf'])
    models = (model_name,)
    base_id = dict(dataset_name=ds, experiment=ctx['experiment'],
                   model_input=model_input, noise_condition=ctx['condition'],
                   level=level, draw=draw, fold=fold)
    missing = {}
    for s in settings:
        L = nbrs.get(s.name)
        if L is None or not L.get('available'):
            why = unavail.get(s.space) or (L or {}).get('reason')
            missing[s.name] = why
            ctx['unavailable'].setdefault(f'{model_input}|{s.name}', why)
    kept, table = screen_settings(settings, nbrs, y_tr_noisy,
                                  metric=ctx['metric'], is_clf=ctx['is_clf'],
                                  y_val=y_val)
    ctx.setdefault('screen', []).append({
        'model_input': model_input, 'fold': fold, 'draw': draw,
        'level': level, 'condition': ctx['condition'],
        'kept': [f'{s_.name}|k={k_}|{sch}' for s_, k_, sch in kept],
        'table': table})
    for s, k, scheme in kept:
        L = dict(nbrs[s.name], scheme=scheme)
        for method, msetting, cleaned in make_dprimes(s, L, k):
            ms = f'k={k}|scheme={scheme}'
            ms = ms if not msetting else f'{ms}|{msetting}'
            if sink is None or not sink.enabled:
                continue
            dprime_rows(sink, dataset_name=ds, cleaned=cleaned, X_tr=X_tr,
                        y_D=y_tr_noisy, X_va=X_va, y_va=y_val,
                        metric=ctx['metric'], task_type=ctx['task_type'],
                        y_clean=y_tr_clean, models=models,
                        is_clf=ctx['is_clf'],
                        identity=row_identity(
                            method=method, method_setting=ms,
                            similarity_setting=s.name, **base_id))
    return missing


def _run_one_condition(data, config, output_dir, dataset_name, **kwargs):
    """N11: one condition at every decided level.

    CHAT 5 ALIGNMENT, 2026-10-05 (HANDOFF_EXPERIMENT_ALIGNMENT section 2.4,
    group C of 2.3 'Still open': n11 switches to 2 + 2 with distance hybrids
    only as detection and neighbour spaces). What each fold now does:
      1. Folds are noise_helpers.noise_outer_folds: the per-fold searches' own
         splits, the count from config['n_splits'] or KIRBY_NOISE_N_FOLDS.
      2. noise_helpers.fold_configurations runs this fold's searches on its
         training rows (clean labels, decided grid, pruning off) and gives each
         of the dataset's two decided models its 2 best base representations,
         its 2 best general hybrids, its top distance-search hybrid and the top
         similarity-search hybrid. Nothing is read from a whole-dataset search.
      3. Evaluation units are '<model>@<configuration>', the 4 configurations
         of each model (base_1, base_2, hybrid, hybrid__general2): 2 models x
         4 = 8 (R5). The two hybrids are rebuilt on the fold's training rows
         from each draw's damaged labels (noise_helpers.fold_model_inputs,
         which keeps the recipe's representation count and the fold search's
         own representation scores: dirty-run bug 1 and n11's own bug, hybrid
         estimation spaces built on all 10 representations).
      4. Every estimator that reads no neighbours x correction runs in every
         estimation space (_build_estimation_spaces), and every unit open on
         that space is fitted on the repaired labels. The neighbour
         estimators run only in the estimation spaces on axis 1 of
         KNN_SEARCH_SPACE_RECORD for this group, screened there
         (_screened_neighbour_grid: group distances x 5/10/20 neighbours x
         rank/inverse_square, scored with no fit, top 3 refitted).
      5. The similarity sweep (_similarity_sweep_own_model, n07's sweep with
         the rows fitted on the unit's model only) runs once per unit: the
         settings of this dataset's group, screened, the top 3 refitted.
    Each (fold, level, draw) is written to a checkpoint file as it finishes.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT N11: NOISE ESTIMATION × REMEDIATION GRID")
    print("=" * 70)

    from experiments._shared import input_fingerprint
    from kirby.noise_robustness import robustness_metrics

    reps = data['reps']
    labels = np.asarray(data['labels'])
    groups = data.get('groups')
    groups_arr = None if groups is None else np.asarray(groups)
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    metric = (metric_name, metric_fn, lower_is_better)
    pred_mode = _nc.predict_mode_for(metric_name)
    n_samples = len(labels)
    smiles = data.get('smiles')
    has_smiles = smiles is not None and len(smiles) > 0
    n_molecules = len(set(smiles)) if has_smiles else n_samples
    knn_grp = dataset_knn_group(data, dataset_name)
    nbr_metrics = neighbor_metrics(knn_grp)
    # The spaces the estimator grid's neighbour estimators may run in
    # (KNN_SEARCH_SPACE_RECORD axis 1 for this group), 2026-10-05 (chat 5).
    group_sp = _ss.group_spaces(knn_grp, has_smiles=has_smiles)

    # R1: the dataset's two decided models (DECISIONS.md Step 1), nothing
    # else. knn runs only inside the distance search. Replaces
    # DOWNSTREAM_MODELS / config['models'] (five models) and a `models` kwarg.
    models = [m for m in decided_models(dataset_name, 'general') if m != 'knn']

    noise_condition = kwargs.get('noise_condition') or NOISE_CONDITION
    levels = _condition_levels(data, dataset_name, noise_condition,
                               kwargs.get('sigmas') or kwargs.get('levels'))
    draws_all = tuple(int(d) for d in (kwargs.get('noise_seeds')
                                       or NOISE_SEEDS))
    noise_groups = noise_groups_for(data)[0]

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
    # The same switch _resolve_estimators reads. Named here because the
    # Gaussian-process smoothing target is the same O(n^3) fit that flag exists
    # to drop, so it has to see the flag too.
    light_estimators = os.environ.get('KIRBY_NOISE_LIGHT_ESTIMATORS') == '1'

    folds, n_folds = noise_outer_folds(data, config)

    print(f"  Dataset: {dataset_name} ({n_samples} rows, {n_molecules} "
          f"molecules, group {knn_grp})")
    print(f"  Metric: {metric_name} (lower is better: {lower_is_better})")
    print(f"  Models (decided): {models}")
    print(f"  Estimators: {len(available_estimators)}")
    if skipped_estimators:
        print(f"  Skipped (regression-only on clf): {skipped_estimators}")
    print(f"  Condition {noise_condition}, levels {list(levels)} "
          f"({level_unit(noise_condition, is_clf)}), draws {list(draws_all)} "
          f"per non-zero level")
    print(f"  Folds: {n_folds}; neighbour distances in the estimator grid: "
          f"{nbr_metrics}")

    # ── The similarity axis (added 2026-09-25; KNN record 2026-10-05) ───────
    # Each neighbour estimator (noise_cache.NEIGHBOUR_DETECTORS that this run
    # enables) reads its neighbours from every setting of this dataset's group,
    # screened per pass (n07.screen_settings inside
    # _similarity_sweep_own_model, n11's copy of n07.similarity_sweep that fits
    # each D' row on the unit's own model only, 2026-10-05: every
    # (setting, k in 5/10/20, counting scheme rank/inverse_square) scored with
    # no model fit; the top 3 refitted). Scaffold-mates stay in the lists. The
    # full correction grid of _run_estimator_grid is applied to each kept
    # setting's scores, and the label target of a neighbour estimator is the
    # neighbours' label under the kept counting scheme. One D-versus-D' row per
    # (estimator, correction, setting, k, scheme), fitted on the unit's model.
    #
    # TODO(similarity-sweep 2026-09-25): 'hybrid' (HybridNoiseEstimator)
    # averages three neighbour estimators but takes no neighbour list, so it
    # is not in the sweep; giving it one is a change to noise_estimation.py.
    sim_settings = sweep_settings(dataset_name, n_molecules, has_smiles)
    # A new rows file: rows written by the whole-dataset design share model
    # input names with these and would be skipped as done.
    sink = RowSink(output_dir, 'n11_estimation_remediation_grid',
                   dataset_name, f'{noise_condition}__{RESULT_LAYOUT}')
    sim_estimators = [e for e in available_estimators
                      if e in NEIGHBOUR_DETECTORS]
    sweep_ctx = sweep_context(
        experiment='n11_estimation_remediation_grid',
        dataset_name=dataset_name, sink=sink, settings=sim_settings,
        metric=metric, task_type=task_type, condition=noise_condition,
        level=None, data=data, factories={})
    print(f"  Similarity axis: {len(sim_settings)} settings x ks "
          f"{list(sweep_ks())} x schemes {list(sweep_schemes())}, top "
          f"{screen_top()} refitted, estimators {sim_estimators}; rows -> "
          f"{sink.path}")

    ckpt = FoldCheckpoint(
        output_dir, 'n11_estimation_remediation_grid', dataset_name,
        condition=noise_condition, tag='units', fingerprint={
            'layout': RESULT_LAYOUT, 'input': input_fingerprint(data, config),
            'condition': noise_condition, 'levels': list(levels),
            'draws': list(draws_all), 'n_folds': int(n_folds),
            'models': models, 'estimators': available_estimators,
            'neighbour_metrics': nbr_metrics,
            'settings': [s.name for s in sim_settings],
            'screen_top': screen_top(),
            'extra_targets': list(EXTRA_TARGETS),
            'smooth_alphas': list(SMOOTH_ALPHAS),
            'weight_strengths': list(WEIGHT_STRENGTHS)})

    fold_records = {}
    units_done = []          # [{'fold', 'level', 'draw', 'value'}]
    unit_meta = {}           # {unit: {'model', 'configuration'}}

    for fold_i, train_idx, val_idx in folds:
        print(f"\n  Fold {fold_i} of {n_folds}: {len(train_idx)} training, "
              f"{len(val_idx)} held-out rows")
        try:
            entries = fold_configurations(data, dataset_name, fold_i=fold_i,
                                          train_idx=train_idx, n_folds=n_folds)
        except Exception as exc:
            print(f"    per-fold searches failed: {type(exc).__name__}: {exc}")
            fold_records[str(fold_i)] = {
                'error': f'per-fold searches failed: {type(exc).__name__}: '
                         f'{exc}'}
            continue
        y_tr_clean = labels[train_idx]
        y_val = labels[val_idx]
        g_tr = None if groups_arr is None else groups_arr[train_idx]
        # Factories built for this fold's training-row count (the svm row cap
        # is checked against it; it read the whole dataset's count before).
        factories = get_model_factories(models, task_type=task_type,
                                        n_samples=len(train_idx))
        sweep_ctx['factories'] = factories

        # Each model's 4 configurations with features chosen from the CLEAN
        # training labels. The base pair is the same matrix at every level; the
        # hybrid pair is the clean-label-features reference.
        clean_inputs = {}
        frec = {'n_train': int(len(train_idx)), 'n_val': int(len(val_idx)),
                'models': {}}
        for m in models:
            entry = entries.get(m)
            if m not in factories:
                frec['models'][m] = {'available': False,
                                     'why': 'no model factory for this name'}
                continue
            ins, unav, names = fold_model_inputs(
                data, dataset_name, m, entry, train_idx=train_idx,
                val_idx=val_idx, y_train=y_tr_clean, factory=factories[m],
                metric=metric)
            clean_inputs[m] = (entry, ins, unav, names)
            rec = configuration_record(entry, names)
            rec['configurations_unavailable'] = {
                sp: why for sp, why in unav.items()
                if sp in CONFIGURATION_SPACES}
            rec['n_features'] = {sp: int(np.shape(X[0])[1])
                                 for sp, X in ins.items()}
            rec['metric'] = metric_name
            rec['lower_is_better'] = bool(lower_is_better)
            # The held-out score of each configuration fitted on the clean
            # training labels in clean-label features; nothing divides by it.
            rec['held_out_score_clean_labels'] = {}
            for sp, (Xtr, Xva) in ins.items():
                try:
                    rec['held_out_score_clean_labels'][sp] = float(metric_fn(
                        y_val, _nc.cached_fit_predict(
                            dataset_name, m, factories[m], Xtr, y_tr_clean,
                            Xva, predict_mode=pred_mode)))
                except Exception as exc:
                    rec['held_out_score_clean_labels'][sp] = None
                    rec.setdefault('fit_errors', {})[sp] = (
                        f'{type(exc).__name__}: {exc}')
            frec['models'][m] = rec
            print(f"    {m}: base {rec.get('base')}, hybrids "
                  f"{[h.get('label') for h in rec.get('hybrids') or []]}, "
                  f"estimation-only hybrids "
                  f"{[sp for sp in NEIGHBOUR_HYBRID_SPACES if sp in (entry or {}).get('recipes', {})]}")
        fold_records[str(fold_i)] = frec

        def _unit(level, draw, _tr=train_idx, _va=val_idx, _ytr=y_tr_clean,
                  _yva=y_val, _g=g_tr, _ci=clean_inputs, _fold=fold_i,
                  _fac=factories):
            y_noisy, cmask, prov = inject_condition(
                data, dataset_name, noise_condition, level, draw,
                train_idx=_tr, groups=noise_groups)
            cmask = np.asarray(cmask, dtype=bool)
            val = {'provenance': prov, 'n_corrupted': int(cmask.sum()),
                   'clean': {}, 'noisy': {}, 'grid': {}, 'spaces': {},
                   'unit_names': {}, 'unavailable': {},
                   'extra_target_errors': {}, 'sim_detection': {},
                   'neighbour_grid_screen': {}}
            sweep_ctx['level'] = float(level)
            sweep_ctx['screen'] = []

            # The fitted configurations of this draw: base pair as built from
            # clean labels (no feature choice), hybrids rebuilt from the
            # damaged labels; plus the estimation-only hybrids, also from the
            # damaged labels.
            unit_pairs, unit_train, neighbour_train = {}, {}, {}
            base_by_model = {}
            for m, (entry, ins_c, unav_c, names_c) in _ci.items():
                base_by_model[m] = list((entry or {}).get('base') or [])
                hyb_n, unav_n, names_n = fold_model_inputs(
                    data, dataset_name, m, entry, train_idx=_tr, val_idx=_va,
                    y_train=y_noisy, factory=_fac[m], metric=metric,
                    spaces=HYBRID_CONFIGURATIONS + NEIGHBOUR_HYBRID_SPACES)
                for sp in CONFIGURATION_SPACES:
                    is_hyb = sp in HYBRID_CONFIGURATIONS
                    pair = hyb_n.get(sp) if is_hyb else ins_c.get(sp)
                    if pair is None:
                        val['unavailable'].setdefault(m, {})[sp] = (
                            (unav_n if is_hyb else unav_c).get(sp)
                            or 'not built')
                        continue
                    u = f'{m}@{sp}'
                    unit_pairs[u] = pair
                    unit_train[u] = (m, sp, pair[0])
                    val['unit_names'][u] = (names_n if is_hyb
                                            else names_c).get(sp)
                for sp in NEIGHBOUR_HYBRID_SPACES:
                    if sp in hyb_n:
                        neighbour_train.setdefault(m, {})[sp] = hyb_n[sp][0]
                    else:
                        val['unavailable'].setdefault(m, {})[sp] = (
                            unav_n.get(sp) or 'not built')
            unit_fac = {u: _fac[m] for u, (m, _sp, _X) in unit_train.items()}

            # Clean and damaged references per unit, in this draw's features.
            for u, (Xtr, Xva) in unit_pairs.items():
                m = unit_train[u][0]
                for key, ys in (('clean', _ytr), ('noisy', y_noisy)):
                    try:
                        val[key][u] = float(metric_fn(_yva, _nc.cached_fit_predict(
                            dataset_name, m, unit_fac[u], Xtr, ys, Xva,
                            predict_mode=pred_mode)))
                    except Exception as exc:
                        val[key][u] = None
                        val.setdefault('reference_errors', {})[
                            f'{u}|{key}'] = f'{type(exc).__name__}: {exc}'

            spaces, models_by_space, space_info = _build_estimation_spaces(
                unit_train, neighbour_train, base_by_model, reps, _tr,
                sorted({m for m, _sp, _X in unit_train.values()}))
            val['spaces'] = {sp: dict(space_info[sp],
                                      models=list(models_by_space[sp]),
                                      n_features=int(np.shape(X)[1]))
                             for sp, X in spaces.items()}
            # The two added smoothing targets are built in each model's
            # fitted configurations and estimation-only hybrids, not in the
            # per-representation spaces (USER DECISION 2026-09-17, below).
            extra_target_spaces = {sp for sp, inf in space_info.items()
                                   if inf['kind'] != 'representation_capped'}
            eval_by_unit = dict(unit_pairs)

            for space_name, X_space in spaces.items():
                _open = models_by_space.get(space_name) or []
                _space_units = {u: f for u, f in unit_fac.items()
                                if unit_train[u][0] in _open}
                if not _space_units:
                    continue
                grid_sp = val['grid'].setdefault(space_name, {})

                # The two smoothing targets that do not belong to any one
                # estimator, built once for this (fold, draw, space) and
                # handed to every estimator below. Both are fitted on this
                # fold's training molecules only and read off molecules the
                # fit did not see, and both take the fold's scaffold groups so
                # the inner split is group-disjoint.
                # USER DECISION 2026-09-17: not computed in every
                # representation. Building one costs five model fits, so they
                # are built in the configurations and hybrid spaces only.
                # ⛔ EXTRA_TARGETS is the list, and this block is what reads it.
                extra_targets = {}
                if not is_clf and space_name in extra_target_spaces:
                    if 'oof' in EXTRA_TARGETS:
                        try:
                            extra_targets['oof'] = _nc.cached_oof_predict(
                                dataset_name, X_space, y_noisy, _g,
                                n_splits=min(EXTRA_TARGET_SPLITS,
                                             len(y_noisy)))
                        except Exception as exc:
                            val['extra_target_errors'].setdefault(
                                f'{space_name}|oof',
                                f'{type(exc).__name__}: {exc}')
                    # The Gaussian-process target is the same O(n^3) fit
                    # KIRBY_NOISE_LIGHT_ESTIMATORS was set up to drop.
                    if 'gp' in EXTRA_TARGETS:
                        if light_estimators:
                            val['extra_target_errors'].setdefault(
                                f'{space_name}|gp',
                                'not computed: KIRBY_NOISE_LIGHT_ESTIMATORS=1')
                        else:
                            try:
                                extra_targets['gp'] = _gp_target(
                                    dataset_name, X_space, y_noisy, _g)
                            except Exception as exc:
                                val['extra_target_errors'].setdefault(
                                    f'{space_name}|gp',
                                    f'{type(exc).__name__}: {exc}')

                _gk = {'args': (None, None, _yva, _space_units, metric_fn,
                                n_samples),
                       'kw': dict(is_clf=is_clf, groups=_g,
                                  corrupted_mask=cmask,
                                  noise_condition=noise_condition,
                                  extra_targets=extra_targets,
                                  eval_by_model=eval_by_unit,
                                  score_metric_name=metric_name)}
                # The estimators that read no neighbours: every estimation
                # space, as before.
                for est_name in available_estimators:
                    if est_name in NEIGHBOR_ESTIMATORS:
                        continue
                    _merge_grid_result(
                        grid_sp.setdefault(est_name, {}),
                        _run_estimator_grid(est_name, None, X_space, y_noisy,
                                            *_gk['args'], **_gk['kw']))
                # The neighbour estimators, 2026-10-05 (chat 5): only in the
                # spaces KNN_SEARCH_SPACE_RECORD axis 1 lists for this
                # dataset's group, and screened there (_screened_neighbour_
                # grid: group distances x 5/10/20 neighbours x two counting
                # schemes scored with no fit, top 3 refitted). Until this
                # date they ran in every estimation space, the capped
                # per-representation spaces included, under every distance
                # at a fixed 10 neighbours with equal weight.
                _nb_ests = [e for e in available_estimators
                            if e in NEIGHBOR_ESTIMATORS]
                _axis = _neighbour_axis_space(space_name,
                                              space_info[space_name],
                                              group_sp)
                if _nb_ests and _axis is None:
                    val['neighbour_grid_screen'][space_name] = {
                        'axis_space': None,
                        'why': ('not a KNN_SEARCH_SPACE_RECORD axis 1 space '
                                f'for group {knn_grp}; neighbour estimators '
                                'not run here')}
                elif _nb_ests:
                    _res, _rec = _screened_neighbour_grid(
                        _nb_ests, _axis, X_space, y_noisy,
                        dataset_name=dataset_name, nbr_metrics=nbr_metrics,
                        metric=metric, is_clf=is_clf, grid_kwargs=_gk)
                    val['neighbour_grid_screen'][space_name] = _rec
                    for est_key, grid_result in _res.items():
                        _merge_grid_result(grid_sp.setdefault(est_key, {}),
                                           grid_result)

            # ── The similarity sweep for this (level, fold, draw). ──
            # One pass per evaluation unit. The extra smoothing targets are
            # the ones built for the unit's own space above (same features,
            # same labels, same groups), read back from noise_cache.
            for u, (m, sp_cfg, _Xtr_u) in unit_train.items():
                if not sim_estimators or not sink.enabled:
                    break
                entry = _ci[m][0] or {}
                _xtr, _xva = unit_pairs[u]
                _xt = {}
                if not is_clf and 'oof' in EXTRA_TARGETS:
                    try:
                        _xt['oof'] = _nc.cached_oof_predict(
                            dataset_name, _xtr, y_noisy, _g,
                            n_splits=min(EXTRA_TARGET_SPLITS, len(y_noisy)))
                    except Exception as exc:
                        val['extra_target_errors'].setdefault(
                            f'sweep:{u}|oof', f'{type(exc).__name__}: {exc}')
                if not is_clf and 'gp' in EXTRA_TARGETS \
                        and not light_estimators:
                    try:
                        _xt['gp'] = _gp_target(dataset_name, _xtr, y_noisy, _g)
                    except Exception as exc:
                        val['extra_target_errors'].setdefault(
                            f'sweep:{u}|gp', f'{type(exc).__name__}: {exc}')

                def _dprimes(setting, lists, k, _xtr=_xtr, _xt=_xt,
                             _y=y_noisy, _cm=cmask):
                    # Where a neighbour estimator moves a label: its
                    # neighbours' label under the kept counting scheme.
                    try:
                        _i, _d = _ss.neighbours_at(lists, k)
                        _nt = _ss.weighted_neighbour_label(
                            _y, _i, _d, lists.get('scheme'), is_clf=is_clf)
                    except Exception:
                        _nt = None
                    for est_name in sim_estimators:
                        got = []
                        res = _run_estimator_grid(
                            est_name, None, _xtr, _y, None, None,
                            _yva, {}, metric_fn, n_samples,
                            is_clf=is_clf, groups=_g,
                            corrupted_mask=_cm,
                            noise_condition=noise_condition,
                            extra_targets=_xt,
                            precomputed_neighbors=lists, k_override=k,
                            dprime_cb=lambda r, c: got.append((r, c)),
                            fit_models=False, score_metric_name=metric_name,
                            neighbour_target=_nt)
                        if '_error' in res:
                            yield est_name, 'estimator_failed', None
                            continue
                        if '_detection' in res:
                            (val['sim_detection'].setdefault(est_name, {})
                             .setdefault(setting.name, {})
                             .setdefault(f"{k}|{lists.get('scheme')}", [])
                             .append(res['_detection']))
                        for remed, cleaned in got:
                            yield est_name, remed, cleaned

                _similarity_sweep_own_model(
                    sweep_ctx, model_name=m, model_input=u, X_tr=_xtr,
                    X_va=_xva, y_tr_noisy=y_noisy, y_tr_clean=_ytr,
                    y_val=_yva, train_idx=_tr, val_idx=_va, groups_tr=_g,
                    fold=_fold, draw=draw,
                    recipes=entry.get('recipes') or {},
                    base_reps=entry.get('base') or [],
                    make_dprimes=_dprimes, level=float(level))
            val['screen'] = sweep_ctx.pop('screen', [])
            return val

        for level in levels:
            draws = draws_all[:1] if float(level) == 0.0 else draws_all
            for draw in draws:
                key = ['unit', int(fold_i), float(level), int(draw)]
                v = ckpt.get(key)
                if v is None:
                    print(f"    level {level:g}, draw {draw}")
                    t0 = time.time()
                    v = _unit(float(level), int(draw))
                    ckpt.put(key, v)
                    print(f"      [{time.time() - t0:.0f}s]")
                else:
                    print(f"    level {level:g}, draw {draw}: read back from "
                          f"{ckpt.path}")
                units_done.append({'fold': int(fold_i), 'level': float(level),
                                   'draw': int(draw), 'value': v})
                for u in (v.get('clean') or {}):
                    m_, _, sp_ = u.partition('@')
                    unit_meta.setdefault(u, {'model': m_,
                                             'configuration': sp_})

    results = {
        'experiment': 'n11_estimation_remediation_grid',
        'result_layout': RESULT_LAYOUT,
        'metric': metric_name,
        'lower_is_better': bool(lower_is_better),
        'task_type': task_type,
        'noise_condition': noise_condition,
        'sigmas': list(levels),
        'levels': list(levels),
        'level_unit': level_unit(noise_condition, is_clf),
        'draws_per_nonzero_level': len(draws_all),
        'level_zero': 'the labels as delivered, run once (one draw)',
        'knn_group': knn_grp,
        'n_folds': int(n_folds),
        'folds_run': [int(f) for f, _t, _v in folds],
        'selection_scope': 'fold_train_rows',
        'models': models,
        'configurations_per_model': list(CONFIGURATION_SPACES),
        'n_fitted_configurations': len(models) * len(CONFIGURATION_SPACES),
        'estimation_only_spaces_per_model': list(NEIGHBOUR_HYBRID_SPACES),
        'folds': fold_records,
        'evaluation_units': unit_meta,
        'evaluation_unit_key': (
            "'<model>@<configuration>': that decided model fitted on one of its "
            "4 configurations of the fold (base_1, base_2, hybrid, "
            "hybrid__general2). Which representation or recipe that is on each "
            "fold is in folds.<i>.models.<model>. Every *_by_model field and "
            "the model half of every remediation key is keyed by it."),
        'estimators': available_estimators,
        'neighbour_metrics': nbr_metrics,
        'extra_smoothing_targets': list(EXTRA_TARGETS),
        'extra_smoothing_targets_note': (
            "'oof' is what a LightGBM model trained on the other training "
            "molecules of this fold predicts for this one, on a "
            "scaffold-group-aware inner split. 'gp' is the posterior mean of a "
            "Gaussian process fitted the same way. Both are computed once per "
            "(fold, draw, space) and handed to every estimator, so two rows "
            "differing only in this prefix differ in where the labels moved to "
            "and in nothing else. A remediation key with no prefix is the "
            "estimator's own target. 'quantile_clip_<alpha>' is not a smoothing "
            "row at all: it appears only under an estimator that exposes both "
            "edges of a per-molecule label range (quantile_forest), leaves "
            "every label inside its range byte-identical, and pulls each label "
            "outside it alpha of the way onto the edge it crossed."),
        'extra_target_spaces': ('every fitted configuration and estimation-only '
                                'hybrid space; not the per-representation '
                                'spaces (user decision 2026-09-17)'),
        'skipped_estimators': {e: 'skipped_regression_estimator_on_clf'
                               for e in skipped_estimators},
        'alphas': ALPHAS,
        'weight_strengths': list(WEIGHT_STRENGTHS),
        'per_sigma': {},
        'recipe_note': (
            "Each fold's configurations and estimation-only hybrids come from "
            "searches on that fold's training rows with clean labels "
            "(noise_helpers.fold_configurations). The recipe is chosen without "
            "the damage and without the held-out rows; the features of each "
            "hybrid are re-chosen from the damaged labels at every draw."),
    }

    # ── Per level: aggregate the units. ─────────────────────────────────────
    provenance = {}
    for lv in levels:
        lv_units = [u for u in units_done if u['level'] == lv]
        if not lv_units:
            continue
        clean_by_fold, noisy_by_fold, denoised_by_fold = {}, {}, {}
        grid_acc, detection_rows, sim_det = {}, {}, {}
        space_records, unavailable, extra_errs = {}, {}, {}
        for ud in lv_units:
            v, f = ud['value'], ud['fold']
            provenance.setdefault(f'{lv:g}', {}).setdefault(
                str(ud['draw']), v.get('provenance'))
            for u, s in (v.get('clean') or {}).items():
                if s is not None and np.isfinite(s):
                    clean_by_fold.setdefault(u, {}).setdefault(f, []).append(s)
            for u, s in (v.get('noisy') or {}).items():
                if s is not None and np.isfinite(s):
                    noisy_by_fold.setdefault(u, {}).setdefault(f, []).append(s)
            for sp, by_est in (v.get('grid') or {}).items():
                for est_key, by_rem in by_est.items():
                    acc = grid_acc.setdefault(sp, {}).setdefault(est_key, {})
                    for rk, sc in by_rem.items():
                        if rk == '_detection':
                            acc.setdefault('_detection_per_pass', []).append(sc)
                            continue
                        if rk == '_failures':
                            acc.setdefault('_failures', {}).update(sc)
                            continue
                        if rk == '_errors':
                            acc.setdefault('_errors', []).extend(sc)
                            continue
                        if rk.startswith('_'):
                            acc[rk] = sc
                            continue
                        if sc is None or not np.isfinite(sc):
                            continue
                        acc.setdefault(rk, []).append(sc)
                        _u = rk.split('|', 1)[0]
                        (denoised_by_fold.setdefault(_u, {})
                         .setdefault(sp, {}).setdefault(est_key, {})
                         .setdefault(rk, {}).setdefault(f, []).append(sc))
            for sp, rec in (v.get('spaces') or {}).items():
                space_records.setdefault(sp, rec)
            if v.get('unavailable'):
                unavailable[f"fold{f}_draw{ud['draw']}"] = v['unavailable']
            for k_, e_ in (v.get('extra_target_errors') or {}).items():
                extra_errs.setdefault(k_, e_)
            for e_, by_s in (v.get('sim_detection') or {}).items():
                for sn, by_k in by_s.items():
                    for k_, rows in by_k.items():
                        (sim_det.setdefault(e_, {}).setdefault(sn, {})
                         .setdefault(k_, [])).extend(rows)

        # One clean and one damaged reference PER UNIT: recovery is a fraction
        # of that unit's own clean-to-damaged gap. A fold's number is the mean
        # over its draws; the reference is the mean over folds.
        clean_f = {u: {f: float(np.mean(v)) for f, v in byf.items()}
                   for u, byf in clean_by_fold.items()}
        clean_by_model = {u: float(np.mean(list(byf.values())))
                          for u, byf in clean_f.items() if byf}
        noisy_by_model = {u: float(np.mean([s for v in byf.values() for s in v]))
                          for u, byf in noisy_by_fold.items() if byf}
        _models_scored = [u for u in sorted(clean_by_model)
                          if u in noisy_by_model]
        if not _models_scored:
            results['per_sigma'][f'{lv:g}'] = {
                'error': 'no evaluation unit produced both a clean and a '
                         'damaged reference score at this level'}
            continue
        gap_by_model, gap_floor_by_model = {}, {}
        for _m in _models_scored:
            _c, _nz = clean_by_model[_m], noisy_by_model[_m]
            # Sign-normalised so a positive gap always means the injected
            # noise cost the model something (fixed 2026-09-12).
            gap_by_model[_m] = (_nz - _c) if lower_is_better else (_c - _nz)
            gap_floor_by_model[_m] = max(1e-10, GAP_MIN_FRAC * abs(_c))
        _ref_model = _models_scored[0]

        sigma_result = {
            'level': float(lv),
            'n_draws': len(draws_all[:1] if lv == 0.0 else draws_all),
            'clean_score': clean_by_model[_ref_model],
            'noisy_score': noisy_by_model[_ref_model],
            'gap': float(gap_by_model[_ref_model]),
            'reference_model': _ref_model,
            'models_scored': list(_models_scored),
            'clean_score_by_model': clean_by_model,
            'noisy_score_by_model': noisy_by_model,
            'gap_by_model': {k: float(v) for k, v in gap_by_model.items()},
            # Held-out score per fold and its spread over folds (fold_spread:
            # no significance test; one fold says so).
            'clean_score_by_fold': {u: fold_spread([byf[f] for f in sorted(byf)])
                                    for u, byf in clean_f.items()},
            'noisy_score_by_fold': {
                u: fold_spread([float(np.mean(byf[f])) for f in sorted(byf)])
                for u, byf in noisy_by_fold.items()},
            'spaces': space_records,
            'configurations_unavailable': unavailable,
            'extra_target_errors': extra_errs,
            'per_space': {},
        }

        for space_name in sorted(grid_acc.keys()):
            space_agg = {}
            for est_key in sorted(grid_acc[space_name].keys()):
                est_agg = {}
                for remed_key, score_list in grid_acc[space_name][est_key].items():
                    if remed_key == '_detection_per_pass':
                        est_agg['detection'] = _mean_detection(score_list)
                        detection_rows.setdefault(est_key, []).extend(score_list)
                        continue
                    if remed_key.startswith('_'):
                        est_agg[remed_key] = score_list
                        continue
                    if not score_list:
                        continue
                    denoised = float(np.mean(score_list))
                    _mn = remed_key.split('|', 1)[0]
                    _c = clean_by_model.get(_mn)
                    _nz = noisy_by_model.get(_mn)
                    if (_c is not None and _nz is not None
                            and abs(gap_by_model[_mn]) > gap_floor_by_model[_mn]):
                        recovery = float((denoised - _nz) / (_c - _nz) * 100)
                    else:
                        recovery = float('nan')
                    _meta = unit_meta.get(_mn, {})
                    est_agg[remed_key] = {
                        'denoised_score': denoised,
                        'lower_is_better': bool(lower_is_better),
                        'recovery': recovery,
                        # 'model' is the evaluation unit ('svm@base_1'); the
                        # two fields after it split it.
                        'model': _mn,
                        'downstream_model': _meta.get('model'),
                        'fitted_on': _meta.get('configuration'),
                        'remediation': remed_key.split('|', 1)[-1],
                    }
                if est_agg:
                    space_agg[est_key] = est_agg
            sigma_result['per_space'][space_name] = space_agg
        sigma_result['detection_by_estimator'] = {
            k: _mean_detection(r) for k, r in detection_rows.items()}
        sigma_result['detection_by_similarity'] = {
            e_: {sn: {k_: _mean_detection(r) for k_, r in by_k.items()}
                 for sn, by_k in by_s.items()}
            for e_, by_s in sim_det.items()}

        # Recovery for a config picked without the fold it is read on. Pure
        # arithmetic over the per-fold scores just collected.
        sigma_result['heldout_recovery_by_model'] = {
            _m: _heldout_recovery(clean_f.get(_m, {}),
                                  noisy_by_fold.get(_m, {}),
                                  denoised_by_fold.get(_m, {}), GAP_MIN_FRAC)
            for _m in _models_scored}
        sigma_result['heldout_recovery'] = (
            sigma_result['heldout_recovery_by_model'][_ref_model])
        results['per_sigma'][f'{lv:g}'] = convert_numpy(sigma_result)

        # Print compact summary
        n_total = n_positive = 0
        best_recovery, best_config = float('-inf'), None
        for space_name, space_data in sigma_result['per_space'].items():
            for est_key, est_data in space_data.items():
                for rk, rv in est_data.items():
                    if rk.startswith('_') or not isinstance(rv, dict):
                        continue
                    n_total += 1
                    r = rv.get('recovery', float('nan'))
                    if r is None or np.isnan(r):
                        continue
                    if r > 0:
                        n_positive += 1
                    if r > best_recovery:
                        best_recovery = r
                        best_config = f'{space_name}/{est_key}/{rk}'
        print(f"\n  === Level {lv:g} ===")
        print(f"    {_ref_model}: clean={sigma_result['clean_score']:.4f}, "
              f"damaged={sigma_result['noisy_score']:.4f}")
        print(f"    Configs: {n_total} total, {n_positive} positive "
              f"({100 * n_positive / max(n_total, 1):.1f}%)")
        if best_config:
            print(f"    Best over all configs on their own folds: "
                  f"{best_config} ({best_recovery:+.1f}%)")

    # Robustness over the levels, per evaluation unit: the damaged score minus
    # the level-0 score in the dataset's own metric and direction, always;
    # auc_norm on full_knn datasets only (fast_knn has 2 levels and auc_norm
    # needs 4, absent there by Adelaide's decision of 2026-10-05).
    robustness = {}
    lvl_keys = [f'{lv:g}' for lv in levels]
    if len(levels) > 1 and 0.0 in levels:
        _units = sorted({u for k in lvl_keys
                         for u in (results['per_sigma'].get(k, {})
                                   .get('noisy_score_by_model') or {})})
        for u in _units:
            curve = [(results['per_sigma'].get(k, {})
                      .get('noisy_score_by_model') or {}).get(u)
                     for k in lvl_keys]
            if curve[0] is None:
                continue
            row = {'levels': list(levels), 'scores': curve,
                   'damaged_minus_level0': [
                       None if c is None else float(c - curve[0])
                       for c in curve]}
            if knn_grp == 'full_knn':
                ok = [(x, c) for x, c in zip(levels, curve) if c is not None]
                rm = robustness_metrics([x for x, _ in ok], [c for _, c in ok],
                                        baseline_threshold=None,
                                        lower_is_better=lower_is_better)
                row['auc_norm'] = rm.get('auc_norm')
            robustness[u] = row
    results['robustness'] = robustness
    results['robustness_reported_as'] = (
        'auc_norm and the plain difference against level 0'
        if knn_grp == 'full_knn' else
        'the plain difference against level 0 only; auc_norm is absent on '
        'fast_knn datasets by decision (2 levels)')
    results['noise_provenance'] = provenance

    results['similarity'] = similarity_record(
        sim_settings, sink, estimators_swept=sim_estimators,
        schemes_swept=list(sweep_schemes()), screen_top=screen_top(),
        screen_scored_on=SCREEN_SCORED_ON, knn_group=knn_grp,
        spaces_unavailable=sweep_ctx['unavailable'],
        label_target_of_neighbour_estimators=(
            "the neighbours' label under the kept counting scheme "
            "(similarity_space.weighted_neighbour_label)"))
    results['screen'] = [dict(e, fold=u['fold'], level=u['level'])
                         for u in units_done
                         for e in (u['value'].get('screen') or [])]
    # The estimator grid's own neighbour screen, 2026-10-05 (chat 5): per
    # (fold, level, draw, estimation space), which axis 1 space it stands for,
    # the kept (distance, k, scheme) and every scored combination. Grid keys
    # of the neighbour estimators are '<estimator>_<distance>|k=<k>|<scheme>',
    # so a key holds only the draws in which that combination was kept.
    results['neighbour_grid_screen'] = [
        dict(rec, space=sp, fold=u['fold'], level=u['level'], draw=u['draw'])
        for u in units_done
        for sp, rec in (u['value'].get('neighbour_grid_screen') or {}).items()]

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
        # comparison. 2026-10-05 (chat 5): each hybrid recipe is chosen per
        # fold by a search on that fold's training rows with the CLEAN
        # training labels (noise_helpers.fold_configurations), and its features
        # are re-chosen from the damaged labels at every draw. The recipe
        # never sees the held-out rows; it does see the clean training labels,
        # which a single representation has no use for. The field keeps its
        # name for readers written before this. Until 2026-10-05 the recipe
        # was read from a whole-dataset search.
        # Every space's recovery for one unit divides by that unit's own
        # clean-minus-damaged gap.
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
                'Every hybrid recipe here was chosen per fold by a search on '
                'that fold\'s training rows with the clean training labels '
                '(the general and distance searches against a model\'s '
                'inner-fold score, the similarity search against the '
                'correlation between each molecule\'s activity and what its '
                'five nearest neighbours predict); its features were re-chosen '
                'from the damaged labels at every draw. Which recipe each '
                'space is on each fold is in results["folds"].'
            ),
        }

        # Robust configs (positive at BOTH sigma levels if we have multiple)
        # This is computed after all sigmas in the final summary merge
        summary[f'sigma_{sigma_key}'] = convert_numpy(s)

    # Cross-sigma robust configs
    # Level 0 (the labels as delivered) has no gap, so it is left out. The
    # keys are f'{level:g}' since 2026-10-05 ('0', not '0.0').
    sigma_keys = [k for k in results['per_sigma'] if float(k) != 0.0]
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
