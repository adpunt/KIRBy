#!/usr/bin/env python3
"""N19: what a measurement count and a measurement spread are worth.

Paper: Noise.

THE QUESTION. A molecule measured six times carries a better label than one
measured once. Nothing in this suite has ever used that, because until
2026-09-16 no loader kept it: every hERG loader averaged a molecule's repeat
measurements together and wrote one number (NOISE_AUDIT_2026_09_10.md 11.1).
Two datasets keep them now -- `herg_ic50`, one row per ChEMBL record, and
`bindingdb_kd`, one row per measured (drug, protein) pair.

NOTHING IS INJECTED. The disagreement is what different laboratories got when
they measured the same thing. This is n17's regime, and n17's split is reused
rather than rebuilt.

HOW A MOLECULE IS GRADED. One row per molecule. A training molecule contributes
a label built from its own measurements. A held-out molecule is graded against
the mean of the measurements not used for training. A molecule is never on both
sides, so nothing a molecule trained on is also what it is graded against.
`n17_replicate_removal.build_entities` builds all of that and is imported here.

TWO WAYS OF BUILDING A TRAINING LABEL, AND THE DIFFERENCE IS THE POINT.

  single    ONE measurement, chosen at random. What a label looks like when it
            came from one laboratory. Its error variance is that molecule's own
            within-molecule variance, and it does NOT fall with the measurement
            count -- one measurement of a molecule measured six times is no more
            precise than one measurement of a molecule measured twice.
  averaged  the mean of ALL that molecule's measurements. What a loader normally
            hands you. Its error variance is the within-molecule variance
            DIVIDED BY the measurement count, so the count is now a precision.

A training molecule and a held-out molecule are different molecules, so a
training molecule using every one of its own measurements leaks nothing.

That split is what makes the weights below testable rather than assumed. Under
`single` the count is not a precision and weighting by it should do nothing.
Under `averaged` it is, and weighting by it is the inverse-variance weight that
meta-analysis has used since Hedges & Olkin 1985. Both run under both, so which
weight wins is measured rather than argued from the algebra.

SIX WAYS OF CHOOSING WHAT TO TRAIN ON. All six are graded on the same held-out
molecules against the same consensus labels, in the same feature space, with the
same folds. Three set a per-molecule fit weight; two change the training label;
one does nothing.

  untouched            every training molecule at its label, unweighted. The
                       thing to beat.
  count_weighted       weight proportional to how many times the molecule was
                       measured.
  spread_weighted      weight proportional to one over the variance of that
                       molecule's own measurements.
  precision_weighted   weight proportional to the count divided by that
                       variance -- the inverse-variance weight for an averaged
                       label.
  shrunk               the training label moves toward a group-aware
                       out-of-fold prediction, and moves further the less
                       precise it is. Each molecule's own measurement variance
                       sets how far.
  calibrated           the same shrinkage, except the variance is not each
                       molecule's own. It is fitted against the measurement
                       count on the molecules measured three or more times, and
                       then applied to every molecule including the ones
                       measured once, which have no variance of their own.

`shrunk` and `calibrated` differ in one thing: whether a molecule's shrinkage
uses its own spread, which is a noisy number at two or three measurements, or a
spread read off a curve fitted on the molecules that have enough measurements to
estimate one. That second one is the trusted-subset idea -- learn the noise
function where it can be learned, apply it where it cannot.

WHAT IS AND IS NOT AN ORACLE. A molecule's own measurement spread is information
a practitioner genuinely holds: they have the measurements. It is not the
held-out answer, and no held-out molecule's label enters any weight or any
shrinkage anywhere in this file. It IS the same quantity n17's `ceiling` ranks
on, and n17 calls that an oracle -- but n17 uses it to REMOVE molecules and
calls it an oracle because no detector can beat it, not because it leaks. Here
it is used as a weight on the molecule's own row.

FOUR CONDITIONS AND FOUR MODEL ARMS ADDED 2026-09-24. They come from an outside
research report on why label cleaning fails at this dataset's accuracy. Its
suggestions B1, B2 and C3 are these. Nothing in the six conditions above moved.

  invvar_weighted           weight 1 / (sigma_u^2 + sigma_e^2 / n), n the
                            number of measurements behind the training label.
                            sigma_e^2 is the variance of one measurement and
                            sigma_u^2 the error left once the label is exact.
                            Both are read off one regression, fitted inside
                            each outer fold's training molecules: squared
                            out-of-fold error against 1/n. The slope is
                            sigma_e^2, the intercept sigma_u^2.
  count_feature             no weight. The molecule's measurement count, as
                            log(count) standardised on the training molecules,
                            is added as one extra input feature.
  invvar_and_count_feature  both of the above at once.
  spread_pooled_weighted    weight 1 / (sigma_u^2 + s^2 / n), s^2 the
                            molecule's own measurement variance pulled toward
                            the training molecules' pooled variance.
                            POOLED_SPREAD_PRIOR sets how many measurements'
                            worth the pooled value counts as. Two or three
                            measurements give a variance too rough to divide
                            by on its own.

`count_weighted` and `precision_weighted` weight by n and by n/s^2. They carry
no sigma_u^2 term. So they treat a label resting on six measurements as six
times as good. That holds only when measurement noise is all of the error.
`invvar_weighted` differs from them in that one term.

⛔ THE COUNT IS A FEATURE A NEW MOLECULE DOES NOT HAVE. A held-out molecule's
count is known here because it was measured. A molecule nobody has measured yet
has no count. Every held-out molecule graded here was measured
MIN_MEASUREMENTS or more times, so at prediction time the feature only takes
values of 3 and up. `count_feature_note` on the result file says so.

THE MODEL ARMS. The report's B2 asks for learners that carry a noise term per
molecule. Two are fitted beside the dataset's two models, each paired only
against itself without the change:

  ngboost_plain / ngboost_count_feature
      NGBoost, from the `ngboost` entry of DEFAULT_MODELS
      (find_best_hybrid.py:2813), without and with the count feature. NGBoost
      predicts a spread per molecule, so with the count in its input the spread
      can depend on it.
  gp_plain / gp_per_molecule_noise
      the `gp` entry's Gaussian process (find_best_hybrid.py:2531). The second
      adds sigma_e^2 / n to each training molecule's diagonal through sklearn's
      `alpha` array. Both fit the same GP_MAX_TRAIN training molecules, drawn
      per fold, because the kernel matrix is n x n and `gp` refuses more than
      1,000. sklearn standardises the labels under normalize_y=True and adds
      `alpha` without rescaling it (sklearn/gaussian_process/_gpr.py:269-274,
      344). So the variance is divided by the training labels' variance before
      it is passed in.

  xgboost_aft_point / xgboost_interval_aft
      Added 2026-09-24 from a second research report (its section J): train on
      the RANGE of a molecule's measurements instead of one number. Both are
      XGBoost with objective survival:aft and a normal error, with n30's
      settings and n30's label map (time = exp(label - centre), centre the
      median label, predictions mapped back with log; n30_censored_likelihood.
      fit_aft_predict). The error scale is the SD of the training labels minus
      the fold's out-of-fold LightGBM prediction, so the arm is absent where
      that prediction failed.
        xgboost_aft_point     every label is a zero-width interval at the
                              training label. The plain version of the same
                              learner: it differs from xgboost_interval_aft in
                              the label bounds and nothing else. It is the arm
                              the interval arm is paired against, rather than
                              the downstream xgboost, whose loss and starting
                              value are not the AFT's.
        xgboost_interval_aft  the interval [min, max] of the measurements the
                              training label rests on. Under `averaged` that is
                              all of a molecule's measurements. Under `single` a
                              gradeable molecule's label rests on ONE
                              measurement, so its interval has zero width; the
                              others rest on all of theirs. A molecule measured
                              once has zero width either way. In a `single`
                              fold where EVERY training molecule has zero
                              width the arm would repeat xgboost_aft_point, so
                              it is not fitted there; `model_arm_skips` holds
                              the reason and the measured count.

WHERE THE FEATURES COME FROM (rewritten 2026-10-05, chat 5 of the alignment,
HANDOFF_EXPERIMENT_ALIGNMENT.md 2.4). Per outer fold and per model -- the
dataset's two decided models (DECISIONS.md Step 1) -- four feature spaces:
`base_1` and `base_2`, its two best single representations, and `hybrid` and
`hybrid__general2`, its two best hybrids. All four are ranked by one search on
that fold's training molecules (_shared.fold_standard_selection, the decided
grid, pruning off). Two models x 4 = 8 fitted configurations. Until
2026-10-05 they came from the whole-dataset stored search, which had seen the
held-out molecules; the similarity hybrid ('hybrid_similarity') was a space
too, and it is gone because n19 finds no neighbours (handoff 2.3 item 1).

THE SEARCH RUNS ON MOLECULES, NOT MEASUREMENTS. The loader keeps one row per
measurement; n19 trains and grades on one row per molecule. The outer folds
(noise_helpers.noise_outer_folds) and the per-fold searches are on the
molecule rows, filed under SEARCH_CONDITION_SUFFIX, so all of a molecule's
measurements sit on one side of every split (handoff 2.3 item 14). The search
sees each training molecule's mean over all its measurements.

Each hybrid is rebuilt on the fold's training molecules with the label build
being scored, through noise_helpers.fold_model_inputs, with the fold search's
own representation scores. The model arms are fitted in the first decided
model's four configurations. Which representations and recipes each fold
used is under `per_fold_configurations`.

The hybrid is built ONCE per (fold, draw, label construction), on the UNTOUCHED
training labels, and all conditions share it. So the only thing differing
between two conditions is the weight or the label handed to the downstream fit.
Whether a recipe searched on the shrunk labels would do better is a different
question and it is n20's.

`sample_weight` appears zero times in `src/kirby/hybrid.py` and zero times in
`src/kirby/find_best_hybrid.py`. So the three weighted conditions weight the
downstream fit and NOT the feature choice, in the hybrid space and in the base
space alike. `weights_reach_feature_selection` is written False on every result
row, because a reader who sees a hybrid beaten by a weight would otherwise
assume the weight had been in the budget split.

NO ORACLE CAVEAT TO CARRY. n11, n12 and n13 record that their recipe was chosen
on clean labels and their experiment then corrupts them. Here nothing is
corrupted and there is no clean column. `recipe_saw_clean_labels` is written
False for that reason.

Usage:
    python n19_measurement_count.py --dataset herg_ic50 --mode full
    python n19_measurement_count.py --dataset bindingdb_kd --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_domain_metric, get_model_factories, as_rep_dtype,
    # Per outer fold, the 2 best base representations and 2 best hybrids of
    # each decided model, from a search on that fold's training rows
    # (handoff 2.4 step 5; chat 5, 2026-10-05).
    fold_standard_selection,
)
from experiments.decided_grid import decided_models
from experiments.noise._ground_truth_providers import build_replicate_groups
from experiments.noise.n17_replicate_removal import (
    build_entities, MIN_MEASUREMENTS,
)
# Chat 5 (2026-10-05): outer folds, the four fitted configurations of one
# model on one fold, and the result-file record of them.
from experiments.noise.noise_helpers import (
    noise_outer_folds, fold_spread, fold_model_inputs, configuration_record,
    CONFIGURATION_SPACES,
)
# The D-versus-D' rows, the controls, the caches and the per-fold checkpoint,
# added 2026-09-25 (plan sections 2-3). n19 finds no neighbours, so it takes no
# similarity setting (plan section 6: "Experiments with no similarity (n19,
# ...) get the output row and the caches only").
from experiments.noise import noise_cache as nc
from experiments.noise import _sim_axis_n15_n22 as sx

# The name the rows file, the checkpoint and the result file are filed under.
EXP_FULL_NAME = 'n19_measurement_count'

# The noise_condition column of every row: nothing injected, the laboratories
# disagree.
ROW_CONDITION = 'real_replicate_disagreement'

# The learner each model arm pair is, for the `model` column of its rows.
MODEL_ARM_LEARNER = {'ngboost_count_feature': 'ngboost',
                     'gp_per_molecule_noise': 'gp',
                     'xgboost_interval_aft': 'xgboost_aft'}

# How a training molecule's label is built from its own measurements.
LABEL_BUILDS = ('single', 'averaged')

# What is done to the training molecules before the downstream fit.
CONDITIONS = ('untouched', 'count_weighted', 'spread_weighted',
              'precision_weighted', 'shrunk', 'calibrated',
              # Added 2026-09-24. See the header's section on them.
              'invvar_weighted', 'count_feature', 'invvar_and_count_feature',
              'spread_pooled_weighted')

# The three that set a per-molecule fit weight. A model whose fit() refuses a
# weight is absent from these three rather than fitted unweighted and recorded
# as if it had been weighted -- the mistake n13 names at :155-162.
WEIGHTED_CONDITIONS = ('count_weighted', 'spread_weighted',
                       'precision_weighted')

# The two that change the training label instead.
RELABEL_CONDITIONS = ('shrunk', 'calibrated')

# The three whose weight needs sigma_u^2 and sigma_e^2 from the regression of
# squared out-of-fold error on 1/count. That regression runs per (fold, space)
# on the training molecules, so these are absent from a fold where it failed.
FITTED_VARIANCE_CONDITIONS = ('invvar_weighted', 'invvar_and_count_feature',
                              'spread_pooled_weighted')

# The two that add the measurement count as an input feature.
COUNT_FEATURE_CONDITIONS = ('count_feature', 'invvar_and_count_feature')

# How many measurements' worth the pooled variance counts as when a molecule's
# own variance is pulled toward it. One value, not swept. At 3, a molecule
# measured twice (1 degree of freedom) sits three quarters of the way to the
# pooled value; one measured ten times (9) sits a quarter of the way.
POOLED_SPREAD_PRIOR = 3

# Learners fitted beside the downstream models. Each is paired against its own
# plain arm, never against a downstream model.
MODEL_ARMS = ('ngboost_plain', 'ngboost_count_feature',
              'gp_plain', 'gp_per_molecule_noise',
              'xgboost_aft_point', 'xgboost_interval_aft')
MODEL_ARM_PAIRS = {'ngboost_count_feature': 'ngboost_plain',
                   'gp_per_molecule_noise': 'gp_plain',
                   'xgboost_interval_aft': 'xgboost_aft_point'}

# The `gp` entry refuses more than 1,000 training molecules
# (find_best_hybrid.py:2781). Both GP arms fit the same draw of this many.
GP_MAX_TRAIN = 1000

# Which feature space each condition is measured in. All of them, always.
#
# 2026-10-05 (chat 5): the four fitted configurations of each decided model on
# each outer fold -- its rank-1 and rank-2 base representation and its two
# best hybrids, all ranked by a search on that fold's training molecules. Two
# models x 4 = 8 (R5). 'hybrid_similarity' is gone: n19 finds no neighbours,
# so the similarity hybrid is not one of its spaces (handoff 2.3 item 1).
BASE_SPACES = ('base_1', 'base_2')
SPACES = tuple(CONFIGURATION_SPACES)

# The per-fold searches run on one row per molecule (the rows n19 trains and
# grades on), not on the loader's one row per measurement, so they are filed
# apart from the dataset's own per-fold searches (handoff 2.3 item 14, dirty-run
# bug 11).
SEARCH_CONDITION_SUFFIX = 'n19_one_row_per_molecule_mean_label'

# The rows file and the checkpoint carry this so nothing written under the
# whole-dataset selection is read back (handoff 2.4 item 11).
SELECTION_VERSION = 'per_fold_2026_10_05'

# How many times WHICH measurement becomes the `single` training label, and
# which measurements form each held-out molecule's consensus, are re-drawn.
# Three, matching n07, n09, n11, n12, n13, n15 and n17.
N_DRAWS = 3

# Folds for the out-of-fold prediction the two relabelling conditions shrink
# toward. It runs inside the training molecules of an outer fold only. An
# inner fold count, not the outer one (handoff 2.4 item 14 leaves these).
OOF_SPLITS = 5

# A weight is a ratio, so it needs a floor: a molecule whose measurements all
# agreed has a variance of zero and an infinite weight. The floor is a share of
# the median defined variance, so it scales with the dataset rather than being
# a number in the label's units.
VARIANCE_FLOOR_SHARE = 0.01


def measurement_stats(labels, replicate_groups):
    """Per-molecule measurement count, mean and variance, in entity order.

    Entity order is `build_entities`' order -- molecules sorted by the row their
    first measurement sits on -- and the caller checks that the two agree rather
    than trusting that they do.

    Returns a dict of arrays:
      row_index       the first row of this molecule, to read features from
      n_measurements  how many times it was measured
      mean_all        the mean of ALL its measurements
      var_all         the variance of its measurements, ddof=1, NaN below two
      min_all         the lowest of its measurements (added 2026-09-24, for the
                      interval arm)
      max_all         the highest

    ddof=1, for the reason n17 records at :176-184: with ddof=0 the measured
    spread grows with how many times a molecule was measured whatever its real
    disagreement, so a quiet molecule measured six times outranks a noisy one
    measured twice. Everything downstream of this ranks or divides by it.

    Reviewed 2026-09-18 alongside n17's identical floor-of-2 question: no
    change requested here. `var_all` at k=2 feeds `weights_for`'s sample weights and
    `shrink_toward`'s label shrinkage directly, so it moves what gets fit, not
    just what gets reported -- that was the point of checking it, and the
    decision was to leave it as-is.
    """
    labels = np.asarray(labels, dtype=np.float64)
    groups = np.asarray(replicate_groups)

    order = {}
    for i, g in enumerate(groups):
        order.setdefault(g, []).append(i)
    gids = sorted(order, key=lambda g: order[g][0])

    n = len(gids)
    out = {
        'row_index': np.zeros(n, dtype=np.int64),
        'n_measurements': np.zeros(n, dtype=np.int64),
        'mean_all': np.zeros(n, dtype=np.float64),
        'var_all': np.full(n, np.nan, dtype=np.float64),
        'min_all': np.zeros(n, dtype=np.float64),
        'max_all': np.zeros(n, dtype=np.float64),
    }
    for j, g in enumerate(gids):
        rows = np.asarray(order[g], dtype=np.int64)
        vals = labels[rows]
        out['row_index'][j] = rows[0]
        out['n_measurements'][j] = len(rows)
        out['mean_all'][j] = float(np.mean(vals))
        out['min_all'][j] = float(np.min(vals))
        out['max_all'][j] = float(np.max(vals))
        if len(rows) >= 2:
            out['var_all'][j] = float(np.var(vals, ddof=1))
    return out


def training_labels(ent, stats, build):
    """The training label for every molecule, and how many measurements it rests on.

    `single`   one measurement, chosen at random -- what `build_entities` put in
               `train_label`. It rests on ONE measurement whatever the molecule's
               count, which is exactly why the count is not a precision here.
    `averaged` the mean of all of this molecule's measurements. It rests on all
               of them.

    A molecule measured once gets the same number either way.

    Returns (y_train, n_behind_label).
    """
    if build == 'single':
        # `build_entities` puts ONE measurement in `train_label` only for the
        # molecules it can grade. A molecule measured once or twice cannot be
        # graded, so it gets the mean of its measurements instead -- n17's
        # choice, at :185-192, and the right one: there is nothing to hold out
        # for it and the mean is the best single number its rows support.
        #
        # So the count behind a `single` label is 1 for a gradeable molecule and
        # its whole measurement count for the rest. Writing 1 everywhere would
        # say a molecule measured twice carries twice the error variance it
        # does, and underweight it by that much.
        n_behind = np.where(ent['gradeable'], 1.0,
                            stats['n_measurements'].astype(np.float64))
        return ent['train_label'].copy(), n_behind
    if build == 'averaged':
        return (stats['mean_all'].copy(),
                stats['n_measurements'].astype(np.float64))
    raise ValueError(f"unknown label build {build!r}")


def interval_bounds(ent, stats, build, y_train):
    """Lower and upper label bound per molecule, for xgboost_interval_aft.

    The range of the measurements the training label rests on (see
    `training_labels`): all of them under `averaged`; under `single`, one for a
    gradeable molecule (zero width, at its label) and all of them otherwise.
    Every interval contains the training label.
    """
    y_train = np.asarray(y_train, dtype=np.float64)
    lo = np.asarray(stats['min_all'], dtype=np.float64).copy()
    hi = np.asarray(stats['max_all'], dtype=np.float64).copy()
    if build == 'single':
        one = np.asarray(ent['gradeable'], dtype=bool)
        lo[one] = y_train[one]
        hi[one] = y_train[one]
    elif build != 'averaged':
        raise ValueError(f"unknown label build {build!r}")
    return lo, hi


def fit_interval_aft_predict(X_tr, lower, upper, X_va, sd):
    """XGBoost survival:aft on [lower, upper] labels; predictions in label units.

    n30's map and settings (n30_censored_likelihood.fit_aft_predict,
    AFT_PARAMS, AFT_ROUNDS): time = exp(label - centre), centre the median of
    the interval midpoints, prediction = log(predicted time) + centre.
    """
    import xgboost as xgb
    from experiments.noise.n30_censored_likelihood import AFT_PARAMS, AFT_ROUNDS
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    centre = float(np.median((lower + upper) / 2.0))
    # Features stay float32 (handoff 2.3 item 18); the bounds are labels.
    dtr = xgb.DMatrix(as_rep_dtype(X_tr))
    dtr.set_float_info('label_lower_bound', np.exp(lower - centre))
    dtr.set_float_info('label_upper_bound', np.exp(upper - centre))
    p = dict(AFT_PARAMS)
    p.update({'objective': 'survival:aft',
              'aft_loss_distribution': 'normal',
              'aft_loss_distribution_scale': float(max(sd, 1e-3))})
    booster = xgb.train(p, dtr, int(AFT_ROUNDS))
    t = booster.predict(xgb.DMatrix(as_rep_dtype(X_va)))
    return np.log(np.clip(t, 1e-300, None)) + centre


def label_variance(stats, n_behind, floor_share=VARIANCE_FLOOR_SHARE):
    """The error variance of each molecule's TRAINING label, and where it came from.

    A molecule's measurements disagree by `var_all`. A label resting on
    `n_behind` of them has an error variance of `var_all / n_behind` -- one
    measurement carries the whole of it, a mean of six carries a sixth.

    A molecule measured once has no variance of its own. It is given the median
    of the ones that do, rather than zero (which is an infinite weight) or
    dropped (which changes how many molecules train, for a reason that has
    nothing to do with the question). How many were filled that way is returned
    and written to the result file.

    Returns (variance, n_filled, floor).
    """
    var = np.asarray(stats['var_all'], dtype=np.float64).copy()
    known = np.isfinite(var)
    if not known.any():
        raise RuntimeError(
            "no molecule in this dataset was measured more than once, so no "
            "measurement variance exists and n19 has nothing to weight by.")
    median_var = float(np.median(var[known]))
    floor = max(floor_share * median_var, 1e-12)
    var[~known] = median_var
    var = np.maximum(var, floor)
    return var / np.maximum(n_behind, 1.0), int((~known).sum()), float(floor)


def calibrated_variance(stats, n_behind, trusted_min=MIN_MEASUREMENTS,
                        floor_share=VARIANCE_FLOOR_SHARE):
    """The error variance read off a curve fitted on the well-measured molecules.

    The trusted subset is the molecules measured `trusted_min` or more times --
    the only ones whose own variance is worth much. Within it, the mean variance
    is taken at each distinct measurement count. A molecule is then given the
    fitted variance for its own count, and a count the trusted subset never saw
    takes the nearest count it did see.

    This is the difference between this and `shrunk`: `shrunk` uses a molecule's
    own variance, which at two measurements is one number from two points.
    This uses a variance estimated across every molecule measured that many
    times.

    Returns (variance, fit, n_trusted). `fit` is {count: variance} and is
    written to the result file, because it IS the noise function and a reader
    should be able to see whether it falls with the count or is flat.
    """
    var = np.asarray(stats['var_all'], dtype=np.float64)
    k = np.asarray(stats['n_measurements'], dtype=np.int64)
    trusted = np.isfinite(var) & (k >= int(trusted_min))
    if not trusted.any():
        raise RuntimeError(
            f"no molecule was measured {trusted_min} or more times, so there "
            f"is no trusted subset to fit a noise function on.")

    fit = {int(c): float(np.mean(var[trusted & (k == c)]))
           for c in np.unique(k[trusted])}
    counts = np.array(sorted(fit), dtype=np.int64)
    values = np.array([fit[int(c)] for c in counts], dtype=np.float64)

    median_var = float(np.median(var[np.isfinite(var)]))
    floor = max(floor_share * median_var, 1e-12)
    # Nearest count the trusted subset actually saw. A molecule measured once
    # takes the smallest fitted count, which is the most pessimistic of them.
    nearest = np.abs(k[:, None] - counts[None, :]).argmin(axis=1)
    fitted = np.maximum(values[nearest], floor)
    return (fitted / np.maximum(n_behind, 1.0),
            {int(c): float(v) for c, v in zip(counts, values)},
            int(trusted.sum()))


def weights_for(condition, n_measurements, variance, n_behind):
    """The per-molecule fit weight, normalised to average one.

    Average one rather than sum one, so a weighted fit and an unweighted fit see
    the same total weight and the difference between them is the SHAPE of the
    weighting and not how much of it there is.

    count_weighted      how many times the molecule was measured
    spread_weighted     one over the variance of its own measurements
    precision_weighted  the count divided by that variance -- the
                        inverse-variance weight for a label that is the mean of
                        that many measurements
    """
    k = np.asarray(n_measurements, dtype=np.float64)
    # `variance` is already var_all / n_behind. The per-measurement variance is
    # what the two spread weights divide by, so multiply it back out.
    per_measurement = np.asarray(variance, dtype=np.float64) * np.maximum(
        n_behind, 1.0)
    if condition == 'count_weighted':
        w = k
    elif condition == 'spread_weighted':
        w = 1.0 / per_measurement
    elif condition == 'precision_weighted':
        w = k / per_measurement
    else:
        raise ValueError(f"{condition!r} does not set a weight")
    mean_w = float(np.mean(w))
    if not np.isfinite(mean_w) or mean_w <= 0:
        raise RuntimeError(f"{condition}: the weights average {mean_w}, which "
                           f"cannot be normalised.")
    return w / mean_w


def shrink_toward(y, oof, variance):
    """Move each training label toward the out-of-fold prediction.

    The amount is `variance / (variance + tau2)`, where tau2 is how far the
    labels sit from the out-of-fold prediction overall. A molecule whose label
    is precise barely moves; one whose label rests on a single noisy
    measurement moves most of the way. That is the James-Stein form, and it is
    one of the families `kirby/utils/noisy_labels/__init__.py:16-19` records as
    deferred.

    Returns (y_shrunk, shrinkage, tau2).
    """
    y = np.asarray(y, dtype=np.float64)
    oof = np.asarray(oof, dtype=np.float64)
    tau2 = float(np.var(y - oof))
    if not np.isfinite(tau2) or tau2 <= 0:
        # Nothing to shrink toward: the prediction already reproduces the label
        # exactly, so every label would move the whole way for no reason.
        return y.copy(), np.zeros_like(y), tau2
    s = np.asarray(variance, dtype=np.float64) / (
        np.asarray(variance, dtype=np.float64) + tau2)
    return (1.0 - s) * y + s * oof, s, tau2


def fit_noise_variance(y, oof, n_behind, floor_share=VARIANCE_FLOOR_SHARE):
    """sigma_u^2 and sigma_e^2 from squared out-of-fold error against 1/count.

    A label resting on n measurements sits from the out-of-fold prediction by
    squared error sigma_u^2 + sigma_e^2 / n on average. Ordinary least squares
    of (y - oof)^2 on 1/n gives sigma_e^2 as the slope, the variance of one
    measurement, and sigma_u^2 as the intercept, the error left at n infinite.

    Both are floored. A negative slope says error does not fall with the count
    here, and it is set to 0 so every molecule gets the same weight. A
    non-positive intercept is set to `floor_share` of the mean squared error,
    so no weight is infinite. Whether either floor fired is returned.

    Returns (sigma_u2, sigma_e2, info), or (None, None, info) when every label
    rests on the same count and there is no slope to fit.
    """
    y = np.asarray(y, dtype=np.float64)
    oof = np.asarray(oof, dtype=np.float64)
    x = 1.0 / np.maximum(np.asarray(n_behind, dtype=np.float64), 1.0)
    r2 = (y - oof) ** 2
    info = {'n_molecules': int(r2.size),
            'distinct_counts_behind_the_label': int(np.unique(x).size),
            'mean_squared_error': float(np.mean(r2))}
    if np.ptp(x) == 0:
        info['reason'] = ('every training label rests on the same number of '
                          'measurements, so 1/count does not vary and no '
                          'slope exists')
        return None, None, info
    slope, intercept = np.polyfit(x, r2, 1)
    floor = max(floor_share * float(np.mean(r2)), 1e-12)
    info.update({
        'slope_fitted': float(slope),
        'intercept_fitted': float(intercept),
        'slope_floored_to_zero': bool(slope < 0),
        'intercept_floored': bool(intercept < floor),
        'intercept_floor': float(floor),
    })
    return max(float(intercept), floor), max(float(slope), 0.0), info


def pooled_spread_variance(var_all, n_measurements,
                           prior=POOLED_SPREAD_PRIOR):
    """Each molecule's measurement variance pulled toward the pooled variance.

    The pooled variance weights every molecule's own variance by its degrees of
    freedom, count - 1. A molecule's pulled variance is
    ((count - 1) * own + prior * pooled) / ((count - 1) + prior). A molecule
    measured once has no own variance and gets the pooled one.

    Pass the TRAINING molecules only. The pooled value is then built from no
    held-out molecule's measurements.

    Returns (pulled variance per molecule, pooled variance).
    """
    var = np.asarray(var_all, dtype=np.float64)
    df = np.asarray(n_measurements, dtype=np.float64) - 1.0
    known = np.isfinite(var) & (df > 0)
    if not known.any():
        raise RuntimeError(
            "no training molecule was measured more than once, so there is no "
            "pooled measurement variance to pull toward.")
    pooled = float(np.sum(df[known] * var[known]) / np.sum(df[known]))
    pulled = np.full(var.shape, pooled, dtype=np.float64)
    pulled[known] = ((df[known] * var[known] + prior * pooled)
                     / (df[known] + prior))
    return pulled, pooled


def inverse_variance_weights(sigma_u2, label_variance):
    """1 / (sigma_u^2 + label variance), normalised to average one."""
    w = 1.0 / (float(sigma_u2) + np.asarray(label_variance, dtype=np.float64))
    mean_w = float(np.mean(w))
    if not np.isfinite(mean_w) or mean_w <= 0:
        raise RuntimeError(f"inverse-variance weights average {mean_w}, "
                           f"which cannot be normalised.")
    return w / mean_w


def with_count_feature(X_tr, X_va, count_tr, count_va):
    """Both matrices with log(measurement count) appended as one feature.

    Standardised on the training molecules, like every other continuous
    feature here. A count every training molecule shares gives a feature of
    zeros. The new feature takes the matrix's own dtype, so a float32 matrix
    stays float32 (handoff 2.3 item 18, 2026-10-05).
    """
    c_tr = np.log(np.asarray(count_tr, dtype=np.float64))
    c_va = np.log(np.asarray(count_va, dtype=np.float64))
    mu, sd = float(c_tr.mean()), float(c_tr.std())
    sd = sd if sd > 0 else 1.0
    dt = np.asarray(X_tr).dtype
    return (np.hstack([X_tr, ((c_tr - mu) / sd)[:, None].astype(dt)]),
            np.hstack([X_va, ((c_va - mu) / sd)[:, None].astype(dt)]))


def _fit_and_score(factory, X_tr, y_tr, X_va, y_va, metric_fn, weights):
    """One fit, one held-out score, or None with the reason recorded.

    A model whose fit() refuses a per-molecule weight comes back with
    'refuses_sample_weight' and the caller drops it from the weighted
    conditions. It is never fitted unweighted and recorded as if it had been
    weighted -- that is the mistake n13 names at :155-162 and guards the same
    way.
    """
    try:
        m = factory()
        if weights is None:
            m.fit(X_tr, y_tr)
        else:
            m.fit(X_tr, y_tr, sample_weight=weights)
        return float(metric_fn(y_va, m.predict(X_va))), None
    except TypeError as exc:
        if weights is not None and 'sample_weight' in str(exc):
            return None, 'refuses_sample_weight'
        return None, f'{type(exc).__name__}: {exc}'
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'


def _molecule_rows_data(data, stats, labels_per_molecule, dataset_name):
    """The dataset as n19 trains and grades it: one row per molecule.

    The loader keeps one row per measurement (herg_ic50: 12,008 rows over
    9,802 molecules). Until 2026-10-05 the hybrid recipes came from a search
    over those measurement rows while n19 trained and graded on molecules
    (dirty-run bug 11). Now the outer folds, the per-fold searches and every
    feature matrix are on these molecule rows, so all of a molecule's
    measurements sit on one side of every split and the search sees the rows
    the experiment scores (handoff 2.3 item 14).

    Representations and augmentation matrices are each molecule's first row
    (`row_index`), float32 (handoff 2.3 item 18). Groups are the loader's
    scaffold groups on that row. The label is `labels_per_molecule`.
    """
    idx = np.asarray(stats['row_index'])
    groups = data.get('groups')
    augs = data.get('augmentations') or None
    smiles = data.get('smiles')
    return {
        'reps': {n: as_rep_dtype(np.asarray(m)[idx])
                 for n, m in data['reps'].items()},
        'augmentations': ({n: as_rep_dtype(np.asarray(m)[idx])
                           for n, m in augs.items()} if augs else None),
        'labels': np.asarray(labels_per_molecule, dtype=np.float64),
        'groups': None if groups is None else np.asarray(groups)[idx],
        'smiles': (None if smiles is None
                   else [smiles[int(i)] for i in idx]),
        'task_type': data['task_type'],
        'dataset_name': dataset_name,
    }


def _entry_from_selection(s_m):
    """One model's fold_standard_selection result in the form
    noise_helpers.fold_model_inputs reads: 'base', 'hybrids', 'recipes'
    {'hybrid', 'hybrid__general2': (Blueprint, aug_condition)},
    'unavailable'. None stays None (that model's search failed).

    n19 calls fold_standard_selection directly rather than
    noise_helpers.fold_configurations. fold_configurations also runs the
    per-fold distance search and similarity search for the neighbour spaces,
    and n19 finds no neighbours, so those two searches would be built and
    never read (2026-10-05)."""
    if s_m is None:
        return None
    recipes, unavailable = {}, {}
    for space, i in (('hybrid', 0), ('hybrid__general2', 1)):
        if len(s_m['hybrids']) > i:
            h = s_m['hybrids'][i]
            recipes[space] = (h['blueprint'], h.get('aug_condition', 'none'))
        else:
            unavailable[space] = (f'the per-fold general search gave this '
                                  f'model fewer than {i + 1} hybrids with '
                                  f'distinct features')
    return {
        'base': list(s_m['base']),
        'base_detail': s_m.get('base_detail'),
        'hybrids': list(s_m['hybrids']),
        'hybrids_skipped_identical_features':
            s_m.get('hybrids_skipped_identical_features', []),
        'fit_configurations': (list(s_m['base'])
                               + [h['label'] for h in s_m['hybrids']]),
        'recipes': recipes,
        'unavailable': unavailable,
        'selection_scope': 'fold_train_rows',
    }


def _fold_record(entries):
    """What the result file says about one fold's configurations, per model:
    noise_helpers.configuration_record (base representations; hybrids with
    allocation, importance, budget in features, evaluator, representation
    count, augmentation condition) plus each one's score on the fold's
    training molecules (the search's own inner folds)."""
    out = {}
    for m, e in entries.items():
        names = {}
        if e is not None:
            for i, sp in enumerate(BASE_SPACES):
                if len(e['base']) > i:
                    names[sp] = e['base'][i]
            for i, sp in enumerate(('hybrid', 'hybrid__general2')):
                if len(e['hybrids']) > i:
                    names[sp] = e['hybrids'][i].get('label')
        rec = configuration_record(e, names)
        if e is not None:
            # configuration_record reads 'evaluator' and 'n_reps'. The hybrids
            # from fold_standard_selection (_shared.top_hybrids) name them
            # 'internal_evaluator' and 'rep_count', so without this both are
            # written as None (2026-10-05).
            for rh, h in zip(rec['hybrids'], e['hybrids']):
                if rh.get('evaluator') is None:
                    rh['evaluator'] = h.get('internal_evaluator')
                if rh.get('n_reps') is None:
                    rh['n_reps'] = h.get('rep_count')
            # No neighbour space is built here; n19 finds no neighbours.
            rec['neighbour_spaces'] = []
            bd = e.get('base_detail') or {}
            rec['base_training_row_scores'] = {
                s['rep']: s.get('score') for s in (bd.get('selected') or [])}
            rec['hybrid_training_row_scores'] = {
                h.get('label'): h.get('score_mean') for h in e['hybrids']}
        out[m] = convert_numpy(rec)
    return out


def run(data, config, output_dir, dataset_name, **kwargs):
    """N19: the measurement count and the measurement spread as weight and as shrinkage."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N19: WHAT A MEASUREMENT COUNT AND A SPREAD ARE WORTH")
    print("=" * 70)

    labels = np.asarray(data['labels'], dtype=np.float64)
    task_type = data['task_type']
    if task_type != 'regression':
        print(f"  SKIPPED: {dataset_name} is {task_type}. A repeat measurement "
              f"of a class label either agrees or does not, so there is no "
              f"spread in the label's own units and no consensus mean.")
        return {'skipped': 'not_regression', 'task_type': task_type}

    rep_groups = build_replicate_groups(data)
    if rep_groups is None:
        print(f"  SKIPPED: {dataset_name}'s loader exposes no repeat structure, "
              f"so no molecule has a measurement count or a spread. "
              f"NOISE_AUDIT_2026_09_10.md 11.1.")
        return {'skipped': 'no_replicate_structure', 'dataset': dataset_name}

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_draws = int(kwargs.get('n_draws', N_DRAWS))
    label_builds = list(kwargs.get('label_builds') or LABEL_BUILDS)
    oof_splits = int(kwargs.get('oof_splits', OOF_SPLITS))

    stats = measurement_stats(labels, rep_groups)
    probe = build_entities(labels, rep_groups, seed=0)
    if not np.array_equal(probe['n_measurements'], stats['n_measurements']):
        raise RuntimeError(
            "n19: measurement_stats and build_entities disagree about how many "
            "times each molecule was measured, so they are not in the same "
            "order and every weight would be attached to the wrong molecule.")
    if not np.array_equal(probe['row_index'], stats['row_index']):
        raise RuntimeError(
            "n19: measurement_stats and build_entities disagree about which row "
            "represents each molecule.")

    n_ent = len(stats['n_measurements'])
    n_grade = int(probe['gradeable'].sum())
    counts = {int(k): int((stats['n_measurements'] >= k).sum())
              for k in (2, 3, 4, 5)}
    print(f"  {len(labels)} measurements over {n_ent} molecules")
    for k, v in counts.items():
        print(f"    molecules measured {k}+ times: {v} of {n_ent} "
              f"({100.0 * v / max(n_ent, 1):.2f}%)")
    print(f"  gradeable molecules (>= {MIN_MEASUREMENTS} measurements): {n_grade}")

    # ── Models: the dataset's two decided models (R1, DECISIONS.md Step 1) ──
    # 2026-10-05 (chat 5): config['models'] (the five-model DOWNSTREAM_MODELS
    # list) and a `models` kwarg are no longer read. knn is never a
    # downstream model here; it runs only on the distance search.
    models_decided = [m for m in decided_models(dataset_name, 'general')
                      if m != 'knn']
    model_factories = get_model_factories(
        models_decided, task_type=task_type, n_samples=n_ent)
    downstream_models = [m for m in models_decided if m in model_factories]
    models_not_built = {
        m: ('get_model_factories returned no factory (over its row limit at '
            f'{n_ent} molecules, or not defined)')
        for m in models_decided if m not in model_factories}
    if not downstream_models:
        raise RuntimeError(
            f"n19: no decided model could be built. Decided "
            f"{models_decided}, got {sorted(model_factories)}.")

    # The learners fitted beside the downstream models. An empty list is
    # allowed and fits none; an unknown name is refused rather than skipped.
    # OPEN (chat 5): these six learners (ngboost, gp and xgboost
    # survival:aft, each with and without the change) are not the dataset's
    # two models. R1 allows two models; the arms are this experiment's own
    # comparison (each paired only against itself). Kept as they were, fitted
    # in the first decided model's four configurations, until Adelaide says
    # keep or drop. 2026-10-05.
    model_arms = list(MODEL_ARMS if kwargs.get('model_arms') is None
                      else kwargs['model_arms'])
    unknown = [a for a in model_arms if a not in MODEL_ARMS]
    if unknown:
        raise ValueError(f"n19: unknown model arm(s) {unknown}; "
                         f"choose from {list(MODEL_ARMS)}")
    arm_factories = {}
    if any(a.startswith('ngboost') for a in model_arms):
        arm_factories.update(get_model_factories(
            ['ngboost'], task_type=task_type, n_samples=n_ent))
    if any(a.startswith('gp') for a in model_arms):
        # Fetched directly, not through get_model_factories: that drops `gp`
        # above 1,000 molecules, and both GP arms fit a 1,000-molecule draw.
        from kirby.find_best_hybrid import _get_gp_regressor
        arm_factories['gp'] = _get_gp_regressor
    if any(a.startswith('xgboost') for a in model_arms):
        try:
            import xgboost  # noqa: F401
            arm_factories['xgboost'] = fit_interval_aft_predict
        except ImportError:
            pass
    missing = sorted({a.split('_')[0] for a in model_arms} - set(arm_factories))
    if missing:
        raise RuntimeError(f"n19: model arm learner(s) {missing} could not be "
                           f"built; is the package installed?")

    # ── One row per molecule, and the outer folds on those rows ──
    # The per-fold searches see each training molecule's mean over ALL its
    # measurements. One search per fold serves every draw and both label
    # builds; a search per (draw, label build) would be six per fold.
    # OPEN (chat 5): which label the per-fold search sees. The mean of all
    # measurements is what a loader normally hands out (the `averaged`
    # build). Under `single` the hybrid is still rebuilt on the single
    # measurement, but its recipe was chosen on the mean. 2026-10-05.
    ent_data = _molecule_rows_data(data, stats, stats['mean_all'],
                                   dataset_name)
    ent_groups = ent_data['groups']
    folds, n_folds = noise_outer_folds(ent_data, config)
    print(f"  {n_folds} outer folds on {n_ent} molecule rows; running "
          f"{[f for f, _, _ in folds]}")

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'noise_regime': 'real_replicate_disagreement',
        'noise_injection': 'none',
        'models': downstream_models,
        'models_decided': models_decided,
        'models_not_built': models_not_built,
        'label_builds': label_builds,
        'conditions': list(CONDITIONS),
        'spaces': list(SPACES),
        'n_draws': n_draws,
        'n_folds': n_folds,
        'folds_run': [int(f) for f, _, _ in folds],
        'selection_scope': 'fold_train_rows',
        'selection_note': (
            'Per outer fold and per decided model: the 2 best base '
            'representations and the 2 best hybrids, ranked by one search on '
            "that fold's training molecules (decided grid, pruning off). "
            'Hybrids are rebuilt on the training molecules with the label '
            'build being scored. Nothing is read from a whole-dataset search.'),
        'search_rows': 'one row per molecule (its first measurement row)',
        'search_labels': ("each training molecule's mean over all of its "
                          'measurements'),
        'search_condition_suffix': SEARCH_CONDITION_SUFFIX,
        'per_fold_configurations': {},
        'min_measurements_to_grade': MIN_MEASUREMENTS,
        'oof_splits': oof_splits,
        'feasibility': {
            'n_rows': int(len(labels)),
            'n_molecules': n_ent,
            'n_molecules_gradeable': n_grade,
            'molecules_by_min_measurements': counts,
        },
        # Two facts a reader needs before reading any number below, both stated
        # rather than left to be inferred from the absence of a caveat.
        'weights_reach_feature_selection': False,
        'weights_reach_feature_selection_note': (
            'sample_weight appears nowhere in src/kirby/hybrid.py or '
            'src/kirby/find_best_hybrid.py. The three weighted conditions '
            'weight the downstream fit only. The features were chosen without '
            'any weight, in the single-representation spaces and the hybrid '
            'spaces alike.'),
        'model_arms': model_arms,
        'model_arm_pairs': dict(MODEL_ARM_PAIRS),
        'gp_max_training_molecules': GP_MAX_TRAIN,
        'pooled_spread_prior_measurements': POOLED_SPREAD_PRIOR,
        'count_feature_note': (
            "count_feature and invvar_and_count_feature add log(measurement "
            "count) as an input feature. A held-out molecule's count is known "
            'only because it was measured; a molecule nobody has measured has '
            f'none. Every graded held-out molecule was measured '
            f'{MIN_MEASUREMENTS} or more times, so at prediction time the '
            'feature only takes those values.'),
        'fitted_variance_note': (
            'invvar_weighted, invvar_and_count_feature and '
            'spread_pooled_weighted read sigma_u^2 and sigma_e^2 off a least '
            'squares fit of squared out-of-fold error on 1/count, per (fold, '
            'space), on the training molecules only. The out-of-fold '
            "prediction is the LightGBM one `shrunk` already uses. Each fit's "
            'numbers are under fitted_noise_variance.'),
        'recipe_saw_clean_labels': False,
        'recipe_saw_clean_labels_note': (
            'Nothing is injected here and there is no clean label column. '
            "The per-fold search saw each training molecule's mean over all "
            'its measurements, and no held-out molecule.'),
        'per_draw': {},
    }

    if n_grade < 30:
        print(f"  SKIPPED: {n_grade} gradeable molecules is too few to split "
              f"into {n_folds} folds and read a difference between six "
              f"conditions.")
        results['skipped'] = 'too_few_gradeable_molecules'
        return results

    # The model arms have no ranking of their own. They are fitted in the first
    # decided model's four configurations; absent where it has none.
    _arm_owner = downstream_models[0]
    results['model_arms_fitted_in'] = _arm_owner
    results['model_arms_note'] = (
        f'The model arms are not downstream models. They are fitted in '
        f"{_arm_owner}'s four configurations on each fold (its base_1, "
        f'base_2, hybrid and hybrid__general2, named per fold under '
        f'per_fold_configurations). A configuration {_arm_owner} lacks on a '
        f'fold has no model arm in it.')

    # ── Per-fold selection, one search per outer fold ──
    # Memoised here and by _shared.fold_searches; every draw and label build
    # of a fold reads the same 2 base + 2 hybrids per model.
    fold_entries = {}

    def _entries_for(fold_i, tr):
        if fold_i not in fold_entries:
            sel = fold_standard_selection(
                ent_data['reps'], ent_data['labels'], ent_groups,
                train_idx=tr, fold_i=int(fold_i), n_outer_folds=n_folds,
                dataset_name=dataset_name, models=list(downstream_models),
                augmentations=ent_data['augmentations'], task_type=task_type,
                condition_suffix=SEARCH_CONDITION_SUFFIX)
            fold_entries[fold_i] = {m: _entry_from_selection(sel.get(m))
                                    for m in downstream_models}
            results['per_fold_configurations'][str(fold_i)] = _fold_record(
                fold_entries[fold_i])
        return fold_entries[fold_i]

    spaces = SPACES

    # Models that refused a per-molecule weight, discovered by being refused
    # rather than assumed. Written to the result file so a missing cell in a
    # weighted condition is traceable to the model rather than to the data.
    refuses_weight = set()

    # ── The main measurement and the checkpoint ──
    # Added 2026-09-25 (plan sections 2-3).
    #
    # ROWS. D is `untouched` -- every training molecule at its label,
    # unweighted -- and D' is each other condition, on the same feature
    # matrix, one d_vs_dprime row per (draw, label build, fold, matrix,
    # condition, model), built from the numbers the loop below already
    # computes: no refit. The model arms give one row per pair, the variant
    # against its own plain arm (MODEL_ARM_PAIRS). similarity_setting is
    # 'none': nothing here finds a neighbour. score_clean is nan: there is no
    # clean label column.
    #
    # CHECKPOINT. The accumulators are pickled after every (draw, label build,
    # fold), so a killed run resumes at the next one; rows resume through
    # done_keys. Both are filed under SELECTION_VERSION since 2026-10-05, so
    # nothing written under the whole-dataset selection is read back.
    metric = (metric_name, metric_fn, lower_is_better)
    rows = sx.Rows(output_dir, EXP_FULL_NAME, dataset_name,
                   condition=SELECTION_VERSION)
    results['rows'] = rows.summary()
    ck = sx.FoldCheckpoint(
        output_dir, EXP_FULL_NAME, dataset_name,
        sx.run_signature(data, config, {
            'models': downstream_models, 'n_draws': n_draws,
            'label_builds': label_builds, 'oof_splits': oof_splits,
            'model_arms': model_arms, 'spaces': list(spaces),
            'selection': SELECTION_VERSION, 'n_folds': n_folds,
            'folds_run': results['folds_run'],
            'search_condition_suffix': SEARCH_CONDITION_SUFFIX}))
    if ck.state is not None:
        results['per_draw'] = ck.state['per_draw']
        refuses_weight = set(ck.state['refuses_weight'])
        results['per_fold_configurations'].update(
            ck.state.get('per_fold_configurations') or {})
    _st0 = ck.state or {}
    _cm_ident = dict(dataset=dataset_name, experiment=EXP_FULL_NAME,
                     noise_condition=ROW_CONDITION)

    def _row_ident(method, build, space, cell_rep, draw, fold_i):
        return sx.identity(**_cm_ident, method=method,
                           method_setting=f'label_build={build}',
                           similarity_setting=sx.NO_SIMILARITY,
                           model_input=f'{space}:{cell_rep}', draw=int(draw),
                           fold=int(fold_i))

    # ── The experiment ──
    for draw in range(n_draws):
        if str(draw) in results['per_draw']:
            print(f"\n  === draw {draw}: already in the checkpoint, skipped ===")
            continue
        t_draw = time.time()
        print(f"\n  === draw {draw} ===")
        ent = build_entities(labels, rep_groups, seed=draw)
        y_grade_all = ent['grade_label']
        gradeable = ent['gradeable']

        draw_block = {'per_label_build': {}}
        if _st0.get('draw') == draw:
            draw_block = _st0['draw_block']

        for build in label_builds:
            if build in draw_block['per_label_build']:
                print(f"    label build {build!r}: already in the checkpoint, "
                      f"skipped")
                continue
            y_train_all, n_behind = training_labels(ent, stats, build)
            aft_lo_all, aft_hi_all = interval_bounds(ent, stats, build,
                                                     y_train_all)
            var_own, n_filled, var_floor = label_variance(stats, n_behind)
            var_cal, noise_fit, n_trusted = calibrated_variance(stats, n_behind)

            build_block = {
                'n_molecules_with_no_variance_of_their_own': n_filled,
                'variance_floor_label_units_squared': var_floor,
                'n_molecules_in_trusted_subset': n_trusted,
                # How many molecules give xgboost_interval_aft an interval of
                # non-zero width. The rest enter it exactly as they enter
                # xgboost_aft_point.
                'n_molecules_with_interval_width_above_zero': int(
                    (aft_hi_all > aft_lo_all).sum()),
                # The noise function itself: measurement count -> the mean
                # variance of the molecules measured that many times. A reader
                # should be able to see whether it falls with the count.
                'fitted_variance_by_measurement_count': noise_fit,
                'per_space': {},
            }
            print(f"    label build {build!r}: "
                  f"{n_filled} molecules have no variance of their own, "
                  f"trusted subset {n_trusted}")

            # {space: {condition: {model: {fold key: score}}}}
            #
            # Keyed by the fold, NOT appended to a list. Two conditions have to
            # be compared on the SAME folds: a condition whose fit raised on
            # fold 2 appends nothing, and two lists of different lengths then
            # line fold 3 up against fold 2. The gain that comes out of that is
            # a difference between folds, which on this data is several times
            # larger than any difference between conditions.
            acc = {s: {c: {m: {} for m in downstream_models}
                       for c in CONDITIONS} for s in spaces}
            errors = {}
            # {space: {feature matrix: [one value per fold]}}. Each model has
            # its own matrix in each space since 2026-10-05, so a feature
            # matrix is named '<model>:<representation or hybrid label>'.
            tau2_seen = {s: {} for s in spaces}
            shrink_seen = {s: {} for s in spaces}
            # {space: [one fit_noise_variance record per (fold, matrix)]}
            noise_fits = {s: [] for s in spaces}
            # {space: {model arm: {fold key: score}}}
            arm_acc = {s: {a: {} for a in model_arms} for s in spaces}
            # {model arm: {fold key: skip record}}. The same in every space:
            # the interval bounds do not depend on the space.
            arm_skips = {}
            folds_scored = 0
            if _st0.get('draw') == draw and _st0.get('build') == build:
                # This label build was part-way through when the run stopped.
                (acc, errors, tau2_seen, shrink_seen, noise_fits, arm_acc,
                 arm_skips, folds_scored) = _st0['build_acc']

            for fold_i, tr, va in folds:
                fold_key = f'd{draw}f{fold_i}'
                _unit = f'd{draw}|{build}|f{fold_i}'
                if ck.is_done(_unit):
                    print(f"      fold {fold_i}: already in the checkpoint, "
                          f"skipped")
                    continue
                fold_rows = []
                va_graded = va[gradeable[va]]
                if len(va_graded) < 5:
                    print(f"      fold {fold_i}: {len(va_graded)} gradeable "
                          f"molecules held out, skipped")
                    continue
                folds_scored += 1
                y_tr = y_train_all[tr]
                y_va = y_grade_all[va_graded]
                g_tr = ent_groups[tr] if ent_groups is not None else None

                # Under `single` a gradeable molecule's interval is its one
                # training measurement, zero width. If every training molecule
                # of this fold has a zero-width interval, xgboost_interval_aft
                # is handed exactly the bounds xgboost_aft_point is handed, and
                # would only repeat its number at the cost of an xgboost fit per
                # space. So it is not fitted in that fold; the reason and the
                # count are written instead. The count is measured, not assumed:
                # a molecule measured twice cannot be graded, keeps both
                # measurements under `single`, and gives a real interval, and
                # in any fold that has one the arm runs. `averaged` is not
                # checked here and runs as before.
                skip_interval_aft = False
                if build == 'single' and 'xgboost_interval_aft' in model_arms:
                    n_zero_width = int(
                        (aft_hi_all[tr] <= aft_lo_all[tr]).sum())
                    if n_zero_width == len(tr):
                        skip_interval_aft = True
                        arm_skips.setdefault('xgboost_interval_aft', {})[
                            fold_key] = {
                            'reason': (
                                'every training molecule has a zero-width '
                                'interval under the single build, so the '
                                'bounds are those xgboost_aft_point is given '
                                'and the fit would repeat it'),
                            'n_training_molecules': int(len(tr)),
                            'n_training_molecules_with_zero_width_interval':
                                n_zero_width,
                        }

                # Every feature space, all built from this fold's training
                # molecules only (2026-10-05, chat 5): per decided model, its
                # four fitted configurations from the per-fold search
                # (noise_helpers.fold_model_inputs). Base representations are
                # scaled with kirby.scaling.FeatureScaler (0/1 features left
                # as they are); hybrids are rebuilt on y_tr, the label build
                # being scored, with the fold search's own representation
                # scores (dirty-run bug 1).
                #
                # {space: [(X_tr, X_va, [model], what the matrix is, whether
                # the model arms run on it)]}.
                entries = _entries_for(fold_i, tr)
                fold_spaces = {}
                for _m in downstream_models:
                    _X, _why, _names = fold_model_inputs(
                        ent_data, dataset_name, _m, entries.get(_m),
                        train_idx=tr, val_idx=va_graded, y_train=y_tr,
                        factory=model_factories[_m], metric=metric)
                    for _sp in spaces:
                        if _sp in _X:
                            fold_spaces.setdefault(_sp, []).append((
                                _X[_sp][0], _X[_sp][1], [_m],
                                str(_names.get(_sp, _sp)), _m == _arm_owner))
                        else:
                            errors[f'{_sp}/{_m}|{fold_key}|unavailable'] = (
                                _why.get(_sp, 'not built'))

                for space, (X_tr_s, X_va_s, cell_models, cell_rep,
                            run_arms) in (
                        (_sp, _c) for _sp in spaces
                        for _c in fold_spaces.get(_sp, ())):
                    # What an error is filed under. Two feature matrices in one
                    # space must not overwrite each other's failures. Each
                    # model has its own matrix since 2026-10-05, so the name
                    # carries the model.
                    _fm = f"{'+'.join(cell_models)}:{cell_rep}"
                    _ek = f'{space}/{_fm}'

                    # The out-of-fold prediction the two relabelling conditions
                    # shrink toward, computed inside this fold's TRAINING
                    # molecules only and in the space the model will be fitted
                    # in. Group-aware, because a random inner fold would put a
                    # molecule's scaffold family on both sides and deflate the
                    # residual the shrinkage keys on
                    # (noisy_labels/core.py:3-8).
                    y_shrunk = y_calib = None
                    oof = None
                    try:
                        # Through noise_cache since 2026-09-25 (plan section
                        # 3): kirby.utils.noisy_labels.oof_predict with the
                        # same arguments and its default seed 0.
                        oof = nc.cached_oof_predict(dataset_name, X_tr_s, y_tr,
                                                    g_tr, n_splits=oof_splits)
                        y_shrunk, s_own, tau2 = shrink_toward(
                            y_tr, oof, var_own[tr])
                        y_calib, _s_cal, _ = shrink_toward(
                            y_tr, oof, var_cal[tr])
                        tau2_seen[space].setdefault(_fm, []).append(
                            float(tau2))
                        shrink_seen[space].setdefault(_fm, []).append(
                            float(np.mean(s_own)))
                    except Exception as exc:
                        errors[f'{_ek}|{fold_key}|oof'] = f'{type(exc).__name__}: {exc}'
                        print(f"      fold {fold_i} {_ek}: out-of-fold "
                              f"prediction failed: {type(exc).__name__}: {exc}")

                    # sigma_u^2 and sigma_e^2 for this fold and space, from the
                    # same out-of-fold prediction. None when it failed or when
                    # 1/count does not vary; the three conditions reading them
                    # are then absent from this fold, with the reason recorded.
                    s2u = s2e = None
                    w_invvar = w_pooled = None
                    if oof is not None:
                        s2u, s2e, nf = fit_noise_variance(y_tr, oof,
                                                          n_behind[tr])
                        nf['fold'] = fold_key
                        nf['feature_matrix'] = _fm
                        noise_fits[space].append(nf)
                        if s2u is None:
                            errors[f'{_ek}|{fold_key}|noise_variance'] = (
                                nf['reason'])
                        else:
                            w_invvar = inverse_variance_weights(
                                s2u, s2e / np.maximum(n_behind[tr], 1.0))
                            try:
                                pulled, pooled = pooled_spread_variance(
                                    stats['var_all'][tr],
                                    stats['n_measurements'][tr])
                                nf['pooled_measurement_variance'] = pooled
                                w_pooled = inverse_variance_weights(
                                    s2u, pulled / np.maximum(n_behind[tr], 1.0))
                            except Exception as exc:
                                errors[f'{_ek}|{fold_key}|pooled_spread'] = (
                                    f'{type(exc).__name__}: {exc}')

                    X_tr_c, X_va_c = with_count_feature(
                        X_tr_s, X_va_s, stats['n_measurements'][tr],
                        stats['n_measurements'][va_graded])

                    for cond in CONDITIONS:
                        X_fit, X_score = X_tr_s, X_va_s
                        if cond in RELABEL_CONDITIONS:
                            y_use = y_shrunk if cond == 'shrunk' else y_calib
                            if y_use is None:
                                continue
                            w = None
                        elif cond in WEIGHTED_CONDITIONS:
                            y_use = y_tr
                            w = weights_for(cond,
                                            stats['n_measurements'][tr],
                                            var_own[tr], n_behind[tr])
                        elif cond == 'count_feature':
                            y_use, w = y_tr, None
                            X_fit, X_score = X_tr_c, X_va_c
                        elif cond in FITTED_VARIANCE_CONDITIONS:
                            w = (w_pooled if cond == 'spread_pooled_weighted'
                                 else w_invvar)
                            if w is None:
                                continue
                            y_use = y_tr
                            if cond in COUNT_FEATURE_CONDITIONS:
                                X_fit, X_score = X_tr_c, X_va_c
                        else:
                            y_use, w = y_tr, None

                        for mname in cell_models:
                            if w is not None and mname in refuses_weight:
                                continue
                            score, err = _fit_and_score(
                                model_factories[mname], X_fit, y_use,
                                X_score, y_va, metric_fn, w)
                            if err == 'refuses_sample_weight':
                                refuses_weight.add(mname)
                                continue
                            if score is None:
                                errors[f'{space}|{cond}|{mname}|{fold_key}'] = err
                                continue
                            acc[space][cond][mname][fold_key] = score
                            # The random and shuffled twins of this
                            # condition's weights or labels, under
                            # KIRBY_NOISE_CONTROLS=1 only. The count-feature
                            # conditions change the features, which neither
                            # twin can deal out, so they get none.
                            if (sx.controls_enabled() and cond != 'untouched'
                                    and cond not in COUNT_FEATURE_CONDITIONS):
                                _cl = {'X': X_fit, 'y': y_use,
                                       'sample_weight': w, 'keep': None,
                                       'label_column': (
                                           y_use if cond in RELABEL_CONDITIONS
                                           else None)}
                                _D = acc[space]['untouched'][mname].get(
                                    fold_key)
                                sx.control_rows(
                                    rows, dataset_name,
                                    _row_ident(cond, build, space, cell_rep,
                                               draw, fold_i),
                                    _cl, X_tr=X_tr_s, y_D=y_tr, X_va=X_va_s,
                                    y_va=y_va, metric=metric,
                                    factories={mname: model_factories[mname]},
                                    seed=sx.seed_of(draw, build, fold_i, space,
                                                    cell_rep, cond, mname),
                                    score_D=(None if _D is None
                                             else {mname: _D}))

                    # ── The model arms ──
                    # One GP draw per (draw, fold), shared by both GP arms and
                    # every space, so the two GP arms differ only in `alpha`.
                    gp_rows = None
                    if not run_arms:
                        continue
                    if any(a.startswith('gp') for a in model_arms):
                        rng = np.random.default_rng(1000 * draw + fold_i)
                        gp_rows = (np.arange(len(tr)) if len(tr) <= GP_MAX_TRAIN
                                   else np.sort(rng.choice(
                                       len(tr), GP_MAX_TRAIN, replace=False)))
                    for arm in model_arms:
                        try:
                            if arm == 'ngboost_plain':
                                m = arm_factories['ngboost']()
                                m.fit(X_tr_s, y_tr)
                                pred = m.predict(X_va_s)
                            elif arm == 'ngboost_count_feature':
                                m = arm_factories['ngboost']()
                                m.fit(X_tr_c, y_tr)
                                pred = m.predict(X_va_c)
                            elif arm == 'gp_plain':
                                m = arm_factories['gp']()
                                m.fit(X_tr_s[gp_rows], y_tr[gp_rows])
                                pred = m.predict(X_va_s)
                            elif arm in ('xgboost_aft_point',
                                         'xgboost_interval_aft'):
                                if oof is None:
                                    continue
                                if (arm == 'xgboost_interval_aft'
                                        and skip_interval_aft):
                                    continue
                                sd = float(np.std(y_tr - oof))
                                if arm == 'xgboost_aft_point':
                                    lo_b = hi_b = y_tr
                                else:
                                    lo_b, hi_b = aft_lo_all[tr], aft_hi_all[tr]
                                pred = fit_interval_aft_predict(
                                    X_tr_s, lo_b, hi_b, X_va_s, sd)
                            else:  # gp_per_molecule_noise
                                if s2e is None:
                                    continue
                                y_sub = y_tr[gp_rows]
                                var_y = float(np.var(y_sub))
                                if var_y <= 0:
                                    continue
                                m = arm_factories['gp']()
                                # Label variance on the standardised scale
                                # normalize_y fits in; see the header. The
                                # 1e-6 the `gp` entry ships is kept on top.
                                m.set_params(alpha=(
                                    s2e / np.maximum(n_behind[tr][gp_rows], 1.0)
                                    / var_y) + 1e-6)
                                m.fit(X_tr_s[gp_rows], y_sub)
                                pred = m.predict(X_va_s)
                            arm_acc[space][arm][fold_key] = float(
                                metric_fn(y_va, pred))
                        except Exception as exc:
                            errors[f'{space}|{arm}|{fold_key}'] = (
                                f'{type(exc).__name__}: {exc}')

                # ── This fold's rows, from the scores just computed ──
                for space, (_X1, _X2, cell_models, cell_rep, run_arms) in (
                        (_sp, _c) for _sp in spaces
                        for _c in fold_spaces.get(_sp, ())):
                    for cond in CONDITIONS:
                        if cond == 'untouched':
                            continue
                        for mname in cell_models:
                            _D = acc[space]['untouched'][mname].get(fold_key)
                            if _D is None:
                                continue
                            fold_rows.append(sx.row(
                                _row_ident(cond, build, space, cell_rep, draw,
                                           fold_i),
                                model=mname, metric=metric, score_D=_D,
                                score_Dprime=acc[space][cond][mname].get(
                                    fold_key)))
                    if not run_arms:
                        continue
                    for variant, plain in MODEL_ARM_PAIRS.items():
                        _D = arm_acc[space].get(plain, {}).get(fold_key)
                        if variant not in model_arms or _D is None:
                            continue
                        fold_rows.append(sx.row(
                            _row_ident(variant, build, space, cell_rep, draw,
                                       fold_i),
                            model=MODEL_ARM_LEARNER[variant], metric=metric,
                            score_D=_D,
                            score_Dprime=arm_acc[space].get(variant, {}).get(
                                fold_key)))
                rows.write(fold_rows)
                ck.save(_unit, {
                    'per_draw': results['per_draw'],
                    'refuses_weight': sorted(refuses_weight),
                    'per_fold_configurations':
                        results['per_fold_configurations'],
                    'draw': draw, 'build': build, 'draw_block': draw_block,
                    'build_acc': (acc, errors, tau2_seen, shrink_seen,
                                  noise_fits, arm_acc, arm_skips,
                                  folds_scored)})

            for space in spaces:
                _t_all = [v for vs in tau2_seen[space].values() for v in vs]
                _s_all = [v for vs in shrink_seen[space].values() for v in vs]
                build_block['per_space'][space] = {
                    # Over every (fold, feature matrix) of this space. In a
                    # single-representation space the per-representation
                    # means are beside it.
                    'mean_tau2_over_folds': (
                        float(np.mean(_t_all)) if _t_all else float('nan')),
                    'mean_shrinkage_over_folds': (
                        float(np.mean(_s_all)) if _s_all else float('nan')),
                    # Per feature matrix: a representation in base_1 /
                    # base_2, a model's recipe in a hybrid space.
                    'mean_tau2_over_folds_by_feature_matrix': {
                        r: float(np.mean(v))
                        for r, v in tau2_seen[space].items()},
                    'mean_shrinkage_over_folds_by_feature_matrix': {
                        r: float(np.mean(v))
                        for r, v in shrink_seen[space].items()},
                    # One record per fold: the slope (sigma_e^2) and intercept
                    # (sigma_u^2) the three fitted-variance conditions used.
                    'fitted_noise_variance': noise_fits[space],
                    'model_arms': {
                        a: {'fold_scores': dict(v), 'n_folds': len(v)}
                        for a, v in arm_acc[space].items()},
                    'per_condition': {
                        c: {'per_model': {
                            m: {'mean': (float(np.mean(list(v.values()))) if v
                                         else float('nan')),
                                'fold_spread': fold_spread(list(v.values())),
                                # Keyed by the fold that produced it, so the
                                # summary can pair two conditions on the folds
                                # they both scored.
                                'fold_scores': {k: float(x)
                                                for k, x in v.items()},
                                'n_folds': len(v)}
                            for m, v in acc[space][c].items()}}
                        for c in CONDITIONS},
                }
            build_block['n_folds_scored'] = folds_scored
            build_block['model_arm_skips'] = arm_skips
            if errors:
                build_block['_errors'] = errors
            draw_block['per_label_build'][build] = convert_numpy(build_block)
            ck.save(f'd{draw}|{build}', {
                'per_draw': results['per_draw'],
                'per_fold_configurations': results['per_fold_configurations'],
                'refuses_weight': sorted(refuses_weight), 'draw': draw,
                'build': None, 'draw_block': draw_block})

        results['per_draw'][str(draw)] = draw_block
        ck.save(f'd{draw}', {'per_draw': results['per_draw'],
                             'per_fold_configurations':
                                 results['per_fold_configurations'],
                             'refuses_weight': sorted(refuses_weight),
                             'draw': None})
        # The result file is written after every draw, marked partial, so a
        # run stopped later still leaves its finished draws on disk (dirty-run
        # bug 12, 2026-10-05). The summary is added once every draw is in.
        if output_dir is not None:
            save_experiment_results(
                dict(results, partial=True, rows=rows.summary(),
                     models_refusing_sample_weight=sorted(refuses_weight)),
                output_dir, 'n19_measurement_count', dataset_name)
        print(f"    [{time.time() - t_draw:.0f}s]")
    results['rows'] = rows.summary()
    results['partial'] = False

    results['models_refusing_sample_weight'] = sorted(refuses_weight)
    results['summary'] = _summarise(results, downstream_models, label_builds,
                                    spaces, metric_name, lower_is_better)
    results['summary']['model_arms'] = _summarise_model_arms(
        results, label_builds, spaces, model_arms, lower_is_better)
    _print_summary(results, metric_name)

    save_experiment_results(results, output_dir, 'n19_measurement_count',
                            dataset_name)
    # The result file holds every draw now, so the per-fold snapshot is spent.
    ck.clear()
    return results


def _fold_scores(results, build, space, cond, model):
    """Every fold score for one cell, keyed by the fold that produced it.

    Pooled across draws, and the keys carry the draw, so two conditions can be
    lined up on exactly the folds they both scored.
    """
    out = {}
    for draw in results['per_draw'].values():
        blk = draw['per_label_build'].get(build, {})
        sp = blk.get('per_space', {}).get(space)
        if not sp:
            continue
        cell = sp['per_condition'][cond]['per_model'].get(model)
        if cell:
            out.update(cell['fold_scores'])
    return out


def _summarise(results, models, label_builds, spaces, metric_name,
               lower_is_better):
    """One row per (label build, space, model): every condition against untouched.

    THE HEADLINE IS A COUNT OF FOLDS, not a mean, and the reason is n18's at
    :83-88. Averaging hides a condition that wins big on one fold and loses on
    four. It also hides something worse here: the held-out molecules are graded
    against a consensus of real measurements, so a fold holding out the
    badly-measured molecules scores worse whatever condition is running. MEASURED
    on the smoke's own data: the spread of the raw scores across folds was 0.26
    to 0.32 while every difference between conditions was 0.01 to 0.18. Read
    unpaired, none of them is a difference.

    So every comparison here is PAIRED: the same fold, the same molecules, the
    same grading labels, two conditions. What varies between folds is common to
    both and cancels.

    Positive gain always means better, whichever way the metric runs.
    """
    out = {'metric': metric_name, 'lower_is_better': bool(lower_is_better),
           'headline': 'folds won against untouched, not mean gain',
           'comparison': 'paired per fold',
           'per_label_build': {}}

    for build in label_builds:
        build_row = {'per_space': {}}
        for space in spaces:
            space_row = {'per_model': {}}
            for m in models:
                base = _fold_scores(results, build, space, 'untouched', m)
                base_vals = list(base.values())
                row = {'untouched_mean': (float(np.mean(base_vals))
                                          if base_vals else float('nan')),
                       'untouched_n_folds': len(base),
                       'per_condition': {}}
                for cond in CONDITIONS:
                    got = _fold_scores(results, build, space, cond, m)
                    shared = sorted(set(base) & set(got))
                    gains = np.array(
                        [(base[k] - got[k]) if lower_is_better
                         else (got[k] - base[k]) for k in shared],
                        dtype=float)
                    gains = gains[np.isfinite(gains)]
                    # A fold is only WON when the two fits actually scored
                    # differently. Two conditions can be the identical fit --
                    # weighting by a measurement count every molecule shares
                    # gives every molecule the same weight -- and the gain is
                    # then floating-point dust. MEASURED on the smoke: a mean
                    # gain of -1.4e-17 over eight folds, one of which came out
                    # positive. Counting that as a win says a condition beat
                    # untouched when it WAS untouched.
                    scale = max(abs(row['untouched_mean']), 1.0)
                    tol = 1e-9 * scale
                    # Spread over the paired gains, through fold_spread: on a
                    # single (draw, fold) there is no spread and it says so
                    # instead of reporting 0 (2026-10-05).
                    _fs = fold_spread(list(gains))
                    row['per_condition'][cond] = {
                        'mean': (float(np.mean(list(got.values())))
                                 if got else float('nan')),
                        'n_folds': len(got),
                        # The folds BOTH this condition and untouched scored.
                        # Everything below is over those and only those.
                        'n_folds_paired': int(gains.size),
                        'folds_won': int((gains > tol).sum()),
                        'folds_lost': int((gains < -tol).sum()),
                        # Folds where the two scored the same number. A large
                        # count here means the two conditions are one fit.
                        'folds_tied': int((np.abs(gains) <= tol).sum()),
                        'tie_tolerance': float(tol),
                        'mean_gain': (float(gains.mean()) if gains.size
                                      else float('nan')),
                        'spread_over_fold_gains': _fs['sd'],
                        'fold_gain_spread': _fs,
                        'fold_gains': [float(x) for x in gains],
                        # A gain that is not larger than the spread of the
                        # per-fold gains behind it is not a gain. Stated as a
                        # comparison of two measured numbers, not a test: at
                        # three draws over five folds there is no replicate
                        # count that would support a confidence interval, and
                        # the suite's standing rule is raw spread only.
                        'gain_exceeds_its_own_spread': (
                            None if _fs['sd'] is None
                            else bool(gains.mean() > _fs['sd'])),
                    }
                space_row['per_model'][m] = row
            build_row['per_space'][space] = space_row
        out['per_label_build'][build] = build_row

    # Does a weighting help MORE in a searched space than in a fixed one? That
    # is the question n02 asks of importance weights and nothing has asked of a
    # measurement count. Only defined where both spaces ran.
    def _cross_for(hybrid_space, base_space):
        """One hybrid space against one single-representation space."""
        cross = {}
        for build in label_builds:
            per_cond = {}
            for cond in CONDITIONS:
                per_model = {}
                for m in models:
                    b = out['per_label_build'][build]['per_space'][base_space][
                        'per_model'][m]['per_condition'][cond]['mean_gain']
                    h = out['per_label_build'][build]['per_space'][
                        hybrid_space]['per_model'][m][
                            'per_condition'][cond]['mean_gain']
                    per_model[m] = {
                        'gain_in_base': b,
                        'gain_in_hybrid': h,
                        'hybrid_minus_base': (float(h - b)
                                              if np.isfinite(b)
                                              and np.isfinite(h)
                                              else float('nan')),
                    }
                per_cond[cond] = per_model
            cross[build] = per_cond
        return cross

    # Every hybrid space this run built, each against each single-
    # representation space. Keyed hybrid space, then base_1 / base_2.
    _hybrid_spaces = [sp for sp in spaces if sp not in BASE_SPACES]
    _base_spaces = [sp for sp in spaces if sp in BASE_SPACES]
    if _hybrid_spaces and _base_spaces:
        out['gain_by_hybrid_space'] = {
            sp: {b: _cross_for(sp, b) for b in _base_spaces}
            for sp in _hybrid_spaces}
    if 'hybrid' in spaces and _base_spaces:
        out['gain_by_space'] = {b: _cross_for('hybrid', b)
                                for b in _base_spaces}
        out['gain_by_space_note'] = (
            "Keyed by single-representation space (base_1, base_2), then "
            "label build. Each entry is one condition's mean paired gain over "
            "untouched in that model's rank-1 or rank-2 single "
            'representation and in the hybrid, and the difference between the '
            'two. It says whether the weighting or the shrinkage is worth more '
            'where the features were chosen. The features were chosen without '
            "any weight in every space. 'gain_by_space' holds each model's "
            "best per-fold hybrid ('hybrid'); 'gain_by_hybrid_space' holds "
            "the same table for both hybrids ('hybrid', 'hybrid__general2'), "
            'keyed by space name.')
    return out


def _summarise_model_arms(results, label_builds, spaces, model_arms,
                          lower_is_better):
    """Each changed model arm against its own plain arm, paired per fold.

    ngboost_count_feature against ngboost_plain, gp_per_molecule_noise against
    gp_plain, xgboost_interval_aft against xgboost_aft_point. Folds where an
    arm was deliberately not fitted (`model_arm_skips`) are counted in
    `n_folds_skipped` and pair nothing. Never against a downstream model: those are different learners
    and the difference would be the learner, not the change.
    """
    out = {'pairs': dict(MODEL_ARM_PAIRS), 'per_label_build': {}}
    for build in label_builds:
        per_space = {}
        # {arm: {fold key: skip record}}, over every draw. Space-independent.
        skips = {}
        for draw in results['per_draw'].values():
            blk = draw['per_label_build'].get(build, {})
            for a, by_fold in (blk.get('model_arm_skips') or {}).items():
                skips.setdefault(a, {}).update(by_fold)
        for space in spaces:
            scores = {}
            for draw in results['per_draw'].values():
                blk = draw['per_label_build'].get(build, {})
                cells = (blk.get('per_space', {}).get(space, {})
                         .get('model_arms', {}))
                for a, cell in cells.items():
                    scores.setdefault(a, {}).update(cell['fold_scores'])
            rows = {}
            for arm, plain in MODEL_ARM_PAIRS.items():
                if arm not in model_arms or plain not in model_arms:
                    continue
                base, got = scores.get(plain, {}), scores.get(arm, {})
                shared = sorted(set(base) & set(got))
                gains = np.array([(base[k] - got[k]) if lower_is_better
                                  else (got[k] - base[k]) for k in shared],
                                 dtype=float)
                gains = gains[np.isfinite(gains)]
                scale = max(abs(float(np.mean(list(base.values()))))
                            if base else 1.0, 1.0)
                tol = 1e-9 * scale
                _fs = fold_spread(list(gains))
                rows[arm] = {
                    'against': plain,
                    'plain_mean': (float(np.mean(list(base.values())))
                                   if base else float('nan')),
                    'arm_mean': (float(np.mean(list(got.values())))
                                 if got else float('nan')),
                    'n_folds_paired': int(gains.size),
                    'folds_won': int((gains > tol).sum()),
                    'folds_lost': int((gains < -tol).sum()),
                    'folds_tied': int((np.abs(gains) <= tol).sum()),
                    'mean_gain': (float(gains.mean()) if gains.size
                                  else float('nan')),
                    'spread_over_fold_gains': _fs['sd'],
                    'fold_gain_spread': _fs,
                    'fold_gains': [float(x) for x in gains],
                    # Folds where the arm was deliberately not fitted, and why.
                    'n_folds_skipped': len(skips.get(arm, {})),
                    'skip_reasons': sorted({v['reason'] for v in
                                            skips.get(arm, {}).values()}),
                }
            per_space[space] = rows
        out['per_label_build'][build] = per_space
    return out


def _print_summary(results, metric_name):
    s = results['summary']
    print("\n" + "=" * 70)
    print(f"SUMMARY ({metric_name}; paired per fold against untouched)")
    if results.get('models_refusing_sample_weight'):
        print(f"  refused a per-molecule weight, so absent from the three "
              f"weighted conditions: "
              f"{', '.join(results['models_refusing_sample_weight'])}")
    for build, brow in s['per_label_build'].items():
        what = ('one measurement per molecule' if build == 'single'
                else 'the mean of every measurement')
        print(f"\n  training label = {what}")
        for space, srow in brow['per_space'].items():
            print(f"    {space}:")
            for m, row in srow['per_model'].items():
                if not np.isfinite(row['untouched_mean']):
                    print(f"      {m}: no fold scored")
                    continue
                print(f"      {m}  untouched {row['untouched_mean']:.4f} "
                      f"over {row['untouched_n_folds']} folds")
                for cond in CONDITIONS:
                    if cond == 'untouched':
                        continue
                    c = row['per_condition'][cond]
                    if not c['n_folds_paired']:
                        print(f"        {cond:19s} no fold scored both")
                        continue
                    mark = '  >spread' if c['gain_exceeds_its_own_spread'] else ''
                    _sd = c['spread_over_fold_gains']
                    _sd_txt = ('one fold, no spread' if _sd is None
                               else f'spread {_sd:.4f}')
                    print(f"        {cond:19s} won {c['folds_won']} of "
                          f"{c['n_folds_paired']} folds, "
                          f"mean {c['mean_gain']:+.4f} "
                          f"({_sd_txt}){mark}")
    ma = s.get('model_arms') or {}
    for build, by_space in (ma.get('per_label_build') or {}).items():
        for space, rows in by_space.items():
            for arm, r in rows.items():
                if r.get('n_folds_skipped'):
                    print(f"  {build}/{space} {arm}: not fitted in "
                          f"{r['n_folds_skipped']} folds: "
                          f"{'; '.join(r['skip_reasons'])}")
                if not r['n_folds_paired']:
                    if not r.get('n_folds_skipped'):
                        print(f"  {build}/{space} {arm}: no fold scored both "
                              f"it and {r['against']}")
                    continue
                _sd = r['spread_over_fold_gains']
                _sd_txt = ('one fold, no spread' if _sd is None
                           else f'spread {_sd:.4f}')
                print(f"  {build}/{space} {arm} against {r['against']}: "
                      f"{r['arm_mean']:.4f} against {r['plain_mean']:.4f}, "
                      f"won {r['folds_won']} of {r['n_folds_paired']} folds, "
                      f"mean {r['mean_gain']:+.4f} ({_sd_txt})")
    print("\n  The count of folds won is the headline. A mean gain no larger "
          "than the")
    print("  spread of the per-fold gains behind it is not a gain.")
    if 'gain_by_space' in s:
        print("  'gain_by_space' in the result file holds the same gains in "
              "each single representation and the hybrid, and the difference.")


def main():
    parser = create_experiment_parser(
        'N19: the measurement count and spread as weight and as shrinkage')
    parser.add_argument('--n-draws', type=int, default=N_DRAWS,
                        help='How many times the training measurement is re-drawn')
    parser.add_argument('--label-builds', type=str, default=None,
                        help="Comma-separated: single, averaged")
    parser.add_argument('--model-arms', type=str, default=None,
                        help=f"Comma-separated, from {','.join(MODEL_ARMS)}; "
                             f"'none' fits none. Default: all six")
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, rep_preset=args.reps)
    kw = {'n_draws': args.n_draws}
    if args.label_builds:
        kw['label_builds'] = [x.strip() for x in args.label_builds.split(',')]
    if args.model_arms:
        kw['model_arms'] = ([] if args.model_arms.strip() == 'none' else
                            [x.strip() for x in args.model_arms.split(',')])
    run(data, config, args.output, args.dataset, **kw)


if __name__ == '__main__':
    main()
