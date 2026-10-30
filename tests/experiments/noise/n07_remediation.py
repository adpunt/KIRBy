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

HOW MANY MOLECULES A CLEANUP FLAGS. remove flags each share in FLAG_SHARES =
0.05, 0.10, 0.20 of the training molecules; knn_smooth flags 0.05 only, and on
a classification set the three neighbour cleanups flag 0.10 (f36 decisions,
2026-10-05, flag_shares_for). Filed as '<cleanup>@flag_share<s>'. Until 2026-09-28 they
flagged `noise_fraction`, which is also the share the injector gives the wide
error (or flips), so each cleanup was handed the injector's own count. Owner's
decision 2026-09-28: no method reads what only exists because the noise was
injected.

Compares the remediation strategies in CLEANUPS after noise detection. Since
2026-10-05 (f36, F36.md section 9) quantile_clip is out of the list and
constant_lift is in; the settings of every smoothing are the constants under
"f36 decisions" below. Two baselines
(no_noise, noisy_baseline) anchor recovery. All strategies keep the iteration's
downstream model fixed so the comparison is apples-to-apples within a column.

WHAT EACH LABEL IS MOVED TOWARD. Four of the seven rewrite labels, and until
2026-09-17 every one of them aimed at the same quantity: the mean label of the
ten nearest training molecules. label_smooth and knn_smooth still do.
oof_smooth aims at what a LightGBM model trained on the other training
molecules predicts for this one, and gp_smooth at the posterior mean of a
Gaussian process fitted the same way; both inner splits are scaffold-group
aware.

⛔ gp_smooth's TARGET WAS A CONSTANT PER INNER FOLD UNTIL 2026-09-21, not a
prediction for the molecule. On saprot, 448 training molecules and 1280
features, the 448 molecules received 3 distinct posterior-mean values and each
was exactly the mean label of the inner fold they were held out of. The cause
was the starting value of the RBF kernel's length scale, 1.0, against a median
distance between two standardised molecules of about 50: every off-diagonal
entry of the kernel matrix underflowed and the optimiser had no gradient to
follow. Starting it at the median distance instead
(src/kirby/utils/noise_estimation.py, GaussianProcessEstimator.estimate) gives
358 distinct values over the same 448 molecules and moves the target's rank
correlation with the clean labels from -0.0441 to 0.5630 spearman. Every result
for this arm measured before 2026-09-21 is a result for shrinkage toward the
inner fold's mean label, not for a Gaussian process.

quantile_clip aims at nothing — a quantile regression forest gives each
molecule the 10th-to-90th-percentile range of the labels its own leaves'
training molecules carry, a label inside that range is left byte-identical, and
a label outside is pulled onto the edge it crossed.

(Earlier versions also included a robust_loss strategy that swapped in
HuberRegressor, but that conflated "different cleanup strategy" with
"different model entirely" — its number wasn't comparable to the other
rows. Removed in the 2026-05-26 audit. If you want to evaluate Huber as
a noise-robust model, run a separate experiment with Huber as the
baseline across all strategies.)

WHAT CHANGED ON 2026-10-05 (chat 5, HANDOFF_EXPERIMENT_ALIGNMENT section
2.4). The paragraphs dated before this one describe the whole-dataset design;
where they disagree with this paragraph, this paragraph is what the code does.
  Models: the dataset's two decided models only (DECISIONS.md Step 1). The
    five noise-suite models, the rf-only per-representation block, the
    ranking of every model on every representation (rank_models_and_reps)
    and the rf and svm every D-versus-D' row added are gone.
  Configurations: per outer fold and decided model, its 2 best base
    representations and its 2 best general hybrids, ranked by searches on
    that fold's training rows (noise_helpers.fold_configurations). 2 models x
    4 = 8 fitted configurations. Each of the 8 runs the cleanups in
    _run_one_condition's strategy list: remove, downweight, label_smooth and
    knn_smooth, plus on regression the REGRESSION_ONLY_CLEANUPS still in
    CLEANUPS (quantile_clip is not, since f36). Those regression-only
    cleanups are skipped on a configuration wider than MAX_CLEANUP_FEATURES.
    huber_weight, prox_smooth and invvar_weight are in CLEANUPS but not in
    the list, so n07 does not run them (decided 2026-10-06, see the list).
  Neighbour spaces: the distance hybrid and the similarity hybrid of the same
    per-fold searches are neighbour spaces for the swept rows only, never a
    fitted configuration.
  Recipes: from the per-fold searches on CLEAN training labels. The features
    of each hybrid are re-chosen from that draw's DAMAGED training labels;
    the clean-label-features oracle arm is kept for the two hybrids.
  Noise: every decided kind at every decided level for the dataset's group
    (noise_helpers.decided_levels), 3 draws per non-zero level, level 0 once.
    Censoring runs one draw (noise_helpers.draws_for, 2026-10-06).
  Folds: noise_helpers.noise_outer_folds, any count including 1.
  Results: per level, per model, per configuration, per cleanup; each
    (fold, level, draw) checkpointed as it finishes.

WHAT CHANGED ON 2026-10-06 (Adelaide delegated the open decisions).
  as_they_come is no longer run: it repeated level 0 of every condition.
  On a classification dataset knn_smooth is not run; it wrote the same labels
    as label_smooth there.
  In the main table (by_level.<level>.scores and the recovery percentages),
    label_smooth and knn_smooth take their neighbours from each pass's top
    screened setting (screen_settings' kept[0]: the neighbour space and
    distance, k, and how much each neighbour counts), not from 9 Euclidean
    neighbours in the model's own features. So does the driving detector
    when it is a neighbour detector (cross_val, the one used, reads none).
    The setting used is recorded per (fold, level, draw, model,
    configuration) under main_table_neighbour_setting.
  Censoring runs one draw. The per-fold searches stay on clean labels.

WHICH FEATURES THE CEILING IS MEASURED IN. Until 2026-09-11 the clean-label
ceiling stayed in the clean-label feature space while every damaged arm had
moved to the noisy one, so part of the gap a recovery percentage divides by was
the feature space changing rather than the noise. The ceiling is fitted inside
each noise draw's own feature space -- clean training labels, no injection,
only the features differ -- and that is what 'no_noise' holds under
by_level.<level>.scores and what every recovery percentage divides by. The
clean-label-feature ceiling is still reported, under
folds.<fold>.no_noise_clean_features, so the two can be compared.

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
from experiments.dataset_policy import noise_dose_units
from kirby.noise_spec import GROUPED_STRATEGIES
from experiments.noise.noise_cache import (fit_takes_sample_weight,
                                           require_weights_accepted,
                                           WEIGHTS_NOT_RUN)
from experiments.noise.noise_helpers import (inject_train_labels,
                                            noise_spec_for,
                                            detection_scores, sweep_conditions,
    conditions_for_task)
# 2026-10-05 (chat 5): the whole-dataset readers this file imported
# (get_precomputed_search, get_best_search_for_downstream, get_rule_hybrids,
# pool_recipes_by_family, extract_top3_base_reps, DOWNSTREAM_MODELS_NOISE,
# fold_reselection, build_fold_reselection_input, top_base_reps, top_hybrids,
# hybrid_deltas) are gone; each fold's configurations come from
# noise_helpers.fold_configurations instead. cv_split stays for
# rank_models_and_reps, which run() no longer calls.
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results,
    cv_split, get_domain_metric, get_model_factories,
)
from kirby.utils.noise_estimation import (
    ESTIMATOR_REGISTRY, ActivityCliffAwareEstimator, GaussianProcessEstimator,
    QuantileForestEstimator, smooth_by_scores, weights_from_scores,
    clip_to_range,
)
from kirby.utils.noisy_labels import oof_predict

# The best general hybrid's configuration name. SPACE_FOR_FAMILY, which mapped
# the whole-dataset pool's two search families onto spaces, went with the pool
# on 2026-10-05 (chat 5).
HYBRID_SPACE = 'hybrid'


# Multi-seed noise averaging. Each fold runs noise injection at multiple
# seeds and aggregates — reduces dependence on a single noise realization.
NOISE_SEEDS = [0, 1, 2]

# Every cleanup this file can apply, i.e. every strategy that is not one of the
# two baselines. Named once here because `_evaluate_fold` used to carry the
# same four-name tuple written out three times, and adding a fifth meant
# finding all three.
CLEANUPS = ('remove', 'downweight', 'label_smooth', 'knn_smooth',
            'oof_smooth', 'gp_smooth',
            # quantile_clip was here until 2026-10-05. Owner's decision from
            # f36 (tests/experiments/framework/F36.md section 9): 370 s a
            # fold for a quantile forest and never more than 0.012 from no
            # smoothing on any of 14 (dataset, input, condition) pairs. The
            # branch below still runs when a caller names it.
            # Added 2026-09-23 from the label-noise menu in the compass review.
            # Both keep every molecule, which is the family that review
            # recommends over deletion on small datasets.
            'huber_weight', 'js_shrink', 'prox_smooth',
            # Added 2026-09-23. The only cleanup that reads no suspicion score
            # and no label: the weight comes from how many times the molecule
            # was measured. See SCORE_FREE_CLEANUPS below.
            'invvar_weight',
            # Added 2026-10-05 from f36 (F36.md section 9). n30's constant
            # lift: the training labels sitting at the ceiling the labels
            # themselves show are raised by one constant, limit + the mean
            # Tobit lift over those rows. Nothing else moves. It declines on
            # a training block whose labels show no ceiling, so it runs on
            # every regression dataset and does something only under
            # censoring. f36: logd 0.102 rmse better than no smoothing,
            # koenig 0.021 spearman better, on both model inputs.
            'constant_lift')

# ---------------------------------------------------------------------------
# f36 decisions, 2026-10-05 (tests/experiments/framework/F36.md section 9).
# Every smoothing setting below was measured on logd, flab_expression_koenig
# and herg, SVM, two folds, against no smoothing. These are the settings the
# noise suite carries; nothing here is swept.
# ---------------------------------------------------------------------------
# label_smooth, oof_smooth, gp_smooth: how far a label moves, times its
# rescaled suspicion score. 0.25 lost at most 0.004 against 0.008 at 0.5,
# with the same gains.
SMOOTH_STRENGTH = 0.25
# js_shrink: the per-molecule variance multiplier. 0.5 lost at most 0.005
# against 0.011 at 1.0.
JS_SHRINK_STRENGTH = 0.5
# knn_smooth on a regression set: the flagged labels are replaced outright by
# their 9 neighbours' mean, so only the smallest flag share is carried. 0.10
# and 0.20 lost 0.04 to 0.11 rmse on every logd pair.
KNN_SMOOTH_FLAG_SHARES = (0.05,)
# prox_smooth on a regression set: EVERY label moves PROX_SMOOTH_SHARE of the
# way toward the mean label of its PROX_SMOOTH_K forest-neighbours, once, and
# no detector is read. Ahead of no smoothing on 9 of 14 pairs and never more
# than 0.005 from it; the old form (the flagged share replaced outright by the
# 10 forest-neighbours' mean) lost 0.003 to 0.023 rmse on logd.
PROX_SMOOTH_K = 5
PROX_SMOOTH_SHARE = 0.25
# Classification: the three neighbour cleanups replace the flagged labels by
# the majority class of the CLF_NEIGHBOURS nearest molecules by
# CLF_NEIGHBOUR_METRIC (prox_smooth: the forest's nearest), at CLF_FLAG_SHARE.
# herg: +0.001 auc; today's 9 by Euclidean -0.002; share 0.2 -0.011.
CLF_NEIGHBOURS = 5
CLF_NEIGHBOUR_METRIC = 'manhattan'
CLF_FLAG_SHARES = (0.10,)
# Regression label_smooth and knn_smooth keep today's 9 Euclidean neighbours.
REG_NEIGHBOURS = 9

# The cleanups that read NO per-molecule suspicion score, so a caller may hand
# them None where the detector would go.
#
#   quantile_clip  asks whether a label lies outside the range of label values
#                  molecules like it carry, which is a question about the label
#                  and the range.
#   invvar_weight  asks how many times the molecule was measured, which is a
#                  fact about the record and not about the label at all.
#
# ⛔ WRITTEN OUT RATHER THAN DERIVED. This was `tuple(c for c in CLEANUPS if c
# != 'quantile_clip')`, so every name added to CLEANUPS became score-driven by
# default. `invvar_weight` takes no score, and under that spelling n07 and n18
# would have fitted a detector for it and `_evaluate_fold` would have dropped
# the whole fold whenever the detector failed -- for a weight the detector
# never touches.
SCORE_FREE_CLEANUPS = ('quantile_clip', 'invvar_weight',
                       # Reads the labels and the out-of-fold prediction only.
                       'constant_lift')

# The cleanups that read the detector's per-molecule suspicion score.
SCORE_DRIVEN_CLEANUPS = tuple(c for c in CLEANUPS
                              if c not in SCORE_FREE_CLEANUPS)

# How many training molecules a thresholding cleanup flags: each share here,
# swept, the same at every condition and level. n24's FLAG_RATES.
#
# ⛔ Until 2026-09-28 the flag share was `noise_fraction`, the SAME number
# _noise_spec writes into the outlier condition's p. So remove, knn_smooth and
# prox_smooth flagged exactly the share the injector gave the wide error, and
# on classification exactly the share it flipped. A real user knows neither.
# Owner's decision 2026-09-28: no method may read what only exists because the
# noise was injected. noise_fraction now reaches the injector only.
FLAG_SHARES = (0.05, 0.10, 0.20)

# The cleanups whose cleaned set depends on the flag share: the three that
# threshold the suspicion score (n25.THRESHOLDING_CLEANUPS), and label_smooth
# on a classification set, which thresholds there and nowhere else. Every
# other cleanup reads no quantile, so a flag share would sweep nothing.
# network_rewrite and ema_rewrite (2026-10-30) replace the flagged share too.
FLAG_SHARE_CLEANUPS = ('remove', 'knn_smooth', 'network_rewrite', 'ema_rewrite')

# The cleanup halves of the in-loop training methods, split out on
# 2026-10-30 (HANDOFF_NOISE_RESTRUCTURE.md step 2; kirby.utils.lnl_parts).
# NOT in CLEANUPS: this file's own sweep is unchanged, and these run when a
# caller names them. Each is a branch of cleaned_training_set below.
#
#   score_weight          the weight is one minus the score. The method's own
#                         weight when the detector returns one minus a weight
#                         (net_meta_weight: Meta-Weight-Net; early learning's
#                         decayed weight is this shape too). Declines on a
#                         score outside [0, 1].
#   spread_invvar_weight  the weight is one over the score, for a detector
#                         whose score is a variance (hetero_gp, ngboost,
#                         heteroscedastic_nn, net_evidential_spread). The
#                         evidential network's and the heteroscedastic GP's
#                         own weighting. Unlike invvar_weight it reads the
#                         score, not a measurement count.
#   network_rewrite       DivideMix's action, once: the flagged labels are
#                         replaced by the prediction of the suite's DNN
#                         trained on the unflagged molecules.
#   ema_rewrite           Jo-SRC's action, once: the flagged labels are
#                         replaced by the prediction of the exponential moving
#                         average copy (omega 0.99) of a DNN trained on every
#                         molecule for 30 epochs.
SPLIT_CLEANUPS = ('score_weight', 'spread_invvar_weight', 'network_rewrite',
                  'ema_rewrite')

# The cleanups that threshold on a CLASSIFICATION set only: there a label is
# a class, so "move it part of the way" is undefined and the flagged share is
# replaced by the neighbours' majority class instead.
CLF_THRESHOLD_CLEANUPS = ('label_smooth', 'knn_smooth', 'prox_smooth')


def reads_flag_share(cleanup, is_clf):
    """Does this cleanup's cleaned training set depend on the flag share?"""
    return (cleanup in FLAG_SHARE_CLEANUPS
            or (bool(is_clf) and cleanup in CLF_THRESHOLD_CLEANUPS))


def flag_shares_for(cleanup, is_clf):
    """The flag shares one cleanup is run at (f36 decisions, 2026-10-05).
    `remove` was not measured by f36 and keeps FLAG_SHARES."""
    if is_clf and cleanup in CLF_THRESHOLD_CLEANUPS:
        return CLF_FLAG_SHARES
    if cleanup == 'knn_smooth':
        return KNN_SMOOTH_FLAG_SHARES
    return FLAG_SHARES


def flag_share_label(cleanup, share):
    """The strategy name one (cleanup, flag share) is filed under."""
    return f'{cleanup}@flag_share{float(share):g}'


def split_strategy(label):
    """(cleanup, flag share or None) from a strategy name."""
    head, sep, tail = str(label).partition('@flag_share')
    return (head, float(tail)) if sep else (head, None)


def expand_flag_shares(strategies, is_clf, shares=None):
    """Each cleanup that reads the flag share once per share, in place.
    shares=None takes each cleanup's own shares (flag_shares_for); a tuple
    applies to every cleanup that reads one."""
    out = []
    for st in strategies:
        if reads_flag_share(st, is_clf):
            out += [flag_share_label(st, sh) for sh in
                    (flag_shares_for(st, is_clf) if shares is None
                     else shares)]
        else:
            out.append(st)
    return out

# Which cleanups read a target that `correction_targets` can hoist, and which
# target each reads. ⛔ DECLARE A NEW CLEANUP HERE. Callers used to write this
# mapping out by hand -- n18 carried its own three-pair list -- so js_shrink,
# added 2026-09-23, was built and then never given a target by anyone. A
# cleanup that silently receives None looks broken rather than untested.
#
# Every one of these can also build its own target when the caller hoists
# none; the hoist exists because the target depends on the features, labels
# and groups and on nothing else, so sweeping settings would otherwise refit
# the same model for every one of them.
#
# ⛔ `invvar_weight` IS NOT HERE AND CANNOT BE. Every target below is built by
# `correction_targets` from the features, the labels and the scaffold groups.
# A measurement count is not in any of those three: it comes from the loader's
# replicate grouping, which says which ROWS are the same molecule. So it
# arrives as its own argument, `measurement_counts`, and
# `measurement_counts_for_rows` below is where it is read off the data dict.
CLEANUP_TARGETS = {
    'oof_smooth': 'oof',
    'gp_smooth': 'gp',
    'quantile_clip': 'quantile',
    'js_shrink': 'oof',          # same out-of-fold prediction as oof_smooth
    'constant_lift': 'oof',      # the same prediction gives the Tobit lift
}


def targets_needed(cleanups):
    """The hoisted targets a list of cleanups needs between them."""
    return sorted({CLEANUP_TARGETS[c] for c in cleanups if c in CLEANUP_TARGETS})

# The three cleanups added 2026-09-17. All three repair a REGRESSION label:
# 'oof_smooth' and 'gp_smooth' move it toward a predicted value, 'quantile_clip'
# pulls it back into a numeric range. None of the three is defined on a nominal
# class label, so on a classification dataset they are not added to the
# strategy list at all and the run records that rather than writing NaN rows.
#
# `invvar_weight` is NOT here, for the reason `huber_weight` is not: it sets a
# row weight and never touches the label, and a row weight is defined whatever
# the label is. Nothing about a measurement count needs the label to be a
# number. It will still decline on most datasets, because the loader kept no
# per-molecule measurement count for them -- that is a different refusal and it
# says so in its own words.
REGRESSION_ONLY_CLEANUPS = ('oof_smooth', 'gp_smooth', 'quantile_clip',
                            # js_shrink blends a label toward a predicted
                            # NUMBER; there is no halfway point between two
                            # nominal classes, so it is regression-only too.
                            'js_shrink',
                            # A ceiling is a number.
                            'constant_lift')

# How many inner folds the out-of-fold prediction that 'oof_smooth' smooths
# toward is built from, and how many the two forests behind 'gp_smooth' and
# 'quantile_clip' use. Five, matching every other out-of-fold split in the noise
# suite. The inner split is group-aware wherever the caller has scaffold groups.
REPAIR_OOF_SPLITS = 5

# The most training molecules `gp_smooth` will fit a Gaussian process on.
# n11 caps the same estimator at GP_MAX = 1000 for the same reason and this
# path had no cap at all. Above it the target is refused with its reason on the
# result rather than fitted on a subsample, because a target fitted on fewer
# molecules is a different quantity and a cleanup silently aiming somewhere
# else is worse than one that says it did not run.
GP_SMOOTH_MAX_ROWS = 1000

# `prox_smooth` builds a molecules-by-molecules proximity, so its memory is
# quadratic where every other cleanup's is linear: 16 MB at 2,000 training
# molecules, 400 MB at 10,000, about 850 MB at nt_H3K4me3's 14,600. It used to
# be refused above 2,000 molecules. Owner's decision 2026-09-25: the limit goes,
# because the jobs have the memory and a refusal removes the cleanup from every
# large dataset. Kept as a name only so old imports do not break; nothing
# reads it.
PROX_SMOOTH_MAX_ROWS = None

# The smallest clean-to-noisy gap a recovery PERCENTAGE may divide by, as a
# fraction of the clean score's own magnitude. Same rule and same number as
# n11 (n11_estimation_remediation_grid.GAP_MIN_FRAC), and it is here for the
# same reason: recovery divides by (clean - noisy), so when the injected noise
# barely moved the model that denominator is a rounding error and the ratio
# explodes. tests/results/noise/n07_remediation/buchwald_hartwig.json on disk
# has a gap of 0.224 rmse against a clean score of 14.35 -- 1.6% -- and reports
# recoveries of +350% and -474% off it. Below the floor the recovery is NaN
# and the reason is written to results['recovery_undefined'], which is what the
# sign guard beside it already does.
GAP_MIN_FRAC = 0.02


# The private `_inject_noise` this module used to carry has been deleted. It picked a
# `fraction` of rows at random and gave each of them Gaussian noise at 0.3 of the label
# RANGE -- a fixed magnitude with no level axis, and a fraction of the range rather than
# of the SD, so it ran at several times the level any other experiment meant by the same
# number. That IS the Outlier condition, so it becomes one: `outlier` with p set from
# `noise_fraction`, at a dose. Selection stays random, which is the point of the
# condition -- a mistyped unit is a property of the record, not of the value.


# The condition n07 injected for the life of the project, and the only one it
# could. Kept as the default so a caller that names none behaves as before; the
# sweep in run() below is what makes the others reachable.
DEFAULT_CONDITION = 'outlier_p05'

# DECIDED 2026-10-06 (Adelaide delegated): n07's run() no longer sweeps this
# condition; it repeated level 0 of every injected condition. The name stays
# because _noise_spec and _evaluate_fold (shared with n18 and n25) read it.
# 2026-09-27, owner's decision: n07 also runs on the labels as recorded. Under
# this condition name nothing is injected at all -- not a zero-level draw, which
# on classification still flips a few labels (calibrate_flip_probability has a
# floor). The 'noisy' training labels ARE the recorded ones, so every cleanup's
# D-versus-D' row answers: does it help on the dataset's own experimental
# noise. 'no_noise' and 'noisy_baseline' are then the same fit, and every
# recovery percentage is NaN by the GAP_MIN_FRAC floor.
AS_THEY_COME = 'as_they_come'


# Scaffold groups for the grouped noise conditions, one entry per dataset, set
# at the top of _run_one_condition from dataset_policy.noise_groups_for(data).
# grouped_shifted and grouped_wider RAISE without them (noise_spec.py:407-412),
# and none of n07's four injection calls passed any -- so n07 crashed on both
# grouped conditions of its own sweep until 2026-09-29. n24 already passes
# them the same way (n24:733, :1041-1043).
_NOISE_GROUPS = {}


def _inject_or_recorded(y_all, spec, *, train_idx, is_clf, dataset_name):
    """inject_train_labels, or the recorded labels when spec is None
    (AS_THEY_COME). Same return shape: (y_train, moved_mask, provenance)."""
    if spec is None:
        y_tr = np.asarray(y_all)[np.asarray(train_idx)].copy()
        return (y_tr, np.zeros(len(y_tr), dtype=bool),
                {'condition': AS_THEY_COME, 'injected': False})
    return inject_train_labels(y_all, spec, train_idx=train_idx,
                               is_clf=is_clf, dataset_name=dataset_name,
                               groups=_NOISE_GROUPS.get(dataset_name))


def _noise_spec(dataset_name, noise_fraction, level, noise_seed, is_clf=False,
                condition=DEFAULT_CONDITION):
    """One injected condition at this dataset's WIDE-ERROR fraction.

    ⛔ `condition` ADDED 2026-09-23. This pinned 'outlier_p05' and nothing
    could change it, so n07 -- the experiment that compares the cleaning
    methods against each other -- ran ONE noise condition at ONE level for the
    life of the project. Everything the suite knows about how the cleaning
    methods behave under different noise had to come through here and could
    not be asked.

    `noise_fraction` NO LONGER OVERRIDES p (removed 2026-09-30, see the body).
    Until then: p is the
    share of training molecules given an error about three times as wide as
    every other molecule's, and it was the knob n07's own design turned. The
    other conditions have no p: gaussian and the heavy tails give every
    molecule the same width, grouped_wider takes a group fraction, censoring
    takes a clipped fraction. Forcing p onto them would either raise or be
    silently ignored, so it is applied only where it means something and the
    registry's own parameters stand everywhere else.

    `noise_fraction` is the share of training molecules that receive an error
    about three times the width every other molecule's receives -- Huber
    contamination at lambda 3. It is NOT the share that is corrupted, which on a
    regression dataset is all of them. Calling it the corrupted fraction is what
    this docstring did until 2026-09-20 and it makes the three cleanups that
    rewrite only the flagged molecules look like they are addressing the whole
    injury; see the note in `_evaluate_fold`.

    For classification the level IS the flip rate and the fraction flipped is
    the level, because a flipped class label has no magnitude to dose-match.
    There the fraction really is the corrupted share.

    `fold_idx` was removed 2026-09-12: the fold must not reach the seed, or a
    molecule's corruption depends on which fold it lands in. See
    noise_helpers.noise_spec_for.
    """
    if condition == AS_THEY_COME:
        return None
    if is_clf:
        return noise_spec_for(dataset_name, 'uniform', noise_fraction,
                              noise_seed=noise_seed, is_clf=True)
    # ⛔ THE p OVERRIDE WAS REMOVED 2026-09-30 (owner: every experiment injects
    # noise the same way). It forced p = noise_fraction (0.10) onto every
    # outlier condition, so outlier_p01 and outlier_p05 -- and the two
    # outlier_only ones -- were injected with 10% wide errors under names that
    # say 1% and 5%. Each condition now carries its own registry p, exactly as
    # noise_spec_for gives it to every other experiment. noise_fraction still
    # sets how many molecules the cleanups flag; it no longer touches the noise.
    return noise_spec_for(dataset_name, condition, level,
                          noise_seed=noise_seed)


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


# ⛔ WHICH DETECTOR DECIDES WHAT IS SUSPECT. Until 2026-09-18 this file fitted
# ActivityCliffAwareEstimator and nothing else, in five places, with no name to
# change and nothing to compare it against.
#
# Two things say that was the wrong one to be stuck with. It is the detector
# n11, n13 and n06 CUT on 2026-09-12 for being neighbor_residual in a second
# name -- measured again here on all three local datasets, Spearman 0.9843 to
# 0.9888 over every molecule. And on n18, over the same molecules and folds,
# swapping it for `cross_val` moved `downweight` from 4 of 4 settings winning a
# majority to 9 of 12, and supplied the best setting for both cleanups that win.
#
# `cross_val` is a different kind of evidence: how badly a model trained on the
# OTHER molecules predicts this label, out of fold on scaffold-disjoint folds,
# rather than the label judged against its neighbours' labels.
#
# GRADED, not assumed. n07 injects the noise, so it knows which molecules were
# corrupted, and it already grades a detector with noise_helpers.detection_scores.
# Both names below are scored that way every fold -- two estimator fits and no
# model fits -- so this run says which one FINDS the corrupted molecules while
# DETECTOR says which one drove the cleanups.
DETECTORS_GRADED = ('cross_val', 'activity_cliff_aware')
DETECTOR = 'cross_val'


def _compute_noise_scores(X_tr_s, y_tr, is_clf=False, detector=None,
                          groups=None, cache_dataset=None):
    """Fit the named detector on (X_tr_s, y_tr) and return its noise scores.

    cache_dataset, added 2026-09-25 (plan section 3): a dataset name routes the
    call through noise_cache.cached_detector_scores, keyed on the content of
    X_tr_s, y_tr and groups and on the detector, which calls this function
    back with cache_dataset=None -- so the scores are the ones computed below,
    stored once per (features, labels, detector, groups). n07 computes the same
    scores in its hybrid block and its per-representation block and again for
    every noise condition at level 0. None, the default, computes here.

    Hoisted out of _evaluate_fold so the caller can compute once per (fold,
    noise_seed) and reuse across the strategies that need it (every cleanup
    except quantile_clip). Previously each strategy refit
    the same estimator on the same data — ~4× wasted work per fold.
    Returns None if the estimator fit fails.

    is_clf threads into the estimator so classification datasets use its
    class-disagreement noise-scoring branch instead of the regression
    neighbor-mean / local-variance path (which divides integer class labels
    by their range — meaningless on nominal labels). Mirrors the per-rep /
    hybrid is_clf threading audited as P0#1.
    """
    import inspect
    name = DETECTOR if detector is None else detector
    if cache_dataset is not None:
        from experiments.noise.noise_cache import cached_detector_scores
        return cached_detector_scores(cache_dataset, X_tr_s, y_tr,
                                      is_clf=is_clf, detector=name,
                                      groups=groups)
    cls = ESTIMATOR_REGISTRY[name]
    # Each estimator takes its own arguments. Handing every one `k` and
    # `metric` is what kept CrossValResidualEstimator, which takes neither,
    # out of this file and out of n18 (fixed there the same day).
    # ⛔ THE GROUPS HAVE TWO NAMES and offering only one refused six estimators.
    # The model-based family (cross_val, gaussian_process, conformal, ngboost
    # and eleven others) names the scaffold array `groups`. The neighbour family
    # (neighbor_residual, local_variance_ratio, distance_weighted,
    # activity_cliff_aware, leave_one_out) names the SAME array
    # `exclude_groups`, because for them it is what a molecule's neighbour list
    # must not contain rather than what a fold must not straddle. Every
    # estimator in ESTIMATOR_REGISTRY takes one or the other; none takes
    # neither. Offering both means the undeliverable check below fires only
    # when an estimator genuinely cannot receive them.
    #
    # Before this, `_smoke_n04_n07` failed outright: DETECTORS_GRADED names
    # activity_cliff_aware, which takes `exclude_groups`, so the check raised
    # TypeError and the whole of n07 stopped.
    # Axis 6 of KNN_SEARCH_SPACE_RECORD (Adelaide, 2026-10-05): a molecule's
    # scaffold-mates stay in its neighbour list. The neighbour family is
    # therefore handed no exclude_groups; the model-based family still gets
    # `groups` for its inner folds.
    offered = {'k': 10, 'metric': None, 'is_clf': is_clf,
               'groups': groups, 'exclude_groups': None}
    accepts = set(inspect.signature(cls.__init__).parameters)
    # ⛔ A FLAG THIS FILTER CANNOT DELIVER IS AN ERROR, NOT A SHRUG.
    #
    # The filter keeps whatever the estimator's constructor happens to name
    # and DROPS the rest with nothing printed. `is_clf` was one of the dropped
    # ones: CrossValResidualEstimator, which DETECTOR names, took no such
    # argument until 2026-09-20, and its `estimate` fitted a
    # RandomForestRegressor whatever it was handed. So on a classification
    # dataset a molecule's suspicion score was the distance from its class
    # NUMBER to a regressor's prediction of that number, while the docstring
    # above said the flag threaded through to a class-disagreement branch. The
    # two disagreed and nothing said so.
    #
    # `groups` sits in the same filter and the same argument applies: an
    # estimator that cannot hold the scaffold groups splits its inner folds at
    # random, and a molecule's own scaffold family then trains the model that
    # predicts it.
    #
    # This raises OUTSIDE the try below on purpose. Inside it the refusal would
    # be caught, printed and turned into None, which is what a genuinely failed
    # fit returns -- a configuration that can never be right would then be
    # indistinguishable from a fold that happened not to converge.
    #
    # Only a flag that would CHANGE something is refused. is_clf=False and
    # groups=None ask for nothing, so an estimator that names neither is used
    # exactly as it was.
    # `groups` counts as delivered if the estimator accepts EITHER name.
    undeliverable = [k for k, v in (('is_clf', is_clf), ('groups', groups))
                     if v is not None and v is not False
                     and not (k in accepts
                              or (k == 'groups' and 'exclude_groups' in accepts))]
    if undeliverable:
        raise TypeError(
            f"detector {name!r} ({cls.__name__}) takes no "
            f"{' or '.join(undeliverable)} argument, so it would be dropped "
            f"and the scores computed as though it had never been asked for.")
    try:
        est = cls(**{k: v for k, v in offered.items() if k in accepts})
        est.fit(X_tr_s, y_tr)
        return est.noise_scores_
    except Exception as exc:
        print(f"      noise_scores estimator failed: "
              f"{type(exc).__name__}: {exc}")
        return None


def correction_targets(X_tr_s, y_tr, groups=None, want=None):
    """The three model-based corrections' targets, fitted ONCE.

    ⛔ WHY THIS EXISTS. `oof_smooth`, `gp_smooth` and `quantile_clip` each fit a
    model on part of the training rows and read a target off the rest, and each
    one refitted it on EVERY call to cleaned_training_set. The target depends on
    the features, the labels and the groups and on nothing else -- not on the
    suspicion score, not on the strength, not on which downstream model is
    about to be fitted. So a caller sweeping settings refits the same model for
    every one of them.

    In n18 that is 3 strength settings x 2 detectors x 2 models = 12 identical
    fits per (fold, space) where 1 would do, for each of the three. The
    Gaussian process is the one that hurts: it is cubic in the training rows
    and fits REPAIR_OOF_SPLITS inner folds per call, so on herg_ki it was 60 GP
    fits per (fold, space) against 5.

    This is the same argument `_compute_noise_scores` carries for the DETECTOR
    ("~4x wasted work per fold"), which was hoisted for it and not for these,
    and it is what n11 already does with its EXTRA_TARGETS -- computed once per
    (fold, draw, space) and handed to every estimator.

    Returns {name: value} for the names in `want`, each None when its fit
    failed, so a caller passes the dict straight to cleaned_training_set and a
    failure drops that one cleanup rather than the fold.
    """
    want = set(want or ('oof', 'gp', 'quantile'))
    out = {}
    n_splits = min(REPAIR_OOF_SPLITS, len(y_tr))
    if 'oof' in want:
        try:
            out['oof'] = oof_predict(X_tr_s, y_tr, groups, n_splits=n_splits)
        except Exception as exc:
            print(f"      oof target failed: {type(exc).__name__}: {exc}")
            out['oof'] = None
    if 'gp' in want:
        try:
            if len(y_tr) > GP_SMOOTH_MAX_ROWS:
                # ⛔ A GAUSSIAN PROCESS IS CUBIC IN THE TRAINING ROWS AND THIS
                # ONE HAS NO CAP. GaussianProcessEstimator takes no max_samples
                # and runs n_restarts=3, so one call is three hyperparameter
                # optimisations of an O(n^3) fit, REPAIR_OOF_SPLITS times over.
                # n11 caps the same estimator at GP_MAX = 1000 and this path
                # did not, which is why gp_smooth did not finish a fold on
                # herg_ki's 1,130-molecule training blocks.
                #
                # Refusing is the honest move rather than subsampling: a target
                # fitted on a subsample is a different quantity from one fitted
                # on the fold, and a cleanup silently aiming at a different
                # target is worse than a cleanup that says it did not run.
                out['gp'] = None
                out['gp_refused_because'] = (
                    f'{len(y_tr)} training molecules is over '
                    f'GP_SMOOTH_MAX_ROWS={GP_SMOOTH_MAX_ROWS}; a Gaussian '
                    f'process is cubic in the rows and this one has no cap')
            else:
                est = GaussianProcessEstimator(n_folds=REPAIR_OOF_SPLITS,
                                               groups=groups)
                est.estimate(X_tr_s, y_tr)
                out['gp'] = np.asarray(est.predictions_, dtype=float)
        except Exception as exc:
            print(f"      gp target failed: {type(exc).__name__}: {exc}")
            out['gp'] = None
    if 'quantile' in want:
        try:
            est = QuantileForestEstimator(n_folds=REPAIR_OOF_SPLITS,
                                          groups=groups)
            est.estimate(X_tr_s, y_tr)
            out['quantile'] = (np.asarray(est.quantile_lo_, dtype=float),
                               np.asarray(est.quantile_hi_, dtype=float))
        except Exception as exc:
            print(f"      quantile target failed: {type(exc).__name__}: {exc}")
            out['quantile'] = None
    return out


def measurement_counts_for_rows(data):
    """How many times each row's molecule was measured, one entry per ROW.

    WHERE THE COUNT COMES FROM, AND IT IS n19's PATH. n19 reaches a
    per-molecule measurement count in two steps:
    `_ground_truth_providers.build_replicate_groups(data)` reads the loader's
    replicate grouping -- which rows are measurements of the same molecule --
    and `n19_measurement_count.measurement_stats` counts the rows in each
    group. Both are called here rather than counted a second way, so the suite
    has one definition of a measurement count.

    A loader hands that grouping back only when it kept one row per
    MEASUREMENT. `tests/experiments/_shared.py` writes `replicate_groups` into
    the data dict for exactly those datasets, and herg_ic50 is the only
    molecular one today. Every other dataset averaged a molecule's repeats into
    one number before an experiment saw it, so there is no count to read and
    this returns None.

    `measurement_stats` is per MOLECULE, in first-row order; this broadcasts it
    back so every row of a molecule carries that molecule's count, because
    `cleaned_training_set` is handed training ROWS.

    ⛔ WHICH REGIME THE CALLER IS IN, because the two need different columns
    and this one builds only the first.

      ONE ROW PER MEASUREMENT, which is what `load_data` hands back for
      herg_ic50: this column is right, and a molecule measured 4 times is 4
      rows each carrying the count 4. Under `invvar_weight` that molecule then
      pulls on the fit 16 times as hard as a molecule measured once, not 4
      times, because the rows are counted as well as weighted. Whether that is
      the intended weighting is a question for whoever runs it.

      ONE ROW PER MOLECULE, which is what `n17_replicate_removal.build_entities`
      produces and what n19 measures in: the label there is already the mean of
      the molecule's measurements, so the count belongs in ENTITY order and
      this column, which is in row order, is the wrong length. Read
      `measurement_stats(...)['n_measurements']` directly for that.
    """
    # n19 imports n17, which imports this file, so both imports are local. A
    # module-level one is a cycle.
    from experiments.noise._ground_truth_providers import build_replicate_groups
    from experiments.noise.n19_measurement_count import measurement_stats

    rep_groups = build_replicate_groups(data)
    if rep_groups is None:
        return None
    rep_groups = np.asarray(rep_groups)
    stats = measurement_stats(np.asarray(data['labels'], dtype=np.float64),
                              rep_groups)
    count_by_group = {rep_groups[first]: int(k) for first, k
                      in zip(stats['row_index'], stats['n_measurements'])}
    return np.array([count_by_group[g] for g in rep_groups], dtype=np.float64)


# Why the last cleaned_training_set call returned None (2026-10-08, A19).
# Before this, a caller saw only None, and n18 and n20 stored either nothing or
# one message listing every refusal the function has. Set by _declined, cleared
# at the start of every call; read it right after the call that returned None.
_DECLINE = {'reason': None}


def _declined(reason):
    _DECLINE['reason'] = str(reason)
    return None


def last_decline_reason():
    """The reason the most recent cleaned_training_set call declined, or None
    when it did not decline (or raised)."""
    return _DECLINE['reason']


def clear_decline_reason():
    _DECLINE['reason'] = None


def cleaned_training_set(strategy, X_tr_s, y_tr, noise_scores, noise_fraction,
                         is_clf=False, groups=None, strength=None,
                         targets=None, measurement_counts=None,
                         dataset_name=None, similarity_neighbours=None,
                         similarity_k=None, similarity_scheme=None):
    """What one of the cleanups hands the downstream fit, before any fit runs.

    similarity_scheme: how much each neighbour counts when label_smooth,
    knn_smooth or prox_smooth read their neighbours from similarity_neighbours
    (KNN_SEARCH_SPACE_RECORD axis 5: 'rank' or 'inverse_square',
    similarity_space.SWEPT_NEIGHBOUR_SCHEMES). None reads the 'scheme' key the
    sweep puts on the lists it hands out (similarity_sweep); when that is
    absent too, every neighbour counts equally, as before 2026-10-05.

    similarity_neighbours / similarity_k, added 2026-09-25 (plan section 1:
    the features neighbours are found in is its own axis). None, the default,
    is the unchanged code path: label_smooth, knn_smooth and prox_smooth find
    neighbours in X_tr_s exactly as before. Given the dict
    similarity_space.neighbour_lists returns, label_smooth and knn_smooth read
    their neighbours from it instead of fitting make_nearest_neighbors on
    X_tr_s, and prox_smooth does too when the setting is forest_proximity.
    similarity_k is how many neighbours; None keeps today's count, which is
    NOT the same for the three: label_smooth and knn_smooth query
    min(10, n-1) rows including the molecule itself and drop the first, so 9
    neighbours; prox_smooth takes min(10, n-1), so 10. Handing in the
    MODEL_INPUT setting's lists reproduces today's numbers bit for bit
    (tests/smoke/smoke_noise_similarity_foundation.py).

    Hoisted out of `_evaluate_fold` on 2026-09-17 so that `n20` can apply the
    same cleanup to the labels the RECIPE SEARCH sees, not only to the labels the
    model is fitted on. Nothing about what each cleanup does changed: the first
    four branches below are the four branches `_evaluate_fold` carried, and
    `_evaluate_fold` now calls this and fits whatever comes back. Three more
    branches were added on 2026-09-17; n20 does not ask for them (its CLEANUPS
    tuple names its own four) and nothing about the first four moved.

    noise_fraction is the share of training rows treated as suspect, so the
    cut-off sits at the (1 - noise_fraction) quantile of `noise_scores`.

    strength is how HARD the two rate-blind cleanups lean on that score, in
    each one's own units, and None keeps the value this function used before it
    was a parameter. It exists because `downweight` and `label_smooth` read no
    quantile at all on a regression set, so sweeping the flag rate for them
    sweeps nothing:

      downweight    `weights_from_scores(..., strength=s)` with the 'inverse'
                    shape gives a row weight of 1/(1+s) at the most suspect end
                    and 1 at the cleanest. The old fixed 1.0 is therefore the
                    gentlest setting the shape has -- a 2:1 ratio between the
                    most and least suspect molecule. 3.0 is 4:1 and 9.0 is 10:1.
      label_smooth  the alpha `smooth_by_scores` moves a label by, 0 to 1.
                    The old fixed 0.5 moves the most suspect label half of the
                    way to its neighbours' mean.

    Nothing in the suite had ever tried a hard setting: n11 sweeps this same
    knob over ALPHAS = [0.1, 0.3, 0.5, 0.7], every one of which is gentler than
    or equal to the fixed value here.

    Returns a dict, or None when the cleanup cannot be applied:

      X               the feature matrix to fit on
      y               the labels to fit on, in the same order as X
      sample_weight   a per-row weight, or None
      keep            a boolean mask over the training rows, or None when the
                      cleanup keeps every row
      label_column    the FULL-LENGTH cleaned label column, or None when this
                      cleanup does not produce one. `remove` drops rows and
                      `downweight` sets weights, so neither has one.

    `label_column` is what a caller hands to find_best_hybrid(train_labels=...).
    `sample_weight` reaches no feature selection at all: `sample_weight` appears
    zero times in src/kirby/hybrid.py and zero times in
    src/kirby/find_best_hybrid.py, checked 2026-09-17.

    measurement_counts: how many times each TRAINING row's molecule was
    measured, one entry per training row, or None. Only `invvar_weight` reads
    it, and it declines when it is None. `measurement_counts_for_rows` above
    builds the whole column from the data dict; a caller slices it to the
    fold's training rows.

    dataset_name: which dataset the cleanup was asked for. Used only in
    `invvar_weight`'s decline line, so a reader of the log sees which dataset
    had no measurement count rather than a bare refusal.

    groups: one scaffold-group label per training row, or None. Only the three
    cleanups added on 2026-09-17 read it, and all three read it for the same
    reason: each fits a model on part of the training rows and reads the result
    off the rest, so a random inner split would put a molecule's scaffold family
    on both sides and the predicted value a label is then moved toward would be
    partly that label itself. With None the inner split is a deterministic
    KFold, which is what the estimators in kirby.utils.noise_estimation already
    fall back to.
    """
    flag_percentile = 100.0 * (1.0 - float(noise_fraction))

    # A similarity setting only means something to the three cleanups that
    # read neighbours. Handed to any other cleanup, or a non-forest setting
    # handed to prox_smooth, it would be ignored, and every setting of a sweep
    # would write the same numbers under 50 different similarity labels. So
    # that is refused. MODEL_INPUT is today's behaviour for every cleanup and
    # is always accepted. Added in review 2026-09-25.
    if similarity_neighbours is not None:
        _sp = (similarity_neighbours.get('setting') or {}).get('space')
        if _sp != 'model_input':
            _reads = {'label_smooth': True, 'knn_smooth': True,
                      'prox_smooth': _sp == 'forest_proximity'}
            if not _reads.get(strategy, False):
                raise ValueError(
                    f"cleaned_training_set: {strategy!r} does not read "
                    f"neighbours from the similarity setting {_sp!r}, so the "
                    f"result would be today's numbers filed under that "
                    f"setting. Pass similarity_neighbours=None (or the "
                    f"MODEL_INPUT lists) for it.")

    def _clf_neighbours(X, n):
        # The CLF_NEIGHBOURS nearest training molecules by
        # CLF_NEIGHBOUR_METRIC, the molecule itself dropped.
        nn = make_nearest_neighbors(n_neighbors=min(CLF_NEIGHBOURS + 1, n),
                                    metric=CLF_NEIGHBOUR_METRIC)
        nn.fit(X)
        _, idx = nn.kneighbors(X)
        return idx[:, 1:]

    def _given_neighbours(k_today):
        # The neighbour indices from the caller's similarity setting, as
        # (n_tr, k) into the training rows with the molecule itself excluded.
        from experiments.noise.similarity_space import neighbours_at
        k = k_today if similarity_k is None else int(similarity_k)
        return neighbours_at(similarity_neighbours, k)[0]

    # A18/A19 (2026-10-08): every return None below goes through _declined,
    # so the caller can read which refusal applied (last_decline_reason).
    _DECLINE['reason'] = None
    _scheme = similarity_scheme
    if _scheme is None and similarity_neighbours is not None:
        _scheme = similarity_neighbours.get('scheme')

    def _given_label(k_today, y_src):
        # The neighbours' label under the swept counting scheme (axis 5):
        # weighted mean on regression, largest summed weight on
        # classification. None when no scheme was given, so the caller keeps
        # its equal-weight mean or majority.
        if similarity_neighbours is None or _scheme is None:
            return None
        from experiments.noise.similarity_space import (
            neighbours_at, weighted_neighbour_label)
        k = k_today if similarity_k is None else int(similarity_k)
        idx, dist = neighbours_at(similarity_neighbours, k)
        return weighted_neighbour_label(y_src, idx, dist, _scheme,
                                        is_clf=is_clf)

    if strategy == 'remove':
        threshold = np.percentile(noise_scores, flag_percentile)
        keep = noise_scores < threshold
        if keep.sum() < 10:
            return _declined(f'remove would leave {int(keep.sum())} training '
                             f'molecules, under 10')
        return {'X': X_tr_s[keep], 'y': y_tr[keep], 'sample_weight': None,
                'keep': keep, 'label_column': None}

    if strategy == 'downweight':
        return {'X': X_tr_s, 'y': y_tr,
                'sample_weight': weights_from_scores(
                    noise_scores,
                    strength=1.0 if strength is None else float(strength)),
                'keep': None, 'label_column': None}

    if strategy == 'huber_weight':
        # Huber's redescending weight, w = min(1, c*s/r), with the scale s the
        # median absolute suspicion score times 1.4826 -- the constant that
        # makes that a consistent estimator of the standard deviation under a
        # normal, and c = 1.345 the tuning constant giving 95 percent
        # efficiency there (Huber 1964).
        #
        # HOW THIS DIFFERS FROM `downweight`, which is the point of having
        # both: downweight's weight is 1/(1 + strength*s_norm), which falls
        # away smoothly from the very first molecule and never reaches zero.
        # This one is FLAT at 1.0 for every molecule inside the scale and only
        # then falls, so a label that looks ordinary is not touched at all.
        # Which shape is right is an empirical question this suite has never
        # asked.
        r = np.abs(np.asarray(noise_scores, dtype=float))
        s_scale = 1.4826 * np.median(r)
        if not np.isfinite(s_scale) or s_scale <= 0:
            # Every score identical: no molecule is more suspect than another,
            # so the honest weight is a flat one rather than a divide by zero.
            w = np.ones(len(r))
        else:
            c = 1.345 if strength is None else float(strength)
            w = np.minimum(1.0, c * s_scale / np.maximum(r, 1e-12))
        w = w / w.mean()          # mean-normalised, as weights_from_scores is
        return {'X': X_tr_s, 'y': y_tr, 'sample_weight': w,
                'keep': None, 'label_column': None}

    if strategy == 'invvar_weight':
        # The inverse-variance weight, taken from the measurement count alone.
        #
        # A molecule whose label is the mean of k repeat measurements carries
        # an error variance of that molecule's own measurement spread divided
        # by k. So the precision of the label rises with k, and the weight that
        # minimises the variance of the fit is proportional to k -- the weight
        # meta-analysis has used since Hedges & Olkin 1985. Normalised to mean
        # one, the same convention `huber_weight` above uses, so a weighted fit
        # and an unweighted fit see the same total weight and the difference
        # between them is the SHAPE of the weighting.
        #
        # HOW THIS DIFFERS FROM EVERY OTHER CLEANUP IN THE FILE: it reads no
        # suspicion score and no label. The other nine ask a detector which
        # labels look wrong. This one asks the record how many times somebody
        # measured it, which is known before any model is fitted and cannot be
        # wrong about a molecule the detector happens to misjudge.
        #
        # ⛔ DECLINING IS THE WHOLE POINT WHEN THERE IS NO COUNT. Falling back
        # to a flat weight would produce an `invvar_weight` column that is
        # bit-identical to the no-cleaning column, under a name that says a
        # method ran.
        if measurement_counts is None:
            print(f"      invvar_weight: no per-molecule measurement count "
                  f"exists for {dataset_name or 'this dataset'}, so there is "
                  f"nothing to weight by; declining rather than returning a "
                  f"flat weight, which would be the no-cleaning column under "
                  f"another name.")
            return _declined(f'invvar_weight: no per-molecule measurement '
                             f'count exists for '
                             f'{dataset_name or "this dataset"}')
        k = np.asarray(measurement_counts, dtype=np.float64)
        if len(k) != len(y_tr):
            # A caller bug, not a fact about the dataset: the counts and the
            # training rows index the same molecules, so a mismatch means every
            # molecule after the first gap would carry another one's weight.
            raise ValueError(
                f"invvar_weight: {len(k)} measurement counts for "
                f"{len(y_tr)} training molecules. They index the same "
                f"molecules, so a mismatch misweights every one after the "
                f"first gap.")
        if not np.all(np.isfinite(k)) or np.any(k < 1):
            raise ValueError(
                f"invvar_weight: a measurement count must be a finite number "
                f"of measurements, one or more. Got a range of {np.nanmin(k)} "
                f"to {np.nanmax(k)}.")
        return {'X': X_tr_s, 'y': y_tr, 'sample_weight': k / k.mean(),
                'keep': None, 'label_column': None}

    if strategy == 'js_shrink':
        # James-Stein / empirical-Bayes shrinkage toward the out-of-fold
        # prediction. Each label moves by shrink_i = sigma2_i / (sigma2_i +
        # tau2), where tau2 is the spread of the label about the prediction
        # across the whole training block -- the signal the model can see --
        # and sigma2_i is how noisy THIS label looks. A label the detector
        # finds ordinary barely moves; one it finds wrong moves most of the
        # way. (James & Stein 1961; Efron & Morris 1975.)
        #
        # HOW THIS DIFFERS FROM `oof_smooth`, which also moves a label toward
        # the same prediction: oof_smooth moves it by `alpha * score_norm`,
        # a fraction the caller chooses and the score scales. Here the fraction
        # is DERIVED -- it is the ratio of the two variances, so there is no
        # alpha to tune and the shrinkage is whatever the data says it should
        # be. If the two disagree, the difference is the tuning.
        # Build the target when the caller hoisted none, exactly as
        # `oof_smooth` does. Declining instead was wrong: n07 passes no targets
        # at all -- its three callers of `_evaluate_fold` leave targets at None
        # -- so js_shrink skipped on every n07 fold and the cleanup would have
        # looked broken rather than untested. n18 DOES hoist them, and its
        # want-list is now derived from CLEANUP_TARGETS so a new cleanup cannot
        # be forgotten there again.
        oof = (targets or {}).get('oof')
        if oof is None:
            try:
                oof = oof_predict(X_tr_s, y_tr, groups,
                                  n_splits=min(REPAIR_OOF_SPLITS, len(y_tr)))
            except Exception as exc:
                print(f"      js_shrink: could not build an out-of-fold "
                      f"prediction ({type(exc).__name__}: {exc}); skipping "
                      f"rather than shrinking toward the fold mean, which is "
                      f"a different method.")
                return _declined(f'js_shrink: the out-of-fold prediction '
                                 f'failed ({type(exc).__name__}: {exc})')
        oof = np.asarray(oof, dtype=float)
        resid = np.asarray(y_tr, dtype=float) - oof
        tau2 = float(np.var(resid))
        if not np.isfinite(tau2) or tau2 <= 0:
            return _declined(f'js_shrink: the out-of-fold residuals have no '
                             f'spread (variance {tau2!r})')
        # The suspicion score is not in label units, so it is put on the same
        # scale as the residuals before being read as a per-molecule variance.
        sc = np.asarray(noise_scores, dtype=float)
        rng_sc = np.ptp(sc)
        if not np.isfinite(rng_sc) or rng_sc <= 0:
            return _declined('js_shrink: the suspicion scores have no spread')
        sc_norm = (sc - sc.min()) / rng_sc
        sigma2 = sc_norm * tau2 * (JS_SHRINK_STRENGTH if strength is None
                                   else float(strength))
        shrink = sigma2 / (sigma2 + tau2)
        y_new = (1.0 - shrink) * np.asarray(y_tr, dtype=float) + shrink * oof
        return {'X': X_tr_s, 'y': y_new, 'sample_weight': None,
                'keep': None, 'label_column': y_new}

    if strategy == 'label_smooth':
        if similarity_neighbours is not None:
            _nbrs = _given_neighbours(
                min(CLF_NEIGHBOURS if is_clf else REG_NEIGHBOURS,
                    len(y_tr) - 1))
        if is_clf:
            # The flagged share replaced by the majority class of the
            # CLF_NEIGHBOURS nearest by CLF_NEIGHBOUR_METRIC (f36, 2026-10-05).
            if similarity_neighbours is None:
                _nbrs = _clf_neighbours(X_tr_s, len(y_tr))
            from scipy.stats import mode as scipy_mode
            nbr_modes = _given_label(
                min(CLF_NEIGHBOURS, len(y_tr) - 1), y_tr)
            if nbr_modes is None:
                nbr_modes = scipy_mode(y_tr[_nbrs], axis=1).mode.ravel()
            # Only correct high-noise-score samples
            threshold = np.percentile(noise_scores, flag_percentile)
            y_smoothed = y_tr.copy()
            y_smoothed[noise_scores >= threshold] = nbr_modes[noise_scores >= threshold]
        else:
            if similarity_neighbours is None:
                nn = make_nearest_neighbors(
                    n_neighbors=min(REG_NEIGHBOURS + 1, len(y_tr)))
                nn.fit(X_tr_s)
                _, nbr_idx = nn.kneighbors(X_tr_s)
                _nbrs = nbr_idx[:, 1:]
            nbr_means = _given_label(min(REG_NEIGHBOURS, len(y_tr) - 1), y_tr)
            if nbr_means is None:
                nbr_means = np.mean(y_tr[_nbrs], axis=1)
            y_smoothed = smooth_by_scores(
                y_tr, noise_scores, nbr_means,
                alpha=SMOOTH_STRENGTH if strength is None else float(strength),
                method='proportional')
        return {'X': X_tr_s, 'y': y_smoothed, 'sample_weight': None,
                'keep': None, 'label_column': y_smoothed}

    if strategy == 'knn_smooth':
        threshold = np.percentile(noise_scores, flag_percentile)
        flagged = noise_scores >= threshold
        if similarity_neighbours is not None:
            _nbrs = _given_neighbours(
                min(CLF_NEIGHBOURS if is_clf else REG_NEIGHBOURS,
                    len(y_tr) - 1))
        elif is_clf:
            _nbrs = _clf_neighbours(X_tr_s, len(y_tr))
        else:
            nn = make_nearest_neighbors(
                n_neighbors=min(REG_NEIGHBOURS + 1, len(y_tr)))
            nn.fit(X_tr_s)
            _, nbr_idx = nn.kneighbors(X_tr_s)
            _nbrs = nbr_idx[:, 1:]
        _w = _given_label(min(CLF_NEIGHBOURS if is_clf else REG_NEIGHBOURS,
                              len(y_tr) - 1), y_tr)
        if is_clf:
            from scipy.stats import mode as scipy_mode
            nbr_modes = (_w if _w is not None else
                         scipy_mode(y_tr[_nbrs], axis=1).mode.ravel())
            y_corrected = y_tr.copy()
            y_corrected[flagged] = nbr_modes[flagged]
        else:
            nbr_means = (_w if _w is not None else
                         np.mean(y_tr[_nbrs], axis=1))
            y_corrected = y_tr.copy()
            y_corrected[flagged] = nbr_means[flagged]
        return {'X': X_tr_s, 'y': y_corrected, 'sample_weight': None,
                'keep': None, 'label_column': y_corrected}

    if strategy == 'prox_smooth':
        # knn_smooth's correction, with the neighbours taken from the FOREST'S
        # OWN notion of similarity instead of Euclidean distance in the feature
        # space. Two molecules are close when the trees keep putting them in
        # the same leaf, so the similarity is importance- and
        # interaction-aware without anyone choosing a feature weighting.
        #
        # WHY THIS IS NOT knn_smooth WITH A DIFFERENT ARGUMENT. Euclidean
        # distance over a wide representation treats every feature alike, and
        # the noise suite has measured what that costs: n12 found 97 percent of
        # every distance being computed over columns of exact zeros, and its
        # top-50 cut of a representation beat the full width in 10 of 12 rows.
        # A proximity never has that problem, because a feature the trees never
        # split on cannot move two molecules apart.
        #
        # ⛔ COST. The forest is fitted on the fold's training molecules and the
        # proximity is a square matrix, so this is O(n^2) in memory where
        # knn_smooth is not. On the 448-molecule folds in this suite that is
        # about 0.8 MB; on a 10,000-molecule training block it would be 400 MB
        # and this cleanup should be refused rather than run.
        #
        # f36 DECISION 2026-10-05 (F36.md section 9). On a REGRESSION set this
        # no longer replaces the flagged share: every label moves
        # PROX_SMOOTH_SHARE of the way toward the mean label of its
        # PROX_SMOOTH_K forest-neighbours, once, and the suspicion score is
        # not read. On a classification set the flagged share is replaced by
        # the majority class of the CLF_NEIGHBOURS forest-neighbours, as
        # before but at that count.
        n_tr = len(y_tr)
        from kirby.utils.noise_estimation import (forest_leaf_ids,
                                                  forest_proximity_distance)
        _k_prox = min(CLF_NEIGHBOURS if is_clf else PROX_SMOOTH_K, n_tr - 1)
        if is_clf:
            threshold = np.percentile(noise_scores, flag_percentile)
            flagged = noise_scores >= threshold
        # The forest sees the training labels AS THEY ARE, corrupted and all.
        # Fitting it on anything cleaner would hand the neighbour list the
        # answer the experiment is asking it to find.
        _given_forest = (similarity_neighbours is not None
                         and (similarity_neighbours.get('setting') or {}).get(
                             'space') == 'forest_proximity')
        if _given_forest:
            # The caller's forest_proximity lists: the same forest and the same
            # row sort as the four lines below (similarity_space.py), held as
            # a k_max list so no n x n matrix is built here.
            nbr_idx = _given_neighbours(_k_prox)
        else:
            leaves, oob = forest_leaf_ids(X_tr_s, y_tr, is_clf)
            D = forest_proximity_distance(leaves, oob)
            np.fill_diagonal(D, np.inf)          # never its own neighbour
            nbr_idx = np.argsort(D, axis=1)[:, :_k_prox]
        y_corrected = np.asarray(y_tr).copy()
        _w = (_given_label(_k_prox, np.asarray(y_tr) if is_clf
                           else np.asarray(y_tr, dtype=float))
              if _given_forest else None)
        if is_clf:
            from scipy.stats import mode as scipy_mode
            nbr_vals = (_w if _w is not None else
                        scipy_mode(np.asarray(y_tr)[nbr_idx], axis=1).mode.ravel())
            y_corrected[flagged] = nbr_vals[flagged]
        else:
            nbr_vals = (_w if _w is not None else
                        np.mean(np.asarray(y_tr, dtype=float)[nbr_idx], axis=1))
            a = PROX_SMOOTH_SHARE if strength is None else float(strength)
            y_corrected = np.asarray(y_tr, dtype=float) + a * (
                nbr_vals - np.asarray(y_tr, dtype=float))
        return {'X': X_tr_s, 'y': y_corrected, 'sample_weight': None,
                'keep': None, 'label_column': y_corrected}

    # ---- The three added 2026-09-17. ------------------------------------
    # All three are regression-only, and all three are refused rather than
    # approximated on a classification dataset.
    if strategy in REGRESSION_ONLY_CLEANUPS and is_clf:
        raise ValueError(
            f"cleaned_training_set: {strategy!r} moves a label to a numeric "
            f"value and is not defined on a class label.")

    if strategy == 'oof_smooth':
        # Same shape as label_smooth — every label moves toward one target in
        # proportion to its suspicion score — with a different target. Where
        # label_smooth aims at the mean label of the ten nearest molecules,
        # this aims at what a LightGBM model trained on the OTHER training
        # molecules predicts for this one. The inner split is group-aware, so
        # no molecule's own scaffold family trains the model that predicts it
        # (kirby/utils/noisy_labels/core.py:127).
        target = (targets or {}).get('oof')
        if target is None:
            target = oof_predict(X_tr_s, y_tr, groups,
                                 n_splits=min(REPAIR_OOF_SPLITS, len(y_tr)))
        y_smoothed = smooth_by_scores(
            y_tr, noise_scores, target,
            alpha=SMOOTH_STRENGTH if strength is None else float(strength),
            method='proportional')
        return {'X': X_tr_s, 'y': y_smoothed, 'sample_weight': None,
                'keep': None, 'label_column': y_smoothed}

    if strategy == 'gp_smooth':
        # The same shape again, aiming at the posterior mean of a Gaussian
        # process fitted on the other training molecules. GaussianProcessEstimator
        # already computed this on every call and kept it on the fitted object
        # as predictions_; nothing in this file read it before today.
        #
        # ⛔ UNTIL 2026-09-21 THAT POSTERIOR MEAN WAS A CONSTANT PER INNER FOLD,
        # NOT A PREDICTION FOR THIS MOLECULE. MEASURED on fold 0 of
        # flab_thermo_tresanco, saprot, 448 training molecules and 1280
        # features, REPAIR_OOF_SPLITS cut to 3 for the probe: the 448 molecules
        # received 3 distinct values -- 64.0567, 65.7113 and 65.7467 label units
        # -- and each one is exactly the mean label of the inner fold that
        # molecule was held out of. The fitted noise level came back at 22.7930
        # label units against a fitted-label standard deviation of 22.8228, so
        # the process assigned all of the variance to noise and its posterior
        # mean fell back to its prior mean. Its rank correlation with the CLEAN
        # labels was -0.0441 spearman.
        #
        # WHAT WAS WRONG, AND IT WAS NOT THE WIDTH. The RBF kernel's length
        # scale started at 1.0. Two standardised molecules described by 1280
        # features sit about sqrt(2 * 1280) = 50.6 apart, exp(-50.6^2 / 2)
        # underflows, so the kernel matrix is the identity and the gradient of
        # the marginal likelihood in the length scale is flat. The three inner
        # folds came back with length scales of 1.01, 1.00 and 1.01 -- the
        # optimiser never moved -- and the white-noise level pinned at 1.0, its
        # own upper bound.
        #
        # FIXED in GaussianProcessEstimator.estimate
        # (src/kirby/utils/noise_estimation.py): the length scale now starts at
        # the median distance between the training molecules it is fitted on.
        # Same fold, same features, same noise draw: 358 distinct posterior-mean
        # values over the 448 molecules, fitted length scales 61.87, 54.26 and
        # 41.85, and 0.5630 spearman against the clean labels where it was
        # -0.0441. The target sits 9.5756 label units from the clean label of
        # the molecules the injector damaged, against 11.0137 before.
        #
        # ⛔ EVERY gp_smooth NUMBER RECORDED BEFORE 2026-09-21 IS A NUMBER FOR
        # SHRINKAGE TOWARD THE INNER FOLD'S MEAN LABEL, not for a Gaussian
        # process: the 13.4% of the clean-to-corrupted gap it recovered on esmc,
        # which put it second from bottom of the seven, and the 47.2% it
        # recovered at strength 1.0. Weighting the kernel inputs by feature
        # importance was measured against the broken process too and came back
        # bit-identical, so that question is open again now the process fits.
        target = (targets or {}).get('gp')
        if target is None:
            # ⛔ THE ROW CAP LIVES HERE TOO, BECAUSE THE FIT IS HERE.
            #
            # GP_SMOOTH_MAX_ROWS was written into `correction_targets` above
            # and nowhere else, and n07 calls `correction_targets` ZERO times
            # -- grep the file. n18 calls it; n07 reaches gp_smooth through
            # this branch, which fitted the Gaussian process with no row check
            # at all. So the cap that exists because gp_smooth did not finish a
            # fold on herg_ki's 1,130-molecule training blocks was not read on
            # the path herg_ki actually takes.
            #
            # Refusing rather than subsampling, for the reason the cap's own
            # note gives: a target fitted on a subsample is a different
            # quantity from one fitted on the fold, and a cleanup silently
            # aiming somewhere else is worse than one that says it did not run.
            if len(y_tr) > GP_SMOOTH_MAX_ROWS:
                print(f"      gp_smooth refused: {len(y_tr)} training "
                      f"molecules is over GP_SMOOTH_MAX_ROWS="
                      f"{GP_SMOOTH_MAX_ROWS}; a Gaussian process is cubic in "
                      f"the rows and this one has no cap.")
                return _declined(
                    f'gp_smooth: {len(y_tr)} training molecules is over '
                    f'GP_SMOOTH_MAX_ROWS={GP_SMOOTH_MAX_ROWS}; a Gaussian '
                    f'process is cubic in the rows and this one has no cap')
            est = GaussianProcessEstimator(n_folds=REPAIR_OOF_SPLITS,
                                           groups=groups)
            est.estimate(X_tr_s, y_tr)
            target = np.asarray(est.predictions_, dtype=float)
        y_smoothed = smooth_by_scores(
            y_tr, noise_scores, target,
            alpha=SMOOTH_STRENGTH if strength is None else float(strength),
            method='proportional')
        return {'X': X_tr_s, 'y': y_smoothed, 'sample_weight': None,
                'keep': None, 'label_column': y_smoothed}

    if strategy == 'constant_lift':
        # n30's constant lift (n30_censored_likelihood.lift_columns), added to
        # the suite from f36 on 2026-10-05. The ceiling is read off the
        # training labels themselves: the largest label, when at least 1% of
        # the rows and at least 2 rows sit on it. The rows at the ceiling are
        # raised to ceiling + the mean Tobit lift over those rows, where each
        # row's Tobit lift is sd x the inverse Mills ratio of (ceiling - its
        # out-of-fold LightGBM prediction) / sd, and sd is the spread of
        # label - prediction over the rows NOT at the ceiling. No other label
        # moves. Declines, and says so, when the labels show no ceiling.
        from experiments.noise import n30_censored_likelihood as n30
        limit = n30.limit_from_labels(y_tr)
        if limit is None:
            _yy = np.asarray(y_tr, dtype=float)
            _top = float(np.max(_yy)) if _yy.size else float('nan')
            return _declined(
                f'constant_lift: the training labels show no ceiling: '
                f'{int(np.sum(_yy >= _top - 1e-12))} of {_yy.size} training '
                f'molecules sit on the largest label {_top:.6g}, under the '
                f'{max(2, int(np.ceil(n30.MIN_SHARE_AT_LIMIT * _yy.size)))} '
                f'needed (at least 2 and {n30.MIN_SHARE_AT_LIMIT:.0%})')
        oof = (targets or {}).get('oof')
        if oof is None:
            oof = oof_predict(X_tr_s, y_tr, groups,
                              n_splits=min(REPAIR_OOF_SPLITS, len(y_tr)))
        oof = np.asarray(oof, dtype=float)
        at = n30.rows_at_limit(y_tr, limit)
        sd = n30.residual_sd(y_tr, oof, at)
        if not np.isfinite(sd) or sd <= 0:
            return _declined(
                f'constant_lift: the spread of label minus out-of-fold '
                f'prediction over the molecules below the ceiling is {sd!r}, '
                f'not positive')
        cols, _info = n30.lift_columns(y_tr, oof, at, limit, sd, (), (0,))
        y_lifted = np.asarray(cols['constant_lift'], dtype=float)
        return {'X': X_tr_s, 'y': y_lifted, 'sample_weight': None,
                'keep': None, 'label_column': y_lifted}

    if strategy == 'quantile_clip':
        # Cut from CLEANUPS on 2026-10-05 (owner, f36); runs when named.
        # Not a move toward a target. Each molecule gets a range of label
        # values — the 10th to the 90th percentile of the labels that training
        # molecules in its own leaves carry, read out of a quantile regression
        # forest fitted on the other training molecules. A label inside its
        # range is left exactly as it is; a label outside is pulled back onto
        # the edge it crossed. So this cleanup rewrites only the labels no
        # molecule like them carries, and the rest of the column is untouched.
        _q = (targets or {}).get('quantile')
        if _q is None:
            est = QuantileForestEstimator(n_folds=REPAIR_OOF_SPLITS,
                                          groups=groups)
            est.estimate(X_tr_s, y_tr)
            _q = (est.quantile_lo_, est.quantile_hi_)
        y_clipped = clip_to_range(y_tr, _q[0], _q[1], alpha=1.0)
        return {'X': X_tr_s, 'y': y_clipped, 'sample_weight': None,
                'keep': None, 'label_column': y_clipped}

    # ---- The cleanup halves of the in-loop methods, 2026-10-30 (SPLIT_CLEANUPS).
    if strategy == 'score_weight':
        from kirby.utils.lnl_parts import score_weight as _sw
        w = _sw(noise_scores)
        if w is None:
            _s = np.asarray(noise_scores, dtype=float)
            return _declined(f'score_weight: the score is not in [0, 1] '
                             f'(range {np.nanmin(_s):.4g} to '
                             f'{np.nanmax(_s):.4g}), so one minus it is not '
                             f'a weight')
        return {'X': X_tr_s, 'y': y_tr, 'sample_weight': w,
                'keep': None, 'label_column': None}

    if strategy == 'spread_invvar_weight':
        from kirby.utils.lnl_parts import spread_invvar_weight as _iv
        w = _iv(noise_scores)
        if w is None:
            _s = np.asarray(noise_scores, dtype=float)
            return _declined(f'spread_invvar_weight: a score is not a '
                             f'positive finite variance (smallest '
                             f'{np.nanmin(_s):.4g})')
        return {'X': X_tr_s, 'y': y_tr, 'sample_weight': w,
                'keep': None, 'label_column': None}

    if strategy in ('network_rewrite', 'ema_rewrite'):
        from kirby.utils import lnl_parts
        threshold = np.percentile(noise_scores, flag_percentile)
        flagged = np.asarray(noise_scores) >= threshold
        if flagged.sum() == 0:
            return _declined(f'{strategy}: no molecule reaches the flag '
                             f'threshold')
        _cls = np.unique(y_tr) if is_clf else None
        if strategy == 'network_rewrite':
            y_new = lnl_parts.network_rewrite(
                X_tr_s, y_tr, flagged, is_clf=is_clf, classes_=_cls,
                groups=groups)
            if y_new is None:
                return _declined(f'network_rewrite would train the guessing '
                                 f'network on {int((~flagged).sum())} '
                                 f'unflagged molecules, under 10')
        else:
            y_new = lnl_parts.ema_rewrite(
                X_tr_s, y_tr, flagged, is_clf=is_clf, classes_=_cls)
        return {'X': X_tr_s, 'y': y_new, 'sample_weight': None,
                'keep': None, 'label_column': y_new}

    raise ValueError(f"cleaned_training_set: unknown cleanup {strategy!r}")


# How many downstream models, and how many representations for each of them,
# the three expensive cleanups are measured under. USER DECISION 2026-09-17.
# Both are chosen from measurements this run makes itself; see
# `rank_models_and_reps` below. Nothing is read from a file an earlier run
# wrote, so this behaves the same on a laptop and on the cluster.
N_TOP_MODELS = 2
N_TOP_REPS_PER_MODEL = 2

# The widest a set of features may be before the three model-fitting cleanups
# refuse to run on it. USER DECISION 2026-09-18: 2000 features.
#
# WHY THERE IS A CEILING AT ALL. These three cleanups fit five models each
# before they repair anything, and how long that takes is driven by the number
# of features more than by anything else. MEASURED 2026-09-17 on this machine,
# one run of all three on a training fold of 1,200 molecules: 275 seconds where
# a molecule is described by 200 features, and 2,351 seconds -- 39 minutes --
# where it is described by 2,048. Over a whole dataset that is the difference
# between roughly nine hours and roughly forty-four.
#
# WHAT IT DOES AND DOES NOT TOUCH. A representation wider than this is still
# measured by the ranking below and its score is still written out, so nothing
# disappears; it is only barred from being one of the two a model's expensive
# cleanups run on. A model whose hybrid budget exceeds it keeps every one of
# the four older cleanups in those hybrid features and loses only the three.
# Nothing else in n07 reads this number.
#
# Overridable per run with --max-cleanup-features or KIRBY_MAX_CLEANUP_FEATURES,
# so it is a setting rather than a number baked into the file.
MAX_CLEANUP_FEATURES = 2000


def max_cleanup_features(kwargs=None):
    """The width ceiling this run uses, and where it came from.

    Order: the run's own argument, then the environment, then
    MAX_CLEANUP_FEATURES. 0 or a negative number means no ceiling.
    """
    val = (kwargs or {}).get('max_cleanup_features')
    src = 'run argument'
    if val is None:
        env = os.environ.get('KIRBY_MAX_CLEANUP_FEATURES')
        if env not in (None, ''):
            val, src = env, 'KIRBY_MAX_CLEANUP_FEATURES'
    if val is None:
        val, src = MAX_CLEANUP_FEATURES, 'n07.MAX_CLEANUP_FEATURES'
    try:
        val = int(val)
    except (TypeError, ValueError):
        val, src = MAX_CLEANUP_FEATURES, 'n07.MAX_CLEANUP_FEATURES'
    return (val if val > 0 else None), src


def rank_models_and_reps(reps, labels, groups, model_factories, model_names,
                         n_folds, mode, sample_X, metric_fn, metric_name,
                         lower_is_better, is_clf,
                         n_top_models=N_TOP_MODELS,
                         n_top_reps=N_TOP_REPS_PER_MODEL,
                         max_features=None):
    """Measure every (downstream model, representation) pair in this run, then
    keep the best models and each kept model's best representations.

    NOTHING IS READ FROM DISK. Until 2026-09-17 the representations were ranked
    by `extract_top3_base_reps`, which reads scores out of a stored search. That
    search ran somewhere else, at some other time, on activity values this
    experiment has not corrupted, and on the cluster it may be stale or absent
    altogether. This function instead fits each model on each representation on
    the UNCORRUPTED activity values, one fit per (model, representation, fold),
    and reads the error on the held-out molecules. Those errors are the ranking,
    and they are written into the result so the ranking can be checked.

    Cost: len(model_names) x len(reps) x n_folds model fits, each one a single
    fit with no corruption and no cleanup. That is one fit where each of the
    three expensive cleanups needs five.

    Returns a dict:
        scores          {model: {representation: mean held-out error}}
        n_features      {representation: how many features it has}
        model_order     every model, best first
        top_models      the first `n_top_models` of model_order
        top_reps        {model: [representation, ...]} for the kept models
        ranked_on       what the ranking measured, in words
    """
    scores = {}
    for m_name in model_names:
        factory = model_factories.get(m_name)
        if factory is None:
            continue
        per_rep = {}
        for rep_name, X_rep in reps.items():
            fold_scores = []
            for train_idx, val_idx in cv_split(groups, n_folds, sample_X,
                                               labels, mode=mode):
                # 0/1 features unscaled (2.3 item 17, 2026-10-05).
                from kirby.scaling import FeatureScaler
                sc = FeatureScaler()
                X_tr = sc.fit_transform(X_rep[train_idx])
                X_va = sc.transform(X_rep[val_idx])
                try:
                    mdl = factory()
                    mdl.fit(X_tr, labels[train_idx])
                    fold_scores.append(float(metric_fn(
                        labels[val_idx],
                        _predict_for_metric(mdl, X_va, metric_name))))
                except Exception as exc:
                    print(f"      ranking fit failed for {m_name} on "
                          f"{rep_name}: {type(exc).__name__}: {exc}")
            if fold_scores:
                per_rep[rep_name] = float(np.mean(fold_scores))
        if per_rep:
            scores[m_name] = per_rep

    def _best(d):
        return min(d.values()) if lower_is_better else max(d.values())

    model_order = sorted(scores, key=lambda m: (
        _best(scores[m]) if lower_is_better else -_best(scores[m])))
    top_models = model_order[:n_top_models]
    n_features = {r: int(np.asarray(X).shape[1]) for r, X in reps.items()}
    # A representation wider than the ceiling keeps its measured score above
    # and is only barred from being SELECTED here, so a reader can still see
    # how it did and what it was excluded for.
    too_wide = ([] if max_features is None
                else sorted(r for r, w in n_features.items()
                            if w > max_features))
    top_reps = {}
    for m_name in top_models:
        eligible = [r for r in scores[m_name] if r not in too_wide]
        ordered = sorted(eligible, key=lambda r: (
            scores[m_name][r] if lower_is_better else -scores[m_name][r]))
        top_reps[m_name] = ordered[:n_top_reps]

    return {
        'scores': scores,
        'n_features': n_features,
        'max_features': max_features,
        'reps_excluded_as_too_wide': {
            r: int(n_features[r]) for r in too_wide},
        'exclusion_note': (
            f'A representation describing a molecule with more than '
            f'{max_features} features was measured and scored, but could not '
            f'be selected for the three model-fitting cleanups. Those three '
            f'fit five models each and their cost is driven by the feature '
            f'count.' if max_features is not None
            else 'No width ceiling was applied.'),
        'model_order': model_order,
        'top_models': top_models,
        'top_reps': top_reps,
        'metric_name': metric_name,
        'lower_is_better': bool(lower_is_better),
        'ranked_on': (
            'Every downstream model was fitted on every representation using '
            'the UNCORRUPTED activity values, once per fold, and scored on the '
            'held-out molecules in the dataset\'s own metric. A model ranks by '
            'its best representation; a representation ranks within that '
            'model. Measured in this run, not read from any stored search.'),
    }


def _evaluate_fold(X_tr_s, X_va_s, y_train, y_val, strategy, noise_fraction,
                   model_factory, metric_fn, is_clf=False, metric_name=None,
                   noise_seed=42, precomputed_y_tr=None, noise_scores=None,
                   dataset_name=None, noise_level=0.5,
                   y_all=None, train_idx=None, train_groups=None,
                   strength=None, targets=None, measurement_counts=None,
                   condition=DEFAULT_CONDITION, cache_dataset=None,
                   cache_model_name=None):
    """Evaluate a single remediation strategy on one fold.

    cache_dataset / cache_model_name, added 2026-09-25 (plan section 3): given
    both, the fit goes through noise_cache.cached_fit_predict under that
    dataset and model name, so the no-noise fit, the noisy-baseline fit and
    every cleanup's fit are stored once per (features, labels, weights, model)
    and the D-versus-D' rows of the same fold read them back instead of
    fitting again. The prediction is the one _predict_for_metric returns
    (predict_mode_for is the same rule). None, the default, fits here as
    before.

    train_groups: one scaffold-group label per TRAINING row, forwarded to
    cleaned_training_set. Only the three cleanups added 2026-09-17 read it; see
    that function's docstring for why they need it.

    measurement_counts: how many times each TRAINING row's molecule was
    measured, forwarded the same way. Only `invvar_weight` reads it and it
    declines without it, so a caller that leaves it None gets a refusal with
    the dataset named rather than a silently flat weight.

    noise_seed: forwarded to the injector so the caller can average across
    multiple noise realizations. Default 42 preserves single-seed behavior.
    y_all / train_idx: the FULL clean label column and this fold's positions in
    it. Needed only on the no-cache path below, because the draw runs over the
    whole column (see noise_helpers.inject_train_labels). Every caller in this
    file takes the recommended path and passes precomputed_y_tr instead.
    precomputed_y_tr / noise_scores: optional caller-supplied y_tr and
    noise_scores. When provided, the per-strategy call doesn't reinject
    noise or refit the estimator — the caller is expected to do that once
    per (fold, noise_seed) and pass them in. This is the recommended path;
    the no-cache path is kept for callers that haven't migrated.

    strategy: a cleanup name, or flag_share_label(cleanup, share). With a
    share in the name, the cleanup flags that share and noise_fraction reaches
    only the injector: n07's own run names every thresholding cleanup this way
    (2026-09-28), because there noise_fraction is also the injector's share.
    With a bare name, noise_fraction is the flag share, as before -- n18 and
    n25 call this with their own swept rate and precomputed labels, so the
    number reaches no injector.

    Returns the fold score (float) or None if the fold failed.
    """
    cleanup, flag_share = split_strategy(strategy)
    # Inject noise (except for clean baseline) — if caller didn't precompute.
    if strategy == 'no_noise':
        y_tr = y_train.copy()
    elif precomputed_y_tr is not None:
        y_tr = precomputed_y_tr
    else:
        if y_all is None or train_idx is None:
            raise ValueError(
                "_evaluate_fold: the no-cache injection path needs y_all (the "
                "full clean label column) and train_idx (this fold's positions "
                "in it). The draw runs over the whole column so that a "
                "molecule's corruption does not depend on its fold. Pass "
                "precomputed_y_tr instead if the caller already injected.")
        _spec = _noise_spec(dataset_name, noise_fraction, noise_level,
                            noise_seed=noise_seed, is_clf=is_clf,
                            condition=condition)
        y_tr, _noise_mask, _ = _inject_or_recorded(
            y_all, _spec, train_idx=train_idx, is_clf=is_clf,
            dataset_name=dataset_name)

    # If caller supplied noise_scores, use them; otherwise compute. Returns
    # None when the strategy needs scores and they're unavailable (loud).
    if cleanup in SCORE_DRIVEN_CLEANUPS:
        if noise_scores is None:
            # train_groups, for the same reason every other call in this file
            # now passes this fold's scaffold groups: without them a molecule's
            # own scaffold family trains the model that predicts it.
            noise_scores = _compute_noise_scores(X_tr_s, y_tr, is_clf=is_clf,
                                                 groups=train_groups)
        if noise_scores is None:
            return None

    # How many training rows the four cleanups touch. It was fixed at the 80th
    # percentile of the noise scores, so a fifth of the rows were flagged while
    # noise_fraction moved a tenth — every cleanup was handed twice as many rows
    # as carried the WIDE error. The flag rate was then noise_fraction itself:
    # 0.1 flags the top tenth, at the 90th percentile. ⛔ SINCE 2026-09-28 IT IS
    # NOT: that handed each cleanup the injector's own share. The share is now
    # swept over FLAG_SHARES and read from the strategy name (see FLAG_SHARES).
    #
    # ⛔ THAT IS NOT "THE CORRUPTED FRACTION", WHICH IS ONE. This comment used to
    # call noise_fraction the corrupted fraction and say the flag rate now
    # matches it. It does not. The Outlier condition is Huber contamination at
    # lambda 3 (src/noise_inject/noiseInject/core.py:65-67, "a randomly chosen
    # fraction get a WIDER error"), so EVERY training label is moved and
    # noise_fraction is the share moved about three times as far. MEASURED
    # 2026-09-20 on fold 0 of flab_thermo_tresanco at injected level 2.0: the
    # distance from a training label to its clean value averaged 39.32 label
    # units over the 39 molecules the injector flagged and 12.67 label units
    # over the other 409, against a clean label standard deviation of 10.12.
    #
    # What follows from it, and it is the whole ranking: a cleanup that rewrites
    # only the flagged tenth leaves the other nine tenths of the damage in place.
    # `remove`, `knn_smooth` and `quantile_clip` are all of that kind. The four
    # that move EVERY label some distance are not, and how far they move is set
    # by `strength` — see cleaned_training_set.
    #
    # The cut-off now lives in `cleaned_training_set` above, which is where the
    # cleanups moved to on 2026-09-17 so n20 can apply the same cleanup to
    # the labels the recipe search sees. What each one does is unchanged.
    # quantile_clip does not read the cut-off: it flags a label by whether it
    # lies outside the range of values molecules like it carry, not by where its
    # suspicion score falls among the others'.

    def _fit_score(X_fit, y_fit, sample_weight=None):
        if cache_dataset is not None and cache_model_name is not None:
            from experiments.noise.noise_cache import (cached_fit_predict,
                                                       predict_mode_for)
            pred = cached_fit_predict(cache_dataset, cache_model_name,
                                      model_factory, X_fit, y_fit, X_va_s,
                                      sample_weight=sample_weight,
                                      predict_mode=predict_mode_for(
                                          metric_name))
            return metric_fn(y_val, pred)
        m = model_factory()
        require_weights_accepted(m, sample_weight)
        if sample_weight is None:
            m.fit(X_fit, y_fit)
        else:
            m.fit(X_fit, y_fit, sample_weight=sample_weight)
        return metric_fn(y_val, _predict_for_metric(m, X_va_s, metric_name))

    # Apply strategy
    try:
        if strategy in ('no_noise', 'noisy_baseline'):
            return _fit_score(X_tr_s, y_tr)

        elif cleanup in CLEANUPS:
            # A cleanup that reads no flag share is handed noise_fraction only
            # because the argument is positional; it computes a percentile
            # from it and never uses it.
            cleaned = cleaned_training_set(cleanup, X_tr_s, y_tr, noise_scores,
                                           (noise_fraction if flag_share is None
                                            else flag_share), is_clf=is_clf,
                                           groups=train_groups,
                                           strength=strength,
                                           targets=targets,
                                           measurement_counts=measurement_counts,
                                           dataset_name=dataset_name)
            if cleaned is None:
                # `remove` left under 10 training rows. Same fall-through as
                # before: no score for this fold rather than a fit on nothing.
                return None
            return _fit_score(cleaned['X'], cleaned['y'],
                              sample_weight=cleaned['sample_weight'])

    except Exception as exc:
        # Surface the failure rather than silently dropping it. Helps
        # distinguish real bugs from "strategy not applicable here" cases.
        print(f"      {strategy} fold failed: {type(exc).__name__}: {exc}")
        return None

    return None


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


# =============================================================================
# SIMILARITY-SWEEP KIT, added 2026-09-25 (plan sections 1-3,
# ~/.claude/plans/pasted-content-id-0428-noise-experiment-fizzy-ladybug.md).
#
# Every noise experiment in n01, n02, n07, n09, n11-n14 and c10 uses the code
# below, so each one sweeps the similarity axis, writes its D-versus-D' rows
# and checkpoints the same way. It lives here because n07 is the module every
# other noise experiment already imports; nothing in it changes what n07's own
# run() computed before 2026-09-25.
#
# What it adds, in order:
#   sweep_settings / sweep_ks   which similarity settings and which k a run
#                               sweeps: similarity_space.settings_from_env()
#                               and SWEPT_KS = 5, 9, 10, 20
#   fold_similarity_spaces      the matrices one fold's settings read -- each
#                               model's top-2 single representations, the two
#                               uncapped augmented spaces, the three hybrids
#                               (2026-10-03) -- every one
#                               built through a noise_cache wrapper
#   fold_neighbours             one neighbour list per setting, through
#                               noise_cache.cached_neighbours
#   RowSink / dprime_rows       the D-versus-D' rows (d_vs_dprime), appended as
#                               each fold finishes, skipped on resume, and the
#                               random / shuffled twins when
#                               KIRBY_NOISE_CONTROLS=1
#   FoldCheckpoint              per-fold partial results for the experiments
#                               that change no data, written as each unit
#                               finishes and read back on resume
#
# ⛔ THE SETTING IS SEPARATE FROM THE MODEL INPUT. The model is fitted on what
# each experiment fitted it on before 2026-09-25; only the neighbour lookup
# moves. KIRBY_SIMILARITY_SETTINGS=model_input sweeps MODEL_INPUT alone, whose
# lists are today's query bit for bit (similarity_space.py, module docstring),
# and every result key an experiment wrote before 2026-09-25 is computed by
# the code that computed it then.
# =============================================================================

# The downstream models every D-versus-D' row is fitted with, on top of
# whatever the experiment fits itself. Owner's decision 2026-09-25: RF and SVM
# at least. Same tuple as d_vs_dprime.DEFAULT_MODELS.
SWEEP_ROW_MODELS = ('rf', 'svm')


def sweep_settings(dataset_name=None, n_molecules=None, has_smiles=True,
                   available_spaces=None):
    """The similarity settings this run sweeps, for this dataset's group.

    KNN_SEARCH_SPACE_RECORD (Adelaide, 2026-10-05): the group comes from the
    molecule count (similarity_space.knn_group: more than 5000 is fast_knn),
    and decides the spaces and distances. model_input is cut (axis 1, number
    10) and is no longer put first. No SMILES removes tanimoto_ecfp4.
    KIRBY_SIMILARITY_SETTINGS still narrows the list (settings_from_env), and
    raises on a name the group cut.

    Pass the dataset name and its molecule count. Called with neither, it
    returns full_knn's list and prints that it did.
    """
    from experiments.noise import similarity_space as ss
    if dataset_name is None and n_molecules is None:
        print("  sweep_settings: no dataset or molecule count given, so "
              "full_knn's settings are swept")
        group = 'full_knn'
    else:
        group = ss.knn_group(dataset_name, n_molecules)
    avail = set(ss.group_spaces(group, has_smiles=has_smiles,
                                 dataset_name=dataset_name))
    if available_spaces is not None:
        avail &= set(available_spaces)
    return list(ss.settings_from_env(available_spaces=sorted(avail),
                                     group=group))


def sweep_ks():
    """Every neighbour count swept: 5, 10, 20 (similarity_space.SWEPT_KS,
    KNN_SEARCH_SPACE_RECORD axis 4; 9 was cut 2026-10-05)."""
    from experiments.noise.similarity_space import SWEPT_KS
    return tuple(int(k) for k in SWEPT_KS)


def sweep_schemes():
    """How much each neighbour counts: 'rank' and 'inverse_square'
    (similarity_space.SWEPT_NEIGHBOUR_SCHEMES, KNN_SEARCH_SPACE_RECORD axis 5)."""
    from experiments.noise.similarity_space import SWEPT_NEIGHBOUR_SCHEMES
    return tuple(SWEPT_NEIGHBOUR_SCHEMES)


# The per-dataset screen (KNN_SEARCH_SPACE_RECORD, "Screening the kNN settings
# per dataset", Adelaide 2026-10-05): every (setting, neighbour count, counting
# scheme) is scored with no model fit on how well its neighbours predict a
# molecule's activity, and only the best SCREEN_TOP are refitted.
# KIRBY_NOISE_SCREEN_TOP overrides; 0 refits every combination.
SCREEN_TOP = 3

# Which molecules the screen predicts. Three ways were weighed:
# 'training_rows': each training molecule from its neighbours among the other
#   training molecules, against the damaged training labels.
# 'held_out': each held-out molecule from its training neighbours, against its
#   clean label (how the record measured it; but those are the molecules every
#   experiment reports on).
# 'inner_holdout': below.
SCREEN_SCORED_ON = 'inner_holdout'
# DECIDED 2026-10-06 (Adelaide delegated; f39_screen_choice measured the other
# two): 'inner_holdout' holds back a fifth of the training molecules, as whole
# scaffold groups, builds every setting's lists on the other four fifths (the
# forest refitted there too), and scores the held-back molecules against the
# training labels the experiment holds. Nothing held out by the experiment is
# read. On herg_ki (5 folds) and caco2 (fold 0), 'training_rows' put forest
# settings in all 3 kept slots on every fold and gave up 0.036 to 0.057
# detection AUROC on 5 of the 6; 'held_out' gave up 0.015 to 0.027 but chooses
# on the reported molecules. 'inner_holdout' itself is not yet measured.
SCREEN_INNER_SHARE = 0.2


def screen_top():
    raw = os.environ.get('KIRBY_NOISE_SCREEN_TOP')
    if raw not in (None, ''):
        return int(raw)
    return SCREEN_TOP


def _screen_prediction(y_src, idx, dist, scheme, is_clf, metric_name):
    """Neighbour prediction for the screen. On classification scored by AUC,
    the weighted share of the higher class; otherwise the weighted label."""
    from experiments.noise import similarity_space as ss
    y_src = np.asarray(y_src)
    if is_clf and 'auc' in str(metric_name or '').lower():
        classes = np.unique(y_src)
        w = ss.neighbour_weights(dist, scheme)
        hit = (y_src[np.asarray(idx)] == classes[-1])
        return (w * hit).sum(1) / w.sum(1)
    return ss.weighted_neighbour_label(
        y_src if is_clf else y_src.astype(float), idx, dist, scheme,
        is_clf=is_clf)


def _inner_holdout_lists(settings, inner):
    """Neighbour lists for the inner hold-out: the held-back molecules (a
    SCREEN_INNER_SHARE of the training scaffold groups, chosen by a fixed
    hash of the group label) looking for neighbours among the rest of the
    training molecules. Returns (nbrs, held_mask) or (None, None)."""
    import zlib
    sp_tr = inner.get('spaces_train') or {}
    y = np.asarray(inner['y_train'])
    g = inner.get('groups_train')
    n = len(y)
    keys = (np.asarray(g) if g is not None else np.arange(n)).astype(str)
    held = np.array([(zlib.crc32(k.encode()) % 1000) < SCREEN_INNER_SHARE * 1000
                     for k in keys])
    if held.sum() < 5 or (~held).sum() < 25:
        return None, None
    a, b = np.where(~held)[0], np.where(held)[0]
    sub_tr = {k: np.asarray(v)[a] for k, v in sp_tr.items() if v is not None}
    sub_va = {k: np.asarray(v)[b] for k, v in sp_tr.items() if v is not None}
    smi = inner.get('smiles_train')
    nb = fold_neighbours(inner['dataset_name'], settings, sub_tr, sub_va, y[a],
                         smiles_train=None if smi is None else [smi[i] for i in a],
                         smiles_val=None if smi is None else [smi[i] for i in b],
                         is_clf=inner.get('is_clf', False))
    return nb, held


def screen_settings(settings, nbrs, y_train, *, metric, is_clf, y_val=None,
                    ks=None, schemes=None, top=None, inner=None):
    """Score every (setting, k, scheme) with no model fit; keep the best `top`.

    nbrs: {setting name: neighbour_lists result} (fold_neighbours).
    y_train: the training labels the experiment holds (damaged).
    y_val: the held-out molecules' labels, read only when SCREEN_SCORED_ON is
        'held_out' (and then the lists must hold validation rows).
    metric: the get_domain_metric triple; the dataset's own metric and
        direction decide the ranking.

    Returns (kept, table): kept is [(SimilaritySetting, k, scheme)] best
    first; table is every scored combination as a dict, for the result file.
    top 0 or None-with-SCREEN_TOP-0 keeps every combination.
    """
    from experiments.noise import similarity_space as ss
    metric_name, metric_fn, lower = metric
    ks = sweep_ks() if ks is None else tuple(ks)
    schemes = sweep_schemes() if schemes is None else tuple(schemes)
    top = screen_top() if top is None else int(top)
    scored_on = SCREEN_SCORED_ON
    if scored_on == 'inner_holdout':
        # inner: {'dataset_name', 'spaces_train' (the same dict the lists were
        # built from), 'y_train', 'groups_train', 'smiles_train', 'is_clf'}.
        got = (None, None) if inner is None else _inner_holdout_lists(settings,
                                                                      inner)
        if got[0] is None:
            scored_on = 'training_rows'
            if inner is None:
                print("  screen_settings: no inner hold-out spaces passed, so "
                      "the screen scores the training molecules")
        else:
            nbrs, held = got
            y_val = np.asarray(y_train)[held]
            y_train = np.asarray(y_train)[~held]
    on_val = scored_on in ('held_out', 'inner_holdout')
    table, cands = [], []
    for s in settings:
        L = nbrs.get(s.name)
        if L is None or not L.get('available'):
            continue
        for k in usable_ks(L, ks):
            try:
                idx, dist = ss.neighbours_at(L, k, which='val' if on_val
                                             else 'train')
            except Exception as exc:
                table.append({'setting': s.name, 'k': int(k), 'scheme': None,
                              'score': None,
                              'error': f'{type(exc).__name__}: {exc}'})
                continue
            y_true = np.asarray(y_val if on_val else y_train)
            for sch in schemes:
                try:
                    pred = _screen_prediction(y_train, idx, dist, sch, is_clf,
                                              metric_name)
                    sc = float(metric_fn(y_true, pred))
                except Exception as exc:
                    table.append({'setting': s.name, 'k': int(k),
                                  'scheme': sch, 'score': None,
                                  'error': f'{type(exc).__name__}: {exc}'})
                    continue
                table.append({'setting': s.name, 'k': int(k), 'scheme': sch,
                              'score': sc})
                if np.isfinite(sc):
                    cands.append((sc, s, int(k), sch))
    cands.sort(key=lambda c: ((c[0] if lower else -c[0]), c[1].name, c[2],
                              c[3]))
    kept = [(s, k, sch) for _, s, k, sch in (cands if top <= 0
                                             else cands[:top])]
    keep_names = {(s.name, k, sch) for s, k, sch in kept}
    for r in table:
        r['kept'] = (r['setting'], r['k'], r['scheme']) in keep_names
        r['scored_on'] = scored_on
        r['metric'] = metric_name
        r['lower_is_better'] = bool(lower)
    return kept, table


def usable_ks(nbrs, ks=None):
    """The ks in `ks` (default sweep_ks()) that a neighbour_lists result holds.

    MODEL_INPUT holds one exact list per k (`exact_by_k`); every other setting
    holds k_eff neighbours per molecule, stably sorted, so any k up to k_eff
    is exact. A k the lists cannot give is left out rather than cut short."""
    ks = sweep_ks() if ks is None else tuple(ks)
    if not nbrs or not nbrs.get('available'):
        return ()
    exact = nbrs.get('exact_by_k')
    if exact is not None:
        return tuple(k for k in ks if k in exact)
    return tuple(k for k in ks if k <= int(nbrs.get('k_eff') or 0))


def settings_need_spaces(settings):
    """The set of spaces a list of settings reads."""
    return {s.space for s in settings}


def _scale_pair(X_tr, X_va):
    # 0/1 features left unscaled, continuous ones standardised (handoff 2.3
    # item 17), as noise_cache.cached_scaled does for the base spaces.
    from kirby.scaling import FeatureScaler
    sc = FeatureScaler()
    return sc.fit_transform(X_tr), sc.transform(X_va)


def fold_similarity_spaces(dataset_name, settings, *, model_input_tr,
                           model_input_va, reps, train_idx, val_idx, y_train,
                           groups_train=None, base_reps=(), recipes=None,
                           augmentations=None, model_name=None,
                           model_factory=None, metric=None, task_type=None,
                           extra_spaces=()):
    """The standardised matrices one fold's similarity settings read.

    Args:
        settings: the SimilaritySetting list; only the spaces it names are
            built, so a run under KIRBY_SIMILARITY_SETTINGS=model_input builds
            nothing here.
        model_input_tr / model_input_va: the matrix the model is fitted on,
            already standardised. Read by MODEL_INPUT and forest_proximity.
        reps: {name: full matrix}; train_idx / val_idx slice them.
        y_train: the labels the caller holds for the training rows (noisy).
            Every build below chooses its features from them, as the
            experiment's own hybrid build does.
        base_reps: the model's top-2 single representations, rank 1 first
            (_shared.noise_base_reps_by_model). base_1, base_2 and the two
            uncapped augmented spaces are built from them.
        recipes: {'hybrid': (Blueprint, aug_condition),
                  'hybrid__general2': (Blueprint, aug_condition),
                  'hybrid_similarity': (Blueprint, aug_condition)}, the model's
            precomputed recipes (pool_recipes_by_family's space names). A
            space whose recipe is absent is reported unavailable, never
            faked.
        augmentations: {name: full matrix}, data['augmentations'].
        model_name / model_factory / metric: the downstream model and the
            get_domain_metric triple, for the builds' accept-or-reject step.
        extra_spaces: spaces to build whether or not a setting names them:
            'base_1', 'base_2', or a recipe space by its key in `recipes`
            (noise_helpers.fold_model_inputs builds a model's four fitted
            configurations this way) -- e.g. 'hybrid__general2' and
            'hybrid__distance1', the owner's set (top 2 general + top 1
            distance hybrids per model and family, 2026-10-02), which n42
            fits models on. A SimilaritySetting accepts only the names in
            similarity_space.HYBRID_SPACES ('hybrid', 'hybrid__general2' since
            2026-10-03, 'hybrid_similarity'), so the others cannot come in
            through `settings`. Default () builds only what `settings`
            names.

    Returns (spaces_train, spaces_val, unavailable), where unavailable is
    {space: reason}. Every build goes through a noise_cache wrapper
    (cached_scaled, cached_hybrid_for_fold), so a space shared by several
    experiments, conditions at level 0, or models holding one recipe is built
    once.
    """
    from experiments.noise import noise_cache as nc
    from experiments.noise import similarity_space as ss
    from experiments.noise.noise_helpers import recipe_is_geometry
    _non_recipe = {'base_1', 'base_2'} | set(ss.AUG_SPACES)
    need = settings_need_spaces(settings) | set(extra_spaces or ())
    tr = {'model_input': model_input_tr}
    va = {'model_input': model_input_va}
    unavailable = {}
    tr_idx = np.asarray(train_idx)
    va_idx = np.asarray(val_idx)
    metric_name, metric_fn, lower_is_better = (metric if metric is not None
                                               else (None, None, None))
    base_reps = list(base_reps or [])[:2]

    for i in (1, 2):
        wanted_base = f'base_{i}' in need
        wanted_aug = [st for st in ss.SWEPT_AUG_STRATEGIES
                      if ss.aug_space_name(i, st) in need]
        if not wanted_base and not wanted_aug:
            continue
        rn = base_reps[i - 1] if len(base_reps) >= i else None
        if rn is None or rn not in reps:
            why = (f'this model has no rank-{i} single representation in the '
                   f'stored search' if rn is None else
                   f'{rn} is not loaded in this run')
            if wanted_base:
                unavailable[f'base_{i}'] = why
            for st in wanted_aug:
                unavailable[ss.aug_space_name(i, st)] = why
            continue
        if wanted_base:
            tr[f'base_{i}'], va[f'base_{i}'] = nc.cached_scaled(
                dataset_name, reps[rn], tr_idx, va_idx)
        if wanted_aug:
            # c21's B_aug, built per fold (similarity_space.base_aug_blueprint):
            # the base representation at full width plus the augmentation
            # features its admission method takes, the base itself excluded
            # from the donors as c21:429 excludes it.
            donors = {a: X for a, X in (augmentations or {}).items() if a != rn}
            if not donors:
                for st in wanted_aug:
                    unavailable[ss.aug_space_name(i, st)] = (
                        'no augmentation representation is loaded for this '
                        'dataset, so there is nothing to add to the base')
                continue
            X_base = reps[rn]
            for st in wanted_aug:
                sp = ss.aug_space_name(i, st)
                try:
                    bp_aug = ss.base_aug_blueprint(
                        rn, int(np.asarray(X_base).shape[1]),
                        metric_name=metric_name, task_type=task_type,
                        lower_is_better=lower_is_better, strategy=st)
                    _model = model_factory() if model_factory else None
                    built = nc.cached_hybrid_for_fold(
                        dataset_name, {rn: X_base[tr_idx]}, {rn: X_base[va_idx]},
                        y_train, bp_aug, groups_train, _model, metric_fn,
                        lower_is_better, model_name=model_name,
                        metric_name=metric_name, train_idx=tr_idx,
                        augmentations_train={a: X[tr_idx]
                                             for a, X in donors.items()},
                        augmentations_val={a: X[va_idx]
                                           for a, X in donors.items()})
                except Exception as exc:
                    built = (None, None)
                    unavailable[sp] = f'build raised {type(exc).__name__}: {exc}'
                if built[0] is None:
                    unavailable.setdefault(sp, 'the augmented build failed')
                    continue
                tr[sp], va[sp] = _scale_pair(built[0], built[1])

    _hyb_spaces = [sp for sp in ss.HYBRID_SPACES if sp in need]
    _hyb_spaces += [sp for sp in (extra_spaces or ())
                    if sp not in _hyb_spaces and sp not in _non_recipe]
    for sp in _hyb_spaces:
        rec = (recipes or {}).get(sp)
        if rec is None or rec[0] is None:
            unavailable[sp] = (
                'no predictive general rank-2 recipe for this model (the '
                "rule read's 'general2' slot, pool_recipes_by_family)"
                if sp == 'hybrid__general2' else
                'no stored recipe of this family for this model '
                '(pool_recipes_by_family)')
            continue
        bp, aug_cond = rec
        geometry = recipe_is_geometry(bp)
        augs = augmentations if (aug_cond or 'none') != 'none' else None
        try:
            built = nc.cached_hybrid_for_fold(
                dataset_name, {k: v[tr_idx] for k, v in reps.items()},
                {k: v[va_idx] for k, v in reps.items()}, y_train, bp,
                groups_train,
                None if (geometry or model_factory is None) else model_factory(),
                metric_fn, lower_is_better,
                model_name=None if geometry else model_name,
                metric_name=metric_name, train_idx=tr_idx,
                augmentations_train=(None if augs is None else
                                     {a: X[tr_idx] for a, X in augs.items()}),
                augmentations_val=(None if augs is None else
                                   {a: X[va_idx] for a, X in augs.items()}))
        except Exception as exc:
            built = (None, None)
            unavailable[sp] = f'build raised {type(exc).__name__}: {exc}'
        if built[0] is None:
            unavailable.setdefault(sp, 'the hybrid build failed')
            continue
        tr[sp], va[sp] = _scale_pair(built[0], built[1])
    return tr, va, unavailable


def fold_neighbours(dataset_name, settings, spaces_train, spaces_val, y_train,
                    *, exclude_groups=None, smiles_train=None, smiles_val=None,
                    is_clf=False):
    """{setting name: neighbour_lists result}, each through
    noise_cache.cached_neighbours. A setting whose space is missing comes back
    with available=False and its reason; the caller skips it and records why.

    exclude_groups: pass the fold's scaffold groups exactly where the method
    being fed passed them before 2026-09-25 (the neighbour detectors in n07),
    and None where it did not (label_smooth, knn_smooth, n11, n12), so the
    MODEL_INPUT lists stay today's lists."""
    from experiments.noise import noise_cache as nc
    out = {}
    for s in settings:
        try:
            out[s.name] = nc.cached_neighbours(
                dataset_name, s, spaces_train=spaces_train,
                spaces_val=spaces_val, y_train=y_train,
                exclude_groups=exclude_groups,
                smiles_train=smiles_train, smiles_val=smiles_val,
                is_clf=is_clf)
        except Exception as exc:
            out[s.name] = {'available': False, 'setting_name': s.name,
                           'reason': f'{type(exc).__name__}: {exc}'}
    return out


def rows_path(output_dir, exp_full_name, dataset_name, condition=None):
    """<output>/<exp_full_name>/<dataset>[__condition].rows.jsonl, next to the
    result file sweep_conditions / run_all write."""
    sfx = f'__{condition}' if condition else ''
    return os.path.join(str(output_dir), exp_full_name,
                        f'{dataset_name}{sfx}.rows.jsonl')


class RowSink:
    """The D-versus-D' rows of one (experiment, dataset[, condition]) file.

    Rows are appended as each fold finishes (d_vs_dprime.append_rows, flushed
    and fsynced) and every key already on disk is skipped (done_keys), so a
    killed run resumes where it stopped. output_dir None disables the sink:
    nothing is fitted for it and nothing is written.

    ⛔ --force-rerun does not reach this file (sweep_conditions consumes it),
    so a forced rerun keeps the rows on disk. Delete the .rows.jsonl to redo
    them; the caches are wiped the same way (locked rule).
    """

    def __init__(self, output_dir, exp_full_name, dataset_name, condition=None):
        from experiments.noise.d_vs_dprime import done_keys
        self.path = (None if output_dir is None else
                     rows_path(output_dir, exp_full_name, dataset_name,
                               condition))
        self.done = done_keys(self.path) if self.path else set()
        self.n_on_disk_at_start = len(self.done)
        self.n_written = 0
        self.n_twins_skipped = 0

    @property
    def enabled(self):
        return self.path is not None

    def is_done(self, identity, models):
        from experiments.noise.d_vs_dprime import row_key
        return all(row_key(dict(identity, model=m)) in self.done
                   for m in models)

    def write(self, rows):
        from experiments.noise.d_vs_dprime import row_key, append_rows
        new = [r for r in rows if row_key(r) not in self.done]
        if not new or not self.enabled:
            return 0
        append_rows(self.path, new)
        for r in new:
            self.done.add(row_key(r))
        self.n_written += len(new)
        return len(new)

    def summary(self):
        return {'rows_file': self.path,
                'rows_on_disk_at_start': self.n_on_disk_at_start,
                'rows_written_this_run': self.n_written,
                'control_twins_skipped': self.n_twins_skipped}


def row_models(extra=()):
    """The models a D' row is fitted with: the experiment's own, once each.

    Until 2026-10-05 rf and svm (SWEEP_ROW_MODELS) were always added, which
    fitted models outside the dataset's two decided ones (R1). SWEEP_ROW_MODELS
    is kept for readers of old rows only."""
    return tuple(dict.fromkeys(m for m in extra if m))


def cleaned_from_weights(X_tr, y_tr, weights):
    """A weight-shaped D' in the dict shape cleaned_training_set returns."""
    return {'X': X_tr, 'y': np.asarray(y_tr), 'keep': None,
            'sample_weight': np.asarray(weights, dtype=float),
            'label_column': None}


def cleaned_from_labels(X_tr, y_new):
    """A label-moving D' in the dict shape cleaned_training_set returns."""
    y_new = np.asarray(y_new)
    return {'X': X_tr, 'y': y_new, 'keep': None, 'sample_weight': None,
            'label_column': y_new}


def _twin_seed(identity):
    """A deterministic seed from a row's identity, so a control twin is the
    same draw on every rerun and a different one for every (method, setting,
    fold, draw)."""
    import json
    import zlib
    blob = json.dumps({k: identity.get(k) for k in sorted(identity)},
                      sort_keys=True, default=str)
    return (int(zlib.crc32(blob.encode('utf-8'))) & 0x7FFFFFFF,)


def dprime_rows(sink, *, dataset_name, cleaned, X_tr, y_D, X_va, y_va, metric,
                task_type, y_clean, models, identity, is_clf=False):
    """Score D and D' for every model in `models`, write the rows, and when
    controls_enabled() the random and shuffled twins of D' as two more
    methods ('<method>__random_twin', '<method>__shuffled_twin').

    identity: every d_vs_dprime.KEY_FIELDS entry but `model` -- dataset,
    experiment, method, method_setting (it carries k), similarity_setting,
    model_input, noise_condition, level, draw, fold.

    cleaned None writes the row with score_Dprime nan (the method declined),
    so a declined cell is on disk and not re-attempted on resume.

    Every fit goes through noise_cache.cached_fit_predict, so the D fit and
    the clean fit of one (fold, features, labels, model) are fitted once
    however many methods and settings are scored against them.

    ⛔ Controls on a classification LABEL move are skipped and counted: a
    dealt or permuted label change need not be a class (controls.py). Both
    twins need the method re-run on a flagged column there, which is a
    per-method rerun= the experiment has not been given.
    """
    from experiments.noise.d_vs_dprime import score_d_dprime, rows_for
    from experiments.noise import controls as ctl
    if not sink.enabled:
        return None
    models = tuple(dict.fromkeys(models))
    scores = None
    if not sink.is_done(identity, models):
        scores = score_d_dprime(dataset_name, cleaned, X_tr, y_D, X_va, y_va,
                                metric=metric, task_type=task_type,
                                y_tr_clean=y_clean, models=models)
        sink.write(rows_for(scores, metric=metric, **identity))
    if cleaned is None or not ctl.controls_enabled():
        return scores
    shape = ctl._shape(cleaned)
    for twin in ('random_twin', 'shuffled_twin'):
        tid = dict(identity, method=f"{identity['method']}__{twin}")
        if sink.is_done(tid, models):
            continue
        if is_clf and shape == 'label':
            sink.n_twins_skipped += 1
            continue
        seed = _twin_seed(tid)
        if twin == 'random_twin':
            tw, _n = ctl.random_twin(cleaned, X_tr, y_D, seed)
        else:
            tw = ctl.shuffled_twin(cleaned, X_tr, y_D, seed)
        if tw is None:
            continue
        sc = score_d_dprime(dataset_name, tw, X_tr, y_D, X_va, y_va,
                            metric=metric, task_type=task_type,
                            y_tr_clean=y_clean, models=models)
        sink.write(rows_for(sc, metric=metric, **tid))
    return scores


def row_identity(*, dataset_name, experiment, method, method_setting,
                 similarity_setting, model_input, noise_condition, level,
                 draw, fold):
    """The KEY_FIELDS of a row except `model`, with plain JSON values."""
    return {'dataset': dataset_name, 'experiment': experiment,
            'method': str(method), 'method_setting': method_setting,
            'similarity_setting': similarity_setting,
            'model_input': str(model_input),
            'noise_condition': (None if noise_condition is None
                                else str(noise_condition)),
            'level': None if level is None else float(level),
            'draw': None if draw is None else int(draw),
            'fold': int(fold)}


# The similarity_setting written on a row whose method reads no neighbours --
# remove, downweight and the other score-driven cleanups under a detector
# that is not a neighbour detector. Such a method is one row, not one per
# setting, because every setting would give it the same numbers.
NO_NEIGHBOURS = 'none'


class FoldCheckpoint:
    """Per-unit partial results for an experiment that changes no data.

    One JSON line per finished unit (a fold, or a (fold, draw) pass),
    {'ckpt_key': [...], 'value': {...}}, appended and fsynced as the unit
    finishes and read back on the next run, so a killed run redoes only the
    unit in flight. Floats round-trip through JSON exactly (repr), so a
    resumed run aggregates the same numbers bit for bit.

    The file sits beside the rows: <output>/<exp>/<dataset>[__cond].<tag>.jsonl.
    Its lines carry no 'dataset' field, so check_noise_run.read_rows skips
    them. `fingerprint` (the caller's input_fingerprint or any JSON-able
    identity of the inputs) is stored in every line; a line with another
    fingerprint is ignored, so changed inputs recompute.
    """

    def __init__(self, output_dir, exp_full_name, dataset_name, condition=None,
                 tag='ckpt', fingerprint=None):
        import json
        sfx = f'__{condition}' if condition else ''
        self.path = (None if output_dir is None else
                     os.path.join(str(output_dir), exp_full_name,
                                  f'{dataset_name}{sfx}.{tag}.jsonl'))
        self.fingerprint = fingerprint
        self._fp = json.dumps(fingerprint, sort_keys=True, default=str)
        self.entries = {}
        self.n_resumed = 0
        if self.path and os.path.exists(self.path):
            with open(self.path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if json.dumps(rec.get('fingerprint'), sort_keys=True,
                                  default=str) != self._fp:
                        continue
                    self.entries[self._k(rec.get('ckpt_key'))] = rec.get('value')

    @staticmethod
    def _k(key):
        import json
        return json.dumps(key, sort_keys=True, default=str)

    def get(self, key):
        v = self.entries.get(self._k(key))
        if v is not None:
            self.n_resumed += 1
        return v

    def put(self, key, value):
        import json
        from experiments.noise.d_vs_dprime import _jsonable
        self.entries[self._k(key)] = value
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        def _clean(v):
            if isinstance(v, dict):
                return {str(k): _clean(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [_clean(x) for x in v]
            return _jsonable(v)
        with open(self.path, 'a') as fh:
            fh.write(json.dumps({'ckpt_key': key, 'fingerprint':
                                 json.loads(self._fp), 'value': _clean(value)},
                                allow_nan=True, default=str) + '\n')
            fh.flush()
            os.fsync(fh.fileno())

    def cached(self, key, compute):
        """The stored value for `key`, or compute(), stored. A None result is
        not stored, so a failed unit is retried."""
        v = self.get(key)
        if v is not None:
            return v
        v = compute()
        if v is not None:
            self.put(key, v)
        return v


# The cleanups whose neighbour lookup the similarity setting replaces
# (cleaned_training_set's refusal rule, n07:709-720). prox_smooth reads only
# the forest_proximity setting.
NEIGHBOUR_CLEANUPS = ('label_smooth', 'knn_smooth', 'prox_smooth')


def _n07_sweep_pass(ctx, *, model_name, model_input, X_tr, X_va, y_tr_noisy,
                    y_tr_clean, y_val, noise_scores, strategies, train_idx,
                    val_idx, groups_tr, fold, draw, recipes, base_reps,
                    corrupted_mask=None, noise_strategy=None,
                    detection_acc=None, grade_detectors=()):
    """The D-versus-D' rows of one (model input, fold, noise draw) pass of n07,
    and the detection grading of every neighbour detector under every
    similarity setting. Added 2026-09-25 (plan sections 1-2).

    ctx: the run's constants -- dataset_name, sink, settings, ks, metric,
    task_type, is_clf, noise_fraction, condition, level, smiles, reps,
    augmentations, experiment.

    Rows are fitted with the pass's own model only (model_name). Per cleanup
    in `strategies` (rewritten 2026-10-05 to say what the code below does):
      a cleanup that reads no neighbours    one row, similarity setting 'none',
                                            the scores the pass already has
                                            (the detector DETECTOR, which is
                                            not a neighbour detector)
      label_smooth, knn_smooth              one row per (setting, k, counting
                                            scheme) the screen kept
                                            (screen_settings, top 3); method
                                            setting 'k=..|scheme=..'
      prox_smooth                           as above, under forest_proximity
                                            settings only
    The settings are ctx['settings'] (sweep_settings: the dataset group's
    spaces and distances, no model_input setting), k in sweep_ks() and the
    schemes in sweep_schemes(). One set of neighbour lists per pass, with a
    molecule's scaffold-mates kept in (axis 6 of KNN_SEARCH_SPACE_RECORD).
    When DETECTOR is a neighbour detector every score-driven cleanup is one row
    per kept (setting, k) instead, scored on those same lists.

    detection_acc / grade_detectors: when a corrupted subset exists, every name
    in grade_detectors that reads neighbours is graded under every (setting, k)
    -- not screened, since grading fits no model -- into
    detection_acc[detector][setting][k], a list of detection_scores readings.
    The detectors read the counting scheme too (2026-10-06).

    Returns this pass's top screened setting, for the main table
    (2026-10-06): {'setting', 'space', 'k', 'scheme', 'lists' (the neighbour
    lists carrying the scheme), 'detector_scores' (DETECTOR's scores on those
    lists when it is a neighbour detector, else None)}, or {'reason': why}
    when the screen kept nothing.
    """
    from experiments.noise import noise_cache as nc
    from experiments.noise.noise_cache import NEIGHBOUR_DETECTORS
    sink = ctx['sink']
    ds = ctx['dataset_name']
    is_clf = ctx['is_clf']
    settings = ctx['settings']
    # 2026-10-05 (chat 5, R1): the pass's own model only. Each configuration
    # belongs to one of the dataset's two decided models; the rf and svm that
    # row_models adds (SWEEP_ROW_MODELS) are not decided models.
    models = (model_name,)
    # invvar_weight's measurement counts, sliced to this fold's training rows
    # (ctx['measurement_counts'] is the whole column, or None).
    _mc_all = ctx.get('measurement_counts')
    _mc_tr = (None if _mc_all is None
              else np.asarray(_mc_all)[np.asarray(train_idx)])
    base_id = dict(dataset_name=ds, experiment=ctx['experiment'],
                   model_input=model_input,
                   noise_condition=ctx['condition'], level=ctx['level'],
                   draw=draw, fold=fold)
    det_reads = DETECTOR in NEIGHBOUR_DETECTORS
    grade = [d for d in grade_detectors if d in NEIGHBOUR_DETECTORS]
    want_grading = (corrupted_mask is not None and detection_acc is not None
                    and bool(grade))
    active = [st for st in strategies
              if st not in ('no_noise', 'noisy_baseline')]
    _base = lambda st: split_strategy(st)[0]  # noqa: E731
    reads_nbrs = [st for st in active if _base(st) in NEIGHBOUR_CLEANUPS]
    fixed = [st for st in active if _base(st) not in NEIGHBOUR_CLEANUPS
             and not (det_reads and _base(st) in SCORE_DRIVEN_CLEANUPS)]
    swept_scored = [st for st in active if _base(st) not in NEIGHBOUR_CLEANUPS
                    and det_reads and _base(st) in SCORE_DRIVEN_CLEANUPS]

    def _clean(st, scores, **sim):
        # The flag share comes from the strategy name (FLAG_SHARES), never
        # from ctx['noise_fraction'], the injector's share (2026-09-28). A
        # cleanup that reads none is handed noise_fraction only because the
        # argument is positional.
        cl, sh = split_strategy(st)
        try:
            return cleaned_training_set(
                cl, X_tr, y_tr_noisy, scores,
                ctx['noise_fraction'] if sh is None else sh,
                is_clf=is_clf, groups=groups_tr, dataset_name=ds,
                measurement_counts=_mc_tr, **sim)
        except Exception as exc:
            _sn = (sim.get('similarity_neighbours') or {}).get('setting_name')
            print(f"      {st} ({model_input}, {_sn}, "
                  f"k={sim.get('similarity_k')}) failed: "
                  f"{type(exc).__name__}: {exc}")
            return None

    def _row(st, cleaned, similarity_setting, method_setting):
        if sink is None or not sink.enabled:
            return
        # The flag share goes into method_setting, the cleanup into method.
        cl, sh = split_strategy(st)
        if sh is not None:
            _fs = f'flag_share={sh:g}'
            method_setting = (_fs if method_setting is None
                              else f'{method_setting}|{_fs}')
        dprime_rows(sink, dataset_name=ds, cleaned=cleaned, X_tr=X_tr,
                    y_D=y_tr_noisy, X_va=X_va, y_va=y_val,
                    metric=ctx['metric'], task_type=ctx['task_type'],
                    y_clean=y_tr_clean, models=models, is_clf=is_clf,
                    identity=row_identity(
                        method=cl, method_setting=method_setting,
                        similarity_setting=similarity_setting, **base_id))

    for st in fixed:
        needs = _base(st) in SCORE_DRIVEN_CLEANUPS
        if needs and noise_scores is None:
            continue
        _row(st, _clean(st, noise_scores if needs else None), NO_NEIGHBOURS,
             None)

    if not (reads_nbrs or swept_scored or want_grading):
        return {'reason': 'no cleanup in this pass reads neighbours'}
    spaces_tr, spaces_va, unavail = fold_similarity_spaces(
        ds, settings, model_input_tr=X_tr, model_input_va=X_va,
        reps=ctx['reps'], train_idx=train_idx, val_idx=val_idx,
        y_train=y_tr_noisy, groups_train=groups_tr, base_reps=base_reps,
        recipes=recipes, augmentations=ctx['augmentations'],
        model_name=model_name, model_factory=ctx['factories'].get(model_name),
        metric=ctx['metric'], task_type=ctx['task_type'])
    for sp, why in unavail.items():
        ctx['unavailable'].setdefault(f'{model_input}|{sp}', why)
    smi = ctx.get('smiles')
    smi_tr = None if smi is None else [smi[i] for i in np.asarray(train_idx)]
    smi_va = None if smi is None else [smi[i] for i in np.asarray(val_idx)]
    # One set of lists for the cleanups and the detectors alike: scaffold-mates
    # stay in (axis 6 of KNN_SEARCH_SPACE_RECORD, Adelaide 2026-10-05). Before
    # that the detectors read a second set with the scaffold groups excluded.
    nb = (fold_neighbours(ds, settings, spaces_tr, spaces_va, y_tr_noisy,
                          smiles_train=smi_tr, smiles_val=smi_va,
                          is_clf=is_clf)
          if (reads_nbrs or swept_scored or want_grading) else {})
    for s in settings:
        L = nb.get(s.name)
        if L is not None and not L.get('available'):
            ctx['unavailable'].setdefault(f'{model_input}|{s.name}',
                                          L.get('reason'))
    # Grading a detector fits no downstream model, so every (setting, k) is
    # graded. Only the refits are screened (screen_settings).
    if want_grading:
        # Graded under every counting scheme too (2026-10-06: the scheme now
        # reaches the detectors); filed under '<setting>|<scheme>'.
        for s in settings:
            L = nb.get(s.name)
            for k in usable_ks(L):
                for sch in sweep_schemes():
                    Ls = dict(L, scheme=sch)
                    for d in grade:
                        sc = nc.cached_detector_scores(
                            ds, X_tr, y_tr_noisy, is_clf=is_clf, detector=d,
                            groups=groups_tr, similarity_neighbours=Ls,
                            similarity_k=k)
                        if sc is None:
                            continue
                        det = detection_scores(corrupted_mask, sc,
                                               noise_strategy)
                        if det is not None:
                            (detection_acc.setdefault(d, {})
                             .setdefault(f'{s.name}|{sch}', {})
                             .setdefault(str(k), []).append(det))
    if not (reads_nbrs or swept_scored):
        return {'reason': 'no cleanup in this pass reads neighbours'}
    kept, table = screen_settings(settings, nb, y_tr_noisy,
                                  metric=ctx['metric'], is_clf=is_clf,
                                  y_val=y_val, inner={
                                      'dataset_name': ds,
                                      'spaces_train': spaces_tr,
                                      'y_train': y_tr_noisy,
                                      'groups_train': groups_tr,
                                      'smiles_train': smi_tr,
                                      'is_clf': is_clf})
    ctx.setdefault('screen', []).append({
        'model_input': model_input, 'fold': fold, 'draw': draw,
        'level': ctx['level'], 'condition': ctx['condition'],
        'kept': [f'{s_.name}|k={k_}|{sch}' for s_, k_, sch in kept],
        'table': table})
    top = ({'reason': 'the screen kept no (setting, k, scheme): no setting '
                      'had usable neighbour lists on this pass'}
           if not kept else None)
    for s, k, scheme in kept:
        L = dict(nb[s.name], scheme=scheme)
        scores_k = None
        if swept_scored or det_reads:
            # The detector reads the counting scheme too (2026-10-06).
            scores_k = nc.cached_detector_scores(
                ds, X_tr, y_tr_noisy, is_clf=is_clf, detector=DETECTOR,
                groups=groups_tr, similarity_neighbours=L,
                similarity_k=k) if det_reads else None
            if swept_scored:
                for st in swept_scored:
                    _row(st, None if scores_k is None
                         else _clean(st, scores_k), s.name,
                         f'k={k}|scheme={scheme}')
        if top is None:
            # kept[0]: the screen's best (setting, k, scheme), which the main
            # table's neighbour cleanups use (2026-10-06).
            top = {'setting': s.name, 'space': s.space, 'k': int(k),
                   'scheme': scheme, 'lists': L,
                   'detector_scores': scores_k if det_reads else None}
        sc_for_nb = scores_k if det_reads else noise_scores
        if sc_for_nb is None:
            continue
        for st in reads_nbrs:
            if _base(st) == 'prox_smooth' and s.space != 'forest_proximity':
                continue
            _row(st, _clean(st, sc_for_nb, similarity_neighbours=L,
                            similarity_k=k), s.name,
                 f'k={k}|scheme={scheme}')
    return top


def sweep_context(*, experiment, dataset_name, sink, settings, metric,
                  task_type, condition, level, data, factories, **extra):
    """The run's constants every sweep helper reads, in one dict."""
    ctx = {'experiment': experiment, 'dataset_name': dataset_name,
           'sink': sink, 'settings': settings, 'metric': metric,
           'task_type': task_type, 'is_clf': task_type == 'classification',
           'condition': condition, 'level': level,
           'smiles': data.get('smiles'), 'reps': data['reps'],
           'augmentations': data.get('augmentations'),
           'factories': factories, 'unavailable': {}}
    ctx.update(extra)
    return ctx


def similarity_sweep(ctx, *, model_name, model_input, X_tr, X_va, y_tr_noisy,
                     y_tr_clean, y_val, train_idx, val_idx, groups_tr, fold,
                     draw, recipes, base_reps, make_dprimes,
                     exclude_groups=None, level=None, condition=None,
                     extra_models=()):
    """Every (similarity setting, k) of one (model input, fold, draw) pass,
    handed to the experiment's own method, and its D' written as rows.

    make_dprimes(setting, lists, k) yields (method, method_setting, cleaned)
    for each D' the experiment's method makes from those neighbours -- the
    dict cleaned_training_set returns, cleaned_from_weights /
    cleaned_from_labels, or None when it declined. method_setting is the
    method's own knob (strength, alpha); k is prefixed to it on the row.

    exclude_groups is passed to the lists exactly as the method passed it
    before 2026-09-25 (see fold_neighbours), so MODEL_INPUT stays today's list.
    `level` and `condition` override ctx['level'] and ctx['condition'] for
    experiments that sweep them inside one call (n13, c10). Returns {setting name: reason} for the settings that could not be
    built on this pass.
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
                           exclude_groups=exclude_groups, smiles_train=smi_tr,
                           smiles_val=smi_va, is_clf=ctx['is_clf'])
    models = row_models([model_name] + list(extra_models))
    base_id = dict(dataset_name=ds, experiment=ctx['experiment'],
                   model_input=model_input,
                   noise_condition=(ctx['condition'] if condition is None
                                    else condition),
                   level=ctx['level'] if level is None else level,
                   draw=draw, fold=fold)
    missing = {}
    for s in settings:
        L = nbrs.get(s.name)
        if L is None or not L.get('available'):
            why = unavail.get(s.space) or (L or {}).get('reason')
            missing[s.name] = why
            ctx['unavailable'].setdefault(f'{model_input}|{s.name}', why)
    # The per-dataset screen (screen_settings): only the best SCREEN_TOP
    # (setting, k, scheme) combinations are refitted. The lists handed to
    # make_dprimes carry the scheme, so cleaned_training_set weights the
    # neighbours by it without the caller passing anything new.
    kept, table = screen_settings(settings, nbrs, y_tr_noisy,
                                  metric=ctx['metric'], is_clf=ctx['is_clf'],
                                  y_val=y_val, inner={
                                      'dataset_name': ds,
                                      'spaces_train': spaces_tr,
                                      'y_train': y_tr_noisy,
                                      'groups_train': groups_tr,
                                      'smiles_train': smi_tr,
                                      'is_clf': ctx['is_clf']})
    ctx.setdefault('screen', []).append({
        'model_input': model_input, 'fold': fold, 'draw': draw,
        'level': ctx['level'] if level is None else level,
        'condition': ctx['condition'] if condition is None else condition,
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


def cached_build_fold_hybrid(dataset_name, train_reps, val_reps, y_train, bp,
                             *, augmentations_train=None,
                             augmentations_val=None, groups=None,
                             **build_kwargs):
    """_shared.build_fold_hybrid, stored through noise_cache (plan section 3).

    Returns what build_fold_hybrid returns, (X_train, X_val, feature_info);
    raises what it raises, and a raise is never stored. Keyed on every recipe
    field (noise_cache.blueprint_fingerprint), the content of every
    representation and augmentation in the order they are handed over, their
    NAMES (feature_info is keyed by them), the labels, the groups, and every
    other keyword -- a model enters as its class and parameters
    (noise_cache.model_signature), a metric function by its qualified name.
    For the callers that built a fold's hybrid with build_fold_hybrid
    directly and pass it every representation (n02, n09, n13): the same call,
    so the same matrices.
    """
    from experiments._shared import build_fold_hybrid
    from experiments.noise import noise_cache as nc

    def _kw(v):
        if v is None or isinstance(v, (bool, int, float, str)):
            return v
        if callable(v) and not hasattr(v, 'get_params'):
            return f'{getattr(v, "__module__", "")}.{getattr(v, "__qualname__", repr(v))}'
        return nc.model_signature(v)
    key = {'blueprint': nc.blueprint_fingerprint(bp, list(train_reps)),
           'rep_names': list(train_reps),
           'train_reps': [nc.array_hash(train_reps[r]) for r in train_reps],
           'val_reps': [nc.array_hash(val_reps.get(r)) for r in train_reps],
           'augmentations': (None if augmentations_train is None else
                             [[a, nc.array_hash(augmentations_train[a]),
                               nc.array_hash((augmentations_val or {}).get(a))]
                              for a in augmentations_train]),
           'y': nc.array_hash(y_train), 'groups': nc.array_hash(groups),
           'kwargs': {k: _kw(v) for k, v in sorted(build_kwargs.items())}}
    return nc.cached(dataset_name, 'build_fold_hybrid', key, lambda: tuple(
        build_fold_hybrid(train_reps, val_reps, y_train, bp,
                          augmentations_train=augmentations_train,
                          augmentations_val=augmentations_val, groups=groups,
                          **build_kwargs)))


def detector_scores_with(ds, detector, X_tr, y_tr, lists, k, *, is_clf=False,
                         groups=None):
    """One neighbour detector's scores with its neighbours taken from `lists`
    (noise_cache.cached_detector_scores). None when the fit failed."""
    from experiments.noise.noise_cache import cached_detector_scores
    return cached_detector_scores(ds, X_tr, y_tr, is_clf=is_clf,
                                  detector=detector, groups=groups,
                                  similarity_neighbours=lists, similarity_k=k)


def similarity_record(settings, sink=None, **extra):
    """What every result file records about the similarity axis: which
    settings and ks were swept, where the rows went, and that every key written
    before 2026-09-25 is the MODEL_INPUT setting at today's k."""
    from experiments.noise import similarity_space as ss
    from experiments.noise.controls import controls_enabled
    rec = {'settings_swept': [s.name for s in settings],
           'ks_swept': list(sweep_ks()),
           'env_KIRBY_SIMILARITY_SETTINGS': os.environ.get(
               'KIRBY_SIMILARITY_SETTINGS'),
           'legacy_keys_similarity_setting': ss.MODEL_INPUT.name,
           'controls_enabled': bool(controls_enabled())}
    if sink is not None:
        rec.update(sink.summary())
    rec.update(extra)
    return rec


# What this file writes, since chat 5 (2026-10-05, HANDOFF_EXPERIMENT_ALIGNMENT
# section 2.4). Stamped on every condition file, written into the condition
# files' fingerprint (run() below) and into the per-unit checkpoint's, so a
# file or a checkpoint line from the whole-dataset design before that date is
# recomputed, never read back. The 'b' (also 2026-10-05) marks the change
# that stopped grading activity_cliff_aware in each configuration's own
# features, so units checkpointed before it are not read back.
# 2026-10-06: the main table's label_smooth and knn_smooth read the top
# screened neighbour setting, as_they_come is dropped, knn_smooth is not run on
# classification and censoring runs one draw, so the layout moved again.
RESULT_LAYOUT = 'fold_aligned_2026-10-06'

# The cleanups whose main-table score (by_level.<level>.scores) reads its
# neighbours from the pass's top screened setting (2026-10-06). prox_smooth is
# not in n07's run list.
MAIN_TABLE_NEIGHBOUR_CLEANUPS = ('label_smooth', 'knn_smooth')

# The cleanups that hand the model a per-molecule weight instead of new labels
# (cleaned_training_set: downweight, huber_weight, invvar_weight). On a model
# whose fit takes no weight (gp) they are recorded under 'weights_not_run'
# with noise_cache.WEIGHTS_NOT_RUN (2026-10-07), not left as a bare None.
WEIGHTING_CLEANUPS = ('downweight', 'huber_weight', 'invvar_weight')

# The two of a model's four fitted configurations whose features are chosen
# from the labels (noise_helpers.CONFIGURATION_SPACES holds all four). Only
# these have a clean-label-features twin (the oracle arm): a single
# representation chooses no features, so its clean-label and damaged-label
# features are the same matrix.
HYBRID_CONFIGURATIONS = ('hybrid', 'hybrid__general2')


def _condition_levels(data, dataset_name, condition):
    """The levels n07 runs one condition at: the decided levels for this
    dataset's group (noise_helpers.decided_levels, KNN_SEARCH_SPACE_RECORD
    "The noise grid", Adelaide 2026-10-05), level 0 first. AS_THEY_COME is
    not run (2026-10-06)."""
    from experiments.noise.noise_helpers import decided_levels
    return tuple(float(x) for x in decided_levels(data, dataset_name,
                                                  condition))


def run(data, config, output_dir, dataset_name, **kwargs):
    """N07 across every injected condition this dataset can be given.

    ⛔ WHY THIS EXISTS. n07 is the experiment that compares the cleaning
    methods against each other, and until 2026-09-23 it injected ONE condition
    -- outlier_p05, pinned in _noise_spec -- at ONE level. Everything this
    suite knows about how the cleaning methods behave under different noise had
    to come through here, and could not be asked.

    Its headline is a downstream score, which is defined under every condition,
    so it sweeps the whole list. The sweep, the per-condition checkpointing,
    the resume and the stride slice are noise_helpers.sweep_conditions.

    2026-10-05 (chat 5): each condition now runs at every decided level, and
    the condition files are fingerprinted with the levels, the draws, the
    outer fold count (KIRBY_NOISE_N_FOLDS included), the folds run
    (KIRBY_NOISE_ONLY_FOLDS) and RESULT_LAYOUT. A file written by the
    whole-dataset design carries none of that, so sweep_conditions recomputes
    it instead of skipping it.
    """
    from experiments.noise.noise_helpers import noise_n_folds, draws_for
    # DECIDED 2026-10-06 (Adelaide delegated): as_they_come is dropped; it
    # repeated level 0 of every injected condition.
    conditions = list(conditions_for_task(data))
    fp_config = dict(config)
    fp_config['n_splits'] = noise_n_folds(config)
    fp_config['noise_levels'] = {
        c: list(_condition_levels(data, dataset_name, c)) for c in conditions}
    fp_config['noise_seeds'] = list(NOISE_SEEDS)
    fp_config['noise_draws'] = {c: list(draws_for(c, NOISE_SEEDS))
                                for c in conditions}
    fp_config['noise_strategies'] = [
        RESULT_LAYOUT,
        f"only_folds={os.environ.get('KIRBY_NOISE_ONLY_FOLDS') or 'all'}"]
    return sweep_conditions(
        _run_one_condition, 'n07_remediation',
        data, fp_config, output_dir, dataset_name,
        default_conditions=conditions,
        kwargs=kwargs, groups_needed=GROUPED_STRATEGIES)


def _main_neighbour_score(st, X_tr, X_va, y_noisy, y_val, noise_scores, top,
                          *, noise_fraction, model_factory, model_name,
                          metric_fn, metric_name, is_clf, groups,
                          measurement_counts, dataset_name):
    """The main-table score of label_smooth or knn_smooth with its neighbours
    taken from the pass's top screened setting (2026-10-06). The same fit as
    _evaluate_fold's (noise_cache.cached_fit_predict, the same model and
    held-out labels); only the neighbour lists differ. Returns (score or
    None, reason or None)."""
    from experiments.noise.noise_cache import (cached_fit_predict,
                                               predict_mode_for)
    if not top or top.get('lists') is None:
        return None, (top or {}).get('reason') or 'no screened setting'
    if noise_scores is None:
        return None, 'the driving detector returned no scores'
    cl, sh = split_strategy(st)
    try:
        cleaned = cleaned_training_set(
            cl, X_tr, y_noisy, noise_scores,
            noise_fraction if sh is None else sh, is_clf=is_clf,
            groups=groups, measurement_counts=measurement_counts,
            dataset_name=dataset_name, similarity_neighbours=top['lists'],
            similarity_k=top['k'])
        if cleaned is None:
            return None, 'the cleanup declined'
        pred = cached_fit_predict(dataset_name, model_name, model_factory,
                                  cleaned['X'], cleaned['y'], X_va,
                                  sample_weight=cleaned['sample_weight'],
                                  predict_mode=predict_mode_for(metric_name))
        return float(metric_fn(y_val, pred)), None
    except Exception as exc:
        print(f"      {st} (top setting {top.get('setting')}) failed: "
              f"{type(exc).__name__}: {exc}")
        return None, f'{type(exc).__name__}: {exc}'


def _draw_means_by_fold(by_fold):
    """{fold: [score per draw]} -> fold_spread over the per-fold draw means."""
    from experiments.noise.noise_helpers import fold_spread
    per_fold = [float(np.mean(v)) for _f, v in sorted(by_fold.items()) if v]
    out = fold_spread(per_fold)
    out['n_scores'] = int(sum(len(v) for v in by_fold.values()))
    return out


def _recovery(clean, noisy, score, lower_is_better):
    """(score - noisy) / (clean - noisy) x 100, or (nan, why). The same rule
    as before 2026-10-05: the gap must say the noise hurt in the metric's own
    direction and be over GAP_MIN_FRAC of the clean score."""
    if any(x is None or not np.isfinite(x) for x in (clean, noisy)):
        return float('nan'), 'no clean or no damaged score'
    gap = clean - noisy
    hurt = (gap < 0) if lower_is_better else (gap > 0)
    if not hurt:
        return float('nan'), (f'the damage did not make this worse: '
                               f'clean={clean:.6g}, damaged={noisy:.6g}, '
                               f'lower_is_better={lower_is_better}')
    floor = max(1e-10, GAP_MIN_FRAC * abs(clean))
    if abs(gap) <= floor:
        return float('nan'), (f'the damage moved the score by {abs(gap):.6g}, '
                               f'under {GAP_MIN_FRAC:.0%} of the clean score '
                               f'({clean:.6g})')
    if score is None or not np.isfinite(score):
        return float('nan'), 'this cleanup has no score here'
    return float((score - noisy) / gap * 100), None


def _run_one_condition(data, config, output_dir, dataset_name, **kwargs):
    """N07: noise remediation strategies, ONE condition at every decided level.

    CHAT 5 ALIGNMENT, 2026-10-05 (HANDOFF_EXPERIMENT_ALIGNMENT section 2.4).
    What each fold now does, in order:
      1. Folds are noise_helpers.noise_outer_folds: the per-fold searches'
         own splits, the count from config['n_splits'] or KIRBY_NOISE_N_FOLDS.
      2. noise_helpers.fold_configurations runs this fold's searches on its
         training rows (clean labels, decided grid, pruning off) and gives each
         of the dataset's two decided models its 2 best base representations,
         its 2 best general hybrids, and the recipes of the neighbour spaces
         (the distance hybrid and the similarity hybrid). Nothing is read from
         a whole-dataset search.
      3. Each model is fitted on its 4 configurations (2 models x 4 = 8, R5).
         The hybrids are rebuilt on the fold's training rows from the damaged
         labels (noise_helpers.fold_model_inputs, through the noise cache) and,
         for the oracle arm, from the clean labels.
      4. Every cleanup runs in every configuration at every level and draw.
         The neighbour cleanups' swept rows go through _n07_sweep_pass (the
         screen keeps the top 3 neighbour settings; scaffold-mates stay in the
         lists). In the main table label_smooth and knn_smooth use the
         screen's top setting (2026-10-06).
    Each (fold, level, draw) is written to a checkpoint file as it finishes.
    """
    from experiments.dataset_policy import noise_groups_for as _ngf
    from experiments._shared import input_fingerprint
    from experiments.decided_grid import decided_models
    from experiments.noise.noise_helpers import (
        noise_outer_folds, fold_configurations, fold_model_inputs,
        configuration_record, inject_condition, dataset_knn_group, level_unit,
        CONFIGURATION_SPACES, draws_for)
    from kirby.noise_robustness import robustness_metrics
    from experiments.noise.noise_cache import NEIGHBOUR_DETECTORS
    _NOISE_GROUPS[dataset_name] = _ngf(data)[0]

    print("\n" + "=" * 70)
    print("EXPERIMENT N07: NOISE REMEDIATION STRATEGIES")
    print("=" * 70)

    task_type = data['task_type']
    is_clf = task_type == 'classification'
    labels = np.asarray(data['labels'])
    groups = data.get('groups')
    groups_arr = None if groups is None else np.asarray(groups)
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    metric = (metric_name, metric_fn, lower_is_better)
    # The injector no longer reads this (inject_condition takes each kind's
    # own parameters); a cleanup whose name carries no flag share is handed it
    # only because the argument is positional. Kept for the record.
    noise_fraction = kwargs.get('noise_fraction', 0.1)
    condition = kwargs.get('noise_condition') or DEFAULT_CONDITION
    if condition == AS_THEY_COME:
        raise ValueError(
            f"n07: {AS_THEY_COME!r} is not run (2026-10-06); it repeats level "
            f"0 of every injected condition.")
    levels = _condition_levels(data, dataset_name, condition)
    # DECIDED 2026-10-06 (Adelaide delegated): censoring reads no seed, so its
    # draws are identical and it runs one; every other kind runs NOISE_SEEDS.
    nonzero_draws = list(draws_for(condition, NOISE_SEEDS))
    knn_grp = dataset_knn_group(data, dataset_name)
    smiles = data.get('smiles')
    has_smiles = smiles is not None and len(smiles) > 0
    n_molecules = len(set(smiles)) if has_smiles else len(labels)

    # R1: the dataset's two decided models (DECISIONS.md Step 1), nothing
    # else. knn runs only inside the distance search. Replaces
    # DOWNSTREAM_MODELS_NOISE / config['models'] (five models) and the rf and
    # svm every D-versus-D' row used to add.
    models = [m for m in decided_models(dataset_name, 'general') if m != 'knn']
    model_factories = get_model_factories(models, task_type=task_type,
                                          n_samples=len(labels))
    models_without_factory = [m for m in models if m not in model_factories]
    print(f"  Models (decided): {models}; condition {condition}; levels "
          f"{list(levels)} ({level_unit(condition, is_clf)}); group {knn_grp}")

    folds, n_folds = noise_outer_folds(data, config)

    strategies = [
        'no_noise',         # clean baseline (no injection)
        'noisy_baseline',   # inject noise, do nothing
        'remove',           # detect + remove flagged samples
        'downweight',       # detect + weight by 1/(1+score)
        'label_smooth',     # detect + smooth labels toward neighbor means
        'knn_smooth',       # replace flagged labels with kNN-mean in hybrid space
    ]
    # DECIDED 2026-10-06 (Adelaide delegated): on a classification dataset
    # only label_smooth runs. There knn_smooth flags the same share and
    # replaces each flagged label with the same neighbours' majority class
    # (cleaned_training_set), so it wrote label_smooth's labels a second time.
    if is_clf:
        strategies.remove('knn_smooth')
    # The cleanups that write a numeric label (oof_smooth, gp_smooth,
    # js_shrink, constant_lift) are not run on a classification dataset.
    # quantile_clip is in REGRESSION_ONLY_CLEANUPS but out of CLEANUPS since
    # f36 (2026-10-05), so it is not run either (it was, through this list,
    # until 2026-10-05).
    # DECIDED 2026-10-06 (Adelaide delegated): huber_weight, prox_smooth and
    # invvar_weight are in CLEANUPS but are not added to this list, so n07
    # does not run them.
    if not is_clf:
        strategies.extend(c for c in REGRESSION_ONLY_CLEANUPS if c in CLEANUPS)
    flag_shares = ([float(x) for x in kwargs['flag_shares']]
                   if kwargs.get('flag_shares') else None)
    strategies = expand_flag_shares(strategies, is_clf, flag_shares)
    expensive = [st for st in strategies
                 if split_strategy(st)[0] in REGRESSION_ONLY_CLEANUPS]
    cap_features, cap_source = max_cleanup_features(kwargs)

    # Measurement counts, for invvar_weight (and only where the loader kept
    # one row per measurement, herg_ic50 today). n07 never passed them before
    # 2026-10-05, so invvar_weight declined on every dataset.
    try:
        mcounts = measurement_counts_for_rows(data)
        mcounts_note = (None if mcounts is not None else
                        'the loader kept no per-molecule measurement count')
    except Exception as exc:
        mcounts, mcounts_note = None, f'{type(exc).__name__}: {exc}'

    # The neighbour settings this dataset's group sweeps (KNN_SEARCH_SPACE_
    # RECORD): group from the molecule count, no model_input, no importance
    # weighting, distances per group; ks and counting schemes are swept and
    # screened inside _n07_sweep_pass.
    sim_settings = sweep_settings(dataset_name, n_molecules, has_smiles)
    # A new rows file: rows written by the whole-dataset design share model
    # input names ('svm:hybrid') with these and would be skipped as done.
    sink = RowSink(output_dir, 'n07_remediation', dataset_name,
                   f'{condition}__{RESULT_LAYOUT}')
    sweep_ctx = sweep_context(
        experiment='n07_remediation', dataset_name=dataset_name, sink=sink,
        settings=sim_settings, metric=metric, task_type=task_type,
        condition=condition, level=None, data=data, factories=model_factories,
        noise_fraction=noise_fraction, measurement_counts=mcounts)
    print(f"  Similarity axis: {len(sim_settings)} settings x ks "
          f"{list(sweep_ks())} x schemes {list(sweep_schemes())}, top "
          f"{screen_top()} refitted; rows -> {sink.path}")

    # DECIDED 2026-10-06 (Adelaide delegated): the per-fold searches
    # (fold_configurations below) stay on the clean training labels.
    ckpt = FoldCheckpoint(
        output_dir, 'n07_remediation', dataset_name, condition=condition,
        tag='units', fingerprint={
            'layout': RESULT_LAYOUT, 'input': input_fingerprint(data, config),
            'condition': condition, 'levels': list(levels),
            'n_folds': int(n_folds), 'models': models,
            'strategies': strategies, 'settings': [s.name for s in sim_settings],
            'screen_top': screen_top(), 'max_cleanup_features': cap_features,
            'draws': nonzero_draws})

    fold_records = {}
    units = []           # [{'fold', 'level', 'draw', 'value'}]

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
        mc_tr = None if mcounts is None else np.asarray(mcounts)[train_idx]

        # Each model's 4 configurations with features chosen from the CLEAN
        # training labels. The base pair is the same matrix at every level;
        # the hybrid pair is the oracle arm.
        clean_inputs = {}
        frec = {'n_train': int(len(train_idx)), 'n_val': int(len(val_idx)),
                'models': {}}
        for m in models:
            entry = entries.get(m)
            if m not in model_factories:
                frec['models'][m] = {'available': False,
                                     'why': 'no model factory for this name'}
                continue
            ins, unav, names = fold_model_inputs(
                data, dataset_name, m, entry, train_idx=train_idx,
                val_idx=val_idx, y_train=y_tr_clean,
                factory=model_factories[m], metric=metric)
            clean_inputs[m] = (entry, ins, unav)
            rec = configuration_record(entry, names)
            rec['configurations_unavailable'] = {
                sp: why for sp, why in unav.items() if sp in CONFIGURATION_SPACES}
            rec['n_features'] = {sp: int(np.shape(X[0])[1])
                                 for sp, X in ins.items()}
            rec['metric'] = metric_name
            rec['lower_is_better'] = bool(lower_is_better)
            frec['models'][m] = rec
            print(f"    {m}: base {rec.get('base')}, hybrids "
                  f"{[h.get('label') for h in rec.get('hybrids') or []]}")
        # The clean-label-feature ceiling of each hybrid configuration: clean
        # labels fitted in clean-label features, one fit per fold. Reported
        # beside the per-draw ceiling; nothing divides by it.
        frec['no_noise_clean_features'] = {}
        for m, (entry, ins, _u) in clean_inputs.items():
            for sp in HYBRID_CONFIGURATIONS:
                if sp not in ins:
                    continue
                s = _evaluate_fold(
                    ins[sp][0], ins[sp][1], y_tr_clean, y_val, 'no_noise',
                    noise_fraction, model_factories[m], metric_fn,
                    is_clf=is_clf, metric_name=metric_name,
                    cache_dataset=dataset_name, cache_model_name=m)
                frec['no_noise_clean_features'].setdefault(m, {})[sp] = (
                    None if s is None else float(s))
        fold_records[str(fold_i)] = frec

        def _unit(level, draw, _tr=train_idx, _va=val_idx, _ytr=y_tr_clean,
                  _yva=y_val, _g=g_tr, _mc=mc_tr, _ci=clean_inputs,
                  _fold=fold_i):
            y_noisy, cmask, prov = inject_condition(
                data, dataset_name, condition, level, draw, train_idx=_tr,
                groups=_NOISE_GROUPS.get(dataset_name))
            cmask = np.asarray(cmask, dtype=bool)
            val = {'provenance': prov, 'n_corrupted': int(cmask.sum()),
                   'scores': {}, 'oracle': {}, 'detection': {},
                   'detection_by_detector': {}, 'unavailable': {},
                   'expensive_skipped': {}, 'main_neighbour_setting': {},
                   'main_neighbour_not_run': {}, 'weights_not_run': {},
                   'tally': {'reached': 0, 'detector_returned_none': 0,
                             'scores_declined': 0, 'graded': 0,
                             'fold_space_unavailable': 0}}
            sweep_ctx['level'] = float(level)
            sweep_ctx['screen'] = []
            unit_sim_det = {}
            for m, (entry, ins_c, unav_c) in _ci.items():
                factory = model_factories[m]
                hyb_n, unav_n, _names = fold_model_inputs(
                    data, dataset_name, m, entry, train_idx=_tr, val_idx=_va,
                    y_train=y_noisy, factory=factory, metric=metric,
                    spaces=HYBRID_CONFIGURATIONS)
                for sp in CONFIGURATION_SPACES:
                    is_hyb = sp in HYBRID_CONFIGURATIONS
                    pair = hyb_n.get(sp) if is_hyb else ins_c.get(sp)
                    if pair is None:
                        val['unavailable'].setdefault(m, {})[sp] = (
                            (unav_n if is_hyb else unav_c).get(sp)
                            or 'not built')
                        val['tally']['fold_space_unavailable'] += 1
                        continue
                    X_tr_n, X_va_n = pair
                    oracle = ins_c.get(sp) if is_hyb else None
                    here = list(strategies)
                    width = int(np.shape(X_tr_n)[1])
                    if cap_features is not None and width > cap_features:
                        here = [st for st in here if st not in expensive]
                        val['expensive_skipped'].setdefault(m, {})[sp] = width
                    # The driving detector. When it reads neighbours, its
                    # main-table scores come from the pass's top screened
                    # setting below (2026-10-06), not from the model's own
                    # features. cross_val, the one used, reads none.
                    det_nbr = DETECTOR in NEIGHBOUR_DETECTORS
                    ns_n = None if det_nbr else _compute_noise_scores(
                        X_tr_n, y_noisy, is_clf=is_clf, groups=_g,
                        cache_dataset=dataset_name)
                    ns_c = (None if (oracle is None or det_nbr)
                            else _compute_noise_scores(
                                oracle[0], y_noisy, is_clf=is_clf, groups=_g,
                                cache_dataset=dataset_name))

                    # D versus D', the neighbour settings and their screen.
                    # Run before the main table, because the main table's
                    # label_smooth and knn_smooth read each pass's top
                    # screened setting.
                    passes = [('scores', f'{m}:{sp}', X_tr_n, X_va_n, ns_n,
                               DETECTORS_GRADED)]
                    if oracle is not None:
                        passes.append(('oracle',
                                       f'{m}:{sp}@clean_label_features',
                                       oracle[0], oracle[1], ns_c, ()))
                    tops = {}
                    for arm, mi, Xtr, Xva, ns, grade in passes:
                        tops[arm] = _n07_sweep_pass(
                            sweep_ctx, model_name=m, model_input=mi,
                            X_tr=Xtr, X_va=Xva, y_tr_noisy=y_noisy,
                            y_tr_clean=_ytr, y_val=_yva, noise_scores=ns,
                            strategies=here, train_idx=_tr, val_idx=_va,
                            groups_tr=_g, fold=_fold, draw=draw,
                            recipes=(entry or {}).get('recipes') or {},
                            base_reps=(entry or {}).get('base') or [],
                            corrupted_mask=cmask, noise_strategy=condition,
                            detection_acc=unit_sim_det,
                            grade_detectors=grade) or {
                                'reason': 'the pass returned nothing'}
                        t = tops[arm]
                        (val['main_neighbour_setting'].setdefault(m, {})
                         .setdefault(sp, {}))[arm] = (
                            {'setting': t['setting'], 'space': t['space'],
                             'k': t['k'], 'scheme': t['scheme']}
                            if t.get('lists') is not None
                            else {'reason': t.get('reason')})
                    if det_nbr:
                        ns_n = (tops.get('scores') or {}).get('detector_scores')
                        ns_c = (None if oracle is None else
                                (tops.get('oracle') or {}).get(
                                    'detector_scores'))

                    # Detection before any cleanup: every graded detector that
                    # reads no neighbours, in this configuration's damaged-label
                    # features, and the driving detector in both feature spaces.
                    # A detector that reads neighbours (activity_cliff_aware) is
                    # not graded here (2026-10-05): here it would take k=10 and
                    # the estimator's default distance in the model's own
                    # features, a neighbour space the kNN record cut (axis 1),
                    # at one setting never swept. _n07_sweep_pass grades it
                    # under every decided (setting, k, scheme), written to
                    # detection_by_similarity.
                    val['tally']['reached'] += 1
                    for dname in DETECTORS_GRADED:
                        if dname in NEIGHBOUR_DETECTORS:
                            continue
                        alt = ns_n if dname == DETECTOR else _compute_noise_scores(
                            X_tr_n, y_noisy, is_clf=is_clf, detector=dname,
                            groups=_g, cache_dataset=dataset_name)
                        if alt is None:
                            val['tally']['detector_returned_none'] += 1
                            continue
                        d = detection_scores(cmask, alt, condition)
                        if d is None:
                            val['tally']['scores_declined'] += 1
                            continue
                        val['tally']['graded'] += 1
                        (val['detection_by_detector'].setdefault(m, {})
                         .setdefault(sp, {}))[dname] = d
                    for kind, ns in (('noisy_space', ns_n), ('clean_space', ns_c)):
                        if ns is None:
                            continue
                        d = detection_scores(cmask, ns, condition)
                        if d is not None:
                            (val['detection'].setdefault(m, {})
                             .setdefault(sp, {}))[kind] = d

                    # DECIDED 2026-10-06 (Adelaide delegated): in these scores
                    # label_smooth and knn_smooth find their neighbours in the
                    # pass's top screened setting (kept[0]: space and
                    # distance, k, and how much each neighbour counts), not 9
                    # Euclidean neighbours in the configuration's own
                    # features. The setting used is in
                    # val['main_neighbour_setting']. The other cleanups and the
                    # two baselines are fitted as before.
                    arms = [('scores', val['scores'], X_tr_n, X_va_n, ns_n)]
                    if oracle is not None:
                        arms.append(('oracle', val['oracle'], oracle[0],
                                     oracle[1], ns_c))
                    for arm, acc, Xtr, Xva, ns in arms:
                        out = acc.setdefault(m, {}).setdefault(sp, {})
                        for st in here:
                            base_st = split_strategy(st)[0]
                            needs = base_st in SCORE_DRIVEN_CLEANUPS
                            if needs and ns is None:
                                # No driving-detector scores for this arm;
                                # _evaluate_fold would otherwise refit the
                                # detector in the configuration's own
                                # features.
                                out[st] = None
                                continue
                            if base_st in MAIN_TABLE_NEIGHBOUR_CLEANUPS:
                                s, why = _main_neighbour_score(
                                    st, Xtr, Xva, y_noisy, _yva, ns,
                                    tops.get(arm), noise_fraction=noise_fraction,
                                    model_factory=factory, model_name=m,
                                    metric_fn=metric_fn,
                                    metric_name=metric_name, is_clf=is_clf,
                                    groups=_g, measurement_counts=_mc,
                                    dataset_name=dataset_name)
                                if why:
                                    (val['main_neighbour_not_run']
                                     .setdefault(m, {}).setdefault(sp, {})
                                     .setdefault(arm, {}))[st] = why
                                out[st] = s
                                continue
                            if (base_st in WEIGHTING_CLEANUPS
                                    and not fit_takes_sample_weight(factory)):
                                (val['weights_not_run']
                                 .setdefault(m, {}).setdefault(sp, {})
                                 .setdefault(arm, {}))[st] = WEIGHTS_NOT_RUN
                                out[st] = None
                                continue
                            s = _evaluate_fold(
                                Xtr, Xva, _ytr, _yva, st, noise_fraction,
                                factory, metric_fn, is_clf=is_clf,
                                metric_name=metric_name,
                                precomputed_y_tr=y_noisy,
                                noise_scores=(ns if needs else None),
                                dataset_name=dataset_name, condition=condition,
                                train_groups=_g, measurement_counts=_mc,
                                cache_dataset=dataset_name, cache_model_name=m)
                            out[st] = None if s is None else float(s)
            val['screen'] = sweep_ctx.pop('screen', [])
            val['sim_detection'] = unit_sim_det
            return val

        for level in levels:
            draws = NOISE_SEEDS[:1] if float(level) == 0.0 else nonzero_draws
            for draw in draws:
                key = ['unit', int(fold_i), float(level), int(draw)]
                v = ckpt.get(key)
                if v is None:
                    print(f"    level {level:g}, draw {draw}")
                    v = _unit(float(level), int(draw))
                    ckpt.put(key, v)
                else:
                    print(f"    level {level:g}, draw {draw}: read back from "
                          f"{ckpt.path}")
                units.append({'fold': int(fold_i), 'level': float(level),
                              'draw': int(draw), 'value': v})

    # ── Summaries. Every spread is over folds (fold_spread); a fold's number
    # is the mean over its draws. No significance test anywhere. ────────────
    def _collect(field):
        out = {}
        for u in units:
            for m, by_sp in (u['value'].get(field) or {}).items():
                for sp, by_st in by_sp.items():
                    for st, s in by_st.items():
                        if s is None or not np.isfinite(s):
                            continue
                        (out.setdefault(u['level'], {}).setdefault(m, {})
                         .setdefault(sp, {}).setdefault(st, {})
                         .setdefault(u['fold'], []).append(float(s)))
        return out

    main, orac = _collect('scores'), _collect('oracle')

    def _table(src):
        return {f'{lv:g}': {m: {sp: {st: _draw_means_by_fold(byf)
                                     for st, byf in by_st.items()}
                                for sp, by_st in by_sp.items()}
                            for m, by_sp in by_m.items()}
                for lv, by_m in src.items()}

    scores_tbl, oracle_tbl = _table(main), _table(orac)

    by_level = {}
    for lv in levels:
        key = f'{lv:g}'
        rec_pct, rec_why, minus_clean = {}, {}, {}
        for m, by_sp in scores_tbl.get(key, {}).items():
            for sp, by_st in by_sp.items():
                clean = (by_st.get('no_noise') or {}).get('mean')
                noisy = (by_st.get('noisy_baseline') or {}).get('mean')
                for st, row in by_st.items():
                    s = row.get('mean')
                    (minus_clean.setdefault(m, {}).setdefault(sp, {}))[st] = (
                        None if (s is None or clean is None)
                        else float(s - clean))
                    if st in ('no_noise', 'noisy_baseline'):
                        continue
                    pct, why = _recovery(clean, noisy, s, lower_is_better)
                    rec_pct.setdefault(m, {}).setdefault(sp, {})[st] = pct
                    if why:
                        rec_why.setdefault(m, {}).setdefault(sp, {})[st] = why
        lv_units = [u for u in units if u['level'] == lv]
        det_rows, det_by_cfg, det_by_det = {}, {}, {}
        for u in lv_units:
            for m, by_sp in (u['value'].get('detection') or {}).items():
                for sp, by_kind in by_sp.items():
                    for kind, d in by_kind.items():
                        det_rows.setdefault(kind, []).append(d)
                        (det_by_cfg.setdefault(m, {}).setdefault(sp, {})
                         .setdefault(kind, [])).append(d)
            for m, by_sp in (u['value'].get('detection_by_detector') or {}).items():
                for sp, by_d in by_sp.items():
                    for dn, d in by_d.items():
                        det_by_det.setdefault(dn, []).append(d)
        sim_det = {}
        for u in lv_units:
            for dn, by_s in (u['value'].get('sim_detection') or {}).items():
                for sn, by_k in by_s.items():
                    for k, rows in by_k.items():
                        (sim_det.setdefault(dn, {}).setdefault(sn, {})
                         .setdefault(k, [])).extend(rows)
        tally = {}
        for u in lv_units:
            for k, v in (u['value'].get('tally') or {}).items():
                tally[k] = tally.get(k, 0) + int(v)
        by_level[key] = {
            'level': float(lv),
            'n_draws': len(NOISE_SEEDS[:1] if lv == 0.0 else nonzero_draws),
            'scores': scores_tbl.get(key, {}),
            'scores_clean_label_features': oracle_tbl.get(key, {}),
            'recovery_pct': rec_pct,
            'recovery_undefined': rec_why,
            'minus_no_noise': minus_clean,
            'detection': {k: _mean_detection(r) for k, r in det_rows.items()},
            'detection_by_configuration': {
                m: {sp: {k: _mean_detection(r) for k, r in by_k.items()}
                    for sp, by_k in by_sp.items()}
                for m, by_sp in det_by_cfg.items()},
            'detection_by_detector': {dn: _mean_detection(r)
                                      for dn, r in det_by_det.items()},
            'detection_by_similarity': {
                dn: {sn: {k: _mean_detection(r) for k, r in by_k.items()}
                     for sn, by_k in by_s.items()}
                for dn, by_s in sim_det.items()},
            'detection_tally': tally,
            'expensive_cleanups_skipped_width': {
                f"fold{u['fold']}_draw{u['draw']}": u['value'].get(
                    'expensive_skipped') for u in lv_units
                if u['value'].get('expensive_skipped')},
            'configurations_unavailable': {
                f"fold{u['fold']}_draw{u['draw']}": u['value'].get(
                    'unavailable') for u in lv_units
                if u['value'].get('unavailable')},
        }

    # Robustness over the levels, per (model, configuration, cleanup). The
    # plain difference against level 0 in the dataset's own metric, always;
    # auc_norm on full_knn datasets only (fast_knn has 2 levels and auc_norm
    # needs 4, absent there by Adelaide's decision of 2026-10-05).
    robustness = {}
    lvl_keys = [f'{lv:g}' for lv in levels]
    if len(levels) > 1 and 0.0 in levels:
        for m in models:
            for sp in CONFIGURATION_SPACES:
                for st in strategies:
                    curve = [((scores_tbl.get(k, {}).get(m, {}).get(sp, {})
                               .get(st) or {}).get('mean')) for k in lvl_keys]
                    if curve[0] is None:
                        continue
                    row = {'levels': list(levels), 'scores': curve,
                           'damaged_minus_level0': [
                               None if c is None else float(c - curve[0])
                               for c in curve]}
                    if knn_grp == 'full_knn':
                        ok = [(x, c) for x, c in zip(levels, curve)
                              if c is not None]
                        rm = robustness_metrics([x for x, _ in ok],
                                                [c for _, c in ok],
                                                baseline_threshold=None,
                                                lower_is_better=lower_is_better)
                        row['auc_norm'] = rm.get('auc_norm')
                    (robustness.setdefault(m, {}).setdefault(sp, {}))[st] = row

    provenance = {}
    for u in units:
        provenance.setdefault(f"{u['level']:g}", {}).setdefault(
            str(u['draw']), u['value'].get('provenance'))

    results = {
        'experiment': 'n07_remediation',
        'result_layout': RESULT_LAYOUT,
        'metric': metric_name,
        'lower_is_better': bool(lower_is_better),
        'task_type': task_type,
        'noise_condition': condition,
        'levels': list(levels),
        'level_unit': level_unit(condition, is_clf),
        'draws_per_nonzero_level': len(nonzero_draws),
        'level_zero': 'the labels as delivered, run once (one draw)',
        'knn_group': knn_grp,
        'robustness_reported_as': (
            'auc_norm and the plain difference against level 0'
            if knn_grp == 'full_knn' else
            'the plain difference against level 0 only; auc_norm is absent '
            'on fast_knn datasets by decision (2 levels)'),
        'dose_units': noise_dose_units(dataset_name),
        'noise_provenance': provenance,
        'n_folds': int(n_folds),
        'folds_run': [int(f) for f, _t, _v in folds],
        'selection_scope': 'fold_train_rows',
        'models': models,
        'models_without_factory': models_without_factory,
        'configurations_per_model': list(CONFIGURATION_SPACES),
        'n_fitted_configurations': len(models) * len(CONFIGURATION_SPACES),
        'folds': fold_records,
        'strategies': strategies,
        'expensive_cleanups': expensive,
        'max_cleanup_features': cap_features,
        'max_cleanup_features_source': cap_source,
        'noise_fraction': noise_fraction,
        'flag_shares': {c: list(flag_shares if flag_shares is not None
                                else flag_shares_for(c, is_clf))
                        for c in CLEANUPS if reads_flag_share(c, is_clf)},
        'measurement_counts_available': mcounts is not None,
        'measurement_counts_note': mcounts_note,
        'detector_used': DETECTOR,
        'detectors_graded': list(DETECTORS_GRADED),
        # Where each graded detector is graded (2026-10-05): one that reads
        # neighbours only per decided neighbour setting
        # (by_level.*.detection_by_similarity), the others in each
        # configuration's own features (by_level.*.detection_by_detector).
        'detectors_graded_per_neighbour_setting': [
            d for d in DETECTORS_GRADED if d in NEIGHBOUR_DETECTORS],
        'detectors_graded_in_configuration_features': [
            d for d in DETECTORS_GRADED if d not in NEIGHBOUR_DETECTORS],
        'noise_regime': 'synthetic',
        'claim_scope': (
            'Remediation ranking holds only under synthetically injected '
            'training-label noise; not validated on real label noise of '
            'unknown structure.'),
        'by_level': by_level,
        'robustness': robustness,
        # 2026-10-06: the neighbour setting the main table's label_smooth and
        # knn_smooth used, per (fold, level, draw, model, configuration, arm);
        # arm 'scores' is the damaged-label features, 'oracle' the
        # clean-label-features twin. A 'reason' entry means the screen kept
        # nothing and those two cleanups have no main-table score there.
        'main_table_neighbour_cleanups': list(
            c for c in MAIN_TABLE_NEIGHBOUR_CLEANUPS
            if any(split_strategy(st)[0] == c for st in strategies)),
        'main_table_neighbour_setting': [
            {'fold': u['fold'], 'level': u['level'], 'draw': u['draw'],
             'model': m, 'configuration': sp, 'arm': arm, **rec}
            for u in units
            for m, by_sp in (u['value'].get('main_neighbour_setting')
                             or {}).items()
            for sp, by_arm in by_sp.items()
            for arm, rec in by_arm.items()],
        'main_table_neighbour_not_run': [
            {'fold': u['fold'], 'level': u['level'], 'draw': u['draw'],
             'model': m, 'configuration': sp, 'arm': arm, 'cleanup': st,
             'reason': why}
            for u in units
            for m, by_sp in (u['value'].get('main_neighbour_not_run')
                             or {}).items()
            for sp, by_arm in by_sp.items()
            for arm, by_st in by_arm.items()
            for st, why in by_st.items()],
        'feature_space_note': (
            "scores: each configuration in the features built from that "
            "draw's DAMAGED training labels (hybrids; a single representation "
            "has no feature choice). scores_clean_label_features: the two "
            "hybrid configurations with features built from the CLEAN "
            "training labels, the oracle arm. no_noise in scores is the clean "
            "training labels fitted in the damaged-label features, and is the "
            "top of every recovery percentage. folds.<i>.no_noise_clean_"
            "features is the clean labels in clean-label features, one fit "
            "per fold; nothing divides by it."),
        'recipe_note': (
            "Each fold's configurations come from searches on that fold's "
            "training rows with clean labels (noise_helpers.fold_"
            "configurations). The recipe is therefore chosen without the "
            "damage; the features of each hybrid are re-chosen from the "
            "damaged labels at every draw."),
        'similarity': similarity_record(
            sim_settings, sink, schemes_swept=list(sweep_schemes()),
            screen_top=screen_top(), screen_scored_on=SCREEN_SCORED_ON,
            knn_group=knn_grp, spaces_unavailable=sweep_ctx['unavailable'],
            detector_driving_cleanups=DETECTOR),
        'screen': [dict(e, fold=u['fold'], level=u['level'])
                   for u in units for e in (u['value'].get('screen') or [])],
        'checkpoint': {'path': ckpt.path, 'units': len(units),
                       'units_read_back': ckpt.n_resumed},
    }
    print(f"\n  {len(units)} (fold, level, draw) units; "
          f"{ckpt.n_resumed} read back from the checkpoint")
    for key in lvl_keys:
        for m, by_sp in by_level[key]['scores'].items():
            for sp, by_st in by_sp.items():
                cl = (by_st.get('no_noise') or {}).get('mean')
                nz = (by_st.get('noisy_baseline') or {}).get('mean')
                print(f"    level {key} {m}/{sp}: no_noise "
                      f"{metric_name}={cl}, noisy_baseline={nz}")
    print(f"  Rows: {sink.n_written} written this run to {sink.path}")
    return results


if __name__ == '__main__':
    parser = create_experiment_parser('N07: Noise Remediation')
    parser.add_argument(
        '--max-cleanup-features', type=int, default=None,
        help=('The widest a set of features may be before oof_smooth, '
              'gp_smooth and quantile_clip refuse to run on it. Those three '
              'fit five models each and their cost is driven by the feature '
              f'count: 275s on 1,200 molecules by 200 features against 39 '
              f'minutes on 1,200 by 2,048. Default '
              f'{MAX_CLEANUP_FEATURES}; 0 removes the ceiling. Also settable '
              'with KIRBY_MAX_CLEANUP_FEATURES.'))
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset,
                  max_cleanup_features=args.max_cleanup_features)
    save_experiment_results(results, args.output, 'n07_remediation', args.dataset)
