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

HOW MANY MOLECULES A CLEANUP FLAGS. remove and knn_smooth (and label_smooth on a
classification set) flag each share in FLAG_SHARES = 0.05, 0.10, 0.20 of the
training molecules, filed as '<cleanup>@flag_share<s>'. Until 2026-09-28 they
flagged `noise_fraction`, which is also the share the injector gives the wide
error (or flips), so each cleanup was handed the injector's own count. Owner's
decision 2026-09-28: no method reads what only exists because the noise was
injected.

Compares 7 remediation strategies after noise detection: remove, downweight,
label_smooth, knn_smooth, oof_smooth, gp_smooth, quantile_clip. The last three
were added 2026-09-17 and run on regression datasets only. Two baselines
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

WHICH SEARCHES THE RECIPE MAY COME FROM. Decided 2026-09-12: the subset of the
precomputed pool this file draws from includes searches run under the
similarity objective — ones whose features were chosen so that the distance
geometry predicts a molecule's activity from its five nearest neighbours — not
prediction-built searches only. Three of the four remediation strategies
(downweight, label_smooth, knn_smooth) overwrite or reweight a label using its
neighbours' labels, which is the same quantity that objective optimises. The
two families are scored in different units, so nothing here ranks one against
the other: which family each model's recipe came from is written to
`hybrid_family_by_model` and to `summary.hybrid_family_per_model`, and every
eligible family's own best sits under `by_objective` there. This is NOT the
`distance=True` flag; n07 does not pass it and that is unchanged.

2026-09-19: both families are BUILT now, not only recorded. The pool names them
in preference order and returns the first that has an entry, and a predictive
recipe always exists, so between those two dates every strategy was measured in
the prediction-built hybrid alone. Every strategy now runs once per (downstream
model, hybrid space): 'hybrid' is the predictive recipe and the three
`per_model_*` keys hold it unchanged, and every space is under the matching
`*_by_space` keys. ⛔ `build_fold_hybrid` REFUSES a model and a metric alongside
a geometry recipe, because create_hybrid fits no downstream model there;
`_fold_space` passes both through `fold_hybrid_scoring_kwargs`, which returns
nothing for such a recipe. Without that its own `except` would have swallowed
the refusal and left the geometry space empty on every fold with nothing said.

WHICH FEATURES THE CEILING IS MEASURED IN. Until 2026-09-11 the clean-label
ceiling stayed in the clean-label feature space while every damaged arm had
moved to the noisy one, so part of the gap a recovery percentage divides by was
the feature space changing rather than the noise. The ceiling is now fitted once
more inside each noise draw's own feature space — clean training labels, no
injection, only the features differ — at 3 extra model fits per model and fold.
That is what `per_model_per_strategy['no_noise']` holds and what every recovery
percentage divides by. The clean-label-feature ceiling is still reported, under
`per_model_no_noise_clean_features`, so the two can be compared.

Usage:
    python n07_remediation.py --dataset logd --mode intermediary
"""
import sys
import os
import copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.preprocessing import StandardScaler
from experiments._shared import make_nearest_neighbors  # honours --distance-metric
from experiments.dataset_policy import (
    noise_dose_units, reference_noise_level,
)
from kirby.noise_spec import GROUPED_STRATEGIES
from experiments.noise.noise_helpers import (inject_train_labels,
                                            noise_spec_for,
                                            detection_scores, sweep_conditions,
    conditions_for_task)
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_model_factories,
    get_precomputed_search, get_best_search_for_downstream, build_fold_hybrid,
    # A geometry recipe fits no downstream model and build_fold_hybrid refuses
    # a model and a metric alongside one; this returns {} for such a recipe.
    fold_hybrid_scoring_kwargs,
    # One feature space per search family the pool holds, added 2026-09-19.
    # n11's loop, in one place. See SPACE_FOR_FAMILY below.
    pool_recipes_by_family, describe_pool_family, POOL_SUBSETS,
    extract_top3_base_reps, DOWNSTREAM_MODELS_NOISE,
    fold_reselection, build_fold_reselection_input,
    # 2026-09-18, owner's instruction: the top 2 rows go in everywhere as an
    # extra, and nothing already reported changes. All three read rows the
    # per-model searches already stored and fit no model: 0 new model fits.
    top_base_reps, top_hybrids, hybrid_deltas,
)
from kirby.utils.noise_estimation import (
    ESTIMATOR_REGISTRY, ActivityCliffAwareEstimator, GaussianProcessEstimator,
    QuantileForestEstimator, smooth_by_scores, weights_from_scores,
    clip_to_range,
)
from kirby.utils.noisy_labels import oof_predict

# Which result-file space each family of stored search is built into. 'hybrid'
# is the predictive one, and every key below that does not name a space holds
# it, so nothing reported before 2026-09-19 moves. See the per-family loop in
# run() for why one recipe is not enough.
HYBRID_SPACE = 'hybrid'
SPACE_FOR_FAMILY = {
    'predictive': HYBRID_SPACE,
    'similarity': 'hybrid_similarity',
}


# Multi-seed noise averaging. Each fold runs noise injection at multiple
# seeds and aggregates — reduces dependence on a single noise realization.
NOISE_SEEDS = [0, 1, 2]

# Every cleanup this file can apply, i.e. every strategy that is not one of the
# two baselines. Named once here because `_evaluate_fold` used to carry the
# same four-name tuple written out three times, and adding a fifth meant
# finding all three.
CLEANUPS = ('remove', 'downweight', 'label_smooth', 'knn_smooth',
            'oof_smooth', 'gp_smooth', 'quantile_clip',
            # Added 2026-09-23 from the label-noise menu in the compass review.
            # Both keep every molecule, which is the family that review
            # recommends over deletion on small datasets.
            'huber_weight', 'js_shrink', 'prox_smooth',
            # Added 2026-09-23. The only cleanup that reads no suspicion score
            # and no label: the weight comes from how many times the molecule
            # was measured. See SCORE_FREE_CLEANUPS below.
            'invvar_weight')

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
SCORE_FREE_CLEANUPS = ('quantile_clip', 'invvar_weight')

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
FLAG_SHARE_CLEANUPS = ('remove', 'knn_smooth', 'prox_smooth')


def reads_flag_share(cleanup, is_clf):
    """Does this cleanup's cleaned training set depend on the flag share?"""
    return (cleanup in FLAG_SHARE_CLEANUPS
            or (bool(is_clf) and cleanup == 'label_smooth'))


def flag_share_label(cleanup, share):
    """The strategy name one (cleanup, flag share) is filed under."""
    return f'{cleanup}@flag_share{float(share):g}'


def split_strategy(label):
    """(cleanup, flag share or None) from a strategy name."""
    head, sep, tail = str(label).partition('@flag_share')
    return (head, float(tail)) if sep else (head, None)


def expand_flag_shares(strategies, is_clf, shares=FLAG_SHARES):
    """Each cleanup that reads the flag share once per share, in place."""
    out = []
    for st in strategies:
        if reads_flag_share(st, is_clf):
            out += [flag_share_label(st, sh) for sh in shares]
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
                            'js_shrink')

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

# 2026-09-27, owner's decision: n07 also runs on the labels as recorded. Under
# this condition name nothing is injected at all -- not a zero-level draw, which
# on classification still flips a few labels (calibrate_flip_probability has a
# floor). The 'noisy' training labels ARE the recorded ones, so every cleanup's
# D-versus-D' row answers: does it help on the dataset's own experimental
# noise. 'no_noise' and 'noisy_baseline' are then the same fit, and every
# recovery percentage is NaN by the GAP_MIN_FRAC floor.
AS_THEY_COME = 'as_they_come'


def _inject_or_recorded(y_all, spec, *, train_idx, is_clf, dataset_name,
                        groups=None):
    """inject_train_labels, or the recorded labels when spec is None
    (AS_THEY_COME). Same return shape: (y_train, moved_mask, provenance).

    groups: the FULL scaffold-group array (dataset_policy.noise_groups_for),
    needed by grouped_shifted and grouped_wider. DIRTY-LOCAL 2026-09-28: the
    three callers in _run_one_condition passed none, so the default sweep
    raised ValueError inside make_noisy_train_labels on grouped_shifted, the
    second condition, and ended the whole n07 sweep."""
    if spec is None:
        y_tr = np.asarray(y_all)[np.asarray(train_idx)].copy()
        return (y_tr, np.zeros(len(y_tr), dtype=bool),
                {'condition': AS_THEY_COME, 'injected': False})
    return inject_train_labels(y_all, spec, train_idx=train_idx,
                               is_clf=is_clf, dataset_name=dataset_name,
                               groups=groups)


def _noise_spec(dataset_name, noise_fraction, level, noise_seed, is_clf=False,
                condition=DEFAULT_CONDITION):
    """One injected condition at this dataset's WIDE-ERROR fraction.

    ⛔ `condition` ADDED 2026-09-23. This pinned 'outlier_p05' and nothing
    could change it, so n07 -- the experiment that compares the cleaning
    methods against each other -- ran ONE noise condition at ONE level for the
    life of the project. Everything the suite knows about how the cleaning
    methods behave under different noise had to come through here and could
    not be asked.

    `noise_fraction` OVERRIDES p, AND ONLY FOR THE OUTLIER FAMILY. p is the
    share of training molecules given an error about three times as wide as
    every other molecule's, and it is the knob n07's own design turns. The
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
    from dataclasses import replace
    if condition == AS_THEY_COME:
        return None
    if is_clf:
        return noise_spec_for(dataset_name, 'uniform', noise_fraction,
                              noise_seed=noise_seed, is_clf=True)
    spec = noise_spec_for(dataset_name, condition, level,
                          noise_seed=noise_seed)
    if str(condition).startswith('outlier') and noise_fraction is not None:
        return replace(spec, params=(('p', float(noise_fraction)),))
    return spec


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
    offered = {'k': 10, 'metric': None, 'is_clf': is_clf,
               'groups': groups, 'exclude_groups': groups}
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


def cleaned_training_set(strategy, X_tr_s, y_tr, noise_scores, noise_fraction,
                         is_clf=False, groups=None, strength=None,
                         targets=None, measurement_counts=None,
                         dataset_name=None, similarity_neighbours=None,
                         similarity_k=None):
    """What one of the cleanups hands the downstream fit, before any fit runs.

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

    def _given_neighbours(k_today):
        # The neighbour indices from the caller's similarity setting, as
        # (n_tr, k) into the training rows with the molecule itself excluded.
        from experiments.noise.similarity_space import neighbours_at
        k = k_today if similarity_k is None else int(similarity_k)
        return neighbours_at(similarity_neighbours, k)[0]

    if strategy == 'remove':
        threshold = np.percentile(noise_scores, flag_percentile)
        keep = noise_scores < threshold
        if keep.sum() < 10:
            return None
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
            return None
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
                return None
        oof = np.asarray(oof, dtype=float)
        resid = np.asarray(y_tr, dtype=float) - oof
        tau2 = float(np.var(resid))
        if not np.isfinite(tau2) or tau2 <= 0:
            return None
        # The suspicion score is not in label units, so it is put on the same
        # scale as the residuals before being read as a per-molecule variance.
        sc = np.asarray(noise_scores, dtype=float)
        rng_sc = np.ptp(sc)
        if not np.isfinite(rng_sc) or rng_sc <= 0:
            return None
        sc_norm = (sc - sc.min()) / rng_sc
        sigma2 = sc_norm * tau2 * (1.0 if strength is None else float(strength))
        shrink = sigma2 / (sigma2 + tau2)
        y_new = (1.0 - shrink) * np.asarray(y_tr, dtype=float) + shrink * oof
        return {'X': X_tr_s, 'y': y_new, 'sample_weight': None,
                'keep': None, 'label_column': y_new}

    if strategy == 'label_smooth':
        if similarity_neighbours is not None:
            _nbrs = _given_neighbours(min(10, len(y_tr) - 1) - 1)
        if is_clf:
            # Label smooth uses neighbor means — use neighbor mode for clf
            if similarity_neighbours is None:
                nn = make_nearest_neighbors(n_neighbors=min(10, len(y_tr) - 1))
                nn.fit(X_tr_s)
                _, nbr_idx = nn.kneighbors(X_tr_s)
                _nbrs = nbr_idx[:, 1:]
            from scipy.stats import mode as scipy_mode
            nbr_modes = scipy_mode(y_tr[_nbrs], axis=1).mode.ravel()
            # Only correct high-noise-score samples
            threshold = np.percentile(noise_scores, flag_percentile)
            y_smoothed = y_tr.copy()
            y_smoothed[noise_scores >= threshold] = nbr_modes[noise_scores >= threshold]
        else:
            if similarity_neighbours is None:
                nn = make_nearest_neighbors(n_neighbors=min(10, len(y_tr) - 1))
                nn.fit(X_tr_s)
                _, nbr_idx = nn.kneighbors(X_tr_s)
                _nbrs = nbr_idx[:, 1:]
            nbr_means = np.mean(y_tr[_nbrs], axis=1)
            y_smoothed = smooth_by_scores(
                y_tr, noise_scores, nbr_means,
                alpha=0.5 if strength is None else float(strength),
                method='proportional')
        return {'X': X_tr_s, 'y': y_smoothed, 'sample_weight': None,
                'keep': None, 'label_column': y_smoothed}

    if strategy == 'knn_smooth':
        threshold = np.percentile(noise_scores, flag_percentile)
        flagged = noise_scores >= threshold
        if similarity_neighbours is None:
            nn = make_nearest_neighbors(n_neighbors=min(10, len(y_tr) - 1))
            nn.fit(X_tr_s)
            _, nbr_idx = nn.kneighbors(X_tr_s)
            _nbrs = nbr_idx[:, 1:]
        else:
            _nbrs = _given_neighbours(min(10, len(y_tr) - 1) - 1)
        if is_clf:
            from scipy.stats import mode as scipy_mode
            nbr_modes = scipy_mode(y_tr[_nbrs], axis=1).mode.ravel()
            y_corrected = y_tr.copy()
            y_corrected[flagged] = nbr_modes[flagged]
        else:
            nbr_means = np.mean(y_tr[_nbrs], axis=1)
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
        n_tr = len(y_tr)
        from kirby.utils.noise_estimation import (forest_leaf_ids,
                                                  forest_proximity_distance)
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
            nbr_idx = _given_neighbours(min(10, n_tr - 1))
        else:
            leaves = forest_leaf_ids(X_tr_s, y_tr, is_clf)
            D = forest_proximity_distance(leaves)
            np.fill_diagonal(D, np.inf)          # never its own neighbour
            k = min(10, n_tr - 1)
            nbr_idx = np.argsort(D, axis=1)[:, :k]
        y_corrected = np.asarray(y_tr).copy()
        if is_clf:
            from scipy.stats import mode as scipy_mode
            nbr_vals = scipy_mode(np.asarray(y_tr)[nbr_idx], axis=1).mode.ravel()
        else:
            nbr_vals = np.mean(np.asarray(y_tr)[nbr_idx], axis=1)
        y_corrected[flagged] = nbr_vals[flagged]
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
            alpha=0.5 if strength is None else float(strength),
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
                return None
            est = GaussianProcessEstimator(n_folds=REPAIR_OOF_SPLITS,
                                           groups=groups)
            est.estimate(X_tr_s, y_tr)
            target = np.asarray(est.predictions_, dtype=float)
        y_smoothed = smooth_by_scores(
            y_tr, noise_scores, target,
            alpha=0.5 if strength is None else float(strength),
            method='proportional')
        return {'X': X_tr_s, 'y': y_smoothed, 'sample_weight': None,
                'keep': None, 'label_column': y_smoothed}

    if strategy == 'quantile_clip':
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
                sc = StandardScaler()
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
#                               model's top-2 single representations, the four
#                               augmented spaces, the two hybrids -- every one
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


def sweep_settings():
    """The similarity settings this run sweeps, in order.

    similarity_space.settings_from_env() reads KIRBY_SIMILARITY_SETTINGS:
    unset or 'all' is every setting (66 when every space exists), 'model_input'
    is today's behaviour alone, a comma list is those names. Under unset or
    'all' MODEL_INPUT is put first, because enumerate_settings leaves it out
    and it is the row every other setting is read against: today's neighbours.
    A named list is taken as named.
    """
    from experiments.noise import similarity_space as ss
    raw = str(os.environ.get('KIRBY_SIMILARITY_SETTINGS') or '').strip()
    settings = list(ss.settings_from_env())
    if raw in ('', 'all') and not any(
            s.space == ss.MODEL_INPUT_SPACE for s in settings):
        settings = [ss.MODEL_INPUT] + settings
    return settings


def sweep_ks():
    """Every neighbour count swept: 5, 9, 10, 20 (similarity_space.SWEPT_KS).
    9 is what label_smooth and knn_smooth use today, 10 what prox_smooth and
    the four neighbour estimators use."""
    from experiments.noise.similarity_space import SWEPT_KS
    return tuple(int(k) for k in SWEPT_KS)


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
    sc = StandardScaler()
    return sc.fit_transform(X_tr), sc.transform(X_va)


def fold_similarity_spaces(dataset_name, settings, *, model_input_tr,
                           model_input_va, reps, train_idx, val_idx, y_train,
                           groups_train=None, base_reps=(), recipes=None,
                           augmentations=None, model_name=None,
                           model_factory=None, metric=None, task_type=None):
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
            (_shared.noise_base_reps_by_model). base_1, base_2 and the four
            augmented spaces are built from them.
        recipes: {'hybrid': (Blueprint, aug_condition),
                  'hybrid_similarity': (Blueprint, aug_condition)}, the model's
            precomputed recipes. A space whose recipe is absent is reported
            unavailable, never faked.
        augmentations: {name: full matrix}, data['augmentations'].
        model_name / model_factory / metric: the downstream model and the
            get_domain_metric triple, for the builds' accept-or-reject step.

    Returns (spaces_train, spaces_val, unavailable), where unavailable is
    {space: reason}. Every build goes through a noise_cache wrapper
    (cached_scaled, cached_hybrid_for_fold), so a space shared by several
    experiments, conditions at level 0, or models holding one recipe is built
    once.
    """
    from experiments.noise import noise_cache as nc
    from experiments.noise import similarity_space as ss
    from experiments.noise.noise_helpers import recipe_is_geometry
    need = settings_need_spaces(settings)
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
        wanted_aug = [st for st in ss.AUG_STRATEGIES
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

    for sp in ('hybrid', 'hybrid_similarity'):
        if sp not in need:
            continue
        rec = (recipes or {}).get(sp)
        if rec is None or rec[0] is None:
            unavailable[sp] = ('no stored recipe of this family for this model '
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
    """RF and SVM, then whatever the experiment fits itself, once each."""
    return tuple(dict.fromkeys(list(SWEEP_ROW_MODELS) + [m for m in extra if m]))


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

    Rows, per cleanup in `strategies`:
      a cleanup that reads no neighbours    one row per model, similarity
                                            setting 'none', the scores the pass
                                            already has (the detector DETECTOR,
                                            which is not a neighbour detector)
      label_smooth, knn_smooth              one row per model per (setting, k)
                                            in the sweep; MODEL_INPUT at k=9 is
                                            the cleanup the legacy block ran
      prox_smooth                           per k under forest_proximity only
    When DETECTOR is a neighbour detector every score-driven cleanup is one row
    per (setting, k) instead, scored on that setting's lists built with the
    scaffold groups excluded, as _compute_noise_scores passes them.

    detection_acc / grade_detectors: when a corrupted subset exists, every name
    in grade_detectors that reads neighbours is graded under every (setting, k)
    into detection_acc[detector][setting][k], a list of detection_scores
    readings.
    """
    from experiments.noise import noise_cache as nc
    from experiments.noise.noise_cache import NEIGHBOUR_DETECTORS
    sink = ctx['sink']
    ds = ctx['dataset_name']
    is_clf = ctx['is_clf']
    settings = ctx['settings']
    models = row_models([model_name])
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
                is_clf=is_clf, groups=groups_tr, dataset_name=ds, **sim)
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
        return
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
    # Lists for the cleanups: no group exclusion, as label_smooth and
    # knn_smooth never excluded one (n07:869-919). Lists for a detector: the
    # scaffold groups excluded, as _compute_noise_scores passes them.
    nb_plain = (fold_neighbours(ds, settings, spaces_tr, spaces_va, y_tr_noisy,
                                smiles_train=smi_tr, smiles_val=smi_va,
                                is_clf=is_clf) if reads_nbrs else {})
    nb_grp = (fold_neighbours(ds, settings, spaces_tr, spaces_va, y_tr_noisy,
                              exclude_groups=groups_tr, smiles_train=smi_tr,
                              smiles_val=smi_va, is_clf=is_clf)
              if (swept_scored or want_grading) else {})
    for s in settings:
        L = nb_plain.get(s.name)
        Lg = nb_grp.get(s.name)
        for L_ in (L, Lg):
            if L_ is not None and not L_.get('available'):
                ctx['unavailable'].setdefault(f'{model_input}|{s.name}',
                                              L_.get('reason'))
        ks = sorted(set(usable_ks(L)) | set(usable_ks(Lg)))
        for k in ks:
            scores_k = None
            if (swept_scored or want_grading) and Lg is not None and \
                    Lg.get('available') and k in usable_ks(Lg):
                for d in grade if want_grading else ():
                    sc = nc.cached_detector_scores(
                        ds, X_tr, y_tr_noisy, is_clf=is_clf, detector=d,
                        groups=groups_tr, similarity_neighbours=Lg,
                        similarity_k=k)
                    if sc is None:
                        continue
                    det = detection_scores(corrupted_mask, sc, noise_strategy)
                    if det is not None:
                        (detection_acc.setdefault(d, {}).setdefault(s.name, {})
                         .setdefault(str(k), []).append(det))
                if swept_scored:
                    scores_k = nc.cached_detector_scores(
                        ds, X_tr, y_tr_noisy, is_clf=is_clf, detector=DETECTOR,
                        groups=groups_tr, similarity_neighbours=Lg,
                        similarity_k=k)
                    for st in swept_scored:
                        _row(st, None if scores_k is None
                             else _clean(st, scores_k), s.name, f'k={k}')
            if L is None or not L.get('available') or k not in usable_ks(L):
                continue
            sc_for_nb = scores_k if det_reads else noise_scores
            if sc_for_nb is None:
                continue
            for st in reads_nbrs:
                if _base(st) == 'prox_smooth' and s.space != 'forest_proximity':
                    continue
                _row(st, _clean(st, sc_for_nb, similarity_neighbours=L,
                                similarity_k=k), s.name, f'k={k}')


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
            continue
        for k in usable_ks(L):
            for method, msetting, cleaned in make_dprimes(s, L, k):
                ms = f'k={k}' if not msetting else f'k={k}|{msetting}'
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
    """
    return sweep_conditions(
        _run_one_condition, 'n07_remediation',
        data, config, output_dir, dataset_name,
        default_conditions=[AS_THEY_COME] + list(conditions_for_task(data)),
        kwargs=kwargs, groups_needed=GROUPED_STRATEGIES)


def _run_one_condition(data, config, output_dir, dataset_name, **kwargs):
    """N07: Noise remediation strategies, ONE condition."""
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
    # 2026-09-20: `noise_helpers.run_all` puts the caller's model list in
    # config['models'] and passes no `models` kwarg. Reading only the module
    # default meant the configured list was silently dropped and every run
    # scored the full five-model roster. Config first, kwarg as override.
    downstream_models = list(
        kwargs.get('models') or config.get('models') or DOWNSTREAM_MODELS_NOISE)
    model_factories = get_model_factories(
        downstream_models, task_type=task_type,
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
    condition = kwargs.get('noise_condition') or DEFAULT_CONDITION
    # How much noise the corrupted rows receive. This experiment had no level axis
    # at all -- the magnitude was hardcoded at 0.3 of the label range -- so the
    # default is one unit of real error for this dataset: 0.54 log units on herg_ki
    # (Kramer et al. 2012), 0.35 on caco2 (Bentz et al. 2013 Table 6), half the label
    # SD where nothing is published.
    noise_level = kwargs.get('noise_level', reference_noise_level(dataset_name))
    # Nothing injected under AS_THEY_COME: the level is 0 and the three draws
    # would be three identical passes, so one is kept.
    if condition == AS_THEY_COME:
        noise_level = 0.0
    _seeds_here = NOISE_SEEDS[:1] if condition == AS_THEY_COME else NOISE_SEEDS
    noise_provenance = {}
    # DIRTY-LOCAL 2026-09-28: scaffold groups for the two grouped conditions,
    # handed to every injection below (see _inject_or_recorded).
    from experiments.dataset_policy import noise_groups_for as _ngf
    _noise_groups = _ngf(data)[0]
    # How well the detector found the corrupted molecules, before any cleanup.
    # One reading per (fold, draw) per feature space. Owner's decision 2026-09-12.
    detection_acc = {}
    # Keyed by detector name rather than by feature space: which name
    # FOUND the corrupted molecules, against DETECTOR, which drove the
    # cleanups. Both are graded in the corrupted-label space.
    detection_by_detector = {}
    # An empty `detection` block used to be unreadable: it looked the same
    # whether the grading was never reached, the detector would not fit, or
    # detection_scores declined because no molecule was corrupted. These say
    # which. Counts of (fold, noise seed) passes, not of molecules.
    detection_tally = {'reached': 0, 'detector_returned_none': 0,
                       'scores_declined': 0, 'graded': 0,
                       'fold_space_unavailable': 0}

    sample_X = next(iter(reps.values()))

    # Per-model blueprint fetch (matched evaluator). Each call is cached.
    # rf is the canonical "top-3 base reps" reference, so fetch it first
    # both for that print and as a fallback if the pool lacks a model's
    # blueprint entry.
    bp_by_model = {}
    aug_by_model = {}
    # Keep the winning per-model SearchResult so we can persist an honest
    # hybrid-vs-baseline re-scoring block (see fold_reselection block below).
    search_by_model = {}
    # Every model in DOWNSTREAM_MODELS_NOISE that was ASKED for, and what
    # happened to it. A model with no blueprint contributes no row to
    # per_model_per_strategy, so without this record the file looks like the
    # experiment was configured for the smaller set. 'ok' means a blueprint
    # was obtained; anything else is the reason it was not.
    models_requested = list(model_factories.keys())
    blueprint_status = {m: 'not_attempted' for m in models_requested}
    # DECISION (2026-09-12): n07's pool subset includes similarity hybrids.
    # Three of its four remediation strategies (downweight, label_smooth,
    # knn_smooth) overwrite or reweight a molecule's label using its
    # neighbours' labels, so the recipe may now also come from a search run
    # under objective='similarity' — one whose features were chosen to make
    # the distance geometry predict a molecule's activity from its five
    # nearest neighbours. This is NOT the distance=True flag, which n07 does
    # not pass and which stays unset.
    POOL_SUBSET = 'predictive_and_similarity'
    # Which search family each model's recipe came from, and every eligible
    # family's own best, so a reader can tell the two apart. The two families
    # are scored in different units (the dataset's own metric against a
    # nearest-neighbour correlation), so nothing here ranks one against the
    # other — get_best_search_for_downstream returns the first family the
    # subset names when the metrics disagree, and says so in objective_choice.
    hybrid_family_by_model = {}
    # Each model's own pool metadata, kept so the per-family loop below reads
    # every family's best for that model and not only the one the pool returned.
    meta_by_model = {}
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
                pool_subset=POOL_SUBSET,
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
                # Copy first. The object here is the recipe stored inside
                # data['precomputed_searches'], shared with every other reader
                # of that pool in this process; setting the field in place
                # changes what they rebuild too.
                try:
                    _bp = copy.copy(_bp)
                    _bp.augmentation_strategy = _aug_cond
                except Exception:
                    pass
            bp_by_model[_mname] = _bp
            aug_by_model[_mname] = _aug_cond
            search_by_model[_mname] = _search
            meta_by_model[_mname] = _meta
            blueprint_status[_mname] = 'ok'
            # Recorded per model, because each model resolves its own recipe
            # and the families can differ between them.
            hybrid_family_by_model[_mname] = {
                'objective': _meta.get('objective'),
                'objective_choice': _meta.get('objective_choice'),
                'pool_subset': _meta.get('pool_subset'),
                'search_metric_name': _meta.get('metric_name'),
                'search_lower_is_better': _meta.get('lower_is_better'),
                'tier': _meta.get('tier'),
                'aug_condition': _aug_cond,
                'by_objective': {
                    _o: {'tier': _d.get('tier'),
                         'aug_condition': _d.get('aug_condition'),
                         'score_mean': _d.get('score_mean'),
                         'metric_name': _d.get('metric_name'),
                         'lower_is_better': _d.get('lower_is_better')}
                    for _o, _d in (_meta.get('by_objective') or {}).items()
                },
            }
            print(f"    {_mname}: recipe from the "
                  f"{_meta.get('objective')!r} search family "
                  f"({_meta.get('objective_choice')}), search metric "
                  f"{_meta.get('metric_name')}")
        except Exception as _exc:
            blueprint_status[_mname] = (
                f'fetch_raised: {type(_exc).__name__}: {_exc}')
            print(f"  WARNING: blueprint fetch failed for {_mname} ({_exc})")

    # ── One recipe per (model, family), not one per model ─────────────────
    # ⛔ Added 2026-09-19. POOL_SUBSETS names the two families in PREFERENCE
    # order and the pool returns the first that has an entry, because their
    # stored scores are in different units. A predictive recipe always exists,
    # so from 2026-09-12 to 2026-09-19 this file asked for both families, built
    # the predictive one and wrote the other into the result file as a score it
    # never measured in.
    #
    # It matters here because three of the four older cleanups -- downweight,
    # label_smooth and knn_smooth -- overwrite or reweight a label using its
    # NEIGHBOURS' labels, and the similarity search chooses features so that
    # exactly that distance predicts activity from the five nearest
    # neighbours. DETECTOR itself is 'cross_val', an out-of-fold model
    # residual, so the suspicion score is not a neighbour reading;
    # activity_cliff_aware, graded beside it, is.
    #
    # HYBRID_SPACE is still the predictive recipe, so every number reported
    # before 2026-09-19 still means what it meant.
    recipes_for_model = {}
    space_objective = {}
    hybrid_spaces_unavailable = {}
    # First per-fold build failure per space, filled by _fold_space below. A
    # space whose every fold failed would otherwise leave no rows and no reason.
    _fold_space_errors = {}
    for _mname in models_requested:
        _recipes, _unavail = pool_recipes_by_family(
            meta_by_model.get(_mname), SPACE_FOR_FAMILY,
            eligible_families=POOL_SUBSETS[POOL_SUBSET])
        _per_space = {}
        for _r in _recipes:
            _r_bp = _r['blueprint']
            _r_aug = _r['aug_condition'] or 'uncapped'
            if getattr(_r_bp, 'augmentation_strategy', None) != _r_aug:
                # Copy first, for the reason the per-model fetch above gives:
                # the recipe object is shared with every other reader of the
                # pool in this process.
                try:
                    _r_bp = copy.copy(_r_bp)
                    _r_bp.augmentation_strategy = _r_aug
                except Exception:
                    pass
            _per_space[_r['space']] = (_r_bp, _r_aug)
            space_objective[_r['space']] = _r['objective']
        recipes_for_model[_mname] = _per_space
        if _unavail:
            hybrid_spaces_unavailable[_mname] = _unavail
    for _sp in sorted(space_objective):
        print(f"  hybrid space {_sp}: family {space_objective[_sp]!r}")
    for _mname, _unavail in hybrid_spaces_unavailable.items():
        for _sp, _why in _unavail.items():
            print(f"  {_mname}: no {_sp} space — {_why}")

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
            'pool_subset': POOL_SUBSET,
            'hybrid_family_by_model': hybrid_family_by_model,
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
            # Same widened subset as the per-model fetch above (decision of
            # 2026-09-12), so this fallback reads the same pool and cannot
            # silently fall back to prediction-built searches only.
            _any_search, _ = get_best_search_for_downstream(
                reps, labels, groups,
                downstream_model=_any_name, budget=budget,
                augmentations=augmentations,
                dataset_name=dataset_name, task_type=task_type,
                data=data,
                pool_subset=POOL_SUBSET,
            )
            top3 = extract_top3_base_reps(_any_search) if _any_search else None
        except Exception:
            top3 = None
    if top3:
        print(f"  Top-3 base reps (ranked by RF baseline): "
              f"{', '.join(f'{n}={s:.4f}' for n, s in top3)}")

    # ── The similarity axis and the D-versus-D' rows, added 2026-09-25 ──────
    # Plan sections 1-2. The model is fitted on exactly what it was fitted on
    # before; the settings below move only the neighbour lookup of
    # label_smooth and knn_smooth and the grading of activity_cliff_aware.
    # Every key the result dict carried before 2026-09-25 is the MODEL_INPUT
    # setting at today's k and is computed by the unchanged code.
    #
    # base_1 / base_2 are each model's top-2 single representations ranked by
    # its own stored search (_shared.noise_base_reps_by_model, owner's decision
    # 2026-09-25), read from the same searches that gave each model its
    # recipe. 0 fits.
    from experiments._shared import noise_base_reps_by_model
    sim_settings = sweep_settings()
    try:
        sim_base_by_model, sim_base_missing = noise_base_reps_by_model(
            search_by_model, models_requested, reps=reps,
            dataset_name=dataset_name, verbose=False)
    except Exception as _exc:
        sim_base_by_model = {}
        sim_base_missing = {'_all': f'{type(_exc).__name__}: {_exc}'}
    sink = RowSink(output_dir, 'n07_remediation', dataset_name,
                   kwargs.get('noise_condition'))
    sim_detection = {}
    sweep_ctx = {
        'experiment': 'n07_remediation', 'dataset_name': dataset_name,
        'sink': sink, 'settings': sim_settings, 'metric': (
            metric_name, metric_fn, lower_is_better),
        'task_type': task_type, 'is_clf': is_clf,
        'noise_fraction': noise_fraction, 'condition': condition,
        'level': noise_level, 'smiles': data.get('smiles'), 'reps': reps,
        'augmentations': augmentations, 'factories': model_factories,
        'unavailable': {},
    }
    print(f"  Similarity axis: {len(sim_settings)} settings x ks "
          f"{list(sweep_ks())}; rows -> {sink.path}")

    strategies = [
        'no_noise',         # clean baseline (no injection)
        'noisy_baseline',   # inject noise, do nothing
        'remove',           # detect + remove flagged samples
        'downweight',       # detect + weight by 1/(1+score)
        'label_smooth',     # detect + smooth labels toward neighbor means
        'knn_smooth',       # replace flagged labels with kNN-mean in hybrid space
    ]
    # Added 2026-09-17. The first two move every label toward a predicted value
    # in proportion to its suspicion score, the same shape as label_smooth with
    # a different target:
    #   oof_smooth     toward what a LightGBM model trained on the other
    #                  training molecules predicts for this one, on a
    #                  scaffold-group-aware inner split
    #   gp_smooth      toward the posterior mean of a Gaussian process fitted
    #                  the same way
    # The third moves nothing toward anything:
    #   quantile_clip  each molecule gets the 10th-to-90th-percentile range of
    #                  the labels its own leaves' training molecules carry, from
    #                  a quantile regression forest; a label inside its range is
    #                  left alone and a label outside is pulled onto the edge
    # All three write a numeric label, so none of them runs on a classification
    # dataset. Which ones ran is recorded below rather than left to be inferred
    # from an absent row.
    if not is_clf:
        strategies.extend(REGRESSION_ONLY_CLEANUPS)
    # Each cleanup that flags a share of the training molecules runs once per
    # share in FLAG_SHARES, filed as '<cleanup>@flag_share<s>' (2026-09-28).
    flag_shares = [float(x) for x in (kwargs.get('flag_shares') or FLAG_SHARES)]
    strategies = expand_flag_shares(strategies, is_clf, flag_shares)
    # robust_loss removed (2026-05-26 audit decision): the strategy swapped
    # the iteration's downstream model for HuberRegressor, conflating "data
    # cleanup" with "different model entirely" — its recovery numbers were
    # not comparable to the four cleanup strategies above. If you want to
    # test Huber separately, run a dedicated experiment with Huber as the
    # baseline model across all strategies.

    # --- Which models and which representations the three expensive cleanups
    # --- run under. USER DECISION 2026-09-17, measured in THIS run.
    #
    # The three cleanups added today each fit five models before they repair
    # anything. On a training fold of 1,200 molecules the three together take
    # 275 seconds where a molecule is described by 200 features and 39 minutes
    # where it is described by 2048. So they are measured under the two
    # downstream models that fit this dataset best, on the two representations
    # each of those models fits best, and in those models' hybrid features.
    #
    # The ranking is measured here, not read from a stored search. A stored
    # search was written by another run on another machine, and on the cluster
    # it can be stale or missing. `rank_models_and_reps` fits every model on
    # every representation using the uncorrupted activity values, one fit per
    # model, representation and fold, and ranks on the held-out error. Every
    # score it measured is written out under 'model_rep_ranking'.
    #
    # The four cleanups n07 already had are NOT restricted by any of this. None
    # of them fits a model, so restricting them would save nothing and would
    # throw away rows earlier runs have.
    ranking = None
    top_models = list(bp_by_model.keys())
    top_reps_by_model = {}
    cap_features, cap_source = max_cleanup_features(kwargs)
    # The same ceiling applied to the hybrid features. A recipe's budget is the
    # most features its hybrid can hold, so a model whose budget exceeds the
    # ceiling keeps the four older cleanups in its hybrid and loses the three.
    hybrid_too_wide = {}
    if cap_features is not None:
        for _m, _bp in bp_by_model.items():
            _b = int(getattr(_bp, 'budget', 0) or 0)
            if _b > cap_features:
                hybrid_too_wide[_m] = _b
    if not is_clf:
        print(f"\n  Ranking every downstream model on every representation, "
              f"uncorrupted activity values, to choose where the three "
              f"model-fitting cleanups run. Ceiling: "
              f"{cap_features if cap_features else 'none'} features, from "
              f"{cap_source}.")
        # Cached, 2026-09-25 (plan section 3): the ranking fits every model on
        # every representation on the CLEAN labels, so it is the same for every
        # noise condition and was refitted once per condition. Keyed on the
        # content of every representation, the labels, the groups, the fold
        # settings, each model's class and parameters and the metric.
        # Representation NAMES are in the key because the returned dict is
        # keyed by them (same exception as cached_hybrid_for_fold's
        # want_feature_info).
        from experiments.noise import noise_cache as _nc
        _rank_models = [m for m in models_with_blueprint]
        _rank_key = {
            'reps': [[r, _nc.array_hash(reps[r])] for r in reps],
            'labels': _nc.array_hash(labels), 'groups': _nc.array_hash(groups),
            'sample_X': _nc.array_hash(sample_X),
            'models': [[m, _nc.model_signature(model_factories[m]())]
                       for m in _rank_models if m in model_factories],
            'n_folds': int(n_folds), 'mode': str(mode),
            'metric': str(metric_name), 'lower_is_better': bool(lower_is_better),
            'is_clf': bool(is_clf), 'max_features': cap_features,
            'n_top_models': N_TOP_MODELS, 'n_top_reps': N_TOP_REPS_PER_MODEL}
        ranking = _nc.cached(
            dataset_name, 'n07_model_rep_ranking', _rank_key,
            lambda: rank_models_and_reps(
                reps, labels, groups, model_factories,
                _rank_models,
                n_folds, mode, sample_X, metric_fn, metric_name,
                lower_is_better, is_clf, max_features=cap_features))
        if ranking['reps_excluded_as_too_wide']:
            print("    barred as too wide: " + ", ".join(
                f"{r} ({w} features)"
                for r, w in ranking['reps_excluded_as_too_wide'].items()))
        top_models = [m for m in ranking['top_models']
                      if m in bp_by_model and m not in hybrid_too_wide]
        _dropped = [m for m in ranking['top_models'] if m in hybrid_too_wide]
        for _m in _dropped:
            print(f"    {_m} keeps the four older cleanups only: its recipe's "
                  f"budget is {hybrid_too_wide[_m]} features, above the "
                  f"{cap_features}-feature ceiling")
        top_reps_by_model = {m: ranking['top_reps'][m] for m in top_models}
        for _m in ranking['model_order']:
            _best_rep = min(ranking['scores'][_m],
                            key=lambda r: (ranking['scores'][_m][r]
                                           if lower_is_better
                                           else -ranking['scores'][_m][r]))
            print(f"    {_m}: best {metric_name}="
                  f"{ranking['scores'][_m][_best_rep]:.4f} on {_best_rep} "
                  f"({ranking['n_features'][_best_rep]} features)"
                  + ("  <- kept" if _m in top_models else ""))
        for _m in top_models:
            print(f"    {_m} keeps: " + ", ".join(
                f"{r} ({ranking['n_features'][r]} features, {metric_name}="
                f"{ranking['scores'][_m][r]:.4f})"
                for r in top_reps_by_model[_m]))

    results = {
        'metric': metric_name,
        # USER DECISION 2026-09-17. Which downstream models and which
        # representations the three model-fitting cleanups were measured under,
        # and every score that chose them. Measured in this run; nothing here
        # came from a stored search.
        'model_rep_ranking': ranking,
        'expensive_cleanups': list(REGRESSION_ONLY_CLEANUPS),
        # USER DECISION 2026-09-18: 2000 features. A set of features wider than
        # this is still measured and still scored, and is only barred from
        # carrying the three model-fitting cleanups, whose cost is driven by
        # the feature count more than by anything else.
        'max_cleanup_features': cap_features,
        'max_cleanup_features_source': cap_source,
        'hybrid_budgets_above_ceiling': dict(hybrid_too_wide),
        'expensive_cleanup_models': list(top_models),
        'expensive_cleanup_reps_by_model': dict(top_reps_by_model),
        'expensive_cleanup_scope_note': (
            'oof_smooth, gp_smooth and quantile_clip each fit five models '
            'before they repair anything, so they run under the two downstream '
            'models that fit this dataset best, on the two representations each '
            'of those models fits best, and in those two models\' hybrid '
            'features. The four older cleanups fit no model and are not '
            'restricted: they still run under every downstream model and on '
            'every representation. model_rep_ranking holds every score that '
            'chose the two and how many features each representation has.'),
        'lower_is_better': lower_is_better,
        'noise_fraction': noise_fraction,
        'flag_shares': flag_shares,
        'flag_share_note': (
            'noise_fraction is the injector\'s share only (the outlier '
            'condition\'s p; the flip rate on classification). A cleanup that '
            'flags a share of the training molecules runs at each share in '
            'flag_shares, filed as <cleanup>@flag_share<s>; the same shares at '
            'every condition and level (owner\'s decision 2026-09-28). Until '
            'then remove, knn_smooth and prox_smooth flagged noise_fraction, '
            'the share the injector moved.'),
        'noise_level': noise_level,
        # Was the literal 'outlier' whatever condition ran; 2026-09-27.
        'noise_condition': condition,
        'dose_units': noise_dose_units(dataset_name),
        'noise_provenance': noise_provenance,
        # Detection, graded before any cleanup ran. Two feature spaces: the one
        # built from the corrupted labels and the one built from the clean ones.
        # Empty on a condition where every training molecule moves.
        #
        # ⛔ PLACEHOLDERS. Both are FILLED IN after the fold loop, near
        # `per_strategy_model`. This dict is built before that loop runs, and
        # writing the comprehension here read `detection_acc` and
        # `detection_by_detector` while both were still the empty dicts they are
        # initialised as -- so every n07 result file on disk carries
        # "detection": {} and "detection_by_detector": {} beside a
        # detection_tally saying the grading was reached 4 times and graded 8,
        # with 0 declined. The tally and the block it counts disagreed, and the
        # tally was right.
        'detection': {},
        'detection_by_detector': {},
        'detection_tally': detection_tally,
        'detection_tally_note': (
            'Counts of (fold, noise seed) passes. reached = got as far as the '
            'grading; fold_space_unavailable = the per-draw feature space could '
            'not be built, which skips the grading with everything else; '
            'detector_returned_none = the estimator would not fit; '
            'scores_declined = no molecule was corrupted, or every one was, so '
            'average precision is undefined; graded = a number was recorded. '
            'An empty detection block reads the same three ways without this.'),
        'detector_used': DETECTOR,
        'detectors_graded': list(DETECTORS_GRADED),
        # Scoping metadata: this experiment studies SYNTHETIC label noise
        # injected on the TRAINING labels only (the Outlier condition for
        # regression / random class flips for classification; validation
        # labels stay clean). No real/"fiction"-scored noise detector is
        # used, so the remediation claim is confined to the synthetic-noise
        # regime.
        'noise_regime': ('recorded labels, nothing injected'
                         if condition == AS_THEY_COME else 'synthetic'),
        # This said 'gaussian_label_noise_train_only' on the regression path
        # while _noise_spec built the outlier condition, contradicting
        # 'noise_condition' three lines up. The regression path gives a
        # randomly chosen noise_fraction of rows a much wider error; it is
        # not a gaussian spread over every row.
        'noise_injection': (
            'none: the recorded labels' if condition == AS_THEY_COME
            else 'outlier_label_noise_train_only' if not is_clf
            else 'class_flip_label_noise_train_only'),
        'claim_scope': (
            'Remediation ranking holds only under synthetically injected '
            'training-label noise; not validated on real label noise of '
            'unknown structure.'),
        'strategies': strategies,
        # The three added 2026-09-17, and whether this run applied them. They
        # rewrite a label to a numeric value, so a classification dataset drops
        # all three; saying so here means an absent row reads as a recorded
        # refusal rather than as a failure.
        'regression_only_strategies': list(REGRESSION_ONLY_CLEANUPS),
        'regression_only_strategies_ran': not is_clf,
        'regression_only_strategies_note': (
            'oof_smooth and gp_smooth move every training label toward a '
            'predicted value in proportion to that molecule\'s suspicion '
            'score — toward a LightGBM out-of-fold prediction and toward a '
            'Gaussian-process posterior mean respectively, both fitted on the '
            'other training molecules under a scaffold-group-aware inner '
            'split. quantile_clip moves no label toward anything: a quantile '
            'regression forest gives each molecule the 10th-to-90th-percentile '
            'range of the labels carried by the training molecules sharing its '
            'leaves, a label inside that range is left byte-identical and a '
            'label outside is pulled onto the edge it crossed. All three are '
            'undefined on a class label and are not run on one.'),
        # 'models' is unchanged: the models that actually produced remediation
        # scores. The three keys below say what the run was ASKED for, so an
        # absent model reads as a recorded drop-out rather than as a smaller
        # configured set. models_requested == models when nothing dropped.
        'models': list(bp_by_model.keys()),
        'models_requested': models_requested,
        'blueprint_status': blueprint_status,
        'models_missing_blueprint': models_missing_blueprint,
        'top3_base_reps': [(n, s) for n, s in top3] if top3 else [],
        # DECISION (2026-09-12): the subset of the precomputed pool this file
        # may take a recipe from now includes searches run under the
        # similarity objective, not prediction-built ones only. Recorded so a
        # reader can tell which family produced each model's recipe.
        'pool_subset': POOL_SUBSET,
        'hybrid_family_by_model': hybrid_family_by_model,
        'hybrid_family_note': (
            "Each model's recipe comes from one search family. 'predictive' "
            "scores a candidate with the dataset's own metric; 'similarity' "
            "scores it by how well a molecule's activity is predicted from "
            "its five nearest neighbours in the resulting space, stored under "
            "the metric name 'readacross'. The two are not on one scale, so "
            "objective_choice reads 'subset_order' when both families were "
            "present: the winner is the first family the subset names and "
            "every family's own best sits in by_objective. Nothing in this "
            "file ranks one family against the other."),
        # DECISION (2026-09-12), the denominator caution: record what feature
        # space each side of every recovery fraction was measured in. The top
        # and the bottom of a recovery fraction must be read together with
        # these two keys.
        'feature_space_by_block': {
            'per_model_per_strategy': (
                'hybrid, features ranked and budget split using that fold and '
                'noise draw\'s NOISY training labels; recipe from the pool '
                'family named in hybrid_family_by_model'),
            'per_model_per_strategy_oracle_features': (
                'hybrid, features ranked and budget split using that fold\'s '
                'CLEAN training labels; recipe from the same pool family'),
            'per_model_no_noise_clean_features': (
                'hybrid, CLEAN-label features, clean training labels — the '
                'ceiling as it was measured before 2026-09-11; nothing '
                'divides by it'),
            'per_strategy_<rep>/<strategy>': (
                'one base representation, no feature selection and no recipe, '
                'so its ceiling and its damaged arms are always in the same '
                'features'),
        },
        # Legacy single-model section kept for backwards compatibility;
        # populated from RF if available, else first available model.
        'per_strategy': {},
        # Canonical per-model results — every reader should prefer this.
        'per_model_per_strategy': {m: {} for m in bp_by_model},
        # The same rows for EVERY hybrid space, keyed {space: {model: {...}}}.
        # Added 2026-09-19 with the second space. The three keys without the
        # suffix hold the predictive space alone and are unchanged.
        'per_model_per_strategy_by_space': {},
        'per_model_per_strategy_oracle_features_by_space': {},
        'per_model_no_noise_clean_features_by_space': {},
        # Why a space that was asked for produced nothing, per model, and the
        # first per-fold build error per space.
        'hybrid_spaces_unavailable': hybrid_spaces_unavailable,
        'hybrid_space_build_errors': _fold_space_errors,
        'hybrid_space_objective': dict(space_objective),
        'hybrid_spaces_note': (
            'One hybrid space per family of stored search, per downstream '
            'model. A predictive search chose its features to predict the '
            "label; a similarity search chose them so that each molecule's "
            'activity is predicted by its five nearest neighbours in the '
            'distance those features define, and fits no downstream model. '
            'The two searches score their own candidates in different units '
            'and nothing here ranks one against the other. Every score under '
            "the '*_by_space' keys is measured by n07 itself in the dataset's "
            'own metric, on the same folds and the same noise draws, so those '
            'ARE comparable.'),
        # The same strategies measured in a feature space chosen from the CLEAN
        # labels, before the noise this experiment studies was injected. That
        # is what every score in this file used to be. It is kept as a named
        # oracle arm so the head start can be read off rather than assumed —
        # subtract it from per_model_per_strategy to see what choosing features
        # with the true labels was worth.
        'per_model_per_strategy_oracle_features': {},
        # The ceiling as it was measured until 2026-09-11: clean training
        # labels fitted in the CLEAN-label feature space, one fit per fold.
        # per_model_per_strategy['no_noise'] is now re-fitted inside each noise
        # draw's own feature space instead, so the top and the bottom of every
        # recovery fraction sit in the same features. This key keeps the old
        # number, named, so the two can be compared. Nothing divides by it.
        'per_model_no_noise_clean_features': {},
        'ceiling_feature_space': 'noisy_per_fold_refit',
        'ceiling_note': (
            "no_noise in per_model_per_strategy is fitted on CLEAN training "
            "labels inside the feature space each noise draw built from its "
            "own noisy training labels, one fit per (fold, noise seed). Only "
            "the features differ from per_model_no_noise_clean_features, "
            "which holds the clean-label-feature ceiling at one fit per fold. "
            "no_noise inside per_model_per_strategy_oracle_features is that "
            "same clean-label-feature ceiling, because every arm in the "
            "oracle block is measured in the clean-label space."),
        'feature_selection_labels': 'noisy_per_fold',
        'feature_selection_note': (
            'Each fold builds its hybrid from that fold and noise draw\'s NOISY '
            'training labels. The RECIPE (allocation method, importance method, '
            'internal evaluator, budget) still comes from the clean-label search '
            'pool, so the choice of recipe remains a head start; only the '
            'per-fold feature ranking and budget split are now noise-aware. '
            'That pool spans both search families named in pool_subset; '
            'hybrid_family_by_model says which one each model drew from.'),
    }

    # --- Honest hybrid-vs-baseline re-scoring input (shared-helper contract) ---
    # Persist, per model, EVERY searched hybrid candidate recipe's per-fold
    # scores AND the per-fold baseline-rep scores. A reader feeds this straight
    # to experiments._shared.fold_reselection, which for each HELD-OUT fold
    # re-picks the winning recipe from the OTHER folds (leave-one-fold-out) and
    # applies the identical held-out tax to both arms. That is the n09
    # selection-honesty principle — "the recipe used at a fold is chosen without
    # that fold" — expressed as pure arithmetic instead of a per-fold re-search.
    # Built from the per-model precomputed searches (the clean-label search
    # space that also SELECTED each model's blueprint); the injected-noise
    # remediation numbers stay in per_model_per_strategy below and keep their
    # own recovery derivation.
    reselection_block = {
        'lower_is_better': lower_is_better,
        'metric_name': metric_name,
        # Which folds contribute no term to either arm. This was hardcoded to
        # [] on the claim that the candidates are arms this file built itself;
        # they are not. build_fold_reselection_input takes them from each
        # search's own candidate list, and it returns [0] whenever that search
        # filtered the list on fold 0 — fold 0's hybrid score is then read off
        # recipes fold 0 helped keep, while the single representations were
        # never filtered. Filled below from what that helper returns per model:
        # the union, so a fold dropped for any model is dropped for all of
        # them and the models stay on one footing. exclude_folds_by_model
        # carries each model's own answer.
        'exclude_folds': [],
        'exclude_folds_by_model': {},
        'fold0_pruned_by_model': {},
        'candidates_by_model': {},
        'baselines_by_model': {},
        # DECISION (2026-09-12), consequence of widening the pool subset: a
        # model whose recipe came from the similarity family carries fold
        # scores in that search's own metric, not the dataset's. The top-level
        # 'lower_is_better' and 'metric_name' above are the DATASET's, so a
        # reader must check this dict before pooling models: any model whose
        # metric here is not the dataset's metric is a separate scale.
        'metric_by_model': {},
        'objective_by_model': {},
    }
    _mixed_metric_models = []
    for _m, _s in search_by_model.items():
        try:
            _blk = build_fold_reselection_input(_s)
        except Exception as _exc:
            print(f"  WARNING: fold_reselection input build failed for {_m} "
                  f"({_exc})")
            continue
        reselection_block['metric_by_model'][_m] = {
            'metric_name': _blk.get('metric_name'),
            'lower_is_better': _blk.get('lower_is_better'),
        }
        _excl = [int(f) for f in (_blk.get('exclude_folds') or [])]
        reselection_block['exclude_folds_by_model'][_m] = _excl
        reselection_block['fold0_pruned_by_model'][_m] = _blk.get('fold0_pruned')
        reselection_block['objective_by_model'][_m] = hybrid_family_by_model.get(
            _m, {}).get('objective')
        if (_blk.get('metric_name') != metric_name
                or bool(_blk.get('lower_is_better')) != bool(lower_is_better)):
            _mixed_metric_models.append(_m)
            print(f"  NOTE: {_m}'s fold_reselection fold scores are in "
                  f"{_blk.get('metric_name')} (lower_is_better="
                  f"{_blk.get('lower_is_better')}), not the dataset's "
                  f"{metric_name} — its recipe came from the "
                  f"{reselection_block['objective_by_model'][_m]!r} search family. "
                  f"Pass that model's own direction to fold_reselection.")
        # Keep only THIS model's slice — a pooled SearchResult's all_results
        # may carry other models' candidates, so stay model-gated.
        if _m in _blk.get('candidates_by_model', {}):
            reselection_block['candidates_by_model'][_m] = \
                _blk['candidates_by_model'][_m]
        if _m in _blk.get('baselines_by_model', {}):
            reselection_block['baselines_by_model'][_m] = \
                _blk['baselines_by_model'][_m]
    reselection_block['models_not_in_dataset_metric'] = _mixed_metric_models
    reselection_block['exclude_folds'] = sorted(
        {f for _folds in reselection_block['exclude_folds_by_model'].values()
         for f in _folds})
    results['fold_reselection'] = reselection_block

    # =========================================================================
    # 2026-09-18, owner's instruction: the top 2 rows as an extra, and the two
    # hybrid deltas. Added beside the existing keys. Nothing above is
    # recomputed, and no representation and no downstream model is dropped from
    # anything above: rank_models_and_reps still fits every downstream model on
    # every representation, and the four cleanups that fit no model still run on
    # every representation.
    # =========================================================================
    # What was absent. results['model_rep_ranking'] ranks representations by a
    # score n07 measured itself on the uncorrupted activity values, and
    # results['fold_reselection'] hands a reader the raw per-fold rows to
    # re-pick from. Neither one states, per downstream model, which two
    # representations and which two hybrid configurations the stored search
    # ranked first, or what the chosen hybrid is worth against them. A reader
    # wanting that had to re-open the search by hand.
    #
    # Read out of search_by_model — each downstream model's OWN search, the one
    # that gave it its blueprint. Two models here can draw from different search
    # families, and their stored scores are then in different units;
    # reselection_block['metric_by_model'] says which model is in which.
    #
    # No width ceiling is applied to this ranking. The
    # max_cleanup_features ceiling above governs which representations may carry
    # the three model-fitting cleanups, not which the search ranked first, so a
    # representation barred there still appears here with its own feature count
    # in 'dimension'.
    #
    # No budget= filter either. get_best_search_for_downstream was called above
    # with this run's budget, but the search it hands back stores rows at
    # several budgets, and filtering to one would report the top 2 of a slice
    # while calling it the top 2. Each selected hybrid entry carries its own
    # 'budget', in features.
    #
    # require_full=False everywhere: a downstream model the search never scored
    # yields a short list with a shortfall_reason instead of raising and killing
    # the run.
    results['top2_base_reps'] = {}
    results['top2_hybrids'] = {}
    results['hybrid_deltas_by_model'] = {}
    # WHICH DOWNSTREAM MODELS THE THREE BLOCKS BELOW COVER, and why a name is
    # missing. Added 2026-09-28. Every reason below used to be printed to the
    # terminal and written nowhere, so a saved result with four models in it
    # read the same as one configured for four.
    #
    # Two lists can disagree. `models_requested` is what this run asked for.
    # `search_by_model` holds the models the pool returned a search for, and a
    # name is absent from it when get_best_search_for_downstream returned no
    # metadata or raised -- 'blueprint_status' above carries that reason. On top
    # of that a search can be missing a model's rows, and then the selector
    # below raises for it. Both kinds land in 'top2_models_dropped'.
    results['top2_models_compared'] = list(search_by_model.keys())
    results['top2_models_dropped'] = {
        _m: 'the pool returned no SearchResult for it: %s'
            % blueprint_status.get(_m, 'unknown')
        for _m in models_requested if _m not in search_by_model}
    for _m, _s in search_by_model.items():
        _ctx = f'n07/{dataset_name}/{_m}'
        try:
            results['top2_base_reps'][_m] = top_base_reps(
                _s, _m, n=2, require_full=False, context=_ctx)
        except Exception as _exc:
            print(f"  WARNING: top-2 base representations failed for {_m} "
                  f"({type(_exc).__name__}: {_exc}); "
                  f"results['top2_base_reps'] carries no entry for it.")
            results['top2_models_dropped'][_m] = (
                'no top-2 base representations: %s: %s'
                % (type(_exc).__name__, _exc))
        _top2_hyb = None
        try:
            _top2_hyb = top_hybrids(_s, _m, n=2, require_full=False,
                                    context=_ctx)
            results['top2_hybrids'][_m] = _top2_hyb
        except Exception as _exc:
            print(f"  WARNING: top-2 hybrids failed for {_m} "
                  f"({type(_exc).__name__}: {_exc}); "
                  f"results['top2_hybrids'] carries no entry for it.")
            results['top2_models_dropped'][_m] = (
                'no top-2 hybrid configurations: %s: %s'
                % (type(_exc).__name__, _exc))
        # The hybrid is passed in by hand. Called with hybrid=None,
        # hybrid_deltas reads _picked['selected'][0]['row'] (_shared.py:7432),
        # and the entry top_hybrids builds (_shared.py:7043) has no 'row' key,
        # so that path raises KeyError before it reaches either delta. That
        # entry does carry 'score_mean' and 'blueprint', which is what
        # hybrid_deltas reads off the hybrid it is given.
        try:
            if _top2_hyb and _top2_hyb['selected']:
                results['hybrid_deltas_by_model'][_m] = hybrid_deltas(
                    _s, _m, hybrid=_top2_hyb['selected'][0], context=_ctx)
        except Exception as _exc:
            print(f"  WARNING: hybrid deltas failed for {_m} "
                  f"({type(_exc).__name__}: {_exc}); "
                  f"results['hybrid_deltas_by_model'] carries no entry for it.")
            results['top2_models_dropped'][_m] = (
                'no hybrid deltas: %s: %s' % (type(_exc).__name__, _exc))
    # delta_vs_source_mean is the hybrid against the mean of the base
    # representations that actually fed it, read off
    # Blueprint.allocation_per_fold so a representation given 0 features is
    # left out. delta_vs_best_rep is the hybrid against the single best base
    # representation for that downstream model. Positive = the hybrid is
    # better, whichever way the metric runs. Both are in the units the search
    # stored, which is not the dataset's metric for a model listed in
    # reselection_block['models_not_in_dataset_metric'].
    for _m, _d in results['hybrid_deltas_by_model'].items():
        # delta_vs_best_rep is None when no base representation carries a score
        # for that downstream model; hybrid_deltas says so in its 'notes'.
        _vb = _d['delta_vs_best_rep']
        print(f"    {_m}: hybrid vs mean of its {_d['n_source_reps']} source "
              f"representations {_d['delta_vs_source_mean']:+.4f}, vs best "
              f"single representation ({_d['best_rep']}) "
              + (f"{_vb:+.4f}" if _vb is not None else "not defined"))

    # --- Hybrid conditions: per-model ---
    # Loop order is (model × fold × seed × strategy) so the per-(fold, seed)
    # noise injection + estimator fit are computed ONCE and reused across
    # all 7 strategies. The previous (model × strategy × fold) order
    # recomputed those 4× per (fold, seed) — once per strategy that needed
    # noise_scores. The audit P1 perf-recompute fix.
    # One pass per (downstream model, hybrid space). Flattened rather than
    # nested so the body below is untouched. Added 2026-09-19: the pool holds
    # two families of stored search and this file built one of them.
    _model_space_passes = [
        (_mn, _mf, _sp, _bp_, _aug_)
        for _mn, _mf in model_factories.items() if _mn in bp_by_model
        for _sp, (_bp_, _aug_) in recipes_for_model.get(_mn, {}).items()
    ]
    for (model_name, model_factory, space_name, model_bp,
            model_aug_cond) in _model_space_passes:
        model_augs = augmentations if model_aug_cond != 'none' else None
        # Under a geometry recipe create_hybrid fits no downstream model, and
        # build_fold_hybrid REFUSES a model and a metric alongside one. Passing
        # them anyway would raise inside _fold_space's except clause and the
        # whole space would come back empty with nothing said. The shared
        # helper returns {} for a geometry recipe and the three unchanged for a
        # predictive one; the model instance is filled in per fold below.
        _build_extra = fold_hybrid_scoring_kwargs(
            model_bp, model=None, metric_fn=metric_fn,
            lower_is_better=lower_is_better)

        # The three model-fitting cleanups run only under the downstream models
        # the ranking above kept. Every other model still gets the four older
        # cleanups and the two baselines, exactly as before.
        model_strategies = [st for st in strategies
                            if st not in REGRESSION_ONLY_CLEANUPS
                            or model_name in top_models]
        per_strategy_scores = {s: [] for s in model_strategies}
        # Same strategies again, measured in the CLEAN-label feature space.
        # That space is the head start: create_hybrid ranks features and splits
        # the budget against the labels it is given, so features chosen before
        # the noise was injected already know which molecules the experiment is
        # about to corrupt. The reported arm above no longer gets it; this one
        # keeps it so the size of the head start is readable rather than
        # assumed. Read it as an oracle, the way n15 labels its clean_bp arm.
        per_strategy_scores_oracle = {s: [] for s in model_strategies}
        # The ceiling in the CLEAN-label feature space: clean training labels,
        # no injection, features chosen before any label was corrupted. One
        # score per fold. Reported on its own so it can be compared against the
        # re-fitted ceiling; no recovery fraction divides by it any more.
        clean_feature_ceiling_scores = []

        for fold_i, (train_idx, val_idx) in enumerate(
                cv_split(groups, n_folds, sample_X, labels, mode=mode)):
            y_train_clean = labels[train_idx]
            # This fold's scaffold groups, for a detector whose inner split has
            # to stay scaffold-disjoint. Without them a molecule's
            # scaffold-mates train the model that predicts it and its residual
            # is small for a reason that has nothing to do with its label.
            _groups_tr = (np.asarray(groups)[train_idx]
                          if groups is not None else None)
            y_val = labels[val_idx]

            train_reps_fold = {k: v[train_idx] for k, v in reps.items()}
            val_reps_fold = {k: v[val_idx] for k, v in reps.items()}
            train_augs = {k: v[train_idx] for k, v in model_augs.items()} if model_augs else None
            val_augs = {k: v[val_idx] for k, v in model_augs.items()} if model_augs else None

            def _fold_space(y_for_features, _tr=train_reps_fold, _va=val_reps_fold,
                            _ta=train_augs, _va_a=val_augs, _tri=train_idx,
                            _mf=model_factory, _bp=model_bp):
                """(X_train, X_val) standardised, features chosen from
                y_for_features. Returns None when the build fails.

                Cached, 2026-09-25 (plan section 3): the clean-label build of
                a fold is the same for every noise condition, and a noisy build
                at level 0 is the clean one, so the unscaled pair is stored
                through noise_cache.cached, keyed like cached_hybrid_for_fold
                (every recipe field, the content of every representation and
                augmentation in the order build_fold_hybrid receives them, the
                train rows, the labels, the groups, the model's class and
                parameters, the metric). A failed build is not stored.
                """
                from experiments.noise import noise_cache as _nc
                _extra = dict(_build_extra)
                if 'model' in _extra:
                    _extra['model'] = _mf()
                _g = (groups[_tri] if groups is not None else None)
                _key = {
                    'blueprint': _nc.blueprint_fingerprint(_bp, list(_tr)),
                    'train_reps': [_nc.array_hash(_tr[r]) for r in _tr],
                    'val_reps': [_nc.array_hash(_va[r]) for r in _tr],
                    'rep_names': (list(_tr) if getattr(_bp, 'rep_count', None)
                                  is not None else None),
                    'augmentations': (None if _ta is None else
                                      [[a, _nc.array_hash(_ta[a]),
                                        _nc.array_hash((_va_a or {}).get(a))]
                                       for a in _ta]),
                    'train_idx': _nc.array_hash(np.asarray(_tri)),
                    'y': _nc.array_hash(y_for_features),
                    'groups': _nc.array_hash(_g),
                    'model_signature': _nc.model_signature(_extra.get('model')),
                    'metric': str(metric_name),
                    'lower_is_better': bool(lower_is_better)}

                def _build():
                    try:
                        X_tr_h, X_va_h, _fi = build_fold_hybrid(
                            _tr, _va, y_for_features, _bp,
                            augmentations_train=_ta, augmentations_val=_va_a,
                            groups=_g, **_extra)
                    except Exception as _exc:
                        _fold_space_errors.setdefault(
                            space_name, f'{type(_exc).__name__}: {_exc}')
                        return None
                    return (X_tr_h, X_va_h)
                _built = _nc.cached(dataset_name, 'n07_fold_space', _key, _build,
                                    store_if=lambda v: v is not None)
                if _built is None:
                    return None
                X_tr_h, X_va_h = _built
                sc = StandardScaler()
                return sc.fit_transform(X_tr_h), sc.transform(X_va_h)

            # Clean-label space. Used for the no_noise ceiling, which is a
            # clean-label condition throughout, and for the oracle arm.
            clean_space = _fold_space(y_train_clean)
            if clean_space is None:
                continue
            X_tr_s, X_va_s = clean_space

            # The clean-label-feature ceiling. Independent of the noise draw,
            # so it runs once per fold, not once per seed. It feeds the oracle
            # block (every arm there is in this space) and its own reported
            # key. It is NOT the ceiling the recovery fractions use.
            no_noise_score = _evaluate_fold(
                X_tr_s, X_va_s, y_train_clean, y_val, 'no_noise',
                noise_fraction, model_factory, metric_fn,
                is_clf=is_clf, metric_name=metric_name,
                cache_dataset=dataset_name, cache_model_name=model_name)
            if no_noise_score is not None:
                clean_feature_ceiling_scores.append(no_noise_score)
                per_strategy_scores_oracle['no_noise'].append(no_noise_score)

            # Noise-dependent strategies: average across NOISE_SEEDS
            for noise_seed in _seeds_here:
                actual_seed = 42 + fold_i * 1000 + noise_seed
                # Compute the noisy y and noise_scores ONCE per (fold, seed);
                # all 6 noise-dependent strategies reuse them.
                _spec = _noise_spec(dataset_name, noise_fraction, noise_level,
                                    noise_seed=noise_seed,
                                    is_clf=is_clf, condition=condition)
                # TODO(similarity-sweep 2026-09-25): plan section 4 -- this
                # call (and the two like it below) passes no scaffold groups,
                # so grouped_shifted and grouped_wider raise here
                # (noise_spec.py:407-412). Not changed in this wiring pass.
                y_tr_noisy, _cmask, _prov = _inject_or_recorded(
                    labels, _spec, train_idx=train_idx, is_clf=is_clf,
                    dataset_name=dataset_name, groups=_noise_groups)
                noise_provenance.setdefault(
                    _spec.spec_id if _spec else AS_THEY_COME, _prov)

                # The feature space every noisy condition is measured in, built
                # from THIS draw's noisy training labels. One build per (fold,
                # seed) — the recipe is fixed, so only the ranking and the
                # budget split are recomputed.
                noisy_space = _fold_space(y_tr_noisy)
                if noisy_space is None:
                    # Everything below this line is skipped, including the
                    # detection grading, so it is counted rather than lost.
                    detection_tally['fold_space_unavailable'] += 1
                    continue
                X_tr_n, X_va_n = noisy_space

                # The ceiling again, inside THIS draw's feature space. Clean
                # training labels, no injection — the only difference from the
                # clean-label-feature ceiling above is which features were
                # chosen. This is what per_model_per_strategy['no_noise']
                # holds, so the clean end and the damaged end of every recovery
                # fraction are measured in the same features. One extra model
                # fit per (fold, noise seed).
                no_noise_same_space = _evaluate_fold(
                    X_tr_n, X_va_n, y_train_clean, y_val, 'no_noise',
                    noise_fraction, model_factory, metric_fn,
                    is_clf=is_clf, metric_name=metric_name,
                    cache_dataset=dataset_name, cache_model_name=model_name)
                if no_noise_same_space is not None:
                    per_strategy_scores['no_noise'].append(no_noise_same_space)

                # ⛔ THE SCAFFOLD GROUPS REACH THE DETECTOR HERE. They did not
                # until 2026-09-20, and the two readings this file takes of the
                # same detector then disagreed. The grading call below hands
                # `activity_cliff_aware` _groups_tr, so its inner split is
                # scaffold-disjoint; these two calls, whose scores are what the
                # cleanups actually flag on, handed none, so a molecule's own
                # scaffold family trained the model that predicted it and its
                # residual was small for a reason that has nothing to do with
                # its label. So the two detectors were graded against each
                # other differing in two ways at once, not one: which estimator
                # they are, and whether the inner split was scaffold-disjoint.
                noise_scores_noisy = _compute_noise_scores(
                    X_tr_n, y_tr_noisy, is_clf=is_clf, groups=_groups_tr,
                    cache_dataset=dataset_name)
                noise_scores_clean_space = _compute_noise_scores(
                    X_tr_s, y_tr_noisy, is_clf=is_clf, groups=_groups_tr,
                    cache_dataset=dataset_name)

                # Did the detector find the molecules that really moved, before
                # any cleanup ran? Owner's decision 2026-09-12. This experiment
                # injects outlier_p05 with p overridden to the corrupted
                # fraction, so a corrupted SUBSET exists and the question has an
                # answer. It costs no model fit: the suspicion scores and the
                # list of corrupted molecules are both already in memory.
                # ⛔ EVERY CANDIDATE DETECTOR IS GRADED, not only the one that
                # drove the cleanups. Added 2026-09-18. The cleanups are the
                # expensive part and they run once, under DETECTOR; grading a
                # second name costs one estimator fit per fold and no model fit
                # at all, and it is the only way this file can say whether the
                # detector it uses is the one that finds the corrupted
                # molecules. Before today there was one hardcoded class and
                # nothing to compare it with.
                detection_tally['reached'] += 1
                for _dname in DETECTORS_GRADED:
                    if _dname == DETECTOR:
                        _alt = noise_scores_noisy
                    else:
                        _alt = _compute_noise_scores(
                            X_tr_n, y_tr_noisy, is_clf=is_clf,
                            detector=_dname, groups=_groups_tr,
                            cache_dataset=dataset_name)
                    if _alt is None:
                        detection_tally['detector_returned_none'] += 1
                        continue
                    _d = detection_scores(_cmask, _alt, (_spec.strategy if _spec else AS_THEY_COME))
                    if _d is None:
                        detection_tally['scores_declined'] += 1
                        continue
                    detection_tally['graded'] += 1
                    detection_by_detector.setdefault(_dname, []).append(_d)

                for _space_name, _ns_arr in (('noisy_space', noise_scores_noisy),
                                             ('clean_space', noise_scores_clean_space)):
                    if _ns_arr is None:
                        continue
                    _det = detection_scores(_cmask, _ns_arr, (_spec.strategy if _spec else AS_THEY_COME))
                    if _det is not None:
                        detection_acc.setdefault(_space_name, []).append(_det)

                for strategy in model_strategies:
                    if strategy == 'no_noise':
                        continue
                    needs_scores = (split_strategy(strategy)[0]
                                    in SCORE_DRIVEN_CLEANUPS)
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
                            dataset_name=dataset_name, noise_level=noise_level,
                            condition=condition,
                            train_groups=(groups[train_idx]
                                          if groups is not None else None),
                            cache_dataset=dataset_name,
                            cache_model_name=model_name)
                        if fold_score is not None:
                            _acc[strategy].append(fold_score)

                # ── D versus D', and the similarity axis. 2026-09-25 ──
                # Plan sections 1-2. One row per (cleanup, similarity
                # setting, k, model) for each of the two feature spaces the
                # legacy block scored: this draw's noisy-label hybrid, and the
                # clean-label ('oracle') hybrid. The row models are RF, SVM
                # and this pass's own model; the own model's D, D' and clean
                # fits are the legacy block's, read back from noise_cache.
                # activity_cliff_aware, graded above in the model's own
                # features, is graded again under every (setting, k).
                # model_input names the pass's model too: the similarity
                # spaces (base_1, base_2, the hybrids) are that model's, so an
                # rf row from xgboost's pass is a different row from rf's own.
                for _mi, _Xtr, _Xva, _ns, _grade in (
                        (f'{model_name}:{space_name}', X_tr_n, X_va_n,
                         noise_scores_noisy, DETECTORS_GRADED),
                        (f'{model_name}:{space_name}@clean_label_features',
                         X_tr_s, X_va_s, noise_scores_clean_space, ())):
                    _n07_sweep_pass(
                        sweep_ctx, model_name=model_name, model_input=_mi,
                        X_tr=_Xtr, X_va=_Xva, y_tr_noisy=y_tr_noisy,
                        y_tr_clean=y_train_clean, y_val=y_val,
                        noise_scores=_ns, strategies=model_strategies,
                        train_idx=train_idx, val_idx=val_idx,
                        groups_tr=_groups_tr, fold=fold_i, draw=noise_seed,
                        recipes=recipes_for_model.get(model_name, {}),
                        base_reps=sim_base_by_model.get(model_name, []),
                        corrupted_mask=_cmask, noise_strategy=(_spec.strategy if _spec else AS_THEY_COME),
                        detection_acc=sim_detection,
                        grade_detectors=_grade)

        # Aggregate per strategy. All strategies now use the iteration's
        # model, so model_used == model_name throughout.
        print(f"\n  {model_name}/{space_name}:")
        for strategy in model_strategies:
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
                # DECISION (2026-09-12): every score carries the space it was
                # measured in and the search family its recipe came from, so a
                # recovery fraction can be checked for a space change between
                # its top and its bottom without leaving the row.
                'feature_space': 'noisy_per_fold',
                'hybrid_family': space_objective.get(space_name),
                # Which hybrid space this row was measured in. One per family
                # of stored search since 2026-09-19.
                'space': space_name,
            }
            (results['per_model_per_strategy_by_space']
                .setdefault(space_name, {})
                .setdefault(model_name, {}))[strategy] = entry
            if space_name == HYBRID_SPACE:
                results['per_model_per_strategy'][model_name][strategy] = entry

            # Same strategy in the clean-label feature space.
            o_scores = per_strategy_scores_oracle[strategy]
            o_mean = float(np.mean(o_scores)) if o_scores else float('nan')
            _oracle_entry = {
                'score': o_mean,
                'n_folds': len(o_scores),
                'model_used': model_name,
                'fold_scores': [float(x) for x in o_scores],
                'feature_space': 'clean_labels',
                'hybrid_family': space_objective.get(space_name),
                'space': space_name,
            }
            (results['per_model_per_strategy_oracle_features_by_space']
                .setdefault(space_name, {})
                .setdefault(model_name, {}))[strategy] = _oracle_entry
            if space_name == HYBRID_SPACE:
                results['per_model_per_strategy_oracle_features'].setdefault(
                    model_name, {})[strategy] = _oracle_entry
            _gapstr = ('' if (np.isnan(o_mean) or np.isnan(score))
                       else f"  oracle_features={o_mean:.4f}")
            print(f"    {strategy:<18} {metric_name}={score:.4f} "
                  f"(n={len(fold_scores)}){_gapstr}")

        # The old ceiling, reported next to the new one rather than instead of
        # it. 'score' here counts one fit per fold; the no_noise row above
        # counts one fit per (fold, noise seed).
        _cf_mean = (float(np.mean(clean_feature_ceiling_scores))
                    if clean_feature_ceiling_scores else float('nan'))
        _ceiling_entry = {
            'score': _cf_mean,
            'n_folds': len(clean_feature_ceiling_scores),
            'model_used': model_name,
            'fold_scores': [float(x) for x in clean_feature_ceiling_scores],
            'feature_space': 'clean_labels',
            'hybrid_family': space_objective.get(space_name),
            'space': space_name,
        }
        (results['per_model_no_noise_clean_features_by_space']
            .setdefault(space_name, {}))[model_name] = _ceiling_entry
        if space_name == HYBRID_SPACE:
            results['per_model_no_noise_clean_features'][model_name] = \
                _ceiling_entry
        print(f"    no_noise in clean-label features "
              f"{metric_name}={_cf_mean:.4f} "
              f"(n={len(clean_feature_ceiling_scores)})")

    # Populate legacy per_strategy from RF (or first available model)
    _legacy_model = 'rf' if 'rf' in bp_by_model else next(iter(bp_by_model))
    # This section reports ONE model, and since 2026-09-17 the three
    # model-fitting cleanups run only under the two models the ranking kept. So
    # when the legacy model is not one of those two, those three rows are empty
    # here. That is not a failure and the numbers are not missing: they are in
    # per_model_per_strategy under the two models that did run them. Said out
    # loud rather than left as three NaNs a reader has to explain.
    # The two detection blocks, now that the fold loop that fills them has run.
    # See the placeholders in the results dict above for what went wrong when
    # these were written at construction time.
    results['detection'] = {_sp: _mean_detection(_rows)
                            for _sp, _rows in detection_acc.items()}
    results['detection_by_detector'] = {
        _dn: _mean_detection(_rows)
        for _dn, _rows in detection_by_detector.items()}

    results['per_strategy_model'] = _legacy_model
    _absent_here = [st for st in strategies
                    if st in REGRESSION_ONLY_CLEANUPS
                    and _legacy_model not in top_models]
    results['per_strategy_absent'] = {
        st: (f'{st} runs only under the downstream models the ranking kept '
             f'({", ".join(top_models) or "none"}), and this section reports '
             f'{_legacy_model}. Its score is in per_model_per_strategy under '
             f'those models.')
        for st in _absent_here}
    if _absent_here:
        print(f"  NOTE: per_strategy reports {_legacy_model}, which the "
              f"ranking did not keep, so {', '.join(_absent_here)} are empty "
              f"there. Their scores are under "
              f"{', '.join(top_models)} in per_model_per_strategy.")
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
    # This section is UNCHANGED by the 2026-09-17 additions: random forest, every
    # representation the dataset loads, the four cleanups n07 already had. The
    # three cleanups added that day are measured per downstream model on that
    # model's top two base representations instead, in the section after this
    # one -- USER DECISION 2026-09-17, 'top 2 base and hybrid reps per
    # downstream evaluator'. Running them here as well would mean one call per
    # representation, fold and noise draw on the representation at its full
    # width, and each of the three fits five models where the older four fit
    # none.
    rep_strategies = [s for s in strategies
                      if s not in REGRESSION_ONLY_CLEANUPS]
    results['per_rep_strategies'] = list(rep_strategies)
    results['per_rep_strategies_note'] = (
        'per_strategy["<representation>/<strategy>"] covers the four cleanups '
        'n07 had before 2026-09-17 and the two baselines, under random forest, '
        'on every representation. oof_smooth, gp_smooth and quantile_clip are '
        'not here: they are in per_model_per_top_rep, which runs all seven '
        'cleanups under each downstream model on that model\'s top two base '
        'representations.')

    for rep_name, X_rep in reps.items():
        per_strategy_scores_rep = {s: [] for s in rep_strategies}

        for fold_i, (train_idx, val_idx) in enumerate(
                cv_split(groups, n_folds, sample_X, labels, mode=mode)):
            y_train_clean = labels[train_idx]
            y_val = labels[val_idx]
            # This fold's scaffold groups, for the detector's inner split.
            _g_tr_rep = (np.asarray(groups)[train_idx]
                         if groups is not None else None)

            # Cached, 2026-09-25: StandardScaler on this fold's training rows,
            # the same arithmetic as the two lines it replaced, stored once
            # per (representation content, fold) across every condition.
            from experiments.noise.noise_cache import cached_scaled
            X_tr_s, X_va_s = cached_scaled(dataset_name, X_rep, train_idx,
                                           val_idx)

            # no_noise: noise-independent, run once per fold
            no_noise_score = _evaluate_fold(
                X_tr_s, X_va_s, y_train_clean, y_val, 'no_noise',
                noise_fraction, rep_model_factory, metric_fn,
                is_clf=is_clf, metric_name=metric_name,
                cache_dataset=dataset_name,
                cache_model_name=results['per_rep_model'])
            if no_noise_score is not None:
                per_strategy_scores_rep['no_noise'].append(no_noise_score)

            for noise_seed in _seeds_here:
                actual_seed = 42 + fold_i * 1000 + noise_seed
                _spec = _noise_spec(dataset_name, noise_fraction, noise_level,
                                    noise_seed=noise_seed,
                                    is_clf=is_clf, condition=condition)
                y_tr_noisy, _, _prov = _inject_or_recorded(
                    labels, _spec, train_idx=train_idx, is_clf=is_clf,
                    dataset_name=dataset_name, groups=_noise_groups)
                noise_provenance.setdefault(
                    _spec.spec_id if _spec else AS_THEY_COME, _prov)
                noise_scores_cache = _compute_noise_scores(
                    X_tr_s, y_tr_noisy, is_clf=is_clf, groups=_g_tr_rep,
                    cache_dataset=dataset_name)

                for strategy in rep_strategies:
                    if strategy == 'no_noise':
                        continue
                    needs_scores = (split_strategy(strategy)[0]
                                    in SCORE_DRIVEN_CLEANUPS)
                    fold_score = _evaluate_fold(
                        X_tr_s, X_va_s, y_train_clean, y_val, strategy,
                        noise_fraction, rep_model_factory, metric_fn,
                        is_clf=is_clf, metric_name=metric_name,
                        noise_seed=actual_seed,
                        precomputed_y_tr=y_tr_noisy,
                        noise_scores=(noise_scores_cache if needs_scores else None),
                        dataset_name=dataset_name, noise_level=noise_level,
                        condition=condition,
                        train_groups=(groups[train_idx]
                                      if groups is not None else None),
                        cache_dataset=dataset_name,
                        cache_model_name=results['per_rep_model'])
                    if fold_score is not None:
                        per_strategy_scores_rep[strategy].append(fold_score)

                # D versus D' and the similarity axis in this representation,
                # 2026-09-25. Same cleanups as the legacy loop above; the
                # similarity spaces are the per-rep model's (random forest).
                _n07_sweep_pass(
                    sweep_ctx, model_name=results['per_rep_model'],
                    model_input=f"{results['per_rep_model']}:{rep_name}",
                    X_tr=X_tr_s, X_va=X_va_s,
                    y_tr_noisy=y_tr_noisy, y_tr_clean=y_train_clean,
                    y_val=y_val, noise_scores=noise_scores_cache,
                    strategies=rep_strategies, train_idx=train_idx,
                    val_idx=val_idx, groups_tr=_g_tr_rep, fold=fold_i,
                    draw=noise_seed,
                    recipes=recipes_for_model.get(results['per_rep_model'],
                                                  {}),
                    base_reps=sim_base_by_model.get(results['per_rep_model'],
                                                    []))

        for strategy in rep_strategies:
            key = f"{rep_name}/{strategy}"
            fold_scores = per_strategy_scores_rep[strategy]
            score = float(np.mean(fold_scores)) if fold_scores else float('nan')
            results['per_strategy'][key] = {
                'score': score,
                'n_folds': len(fold_scores),
                'fold_scores': [float(x) for x in fold_scores],
                # One representation on its own: no feature selection, no
                # recipe, so no search family and no space change between this
                # block's ceiling and its damaged arms (2026-09-12).
                'feature_space': 'single_base_rep',
                'hybrid_family': None,
            }
            print(f"  {key}... {metric_name}={score:.4f}")

    # --- The best representations of the best downstream models --------------
    # USER DECISION 2026-09-17: "top 2 base and hybrid reps per downstream
    # evaluator", with both choices measured in this run rather than read from
    # a stored search. The two models and each one's two representations come
    # from `rank_models_and_reps` above, which fitted every model on every
    # representation using the uncorrupted activity values and ranked on the
    # held-out error. All seven cleanups run here, so the three added today can
    # be read against the four older ones inside one set of features.
    #
    # There is no second, uncorrupted-value set of features here and no oracle
    # arm, because a representation on its own has no feature selection:
    # nothing in this part reads the activity values to decide which features
    # to keep, so there is no second set of features for an oracle to differ
    # in. Every score carries the number of features it was measured on.
    results['per_model_per_top_rep'] = {}

    for model_name in top_models:
        model_factory = model_factories.get(model_name)
        chosen = top_reps_by_model.get(model_name, [])
        if model_factory is None or not chosen:
            continue
        print(f"\n  {model_name} on its best {len(chosen)} representations: "
              + ", ".join(
                  f"{r} ({ranking['n_features'][r]} features, {metric_name}="
                  f"{ranking['scores'][model_name][r]:.4f})" for r in chosen))

        for rep_name in chosen:
            X_rep = reps[rep_name]
            acc = {st: [] for st in strategies}

            for fold_i, (train_idx, val_idx) in enumerate(
                    cv_split(groups, n_folds, sample_X, labels, mode=mode)):
                y_train_clean = labels[train_idx]
                y_val = labels[val_idx]
                g_tr = groups[train_idx] if groups is not None else None

                # Cached, 2026-09-25: the same StandardScaler arithmetic.
                from experiments.noise.noise_cache import cached_scaled
                X_tr_s, X_va_s = cached_scaled(dataset_name, X_rep, train_idx,
                                               val_idx)

                # Uncorrupted activity values, one fit per fold: this does not
                # depend on which molecules the corruption picked.
                nn_score = _evaluate_fold(
                    X_tr_s, X_va_s, y_train_clean, y_val, 'no_noise',
                    noise_fraction, model_factory, metric_fn,
                    is_clf=is_clf, metric_name=metric_name,
                    cache_dataset=dataset_name, cache_model_name=model_name)
                if nn_score is not None:
                    acc['no_noise'].append(nn_score)

                for noise_seed in _seeds_here:
                    actual_seed = 42 + fold_i * 1000 + noise_seed
                    _spec = _noise_spec(dataset_name, noise_fraction,
                                        noise_level, noise_seed=noise_seed,
                                        is_clf=is_clf, condition=condition)
                    y_tr_noisy, _cm, _prov = _inject_or_recorded(
                        labels, _spec, train_idx=train_idx, is_clf=is_clf,
                        dataset_name=dataset_name, groups=_noise_groups)
                    noise_provenance.setdefault(
                        _spec.spec_id if _spec else AS_THEY_COME, _prov)
                    ns_cache = _compute_noise_scores(X_tr_s, y_tr_noisy,
                                                     is_clf=is_clf,
                                                     groups=g_tr,
                                                     cache_dataset=dataset_name)

                    for strategy in strategies:
                        if strategy == 'no_noise':
                            continue
                        needs = (split_strategy(strategy)[0]
                                 in SCORE_DRIVEN_CLEANUPS)
                        fs = _evaluate_fold(
                            X_tr_s, X_va_s, y_train_clean, y_val, strategy,
                            noise_fraction, model_factory, metric_fn,
                            is_clf=is_clf, metric_name=metric_name,
                            noise_seed=actual_seed,
                            precomputed_y_tr=y_tr_noisy,
                            noise_scores=(ns_cache if needs else None),
                            dataset_name=dataset_name,
                            noise_level=noise_level, condition=condition,
                            train_groups=g_tr, cache_dataset=dataset_name,
                            cache_model_name=model_name)
                        if fs is not None:
                            acc[strategy].append(fs)

                    # D versus D' and the similarity axis on this model's
                    # top representation, 2026-09-25. All seven cleanups, as
                    # the legacy loop above.
                    _n07_sweep_pass(
                        sweep_ctx, model_name=model_name,
                        model_input=f'{model_name}:top_rep:{rep_name}',
                        X_tr=X_tr_s,
                        X_va=X_va_s, y_tr_noisy=y_tr_noisy,
                        y_tr_clean=y_train_clean, y_val=y_val,
                        noise_scores=ns_cache, strategies=strategies,
                        train_idx=train_idx, val_idx=val_idx, groups_tr=g_tr,
                        fold=fold_i, draw=noise_seed,
                        recipes=recipes_for_model.get(model_name, {}),
                        base_reps=sim_base_by_model.get(model_name, []))

            for strategy in strategies:
                fold_scores = acc[strategy]
                score = (float(np.mean(fold_scores)) if fold_scores
                         else float('nan'))
                (results['per_model_per_top_rep']
                 .setdefault(model_name, {})
                 .setdefault(rep_name, {})[strategy]) = {
                    'score': score,
                    'n_folds': len(fold_scores),
                    'model_used': model_name,
                    'fold_scores': [float(x) for x in fold_scores],
                    'feature_space': 'single_base_rep',
                    # How many features this number was measured on. These
                    # three cleanups scale with it more than with anything
                    # else, so no score of theirs is reported without it.
                    'n_features': int(ranking['n_features'][rep_name]),
                    'hybrid_family': None,
                }
                print(f"    {rep_name}/{strategy:<18} "
                      f"{metric_name}={score:.4f} (n={len(fold_scores)})")

    # Recovery inside each (model, top representation). The reference is that
    # same model on that same representation, so nothing here divides by
    # another model's damage or by a different set of features.
    per_top_rep_recovery = {}
    for m_name, by_rep in results['per_model_per_top_rep'].items():
        for rep_name, strat_rows in by_rep.items():
            r_clean = strat_rows.get('no_noise', {}).get('score', float('nan'))
            r_noisy = strat_rows.get('noisy_baseline', {}).get('score',
                                                               float('nan'))
            r_gap = (r_clean - r_noisy
                     if not (np.isnan(r_clean) or np.isnan(r_noisy))
                     else float('nan'))
            r_hurt = (r_gap < 0) if lower_is_better else (r_gap > 0)
            r_floor = (max(1e-10, GAP_MIN_FRAC * abs(r_clean))
                       if not np.isnan(r_clean) else 1e-10)
            row = {}
            for strategy in strategies:
                if strategy in ('no_noise', 'noisy_baseline'):
                    continue
                sc_ = strat_rows.get(strategy, {}).get('score', float('nan'))
                if (not np.isnan(sc_) and not np.isnan(r_gap)
                        and abs(r_gap) > r_floor and r_hurt):
                    row[strategy] = float((sc_ - r_noisy) / r_gap * 100)
                else:
                    row[strategy] = float('nan')
            if not np.isnan(r_gap) and r_hurt and abs(r_gap) <= r_floor:
                results.setdefault('recovery_undefined', {})[
                    f'{m_name}/{rep_name}'] = (
                    f'the injected noise moved this model on this '
                    f'representation by {abs(r_gap):.6g} {metric_name}, which '
                    f'is under {GAP_MIN_FRAC:.0%} of its clean score '
                    f'({r_clean:.6g}); a percentage of that is a percentage '
                    f'of a rounding error')
            elif not np.isnan(r_gap) and not r_hurt:
                results.setdefault('recovery_undefined', {})[
                    f'{m_name}/{rep_name}'] = (
                    f'noise did not degrade this model on this representation: '
                    f'clean={r_clean:.6g}, noisy={r_noisy:.6g}, '
                    f'lower_is_better={lower_is_better}')
            per_top_rep_recovery.setdefault(m_name, {})[rep_name] = row
    results['per_model_per_top_rep_recovery_pct'] = per_top_rep_recovery

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
        _floor = (max(1e-10, GAP_MIN_FRAC * abs(ref_clean))
                  if not np.isnan(ref_clean) else 1e-10)
        if (not np.isnan(s) and not np.isnan(ref_gap)
                and abs(ref_gap) > _floor and _hurt):
            recovery[key] = float((s - ref_noisy) / ref_gap * 100)
        else:
            recovery[key] = float('nan')
            if not np.isnan(ref_gap) and not _hurt:
                results.setdefault('recovery_undefined', {})[key] = (
                    f'noise did not degrade this reference: clean={ref_clean:.6g}, '
                    f'noisy={ref_noisy:.6g}, lower_is_better={lower_is_better}')
            elif not np.isnan(ref_gap) and abs(ref_gap) <= _floor:
                results.setdefault('recovery_undefined', {})[key] = (
                    f'the injected noise moved this reference by '
                    f'{abs(ref_gap):.6g} {metric_name}, under '
                    f'{GAP_MIN_FRAC:.0%} of its clean score ({ref_clean:.6g}); '
                    f'a percentage of that is a percentage of a rounding error')

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
        # Clean training labels, fitted in each noise draw's own feature space.
        # This is the top of every recovery fraction below.
        'clean_score': clean,
        # The same clean training labels fitted in the clean-label feature
        # space, for the model this legacy block reports. Comparison only —
        # nothing divides by it.
        'clean_score_clean_features': results.get(
            'per_model_no_noise_clean_features', {}).get(
                _legacy_model, {}).get('score', float('nan')),
        'ceiling_feature_space': 'noisy_per_fold_refit',
        'noisy_score': noisy,
        'noise_gap': float(gap) if not np.isnan(gap) else None,
        'recovery_pct': recovery,
        # Hybrid (RF) and per-rep have different reference gaps; report separately.
        'best_hybrid_strategy': _arg_max_finite(hybrid_recovery_only),
        'best_strategy_per_rep': best_strategy_per_rep,
        # Canonical multi-model view: per-model recovery + per-model winners.
        'per_model_recovery_pct': per_model_recovery,
        'best_strategy_per_model': per_model_best,
        # USER DECISION 2026-09-17, the base half of 'top 2 base and hybrid reps
        # per downstream evaluator'. One recovery percentage per (model,
        # representation, cleanup), each against that model's own uncorrupted
        # and corrupted reference on that same representation. Which two models
        # and which two representations, and every score that chose them, are in
        # results['model_rep_ranking'].
        'per_model_per_top_rep_recovery_pct': per_top_rep_recovery,
        'best_strategy_per_model_per_top_rep': {
            _m: {_r: _arg_max_finite(_row) for _r, _row in _by_rep.items()}
            for _m, _by_rep in per_top_rep_recovery.items()
        },
        'expensive_cleanup_models': list(top_models),
        'expensive_cleanup_reps_by_model': dict(top_reps_by_model),
        'model_rep_ranking': ranking,
        # DECISION (2026-09-12): which search family each model's recipe came
        # from, next to that model's recovery percentage, and the space both
        # halves of that percentage were measured in. The percentage is a
        # noisy-label-feature number over a noisy-label-feature number; the
        # clean-label-feature ceiling is reported but is not its divisor.
        'pool_subset': POOL_SUBSET,
        'hybrid_family_per_model': {
            m: hybrid_family_by_model.get(m, {}).get('objective')
            for m in per_model_recovery
        },
        'recovery_numerator_feature_space': 'noisy_per_fold',
        'recovery_denominator_feature_space': 'noisy_per_fold',
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

    # The similarity axis, added 2026-09-25. Every key above is the MODEL_INPUT
    # setting at today's k (label_smooth and knn_smooth read 9 neighbours; the
    # graded activity_cliff_aware reads 10) and is computed by the code that
    # computed it before. The swept numbers are the D-versus-D' rows in
    # results['similarity']['rows_file'] and, for detection, the block below:
    # {detector: {setting: {k: mean detection reading over (fold, draw)}}}.
    results['similarity'] = similarity_record(
        sim_settings, sink,
        legacy_k={'label_smooth': 9, 'knn_smooth': 9,
                  'activity_cliff_aware': 10},
        base_reps_by_model=sim_base_by_model,
        base_reps_missing=sim_base_missing,
        spaces_unavailable=sweep_ctx['unavailable'],
        detector_driving_cleanups=DETECTOR,
        detector_reads_neighbours=DETECTOR in __import__(
            'experiments.noise.noise_cache',
            fromlist=['NEIGHBOUR_DETECTORS']).NEIGHBOUR_DETECTORS)
    results['detection_by_similarity'] = {
        _d: {_sn: {_k: _mean_detection(_rows) for _k, _rows in _byk.items()}
             for _sn, _byk in _bys.items()}
        for _d, _bys in sim_detection.items()}
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
