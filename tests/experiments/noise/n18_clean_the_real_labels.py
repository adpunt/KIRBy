#!/usr/bin/env python3
"""N18: Does cleaning help on the labels as they come, with nothing injected?

Paper: Noise.

THE QUESTION, AND WHY IT IS NOT n07's. Train a model on the data as it arrived
and write down its held-out score. Run a cleanup over the training labels. Train
a fresh model of the same kind on the cleaned training data and write down its
held-out score. If the second beats the first on fold after fold, the cleanup did
something.

That is the whole experiment. It needs no ground truth about which labels are
wrong, and no molecule measured more than once.

WHAT THIS COVERS THAT NOTHING ELSE DID. n07 runs these same four cleanups, but it
corrupts the labels first and then measures how much of the damage it got back.
Every cleanup number in the noise suite is of that shape. So the suite could say
"this cleanup recovers 40% of what I broke" and could not say "this cleanup helps
on your data". n17 can say the second, but only on a dataset where molecules were
measured three or more times, which as of 2026-09-16 is two datasets. This one
runs anywhere.

WHAT IT CANNOT SAY. Nothing here identifies a wrong label. There is no list of
corrupted molecules, because nobody corrupted any. A win here says the cleanup
helped the model; it does not say the cleanup found errors. It could equally have
removed molecules that were merely hard.

THE HELD-OUT LABELS ARE NOT CLEANED AND ARE NOT CLEAN. They carry whatever error
the assay carries. Every score below is measured against them. That makes a win
harder to get, not easier: the cleanup has to improve agreement with labels that
are themselves imperfect.

THE HEADLINE IS A COUNT OF FOLDS, NOT A MEAN. "Consistently across folds" is the
claim, so the reported number is how many folds the cleaned model beat the
uncleaned one on, out of how many folds ran. The mean difference and the spread
across folds are written beside it, and the mean is never the headline. Averaging
five folds hides a cleanup that wins big on one and loses on four.

WHAT A FLAG RATE IS HERE. In n07 the share of rows flagged is set to the share
that was corrupted, because n07 knows. Here nobody knows, so it is a setting to
sweep rather than a number to look up. FLAG_RATES holds what is swept, and a
cleanup that only wins at one of them has not shown much.

QM9 IS THE CONTROL, NOT AN EXCLUSION. Its labels come from a calculation, so
there is no experimental error in them to remove. A cleanup that "helps" on QM9
is finding something other than measurement error -- most likely it is throwing
away the molecules the model finds hard. Run it there on purpose and read it that
way.

WHERE EVERY CLEANUP IS MEASURED. In two feature spaces: one base representation,
and a hybrid whose recipe comes from the precomputed pool and whose features are
rebuilt per fold. That is n19's two-space treatment, added here on 2026-09-17.
The recipe is fetched PER DOWNSTREAM MODEL since 2026-09-23: rf is scored in the
features rf's recipe chose, svm in svm's. Before that one recipe was fetched for
the first model in the list and every model was scored on its features.
Until then this file read `sorted(reps)[0]` and every number in it came from
whichever representation sorted first, with nothing in the output saying so.

THE SINGLE-REPRESENTATION COMPARISON is each downstream model's top 2 base
representations by that model's own score in the stored search -- spaces
`base_1` and `base_2` -- the same rule as the rest of the study
(_shared.noise_base_reps_by_model, owner's decision 2026-09-25). Which
representations those are is written to `base_reps_by_model`.

THE HYBRID IS BUILT ON THE TRAINING MOLECULES OF A FOLD, AT THEIR UNCLEANED
LABELS. It is built once per fold per downstream model and every cleanup, flag
rate and detector that model is scored under shares it, so the only thing
differing between two cleanups is what the downstream fit was handed. The held-out molecules' labels reach no feature
ranking and no budget split. Whether a recipe searched AFTER the cleaning would
do better is a different question and it is n20's.

The detector is re-run in each space. It reads features, so the molecules it
flags in the hybrid are not the molecules it flags in the base representation --
which is the whole reason a cleanup can move between the two.

A MODEL THE POOL HOLDS NO SEARCH FOR HAS NO SPACE AT ALL. Its single
representations are ranked from that stored search, so without it there is
neither a hybrid nor a base_1/base_2 for that model. `base_reps_missing` and
`hybrid_unavailable_by_model` say which models and why, and `spaces_run` says
which spaces ran for the rest.

NO ORACLE CAVEAT TO CARRY. n11, n12 and n13 record that their recipe was chosen
on clean labels and their experiment then corrupts them. Nothing is corrupted
here and there is no clean label column, so the recipe saw the same labels this
experiment trains on. `recipe_saw_clean_labels` is written False for that reason.

Usage:
    python n18_clean_the_real_labels.py --dataset logd --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import functools
import time
import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_model_factories, DOWNSTREAM_MODELS,
    get_best_search_for_downstream,
    top_base_reps, top_hybrids, select_model_rep_pairs, hybrid_deltas,
    searched_models, resolve_comparison_models,
    noise_base_reps_by_model, base_rank_groups, NOISE_BASE_REP_COUNT,
    pool_recipes_by_family, POOL_SUBSETS,
)
# One per-fold hybrid build, shared with n17 and n19. It lived in n19 until
# 2026-09-17.
from experiments.noise.noise_helpers import (hybrid_for_fold,
                                             pool_recipe_per_downstream_model,
                                             group_models_by_recipe)
# The four cleanups are n07's, called directly rather than copied. n07 injects
# noise unless it is handed a training label vector, and it is handed one here --
# the labels exactly as the loader produced them. So the same code runs, over the
# real labels, with nothing corrupted.
from experiments.noise.n07_remediation import (_evaluate_fold, targets_needed,
                                               correction_targets,
                                               SCORE_FREE_CLEANUPS,
                                               cleaned_training_set)
from kirby.utils.noise_estimation import ESTIMATOR_REGISTRY
# The similarity axis, the D-versus-D' rows, the controls and the per-fold
# checkpoint, added 2026-09-25 (plan sections 1-3). See _sim_axis_n15_n22.
from experiments.noise import noise_cache as nc
from experiments.noise import _sim_axis_n15_n22 as sx
from experiments.noise.similarity_space import SWEPT_KS

# The name the rows file, the checkpoint and the result file are filed under.
EXP_FULL_NAME = 'n18_clean_the_real_labels'

# The noise_condition column of every row this file writes: nothing injected.
ROW_CONDITION = 'as_they_come'

# Downstream models whose fit is seeded, so _evaluate_fold may read it back
# from noise_cache (n07._evaluate_fold, cache_dataset=). dnn is fitted as
# before. Same list as _sim_axis_n15_n22.CACHEABLE_MODELS.
_CACHED_FIT_MODELS = sx.CACHEABLE_MODELS

# The cleanup applied to the training rows. 'no_cleaning' is the score to beat.
#
# ⛔ THIS LIST WAS FOUR AND n07's WAS SEVEN. The three below were added to
# n07.cleaned_training_set on 2026-09-17 and never reached this file, so every
# n18 number before 2026-09-18 came from a menu of four: two that are blunt
# (delete the molecule, lean away from it) and two that move a label toward the
# mean label of its ten nearest neighbours. None of them ESTIMATES what the
# label should have been from a model.
#
#   oof_smooth     move the label toward what a LightGBM trained on the OTHER
#                  training molecules predicts for it, on a scaffold-group-aware
#                  inner split, in proportion to how suspect it is.
#   gp_smooth      the same shape, aiming at a Gaussian process posterior mean.
#   quantile_clip  not a move toward a target at all. Each molecule gets the
#                  10th-to-90th-percentile range of the labels carried by the
#                  training molecules sharing its leaves in a quantile
#                  regression forest. A label inside its range is left exactly
#                  as it is; one outside is pulled back to the edge it crossed.
#                  So it rewrites only the labels no molecule like them carries.
#
# All three are REGRESSION ONLY -- none is defined on a nominal class label --
# and n07 gates them the same way (REGRESSION_ONLY_CLEANUPS). `cleanups_run`
# on the result file says which ones this run actually scored.
ALL_CLEANUPS = ('remove', 'downweight', 'label_smooth', 'knn_smooth',
                'oof_smooth', 'gp_smooth', 'quantile_clip')

# Not defined on a nominal class label, so they are dropped on a
# classification dataset rather than writing NaN rows. Same tuple as n07's.
REGRESSION_ONLY_CLEANUPS = ('oof_smooth', 'gp_smooth', 'quantile_clip')


def cleanups_for(is_clf):
    """The cleanups this task type can actually run."""
    if not is_clf:
        return ALL_CLEANUPS
    return tuple(c for c in ALL_CLEANUPS
                 if c not in REGRESSION_ONLY_CLEANUPS)


# Kept as a name because the module-level accumulators below are built from it
# before `is_clf` is known. run() narrows it to cleanups_for(is_clf).
CLEANUPS = ALL_CLEANUPS

# How many training rows in every hundred are treated as suspect. n07 sets this
# to the share it corrupted; nobody can do that here, so it is swept.
#
# ⛔ THE OLD LOW END WAS STILL WELL ABOVE THE SHARE OF LABELS THAT ARE WRONG.
# The grid was (0.05, 0.10, 0.20). The repository's own ChEMBL probe
# (NOISE_AUDIT_2026_09_10.md 14.11, run 2026-09-18 on 12,010 CHEMBL240 IC50
# records) measured how far a compound's repeat measurements actually disagree:
# across documents the median spread is 0.000 log units and 64 of 9,802
# compounds -- 0.65% -- exceed 1 log unit. herg_ki's own loader then deletes
# every molecule whose spread is over 1 log unit before an experiment sees it.
#
# So `remove` at the old lowest rate was deleting about eight times as many
# molecules as are plausibly mislabelled, and what it deleted beyond that was
# molecules that are merely hard -- which is this file's own caveat at :23-26,
# arriving through the grid rather than through the method. 0.01 replaces 0.10,
# so the sweep still has three values and now spans the measured error share at
# one end and a deliberately aggressive setting at the other.
FLAG_RATES = (0.01, 0.05, 0.20)

# ⛔ TWO OF THE FOUR CLEANUPS DO NOT READ THE FLAG RATE ON A REGRESSION SET.
# Read out of n07.cleaned_training_set on 2026-09-18:
#   remove        thresholds at the (1 - rate) quantile        -- reads it
#   knn_smooth    thresholds at the same quantile              -- reads it
#   downweight    weights_from_scores(scores, strength=1.0),
#                 which takes no threshold at all              -- does NOT
#   label_smooth  on a CLASSIFICATION set it thresholds; on a
#                 REGRESSION set it calls smooth_by_scores with
#                 method='proportional' over every row         -- clf only
#
# Two things went wrong because nothing recorded that. The cheaper one is
# compute: a rate-blind cleanup was handed an identical cleaned column three
# times and a model was fitted on it three times -- 4 of every 12 cleanup fits
# per (space, detector, model) per fold.
#
# The one that matters is the headline. `settings_scored` counts one setting
# per (detector, flag rate, model), so a rate-blind cleanup's single setting
# was counted three times and a rate-reading cleanup's three settings were
# counted three times, and "won a majority in 8 of 12 settings" meant two
# different things depending on which cleanup it was said about. A rate-blind
# cleanup is now scored once, at the first rate, and the other two rates are
# absent rather than duplicated.
def reads_flag_rate(cleanup, is_clf):
    """Does this cleanup's cleaned training set depend on the flag rate?"""
    if cleanup in ('remove', 'knn_smooth'):
        return True
    if cleanup == 'downweight':
        return False
    if cleanup in ('label_smooth', 'oof_smooth', 'gp_smooth'):
        # All three smooth every row in proportion to its score on a regression
        # set, and threshold only on a classification one.
        return bool(is_clf)
    if cleanup == 'quantile_clip':
        # Reads neither axis. A label is inside the range its own leaves carry
        # or it is not; there is no quantile of the suspicion scores and no
        # strength. It is one column, scored once -- see setting_axis.
        return False
    # An unknown cleanup is assumed to read it: scoring it at every rate is
    # wasteful, scoring it at one and calling that the whole sweep is wrong.
    return True


# What the two rate-blind cleanups sweep INSTEAD of the flag rate: how hard
# they lean on the suspicion score, in each one's own units. Three values each,
# so every cleanup is scored at three settings and the counts stay comparable.
#
# ⛔ THE OLD FIXED VALUES WERE THE GENTLEST SETTINGS THE METHODS HAVE.
# `downweight` used strength 1.0 with the 'inverse' shape, which is a row
# weight of 1/(1+1) = 0.5 at the most suspect end against 1 at the cleanest --
# a ratio of 2 to 1. 3.0 is 4 to 1 and 9.0 is 10 to 1. `label_smooth` used
# alpha 0.5, which moves the most suspect label half of the way to its
# neighbours' mean. Nothing in the suite had tried a harder setting: n11 sweeps
# this same knob over ALPHAS = [0.1, 0.3, 0.5, 0.7], every one of which is
# gentler than or equal to the fixed value.
#
# That matters because downweight is the cleanup that WINS on herg_ki -- a
# majority of folds in every setting it was scored in, and all five folds in
# one, at +0.0047 rmse. A method that wins at its weakest setting is a method
# whose strength has not been measured.
STRENGTHS = {
    'downweight': (1.0, 3.0, 9.0),      # 2:1, 4:1 and 10:1 row weights
    'label_smooth': (0.25, 0.5, 1.0),   # a quarter, half, all the way
    # The two model-based corrections have the same alpha as label_smooth --
    # all three call smooth_by_scores with method='proportional' and differ
    # only in what they aim at -- so they get the same three settings, and the
    # comparison between the three targets is like for like.
    'oof_smooth': (0.25, 0.5, 1.0),
    'gp_smooth': (0.25, 0.5, 1.0),
}


def setting_axis(cleanup, is_clf):
    """(axis name, the values) this cleanup is actually swept over.

    `quantile_clip` comes back with ONE value and the axis name 'none': a label
    is inside the range its own leaves carry or it is not, so there is no
    quantile of the suspicion scores to set and no strength to lean by. It is
    scored once and the other two slots are skipped, the same way the rate-blind
    cleanups used to be before they were given the strength axis.
    """
    if reads_flag_rate(cleanup, is_clf):
        return 'flag_rate', tuple(FLAG_RATES)
    vals = STRENGTHS.get(cleanup)
    if not vals:
        return 'none', (None,)
    return 'strength', tuple(vals)

# Which per-molecule suspicion score decides what gets flagged. Swept, because
# "did cleaning help" is a question about the whole cleanup and the detector is
# part of it.
#
# ⛔ activity_cliff_aware WAS THE SECOND ENTRY AND IS NOT ANY MORE, measured
# 2026-09-18 on all three datasets in the space this file runs them in:
#
#   dataset                molecules   spearman   same molecules flagged at
#                                      all rows   0.05    0.10    0.20
#   herg_ki                    1,412     0.9888   78.9%   86.5%   90.8%
#   caco2                      2,158     0.9853   83.3%   82.4%   89.4%
#   flab_thermo_tresanco         672     0.9843   88.2%   83.6%   90.3%
#
# So it and neighbor_residual are near enough one method in two names to be a
# poor use of the second slot -- which is what n11, n13 and n06 concluded on
# 2026-09-12, on the Spearman alone.
#
# What sits there now is a DIFFERENT signal: how badly a model trained on the
# OTHER molecules predicts this label, out of fold on scaffold-disjoint folds.
# Both entries in this tuple used to be a molecule judged against its
# neighbours' LABELS; this one is a molecule judged against a fitted model, so
# the sweep now covers two kinds of evidence rather than one kind twice.
#
# It could not be named here until 2026-09-18: `_suspicion` handed every
# detector `k` and `metric`, which CrossValResidualEstimator takes neither of.
DETECTORS = ('neighbor_residual', 'cross_val')
DETECTOR_K = 10

# ⛔ THE SHUFFLED-SCORE CONTROL, added 2026-09-25. Every cleanup is run a second
# time with the detector's suspicion scores dealt to the training molecules in
# a random order. Same scores, same flag rate, same strength -- so `remove`
# deletes the same NUMBER of molecules, `downweight` hands out the same SET of
# row weights, the smoothers move labels by the same set of amounts -- but to
# molecules the detector did not pick.
#
# Why it is needed: on flab_thermo_tresanco in the hybrid space every cleanup's
# mean gain over its settings was positive (downweight +0.0128 spearman over 30
# settings, remove +0.0149 over 15; loop_n18/D2, 2026-09-21), and on the same
# dataset deleting 45 RANDOM molecules moved spearman by -0.0628 to +0.0688
# (loop_n23/random_drop_probe.json). Without this control a gain cannot be told
# apart from "perturbing the training set of this model helps". The only
# random-drop control before this lived outside the suite
# (loop_n18/random_drop_probe.py) and ran on herg_ki and caco2 only.
#
# Each seed is reported on its own; no mean across seeds is written.
# Score-free cleanups (n07.SCORE_FREE_CLEANUPS, e.g. quantile_clip) read no
# score, so shuffling changes nothing for them and they are not re-run.
SHUFFLE_SEEDS = (0, 1, 2)

# Which feature space each cleanup is measured in.
#
# ⛔ THE THIRD ONE IS THE POINT, AND IT WAS MISSING. This file asks the pool
# with pool_subset='predictive_and_similarity', which reads as though both
# families are being tested. They are not. POOL_SUBSETS names them in
# PREFERENCE order and get_best_search_for_downstream returns the FIRST family
# that has an entry, because the two score in units that cannot be compared --
# one is the dataset's own metric, the other a correlation between each
# held-out molecule's activity and what its five nearest neighbours predict. A
# predictive recipe always exists, so the pool always returned it and the
# similarity search only ever appeared in the metadata. Both n18 runs on
# 2026-09-18 printed `searched under objective predictive`.
#
# That matters more here than anywhere else in the suite. Every detector in
# this file scores a molecule by comparing it to its NEAREST NEIGHBOURS, so the
# whole method rests on the distance those neighbours are found in. The
# similarity-objective search picks features so that exactly that distance
# predicts activity from the five nearest neighbours -- it is the feature space
# built for the question this file asks -- and it was never measured in.
#
# 'hybrid' is the predictive recipe, unchanged, so every number reported before
# today still means what it meant. 'hybrid_similarity' is the geometry one,
# taken from meta['by_objective']['similarity']['best_config'], and is absent
# with a stated reason when that family has no entry.
# base_1 and base_2 are each downstream model's rank-1 and rank-2 single
# representation (see the module docstring).
BASE_SPACES = tuple(f'base_{_k}' for _k in range(1, NOISE_BASE_REP_COUNT + 1))
SPACES = BASE_SPACES + ('hybrid', 'hybrid_similarity')
# The owner's set (2026-10-02): per model and family, the top 2 hybrids from
# the general searches and the top 1 from the distance searches. 'hybrid' and
# 'hybrid_similarity' stay the general rank-1 recipes; the other two slots of
# each family get their own space. They are listed after SPACES so the seeds
# that SPACES.index gave the existing spaces do not move.
SPACE_FOR_FAMILY = {'predictive': 'hybrid', 'similarity': 'hybrid_similarity'}
RULE_SPACES = tuple(f'{_b}__{_slot}' for _b in ('hybrid', 'hybrid_similarity')
                    for _slot in ('general2', 'distance1'))
ALL_SPACES = SPACES + RULE_SPACES

# Which searches the recipe may come from. Same as n02, n06, n07, n11, n12, n13,
# n17 and n19 since 2026-09-12: a search whose features were chosen to predict
# the label, and one whose features were chosen so that the distance they define
# predicts each molecule's activity from its five nearest neighbours. Three of
# the four cleanups here overwrite or reweight a label using its neighbours'
# labels, which is the quantity that second objective optimises.
POOL_SUBSET = 'predictive_and_similarity'


def _suspicion(detector, X_tr_s, y_tr, is_clf, groups=None, dataset=None):
    """One suspicion score per training row, or None if the detector failed.

    ⛔ CHANGED 2026-09-25: THE NEIGHBOUR DETECTORS NOW EXCLUDE SAME-SCAFFOLD
    NEIGHBOURS. Until today `offered` named `groups` only, which
    CrossValResidualEstimator reads and NeighborResidualEstimator does not --
    that one names the same array `exclude_groups` -- so neighbor_residual
    found each molecule's neighbours among its own scaffold family, while n07,
    n24 and n25 hand the same detector exclude_groups (plan section 4: "n18
    and n23 do not exclude same-scaffold molecules from the neighbour
    detectors ... The same detector gives different scores in different
    experiments"). Both names are offered now, the way
    n07._compute_noise_scores offers them (n07_remediation.py:432-433).
    EVERY neighbor_residual NUMBER IN THIS FILE MOVES WITH THIS; the cross_val
    numbers do not.

    Computed through noise_cache.cached_detector_scores (plan section 3), which
    calls n07._compute_noise_scores: the same estimator class with the same
    k=10, metric=None, groups and exclude_groups, fitted with .fit(), whose
    noise_scores_ is estimate()'s return (noise_estimation.py:79-81). A failed
    fit is None, as before. `dataset` names the cache directory only.

    Returning None drops the four cleanups from this fold rather than scoring
    them against a silent fallback.

    ⛔ THE ARGUMENTS ARE CHOSEN BY THE ESTIMATOR'S OWN SIGNATURE. This used to
    hand `k` and `metric` to every detector and catch the TypeError, which
    worked only because both detectors it ran were neighbour-based and took
    them. `CrossValResidualEstimator` takes neither, so both attempts raised
    and the fold dropped all four cleanups with one printed line -- the
    out-of-fold residual, which is the standard detector in this literature,
    could not be named here at all.

    `groups` is the fold's training scaffold groups, and it matters for exactly
    the reason CrossValResidualEstimator's own docstring gives: without it a
    molecule's scaffold-mates train the model that predicts it, so its residual
    is small for a reason that has nothing to do with its label.
    """
    if DETECTOR_K != 10:
        # cached_detector_scores offers n07's k=10; a different DETECTOR_K
        # here would be silently overridden.
        raise ValueError(f'n18 DETECTOR_K={DETECTOR_K} but the shared scorer '
                         f'offers k=10')
    try:
        s = nc.cached_detector_scores(dataset or 'n18', X_tr_s, y_tr,
                                      is_clf=is_clf, detector=detector,
                                      groups=groups)
    except Exception as exc:
        print(f"      {detector} failed: {type(exc).__name__}: {exc}")
        return None
    return None if s is None else np.asarray(s, dtype=float)


def _won(cleaned, base, lower_is_better):
    """Did the cleaned model beat the uncleaned one on this fold?"""
    if not (np.isfinite(cleaned) and np.isfinite(base)):
        return None
    return bool(cleaned < base) if lower_is_better else bool(cleaned > base)


def _drop_blueprint(obj):
    """Strip the pickled Blueprint out of the top-2 blocks before they are saved.

    It does serialise -- convert_numpy routes it through Blueprint.to_dict
    (find_best_hybrid.py:1438) -- but it carries kept_augmentations, which is
    one list of feature indices per fold per donor representation, and every
    recipe field on it (allocation method, importance method, budget, internal
    evaluator, step size, representation count) is already a key of the entry
    it hangs off. Which representations actually fed a configuration is
    recorded by hybrid_deltas under 'source_reps'.
    """
    if isinstance(obj, dict):
        return {k: _drop_blueprint(v) for k, v in obj.items()
                if k != 'blueprint'}
    if isinstance(obj, list):
        return [_drop_blueprint(v) for v in obj]
    return obj


def _print_deltas(deltas_by_model, models, metric_name):
    """The top configuration against its two reference numbers, per model.

    Two numbers and not one, because they answer different questions. The first
    is the configuration against the MEAN of the representations the budget
    actually gave features to, read off Blueprint.allocation_per_fold, so a
    representation given 0 features is not in that mean. The second is it
    against the single best base representation for the same downstream model.
    A configuration can win the first and lose the second.
    """
    if not deltas_by_model:
        return
    print(f"\n  TOP HYBRID vs TWO REFERENCE NUMBERS ({metric_name}, "
          f"+ means the hybrid won):")
    for m in models:
        hd = deltas_by_model.get(m)
        if not hd:
            continue
        dsm, dbr = hd.get('delta_vs_source_mean'), hd.get('delta_vs_best_rep')
        src_mean, n_src = hd.get('source_mean'), hd.get('n_source_reps')
        best_rep, best_sc = hd.get('best_rep'), hd.get('best_rep_score')
        left = (f"vs the mean of the {n_src} representations that fed it "
                f"({src_mean:.4f}): {dsm:+.4f}"
                if dsm is not None and src_mean is not None
                else "vs the mean of its source representations: --")
        right = (f"vs the best single representation {best_rep} "
                 f"({best_sc:.4f}): {dbr:+.4f}"
                 if dbr is not None and best_sc is not None
                 else "vs the best single representation: --")
        print(f"    {m}: {left}; {right}")


def run(data, config, output_dir, dataset_name, **kwargs):
    """N18: cleaning the labels as they come."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N18: DOES CLEANING HELP ON THE LABELS AS THEY COME")
    print("=" * 70)

    labels = np.asarray(data['labels'])
    reps = data['reps']
    groups = data.get('groups')
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'

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
        downstream_models, task_type=task_type, n_samples=len(labels))
    downstream_models = [m for m in downstream_models if m in model_factories]
    if not downstream_models:
        raise RuntimeError(
            f"n18: no downstream model could be built. Asked for "
            f"{list(kwargs.get('models') or DOWNSTREAM_MODELS)}, got "
            f"{sorted(model_factories)}.")

    flag_rates = list(kwargs.get('flag_rates') or FLAG_RATES)
    detectors = list(kwargs.get('detectors') or DETECTORS)
    # An empty tuple switches the control off; None means the default.
    shuffle_seeds = list(SHUFFLE_SEEDS if kwargs.get('shuffle_seeds') is None
                         else kwargs['shuffle_seeds'])
    # {model: [rank-1 rep, rank-2 rep]} given directly (a smoke or a direct
    # caller). None means: rank them from the stored search, below.
    base_reps_override = kwargs.get('base_reps')

    # What this task type can actually run. The three model-based corrections
    # are not defined on a nominal class label, so on a classification dataset
    # they are dropped rather than writing NaN rows -- n07 gates them the same
    # way. Every loop below reads this, not the module tuple.
    CLEANUPS = cleanups_for(is_clf)

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'noise_regime': 'whatever_the_assay_carries',
        'noise_injection': 'none',
        'models': downstream_models,
        'cleanups': list(CLEANUPS),
        'cleanups_not_run': [c for c in ALL_CLEANUPS if c not in CLEANUPS],
        'flag_rates': flag_rates,
        'cleanups_that_read_the_flag_rate': [
            c for c in CLEANUPS if reads_flag_rate(c, is_clf)],
        'swept_axis_per_cleanup': {
            c: {'axis': setting_axis(c, is_clf)[0],
                'values': list(setting_axis(c, is_clf)[1])}
            for c in CLEANUPS},
        'flag_rate_note': (
            'A cleanup absent from cleanups_that_read_the_flag_rate produces '
            'the same cleaned training set at every flag rate, so the rate is '
            'not its axis. It is swept over STRENGTHS instead -- how hard it '
            'leans on the suspicion score -- at the same three slots, and '
            'every row says which axis it sits on under swept_axis and where '
            'under setting. swept_axis_per_cleanup lists both.'),
        'detectors': detectors,
        'shuffle_seeds': shuffle_seeds,
        'shuffled_control_note': (
            'per_space[space]["shuffled_control"] holds every cleanup re-run '
            'with the suspicion scores permuted across the training molecules: '
            'the same number removed, the same set of weights, the same set of '
            'label moves, dealt to molecules the detector did not pick. '
            'real_minus_shuffled is signed so positive means the detector\'s '
            'own ranking beat the random one. Each seed is its own row. '
            'Score-free cleanups are not re-run: shuffling changes nothing '
            'for them.'),
        'claim_scope': (
            'No noise is injected. The uncleaned model trains on the labels as '
            'the loader produced them. Every score is measured against held-out '
            'labels that were NOT cleaned and are NOT clean. A win says the '
            'cleanup helped the model; it does NOT say the cleanup found errors.'),
        # Three facts a reader needs before reading any number below, each
        # stated rather than left to be inferred from the absence of a caveat.
        'recipe_saw_clean_labels': False,
        'recipe_saw_clean_labels_note': (
            'Nothing is injected here and there is no clean label column, so '
            'the recipe saw the same labels this experiment trains on. The '
            'oracle caveat n11, n12 and n13 carry does not apply.'),
        'hybrid_sees_cleaned_labels': False,
        'hybrid_sees_cleaned_labels_note': (
            'The hybrid is built once per fold per downstream model from that '
            'fold\'s training molecules at their UNCLEANED labels, and every '
            'cleanup, flag rate and detector shares it. No cleanup reaches the '
            'feature ranking or the budget split. Whether a recipe searched after the cleaning '
            'would do better is n20\'s question.'),
        'weights_reach_feature_selection': False,
        'weights_reach_feature_selection_note': (
            'sample_weight appears nowhere in src/kirby/hybrid.py or '
            'src/kirby/find_best_hybrid.py, so the downweight cleanup weights '
            'the downstream fit only. The features were chosen without any '
            'weight, in the base space and the hybrid space alike.'),
        'is_computed_label_control': bool(dataset_name == 'qm9'),
        'per_space': {},
    }
    if results['is_computed_label_control']:
        print("  qm9: labels come from a calculation, so there is no measurement\n"
              "  error here to remove. This run is the control — a cleanup that\n"
              "  wins here is removing hard molecules, not wrong ones.")

    splits = list(cv_split(groups, n_folds, next(iter(reps.values())), labels,
                           mode=mode))
    print(f"  {len(labels)} molecules, {len(splits)} folds, "
          f"models {downstream_models}")
    print(f"  flag rates {flag_rates}, detectors {detectors}")

    # ── The hybrid recipe, fetched once PER DOWNSTREAM MODEL ──
    # The pool is keyed on the dataset as the precompute built it, so the lookup
    # is handed the representations, labels and fold grouping exactly as n02,
    # n11, n12, n13, n17 and n19 hand them over. What comes back is a recipe --
    # a representation set, an allocation method, an importance method, a budget
    # and an internal evaluator -- and the features are rebuilt per fold below.
    #
    # ⛔ IT WAS ONE RECIPE, ASKED FOR downstream_models[0], AND EVERY MODEL WAS
    # SCORED ON IT. downstream_models[0] is 'rf' in every run this file has
    # made, so a run over the five models in DOWNSTREAM_MODELS -- rf, xgboost,
    # svm, dnn, lightgbm -- wrote five numbers measured in the features the
    # random forest's recipe chose, and the difference between two of those
    # numbers was not a difference between the two models.
    # Each model now gets the recipe the search picked FOR it. A model the pool
    # holds no search for gets NO hybrid space and keeps every
    # base-representation number; it is never handed another model's recipe.
    # Changed 2026-09-23.
    by_model, no_recipe = pool_recipe_per_downstream_model(
        functools.partial(
            get_best_search_for_downstream, reps, labels, groups,
            budget=None, augmentations=data.get('augmentations'),
            dataset_name=dataset_name, task_type=task_type, data=data,
            pool_subset=POOL_SUBSET),
        downstream_models)

    # ⛔ WHAT WENT WRONG IS RECORDED, NOT JUST THAT SOMETHING DID. Two
    # different things put this file in the base representation alone and
    # they call for opposite responses. One is a dataset the precompute
    # never covered: this file was built to survive it and no action fixes
    # it. The other is a stored search written under an older column list
    # -- result_cache.StaleResultDB -- which one rebuild of that one file
    # fixes. Both were reported here as "no recipe came back from the
    # precomputed pool", so a result file could not tell them apart and a
    # run that should have been re-launched after a rebuild read as a run
    # with nothing to rebuild.
    #
    # Measured on flab_thermo_tresanco 2026-09-20: commit 9c6f7d7 added
    # distance_metric to the search store's identity that morning, every
    # stored search on disk was written without it, and three n18 runs
    # dropped both hybrid spaces while their result files said the pool was
    # empty. The same commit made four callers in find_best_hybrid and
    # _shared re-raise StaleResultDB for exactly this reason; this file
    # still swallows it, because dropping the run would lose every
    # base-representation number -- so it records the cause instead.
    # pool_recipe_per_downstream_model catches it per model and says which of
    # the two happened; the key below carries one sentence per model that has
    # no recipe.
    _raised = [d['lookup_error'] for d in no_recipe.values()
               if d['lookup_error']]
    _lookup_error = _raised[0] if _raised else None
    results['hybrid_unavailable_by_model'] = {
        m: d['why'] for m, d in no_recipe.items()}

    # {model: {space: recipe}}. A model missing from a space is a model that
    # space was never built for, and its rows in that space stay empty.
    recipe_for_space_by_model = {}
    # The owner's set (2026-10-02): every slot of every family becomes its own
    # space -- 'hybrid' / 'hybrid_similarity' (general rank 1, as before), and
    # '<space>__general2' / '<space>__distance1'. pool_recipes_by_family reads
    # the set pool_recipe_per_downstream_model attached to the metadata. Each
    # family's own winner is still taken by name, not by preference, so both
    # families are measured. Every space or slot not built is written to
    # hybrid_spaces_unavailable with its reason.
    hybrid_unavailable = {}
    recipe_info_by_model = {}
    for _m, _got in by_model.items():
        _rs, _un = pool_recipes_by_family(
            _got['meta'], SPACE_FOR_FAMILY,
            eligible_families=POOL_SUBSETS.get(POOL_SUBSET))
        recipe_for_space_by_model[_m] = {r['space']: r['blueprint']
                                         for r in _rs
                                         if r['space'] in ALL_SPACES}
        recipe_info_by_model[_m] = {
            r['space']: {'slot': r.get('slot', 'general1'),
                         'lineage': r.get('lineage', 'general'),
                         'family': r['objective'], 'tier': r['tier'],
                         'aug_condition': r['aug_condition'],
                         'budget_relaxed': r.get('budget_relaxed'),
                         'search_score_mean': r['search_score_mean']}
            for r in _rs}
        for _sp, _why in _un.items():
            hybrid_unavailable.setdefault(_sp, {})[_m] = _why
    results['hybrid_spaces_unavailable'] = hybrid_unavailable
    results['hybrid_slot_by_model'] = convert_numpy(recipe_info_by_model)

    # ── The single-representation comparison, per downstream model ──
    if base_reps_override is not None:
        base_by_model = {m: list(base_reps_override[m])
                         for m in downstream_models if base_reps_override.get(m)}
        base_missing = {m: 'not given by the caller' for m in downstream_models
                        if m not in base_by_model}
    else:
        base_by_model, base_missing = noise_base_reps_by_model(
            {m: d['search'] for m, d in by_model.items()}, downstream_models,
            reps=reps, dataset_name=dataset_name)
    results['base_reps_by_model'] = base_by_model
    results['base_reps_missing'] = base_missing
    base_spaces_run = tuple(
        f'base_{k}' for k in range(1, NOISE_BASE_REP_COUNT + 1)
        if base_rank_groups(base_by_model, k, downstream_models))

    bp = None
    meta = None
    if not by_model:
        # Every cleanup still runs, in the base representation alone. Dropping
        # the run would lose every base-space number because one of two spaces
        # is missing.
        spaces = base_spaces_run
        results['hybrid_available'] = False
        results['hybrid_lookup_error'] = _lookup_error
        results['hybrid_unavailable_reason'] = (
            (f'the recipe lookup RAISED -- {_lookup_error} -- so neither '
             f'hybrid space could be built. This is NOT a dataset the '
             f'precompute never covered: something is wrong with the stored '
             f'search and it can be fixed. Every cleanup ran in the single '
             f'representations only.')
            if _lookup_error else
            (f'no recipe came back from the precomputed pool for subset '
             f'{POOL_SUBSET!r} for any downstream model, so the hybrid space '
             f'could not be built. Every cleanup ran in the single '
             f'representations only.'))
        print(f"  no hybrid recipe from pool subset {POOL_SUBSET} for any "
              f"downstream model; single representations only")
    else:
        # A space runs when at least one downstream model has a recipe for it,
        # and inside it only the models that do.
        spaces = tuple(s for s in ALL_SPACES if s in base_spaces_run or (
            s not in BASE_SPACES and any(
                s in _r for _r in recipe_for_space_by_model.values())))
        results['hybrid_available'] = True
        results['hybrid_lookup_error'] = _lookup_error
        results['hybrid_models_with_a_recipe'] = [
            m for m in downstream_models if m in by_model]
        results['hybrid_models_without_one'] = [
            m for m in downstream_models if m not in by_model]
        # The keys that name ONE recipe describe this model's. Every model's is
        # in hybrid_recipe_by_model, and every cleanup score was measured in
        # the features of the model's own recipe.
        _first = results['hybrid_models_with_a_recipe'][0]
        meta = by_model[_first]['meta']
        bp = by_model[_first]['blueprint']
        results['hybrid_recipe_keys_name_one_model'] = _first
        results['hybrid_recipe_keys_note'] = (
            'hybrid_family, hybrid_family_choice, hybrid_pool_subset, '
            'hybrid_tier, hybrid_aug_condition, hybrid_budget, hybrid_reps, '
            'hybrid_pool_by_objective and the hybrid_similarity_* keys all '
            f'describe the recipe the search picked for {_first}. Every other '
            'downstream model was scored in the features of ITS OWN recipe, '
            'and all of them are in hybrid_recipe_by_model.')
        results['hybrid_family'] = meta['objective']
        results['hybrid_family_choice'] = meta['objective_choice']
        results['hybrid_pool_subset'] = meta['pool_subset']
        results['hybrid_tier'] = meta['tier']
        results['hybrid_aug_condition'] = meta['aug_condition']
        results['hybrid_budget'] = getattr(bp, 'budget', None)
        results['hybrid_reps'] = sorted(getattr(bp, 'reps', []) or [])
        results['hybrid_pool_by_objective'] = convert_numpy({
            o: {'tier': d['tier'], 'aug_condition': d['aug_condition'],
                'score_mean': d['score_mean'], 'metric_name': d['metric_name'],
                'lower_is_better': d['lower_is_better']}
            for o, d in (meta.get('by_objective') or {}).items()
        })
        # One entry per (downstream model, space): the recipe that model's
        # features were rebuilt from in that space.
        results['hybrid_recipe_by_model'] = convert_numpy({
            m: {sp: {'reps': sorted(getattr(b, 'reps', []) or []),
                     'budget': getattr(b, 'budget', None),
                     'allocation_method': getattr(b, 'allocation_method', None),
                     'importance_method': getattr(b, 'importance_method', None),
                     'internal_evaluator': getattr(b, 'internal_evaluator',
                                                   None),
                     'searched_under_objective': getattr(b, 'objective', None)}
                for sp, b in _r.items()}
            for m, _r in recipe_for_space_by_model.items()})
        for m in results['hybrid_models_with_a_recipe']:
            _r = recipe_for_space_by_model[m]
            _bp_m = _r.get('hybrid', by_model[m]['blueprint'])
            print(f"  {m}: recipe of "
                  f"{len(sorted(getattr(_bp_m, 'reps', []) or []))} "
                  f"representations, budget {getattr(_bp_m, 'budget', None)} "
                  f"features, searched under objective "
                  f"{by_model[m]['meta']['objective']} (pool subset "
                  f"{by_model[m]['meta']['pool_subset']}); spaces "
                  f"{sorted(_r)}")
        for m in results['hybrid_models_without_one']:
            print(f"  {m}: no recipe — {no_recipe[m]['why']}")

        # The similarity keys describe the first model that HAS a similarity
        # recipe, which need not be the model the keys above describe.
        _sim_model = next(
            (m for m in downstream_models
             if 'hybrid_similarity' in recipe_for_space_by_model.get(m, {})),
            None)
        _sim = ((by_model[_sim_model]['meta'].get('by_objective') or {})
                .get('similarity') or {}) if _sim_model else {}
        _bp_sim = (_sim.get('best_config') or {}).get('blueprint')
        if 'hybrid_similarity' not in spaces:
            results['hybrid_similarity_available'] = False
            _any_sim_family = any(
                'similarity' in (d['meta'].get('by_objective') or {})
                for d in by_model.values())
            results['hybrid_similarity_unavailable_reason'] = (
                'the pool held no search under objective=similarity for this '
                'dataset, so the space whose features were chosen to make the '
                'neighbour distance predictive could not be built for any '
                'downstream model. Every cleanup still ran in the other '
                'spaces.'
                if not _any_sim_family else
                'the similarity family was in the pool but no downstream '
                "model's winning configuration under it carried a recipe.")
            print(f"  no similarity-objective recipe; "
                  f"{results['hybrid_similarity_unavailable_reason']}")
        else:
            results['hybrid_similarity_available'] = True
            results['hybrid_similarity_keys_name_one_model'] = _sim_model
            results['hybrid_similarity_models'] = [
                m for m in downstream_models
                if 'hybrid_similarity' in recipe_for_space_by_model.get(m, {})]
            results['hybrid_similarity_budget'] = getattr(_bp_sim, 'budget',
                                                          None)
            results['hybrid_similarity_reps'] = sorted(
                getattr(_bp_sim, 'reps', []) or [])
            results['hybrid_similarity_score_mean'] = _sim.get('score_mean')
            results['hybrid_similarity_metric_name'] = _sim.get('metric_name')
            results['hybrid_similarity_note'] = (
                "Its features were chosen so that the distance they define "
                "predicts each molecule's activity from its five nearest "
                "neighbours. Its score_mean is that correlation and is NOT in "
                "the dataset's own metric, so it cannot be compared with the "
                "predictive hybrid's. Every cleanup score measured IN it is in "
                "the dataset's metric and is comparable with everything else "
                "in this file.")
            print(f"  similarity recipe ({_sim_model}): "
                  f"{len(results['hybrid_similarity_reps'])} representations, "
                  f"budget {results['hybrid_similarity_budget']}, scored "
                  f"{_sim.get('score_mean')} under "
                  f"{_sim.get('metric_name')}")
    results['spaces_run'] = list(spaces)

    # ── What each downstream model would have picked for itself ──
    # Added 2026-09-18, ADDITIVE. Nothing above or below moves: every
    # fold-win count in
    # 'per_space' is still measured in the spaces 'spaces_run' names. These 4
    # keys sit beside them.
    #
    # What was absent without them: this file scores 5 downstream models on 1
    # base representation, and the result file recorded neither which
    # representations nor which configurations the other 4 downstream models
    # rank highest. A reader could not tell whether a model lost because the
    # cleanup did nothing or because it was scored on another model's recipe.
    #
    # 0 new model fits. All four read rows the search already stored:
    # search.baselines for the representations, search.all_results for the
    # configurations. Each model is read out of the SearchResult the pool
    # returned for that model, since 2026-09-23; before that one search was
    # fetched for downstream_models[0] and every model read out of it.
    results['top2_base_reps'] = {}
    results['top2_hybrids'] = {}
    results['top2_pairs_by_model'] = {}
    results['hybrid_deltas_by_model'] = {}
    _searches = {m: d['search'] for m, d in by_model.items()
                 if d['search'] is not None}
    if not _searches:
        results['top2_unavailable_reason'] = (
            'the pool returned no SearchResult for any downstream model, so '
            'there are no stored rows to rank. Every cleanup number in this '
            'file stands.')
    else:
        # WHICH DOWNSTREAM MODELS THIS TABLE COVERS. Two lists decide it and
        # they can disagree. `downstream_models` is the list this run fits,
        # taken from config['models'] or from DOWNSTREAM_MODELS when the runner
        # named none. The other list is the models a stored search scored,
        # which searched_models reads off that search. A model in the first and
        # not in the second has no rows of its own to rank. Both kinds of
        # missing model are now written to the result file: the model whose
        # pool lookup returned no search at all, and the model whose own search
        # scored other models and not it. Neither is read out of another
        # model's search -- resolve_comparison_models never substitutes one
        # model for another, and each model below is read out of its own.
        _requested = list(downstream_models)
        _compare, _model_dropped = [], {}
        for _m in _requested:
            _s_m = _searches.get(_m)
            if _s_m is None:
                _model_dropped[_m] = (
                    'the pool returned no stored search for this downstream '
                    'model, so there are no rows of its own to rank')
                continue
            _keep, _why = resolve_comparison_models(_s_m, [_m])
            if not _keep:
                _model_dropped[_m] = _why[0][1]
                continue
            _compare.append(_m)
        results['top2_models_requested'] = _requested
        results['top2_models_compared'] = list(_compare)
        results['top2_models_dropped'] = _model_dropped
        results['top2_models_scored_by_each_search'] = {
            _m: searched_models(_s) for _m, _s in _searches.items()}
        for _m, _why in _model_dropped.items():
            print(f"  top-2 rows absent for {_m}: {_why}")
        for _m in _compare:
            search = _searches[_m]
            # require_full=False: a downstream model this search never scored
            # comes back a short list carrying its own shortfall_reason,
            # instead of raising and losing every cleanup number below it.
            try:
                results['top2_base_reps'][_m] = top_base_reps(
                    search, _m, 2, require_full=False)
            except Exception as _e:
                print(f"  top-2 base representations for {_m} unavailable: "
                      f"{type(_e).__name__}: {_e}")
            _th = None
            try:
                _th = top_hybrids(search, _m, 2, require_full=False)
                results['top2_hybrids'][_m] = _drop_blueprint(_th)
            except Exception as _e:
                print(f"  top-2 hybrid configurations for {_m} unavailable: "
                      f"{type(_e).__name__}: {_e}")
            # The rank-1 configuration is handed over explicitly. Called with
            # hybrid=None, hybrid_deltas picks its own and reads it as
            # _picked['selected'][0]['row'] (_shared.py:7432); a selected entry
            # from top_hybrids carries no 'row' key, so that path raises
            # KeyError('row'). Read in _shared.py this session. The entry does
            # carry 'score_mean' and 'blueprint', which is all hybrid_deltas
            # reads off the configuration it is given.
            if _th and _th['selected']:
                try:
                    results['hybrid_deltas_by_model'][_m] = hybrid_deltas(
                        search, _m, hybrid=_th['selected'][0])
                except Exception as _e:
                    print(f"  hybrid deltas for {_m} unavailable: "
                          f"{type(_e).__name__}: {_e}")
        # The same grid in one block: per downstream model, 2 base
        # representations and 2 hybrid configurations, plus 'n_distinct_units'
        # -- how many separate things would have to be BUILT once one
        # representation picked by three downstream models is counted once.
        # It takes ONE SearchResult and reads every model out of it, so it is
        # given the one the pool returned for the model named in
        # 'top2_pairs_search_of_model' and that is written down.
        _grid_model = next(m for m in downstream_models if m in _searches)
        results['top2_pairs_search_of_model'] = _grid_model
        try:
            results['top2_pairs_by_model'] = _drop_blueprint(
                select_model_rep_pairs(_searches[_grid_model],
                                       models=list(downstream_models),
                                       n_base=2, n_hybrids=2,
                                       require_full=False))
        except Exception as _e:
            print(f"  top-2 model-and-representation grid unavailable: "
                  f"{type(_e).__name__}: {_e}")
        _print_deltas(results['hybrid_deltas_by_model'], downstream_models,
                      metric_name)

    # {space: {detector: {flag_rate: {model: {cleanup: [per-fold score]}}}}}
    acc = {s: {d: {r: {m: {c: [] for c in ('no_cleaning',) + CLEANUPS}
                       for m in downstream_models} for r in flag_rates}
               for d in detectors} for s in spaces}
    # Which FOLDS produced a score, as a set of fold numbers per (space,
    # detector, flag rate). A count until 2026-09-23, when a hybrid space
    # became one feature matrix per downstream model: the loop below then
    # reaches the same (space, detector, rate) once per model, and a count
    # would have said five folds ran where one did.
    folds_ran = {s: {d: {r: set() for r in flag_rates} for d in detectors}
                 for s in spaces}
    # The shuffled-score control, per seed. Kept out of `acc` so that every
    # count in per_detector and in the summary still means the real detector.
    ctrl_cleanups = tuple(c for c in CLEANUPS if c not in SCORE_FREE_CLEANUPS)
    acc_ctrl = {s: {d: {k: {r: {m: {c: [] for c in ctrl_cleanups}
                                for m in downstream_models}
                            for r in flag_rates}
                        for k in shuffle_seeds}
                    for d in detectors} for s in spaces}
    hybrid_failures = []

    # ── The main measurement, the similarity axis and the checkpoint ──
    # Added 2026-09-25 (plan sections 1-3).
    #
    # ROWS. Every cleanup's fit on D' (the cleaned training fold) beside the
    # uncleaned fit on D, one d_vs_dprime row per (fold, feature matrix,
    # detector, setting, model), appended as each fold finishes. The rows of
    # the code path above are built from the numbers it already computed --
    # no refit -- and carry similarity_setting 'model_input|none|process|
    # k=default' when a neighbour list reached them (the neighbour detector,
    # or label_smooth / knn_smooth), 'none' when nothing did.
    #
    # THE SIMILARITY SWEEP. For each downstream model in each feature matrix,
    # the neighbours neighbor_residual and label_smooth / knn_smooth read are
    # found in each setting of sx.sweep_settings() -- that model's top-2
    # single representations, each plus augmented features, its two hybrids,
    # ECFP4 Tanimoto, forest proximity, x four weightings x two distances --
    # at every k in SWEPT_KS. The model is still fitted on the feature matrix
    # it was fitted on before; only the neighbours move. The detector's lists
    # exclude same-scaffold neighbours (exclude_groups=g_tr, as _suspicion now
    # does); the smoothers' do not, as n07's own smoothers do not
    # (n07:829-855). The sweep adds rows only: nothing in per_space or summary
    # reads it. KIRBY_SIMILARITY_SETTINGS=model_input switches it off, and the
    # file is then today's run.
    #
    # CHECKPOINT. The accumulators are pickled after every fold, so a killed
    # run resumes at the next fold (sx.FoldCheckpoint); the rows resume through
    # d_vs_dprime.done_keys.
    metric = (metric_name, metric_fn, lower_is_better)
    rows = sx.Rows(output_dir, EXP_FULL_NAME, dataset_name)
    sweep = sx.sweep_settings()
    results['rows'] = rows.summary()
    results['similarity_settings_swept'] = [s.name for s in sweep]
    results['similarity_ks_swept'] = list(SWEPT_KS) if sweep else []
    # Which neighbours every number OUTSIDE the rows and the similarity axis was
    # computed with, so each result record names its similarity setting.
    results['similarity_setting_of_the_default_path'] = sx.TODAY
    results['similarity_note'] = (
        "The similarity axis writes rows only (rows_path). Its settings are "
        "the features neighbours are found in; every fit stays on the feature "
        "matrix of the space it is filed under. An empty "
        "similarity_settings_swept means KIRBY_SIMILARITY_SETTINGS=model_input: "
        "today's run.")
    sim_unavailable = {}
    ck = sx.FoldCheckpoint(
        output_dir, EXP_FULL_NAME, dataset_name,
        sx.run_signature(data, config, {
            'models': downstream_models, 'flag_rates': flag_rates,
            'detectors': detectors, 'shuffle_seeds': shuffle_seeds,
            'base_reps': base_reps_override, 'spaces': list(spaces)}))
    if ck.state is not None:
        acc = ck.state['acc']
        folds_ran = ck.state['folds_ran']
        acc_ctrl = ck.state['acc_ctrl']
        hybrid_failures = ck.state['hybrid_failures']
        sim_unavailable = ck.state.get('sim_unavailable', {})

    def _cache_name(mname):
        # The name _evaluate_fold files a cached fit under, or None to fit it
        # directly (dnn: its fit is not reproducible, see noise_cache).
        return mname if mname in _CACHED_FIT_MODELS else None

    def _setting_label(cleanup, slot, rate):
        # The axis a cleanup is swept on and its value at this slot, as
        # setting_axis defines them.
        _axis, _vals = setting_axis(cleanup, is_clf)
        return f"{_axis}={rate if _axis == 'flag_rate' else _vals[slot]}"

    def _today_ident(method, cleanup, detector, slot, rate, model_input, fold):
        # A row of the code path above. A neighbour list reached it when the
        # detector reads one or the cleanup does; then its neighbours were
        # found in the model-input matrix, each at today's count.
        reads = (detector in nc.NEIGHBOUR_DETECTORS
                 or cleanup in sx.SMOOTHERS_READING_NEIGHBOURS)
        return sx.identity(
            dataset=dataset_name, experiment=EXP_FULL_NAME, method=method,
            method_setting=(f'detector={detector}|'
                            f'{_setting_label(cleanup, slot, rate)}'),
            similarity_setting=sx.TODAY if reads else sx.NO_SIMILARITY,
            model_input=model_input, noise_condition=ROW_CONDITION,
            fold=int(fold))

    def _sweep_cell(X_tr_s, X_va_s, cell_models, mi, g_tr, y_train, y_val,
                    train_idx, tr_reps, va_reps, smiles_tr, scores_by_detector,
                    targets, base_score, fold_i):
        """Every cleanup that a similarity setting reaches, per setting and k,
        on one feature matrix. A setting reaches a (detector, cleanup) pair
        when the detector reads neighbours (neighbor_residual) or the cleanup
        does (label_smooth, knn_smooth); a pair it reaches neither way is the
        'none' row the code path above already wrote. The model is fitted on
        this matrix as before; the D score is the one base_score holds."""
        for mname in cell_models:
            fac = {mname: model_factories[mname]}
            sp_tr, _sp_va, unav = sx.build_spaces(
                dataset_name, sweep, X_model_tr=X_tr_s, train_reps=tr_reps,
                val_reps=va_reps, train_idx=train_idx,
                base_names=base_by_model.get(mname),
                recipes=recipe_for_space_by_model.get(mname),
                y_train=y_train, groups_train=g_tr, model_name=mname,
                model_factory=model_factories[mname], metric=metric,
                task_type=task_type,
                augmentations_train=_aug_rows(train_idx),
                augmentations_val=_aug_rows(val_idx_of[fold_i]))
            for setting in sweep:
                if setting.space in unav:
                    sim_unavailable.setdefault(setting.name, {})[mname] = \
                        unav[setting.space]
                    continue
                nb_plain, why = sx.neighbour_lists(
                    dataset_name, setting, sp_tr, y_train,
                    smiles_train=smiles_tr, is_clf=is_clf)
                if nb_plain is None:
                    sim_unavailable.setdefault(setting.name, {})[mname] = why
                    continue
                nb_excl = nb_plain
                if g_tr is not None:
                    nb_excl, _why_e = sx.neighbour_lists(
                        dataset_name, setting, sp_tr, y_train,
                        exclude_groups=g_tr, smiles_train=smiles_tr,
                        is_clf=is_clf)
                for k in sx.ks_held(nb_plain):
                    sim = sx.sim_name(setting, k)
                    for detector in detectors:
                        det_reads = detector in nc.NEIGHBOUR_DETECTORS
                        if det_reads:
                            if nb_excl is None or k not in sx.ks_held(nb_excl):
                                continue
                            try:
                                scores_k = nc.cached_detector_scores(
                                    dataset_name, X_tr_s, y_train,
                                    is_clf=is_clf, detector=detector,
                                    groups=g_tr, similarity_neighbours=nb_excl,
                                    similarity_k=k)
                            except Exception as exc:
                                print(f"      {detector} under {sim} failed: "
                                      f"{type(exc).__name__}: {exc}")
                                continue
                        else:
                            scores_k = scores_by_detector.get(detector)
                        if scores_k is None:
                            continue
                        for cleanup in CLEANUPS:
                            reads = sx.strategy_reads_setting(cleanup, setting)
                            if not (det_reads or reads):
                                continue
                            _axis, _vals = setting_axis(cleanup, is_clf)
                            for slot, rate in enumerate(flag_rates):
                                if slot >= len(_vals):
                                    continue
                                _strength = (None if _axis == 'flag_rate'
                                             else _vals[slot])
                                ident = sx.identity(
                                    dataset=dataset_name,
                                    experiment=EXP_FULL_NAME, method=cleanup,
                                    method_setting=(
                                        f'detector={detector}|'
                                        f'{_setting_label(cleanup, slot, rate)}'),
                                    similarity_setting=sim, model_input=mi,
                                    noise_condition=ROW_CONDITION,
                                    fold=int(fold_i))
                                if rows.is_done(ident, [mname]) and not \
                                        sx.controls_enabled():
                                    continue

                                def _clean(sc_, rate_, _c=cleanup,
                                           _s=_strength, _r=reads, _k=k):
                                    return cleaned_training_set(
                                        _c, X_tr_s, y_train, sc_, rate_,
                                        is_clf=is_clf, groups=g_tr,
                                        strength=_s, targets=targets,
                                        dataset_name=dataset_name,
                                        similarity_neighbours=(
                                            nb_plain if _r else None),
                                        similarity_k=_k if _r else None)
                                try:
                                    cleaned = _clean(scores_k, rate)
                                except Exception as exc:
                                    print(f"      {cleanup} under {sim} "
                                          f"failed: {type(exc).__name__}: "
                                          f"{exc}")
                                    cleaned = None
                                _kw = dict(X_tr=X_tr_s, y_D=y_train,
                                           X_va=X_va_s, y_va=y_val,
                                           metric=metric, factories=fac,
                                           score_D={mname: base_score[mname]})
                                sx.dprime_rows(rows, dataset_name, ident,
                                               cleaned, **_kw)
                                sx.control_rows(
                                    rows, dataset_name, ident, cleaned,
                                    seed=sx.seed_of(fold_i, mi, mname, cleanup,
                                                    detector, sim, slot),
                                    is_clf=is_clf, rerun=_clean, **_kw)

    augmentations_all = data.get('augmentations') or None

    def _aug_rows(idx):
        # The augmentation donors sliced to these rows, for the four augmented
        # similarity spaces. None when the loader carries none.
        if not augmentations_all:
            return None
        return {a: np.asarray(m[idx], dtype=np.float64)
                for a, m in augmentations_all.items()}

    val_idx_of = {i: va for i, (_tr, va) in enumerate(splits)}

    t0 = time.time()
    for fold_i, (train_idx, val_idx) in enumerate(splits):
        if ck.is_done(fold_i):
            print(f"    fold {fold_i}: already in the checkpoint, skipped")
            continue
        y_train = labels[train_idx]
        y_val = labels[val_idx]
        smiles_tr = sx.smiles_for(data, train_idx)
        fold_rows = []

        # ── The feature spaces for this fold ──
        # Each single-representation space holds one matrix per distinct
        # representation at that rank, scaled on this fold's training
        # molecules and shared by the models that rank it there.
        #
        # ⛔ A HYBRID SPACE IS ONE MATRIX PER DOWNSTREAM MODEL. Each model's
        # features are rebuilt from the recipe the search picked for THAT
        # model, and the accept/reject test inside the build is scored with
        # that model. Until 2026-09-23 a hybrid space was one matrix, built
        # from downstream_models[0]'s recipe with downstream_models[0]'s
        # estimator, and every model was scored on it.
        #
        # {space: [(training matrix, held-out matrix, the models scored on
        # it)]}. Every matrix is built from this fold's training molecules at
        # their UNCLEANED labels, and every cleanup shares it.
        fold_spaces = {}
        # This fold's training scaffold groups. Bound here and not inside the
        # hybrid branch below, because the detector reads them too and a run
        # with no hybrid recipe still has to reach them.
        g_tr = (np.asarray(groups)[train_idx]
                if groups is not None else None)
        for _k, _bsp in enumerate(BASE_SPACES, start=1):
            if _bsp not in spaces:
                continue
            _cells = []
            for _rn, _ms in base_rank_groups(base_by_model, _k,
                                             downstream_models).items():
                scaler = StandardScaler()
                _cells.append((
                    scaler.fit_transform(np.asarray(reps[_rn][train_idx],
                                                    dtype=np.float64)),
                    scaler.transform(np.asarray(reps[_rn][val_idx],
                                                dtype=np.float64)),
                    list(_ms), _rn))
            fold_spaces[_bsp] = _cells
        _tr_reps = {k: np.asarray(v[train_idx], dtype=np.float64)
                    for k, v in reps.items()}
        _va_reps = {k: np.asarray(v[val_idx], dtype=np.float64)
                    for k, v in reps.items()}
        # hybrid_for_fold refuses the model and the metric under a geometry
        # recipe on its own -- it reads the objective off the blueprint -- so
        # one call serves the predictive space and the similarity one.
        for _sp in spaces:
            if _sp in BASE_SPACES:
                continue
            # ONE BUILD PER DISTINCT RECIPE, NOT PER MODEL -- but only where
            # the model provably does not enter the build. Under a recipe from
            # the similarity family no downstream model is fitted inside
            # build_fold_hybrid and the model is not passed at all
            # (hybrid_for_fold's own docstring, and it reads the objective off
            # the blueprint). So two models holding the SAME geometry recipe
            # get bit-identical matrices and the second build is waste: five
            # downstream models meant five builds of one matrix per fold.
            #
            # It is NOT assumed that they hold the same one. The pool looks the
            # recipe up per model -- get_best_search_for_downstream skips a
            # stored search the model is absent from and then takes
            # get_best_for_downstream for it (_shared.py:6133-6138) -- so two
            # models can come back with different similarity recipes. The
            # grouping below compares the recipes it actually got and falls
            # back to one build per model when they differ, so the worst case
            # is what this did before.
            #
            # Under a predictive recipe the model IS fitted inside the build,
            # so no grouping is attempted there.
            _groups = group_models_by_recipe(
                {_m: recipe_for_space_by_model.get(_m, {}).get(_sp)
                 for _m in downstream_models},
                downstream_models)

            _cells = []
            for _bp_m, _mnames, _shareable in _groups:
                # The model instance is passed only when the build fits one.
                # Through noise_cache since 2026-09-25 (plan section 3): the
                # same hybrid_for_fold call, stored under its inputs, so the
                # similarity sweep's 'hybrid' / 'hybrid_similarity' spaces for
                # this model read this very build back instead of repeating it.
                X_tr_h, X_va_h = nc.cached_hybrid_for_fold(
                    dataset_name, _tr_reps, _va_reps, y_train, _bp_m, g_tr,
                    None if _shareable else model_factories[_mnames[0]](),
                    metric_fn, lower_is_better,
                    model_name=None if _shareable else _mnames[0],
                    metric_name=metric_name, train_idx=train_idx)
                if X_tr_h is None:
                    for mname in _mnames:
                        hybrid_failures.append({'fold': int(fold_i),
                                                'space': _sp,
                                                'model': mname})
                    continue
                sc_h = StandardScaler()
                _cells.append((sc_h.fit_transform(X_tr_h),
                               sc_h.transform(X_va_h), list(_mnames),
                               'recipe of ' + '+'.join(_mnames)))
            if _cells:
                fold_spaces[_sp] = _cells

        # One pass per (space, feature matrix): once for the base
        # representation, once per downstream model in each hybrid space.
        for space, (X_tr_s, X_va_s, cell_models, cell_label) in (
                (_sp, _cell) for _sp in spaces
                for _cell in fold_spaces.get(_sp, ())):

            # The score to beat, ON THIS MATRIX. One fit per (fold, matrix,
            # model), shared by every cleanup, every flag rate and every
            # detector — they all have to beat the same number or the
            # comparison is not paired.
            # ⛔ THE THREE MODEL-BASED CORRECTIONS' TARGETS, ONCE PER
            # (FOLD, FEATURE MATRIX). Each depends on the features, the labels
            # and the groups and on nothing else -- not the suspicion score,
            # not the strength, not which model is about to be fitted. They
            # are computed once per matrix, which is once per (fold, space)
            # for the base representation and once per (fold, space, model)
            # in a hybrid space, because those matrices differ. Left in
            # cleaned_training_set they were refitted for every setting: 3
            # strengths x 2 detectors x 2 models = 12 identical fits where 1
            # does, and the Gaussian process is cubic in the training rows with
            # 5 inner folds per call. Same argument _compute_noise_scores
            # carries for the detector, and what n11 does with EXTRA_TARGETS.
            _targets = correction_targets(
                X_tr_s, y_train, groups=g_tr,
                # Derived from n07.CLEANUP_TARGETS rather than written out
                # here. The hand-written list this replaces named three
                # cleanups, so js_shrink -- added 2026-09-23 and needing the
                # same out-of-fold prediction oof_smooth uses -- was never
                # given one.
                want=targets_needed(CLEANUPS))

            base_score = {}
            for mname in cell_models:
                sc = _evaluate_fold(
                    X_tr_s, X_va_s, y_train, y_val, 'no_noise', 0.0,
                    model_factories[mname], metric_fn, is_clf=is_clf,
                    metric_name=metric_name, dataset_name=dataset_name,
                    precomputed_y_tr=y_train,
                    cache_dataset=dataset_name,
                    cache_model_name=_cache_name(mname))
                base_score[mname] = float(sc) if sc is not None else float('nan')

            def _cleaned_score(cleanup, rate, slot, mname, scores):
                """Held-out score after `cleanup` driven by `scores`, or None
                when this cleanup has no setting at this slot."""
                _axis, _vals = setting_axis(cleanup, is_clf)
                if slot >= len(_vals):
                    return None
                _strength = None if _axis == 'flag_rate' else _vals[slot]
                sc = _evaluate_fold(
                    X_tr_s, X_va_s, y_train, y_val, cleanup, rate,
                    model_factories[mname], metric_fn,
                    is_clf=is_clf, metric_name=metric_name,
                    dataset_name=dataset_name,
                    precomputed_y_tr=y_train, noise_scores=scores,
                    strength=_strength, targets=_targets,
                    train_groups=g_tr, cache_dataset=dataset_name,
                    cache_model_name=_cache_name(mname))
                return float(sc) if sc is not None else float('nan')

            # This matrix's default-path scores per detector, which the
            # similarity sweep below reuses for a detector that reads no
            # neighbours (cross_val).
            scores_by_detector = {}
            _mi = f'{space}:{cell_label}'
            for det_i, detector in enumerate(detectors):
                # Re-run per space. The detector reads features, so the
                # molecules it flags in the hybrid are not the ones it flags in
                # the base representation.
                scores = _suspicion(detector, X_tr_s, y_train, is_clf,
                                    groups=g_tr, dataset=dataset_name)
                if scores is None:
                    continue
                scores_by_detector[detector] = scores
                # The control: the same scores, dealt at random. Seeded on
                # (seed, fold, space, detector) so a re-run reproduces it and
                # two spaces or detectors never share a permutation.
                for seed in shuffle_seeds:
                    _perm = np.random.default_rng(
                        [int(seed), int(fold_i), ALL_SPACES.index(space),
                         det_i]).permutation(len(scores))
                    _shuf = np.asarray(scores)[_perm]
                    for slot, rate in enumerate(flag_rates):
                        for mname in cell_models:
                            for cleanup in ctrl_cleanups:
                                sc = _cleaned_score(cleanup, rate, slot,
                                                    mname, _shuf)
                                if sc is not None:
                                    acc_ctrl[space][detector][seed][rate][
                                        mname][cleanup].append(sc)
                                    fold_rows.append(sx.row(
                                        _today_ident(
                                            f'{cleanup}__score_shuffled_seed{seed}',
                                            cleanup, detector, slot, rate, _mi,
                                            fold_i),
                                        model=mname, metric=metric,
                                        score_D=base_score[mname],
                                        score_Dprime=sc))
                for slot, rate in enumerate(flag_rates):
                    folds_ran[space][detector][rate].add(int(fold_i))
                    for mname in cell_models:
                        acc[space][detector][rate][mname]['no_cleaning'].append(
                            base_score[mname])
                        for cleanup in CLEANUPS:
                            # Each cleanup is swept over the axis it actually
                            # reads. `remove` and `knn_smooth` threshold at the
                            # (1 - rate) quantile, so the rate is their axis.
                            # `downweight` and `label_smooth` read no quantile
                            # on a regression set, so sweeping the rate for
                            # them swept nothing -- they get the strength axis
                            # instead, at the same three slots. See
                            # setting_axis.
                            _axis, _vals = setting_axis(cleanup, is_clf)
                            if slot >= len(_vals):
                                # This cleanup has fewer settings than slots.
                                # Only quantile_clip does: it reads neither
                                # axis, so scoring it here would be the same
                                # column fitted again.
                                continue
                            _strength = (None if _axis == 'flag_rate'
                                         else _vals[slot])
                            sc = _evaluate_fold(
                                X_tr_s, X_va_s, y_train, y_val, cleanup, rate,
                                model_factories[mname], metric_fn,
                                is_clf=is_clf, metric_name=metric_name,
                                dataset_name=dataset_name,
                                precomputed_y_tr=y_train, noise_scores=scores,
                                strength=_strength,
                                targets=_targets,
                                # ⛔ REQUIRED BY THE THREE MODEL-BASED
                                # CORRECTIONS. oof_smooth, gp_smooth and
                                # quantile_clip each fit a model on part of the
                                # training rows and read the target off the
                                # rest. Without the scaffold groups that inner
                                # split is random, so a molecule's own scaffold
                                # family trains the model that says what its
                                # label should have been -- which is the leak
                                # cleaned_training_set's docstring warns about.
                                train_groups=g_tr,
                                cache_dataset=dataset_name,
                                cache_model_name=_cache_name(mname))
                            acc[space][detector][rate][mname][cleanup].append(
                                float(sc) if sc is not None else float('nan'))
                            fold_rows.append(sx.row(
                                _today_ident(cleanup, cleanup, detector, slot,
                                             rate, _mi, fold_i),
                                model=mname, metric=metric,
                                score_D=base_score[mname], score_Dprime=sc))

            # ── The similarity sweep on this feature matrix ──
            if sweep:
                _sweep_cell(X_tr_s, X_va_s, cell_models, _mi, g_tr, y_train,
                            y_val, train_idx, _tr_reps, _va_reps, smiles_tr,
                            scores_by_detector, _targets, base_score, fold_i)

        rows.write(fold_rows)
        ck.save(fold_i, {'acc': acc, 'folds_ran': folds_ran,
                         'acc_ctrl': acc_ctrl,
                         'hybrid_failures': hybrid_failures,
                         'sim_unavailable': sim_unavailable})
        print(f"    fold {fold_i}: [{time.time() - t0:.0f}s]")
    results['rows'] = rows.summary()
    if sim_unavailable:
        results['similarity_settings_unavailable'] = sim_unavailable

    if hybrid_failures:
        # Named rather than left as a shorter list of folds. Those folds still
        # carry every base-space number.
        # One entry per (fold, space, downstream model) that could not be
        # built, so a geometry recipe failing on a fold is not confused with
        # the predictive one failing on it, and one model's build failing is
        # not read as every model's. Those folds keep every number from the
        # matrices that did build.
        results['folds_with_no_hybrid'] = hybrid_failures
        print(f"  a hybrid space could not be built on "
              f"{len(hybrid_failures)} (fold, space, model) triples: "
              f"{hybrid_failures}; the spaces that built keep their numbers")

    # ── Fold win counts, which are the headline ──
    for space in spaces:
        space_block = {}
        for detector in detectors:
            space_block[detector] = {}
            for rate in flag_rates:
                block = {'n_folds_ran': len(folds_ran[space][detector][rate]),
                         'per_model': {}}
                for mname in downstream_models:
                    cell = acc[space][detector][rate][mname]
                    base = np.asarray(cell['no_cleaning'], dtype=float)
                    row = {'no_cleaning_mean': float(np.nanmean(base))
                           if base.size else float('nan')}
                    for cleanup in CLEANUPS:
                        _axis, _vals = setting_axis(cleanup, is_clf)
                        _slot = flag_rates.index(rate)
                        if _slot >= len(_vals):
                            # quantile_clip reads neither axis, so it was
                            # scored at the first slot only. Nothing to write
                            # here; a duplicate row would be counted as a
                            # second setting by the summary.
                            continue
                        got = np.asarray(cell[cleanup], dtype=float)
                        wins = [_won(c, b, lower_is_better)
                                for c, b in zip(got, base)]
                        scored = [w for w in wins if w is not None]
                        diff = np.array(
                            [(b - c) if lower_is_better else (c - b)
                             for c, b in zip(got, base)
                             if np.isfinite(c) and np.isfinite(b)], dtype=float)
                        row[cleanup] = {
                            # Which knob this row's setting is, and its value.
                            # 'flag_rate' rows differ by how many molecules
                            # were flagged; 'strength' rows differ by how hard
                            # the cleanup leaned on the score.
                            'swept_axis': _axis,
                            'setting': (rate if _axis == 'flag_rate'
                                        else _vals[_slot]),
                            # The headline. Folds the cleaned model beat the
                            # uncleaned one on, out of the folds both scored.
                            'folds_won': int(sum(scored)),
                            'folds_scored': len(scored),
                            # Beside it, never instead of it.
                            'mean_gain': float(diff.mean()) if diff.size else float('nan'),
                            'fold_gains': [float(x) for x in diff],
                            'spread_over_folds': float(diff.std()) if diff.size else float('nan'),
                            'cleaned_mean': float(np.nanmean(got)) if got.size else float('nan'),
                        }
                    block['per_model'][mname] = row
                space_block[detector][str(rate)] = convert_numpy(block)
        results['per_space'][space] = {'per_detector': space_block}

        # ── The shuffled-score control, paired fold by fold ──
        # {detector: {rate: {model: {cleanup: {seed: row}}}}}. Each row is the
        # SAME setting as the real row in per_detector, with the scores dealt
        # at random.
        ctrl_block = {}
        for detector in detectors:
            ctrl_block[detector] = {}
            for rate in flag_rates:
                by_model = {}
                for mname in downstream_models:
                    base = np.asarray(
                        acc[space][detector][rate][mname]['no_cleaning'],
                        dtype=float)
                    by_cleanup = {}
                    for cleanup in ctrl_cleanups:
                        real = np.asarray(
                            acc[space][detector][rate][mname][cleanup],
                            dtype=float)
                        per_seed = {}
                        for seed in shuffle_seeds:
                            shuf = np.asarray(
                                acc_ctrl[space][detector][seed][rate][mname][
                                    cleanup], dtype=float)
                            if not shuf.size or shuf.size != real.size:
                                continue
                            sign = -1.0 if lower_is_better else 1.0
                            ok = np.isfinite(shuf) & np.isfinite(base)
                            gain = sign * (shuf - base)
                            both = ok & np.isfinite(real)
                            r_minus_s = sign * (real - shuf)
                            per_seed[str(seed)] = {
                                'shuffled_mean': float(np.nanmean(shuf)),
                                # Against no cleaning, same sign as the real
                                # row's mean_gain.
                                'shuffled_mean_gain': (float(gain[ok].mean())
                                                       if ok.any()
                                                       else float('nan')),
                                'shuffled_folds_won': int(np.sum(gain[ok] > 0)),
                                # Against the real detector: positive means
                                # the detector's own ranking did better.
                                'real_minus_shuffled': (
                                    float(r_minus_s[both].mean())
                                    if both.any() else float('nan')),
                                'real_minus_shuffled_per_fold': [
                                    float(x) for x in r_minus_s[both]],
                                'folds_real_beat_shuffled': int(
                                    np.sum(r_minus_s[both] > 0)),
                                'folds_scored': int(both.sum()),
                            }
                        if per_seed:
                            by_cleanup[cleanup] = per_seed
                    by_model[mname] = by_cleanup
                ctrl_block[detector][str(rate)] = convert_numpy(by_model)
        results['per_space'][space]['shuffled_control'] = ctrl_block

    results['summary'] = _summarise(results, downstream_models, flag_rates,
                                    detectors, spaces, metric_name,
                                    cleanups=CLEANUPS)
    results['summary']['shuffled_control'] = _summarise_control(
        results, spaces, shuffle_seeds, ctrl_cleanups)
    _print_summary(results, metric_name)

    # (results, output_dir, experiment_id, dataset) — see the note in n17; the
    # arguments were rotated here the same way and with the same two effects.
    save_experiment_results(results, output_dir, 'n18_clean_the_real_labels',
                            dataset_name)
    # The result file holds every fold now, so the per-fold snapshot is spent.
    ck.clear()
    return results


def _summarise(results, models, flag_rates, detectors, spaces, metric_name,
               cleanups=None):
    """Which (cleanup, detector, flag rate, model) won on most folds, per space.

    A cleanup that wins on every fold at one flag rate and loses at the others
    has shown less than one that wins at all of them, so the count of settings it
    won a majority of folds at is recorded beside its best single setting.

    Every count is taken inside ONE space. A setting that swept the base
    representation and lost in the hybrid is two readings, not an average.
    """
    cleanups = tuple(cleanups) if cleanups else CLEANUPS
    out = {'metric': metric_name, 'headline': 'folds won, not mean gain',
           'spaces': list(spaces), 'per_space': {}}
    for space in spaces:
        per_cleanup = {}
        for cleanup in cleanups:
            rows = []
            for d in detectors:
                for r in flag_rates:
                    blk = (results['per_space'].get(space, {})
                           .get('per_detector', {}).get(d, {}).get(str(r)))
                    if not blk:
                        continue
                    for m in models:
                        # Absent, not zero: a cleanup that reads neither axis
                        # is scored at the first slot only, so it has no row at
                        # the others. Counting a missing row as a setting would
                        # make the counts mean different things per cleanup,
                        # which is the defect this file already fixed once.
                        cell = blk['per_model'][m].get(cleanup)
                        if not cell or cell['folds_scored'] == 0:
                            continue
                        rows.append({
                            'detector': d, 'model': m,
                            'swept_axis': cell.get('swept_axis', 'flag_rate'),
                            'setting': cell.get('setting', r),
                            # Kept so a reader written before the strength axis
                            # existed still finds the key it looks for. On a
                            # strength row it is the slot, not a quantile.
                            'flag_rate': r,
                            'folds_won': cell['folds_won'],
                            'folds_scored': cell['folds_scored'],
                            'mean_gain': cell['mean_gain'],
                        })
            if not rows:
                per_cleanup[cleanup] = {'settings_scored': 0}
                continue
            majority = [x for x in rows
                        if x['folds_won'] * 2 > x['folds_scored']]
            clean_sweep = [x for x in rows
                           if x['folds_won'] == x['folds_scored']]
            best = max(rows, key=lambda x: (x['folds_won'] / max(x['folds_scored'], 1),
                                            x['mean_gain']))
            per_cleanup[cleanup] = {
                'settings_scored': len(rows),
                'settings_winning_most_folds': len(majority),
                'settings_winning_every_fold': len(clean_sweep),
                'best_setting': best,
            }
        out['per_space'][space] = {'per_cleanup': per_cleanup}

    # Does a cleanup sweep more settings where the features were chosen? Only
    # defined where both spaces ran.
    if 'hybrid' in spaces and any(s in spaces for s in BASE_SPACES):
        # One entry per space that ran, rather than the two this block used to
        # name. The third space -- the one whose features were chosen to make
        # the neighbour distance predictive -- would otherwise be absent from
        # the only table that puts the spaces side by side.
        out['sweeps_by_space'] = {
            c: {
                f'{_k}_in_{_sp}': out['per_space'][_sp]['per_cleanup'][c].get(
                    _k, 0)
                for _sp in out['per_space']
                for _k in ('settings_winning_every_fold', 'settings_scored')
            }
            for c in cleanups}
        out['sweeps_by_space_note'] = (
            'How many settings each cleanup won EVERY fold at, in the base '
            'representation and in the hybrid, out of how many settings were '
            'scored in each. A setting is one (detector, flag rate, model).')
    return out


def _summarise_control(results, spaces, seeds, cleanups):
    """Per space and cleanup: does the detector's ranking beat a random one?

    One setting is one (detector, flag rate or strength, model). For each seed
    separately: in how many settings the real cleanup beat the shuffled one on
    most folds, and the mean over settings of the real and the shuffled gain
    against no cleaning. No number here is averaged across seeds.
    """
    out = {'what_one_setting_is': '(detector, flag rate or strength, model)',
           'per_space': {}}
    for space in spaces:
        ctrl = results['per_space'].get(space, {}).get('shuffled_control', {})
        real_blk = results['per_space'].get(space, {}).get('per_detector', {})
        per_cleanup = {}
        for cleanup in cleanups:
            per_seed = {}
            for seed in seeds:
                s = str(seed)
                real_g, shuf_g, beat, n = [], [], 0, 0
                for det, by_rate in ctrl.items():
                    for rate, by_model in by_rate.items():
                        for m, by_c in by_model.items():
                            row = (by_c.get(cleanup) or {}).get(s)
                            real = ((real_blk.get(det, {}).get(rate, {})
                                     .get('per_model', {}).get(m, {})
                                     .get(cleanup)))
                            if not row or not real or not row['folds_scored']:
                                continue
                            n += 1
                            real_g.append(real['mean_gain'])
                            shuf_g.append(row['shuffled_mean_gain'])
                            if row['folds_real_beat_shuffled'] * 2 > row['folds_scored']:
                                beat += 1
                if n:
                    per_seed[s] = {
                        'settings': n,
                        'settings_where_real_beat_shuffled_on_most_folds': beat,
                        'mean_real_gain': float(np.nanmean(real_g)),
                        'mean_shuffled_gain': float(np.nanmean(shuf_g)),
                    }
            per_cleanup[cleanup] = per_seed
        out['per_space'][space] = per_cleanup
    return out


def _print_summary(results, metric_name):
    s = results['summary']
    print("\n" + "=" * 70)
    print(f"SUMMARY — folds won against no cleaning ({metric_name})")
    if results['is_computed_label_control']:
        print("  THIS IS THE CONTROL DATASET. Its labels are calculated, so a win")
        print("  here is a cleanup removing hard molecules, not wrong ones.")
    if not results.get('hybrid_available', True):
        print("  no hybrid recipe was available; base representation only")
    for space, srow in s['per_space'].items():
        print(f"\n  {space}:")
        for cleanup, row in srow['per_cleanup'].items():
            if not row.get('settings_scored'):
                print(f"    {cleanup:14s} no setting produced a score")
                continue
            b = row['best_setting']
            print(f"    {cleanup:14s} won most folds in "
                  f"{row['settings_winning_most_folds']} of "
                  f"{row['settings_scored']} settings; every fold in "
                  f"{row['settings_winning_every_fold']}")
            print(f"                   best: {b['model']}/{b['detector']}"
                  f"/{b.get('swept_axis', 'flag_rate')} "
                  f"{b.get('setting', b.get('flag_rate'))} — "
                  f"{b['folds_won']} of {b['folds_scored']} folds, "
                  f"mean gain {b['mean_gain']:+.4f}")
    sc = s.get('shuffled_control', {}).get('per_space', {})
    if sc:
        print(f"\n  SHUFFLED-SCORE CONTROL — the same cleanup with the suspicion "
              f"scores dealt at random ({metric_name}, + means better than no "
              f"cleaning):")
        for space, by_c in sc.items():
            print(f"\n  {space}:")
            for cleanup, by_seed in by_c.items():
                for seed, row in by_seed.items():
                    print(f"    {cleanup:14s} seed {seed}: real beat shuffled on "
                          f"most folds in "
                          f"{row['settings_where_real_beat_shuffled_on_most_folds']}"
                          f" of {row['settings']} settings; mean gain real "
                          f"{row['mean_real_gain']:+.4f}, shuffled "
                          f"{row['mean_shuffled_gain']:+.4f}")
    print("\n  One setting winning on one fold is not a result. Read the counts.")
    if 'sweeps_by_space' in s:
        print("  'sweeps_by_space' in the result file holds the sweep counts in "
              "both spaces side by side.")


def main():
    parser = create_experiment_parser(
        'N18: does cleaning help on the labels as they come')
    parser.add_argument('--flag-rates', type=str, default=None,
                        help='Comma-separated shares of rows to treat as suspect')
    parser.add_argument('--shuffle-seeds', type=str, default=None,
                        help='Comma-separated seeds for the shuffled-score '
                             'control. Default SHUFFLE_SEEDS; "" switches it off')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, rep_preset=args.reps)
    kw = {}
    if args.flag_rates:
        kw['flag_rates'] = [float(x) for x in args.flag_rates.split(',')]
    if args.shuffle_seeds is not None:
        kw['shuffle_seeds'] = [int(x) for x in args.shuffle_seeds.split(',')
                               if x.strip()]
    run(data, config, args.output, args.dataset, **kw)


if __name__ == '__main__':
    main()
