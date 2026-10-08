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

WHERE EVERY CLEANUP IS MEASURED (2026-10-05, chat 5). Per outer fold, each of
the dataset's two decided models (DECISIONS.md Step 1) is fitted on its own 4
configurations: its 2 best single representations (base_1, base_2) and its 2
best general hybrids (hybrid, hybrid__general2), all ranked by searches run on
that fold's training rows (noise_helpers.fold_configurations). 2 models x 4 =
8. Until 2026-10-05 this file fitted five models in up to 8 spaces each, every
recipe and every representation ranking read from searches run once over all
rows, so the held-out fold had a say in what it was scored on.

The distance hybrid and the similarity hybrid are neighbour spaces only: the
detectors and smoothers here judge a molecule against its nearest neighbours,
so both are among the neighbour settings the per-fold screen picks from, and
the top 3 settings are refitted into the rows file. No model is fitted on
them.

THE HYBRID IS BUILT ON THE TRAINING MOLECULES OF A FOLD, AT THEIR UNCLEANED
LABELS. It is searched and built once per fold per model and every cleanup,
flag rate and detector that model is scored under shares it, so the only
thing differing between two cleanups is what the downstream fit was handed.
The held-out molecules' labels reach no search, no feature ranking and no
budget split. Whether a recipe searched AFTER the cleaning would do better is
a different question and it is n20's.

The detector is re-run in each configuration. It reads features, so the
molecules it flags in a hybrid are not the molecules it flags in a single
representation -- which is the whole reason a cleanup can move between them.

A CONFIGURATION THAT CANNOT BE BUILT IS NAMED, NOT FAKED. A model short of 2
hybrids with distinct features, or whose per-fold search failed, loses those
configurations on that fold; `configurations_unavailable` and `fold_records`
say which and why, and the other configurations keep their numbers.

NO ORACLE CAVEAT TO CARRY. n11, n12 and n13 record that their recipe was chosen
on clean labels and their experiment then corrupts them. Nothing is corrupted
here and there is no clean label column, so the per-fold searches saw the same
labels this experiment trains on. `recipe_saw_clean_labels` is written False
for that reason.

DECIDED 2026-10-06 (Adelaide delegated), what changed in this file:
  - label_smooth and knn_smooth in the main table find their neighbours in
    each (fold, configuration)'s top screened neighbour setting (kept[0]:
    space and distance, neighbour count, counting scheme), and
    neighbor_residual reads the same lists. Until 2026-10-06 they found 9 or
    10 Euclidean neighbours in the configuration's own features, each
    counting equally. The screen predicts a held-back fifth of the training
    molecules (n07.screen_settings, inner=).
  - constant_lift reads no detector score, so it is scored once per
    (configuration, model) under its own heading, 'no_detector', not under
    the first detector.
  - herg_ic50 is collapsed to one row per molecule, labelled with the mean of
    its measurements, before the folds are drawn (9,802 molecules from
    12,008 measurement rows). svm fits on it now. Its per-fold searches are
    filed under MOLECULE_MEAN_SUFFIX.
  - On a classification dataset knn_smooth is not run: it writes the same
    labels as label_smooth there.

Usage:
    python n18_clean_the_real_labels.py --dataset logd --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np

# 2026-10-05 (chat 5): the whole-dataset readers this file imported
# (get_best_search_for_downstream, pool_recipe_per_downstream_model,
# pool_recipes_by_family, noise_base_reps_by_model, top_base_reps,
# top_hybrids, select_model_rep_pairs, hybrid_deltas, cv_split,
# DOWNSTREAM_MODELS) are gone. Folds, models and configurations come from
# the per-fold helpers in noise_helpers.
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_domain_metric, get_model_factories,
)
from experiments.decided_grid import decided_models
from experiments.noise.noise_helpers import (
    noise_outer_folds, fold_spread, fold_configurations, fold_model_inputs,
    configuration_record, dataset_knn_group, CONFIGURATION_SPACES)
# The cleanups are n07's, called directly rather than copied. n07 injects
# noise unless it is handed a training label vector, and it is handed one here --
# the labels exactly as the loader produced them. So the same code runs, over the
# real labels, with nothing corrupted. The neighbour settings, the screen and
# the neighbour-space builds are n07's shared ones too.
from experiments.noise.n07_remediation import (
    _evaluate_fold, targets_needed, correction_targets, SCORE_FREE_CLEANUPS,
    cleaned_training_set, sweep_settings, sweep_ks, sweep_schemes,
    screen_settings, screen_top, SCREEN_SCORED_ON, fold_similarity_spaces,
    fold_neighbours, similarity_record, MAIN_TABLE_NEIGHBOUR_CLEANUPS,
    WEIGHTING_CLEANUPS, last_decline_reason, clear_decline_reason)
from experiments._shared import as_rep_dtype
# The D-versus-D' rows, the controls and the per-fold checkpoint, added
# 2026-09-25 (plan sections 1-3). See _sim_axis_n15_n22.
from experiments.noise import noise_cache as nc
from experiments.noise import _sim_axis_n15_n22 as sx

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
#
# 2026-10-05 (f36, F36.md section 9): quantile_clip is cut and n30's
# constant_lift is added. constant_lift raises the training labels sitting at
# the ceiling the labels themselves show by one constant, and declines on a
# dataset whose labels show no ceiling.
ALL_CLEANUPS = ('remove', 'downweight', 'label_smooth', 'knn_smooth',
                'oof_smooth', 'gp_smooth', 'constant_lift')

# Not defined on a nominal class label, so they are dropped on a
# classification dataset rather than writing NaN rows. Same tuple as n07's.
REGRESSION_ONLY_CLEANUPS = ('oof_smooth', 'gp_smooth', 'constant_lift')

# DECIDED 2026-10-06 (Adelaide delegated): on a classification dataset
# label_smooth runs and knn_smooth does not. Both flip a flagged label to its
# neighbours' majority class there, so they write the same labels.
CLASSIFICATION_DUPLICATE_CLEANUPS = ('knn_smooth',)


def cleanups_for(is_clf):
    """The cleanups this task type can actually run."""
    if not is_clf:
        return ALL_CLEANUPS
    return tuple(c for c in ALL_CLEANUPS
                 if c not in REGRESSION_ONLY_CLEANUPS
                 and c not in CLASSIFICATION_DUPLICATE_CLEANUPS)


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
    if cleanup in SCORE_FREE_CLEANUPS:
        # Read neither axis (n07.SCORE_FREE_CLEANUPS: quantile_clip,
        # invvar_weight, constant_lift). quantile_clip: a label is inside the
        # range its own leaves carry or it is not. constant_lift (2026-10-05):
        # n07.cleaned_training_set reads only the labels and the out-of-fold
        # prediction, never noise_fraction, strength or noise_scores. Each is
        # one column, scored once -- see setting_axis. Until this branch
        # existed constant_lift fell through to True and was fitted at all
        # three flag rates on one identical column.
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

    The score-free cleanups (`quantile_clip`, `constant_lift`) come back with
    ONE value and the axis name 'none': neither reads a quantile of the
    suspicion scores or a strength. It is
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

# Which feature spaces each cleanup is measured in (2026-10-05, chat 5).
#
# Per outer fold, per decided model: its 2 best single representations
# (base_1, base_2) and its 2 best general hybrids (hybrid, hybrid__general2),
# all ranked by a search on that fold's training rows
# (noise_helpers.CONFIGURATION_SPACES). 2 models x 4 = 8 fitted configurations
# (R5). Until 2026-10-05 this file fitted up to 8 spaces per model for five
# models, all read from searches run once over every row: base_1, base_2,
# 'hybrid', 'hybrid_similarity', and the general rank-2 and distance rank-1
# slot of each family.
#
# The distance hybrid ('hybrid__distance1') and the similarity hybrid
# ('hybrid_similarity') are NEIGHBOUR spaces only (handoff 2.3 item 1). Every
# detector and smoother here judges a molecule against its nearest
# neighbours, so both are in the neighbour settings the screen chooses from
# (_sweep_cell). No model is fitted on them.
SPACES = tuple(CONFIGURATION_SPACES)

# What this file writes since chat 5 (2026-10-05, HANDOFF_EXPERIMENT_ALIGNMENT
# section 2.4). In the rows file's name, the result file and the checkpoint's
# signature, so nothing written by the whole-dataset design is read back.
# b: constant_lift scored once per (configuration, model), 2026-10-05.
# 2026-10-06: the main table's neighbours come from the top screened setting,
# constant_lift sits under 'no_detector', herg_ic50 is one row per molecule,
# knn_smooth is not run on classification.
RESULT_LAYOUT = 'fold_aligned_2026-10-06'

# The heading the score-free cleanups (constant_lift) are filed under in
# acc, per_space and the summary: they read no detector score, so they are
# scored once per (fold, configuration, model) and not once per detector.
NO_DETECTOR = 'no_detector'

# The per-fold searches on herg_ic50's one-row-per-molecule data are filed
# under this name, so they are never read for the measurement rows (or the
# other way round). n19 and n26 use their own names for the same rows.
MOLECULE_MEAN_SUFFIX = 'n18_one_row_per_molecule_mean_label'


def one_row_per_molecule(data, dataset_name):
    """(data, record). A loader that keeps one row per measurement
    (herg_ic50: 12,008 rows over 9,802 molecules) hands back
    data['replicate_groups']; this returns the dataset with one row per
    molecule, its label the mean of the molecule's measurements and its
    features, group and SMILES those of its first row. Molecules are in the
    order of their first row. Representations are float32.

    Returns (data, None) unchanged when there is no replicate grouping, when
    every row is its own molecule, or on a classification dataset (a mean of
    class labels is not a class); the record then says why.

    DECIDED 2026-10-06 (Adelaide delegated): n18 collapses herg_ic50 this way
    so svm fits (get_model_factories skips svm above 10,000 rows)."""
    rg = data.get('replicate_groups')
    if rg is None:
        return data, None
    labels = np.asarray(data['labels'])
    n = len(labels)
    rg = np.asarray(rg)
    if len(rg) != n:
        raise RuntimeError(
            f"n18: {dataset_name} has {len(rg)} replicate group ids for {n} "
            f"rows")
    _u, first, inverse = np.unique(rg, return_index=True, return_inverse=True)
    if len(first) == n:
        return data, None
    if data.get('task_type') == 'classification':
        return data, {'collapsed': False,
                      'why': 'classification labels: a mean is not a class'}
    order = np.argsort(first, kind='stable')
    idx = first[order]
    counts = np.bincount(inverse)
    means = (np.bincount(inverse, weights=labels.astype(np.float64))
             / counts)[order]
    out = dict(data)

    def _take(v):
        if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n:
            return v[idx]
        if isinstance(v, list) and len(v) == n:
            return [v[int(i)] for i in idx]
        return v

    out['reps'] = {k: as_rep_dtype(np.asarray(m)[idx])
                   for k, m in data['reps'].items()}
    augs = data.get('augmentations')
    if augs:
        out['augmentations'] = {k: as_rep_dtype(np.asarray(m)[idx])
                                for k, m in augs.items()}
    out['labels'] = means
    for key in ('groups', 'smiles'):
        if data.get(key) is not None:
            v = data[key]
            out[key] = (np.asarray(v)[idx] if not isinstance(v, list)
                        else [v[int(i)] for i in idx])
    # Every other per-row array (ChEMBL documents and assays, metadata ids)
    # is read at the molecule's first row, so nothing keeps the old length.
    for key, v in data.items():
        if key in ('reps', 'augmentations', 'labels', 'groups', 'smiles',
                   'replicate_groups'):
            continue
        if isinstance(v, dict):
            out[key] = {k2: _take(v2) for k2, v2 in v.items()}
        else:
            out[key] = _take(v)
    out['replicate_groups'] = None
    rec = {'collapsed': True,
           'n_measurement_rows': int(n),
           'n_molecules': int(len(idx)),
           'molecules_measured_more_than_once': int((counts > 1).sum()),
           'label': 'mean of the molecule\'s measurements',
           'features_from': 'the molecule\'s first row',
           'search_condition_suffix': MOLECULE_MEAN_SUFFIX}
    print(f"  {dataset_name}: {n} measurement rows collapsed to {len(idx)} "
          f"molecules (label = mean of each molecule's measurements)")
    return out, rec


def _suspicion(detector, X_tr_s, y_tr, is_clf, groups=None, dataset=None):
    """One suspicion score per training row, or None if the detector failed.

    2026-10-06: the main table calls this for detectors that read no
    neighbours (cross_val) only. neighbor_residual reads the top screened
    setting's lists instead (run(), the detector loop); the history below is
    about the path it used before.

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

    ⛔ REVERSED 2026-10-05 (KNN_SEARCH_SPACE_RECORD axis 6, Adelaide): a
    molecule's scaffold-mates stay in its neighbour list. n07._compute_noise_
    scores now hands the neighbour detectors exclude_groups=None, so
    neighbor_residual here finds neighbours among the full training set again.
    cross_val still gets `groups` for its inner folds.

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
    """'won', 'lost' or 'tied' for the cleaned model against the uncleaned
    one on this fold, or None when either score is missing.

    2026-10-05 (dirty-run bug, n18): a tie used to come back False and was
    counted as a loss. A cleanup that changes nothing (it declined, or it
    moved no label) ties exactly, so every such fold read as the cleanup
    hurting. A tie is now its own outcome.
    """
    if not (np.isfinite(cleaned) and np.isfinite(base)):
        return None
    if cleaned == base:
        return 'tied'
    better = (cleaned < base) if lower_is_better else (cleaned > base)
    return 'won' if better else 'lost'


# 2026-10-05 (chat 5): _drop_blueprint and _print_deltas are gone with the
# block they served, which ranked each model's top 2 representations and
# hybrids out of the whole-dataset stored search and printed the top hybrid
# against the mean of the representations that fed it (a mean baseline,
# dirty-run bug, n18). The per-fold configurations are in fold_records.


def run(data, config, output_dir, dataset_name, **kwargs):
    """N18: cleaning the labels as they come.

    CHAT 5 ALIGNMENT, 2026-10-05 (HANDOFF_EXPERIMENT_ALIGNMENT section 2.4).
    What each outer fold does, in order:
      1. Folds are noise_helpers.noise_outer_folds: the per-fold searches' own
         splits, the count from config['n_splits'] or KIRBY_NOISE_N_FOLDS.
      2. noise_helpers.fold_configurations runs this fold's searches on its
         training rows (decided grid, pruning off, the labels as delivered)
         and gives each of the dataset's two decided models its 2 best base
         representations, its 2 best general hybrids, and the recipes of the
         two neighbour spaces (the distance hybrid and the similarity
         hybrid). Nothing is read from a whole-dataset search.
      3. Each model is fitted on its 4 configurations (2 models x 4 = 8, R5).
         Every cleanup, detector and flag rate or strength is scored in each.
      4. The neighbour settings of KNN_SEARCH_SPACE_RECORD are screened per
         (fold, configuration) with no model fit, on a held-back fifth of the
         training molecules. The top one is the main table's neighbour
         setting (2026-10-06); the top 3 (setting, neighbour count, counting
         scheme) are refitted, as rows.
      5. constant_lift is scored once per (configuration, model) under
         'no_detector'. herg_ic50 runs on one row per molecule (mean label).
    The accumulators are checkpointed and the rows written as each fold
    finishes.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT N18: DOES CLEANING HELP ON THE LABELS AS THEY COME")
    print("=" * 70)

    # herg_ic50: one row per molecule (mean label) before anything else reads
    # the data, so the folds, the searches and every fit are on molecules.
    data, collapse_record = one_row_per_molecule(data, dataset_name)
    collapsed = bool(collapse_record and collapse_record.get('collapsed'))
    search_suffix = MOLECULE_MEAN_SUFFIX if collapsed else None
    row_condition = (f'{RESULT_LAYOUT}__{MOLECULE_MEAN_SUFFIX}' if collapsed
                     else RESULT_LAYOUT)

    labels = np.asarray(data['labels'])
    reps = data['reps']
    groups = data.get('groups')
    groups_arr = None if groups is None else np.asarray(groups)
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    metric = (metric_name, metric_fn, lower_is_better)

    # R1 (2026-10-05, chat 5): the dataset's two decided models (DECISIONS.md
    # Step 1), nothing else. knn runs only inside the distance search. This
    # replaces config['models'] / DOWNSTREAM_MODELS (rf, xgboost, svm, dnn,
    # lightgbm), so a `models` kwarg is no longer read.
    models_decided = [m for m in decided_models(dataset_name, 'general')
                      if m != 'knn']
    model_factories = get_model_factories(
        models_decided, task_type=task_type, n_samples=len(labels))
    # DECIDED 2026-10-06 (Adelaide delegated): herg_ic50 is collapsed to one
    # row per molecule (mean label) above, 9,802 rows, so svm is under
    # get_model_factories' 10,000-row limit and fits. Until 2026-10-06 the
    # 12,008 measurement rows were used and only dnn ran on herg_ic50.
    models_without_factory = {
        m: ('get_model_factories built no model of this name for this '
            f'dataset ({len(labels)} rows; a model with a row cap is skipped '
            'above it)')
        for m in models_decided if m not in model_factories}
    downstream_models = [m for m in models_decided if m in model_factories]
    for m, why in models_without_factory.items():
        print(f"  {m}: not fitted -- {why}")
    if not downstream_models:
        raise RuntimeError(
            f"n18: no decided model could be built. Decided {models_decided}, "
            f"got {sorted(model_factories)}.")

    flag_rates = list(kwargs.get('flag_rates') or FLAG_RATES)
    detectors = list(kwargs.get('detectors') or DETECTORS)
    # An empty tuple switches the control off; None means the default.
    shuffle_seeds = list(SHUFFLE_SEEDS if kwargs.get('shuffle_seeds') is None
                         else kwargs['shuffle_seeds'])

    # What this task type can actually run. The three model-based corrections
    # are not defined on a nominal class label, so on a classification dataset
    # they are dropped rather than writing NaN rows -- n07 gates them the same
    # way. Every loop below reads this, not the module tuple.
    CLEANUPS = cleanups_for(is_clf)

    folds, n_folds = noise_outer_folds(data, config)
    smiles = data.get('smiles')
    has_smiles = smiles is not None and len(smiles) > 0
    n_molecules = len(set(smiles)) if has_smiles else len(labels)
    knn_grp = dataset_knn_group(data, dataset_name)
    # The neighbour settings this dataset's group sweeps (KNN_SEARCH_SPACE_
    # RECORD): no model-input space, no importance weighting, distances per
    # group. Neighbour counts (5, 10, 20) and counting schemes (rank,
    # inverse_square) are swept and screened per configuration below.
    #
    # DECIDED 2026-10-06 (Adelaide delegated): the spaces of a base
    # representation plus its added features are kept only on
    # ginkgo_colloidal and ginkgo_hic. n07.sweep_settings applies this; this
    # file adds no such space itself.
    sim_settings = sweep_settings(dataset_name, n_molecules, has_smiles)

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'result_layout': RESULT_LAYOUT,
        'noise_regime': 'whatever_the_assay_carries',
        'noise_injection': 'none',
        'models': downstream_models,
        'models_decided': models_decided,
        'models_without_factory': models_without_factory,
        'n_folds': int(n_folds),
        'folds_run': [int(f) for f, _tr, _va in folds],
        'selection_scope': 'fold_train_rows',
        'configuration_spaces': list(SPACES),
        'configuration_note': (
            'Per fold, per decided model: base_1 and base_2 are its 2 best '
            'single representations and hybrid and hybrid__general2 its 2 best '
            'general hybrids, all ranked by a search on that fold\'s training '
            'rows. Each model is fitted on its own 4; 2 models x 4 = 8. The '
            'distance hybrid (hybrid__distance1) and the similarity hybrid '
            '(hybrid_similarity) are neighbour spaces only: they reach the '
            'rows file through the neighbour settings, and no model is fitted '
            'on them. fold_records names every configuration per fold.'),
        'cleanups': list(CLEANUPS),
        'cleanups_not_run': [c for c in ALL_CLEANUPS if c not in CLEANUPS],
        'cleanups_not_run_why': {
            c: ('classification: writes the same labels as label_smooth'
                if c in CLASSIFICATION_DUPLICATE_CLEANUPS
                else 'regression only: not defined on a class label')
            for c in ALL_CLEANUPS if c not in CLEANUPS},
        'rows_collapsed_to_molecules': collapse_record,
        'search_condition_suffix': search_suffix,
        'main_table_neighbours': (
            'neighbor_residual, label_smooth and knn_smooth read the '
            'neighbour lists of each (fold, configuration)\'s top screened '
            'setting (space and distance, neighbour count, counting scheme); '
            'fold_records[fold].models[model].main_neighbour_setting names '
            'it, and main_neighbour_not_run says where no setting was kept. '
            'The screen predicts a held-back fifth of the training molecules '
            'from the rest, with no model fit.'),
        'no_detector_note': (
            f'Score-free cleanups ({", ".join(c for c in CLEANUPS if c in SCORE_FREE_CLEANUPS) or "none here"}) '
            f'read no detector score. They are scored once per (fold, '
            f'configuration, model) and filed under the detector heading '
            f'"{NO_DETECTOR}", at the first flag-rate slot only.'),
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
        'ties_note': (
            'A fold where the cleaned and the uncleaned model score exactly '
            'the same is a tie: counted in folds_tied, never as a win or a '
            'loss. A setting wins most folds when folds_won > folds_lost; it '
            'wins every fold when folds_won == folds_scored. Until 2026-10-05 '
            'a tie was counted as a loss.'),
        'claim_scope': (
            'No noise is injected. The uncleaned model trains on the labels as '
            'the loader produced them. Every score is measured against held-out '
            'labels that were NOT cleaned and are NOT clean. A win says the '
            'cleanup helped the model; it does NOT say the cleanup found errors.'),
        # Three facts a reader needs before reading any number below, each
        # stated rather than left to be inferred from the absence of a caveat.
        'recipe_saw_clean_labels': False,
        'recipe_saw_clean_labels_note': (
            'Nothing is injected here and there is no clean label column. The '
            'per-fold searches ranked representations and hybrids on each '
            'fold\'s training rows at the labels as delivered, the same labels '
            'this experiment trains on. The oracle caveat n11, n12 and n13 '
            'carry does not apply.'),
        'hybrid_sees_cleaned_labels': False,
        'hybrid_sees_cleaned_labels_note': (
            'Each hybrid is searched and built once per fold per decided model '
            'from that fold\'s training molecules at their UNCLEANED labels, '
            'and every cleanup, flag rate and detector shares it. No cleanup '
            'reaches the feature ranking or the budget split. Whether a recipe '
            'searched after the cleaning would do better is n20\'s question.'),
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

    print(f"  {len(labels)} rows, {len(folds)} of {n_folds} folds, "
          f"models {downstream_models} (decided)")
    print(f"  flag rates {flag_rates}, detectors {detectors}")
    print(f"  neighbour settings: {len(sim_settings)} ({knn_grp}) x ks "
          f"{list(sweep_ks())} x schemes {list(sweep_schemes())}, top "
          f"{screen_top()} refitted per configuration")

    # The detector headings: every detector, plus NO_DETECTOR when a
    # score-free cleanup runs (constant_lift; 2026-10-06).
    free_cleanups = tuple(c for c in CLEANUPS if c in SCORE_FREE_CLEANUPS)
    det_keys = list(detectors) + ([NO_DETECTOR] if free_cleanups else [])

    def _cleanups_under(det):
        # Score-free cleanups only under NO_DETECTOR, the rest only under a
        # real detector.
        return (free_cleanups if det == NO_DETECTOR else
                tuple(c for c in CLEANUPS if c not in SCORE_FREE_CLEANUPS))

    # {space: {detector: {flag_rate: {model: {cleanup: [per-fold score]}}}}}
    acc = {s: {d: {r: {m: {c: [] for c in ('no_cleaning',) + CLEANUPS}
                       for m in downstream_models} for r in flag_rates}
               for d in det_keys} for s in SPACES}
    # Which FOLDS produced a score, as a set of fold numbers per (space,
    # detector, flag rate). The loop reaches the same (space, detector, rate)
    # once per model, so a count would say two folds ran where one did.
    folds_ran = {s: {d: {r: set() for r in flag_rates} for d in det_keys}
                 for s in SPACES}
    # The shuffled-score control, per seed. Kept out of `acc` so that every
    # count in per_detector and in the summary still means the real detector.
    ctrl_cleanups = tuple(c for c in CLEANUPS if c not in SCORE_FREE_CLEANUPS)
    acc_ctrl = {s: {d: {k: {r: {m: {c: [] for c in ctrl_cleanups}
                                for m in downstream_models}
                            for r in flag_rates}
                        for k in shuffle_seeds}
                    for d in detectors} for s in SPACES}
    # One entry per (fold, configuration, model) that could not be built.
    config_failures = []
    fold_records = {}
    screen_log = []
    sim_unavailable = {}

    # ── Rows, the neighbour settings and the checkpoint ──
    #
    # ROWS. Every cleanup's fit on D' (the cleaned training fold) beside the
    # uncleaned fit on D, one d_vs_dprime row per (fold, configuration,
    # detector, setting, model), appended as each fold finishes. The rows of
    # the main table below are built from the numbers it already computed
    # -- no refit. When a neighbour list reached one, its similarity_setting
    # is the top screened setting's name and its method_setting carries the
    # neighbour count, the counting scheme and 'table=main'; 'none' when
    # nothing read neighbours.
    #
    # THE NEIGHBOUR SETTINGS (KNN_SEARCH_SPACE_RECORD). For each configuration,
    # every setting's neighbours are found among the training molecules
    # (scaffold-mates stay in the lists, axis 6), and every (setting, k,
    # scheme) is scored with no model fit on how well its neighbours predict a
    # held-back fifth of the training molecules (n07.screen_settings, inner=).
    # The top one (kept[0]) is the main table's neighbour setting
    # (2026-10-06). The top 3 are refitted into the rows file:
    # neighbor_residual reads their lists under the kept counting scheme, and
    # label_smooth / knn_smooth average their neighbours' labels under it. The
    # model is still fitted on the configuration's own features; only the
    # neighbours move.
    #
    # The rows file and the checkpoint carry RESULT_LAYOUT, so nothing written
    # by the whole-dataset design (before 2026-10-05) is read back as done.
    rows = sx.Rows(output_dir, EXP_FULL_NAME, dataset_name,
                   condition=row_condition)
    results['rows'] = rows.summary()
    ck = sx.FoldCheckpoint(
        output_dir, EXP_FULL_NAME, dataset_name,
        sx.run_signature(data, config, {
            'layout': RESULT_LAYOUT, 'collapsed': collapsed,
            'models': downstream_models,
            'n_folds': int(n_folds), 'flag_rates': flag_rates,
            'detectors': detectors, 'shuffle_seeds': shuffle_seeds,
            'cleanups': list(CLEANUPS), 'spaces': list(SPACES),
            'settings': [s.name for s in sim_settings],
            'screen_top': screen_top()}),
        condition=row_condition)
    if ck.state is not None:
        acc = ck.state['acc']
        folds_ran = ck.state['folds_ran']
        acc_ctrl = ck.state['acc_ctrl']
        config_failures = ck.state['config_failures']
        sim_unavailable = ck.state.get('sim_unavailable', {})
        fold_records = ck.state.get('fold_records', {})
        screen_log = ck.state.get('screen_log', [])

    def _cache_name(mname):
        # The name _evaluate_fold files a cached fit under, or None to fit it
        # directly (dnn: its fit is not reproducible, see noise_cache).
        return mname if mname in _CACHED_FIT_MODELS else None

    def _setting_label(cleanup, slot, rate):
        # The axis a cleanup is swept on and its value at this slot, as
        # setting_axis defines them.
        _axis, _vals = setting_axis(cleanup, is_clf)
        return f"{_axis}={rate if _axis == 'flag_rate' else _vals[slot]}"

    def _today_ident(method, cleanup, detector, slot, rate, model_input, fold,
                     top=None):
        # A row of the main table. A neighbour list reached it when the
        # detector reads one or the cleanup does; then its neighbours came
        # from the top screened setting `top` (2026-10-06).
        reads = (detector in nc.NEIGHBOUR_DETECTORS
                 or cleanup in sx.SMOOTHERS_READING_NEIGHBOURS)
        ms = f'detector={detector}|'
        sim = sx.NO_SIMILARITY
        if reads:
            if top and top.get('lists') is not None:
                ms += f"k={int(top['k'])}|scheme={top['scheme']}|"
                sim = top['setting']
            else:
                sim = 'not_run'
        ms += f'{_setting_label(cleanup, slot, rate)}|table=main'
        return sx.identity(
            dataset=dataset_name, experiment=EXP_FULL_NAME, method=method,
            method_setting=ms, similarity_setting=sim,
            model_input=model_input, noise_condition=ROW_CONDITION,
            fold=int(fold))

    def _screen_cell(X_tr_s, X_va_s, mname, entry, mi, g_tr, y_train, y_val,
                     train_idx, val_idx, smi_tr, smi_va, fold_i):
        """Neighbour lists for every setting on one configuration of one
        model, and the screen over (setting, k, scheme). Returns (nb, kept,
        top): top is kept[0] as {'setting', 'space', 'k', 'scheme', 'lists'
        (with the scheme on them)}, or {'reason'} when nothing was kept."""
        if not sim_settings:
            return {}, [], {'reason': 'this dataset has no neighbour settings'}
        sp_tr, sp_va, unav = fold_similarity_spaces(
            dataset_name, sim_settings, model_input_tr=X_tr_s,
            model_input_va=X_va_s, reps=reps, train_idx=train_idx,
            val_idx=val_idx, y_train=y_train, groups_train=g_tr,
            base_reps=(entry or {}).get('base') or [],
            recipes=(entry or {}).get('recipes') or {},
            augmentations=data.get('augmentations'), model_name=mname,
            model_factory=model_factories[mname], metric=metric,
            task_type=task_type)
        # Axis 6: one list per setting, scaffold-mates included, for the
        # detector and the smoothers alike.
        nb = fold_neighbours(dataset_name, sim_settings, sp_tr, sp_va,
                             y_train, smiles_train=smi_tr, smiles_val=smi_va,
                             is_clf=is_clf)
        for s in sim_settings:
            L = nb.get(s.name)
            if L is None or not L.get('available'):
                sim_unavailable.setdefault(s.name, {})[mi] = (
                    unav.get(s.space) or (L or {}).get('reason'))
        # DECIDED 2026-10-06 (Adelaide delegated): the screen predicts a
        # held-back fifth of the training molecules from the rest (inner=).
        # Without it the screen scores the training molecules themselves.
        kept, table = screen_settings(sim_settings, nb, y_train,
                                      metric=metric, is_clf=is_clf,
                                      y_val=y_val, inner={
                                          'dataset_name': dataset_name,
                                          'spaces_train': sp_tr,
                                          'y_train': y_train,
                                          'groups_train': g_tr,
                                          'smiles_train': smi_tr,
                                          'is_clf': is_clf})
        screen_log.append({
            'fold': int(fold_i), 'model_input': mi,
            'kept': [f'{s_.name}|k={k_}|{sch}' for s_, k_, sch in kept],
            'table': table})
        if not kept:
            return nb, kept, {'reason': 'the screen kept no (setting, k, '
                                        'scheme): no setting had usable '
                                        'neighbour lists'}
        s0, k0, sch0 = kept[0]
        return nb, kept, {'setting': s0.name, 'space': s0.space,
                          'k': int(k0), 'scheme': sch0,
                          'lists': dict(nb[s0.name], scheme=sch0)}

    def _sweep_cell(X_tr_s, X_va_s, mname, mi, g_tr, y_train, y_val, nb,
                    kept, scores_by_detector, targets, base_score, fold_i):
        """The screened neighbour settings on one configuration of one model,
        as rows.

        A (setting, k, scheme) reaches a (detector, cleanup) pair when the
        detector reads neighbours (neighbor_residual) or the cleanup does
        (label_smooth, knn_smooth); a pair it reaches neither way is the
        'none' row the main table already wrote. Since 2026-10-06 the
        counting scheme reaches the neighbour detectors too (the lists carry
        it), so every reached pair is scored per (setting, k, scheme)."""
        fac = {mname: model_factories[mname]}
        for s, k, scheme in kept:
            L = dict(nb[s.name], scheme=scheme)
            for detector in detectors:
                det_reads = detector in nc.NEIGHBOUR_DETECTORS
                if det_reads:
                    try:
                        scores_k = nc.cached_detector_scores(
                            dataset_name, X_tr_s, y_train, is_clf=is_clf,
                            detector=detector, groups=g_tr,
                            similarity_neighbours=L, similarity_k=k)
                    except Exception as exc:
                        print(f"      {detector} under {s.name} k={k} "
                              f"failed: {type(exc).__name__}: {exc}")
                        continue
                else:
                    scores_k = scores_by_detector.get(detector)
                if scores_k is None:
                    continue
                for cleanup in CLEANUPS:
                    if cleanup in SCORE_FREE_CLEANUPS:
                        # Reads no detector score and no neighbour list, so
                        # its column is the one the main table already
                        # scored (2026-10-05).
                        continue
                    reads = sx.strategy_reads_setting(cleanup, s)
                    if not (det_reads or reads):
                        continue
                    _axis, _vals = setting_axis(cleanup, is_clf)
                    for slot, rate in enumerate(flag_rates):
                        if slot >= len(_vals):
                            continue
                        _strength = (None if _axis == 'flag_rate'
                                     else _vals[slot])
                        ms = (f'detector={detector}|k={int(k)}'
                              f'|scheme={scheme}'
                              f'|{_setting_label(cleanup, slot, rate)}')
                        ident = sx.identity(
                            dataset=dataset_name, experiment=EXP_FULL_NAME,
                            method=cleanup, method_setting=ms,
                            similarity_setting=s.name, model_input=mi,
                            noise_condition=ROW_CONDITION, fold=int(fold_i))
                        if rows.is_done(ident, [mname]) and not \
                                sx.controls_enabled():
                            continue
                        _nb = L if reads else None

                        def _clean(sc_, rate_, _c=cleanup, _s=_strength,
                                   _nbl=_nb, _k=(int(k) if reads else None)):
                            return cleaned_training_set(
                                _c, X_tr_s, y_train, sc_, rate_,
                                is_clf=is_clf, groups=g_tr, strength=_s,
                                targets=targets, dataset_name=dataset_name,
                                similarity_neighbours=_nbl, similarity_k=_k)
                        try:
                            cleaned = _clean(scores_k, rate)
                        except Exception as exc:
                            print(f"      {cleanup} under {s.name} k={k} "
                                  f"failed: {type(exc).__name__}: {exc}")
                            cleaned = None
                        _kw = dict(X_tr=X_tr_s, y_D=y_train, X_va=X_va_s,
                                   y_va=y_val, metric=metric, factories=fac,
                                   score_D={mname: base_score})
                        sx.dprime_rows(rows, dataset_name, ident, cleaned,
                                       **_kw)
                        sx.control_rows(
                            rows, dataset_name, ident, cleaned,
                            seed=sx.seed_of(fold_i, mi, mname, cleanup,
                                            detector, s.name, int(k), scheme,
                                            slot),
                            is_clf=is_clf, rerun=_clean, **_kw)

    t0 = time.time()
    for fold_i, train_idx, val_idx in folds:
        if ck.is_done(fold_i):
            print(f"    fold {fold_i}: already in the checkpoint, skipped")
            continue
        print(f"\n  Fold {fold_i} of {n_folds}: {len(train_idx)} training, "
              f"{len(val_idx)} held-out rows")
        y_train = labels[train_idx]
        y_val = labels[val_idx]
        g_tr = None if groups_arr is None else groups_arr[train_idx]
        smi_tr = sx.smiles_for(data, train_idx)
        smi_va = sx.smiles_for(data, val_idx)
        fold_rows = []
        frec = {'n_train': int(len(train_idx)), 'n_val': int(len(val_idx)),
                'models': {}}

        # ── This fold's configurations, from searches on its training rows ──
        # On every dataset but a collapsed one n18 runs on the rows as
        # loaded, so the per-fold searches are the dataset's standard ones.
        # herg_ic50's one-row-per-molecule searches are filed under
        # MOLECULE_MEAN_SUFFIX.
        try:
            entries = fold_configurations(data, dataset_name, fold_i=fold_i,
                                          train_idx=train_idx,
                                          n_folds=n_folds,
                                          condition_suffix=search_suffix)
        except Exception as exc:
            print(f"    per-fold searches failed: {type(exc).__name__}: {exc}")
            frec['error'] = (f'per-fold searches failed: '
                             f'{type(exc).__name__}: {exc}')
            fold_records[str(fold_i)] = frec
            continue

        # {space: [(X_train, X_val, model, entry, what it is)]}, one cell per
        # model holding that configuration. Every matrix is built from this
        # fold's training molecules at their UNCLEANED labels, and every
        # cleanup shares it.
        fold_spaces = {sp: [] for sp in SPACES}
        for mname in downstream_models:
            entry = entries.get(mname)
            ins, unav, names = fold_model_inputs(
                data, dataset_name, mname, entry, train_idx=train_idx,
                val_idx=val_idx, y_train=y_train,
                factory=model_factories[mname], metric=metric)
            rec = configuration_record(entry, names)
            rec['configurations_unavailable'] = {
                sp: why for sp, why in unav.items() if sp in SPACES}
            rec['n_features'] = {sp: int(np.shape(X[0])[1])
                                 for sp, X in ins.items()}
            rec['metric'] = metric_name
            rec['lower_is_better'] = bool(lower_is_better)
            rec['held_out_score_no_cleaning'] = {}
            rec['main_neighbour_setting'] = {}
            rec['main_neighbour_not_run'] = {}
            frec['models'][mname] = rec
            print(f"    {mname}: base {rec.get('base')}, hybrids "
                  f"{[h.get('label') for h in rec.get('hybrids') or []]}")
            for sp in SPACES:
                if sp in ins:
                    fold_spaces[sp].append((ins[sp][0], ins[sp][1], mname,
                                            entry, names.get(sp, sp)))
                else:
                    config_failures.append({
                        'fold': int(fold_i), 'space': sp, 'model': mname,
                        'why': rec['configurations_unavailable'].get(
                            sp, 'not built')})

        # One pass per (configuration, model).
        for space in SPACES:
            for X_tr_s, X_va_s, mname, entry, cell_label in fold_spaces[space]:
                # ⛔ THE THREE MODEL-BASED CORRECTIONS' TARGETS, ONCE PER
                # (FOLD, FEATURE MATRIX). Each depends on the features, the
                # labels and the groups and on nothing else -- not the
                # suspicion score, not the strength. Left in
                # cleaned_training_set they were refitted for every setting,
                # and the Gaussian process is cubic in the training rows with
                # 5 inner folds per call.
                _targets = correction_targets(
                    X_tr_s, y_train, groups=g_tr,
                    want=targets_needed(CLEANUPS))

                # The score to beat, ON THIS MATRIX. One fit per (fold,
                # configuration, model), shared by every cleanup, every flag
                # rate and every detector, so the comparison is paired.
                sc = _evaluate_fold(
                    X_tr_s, X_va_s, y_train, y_val, 'no_noise', 0.0,
                    model_factories[mname], metric_fn, is_clf=is_clf,
                    metric_name=metric_name, dataset_name=dataset_name,
                    precomputed_y_tr=y_train,
                    cache_dataset=dataset_name,
                    cache_model_name=_cache_name(mname))
                base_score = float(sc) if sc is not None else float('nan')
                _mrec = frec['models'][mname]
                _mrec['held_out_score_no_cleaning'][space] = base_score
                _mi = f'{mname}:{space}'

                # DECIDED 2026-10-06 (Adelaide delegated): the main table's
                # neighbours are this (fold, configuration)'s top screened
                # setting (kept[0]: space and distance, k, counting scheme).
                # neighbor_residual reads its lists and label_smooth /
                # knn_smooth average its neighbours' labels. Until
                # 2026-10-06 both found 9 or 10 Euclidean neighbours in the
                # configuration's own features, each counting equally, and
                # the screened settings reached the rows file only.
                nb, kept, top = _screen_cell(
                    X_tr_s, X_va_s, mname, entry, _mi, g_tr, y_train, y_val,
                    train_idx, val_idx, smi_tr, smi_va, fold_i)
                if top.get('lists') is not None:
                    _mrec['main_neighbour_setting'][space] = {
                        'setting': top['setting'], 'space': top['space'],
                        'k': top['k'], 'scheme': top['scheme']}
                else:
                    _mrec['main_neighbour_not_run'][space] = top.get('reason')

                def _cleaned_score(cleanup, rate, slot, scores,
                                   _X=X_tr_s, _V=X_va_s, _m=mname,
                                   _t=_targets, _top=top):
                    """Held-out score after `cleanup` driven by `scores`, or
                    None when this cleanup has no setting at this slot. nan
                    when it ran and produced no score; then the reason is
                    stored under fold_records[fold].models[model]
                    .cleanups_declined[configuration]['<cleanup>|<rate>']
                    (A19, 2026-10-08)."""
                    def _why(reason, _cleanup=cleanup, _rate=rate):
                        (_mrec.setdefault('cleanups_declined', {})
                         .setdefault(space, {}))[
                             f'{_cleanup}|{_rate}'] = reason
                    _axis, _vals = setting_axis(cleanup, is_clf)
                    if slot >= len(_vals):
                        return None
                    if (cleanup in WEIGHTING_CLEANUPS and
                            not nc.fit_takes_sample_weight(
                                model_factories[_m])):
                        # 2026-10-07: gp takes no per-molecule weight.
                        _mrec.setdefault('weights_not_run', {})[cleanup] = (
                            nc.WEIGHTS_NOT_RUN)
                        return float('nan')
                    _strength = None if _axis == 'flag_rate' else _vals[slot]
                    if cleanup in MAIN_TABLE_NEIGHBOUR_CLEANUPS:
                        # Neighbours from the top screened setting, with its
                        # k and counting scheme (the lists carry the scheme).
                        if _top.get('lists') is None or scores is None:
                            _why('no neighbour setting was kept on this '
                                 'configuration' if _top.get('lists') is None
                                 else 'the detector gave no suspicion score')
                            return float('nan')
                        clear_decline_reason()
                        try:
                            cleaned = cleaned_training_set(
                                cleanup, _X, y_train, scores, rate,
                                is_clf=is_clf, groups=g_tr,
                                strength=_strength, targets=_t,
                                dataset_name=dataset_name,
                                similarity_neighbours=_top['lists'],
                                similarity_k=_top['k'])
                        except Exception as exc:
                            print(f"      {cleanup} (top setting "
                                  f"{_top.get('setting')}) failed: "
                                  f"{type(exc).__name__}: {exc}")
                            _why(f'raised {type(exc).__name__}: {exc}')
                            return float('nan')
                        if cleaned is None:
                            _why(last_decline_reason() or 'declined')
                            return float('nan')
                        return sx.fit_score(
                            dataset_name, _m, model_factories[_m],
                            cleaned['X'], cleaned['y'], _V, y_val, metric,
                            sample_weight=cleaned['sample_weight'])
                    clear_decline_reason()
                    s_ = _evaluate_fold(
                        _X, _V, y_train, y_val, cleanup, rate,
                        model_factories[_m], metric_fn,
                        is_clf=is_clf, metric_name=metric_name,
                        dataset_name=dataset_name,
                        precomputed_y_tr=y_train, noise_scores=scores,
                        strength=_strength, targets=_t,
                        # ⛔ REQUIRED BY THE MODEL-BASED CORRECTIONS. Each fits
                        # a model on part of the training rows and reads the
                        # target off the rest. Without the scaffold groups
                        # that inner split is random, so a molecule's own
                        # scaffold family trains the model that says what its
                        # label should have been.
                        train_groups=g_tr, cache_dataset=dataset_name,
                        cache_model_name=_cache_name(_m))
                    if s_ is None:
                        _why(last_decline_reason() or (
                            'no score: the detector gave no suspicion score '
                            'or the fit raised (printed above)'))
                    return float(s_) if s_ is not None else float('nan')

                # DECIDED 2026-10-06 (Adelaide delegated): a score-free
                # cleanup (constant_lift) reads no detector score, so it is
                # scored once per (fold, configuration, model) under its own
                # heading NO_DETECTOR, at the first flag-rate slot. Until
                # 2026-10-06 it was scored under the first detector, and not
                # at all in a fold where that detector failed.
                if free_cleanups:
                    _r0 = flag_rates[0]
                    folds_ran[space][NO_DETECTOR][_r0].add(int(fold_i))
                    acc[space][NO_DETECTOR][_r0][mname]['no_cleaning'].append(
                        base_score)
                    for cleanup in free_cleanups:
                        s_ = _cleaned_score(cleanup, _r0, 0, None)
                        if s_ is None:
                            s_ = float('nan')
                        acc[space][NO_DETECTOR][_r0][mname][cleanup].append(s_)
                        fold_rows.append(sx.row(
                            _today_ident(cleanup, cleanup, NO_DETECTOR, 0,
                                         _r0, _mi, fold_i),
                            model=mname, metric=metric,
                            score_D=base_score, score_Dprime=s_))

                # The main table's scores per detector, which the neighbour
                # settings below reuse for a detector that reads no
                # neighbours (cross_val).
                scores_by_detector = {}
                for det_i, detector in enumerate(detectors):
                    # Re-run per configuration. The detector reads features,
                    # so the molecules it flags in a hybrid are not the ones it
                    # flags in a single representation.
                    if detector in nc.NEIGHBOUR_DETECTORS:
                        if top.get('lists') is None:
                            _mrec['main_neighbour_not_run'].setdefault(
                                f'{space}|{detector}', top.get('reason'))
                            continue
                        try:
                            scores = nc.cached_detector_scores(
                                dataset_name, X_tr_s, y_train, is_clf=is_clf,
                                detector=detector, groups=g_tr,
                                similarity_neighbours=top['lists'],
                                similarity_k=top['k'])
                        except Exception as exc:
                            print(f"      {detector} under {top['setting']} "
                                  f"failed: {type(exc).__name__}: {exc}")
                            scores = None
                        scores = (None if scores is None
                                  else np.asarray(scores, dtype=float))
                    else:
                        scores = _suspicion(detector, X_tr_s, y_train, is_clf,
                                            groups=g_tr, dataset=dataset_name)
                    if scores is None:
                        continue
                    scores_by_detector[detector] = scores
                    # The control: the same scores, dealt at random. Seeded on
                    # (seed, fold, configuration, detector) so a re-run
                    # reproduces it.
                    for seed in shuffle_seeds:
                        _perm = np.random.default_rng(
                            [int(seed), int(fold_i), SPACES.index(space),
                             det_i]).permutation(len(scores))
                        _shuf = np.asarray(scores)[_perm]
                        for slot, rate in enumerate(flag_rates):
                            for cleanup in ctrl_cleanups:
                                s_ = _cleaned_score(cleanup, rate, slot, _shuf)
                                if s_ is None:
                                    continue
                                acc_ctrl[space][detector][seed][rate][mname][
                                    cleanup].append(s_)
                                fold_rows.append(sx.row(
                                    _today_ident(
                                        f'{cleanup}__score_shuffled_seed{seed}',
                                        cleanup, detector, slot, rate, _mi,
                                        fold_i, top=top),
                                    model=mname, metric=metric,
                                    score_D=base_score, score_Dprime=s_))
                    for slot, rate in enumerate(flag_rates):
                        folds_ran[space][detector][rate].add(int(fold_i))
                        acc[space][detector][rate][mname]['no_cleaning'].append(
                            base_score)
                        for cleanup in _cleanups_under(detector):
                            # Each cleanup is swept over the axis it actually
                            # reads; see setting_axis.
                            s_ = _cleaned_score(cleanup, rate, slot, scores)
                            if s_ is None:
                                # Fewer settings than slots: scoring it here
                                # would be the same column fitted again.
                                continue
                            acc[space][detector][rate][mname][cleanup].append(
                                s_)
                            fold_rows.append(sx.row(
                                _today_ident(cleanup, cleanup, detector, slot,
                                             rate, _mi, fold_i, top=top),
                                model=mname, metric=metric,
                                score_D=base_score, score_Dprime=s_))

                # ── The screened neighbour settings on this configuration ──
                if kept:
                    _sweep_cell(X_tr_s, X_va_s, mname, _mi, g_tr, y_train,
                                y_val, nb, kept, scores_by_detector, _targets,
                                base_score, fold_i)

        rows.write(fold_rows)
        fold_records[str(fold_i)] = frec
        ck.save(fold_i, {'acc': acc, 'folds_ran': folds_ran,
                         'acc_ctrl': acc_ctrl,
                         'config_failures': config_failures,
                         'sim_unavailable': sim_unavailable,
                         'fold_records': fold_records,
                         'screen_log': screen_log})
        print(f"    fold {fold_i}: [{time.time() - t0:.0f}s]")

    spaces = tuple(s for s in SPACES
                   if any(folds_ran[s][d][r] for d in det_keys
                          for r in flag_rates))
    results['spaces_run'] = list(spaces)
    results['fold_records'] = convert_numpy(fold_records)
    # A19 (2026-10-08): every (fold, model, configuration, cleanup, rate) whose
    # cleanup returned no score, with the reason that applied. The per-unit
    # reasons are in fold_records[fold].models[model].cleanups_declined.
    _declined = {}
    for _f, _fr in fold_records.items():
        for _m, _mr in (_fr.get('models') or {}).items():
            for _sp, _by in (_mr.get('cleanups_declined') or {}).items():
                for _cr, _why in _by.items():
                    (_declined.setdefault(_cr.split('|', 1)[0], {})
                     [f'fold{_f}|{_m}@{_sp}|{_cr.split("|", 1)[1]}']) = _why
    results['cleanups_declined'] = _declined
    results['screen'] = convert_numpy(screen_log)
    results['rows'] = rows.summary()
    results['similarity'] = similarity_record(
        sim_settings, rows, knn_group=knn_grp,
        schemes_swept=list(sweep_schemes()), screen_top=screen_top(),
        screen_scored_on=SCREEN_SCORED_ON,
        main_table_similarity_setting=(
            'per (fold, configuration): the top screened setting, named in '
            'fold_records[fold].models[model].main_neighbour_setting'))
    if sim_unavailable:
        results['similarity_settings_unavailable'] = sim_unavailable
    if config_failures:
        # Named rather than left as a shorter list of folds. Those folds still
        # carry every number from the configurations that did build.
        results['configurations_unavailable'] = config_failures
        print(f"  {len(config_failures)} (fold, configuration, model) "
              f"triples could not be built; the rest keep their numbers")

    # ── Fold win counts, which are the headline ──
    for space in spaces:
        space_block = {}
        for detector in det_keys:
            space_block[detector] = {}
            # NO_DETECTOR's cleanups are scored at the first slot only.
            for rate in (flag_rates[:1] if detector == NO_DETECTOR
                         else flag_rates):
                block = {'n_folds_ran': len(folds_ran[space][detector][rate]),
                         'per_model': {}}
                for mname in downstream_models:
                    cell = acc[space][detector][rate][mname]
                    base = np.asarray(cell['no_cleaning'], dtype=float)
                    row = {'no_cleaning_mean': float(np.nanmean(base))
                           if base.size else float('nan')}
                    for cleanup in _cleanups_under(detector):
                        _axis, _vals = setting_axis(cleanup, is_clf)
                        _slot = flag_rates.index(rate)
                        if _slot >= len(_vals):
                            # A cleanup that reads neither axis was scored at
                            # the first slot only. A duplicate row would be
                            # counted as a second setting by the summary.
                            continue
                        got = np.asarray(cell[cleanup], dtype=float)
                        outcomes = [_won(c, b, lower_is_better)
                                    for c, b in zip(got, base)]
                        n_won = sum(o == 'won' for o in outcomes)
                        n_lost = sum(o == 'lost' for o in outcomes)
                        n_tied = sum(o == 'tied' for o in outcomes)
                        diff = np.array(
                            [(b - c) if lower_is_better else (c - b)
                             for c, b in zip(got, base)
                             if np.isfinite(c) and np.isfinite(b)], dtype=float)
                        row[cleanup] = {
                            # Which knob this row's setting is, and its value.
                            'swept_axis': _axis,
                            'setting': (rate if _axis == 'flag_rate'
                                        else _vals[_slot]),
                            # The headline. Folds the cleaned model beat the
                            # uncleaned one on, out of the folds both scored.
                            # A tie is its own count (dirty-run bug, n18: ties
                            # were counted as losses until 2026-10-05).
                            'folds_won': int(n_won),
                            'folds_lost': int(n_lost),
                            'folds_tied': int(n_tied),
                            'folds_scored': int(n_won + n_lost + n_tied),
                            # Beside it, never instead of it.
                            'mean_gain': float(diff.mean()) if diff.size else float('nan'),
                            'fold_gains': [float(x) for x in diff],
                            'spread_over_folds': fold_spread(list(diff)),
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
                                'shuffled_folds_tied': int(np.sum(gain[ok] == 0)),
                                # Against the real detector: positive means
                                # the detector's own ranking did better.
                                'real_minus_shuffled': (
                                    float(r_minus_s[both].mean())
                                    if both.any() else float('nan')),
                                'real_minus_shuffled_per_fold': [
                                    float(x) for x in r_minus_s[both]],
                                'folds_real_beat_shuffled': int(
                                    np.sum(r_minus_s[both] > 0)),
                                'folds_real_tied_shuffled': int(
                                    np.sum(r_minus_s[both] == 0)),
                                'folds_scored': int(both.sum()),
                            }
                        if per_seed:
                            by_cleanup[cleanup] = per_seed
                    by_model[mname] = by_cleanup
                ctrl_block[detector][str(rate)] = convert_numpy(by_model)
        results['per_space'][space]['shuffled_control'] = ctrl_block

    results['summary'] = _summarise(results, downstream_models, flag_rates,
                                    det_keys, spaces, metric_name,
                                    cleanups=CLEANUPS)
    results['summary']['shuffled_control'] = _summarise_control(
        results, spaces, shuffle_seeds, ctrl_cleanups)
    _print_summary(results, metric_name)

    # (results, output_dir, experiment_id, dataset) — see the note in n17; the
    # arguments were rotated here the same way and with the same two effects.
    save_experiment_results(results, output_dir, 'n18_clean_the_real_labels',
                            dataset_name)
    # The result file holds every fold of the split now, so the per-fold
    # snapshot is spent. A run of some folds only (KIRBY_NOISE_ONLY_FOLDS)
    # keeps it, so the other folds add to it later.
    if len(folds) == int(n_folds):
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
                            'folds_lost': cell.get('folds_lost'),
                            'folds_tied': cell.get('folds_tied', 0),
                            'folds_scored': cell['folds_scored'],
                            'mean_gain': cell['mean_gain'],
                        })
            if not rows:
                per_cleanup[cleanup] = {'settings_scored': 0}
                continue
            # Ties set aside: most folds means more won than lost
            # (2026-10-05; a tie was a loss before).
            majority = [x for x in rows
                        if x['folds_won'] > (x['folds_lost'] if
                                             x['folds_lost'] is not None else
                                             x['folds_scored'] - x['folds_won'])]
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
    if 'hybrid' in spaces and any(s in spaces for s in ('base_1', 'base_2')):
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
                            # Ties set aside, as in _summarise.
                            _lost = (row['folds_scored']
                                     - row['folds_real_beat_shuffled']
                                     - row.get('folds_real_tied_shuffled', 0))
                            if row['folds_real_beat_shuffled'] > _lost:
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
                  f"{b['folds_won']} of {b['folds_scored']} folds "
                  f"({b.get('folds_tied', 0)} tied), "
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
