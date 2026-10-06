#!/usr/bin/env python3
"""N17: Does cleaning help when the noise is REAL, not injected?

Paper: Noise.

THE QUESTION. Every other experiment in this suite corrupts clean labels on
purpose and then asks whether a detector can find what it corrupted. That is a
controlled measurement and it is worth having, but a reader can fairly say the
noise was drawn from a distribution the same script chose. This experiment asks
the same question where nobody chose the noise: the labels disagree because
different laboratories measured the same thing and got different numbers.

WHERE THE THREE PIECES COME FROM. A removal claim needs three quantities from
three different places, or it grades itself.

  a per-molecule error figure   the spread of one entity's repeat measurements
  a training label              ONE of those measurements, chosen at random
  a grading label               the mean of the measurements NOT chosen

Nothing here is injected and nothing is drawn from a distribution. The training
label is one laboratory's number, with whatever that laboratory got wrong still
in it. The grading label is what the other laboratories agreed on.

WHAT AN ENTITY IS. It depends on the dataset, and the loader decides, not this
file. On a drug-target set an entity is one (drug, protein) pair, so repeat rows
are the same pair measured more than once. `_ground_truth_providers
.build_replicate_groups` builds the ids and this experiment does not second-guess
it. A dataset whose loader exposes no repeat structure cannot run here and says
so rather than inventing one.

HOW THE SPLIT WORKS, AND WHY NO ENTITY IS ON BOTH SIDES. One row per entity, then
ordinary grouped cross-validation over those rows. In a fold, the training
entities contribute their single-measurement label and the validation entities
are graded against their consensus label. An entity is never in both, so the
single measurement an entity trained on is never the thing it is graded against.

Only entities with MIN_MEASUREMENTS or more can be graded: below three there is
no consensus left once one measurement is held out. Entities measured once or
twice still train, at face value, and are never graded. That is deliberate --
throwing them out would shrink the training set for a reason that has nothing to
do with the question.

THE WAYS OF CHOOSING WHAT TO TRAIN ON. All of them are graded on exactly the
same validation entities, against exactly the same consensus labels. The only
thing that differs is which training entities survive, with what label, and at
what row weight.

  no_cleaning  every training entity, untouched. The thing to beat.
  the seven    the corrections n07 measures, applied to the entities THIS
  corrections  experiment's detector flags. `remove` takes them out of training;
               `downweight` gives every entity a row weight; `label_smooth`,
               `knn_smooth`, `oof_smooth` and `gp_smooth` move a label toward a
               target; `constant_lift` raises the labels at the ceiling by one
               constant, and declines where the labels show no ceiling. They
               are n07's, not this file's -- see CORRECTIONS.
  ceiling      the entities the ranking quantity ranks worst are removed. It uses
               the answer, so no detector can beat it. It says how much is on the
               table before any detector is involved.
  floor        the same NUMBER of entities, chosen at random FROM THE SAME
               ENTITIES THE CEILING COULD HAVE PICKED. It says how much of any
               gain is just from training on fewer entities.

Which entities the ceiling can reach follows the quantity it ranks on, and its
floor follows it -- see SPREAD_SCORE. The ceiling and its floor drew from
different sets until 2026-09-16 and the difference between them was partly that.

A cleaning result is a correction beating `no_cleaning`. `remove` is read on top
of that against the distance between `floor` and `ceiling`, because it is the one
correction that trains on fewer entities: beating `no_cleaning` while sitting at
`floor_any` is not a removal result, it is a smaller training set. The other six
train on every entity, so a removal floor is not their control and they are read
against `no_cleaning` alone.

WHERE EVERY ARM IS MEASURED. Since 2026-10-05 (chat 5, HANDOFF_EXPERIMENT_
ALIGNMENT section 2.4): per outer fold, per each of the dataset's two decided
models, in that model's 2 best single representations and its 2 best hybrids,
all four ranked by searches on that fold's training entities
(noise_helpers.fold_configurations). 2 models x 4 = 8 fitted configurations.
Before that, every loaded representation and up to six hybrids per model were
fitted, with five models, and every hybrid recipe came from a search over the
whole dataset, held-out entities included.

THE EXPERIMENT'S OWN ROWS. One row per entity (entity_data). The folds, the
per-fold searches, the hybrid builds and the fits all read these rows, so all
of an entity's measurements are on one side of every fold. On herg_ic50 the
loader keeps one row per measurement, 12,008 rows for 9,802 molecules; the old
search ran on those rows (dirty-run bug 11). The searches see each training
entity's mean over all its measurements (SEARCH_LABELS).

THE HYBRID IS BUILT ON THE TRAINING ENTITIES OF A FOLD AND NOTHING ELSE. Its
recipe comes from the fold's own search and its features are ranked and its
budget split against those entities' single-measurement training labels --
the same numbers `no_cleaning` fits on. The consensus labels the held-out
entities are graded against never reach it, in any fold, under any arm.

It is built ONCE per (draw, fold, model) and all the arms share it, so the
only thing differing between two arms is which training entities survive and
with what label. Whether a recipe searched after the removal would do better
is a different question and it is n20's.

NEIGHBOUR SETTINGS. The detector and two of the corrections read neighbours.
Per (draw, fold, configuration) every setting of the kNN record
(KNN_SEARCH_SPACE_RECORD.md) is screened with no model fit: each setting
predicts a held-back fifth of the training entities from the rest
(n07.screen_settings with inner=). Since 2026-10-06 (Adelaide delegated) the
main table (per_draw, summary) takes the detector's neighbours and
label_smooth's and knn_smooth's from the top screened setting (kept[0]: the
neighbour space and distance, k, and how much each neighbour counts). Before
that they used 9-10 Euclidean neighbours in the configuration's own features,
each counting equally. The setting used is recorded per (draw, fold, model,
configuration) under configurations_by_fold ... main_table_neighbour_setting.
Every correction is also refitted under each of the top 3 settings, into the
rows file only. Forest leaves are fitted on each configuration's own matrix,
as in n07.
The distance hybrid and the similarity hybrid are neighbour spaces there,
never fitted configurations.

WHAT THIS DOES NOT CLAIM. It says nothing about which detector is best in
general -- that is n11, under injected noise where the answer is known. It says
nothing about datasets whose loaders collapse their repeats before anything sees
them, which as of 2026-09-12 is most of them (NOISE_AUDIT_2026_09_10.md 11.1).

WHAT IT COSTS. Eleven arms (no_cleaning, seven corrections, ceiling, two
floors) by one fit each, in 8 configurations, on every fold of every draw:
at five folds and three draws, 1,320 downstream fits, plus the per-fold
searches (three per fold: general, distance, similarity; the same for every
draw) and the hybrid builds. The neighbour sweep adds up to 3 kept settings x
7 corrections refits per configuration. Not measured.

Usage:
    python n17_replicate_removal.py --dataset bindingdb_kd --mode full
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
)
from experiments.decided_grid import decided_models
from experiments.noise._ground_truth_providers import build_replicate_groups
# 2026-10-05 (chat 5): folds, the 8 fitted configurations per fold and the
# spread over folds, shared by every noise experiment. They replace the
# whole-dataset pool lookup (get_best_search_for_downstream,
# pool_recipe_per_downstream_model, pool_recipes_by_family,
# noise_base_reps_by_model) and cv_split.
from experiments.noise.noise_helpers import (
    noise_outer_folds, fold_spread, fold_configurations, fold_model_inputs,
    configuration_record, CONFIGURATION_SPACES)
# The seven corrections, defined once for the whole suite. n17 held two of its
# own inline until 2026-09-21; see CORRECTIONS. The neighbour settings, their
# screen and the neighbour spaces are n07's too since 2026-10-05.
from experiments.noise.n07_remediation import (
    cleaned_training_set, correction_targets, targets_needed,
    sweep_settings as n07_sweep_settings, sweep_ks as n07_sweep_ks,
    sweep_schemes as n07_sweep_schemes, screen_top as n07_screen_top,
    screen_settings as n07_screen_settings,
    fold_similarity_spaces as n07_fold_similarity_spaces,
    fold_neighbours as n07_fold_neighbours,
    similarity_record as n07_similarity_record)
from kirby.utils.noise_estimation import get_noise_estimator
from experiments.noise.similarity_space import FOREST_SPACE
# The D-versus-D' rows, the controls and the per-fold checkpoint, added
# 2026-09-25 (plan sections 1-3). See _sim_axis_n15_n22.
from experiments.noise import noise_cache as nc
from experiments.noise import _sim_axis_n15_n22 as sx

# The name the rows file, the checkpoint and the result file are filed under.
EXP_FULL_NAME = 'n17_replicate_removal'

# What this file writes since chat 5 (2026-10-05, HANDOFF_EXPERIMENT_ALIGNMENT
# section 2.4). It names the rows file and the checkpoint and is in the
# checkpoint's signature, so a file from the whole-dataset design is never
# read back as this one. The 'b' (2026-10-05): the neighbour screen moved
# from one per model to one per configuration, so forest leaves are screened.
# 2026-10-06: the screen predicts a held-back fifth of the training entities,
# the main table's detector, label_smooth and knn_smooth read the top screened
# setting, and the detector reads the counting scheme, so the numbers in
# per_draw and in the rows file changed.
RESULT_LAYOUT = 'fold_aligned_2026-10-06'

# The per-fold searches run on this experiment's own rows -- one row per
# entity, not the loader's rows -- so they are filed apart from the dataset's
# standard per-fold searches under this suffix (fold_configurations'
# condition_suffix). Specific to n17: another experiment that collapses
# herg_ic50 to one row per molecule may order its rows differently.
SEARCH_SUFFIX = 'n17_one_row_per_entity_mean'

# Which labels the per-fold searches see. 2026-10-05 (chat 5).
# DECIDED 2026-10-06 (Adelaide delegated): the searches see each training
# entity's mean over ALL its measurements. There is no clean label here; this
# is the nearest thing to the clean labels the other noise experiments search
# on. It does not change between draws, so each fold has one set of searches
# for all three draws. No held-out entity reaches it.
SEARCH_LABELS = ("each training entity's mean over all of its measurements "
                 "(entity_data); the same for every draw")

# The noise_condition column of every row this file writes: nothing injected,
# the laboratories disagree.
ROW_CONDITION = 'real_replicate_disagreement'

# Below this an entity cannot be graded: hold one measurement out of two and the
# "consensus of the rest" is a single measurement, which is not a consensus.
MIN_MEASUREMENTS = 3

# `true_spread` (below) is defined from 2 measurements, one lower than
# MIN_MEASUREMENTS. Checked on herg_ic50, 2026-09-18: of the entities the
# ceiling removes, 70% (689 of 980) carry a spread computed from exactly 2
# measurements. Two alternatives were tried and rejected: requiring 3+
# measurements before an entity is eligible drops the ceiling's reachable
# population from 1,218 entities to 405; shrinking each entity's variance
# toward the population mean (weighted by measurement count) flagged MORE
# 2-measurement entities, not fewer, because entities measured more often
# turned out to disagree more, which pulled the population mean up and
# dragged every 2-measurement entity's shrunk estimate up with it. Decision:
# 2 measurements is an acceptable floor given the data this runs on. Do not
# re-raise this without a new number attached.

# Share of training entities the detector flags, and the share the ceiling
# removes. Held equal on purpose: `drop`, `ceiling` and `floor` must remove the
# same NUMBER of entities, or the comparison measures training-set size.
FLAG_FRACTION = 0.10

# The detector. neighbour_residual scores an entity by how far its label sits
# from its nearest neighbours' labels. It is the quantity the literature names
# and it needs no model fit of its own.
DETECTOR = 'neighbor_residual'
DETECTOR_K = 10

# WHICH QUANTITY THE CEILING RANKS ENTITIES BY. Swept from the caller.
#
# 'measurement_spread' is what this file ranked on from the day it was written:
# the standard deviation of ALL of an entity's repeat measurements, which is how
# far the laboratories disagreed with each other.
#
# 'training_label_error' is how far THIS ENTITY'S TRAINING LABEL sits from the
# mean of its other measurements, in the label's own units. That is the quantity
# the arm is supposed to be an oracle for: an entity whose laboratories disagree
# wildly, but whose randomly chosen training measurement happens to land on the
# consensus, carries a correct training label and removing it removes a correct
# label.
#
# The two differ most on the entities measured EXACTLY TWICE. Their training
# label is the mean of both measurements -- `build_entities` holds nothing out
# below MIN_MEASUREMENTS -- so under 'training_label_error' their label sits 0
# from the mean of the rest and they can never be ranked. Under
# 'measurement_spread' they carry the full disagreement of their two
# laboratories and the ceiling takes them first.
#
# Which entities each one can reach is therefore different, and the floor it is
# read against follows it: 'measurement_spread' reaches every entity measured
# twice or more, 'training_label_error' reaches only the entities measured
# MIN_MEASUREMENTS times or more.
SPREAD_SCORE = 'measurement_spread'
SPREAD_SCORES = ('measurement_spread', 'training_label_error')

# THE SMALLEST FLOOR-TO-CEILING DISTANCE THIS FILE WILL DIVIDE BY. Added
# 2026-09-23.
#
# `removal_position` divides a correction's gain by the distance between
# `floor_measured` and `ceiling`. Until today the only condition on that
# distance was that it be above zero: the test at `_summarise` was
# `span > 0`. On herg_ic50 at a removal share of 0.01 the distance came back
# +0.0071 rmse against a no-cleaning score of 0.7399 rmse, which is 0.96% of
# it, and `remove` and `repair` were divided by it and printed at -2.003 and
# -2.340 of the way from floor to ceiling. Those two numbers say a correction
# landed two whole distances below a distance of seven thousandths of an rmse
# unit. Nothing in the data supports reading them.
#
# n06, n07, n11, n12 and n21 each refuse their own denominator below 2% of the
# score it is measured against, under the name GAP_MIN_FRAC. Their denominator
# is the clean-to-corrupted gap and this one is a floor-to-ceiling distance, so
# the two are not the same quantity and the number is carried across by choice,
# not because anything measured it here. It is a separate name for that reason.
SPAN_MIN_FRAC = 0.02

# TWO floors, because there are two things to control for and one floor cannot be
# the control for both. Added 2026-09-16 after the first floor was fixed and the
# fix removed a control instead of repairing one.
#
# `ceiling` can only reach entities measured twice or more, because only those have
# a disagreement to rank. So its control must draw from exactly those: that is
# `floor_measured`.
#
# `drop` and `repair` pick with a detector that reads features and labels, so they
# can reach ANY training entity, including the ones measured once. Their control
# must draw from all of training: that is `floor_any`.
#
# Comparing `drop` against `floor_measured` would compare a pick from everything
# against a pick from a small, well-studied minority, and most of the difference
# would be which minority.
#
# ⛔ THE CORRECTION AXIS IS n07's, NOT THIS FILE'S OWN. Changed 2026-09-21.
#
# Until today this file wrote its own two corrections inline: `drop` removed the
# flagged entities and `repair` overwrote their labels with the mean of their ten
# nearest neighbours. Removal is one correction of the seven the suite measures,
# and n07 ranked it FOURTH of the seven at 50.0% of the clean-to-corrupted gap
# recovered, behind quantile_clip at 72.2% and knn_smooth at 70.4%. So the one
# experiment in the suite where the noise is nobody's choice was trying the
# fourth-best correction and one other, and calling the answer negative.
#
# Every correction below is n07's `cleaned_training_set`, called with this
# experiment's own detector scores and flag fraction. Nothing is reimplemented
# here. The two names that left:
#
#   drop    -> `remove`.     n17 removed exactly n_flag entities by rank; n07
#                            cuts at the (1 - flag_fraction) quantile of the
#                            same scores, so with no tied score it removes the
#                            same entities.
#   repair  -> `knn_smooth`. n17 averaged the 10 nearest training entities; n07
#                            averages 9, because it asks for min(10, n-1)
#                            neighbours and then drops the entity itself.
#
# ⛔ WHICH CORRECTIONS REMOVE AN ENTITY AND WHICH DO NOT, because the summary
# reads each one against a different control. `remove` is the only one that
# trains on fewer entities. `downweight` gives every entity a row weight.
# `label_smooth`, `knn_smooth`, `oof_smooth` and `gp_smooth` move a label toward
# a target. `constant_lift` raises the labels at the ceiling by one constant
# (it declines where the labels show no ceiling). All six of those train on
# every entity, so a floor that removes entities is not their control
# and they are read against `no_cleaning` alone. Until today `repair` was read
# against `floor_any`, which removes entities, although `repair` removes none.
# 2026-10-05 (f36): quantile_clip cut, constant_lift added (raises the labels
# at the ceiling by one constant; declines where the labels show no ceiling).
CORRECTIONS = ('remove', 'downweight', 'label_smooth', 'knn_smooth',
               'oof_smooth', 'gp_smooth', 'constant_lift')

# The one correction that trains on fewer entities than `no_cleaning` does.
REMOVING_CORRECTIONS = ('remove',)

ARMS = (('no_cleaning',) + CORRECTIONS
        + ('ceiling', 'floor_measured', 'floor_any'))

# How many times the whole thing is repeated. WHICH measurement becomes the
# training label is a random choice, so one run is one draw of it. Three, to
# match every other experiment in this suite.
N_DRAWS = 3

# The configurations every arm is fitted in, per fold and per decided model
# (noise_helpers.CONFIGURATION_SPACES, 2026-10-05): the model's two best
# single representations and its two best hybrids on that fold's training
# entities. The summary compares each hybrid against each single
# representation.
HYBRID_SPACE = 'hybrid'
BASE_CONFIGURATIONS = ('base_1', 'base_2')
HYBRID_CONFIGURATIONS = ('hybrid', 'hybrid__general2')

# ⛔ WHY THE DISTANCE AND SIMILARITY HYBRIDS ARE NEIGHBOUR SPACES HERE.
# The corrections pick which entities to remove or relabel with DETECTOR,
# 'neighbor_residual' -- an entity scored against its nearest neighbours --
# and label_smooth / knn_smooth move a label toward its neighbours' labels.
# So n17 measures something with neighbours (handoff 2.3 item 1), and the top
# hybrid of the per-fold distance search ('hybrid__distance1') and of the
# per-fold similarity search ('hybrid_similarity') are among the neighbour
# settings the sweep screens. They are never fitted configurations. Until
# 2026-10-05 'hybrid_similarity' was a fitted space built from a
# whole-dataset recipe.


def build_entities(labels, replicate_groups, seed):
    """One row per entity: a training label, a grading label, and the truth.

    Returns a dict of arrays, all in the same order, one entry per entity:

      row_index       a row of the original data to read features from. Every row
                      of one entity carries the same features -- they are the same
                      drug against the same protein -- so the first one will do.
      n_measurements  how many times this entity was measured.
      train_label     ONE measurement, chosen at random, for entities that can be
                      graded. The mean of all of them for entities that cannot,
                      because there is nothing to hold out for them and the mean is
                      the best single number their rows support.
      grade_label     the mean of the measurements NOT chosen. NaN where the entity
                      has too few measurements to grade.
      gradeable       True where grade_label is a real number.
      true_spread     the standard deviation of ALL of this entity's measurements,
                      which is how far the laboratories actually disagreed. NaN for
                      an entity measured once. This is the answer the ceiling uses
                      and no detector is allowed to see.
      all_mean        the mean of ALL of this entity's measurements. The same in
                      every draw. Added 2026-10-05: the labels the per-fold
                      searches see on the training entities (SEARCH_LABELS);
                      nothing is fitted on it and nothing is graded on it.
    """
    labels = np.asarray(labels, dtype=np.float64)
    groups = np.asarray(replicate_groups)
    rng = np.random.default_rng(int(seed))

    order = {}
    for i, g in enumerate(groups):
        order.setdefault(g, []).append(i)

    gids = sorted(order, key=lambda g: order[g][0])
    n = len(gids)
    out = {
        'group_id': np.array(gids, dtype=object),
        'row_index': np.zeros(n, dtype=np.int64),
        'n_measurements': np.zeros(n, dtype=np.int64),
        'train_label': np.zeros(n, dtype=np.float64),
        'grade_label': np.full(n, np.nan, dtype=np.float64),
        'gradeable': np.zeros(n, dtype=bool),
        'true_spread': np.full(n, np.nan, dtype=np.float64),
        'all_mean': np.zeros(n, dtype=np.float64),
    }
    for j, g in enumerate(gids):
        rows = np.asarray(order[g], dtype=np.int64)
        vals = labels[rows]
        k = len(rows)
        out['row_index'][j] = rows[0]
        out['n_measurements'][j] = k
        out['all_mean'][j] = float(np.mean(vals))
        if k >= 2:
            # ddof=1, and this matters. With ddof=0 the measured spread of an
            # entity grows with HOW MANY times it was measured, whatever its real
            # disagreement: two measurements of a wide laboratory report about
            # 0.56 of its true width, six of a narrow one about 0.91 of its own.
            # So a ddof=0 ceiling partly ranks entities by measurement count, and
            # the ceiling is the one arm that must rank by disagreement alone.
            # The variance with ddof=1 is unbiased at every count, so the square
            # root of it is what is stored and what the ceiling ranks on.
            out['true_spread'][j] = float(np.sqrt(np.var(vals, ddof=1)))
        if k >= MIN_MEASUREMENTS:
            held = int(rng.integers(k))
            rest = np.delete(vals, held)
            out['train_label'][j] = float(vals[held])
            out['grade_label'][j] = float(np.mean(rest))
            out['gradeable'][j] = True
        else:
            out['train_label'][j] = float(np.mean(vals))
    return out


def detector_scores(X_train, y_train, dataset=None, similarity_neighbours=None,
                    similarity_k=None):
    """Per-entity suspicion score from the training features and labels only.

    It never sees `true_spread` and never sees `grade_label`. Returns None if the
    detector raises, and the caller then records every correction as refused for
    that fold rather than scoring one against a silent fallback.

    Through noise_cache.cached_detector_scores since 2026-09-25 (plan section
    3). With `dataset` None it is the call it always was. With a dataset name
    the scores are stored under their inputs: the same NeighborResidualEstimator
    with k=10, metric=None, is_clf=False and no groups (n17 excludes no
    neighbour, today as before), fitted with .fit(), whose noise_scores_ is
    estimate()'s return (noise_estimation.py:79-81).

    similarity_neighbours / similarity_k: a similarity setting's neighbour
    lists and how many to read (plan section 1). None is today's: neighbours
    in X_train under the process-wide distance.
    """
    if dataset is None and similarity_neighbours is None:
        try:
            est = get_noise_estimator(DETECTOR, k=DETECTOR_K, metric=None)
            return np.asarray(est.estimate(X_train, y_train), dtype=float)
        except Exception as exc:
            print(f"      detector failed: {type(exc).__name__}: {exc}")
            return None
    if DETECTOR_K != 10:
        raise ValueError(f'n17 DETECTOR_K={DETECTOR_K} but the shared scorer '
                         f'offers k=10')
    try:
        s = nc.cached_detector_scores(
            dataset or 'n17', X_train, y_train, is_clf=False,
            detector=DETECTOR, groups=None,
            similarity_neighbours=similarity_neighbours,
            similarity_k=similarity_k)
    except Exception as exc:
        print(f"      detector failed: {type(exc).__name__}: {exc}")
        return None
    return None if s is None else np.asarray(s, dtype=float)


def build_arm(arm, X_train, y_train, scores, rank_quantity_train, n_flag, rng,
              flag_fraction, groups_train=None, top=None):
    """The training rows, labels and row weights one arm fits on.

    top: the screen's best setting as {'setting', 'k', 'scheme', 'lists'}
    (lists carry the scheme), or None. label_smooth and knn_smooth read their
    neighbours from it (DECIDED 2026-10-06). The other corrections read no
    neighbour list; `scores` already came from it.

    Returns (X, y, sample_weight, n_removed, n_relabelled) or None when the arm
    cannot be built.

    The two counts are separate because the arms do different things to reach the
    same flagged entities. `remove`, `ceiling` and the two floors take entities
    OUT of training. The other six corrections keep every entity and change its
    label or its row weight. One field named "removed" covering both would say a
    label-moving run trained on fewer entities than it did.

    `rank_quantity_train` is whatever SPREAD_SCORE names, per training entity,
    NaN where this entity cannot be ranked by it. The ceiling ranks on it and
    `floor_measured` draws at random from exactly the entities it is finite on.
    """
    n = len(y_train)
    if arm == 'no_cleaning':
        return X_train, y_train, None, 0, 0

    if arm in CORRECTIONS:
        if scores is None:
            return None
        # n07's own function, so the seven corrections are defined in one place
        # and this file holds none of them. is_clf is False: n17 refuses a
        # classification dataset before any fold starts, because a repeat
        # measurement of a class label has no spread in the label's own units.
        reads = bool(top) and sx.strategy_reads_setting(arm, top['setting'])
        built = cleaned_training_set(
            arm, X_train, y_train, scores, flag_fraction,
            is_clf=False, groups=groups_train, strength=None, targets=None,
            similarity_neighbours=top['lists'] if reads else None,
            similarity_k=top['k'] if reads else None,
            similarity_scheme=top['scheme'] if reads else None)
        if built is None:
            # gp_smooth refuses above GP_SMOOTH_MAX_ROWS training entities, and
            # remove refuses when fewer than 10 would be left. Both say so.
            return None
        Xa = built['X']
        ya = np.asarray(built['y'], dtype=np.float64)
        keep = built['keep']
        n_removed = int(n - len(ya)) if keep is not None else 0
        if keep is not None and len(ya) < 10:
            return None
        # How many labels this correction actually moved, counted against the
        # column it was given rather than assumed from the flag rate.
        # `constant_lift` leaves every label below the ceiling untouched, and
        # `downweight` and `remove` move none at all.
        col = built.get('label_column')
        n_relabelled = (int(np.count_nonzero(np.asarray(col, dtype=np.float64)
                                             != y_train))
                        if col is not None else 0)
        return Xa, ya, built.get('sample_weight'), n_removed, n_relabelled

    # THE CEILING AND THE FLOOR DRAW FROM THE SAME ENTITIES. That is what makes
    # the floor a control for the ceiling rather than a different experiment.
    #
    # An entity the ranking quantity is not defined on can never be ranked and so
    # can never be removed by the ceiling. Until 2026-09-16 the floor removed its
    # entities from ALL of training while the ceiling could only reach the ones
    # with a spread, so the two were removing from different populations and their
    # difference was partly that. MEASURED on herg_ic50: only 1,218 of 9,802
    # molecules have a spread at all, so a ceiling asked for ten in every hundred
    # of the whole training set was removing about 80 in every hundred of every
    # molecule that had ever been measured twice. Those are not a random 80%: they
    # average 2.88 measurements against 1.23 over all molecules, and 29.7% of them
    # are measured three or more times against 4.1% overall. So the ceiling was
    # removing the well-studied molecules, and the floor was not.
    #
    # Both now draw from the entities the ranking quantity is finite on, and the
    # count is capped at how many of those there are.
    if arm in ('ceiling', 'floor_measured'):
        eligible = np.flatnonzero(np.isfinite(rank_quantity_train))
        if eligible.size == 0:
            return None
        take = int(min(n_flag, eligible.size))
        if arm == 'ceiling':
            order = eligible[np.argsort(-rank_quantity_train[eligible],
                                        kind='stable')]
            chosen = order[:take]
        else:
            chosen = rng.choice(eligible, size=take, replace=False)
        keep = np.ones(n, dtype=bool)
        keep[chosen] = False
        if keep.sum() < 10:
            return None
        return X_train[keep], y_train[keep], None, int(take), 0

    if arm == 'floor_any':
        keep = np.ones(n, dtype=bool)
        keep[rng.choice(n, size=int(min(n_flag, n)), replace=False)] = False
        if keep.sum() < 10:
            return None
        return X_train[keep], y_train[keep], None, int((~keep).sum()), 0

    raise ValueError(f"unknown arm {arm!r}")


def entity_data(data, ent):
    """The experiment's own rows as a data dict: one row per entity.

    2026-10-05 (chat 5, handoff 2.3 item 14, dirty-run bug 11). The per-fold
    searches and the folds run on these rows, not on the loader's rows. On
    herg_ic50 the loader keeps one row per measurement (12,008 rows for
    9,802 molecules), so a search on its rows saw a molecule's measurements on
    both sides of a fold and skipped svm above 10,000 rows. Here every
    measurement of an entity sits in its one row.

    reps and augmentations are each entity's first row (every row of one
    entity carries the same features), in REP_DTYPE (float32). labels are the
    mean of ALL of an entity's measurements, the labels the per-fold searches
    see; see SEARCH_LABELS. groups are the loader's fold groups read at the
    same row.
    """
    ri = np.asarray(ent['row_index'])
    smi = data.get('smiles')
    augs = data.get('augmentations') or None
    return {
        'reps': {n: as_rep_dtype(np.asarray(m)[ri])
                 for n, m in data['reps'].items()},
        'labels': np.asarray(ent['all_mean'], dtype=np.float64),
        'groups': (None if data.get('groups') is None
                   else np.asarray(data['groups'])[ri]),
        'task_type': data['task_type'],
        'smiles': (None if smi is None or len(smi) == 0
                   else np.asarray(smi, dtype=object)[ri]),
        'augmentations': (None if not augs else
                          {a: as_rep_dtype(np.asarray(m)[ri])
                           for a, m in augs.items()}),
    }


def run(data, config, output_dir, dataset_name, **kwargs):
    """N17: removal on real repeat-measurement disagreement."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N17: DOES CLEANING HELP WHEN THE NOISE IS REAL")
    print("=" * 70)

    labels = np.asarray(data['labels'], dtype=np.float64)
    task_type = data['task_type']
    if task_type != 'regression':
        # A repeat measurement of a class label either agrees or does not, so a
        # spread in the label's own units does not exist and neither does a
        # consensus mean.
        # DECIDED 2026-10-06 (Adelaide delegated): on classification datasets
        # label_smooth runs and knn_smooth does not. n17 never reaches a
        # classification dataset, because of this skip, so nothing below
        # needs the rule.
        print(f"  SKIPPED: {dataset_name} is {task_type}. This experiment needs "
              f"repeat measurements on a continuous label.")
        return {'skipped': 'not_regression', 'task_type': task_type}

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    metric = (metric_name, metric_fn, lower_is_better)

    rep_groups = build_replicate_groups(data)
    if rep_groups is None:
        # Not a failure of this experiment. Most loaders collapse their repeat
        # measurements before anything downstream sees them, so there is nothing
        # here to hold out. NOISE_AUDIT_2026_09_10.md section 11.1.
        print(f"  SKIPPED: {dataset_name}'s loader exposes no repeat structure. "
              f"Neither data['replicate_groups'] nor metadata drug/protein ids "
              f"are present, so there is no second measurement to hold out.")
        return {'skipped': 'no_replicate_structure', 'dataset': dataset_name}

    # ── Feasibility, printed before anything is fitted ──
    probe = build_entities(labels, rep_groups, seed=0)
    n_ent = len(probe['n_measurements'])
    n_grade = int(probe['gradeable'].sum())

    # 2026-10-05 (chat 5, R1): the dataset's two decided models
    # (DECISIONS.md Step 1, decided_grid), nothing else. Until then this read
    # config['models'] or DOWNSTREAM_MODELS, five models. n_samples is the
    # entity count, the rows every model is fitted on here: on herg_ic50 that
    # is 9,802, under svm's 10,000-row cap, where the loader's 12,008
    # measurement rows were over it.
    models = list(decided_models(dataset_name, 'general'))
    model_factories = get_model_factories(models, task_type=task_type,
                                          n_samples=n_ent)
    models_without_factory = [m for m in models if m not in model_factories]
    if not [m for m in models if m in model_factories]:
        raise RuntimeError(
            f"n17: no decided model could be built. Decided {models}, got "
            f"{sorted(model_factories)}.")
    if kwargs.get('models') or kwargs.get('base_reps'):
        print("  n17: 'models' and 'base_reps' are no longer taken; each fold "
              "fits the dataset's two decided models on their own 2 base "
              "representations and 2 hybrids (2026-10-05)")

    n_draws = int(kwargs.get('n_draws', N_DRAWS))

    # The share of training entities every removing arm takes out, swept from
    # the caller rather than edited into the constant above. It sets how far
    # the ceiling and `floor_measured` overlap, and on a dataset where few
    # entities were measured twice the two can end up removing almost the same
    # set: at 0.10 on herg_ic50 a training fold of about 6,535 entities is
    # asked for about 653, while only about 812 of those entities have a
    # disagreement to rank at all, so both arms take four in five of the same
    # small population. The default is the constant, so a caller that passes
    # nothing gets exactly what this file did before.
    flag_fraction = float(kwargs.get('flag_fraction', FLAG_FRACTION))

    # WHICH ARMS RUN. Default: every one of them. A caller names fewer only
    # when the ones it leaves out would produce a number that already exists at
    # the identical configuration -- the six corrections never read the ranking
    # quantity, so a run that changes only SPREAD_SCORE would refit them for
    # nothing. `no_cleaning` is always kept, because every gain is read against
    # it. The list that ran is written to 'arms', so no reader has to assume.
    arms = tuple(kwargs.get('arms') or ARMS)
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise ValueError(f"n17: unknown arm(s) {unknown}. Known: {list(ARMS)}.")
    if 'no_cleaning' not in arms:
        arms = ('no_cleaning',) + arms

    # Which quantity the ceiling ranks entities by, and therefore which entities
    # it and `floor_measured` can reach at all. See SPREAD_SCORE.
    spread_score = str(kwargs.get('spread_score', SPREAD_SCORE))
    if spread_score not in SPREAD_SCORES:
        raise ValueError(
            f"n17: spread_score={spread_score!r} is not one of "
            f"{list(SPREAD_SCORES)}.")

    # ── The experiment's own rows, and the folds over them ──
    # One row per entity (entity_data). Every per-fold search, every hybrid
    # build and every fit reads these rows, so all of one entity's
    # measurements are on one side of every fold. The folds are
    # nested_cv_outer_folds over them (noise_outer_folds), the same splits the
    # per-fold searches are filed under; the count is config['n_splits'] or
    # KIRBY_NOISE_N_FOLDS. They do not depend on which measurement a draw
    # picks, so every draw runs on the same folds and the same searches.
    ent_data = entity_data(data, probe)
    folds, n_folds = noise_outer_folds(ent_data, config)
    ent_groups = ent_data['groups']
    smiles_ent = ent_data['smiles']

    # The neighbour settings of this dataset's group (KNN_SEARCH_SPACE_RECORD,
    # Adelaide 2026-10-05): group from the entity count, no model_input
    # setting, no importance weighting, distances per group, neighbour counts
    # 5/10/20 and the counting schemes rank and inverse_square, screened per
    # (draw, fold, configuration) and the top 3 refitted
    # (n07.screen_settings).
    # Scaffold-mates stay in every list (axis 6): no exclude_groups anywhere.
    sim_settings = n07_sweep_settings(dataset_name, n_ent,
                                      has_smiles=smiles_ent is not None)

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'result_layout': RESULT_LAYOUT,
        'noise_regime': 'real_replicate_disagreement',
        'noise_injection': 'none',
        'models': [m for m in models if m in model_factories],
        'models_source': ("decided_grid.decided_models(dataset, 'general'): "
                          "the dataset's two models from DECISIONS.md Step 1"),
        'models_without_factory': models_without_factory,
        'configurations': list(CONFIGURATION_SPACES),
        'configurations_note': (
            'Per outer fold, per decided model: base_1 and base_2 are the '
            "model's 2 best single representations and hybrid and "
            'hybrid__general2 its 2 best hybrids, all ranked by searches on '
            "that fold's training entities (noise_helpers.fold_configurations)."
            ' 2 models x 4 = 8 fitted configurations. Which representation '
            'base_1 is can differ between folds; configurations_by_fold names '
            'it on every fold. Before 2026-10-05 every loaded representation '
            'and up to six hybrids from the whole-dataset searches were fitted '
            'with five models.'),
        'n_folds': int(n_folds),
        'folds_run': [int(f) for f, _, _ in folds],
        'selection_scope': 'fold_train_rows',
        'search_rows': (
            'one row per entity; every per-fold search, hybrid build and fit '
            "reads these rows, so all of an entity's measurements are on one "
            'side of every fold'),
        'search_condition_suffix': SEARCH_SUFFIX,
        'search_labels': SEARCH_LABELS,
        'arms': list(arms),
        'arms_all': list(ARMS),
        'detector': DETECTOR,
        'detector_k': DETECTOR_K,
        'flag_fraction': flag_fraction,
        'flag_fraction_default': FLAG_FRACTION,
        'corrections': list(CORRECTIONS),
        'corrections_source': (
            "experiments.noise.n07_remediation.cleaned_training_set, called "
            "with this experiment's own detector scores and flag fraction. "
            "Until 2026-09-21 this file held two corrections of its own "
            "inline: 'drop', which is n07's 'remove', and 'repair', which is "
            "n07's 'knn_smooth' over one more neighbour."),
        'corrections_that_remove_entities': list(REMOVING_CORRECTIONS),
        'spread_score': spread_score,
        'spread_score_default': SPREAD_SCORE,
        'spread_score_note': (
            "What the ceiling ranks entities by. 'measurement_spread' is the "
            "standard deviation of all of an entity's repeat measurements. "
            "'training_label_error' is how far the one measurement that became "
            "this entity's training label sits from the mean of its others, in "
            "the label's own units; it is undefined for an entity measured "
            "fewer than MIN_MEASUREMENTS times, whose training label is "
            "already the mean of every measurement it has."),
        'min_measurements_to_grade': MIN_MEASUREMENTS,
        'n_draws': n_draws,
        'claim_scope': (
            'Training labels are ONE real measurement per entity; grading labels '
            'are the mean of that entity\'s OTHER measurements. No noise is '
            'injected anywhere. A cleaning result is a correction beating '
            'no_cleaning. `remove` is read on top of that against the distance '
            'between floor_any and ceiling, because it is the one correction '
            'that trains on fewer entities.'),
        'hybrid_sees_grading_labels': False,
        'hybrid_sees_grading_labels_note': (
            "The per-fold searches read only the fold's training entities "
            '(their mean over all their measurements), and every hybrid is '
            "rebuilt on those entities' single-measurement training labels. "
            'No measurement of a held-out entity reaches a search, a feature '
            'ranking or a budget split. Before 2026-10-05 the recipe came '
            'from a search over every measurement row of the dataset, held-out '
            "entities' included."),
        'similarity': n07_similarity_record(
            sim_settings, None, schemes=list(n07_sweep_schemes()),
            screen_top=n07_screen_top(),
            neighbour_spaces_note=(
                'Neighbour spaces only, never fitted configurations: the '
                "model's 2 base representations, base_1 with augmentation "
                'features where the group keeps it, its 2 hybrids, the top '
                'hybrid of the per-fold distance search (hybrid__distance1), '
                'the top hybrid of the per-fold similarity search '
                '(hybrid_similarity), ECFP4 bits with Tanimoto where SMILES '
                'exist, forest leaves.'),
            detector_reads_scheme=True,
            screen_scored_on=(
                'a held-back fifth of the training entities (by scaffold '
                'group), predicted from the other training entities '
                '(n07.screen_settings, inner=)')),
        'main_table_neighbours': (
            "DECIDED 2026-10-06 (Adelaide delegated): in per_draw and summary "
            "the detector (neighbor_residual), label_smooth and knn_smooth "
            "read their neighbours from each (draw, fold, model, "
            "configuration)'s top screened setting: neighbour space and "
            "distance, k, and how much each neighbour counts. Recorded under "
            "configurations_by_fold.<fold>.<draw>.models.<model>"
            ".main_table_neighbour_setting.<configuration>. Before this, 9-10 "
            "Euclidean neighbours in the configuration's own features, each "
            "counting equally."),
        'configurations_by_fold': {},
        'screen': [],
        'per_draw': {},
    }

    counts = {int(k): int((probe['n_measurements'] >= k).sum())
              for k in (2, 3, 4, 5)}
    results['feasibility'] = {
        'n_rows': int(len(labels)),
        'n_entities': n_ent,
        'n_entities_gradeable': n_grade,
        'entities_by_min_measurements': counts,
        'mean_true_spread_label_units': float(
            np.nanmean(probe['true_spread'])) if n_grade else float('nan'),
    }
    print(f"  rows: {len(labels)}, entities: {n_ent}")
    for k, v in counts.items():
        print(f"    entities with {k}+ measurements: {v} of {n_ent} "
              f"({100.0 * v / max(n_ent, 1):.2f}%)")
    print(f"  gradeable entities (>= {MIN_MEASUREMENTS} measurements): {n_grade}")
    print(f"  models (decided): {results['models']}; folds {n_folds} "
          f"(running {results['folds_run']}); neighbour settings "
          f"{len(sim_settings)} x ks {list(n07_sweep_ks())} x schemes "
          f"{list(n07_sweep_schemes())}, top {n07_screen_top()} refitted")
    if n_grade < 30:
        print(f"  SKIPPED: {n_grade} gradeable entities is too few to split into "
              f"{n_folds} folds and read a difference between five arms.")
        results['skipped'] = 'too_few_gradeable_entities'
        return results

    # ── Rows, checkpoint ──
    # ROWS. Every arm's fit beside no_cleaning's on the same configuration, one
    # d_vs_dprime row per (draw, fold, configuration, arm, model), from the
    # numbers the loop below already computes -- no refit. The seven
    # corrections in the main table carry the top screened setting's name as
    # similarity_setting and 'main_table|k=..|scheme=..' at the front of
    # method_setting (2026-10-06); ceiling and the two floors read no
    # neighbour and carry 'none'.
    # score_clean is nan: there is no clean label column here.
    #
    # THE NEIGHBOUR SWEEP. Per (draw, fold, configuration) every setting is
    # screened with no model fit and the top 3 (setting, k, scheme) are kept;
    # every correction is then refitted under those 3 on that configuration.
    # Every setting but forest leaves is built once per (draw, fold, model);
    # forest leaves are fitted on the configuration's own matrix, as n07
    # does (2026-10-05). Rows only; nothing in per_draw or summary reads it.
    #
    # Both files carry RESULT_LAYOUT in their name and signature, so a rows
    # file or checkpoint from the whole-dataset design is never read back.
    rows = sx.Rows(output_dir, EXP_FULL_NAME, dataset_name,
                   condition=RESULT_LAYOUT)
    ck = sx.FoldCheckpoint(
        output_dir, EXP_FULL_NAME, dataset_name,
        sx.run_signature(data, config, {
            'layout': RESULT_LAYOUT, 'models': results['models'],
            'n_draws': n_draws, 'flag_fraction': flag_fraction, 'arms': arms,
            'spread_score': spread_score, 'n_folds': int(n_folds),
            'folds_run': results['folds_run'],
            'search_suffix': SEARCH_SUFFIX,
            'settings': [s_.name for s_ in sim_settings],
            'screen_top': n07_screen_top()}),
        condition=RESULT_LAYOUT)
    sim_unavailable = {}
    if ck.state is not None:
        results['per_draw'] = ck.state['per_draw']
        sim_unavailable = ck.state.get('sim_unavailable', {})
        results['configurations_by_fold'] = ck.state.get(
            'configurations_by_fold', {})
        results['screen'] = ck.state.get('screen', [])
    _cm_ident = dict(dataset=dataset_name, experiment=EXP_FULL_NAME,
                     noise_condition=ROW_CONDITION)

    def _method_setting(arm):
        ms = f'flag_fraction={flag_fraction}|spread_score={spread_score}'
        return (f'detector={DETECTOR}|' + ms) if arm in CORRECTIONS else ms

    # The per-fold searches, once per fold for every draw (fold_searches also
    # stores them on disk under SEARCH_SUFFIX, so a resumed run reads them
    # back). {fold: entries, or the exception the searches raised}.
    entries_by_fold = {}

    def _entries(fold_i, tr):
        if fold_i not in entries_by_fold:
            try:
                entries_by_fold[fold_i] = fold_configurations(
                    ent_data, dataset_name, fold_i=fold_i, train_idx=tr,
                    n_folds=n_folds, condition_suffix=SEARCH_SUFFIX)
            except Exception as exc:
                print(f"    fold {fold_i}: per-fold searches failed: "
                      f"{type(exc).__name__}: {exc}")
                entries_by_fold[fold_i] = exc
        return entries_by_fold[fold_i]

    def _model_neighbours(mname, entry, tr, va_graded, y_tr, g_tr):
        """One model's neighbour spaces and lists for one (draw, fold), every
        setting but forest leaves: (spaces_train, spaces_val, unavailable,
        lists by setting name, smiles_train, smiles_val). Forest leaves are
        fitted on the configuration's own matrix, so _configuration_screen
        adds them per configuration (2026-10-05)."""
        if not sim_settings or entry is None:
            return None
        sp_tr, sp_va, unav = n07_fold_similarity_spaces(
            dataset_name, sim_settings, model_input_tr=None,
            model_input_va=None, reps=ent_data['reps'], train_idx=tr,
            val_idx=va_graded, y_train=y_tr, groups_train=g_tr,
            base_reps=entry.get('base') or [],
            recipes=entry.get('recipes') or {},
            augmentations=ent_data['augmentations'], model_name=mname,
            model_factory=model_factories[mname], metric=metric,
            task_type=task_type)
        smi_tr = (None if smiles_ent is None
                  else list(smiles_ent[np.asarray(tr)]))
        smi_va = (None if smiles_ent is None
                  else list(smiles_ent[np.asarray(va_graded)]))
        shared = [s_ for s_ in sim_settings if s_.space != FOREST_SPACE]
        nbrs = n07_fold_neighbours(dataset_name, shared, sp_tr, sp_va,
                                   y_tr, smiles_train=smi_tr,
                                   smiles_val=smi_va, is_clf=False)
        return sp_tr, sp_va, unav, nbrs, smi_tr, smi_va

    def _configuration_screen(shared, mname, space, X_tr, X_va, y_tr, g_tr,
                              draw, fold_i):
        """The screen for one configuration of one model: the shared lists
        plus forest leaves fitted on this configuration's matrix, as n07
        passes the configuration's matrix (n07_remediation.py:2441).
        Returns (kept [(setting, k, scheme)], lists by setting name)."""
        if shared is None:
            return [], {}
        sp_tr, sp_va, unav, nbrs, smi_tr, smi_va = shared
        nbrs = dict(nbrs)
        forest = [s_ for s_ in sim_settings if s_.space == FOREST_SPACE]
        if forest:
            nbrs.update(n07_fold_neighbours(
                dataset_name, forest, dict(sp_tr, model_input=X_tr),
                dict(sp_va, model_input=X_va), y_tr, smiles_train=smi_tr,
                smiles_val=smi_va, is_clf=False))
        for s_ in sim_settings:
            L = nbrs.get(s_.name)
            if L is None or not L.get('available'):
                sim_unavailable.setdefault(s_.name, {})[
                    f'{mname}:{space}'] = (
                    unav.get(s_.space) or (L or {}).get('reason'))
        # The screen predicts a held-back fifth of the training entities from
        # the rest (2026-10-06). spaces_train is the dict the lists above were
        # built from, with this configuration's matrix where forest leaves
        # read it.
        kept, table = n07_screen_settings(
            sim_settings, nbrs, y_tr, metric=metric, is_clf=False,
            inner={'dataset_name': dataset_name,
                   'spaces_train': (dict(sp_tr, model_input=X_tr) if forest
                                    else sp_tr),
                   'y_train': y_tr, 'groups_train': g_tr,
                   'smiles_train': smi_tr, 'is_clf': False})
        results['screen'].append({
            'model': mname, 'configuration': space, 'draw': int(draw),
            'fold': int(fold_i),
            'kept': [f'{s_.name}|k={k_}|{sch}' for s_, k_, sch in kept],
            'table': table})
        return kept, nbrs

    def _sweep_cell(X_tr, X_va, mname, mi, g_tr, y_tr, y_va, kept, nbrs,
                    cell_D, draw, fold_i):
        """Every correction under the screen's kept (setting, k, scheme) on
        one configuration of one model. The D score is no_cleaning's on this
        configuration (cell_D)."""
        if mname not in cell_D or not kept:
            return
        targets = correction_targets(X_tr, y_tr, groups=g_tr,
                                     want=targets_needed(CORRECTIONS))
        fac = {mname: model_factories[mname]}
        for setting, k, scheme in kept:
            # The detector reads the counting scheme too since 2026-10-06
            # (the lists carry it), so every kept (setting, k, scheme) has
            # its own detector scores, and a correction that reads no
            # neighbour list still moves with them.
            L = dict(nbrs[setting.name], scheme=scheme)
            scores_k = detector_scores(
                X_tr, y_tr, dataset=dataset_name,
                similarity_neighbours=L, similarity_k=k)
            for arm in [a for a in arms if a in CORRECTIONS]:
                reads = sx.strategy_reads_setting(arm, setting)
                ms = f'k={k}|scheme={scheme}'
                ident = sx.identity(**_cm_ident, method=arm,
                                    method_setting=f'{ms}|'
                                    + _method_setting(arm),
                                    similarity_setting=setting.name,
                                    model_input=mi, draw=int(draw),
                                    fold=int(fold_i))
                if rows.is_done(ident, [mname]) and not \
                        sx.controls_enabled():
                    continue

                def _clean(sc_, rate_, _a=arm, _r=reads, _k=k, _L=L,
                           _sch=scheme):
                    return cleaned_training_set(
                        _a, X_tr, y_tr, sc_, rate_, is_clf=False,
                        groups=g_tr, strength=None, targets=targets,
                        similarity_neighbours=_L if _r else None,
                        similarity_k=_k if _r else None,
                        similarity_scheme=_sch if _r else None)
                cleaned = None
                if scores_k is not None:
                    try:
                        cleaned = _clean(scores_k, flag_fraction)
                    except Exception as exc:
                        print(f"      {arm} under {setting.name} k={k} "
                              f"{scheme} failed: {type(exc).__name__}: {exc}")
                _kw = dict(X_tr=X_tr, y_D=y_tr, X_va=X_va, y_va=y_va,
                           metric=metric, factories=fac,
                           score_D={mname: cell_D[mname]})
                sx.dprime_rows(rows, dataset_name, ident, cleaned, **_kw)
                sx.control_rows(
                    rows, dataset_name, ident, cleaned,
                    seed=sx.seed_of(draw, fold_i, mi, mname, arm,
                                    setting.name, k, scheme),
                    rerun=_clean, **_kw)

    # ── The experiment ──
    for draw in range(n_draws):
        if str(draw) in results['per_draw']:
            print(f"\n  === draw {draw}: already in the checkpoint, skipped ===")
            continue
        t_draw = time.time()
        print(f"\n  === draw {draw} (which measurement trains, chosen afresh) ===")
        ent = build_entities(labels, rep_groups, seed=draw)

        y_train_all = ent['train_label']
        y_grade_all = ent['grade_label']
        gradeable = ent['gradeable']

        # What the ceiling ranks on, per entity, NaN where it cannot rank this
        # entity at all. Under 'training_label_error' an entity measured fewer
        # than MIN_MEASUREMENTS times trained on the mean of every measurement
        # it has, so its training label sits 0 from the mean of the rest and
        # there is nothing for the ceiling to find in it.
        if spread_score == 'training_label_error':
            rank_all = np.where(
                gradeable,
                np.abs(ent['train_label'] - ent['grade_label']),
                np.nan)
        else:
            rank_all = ent['true_spread']

        # {configuration: {arm: {model: [fold scores]}}}
        acc = {s: {a: {m: [] for m in models} for a in arms}
               for s in CONFIGURATION_SPACES}
        # {configuration: {arm: {model: [one count per fold]}}}. Each model
        # has its own matrix in every configuration, so the detector flags
        # different entities for each.
        removed = {s: {a: {m: [] for m in models} for a in arms}
                   for s in CONFIGURATION_SPACES}
        relabelled = {s: {a: {m: [] for m in models} for a in arms}
                      for s in CONFIGURATION_SPACES}
        # {configuration: {model: {arm: [folds it refused to build on]}}}. An
        # arm that produced no number is not the same thing as an arm that
        # produced a bad one, and gp_smooth refuses above GP_SMOOTH_MAX_ROWS
        # training entities on every fold of this dataset.
        refused = {s: {m: {} for m in models} for s in CONFIGURATION_SPACES}
        # {configuration: [{'fold', 'model', 'why'}]}: the configurations that
        # could not be built on a fold.
        config_failures = {}
        n_eligible, n_flag_per_fold = [], []
        fold_sizes = []
        fold_errors = {}
        if ck.state is not None and ck.state.get('draw') == draw:
            # This draw was part-way through when the run stopped.
            (acc, removed, relabelled, refused, config_failures, n_eligible,
             n_flag_per_fold, fold_sizes, fold_errors) = ck.state['draw_acc']

        def _save(unit, _d=draw):
            ck.save(unit, {
                'per_draw': results['per_draw'], 'draw': _d,
                'sim_unavailable': sim_unavailable,
                'configurations_by_fold': results['configurations_by_fold'],
                'screen': results['screen'],
                'draw_acc': (acc, removed, relabelled, refused,
                             config_failures, n_eligible, n_flag_per_fold,
                             fold_sizes, fold_errors)})

        for fold_i, tr, va in folds:
            if ck.is_done(f'd{draw}f{fold_i}'):
                print(f"    fold {fold_i}: already in the checkpoint, skipped")
                continue
            fold_rows = []
            va_graded = va[gradeable[va]]
            if len(va_graded) < 5:
                print(f"    fold {fold_i}: only {len(va_graded)} gradeable "
                      f"entities held out, skipped")
                fold_errors[str(fold_i)] = (
                    f'only {len(va_graded)} gradeable entities held out')
                _save(f'd{draw}f{fold_i}')
                continue
            entries = _entries(fold_i, tr)
            if isinstance(entries, Exception):
                fold_errors[str(fold_i)] = (
                    f'per-fold searches failed: {type(entries).__name__}: '
                    f'{entries}')
                _save(f'd{draw}f{fold_i}')
                continue
            fold_sizes.append(int(len(va_graded)))

            y_tr = y_train_all[tr]
            y_va = y_grade_all[va_graded]
            g_tr = ent_groups[tr] if ent_groups is not None else None

            n_flag = max(1, int(round(flag_fraction * len(tr))))
            # How many training entities the ceiling could possibly reach. Only
            # an entity measured twice or more has a disagreement to rank by.
            # When this is at or below n_flag, the ceiling and `floor_measured`
            # both remove EVERY one of them, which is the same set, and their
            # difference is exactly zero for a reason that has nothing to do
            # with the data. MEASURED on bindingdb_kd: 2,014 of 26,531 pairs are
            # measured twice or more, so a training fold holds about 1,611 of
            # them while a tenth of the fold is 2,122 -- the ceiling exhausts
            # the pool and the span came out +0.0000.
            n_eligible.append(int(np.isfinite(rank_all[tr]).sum()))
            n_flag_per_fold.append(int(n_flag))

            frec = {'n_train_entities': int(len(tr)),
                    'n_held_out_entities': int(len(va)),
                    'n_held_out_graded': int(len(va_graded)),
                    'metric': metric_name,
                    'lower_is_better': bool(lower_is_better),
                    'models': {}}

            for mname in models:
                entry = entries.get(mname)
                if mname not in model_factories:
                    frec['models'][mname] = {
                        'available': False,
                        'why': 'no model factory for this name'}
                    continue
                # The model's 4 configurations on this fold. The base pair is
                # the same matrix in every draw; the hybrid pair is rebuilt
                # from the fold's per-fold recipe on this draw's
                # single-measurement training labels, the labels no_cleaning
                # fits on. 0/1 features are left unscaled (FeatureScaler).
                ins, unav, names = fold_model_inputs(
                    ent_data, dataset_name, mname, entry, train_idx=tr,
                    val_idx=va_graded, y_train=y_tr,
                    factory=model_factories[mname], metric=metric)
                rec = configuration_record(entry, names)
                rec['configurations_unavailable'] = {
                    sp: why for sp, why in unav.items()
                    if sp in CONFIGURATION_SPACES}
                rec['n_features'] = {sp: int(np.shape(X[0])[1])
                                     for sp, X in ins.items()}
                rec['metric'] = metric_name
                rec['lower_is_better'] = bool(lower_is_better)
                frec['models'][mname] = rec
                for sp in CONFIGURATION_SPACES:
                    if sp not in ins:
                        config_failures.setdefault(sp, []).append({
                            'fold': int(fold_i), 'model': mname,
                            'why': unav.get(sp, 'not built')})
                print(f"    fold {fold_i} {mname}: base {rec.get('base')}, "
                      f"hybrids "
                      f"{[h.get('label') for h in rec.get('hybrids') or []]}")

                shared_nbrs = _model_neighbours(mname, entry, tr,
                                                va_graded, y_tr, g_tr)

                for space in CONFIGURATION_SPACES:
                    if space not in ins:
                        continue
                    X_tr, X_va = ins[space]
                    # A fresh generator per (draw, fold, configuration),
                    # seeded the same way in every one of them, so
                    # `floor_measured` and `floor_any` remove the IDENTICAL
                    # entities in every configuration of every model.
                    rng = np.random.default_rng([1000 + draw, fold_i])

                    # The screen, before any fit: every neighbour setting
                    # predicts a held-back fifth of this fold's training
                    # entities; kept is best first.
                    kept, nbrs = _configuration_screen(
                        shared_nbrs, mname, space, X_tr, X_va, y_tr, g_tr,
                        draw, fold_i)
                    # DECIDED 2026-10-06 (Adelaide delegated): the main
                    # table's detector (neighbor_residual) and label_smooth /
                    # knn_smooth read their neighbours from kept[0], this
                    # configuration's top screened setting (neighbour space
                    # and distance, k, and how much each neighbour counts),
                    # not 9-10 Euclidean neighbours in the configuration's
                    # own features. With no kept setting the seven
                    # corrections are recorded as refused on this fold, with
                    # the reason under main_table_neighbour_setting.
                    if kept:
                        _s0, _k0, _sch0 = kept[0]
                        top = {'setting': _s0, 'k': int(_k0),
                               'scheme': _sch0,
                               'lists': dict(nbrs[_s0.name], scheme=_sch0)}
                        scores = detector_scores(
                            X_tr, y_tr, dataset=dataset_name,
                            similarity_neighbours=top['lists'],
                            similarity_k=top['k'])
                        rec.setdefault('main_table_neighbour_setting', {})[
                            space] = {'setting': _s0.name,
                                      'space': _s0.space, 'k': int(_k0),
                                      'scheme': _sch0,
                                      'detector_scored': scores is not None}
                        main_sim = _s0.name
                        main_ms = f'main_table|k={int(_k0)}|scheme={_sch0}|'
                    else:
                        top, scores = None, None
                        rec.setdefault('main_table_neighbour_setting', {})[
                            space] = {'setting': None, 'reason': (
                                'no neighbour settings for this dataset'
                                if not sim_settings or shared_nbrs is None
                                else 'the screen kept no (setting, k, '
                                     'scheme): no setting had usable '
                                     'neighbour lists')}
                        main_sim, main_ms = sx.NO_SIMILARITY, 'main_table|'
                    mi = f'{mname}:{space}'
                    cell_scores = {}

                    for arm in arms:
                        built = build_arm(arm, X_tr, y_tr, scores,
                                          rank_all[tr], n_flag, rng,
                                          flag_fraction, groups_train=g_tr,
                                          top=top)
                        if built is None:
                            refused[space][mname].setdefault(arm, []).append(
                                int(fold_i))
                            continue
                        Xa, ya, sw, n_rm, n_rl = built
                        removed[space][arm][mname].append(n_rm)
                        relabelled[space][arm][mname].append(n_rl)
                        try:
                            # downweight is the one arm that hands the fit a
                            # row weight. A model whose fit takes none is not
                            # asked to take one; that arm is then recorded as
                            # failed rather than fitted on flat weights,
                            # which would be no_cleaning under another name.
                            if mname in sx.CACHEABLE_MODELS:
                                # Through noise_cache: the same fit and
                                # model.predict, stored under the content of
                                # the matrix, the labels and the weights, so
                                # the sweep's D' fits of the same data read
                                # it back. Seeded models only.
                                _pred = nc.cached_fit_predict(
                                    dataset_name, mname,
                                    model_factories[mname], Xa, ya, X_va,
                                    sample_weight=(None if sw is None
                                                   else np.asarray(sw)),
                                    predict_mode='predict')
                            else:
                                m_ = model_factories[mname]()
                                if sw is None:
                                    m_.fit(Xa, ya)
                                else:
                                    m_.fit(Xa, ya,
                                           sample_weight=np.asarray(sw))
                                _pred = m_.predict(X_va)
                            acc[space][arm][mname].append(
                                float(metric_fn(y_va, _pred)))
                            cell_scores[arm] = acc[space][arm][mname][-1]
                        except Exception as exc:
                            print(f"      {mname}/{space}/{arm} fold {fold_i} "
                                  f"failed: {type(exc).__name__}: {exc}")

                    # One row per arm against no_cleaning on this matrix.
                    cell_D = ({mname: cell_scores['no_cleaning']}
                              if 'no_cleaning' in cell_scores else {})
                    if cell_D:
                        for arm in arms:
                            if arm == 'no_cleaning':
                                continue
                            fold_rows.append(sx.row(
                                sx.identity(
                                    **_cm_ident, method=arm,
                                    method_setting=(
                                        (main_ms if arm in CORRECTIONS
                                         else '') + _method_setting(arm)),
                                    similarity_setting=(
                                        main_sim if arm in CORRECTIONS
                                        else sx.NO_SIMILARITY),
                                    model_input=mi, draw=int(draw),
                                    fold=int(fold_i)),
                                model=mname, metric=metric,
                                score_D=cell_D[mname],
                                score_Dprime=cell_scores.get(arm)))
                    if mname in cell_D:
                        _sweep_cell(X_tr, X_va, mname, mi, g_tr, y_tr, y_va,
                                    kept, nbrs, cell_D, draw, fold_i)

            results['configurations_by_fold'].setdefault(str(fold_i), {})[
                str(draw)] = convert_numpy(frec)
            # Written as each fold finishes (dirty-run bug 12): the rows file
            # gets this fold's rows and the checkpoint every accumulator, so
            # a killed run loses at most the fold in flight.
            rows.write(fold_rows)
            _save(f'd{draw}f{fold_i}')

        # True when every fold asked to remove at least as many entities as had a
        # disagreement at all. The ceiling and `floor_measured` then removed the
        # identical set and the distance between them is zero by construction.
        _exhausted = bool(n_eligible) and all(
            f >= e for f, e in zip(n_flag_per_fold, n_eligible))
        draw_block = {
            'n_folds_scored': len(fold_sizes),
            'folds_not_scored': dict(fold_errors),
            'gradeable_entities_per_fold': fold_sizes,
            'entities_with_a_disagreement_per_fold': n_eligible,
            'entities_asked_to_remove_per_fold': n_flag_per_fold,
            'ceiling_exhausted_the_pool': _exhausted,
            # The configurations that could not be built, as {'fold', 'model',
            # 'why'}. The other configurations of that model on that fold keep
            # their numbers.
            'configurations_not_built': {
                k: list(v) for k, v in config_failures.items()},
            'per_space': {},
        }
        for space in CONFIGURATION_SPACES:
            per_arm = {}
            for arm in arms:
                per_arm[arm] = {
                    'per_model': {
                        m: {'mean': fold_spread(v)['mean'],
                            'fold_spread': fold_spread(v),
                            'fold_scores': [float(x) for x in v],
                            'n_folds': len(v),
                            # Entities taken OUT of training, per fold.
                            'entities_removed_per_fold':
                                removed[space][arm][m],
                            # Entities kept in training with a CHANGED label,
                            # per fold.
                            'entities_relabelled_per_fold':
                                relabelled[space][arm][m],
                            'folds_the_arm_refused':
                                refused[space][m].get(arm, [])}
                        for m, v in acc[space][arm].items()},
                }
            draw_block['per_space'][space] = {'per_arm': per_arm}
        results['per_draw'][str(draw)] = convert_numpy(draw_block)
        ck.save(f'd{draw}', {
            'per_draw': results['per_draw'], 'draw': None,
            'sim_unavailable': sim_unavailable,
            'configurations_by_fold': results['configurations_by_fold'],
            'screen': results['screen']})
        print(f"    [{time.time() - t_draw:.0f}s]")
    results['rows'] = rows.summary()
    if sim_unavailable:
        results['similarity_settings_unavailable'] = sim_unavailable

    results['summary'] = _summarise(results, results['models'],
                                    list(CONFIGURATION_SPACES), metric_name,
                                    lower_is_better, arms)
    _print_summary(results['summary'], metric_name)

    # (results, output_dir, experiment_id, dataset) — the order every other
    # experiment in the directory uses.
    save_experiment_results(results, output_dir, 'n17_replicate_removal',
                            dataset_name)
    # The result file holds every draw now, so the per-fold snapshot is spent.
    ck.clear()
    return results


def _summarise(results, models, spaces, metric_name, lower_is_better,
               arms=ARMS):
    """One row per (space, model): each arm's score, and where the two detector
    arms sit.

    `removal_position` is where an arm sits between floor and ceiling. 0.0 is
    level with the floor -- the gain is training-set size and nothing else. 1.0 is
    level with the ceiling -- the detector found what the repeat measurements say
    is actually worst. It is left undefined when the ceiling does not beat its
    floor, and also when it beats it by less than SPAN_MIN_FRAC of the
    no-cleaning score, because there is then no distance worth being a fraction
    of.

    Every number is computed inside one space. Nothing here averages a
    representation against another one or against the hybrid.
    """
    draws = list(results['per_draw'].values())
    out = {'metric': metric_name, 'lower_is_better': bool(lower_is_better),
           'spaces': list(spaces), 'per_space': {}}
    exhausted = any(d.get('ceiling_exhausted_the_pool') for d in draws)

    for space in spaces:
        per_model = {}
        for m in models:
            arm_means = {}
            for arm in arms:
                vals = []
                for d in draws:
                    sp = d['per_space'].get(space)
                    if not sp:
                        continue
                    x = sp['per_arm'][arm]['per_model'][m]['mean']
                    # None since 2026-10-05: fold_spread's mean when no fold
                    # scored.
                    if x is not None and np.isfinite(x):
                        vals.append(x)
                arm_means[arm] = {
                    'mean_over_draws': float(np.mean(vals)) if vals else float('nan'),
                    # The spread across draws is the spread over WHICH measurement
                    # became the training label. A difference between arms smaller
                    # than this is not a difference between arms.
                    'std_over_draws': float(np.std(vals)) if vals else float('nan'),
                    'per_draw': vals,
                    'n_draws_scored': len(vals),
                }
            _nan = {'mean_over_draws': float('nan')}

            def _mean(a, _am=arm_means, _n=_nan):
                return _am.get(a, _n)['mean_over_draws']

            base = _mean('no_cleaning')
            ceil = _mean('ceiling')
            # Each pick is read against the floor drawn from the same entities
            # it could have picked from. The ceiling reaches only the entities
            # the ranking quantity is defined on; `remove` reaches all of
            # training, because its detector reads features and labels.
            #
            # ⛔ ONLY AN ARM THAT REMOVES ENTITIES GETS A REMOVAL FLOOR.
            # `repair` was given `floor_any` until 2026-09-21 although it
            # removed no entity at all, so it was read against a control that
            # trains on fewer entities than it does. The six label-moving and
            # row-weighting corrections train on every entity and are read
            # against `no_cleaning` alone.
            floor_for = {}
            if 'ceiling' in arms and 'floor_measured' in arms:
                floor_for['ceiling'] = _mean('floor_measured')
            if 'floor_any' in arms:
                for _c in REMOVING_CORRECTIONS:
                    if _c in arms:
                        floor_for[_c] = _mean('floor_any')
            floor = floor_for.get('ceiling', float('nan'))

            def _gain(x, _base=base):
                # Positive always means better, whichever way the metric runs.
                if not (np.isfinite(x) and np.isfinite(_base)):
                    return float('nan')
                return float(_base - x) if lower_is_better else float(x - _base)

            # The distance a removing arm is measured against. It is only a
            # distance when the ceiling actually beats the floor. When it does not,
            # removing the entities that really disagree most did not help, so there
            # is nothing on the table and no arm can be a fraction of it.
            span = _gain(ceil) - _gain(floor)
            # A span is only a distance when the ceiling and its floor removed
            # DIFFERENT entities. Where the flag count met or passed the number of
            # entities with a disagreement, both removed all of them, so the span is
            # zero by construction and says nothing about the data.
            #
            # And a span above zero is still not a distance when it is a few
            # thousandths of the score it is measured against: dividing by it
            # prints a removal position of hundreds of percent. See
            # SPAN_MIN_FRAC.
            span_positive = bool(np.isfinite(span) and span > 0)
            span_floor = (max(1e-10, SPAN_MIN_FRAC * abs(base))
                          if np.isfinite(base) else 1e-10)
            span_too_small = bool(span_positive and not (span > span_floor))
            span_is_real = bool(span_positive and not span_too_small
                                and not exhausted)
            row = {
                'arm_scores': arm_means,
                'gain_over_no_cleaning': {a: _gain(arm_means[a]['mean_over_draws'])
                                          for a in arms},
                'floor_to_ceiling_span': float(span) if np.isfinite(span) else float('nan'),
                'smallest_span_this_file_divides_by': float(span_floor),
                'span_min_frac_of_no_cleaning': SPAN_MIN_FRAC,
                'ceiling_beats_floor': span_is_real,
                'span_under_its_minimum': span_too_small,
                'ceiling_exhausted_the_pool': exhausted,
                'nothing_on_the_table': (
                    None if span_is_real else
                    'The flag count met or passed the number of entities that have a '
                    'disagreement at all, so the ceiling and its floor removed the '
                    'identical set. The distance between them is zero by '
                    'construction and says nothing about the data. Lower the flag '
                    'rate, or read this dataset on gain_over_its_own_floor instead.'
                    if exhausted else
                    f'The ceiling beat its floor by {span:.6g} {metric_name}, '
                    f'which is under {SPAN_MIN_FRAC:.0%} of the no-cleaning '
                    f'score ({base:.6g} {metric_name}). A distance that small '
                    'in the denominator turns a small swing in a correction '
                    'into a removal position of hundreds of percent, so every '
                    'removal position is left undefined. The gain each arm '
                    'made beyond its own floor is still reported and needs no '
                    'ceiling.'
                    if span_too_small else
                    'The ceiling did not beat the floor, so removing the entities '
                    'whose repeat measurements really disagree most helped no more '
                    'than removing the same number at random. No detector can be '
                    'graded against a distance that is not there, so every removal '
                    'position is left undefined.'),
            }
            # What each pick cost beyond a random pick from the SAME entities. This is
            # the number that says the pick was better than luck, and it needs no
            # ceiling at all.
            row['gain_over_its_own_floor'] = {}
            for arm in floor_for:
                g = _gain(_mean(arm))
                f = _gain(floor_for[arm])
                row['gain_over_its_own_floor'][arm] = (
                    float(g - f) if np.isfinite(g) and np.isfinite(f) else float('nan'))
            for arm in REMOVING_CORRECTIONS:
                if arm not in floor_for:
                    continue
                g = _gain(_mean(arm))
                row.setdefault('removal_position', {})[arm] = (
                    float((g - _gain(floor_for[arm])) / span)
                    if span_is_real and np.isfinite(g) else float('nan'))
            per_model[m] = row
        out['per_space'][space] = {'per_model': per_model}

    # Does a pick cost more, or less, where the features were chosen? Only
    # defined where the hybrid ran, and stated against EACH representation
    # rather than against one of them chosen here.
    base_spaces = [s for s in spaces if s in BASE_CONFIGURATIONS]

    def _cross_for(hybrid_space):
        """One hybrid space against every base representation, per arm."""
        cross = {}
        for arm in ('ceiling',) + tuple(REMOVING_CORRECTIONS):
            per_model = {}
            for m in models:
                h = out['per_space'][hybrid_space]['per_model'][m][
                    'gain_over_its_own_floor'][arm]
                per_model[m] = {
                    'gain_in_hybrid': h,
                    'gain_in_base': {
                        s: out['per_space'][s]['per_model'][m][
                            'gain_over_its_own_floor'][arm]
                        for s in base_spaces},
                    'hybrid_minus_base': {
                        s: (float(h - out['per_space'][s]['per_model'][m][
                            'gain_over_its_own_floor'][arm])
                            if np.isfinite(h) and np.isfinite(
                                out['per_space'][s]['per_model'][m][
                                    'gain_over_its_own_floor'][arm])
                            else float('nan'))
                        for s in base_spaces},
                }
            cross[arm] = per_model
        return cross

    # Every hybrid space this run built, each against every base
    # representation. 2026-09-19: a second hybrid space exists, so this is a
    # dict keyed by space name. 'gain_by_space' below keeps the shape it had,
    # holding the predictive space alone, so nothing already reported moves.
    hybrid_spaces = [s for s in spaces if s not in base_spaces]
    if hybrid_spaces and base_spaces:
        out['gain_by_hybrid_space'] = {
            hs: _cross_for(hs) for hs in hybrid_spaces}
    if HYBRID_SPACE in spaces and base_spaces:
        cross = _cross_for(HYBRID_SPACE)
        out['gain_by_space'] = cross
        out['gain_by_space_note'] = (
            "Each entry is what one arm gained beyond a random pick from the "
            'entities it could have reached, in the hybrid and in each base '
            'representation, and the difference between them. It says whether '
            'removal is worth more where the features were chosen. '
            "'gain_by_space' holds the space built from the predictive search "
            "alone; 'gain_by_hybrid_space' holds the same table for every "
            'hybrid space this run built, keyed by space name.')
    return out


def _print_summary(summary, metric_name):
    print("\n" + "=" * 70)
    print(f"SUMMARY ({metric_name}; positive gain = better than no cleaning)")
    for space, srow in summary['per_space'].items():
        print(f"\n  {space}:")
        for m, row in srow['per_model'].items():
            g = row['gain_over_no_cleaning']
            pos = row.get('removal_position', {})
            if not np.isfinite(row['arm_scores']['no_cleaning']['mean_over_draws']):
                print(f"    {m}: no fold scored")
                continue
            print(f"    {m}:")
            own = row['gain_over_its_own_floor']
            _nan = float('nan')
            print(f"      floor_any {g.get('floor_any', _nan):+.4f}   "
                  f"floor_measured {g.get('floor_measured', _nan):+.4f}   "
                  f"ceiling {g.get('ceiling', _nan):+.4f}   "
                  f"span {row['floor_to_ceiling_span']:+.4f}")
            print("      beyond its own floor: "
                  + '  '.join(f"{k} {v:+.4f}" for k, v in own.items()))
            if row.get('ceiling_exhausted_the_pool'):
                print("      the ceiling removed EVERY entity that has a "
                      "disagreement, and so did its floor —")
                print("      they removed the same set, so the distance between "
                      "them is zero by construction.")
                print("      Read this dataset on 'beyond its own floor', or "
                      "lower the flag rate.")
            elif row.get('span_under_its_minimum'):
                print(f"      the ceiling beat its floor by "
                      f"{row['floor_to_ceiling_span']:+.4f} {metric_name}, under "
                      f"{SPAN_MIN_FRAC:.0%} of the no-cleaning score")
                print(f"      ({row['arm_scores']['no_cleaning']['mean_over_draws']:.4f} "
                      f"{metric_name}, so the smallest distance this file will "
                      f"divide by is "
                      f"{row['smallest_span_this_file_divides_by']:.4f}) —")
                print("      dividing by it would print a position of hundreds "
                      "of percent, so no position is reported")
            elif not row['ceiling_beats_floor']:
                print("      the ceiling did not beat the floor — nothing is on "
                      "the table here, so no position is reported")
            # Every correction, best first in the metric's own direction, so
            # the line at the top is the one that did most to the held-out
            # score. The share of the floor-to-ceiling distance is printed only
            # for the correction that removes entities; it is not defined for a
            # correction that removes none.
            _ranked = sorted(
                (a for a in CORRECTIONS if np.isfinite(g.get(a, float('nan')))),
                key=lambda a: -g[a])
            for arm in _ranked:
                if arm in REMOVING_CORRECTIONS:
                    p = pos.get(arm, float('nan'))
                    where = ('no distance to measure against'
                             if not np.isfinite(p)
                             else f'{p:.0%} of the way from floor to ceiling')
                else:
                    where = 'trains on every molecule, read against no_cleaning'
                print(f"      {arm:14s} {g[arm]:+.4f}   {where}")
            _absent = [a for a in CORRECTIONS
                       if not np.isfinite(g.get(a, float('nan')))]
            if _absent:
                print(f"      no number from: {', '.join(_absent)}")
    print("\n  A gain at the floor is a smaller training set, not a removal "
          "result.")
    if 'gain_by_space' in summary:
        print("  'gain_by_space' in the result file holds the same gain in the "
              "hybrid and in each")
        print("  base representation, and the difference.")


def main():
    parser = create_experiment_parser('N17: removal on real repeat disagreement')
    parser.add_argument('--n-draws', type=int, default=N_DRAWS,
                        help='How many times the training measurement is re-drawn')
    parser.add_argument('--flag-fraction', type=float, default=FLAG_FRACTION,
                        help='Share of training entities every removing arm '
                             'takes out')
    parser.add_argument('--spread-score', type=str, default=SPREAD_SCORE,
                        choices=list(SPREAD_SCORES),
                        help='What the ceiling ranks entities by')
    # --base-reps removed 2026-10-05 (chat 5): each fold fits each decided
    # model on its own 2 best single representations, chosen on that fold's
    # training entities.
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, rep_preset=args.reps)
    kw = {'n_draws': args.n_draws, 'flag_fraction': args.flag_fraction,
          'spread_score': args.spread_score}
    run(data, config, args.output, args.dataset, **kw)


if __name__ == '__main__':
    main()
