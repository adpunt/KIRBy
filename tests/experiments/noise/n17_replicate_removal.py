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
               target; `quantile_clip` moves only the labels outside the range of
               label values entities in their own forest leaves carry. They are
               n07's, not this file's -- see CORRECTIONS.
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

WHERE EVERY ARM IS MEASURED. In every representation the run loaded, one at a
time, and in a hybrid per family of stored search, each recipe coming from the
precomputed pool and each rebuilt per fold. There are two families: a search
that chose its features to predict the label, and one that chose them so that
each molecule's activity is predicted by its five nearest neighbours in the
distance those features define. Their spaces are 'hybrid' and
'hybrid_similarity'. Before 2026-09-19 only the first was built and the second
was written into the result file as a score nothing had measured in.

Until 2026-09-17 this file read `sorted(reps)[0]` and measured the six arms in
that one representation. Whichever representation sorted first became the whole
of the feature axis, and nothing in the output said so. The runs in
NOISE_AUDIT_2026_09_10.md 14.11 were read in `drug_maccs` on bindingdb_kd and
`ecfp4` on herg_ic50 for that reason, with `rf` as the only model. Both were
cuts for compute with no stated reason (audit 14.7), so both are gone: the
roster of five models runs, and every representation loaded runs.

THE HYBRID IS BUILT ON THE TRAINING MOLECULES OF A FOLD AND NOTHING ELSE. Its
features are ranked and its budget split against those molecules'
single-measurement training labels -- the same numbers `no_cleaning` fits on.
The consensus labels the held-out molecules are graded against never reach it,
in any fold, under any arm.

It is built ONCE per (draw, fold) and all six arms share it, so the only thing
differing between two arms is which training entities survive and with what
label. Whether a recipe searched after the removal would do better is a
different question and it is n20's.

NO ORACLE CAVEAT TO CARRY. n11, n12 and n13 record that their recipe was chosen
on clean labels and their experiment then corrupts them. Nothing is corrupted
here and there is no clean label column, so the recipe saw the same labels this
experiment trains on. `recipe_saw_clean_labels` is written False for that reason.

WHAT THIS DOES NOT CLAIM. It says nothing about which detector is best in
general -- that is n11, under injected noise where the answer is known. It says
nothing about datasets whose loaders collapse their repeats before anything sees
them, which as of 2026-09-12 is most of them (NOISE_AUDIT_2026_09_10.md 11.1).

WHAT IT COSTS. Six arms by five models by one fit each, in every space, on every
fold of every draw. At ten representations plus the two hybrid spaces, five
folds and three draws that is 5,400 downstream fits and 30 hybrid builds. The
second hybrid space added 450 of those fits and 15 of those builds on
2026-09-19; it runs no search of its own, because the recipe it uses is one the
same pool lookup already returned. Nothing is dropped to
bring that down; a run that cannot afford it should be given fewer
representations by name, and the list it ran is written to `base_reps`.

Usage:
    python n17_replicate_removal.py --dataset bindingdb_kd --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_model_factories, DOWNSTREAM_MODELS,
    get_best_search_for_downstream,
    # One feature space per search family the pool holds, added 2026-09-19.
    # n11's loop, in one place. See the POOL_SUBSET comment below.
    pool_recipes_by_family, describe_pool_family, POOL_SUBSETS,
    # The top-2 selectors and the two hybrid deltas, added 2026-09-18. All four
    # read rows the search already stored and fit nothing.
    top_base_reps, top_hybrids, select_model_rep_pairs, hybrid_deltas,
)
from experiments.noise._ground_truth_providers import build_replicate_groups
# One per-fold hybrid build, shared with n18 and n19. It lived in n19 until
# 2026-09-17.
from experiments.noise.noise_helpers import hybrid_for_fold
# The seven corrections, defined once for the whole suite. n17 held two of its
# own inline until 2026-09-21; see CORRECTIONS.
from experiments.noise.n07_remediation import cleaned_training_set
from kirby.utils.noise_estimation import get_noise_estimator

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
# a target. `quantile_clip` moves only the labels that lie outside the range of
# label values the entities in their own forest leaves carry. All six of those
# train on every entity, so a floor that removes entities is not their control
# and they are read against `no_cleaning` alone. Until today `repair` was read
# against `floor_any`, which removes entities, although `repair` removes none.
CORRECTIONS = ('remove', 'downweight', 'label_smooth', 'knn_smooth',
               'oof_smooth', 'gp_smooth', 'quantile_clip')

# The one correction that trains on fewer entities than `no_cleaning` does.
REMOVING_CORRECTIONS = ('remove',)

ARMS = (('no_cleaning',) + CORRECTIONS
        + ('ceiling', 'floor_measured', 'floor_any'))

# How many times the whole thing is repeated. WHICH measurement becomes the
# training label is a random choice, so one run is one draw of it. Three, to
# match every other experiment in this suite.
N_DRAWS = 3

# The name the hybrid space carries in the result file. Every other space is
# named 'base:<representation>', so a reader can tell which of the two kinds a
# row came from without a lookup.
HYBRID_SPACE = 'hybrid'
BASE_SPACE_PREFIX = 'base:'

# The second hybrid space, added 2026-09-19. See SPACE_FOR_FAMILY.
HYBRID_SIMILARITY_SPACE = 'hybrid_similarity'

# Which searches the recipe may come from. Same as n02, n06, n07, n11, n12, n13
# and n19 since 2026-09-12: a search whose features were chosen to predict the
# label, and one whose features were chosen so that the distance they define
# predicts each molecule's activity from its five nearest neighbours.
POOL_SUBSET = 'predictive_and_similarity'

# ⛔ ONE SPACE PER FAMILY, NOT ONE RECIPE. Added 2026-09-19.
#
# POOL_SUBSETS names the two families in PREFERENCE order and the pool returns
# the first that has an entry, because their stored scores are in different
# units -- a predictive search's score is the dataset's own metric, a
# similarity search's is the correlation between each held-out molecule's
# activity and what its five nearest neighbours predict. A predictive recipe
# always exists, so from 2026-09-12 to 2026-09-19 this file asked for both
# families, built the predictive one and wrote the other into the result file
# as a score it never measured in.
#
# That matters here in particular. Two of the six arms, `drop` and `repair`,
# pick which entities to remove or relabel with DETECTOR, which is
# 'neighbor_residual' at k = 10 -- a molecule scored against its ten nearest
# neighbours in whatever space it is read in. The similarity search chooses
# features so that exactly that distance predicts activity from the five
# nearest neighbours. The other four arms pick without a detector: `ceiling`
# reads the true spread, the two floors draw at random, and `no_cleaning`
# removes nothing.
#
# 'hybrid' is still the predictive recipe, so every number reported before
# 2026-09-19 still means what it meant.
SPACE_FOR_FAMILY = {
    'predictive': HYBRID_SPACE,
    'similarity': HYBRID_SIMILARITY_SPACE,
}


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
    }
    for j, g in enumerate(gids):
        rows = np.asarray(order[g], dtype=np.int64)
        vals = labels[rows]
        k = len(rows)
        out['row_index'][j] = rows[0]
        out['n_measurements'][j] = k
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


def detector_scores(X_train, y_train):
    """Per-entity suspicion score from the training features and labels only.

    It never sees `true_spread` and never sees `grade_label`. Returns None if the
    detector raises, and the caller then drops the two detector arms for that fold
    rather than scoring them against a silent fallback.
    """
    try:
        est = get_noise_estimator(DETECTOR, k=DETECTOR_K, metric=None)
        return np.asarray(est.estimate(X_train, y_train), dtype=float)
    except Exception as exc:
        print(f"      detector failed: {type(exc).__name__}: {exc}")
        return None


def build_arm(arm, X_train, y_train, scores, rank_quantity_train, n_flag, rng,
              flag_fraction, groups_train=None):
    """The training rows, labels and row weights one arm fits on.

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
        built = cleaned_training_set(
            arm, X_train, y_train, scores, flag_fraction,
            is_clf=False, groups=groups_train, strength=None, targets=None)
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
        # `quantile_clip` leaves every label inside its own range untouched, and
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
    """N17: removal on real repeat-measurement disagreement."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N17: DOES CLEANING HELP WHEN THE NOISE IS REAL")
    print("=" * 70)

    labels = np.asarray(data['labels'], dtype=np.float64)
    reps = data['reps']
    task_type = data['task_type']
    if task_type != 'regression':
        # A repeat measurement of a class label either agrees or does not, so a
        # spread in the label's own units does not exist and neither does a
        # consensus mean.
        print(f"  SKIPPED: {dataset_name} is {task_type}. This experiment needs "
              f"repeat measurements on a continuous label.")
        return {'skipped': 'not_regression', 'task_type': task_type}

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'

    rep_groups = build_replicate_groups(data)
    if rep_groups is None:
        # Not a failure of this experiment. Most loaders collapse their repeat
        # measurements before anything downstream sees them, so there is nothing
        # here to hold out. NOISE_AUDIT_2026_09_10.md section 11.1.
        print(f"  SKIPPED: {dataset_name}'s loader exposes no repeat structure. "
              f"Neither data['replicate_groups'] nor metadata drug/protein ids "
              f"are present, so there is no second measurement to hold out.")
        return {'skipped': 'no_replicate_structure', 'dataset': dataset_name}

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
            f"n17: no downstream model could be built. Asked for "
            f"{list(kwargs.get('models') or DOWNSTREAM_MODELS)}, got "
            f"{sorted(model_factories)}.")

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

    # EVERY representation the caller loaded, each its own space. Not
    # `sorted(reps)[0]`, which is what this read until 2026-09-17 -- that made
    # whichever representation sorted first the whole feature axis and said
    # nothing about it in the output. A caller that wants fewer names them.
    base_rep_names = [r for r in (kwargs.get('base_reps') or sorted(reps))
                      if r in reps]
    if not base_rep_names:
        raise RuntimeError(
            f"n17: none of the requested base representations "
            f"{list(kwargs.get('base_reps') or [])} is loaded. Loaded: "
            f"{sorted(reps)}.")

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'noise_regime': 'real_replicate_disagreement',
        'noise_injection': 'none',
        'base_reps': list(base_rep_names),
        'base_reps_note': (
            'One space per representation, all of them. Before 2026-09-17 this '
            'experiment measured every arm in sorted(reps)[0] alone and the '
            'result file did not say which representation that was.'),
        'models': downstream_models,
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
        # Three facts a reader needs before reading any number below, each
        # stated rather than left to be inferred from the absence of a caveat.
        'recipe_saw_clean_labels': False,
        'recipe_saw_clean_labels_note': (
            'Nothing is injected here and there is no clean label column, so '
            'the recipe saw the same labels this experiment trains on. The '
            'oracle caveat n11, n12 and n13 carry does not apply.'),
        'hybrid_sees_grading_labels': False,
        'hybrid_sees_grading_labels_note': (
            'The hybrid is built from the training entities of a fold and their '
            'single-measurement training labels. The consensus labels the '
            'held-out entities are graded against reach no feature ranking and '
            'no budget split.'),
        'per_draw': {},
    }

    # ── Feasibility, printed before anything is fitted ──
    probe = build_entities(labels, rep_groups, seed=0)
    n_ent = len(probe['n_measurements'])
    n_grade = int(probe['gradeable'].sum())
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
    if n_grade < 30:
        print(f"  SKIPPED: {n_grade} gradeable entities is too few to split into "
              f"{n_folds} folds and read a difference between five arms.")
        results['skipped'] = 'too_few_gradeable_entities'
        return results

    # ── The hybrid recipe, fetched once ──
    # The pool is keyed on the dataset as the precompute built it, so the lookup
    # is handed the ROW-level representations, labels and fold grouping, exactly
    # as n02, n11, n12, n13 and n19 hand them over. What comes back is a recipe
    # -- a representation set, an allocation method, an importance method, a
    # budget and an internal evaluator -- and the features are rebuilt per fold
    # on the entity rows below.
    bp = None
    # Bound before the try, because the assignment below is inside it: when the
    # lookup raised, `search` stayed undefined and reading it afterwards was an
    # UnboundLocalError rather than a None. Added 2026-09-18 with the top-2
    # block, which reads it.
    search = None
    try:
        search, meta = get_best_search_for_downstream(
            reps, labels, data.get('groups'),
            downstream_model=downstream_models[0], budget=None,
            augmentations=data.get('augmentations'),
            dataset_name=dataset_name, task_type=task_type,
            data=data,
            pool_subset=POOL_SUBSET,
        )
        bp = meta['best_config']['blueprint'] if meta else None
    except Exception as exc:
        meta = None
        print(f"  hybrid recipe lookup failed: {type(exc).__name__}: {exc}")

    # One recipe per family the pool holds, and a written reason for each family
    # that produced none. See SPACE_FOR_FAMILY. `bp` above stays the recipe the
    # pool RETURNED, which is what every 'hybrid_*' key below already described.
    hybrid_recipes, hybrid_unavailable = pool_recipes_by_family(
        meta, SPACE_FOR_FAMILY,
        eligible_families=POOL_SUBSETS.get(POOL_SUBSET))
    recipe_for_space = {r['space']: r['blueprint'] for r in hybrid_recipes}
    results['hybrid_spaces_unavailable'] = hybrid_unavailable

    base_spaces = [f'{BASE_SPACE_PREFIX}{r}' for r in base_rep_names]
    if not hybrid_recipes:
        # Every arm still runs, in every base representation. Dropping the run
        # would lose six real answers per representation because one space of
        # the several is missing.
        spaces = list(base_spaces)
        results['hybrid_available'] = False
        results['hybrid_unavailable_reason'] = (
            'no recipe came back from the precomputed pool for subset '
            f'{POOL_SUBSET!r}, so the hybrid space could not be built. Every '
            'arm ran in every base representation.')
        print(f"  no hybrid recipe from pool subset {POOL_SUBSET}; "
              f"base representations only")
    else:
        spaces = base_spaces + [r['space'] for r in hybrid_recipes]
        results['hybrid_available'] = True
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
        # Which of the spaces below holds the recipe the pool returned. The
        # other hybrid spaces are the other families' own winners.
        results['hybrid_space_of_pool_winner'] = SPACE_FOR_FAMILY.get(
            meta['objective'], f"hybrid_{meta['objective']}")
        results['hybrid_spaces'] = convert_numpy({
            r['space']: {
                'built_under_objective': r['objective'],
                'features_chosen_by': describe_pool_family(r['objective']),
                'is_the_recipe_the_pool_returned': r['is_pool_winner'],
                'tier': r['tier'], 'aug_condition': r['aug_condition'],
                'budget': r['budget'],
                'allocation_method': r['allocation_method'],
                'importance_method': r['importance_method'],
                'reps': sorted(getattr(r['blueprint'], 'reps', []) or []),
                'search_score_mean': r['search_score_mean'],
                'search_metric_name': r['search_metric_name'],
            } for r in hybrid_recipes})
        results['hybrid_spaces_note'] = (
            'One hybrid space per family of stored search. A predictive search '
            'chose its features to predict the label and its search_score_mean '
            "is in the dataset's own metric. A similarity search chose its "
            "features so that each molecule's activity is predicted by its "
            'five nearest neighbours in the distance those features define, '
            'fits no downstream model, and its search_score_mean is that '
            'correlation. The two search scores are in different units and '
            'nothing here ranks one against the other. What IS comparable is '
            "what every space below is scored in: this experiment's own arms, "
            "in the dataset's own metric, on the same folds.")
        for r in hybrid_recipes:
            print(f"  hybrid space {r['space']}: family {r['objective']!r}, "
                  f"budget {r['budget']} features, allocation "
                  f"{r['allocation_method']}, tier {r['tier']}/"
                  f"{r['aug_condition']}, search score "
                  f"{r['search_score_mean']} in {r['search_metric_name']!r}")
        for _sp, _why in hybrid_unavailable.items():
            print(f"  no {_sp} space: {_why}")
    results['spaces_run'] = list(spaces)
    print(f"  spaces: {len(spaces)} — {', '.join(spaces)}")
    print(f"  models: {', '.join(downstream_models)}")

    # ── Top-2 rows and the two hybrid deltas, beside what is reported ──
    # Owner's instruction 2026-09-18: add the top 2 as an extra and change
    # nothing already reported. Every existing key above is untouched, the six
    # arms still run in every representation the caller loaded, and 'base_reps'
    # still names all of them.
    #
    # These rankings come out of the stored search, NOT out of this
    # experiment's own arms. The search scored representations on the full
    # labels; n17 scores its arms on one measurement per entity against the
    # consensus of the rest. They are different quantities and nothing here
    # mixes them.
    #
    # The search was fetched with downstream_model=downstream_models[0], but
    # search.baselines and search.all_results carry a score under every
    # downstream model the search scored, so the selectors read the rest out of
    # the same object. 0 new model fits: the 4,950 downstream fits this file
    # costs are unchanged.
    if search is not None:
        results.update(_top2_and_deltas(
            search, list(downstream_models), f'n17/{dataset_name}'))
    else:
        results['top2_extra_absent_reason'] = (
            'the pool lookup returned no SearchResult, so there are no stored '
            'rows to rank and no baseline scores to difference against')

    # ── The experiment ──
    for draw in range(n_draws):
        t_draw = time.time()
        print(f"\n  === draw {draw} (which measurement trains, chosen afresh) ===")
        ent = build_entities(labels, rep_groups, seed=draw)

        # ONE row per entity, and only the rows this experiment reads are cast.
        # NOT np.asarray(mat, dtype=float64) on the whole matrix first: this
        # experiment reads one row per entity, and on a drug-target set the rows
        # outnumber the entities -- 30,000 rows against 26,531 pairs on
        # bindingdb_kd. Casting first makes a float64 copy of every row, which
        # on a 2,048-bit fingerprint is about 490 MB that is then discarded.
        ent_reps = {name: np.asarray(mat[ent['row_index']], dtype=np.float64)
                    for name, mat in reps.items()}

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

        # Fold geometry over ENTITIES. data['groups'] is per row, so it is read
        # at each entity's representative row.
        row_groups = data.get('groups')
        ent_groups = (np.asarray(row_groups)[ent['row_index']]
                      if row_groups is not None else None)

        # {space: {arm: {model: [fold scores]}}}
        acc = {s: {a: {m: [] for m in downstream_models} for a in arms}
               for s in spaces}
        removed = {s: {a: [] for a in arms} for s in spaces}
        relabelled = {s: {a: [] for a in arms} for s in spaces}
        # {space: {arm: [folds it refused to build on]}}. An arm that produced
        # no number is not the same thing as an arm that produced a bad one, and
        # gp_smooth refuses above GP_SMOOTH_MAX_ROWS training entities on every
        # fold of this dataset.
        refused = {s: {} for s in spaces}
        # {space: [fold numbers whose build failed]}. A list of folds until
        # 2026-09-19, when a second hybrid space arrived and the folds of the
        # two would otherwise have been added into one list.
        hybrid_failures = {}
        n_eligible, n_flag_per_fold = [], []
        fold_sizes = []

        splits = list(cv_split(ent_groups, n_folds,
                               ent_reps[base_rep_names[0]], y_train_all,
                               mode=mode))
        for fold_i, (tr, va) in enumerate(splits):
            va_graded = va[gradeable[va]]
            if len(va_graded) < 5:
                print(f"    fold {fold_i}: only {len(va_graded)} gradeable "
                      f"entities held out, skipped")
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

            # ── The feature spaces for this fold ──
            # Each base representation is scaled on this fold's training
            # entities. The hybrid is built ONCE, here, from those same training
            # entities and their single-measurement training labels, and all six
            # arms share it. The consensus labels y_va never enter it.
            fold_spaces = {}
            for rname in base_rep_names:
                sc = StandardScaler()
                X_all = ent_reps[rname]
                fold_spaces[f'{BASE_SPACE_PREFIX}{rname}'] = (
                    sc.fit_transform(X_all[tr]), sc.transform(X_all[va_graded]))
            # One build per family the pool holds, all on the same training
            # entities and the same single-measurement training labels.
            # hybrid_for_fold reads the objective off the recipe and drops the
            # model, the metric and its direction under a geometry recipe,
            # where build_fold_hybrid refuses the pair -- so one call serves
            # both families.
            for _sp in spaces:
                _bp = recipe_for_space.get(_sp)
                if _bp is None:
                    continue
                X_tr_h, X_va_h = hybrid_for_fold(
                    {k: v[tr] for k, v in ent_reps.items()},
                    {k: v[va_graded] for k, v in ent_reps.items()},
                    y_tr, _bp, g_tr,
                    model_factories[downstream_models[0]](),
                    metric_fn, lower_is_better)
                if X_tr_h is None:
                    hybrid_failures.setdefault(_sp, []).append(int(fold_i))
                else:
                    sc_h = StandardScaler()
                    fold_spaces[_sp] = (
                        sc_h.fit_transform(X_tr_h), sc_h.transform(X_va_h))

            for space in spaces:
                if space not in fold_spaces:
                    continue
                X_tr, X_va = fold_spaces[space]

                # A fresh generator per (draw, fold, space), seeded the same way
                # in every space. `floor_measured` and `floor_any` then remove
                # the IDENTICAL entities in the base representations and in the
                # hybrid, so a base-against-hybrid reading of any arm is not
                # partly two different random picks.
                rng = np.random.default_rng([1000 + draw, fold_i])

                # The detector reads features, so it is re-run in each space --
                # that is the whole reason a detector arm can move between them.
                scores = detector_scores(X_tr, y_tr)

                for arm in arms:
                    built = build_arm(arm, X_tr, y_tr, scores,
                                      rank_all[tr], n_flag, rng,
                                      flag_fraction, groups_train=g_tr)
                    if built is None:
                        refused[space].setdefault(arm, []).append(int(fold_i))
                        continue
                    Xa, ya, sw, n_rm, n_rl = built
                    removed[space][arm].append(n_rm)
                    relabelled[space][arm].append(n_rl)
                    for mname in downstream_models:
                        try:
                            m = model_factories[mname]()
                            # downweight is the one arm that hands the fit a row
                            # weight. A model whose fit takes none is not asked
                            # to take one; that arm is then recorded as refused
                            # rather than fitted on flat weights, which would be
                            # no_cleaning under another name.
                            if sw is None:
                                m.fit(Xa, ya)
                            else:
                                m.fit(Xa, ya, sample_weight=np.asarray(sw))
                            acc[space][arm][mname].append(
                                float(metric_fn(y_va, m.predict(X_va))))
                        except Exception as exc:
                            print(f"      {space}/{arm}/{mname} fold {fold_i} "
                                  f"failed: {type(exc).__name__}: {exc}")

        # True when every fold asked to remove at least as many entities as had a
        # disagreement at all. The ceiling and `floor_measured` then removed the
        # identical set and the distance between them is zero by construction.
        _exhausted = bool(n_eligible) and all(
            f >= e for f, e in zip(n_flag_per_fold, n_eligible))
        draw_block = {
            'n_folds_scored': len(fold_sizes),
            'gradeable_entities_per_fold': fold_sizes,
            'entities_with_a_disagreement_per_fold': n_eligible,
            'entities_asked_to_remove_per_fold': n_flag_per_fold,
            'ceiling_exhausted_the_pool': _exhausted,
            # Folds whose hybrid could not be built. Those folds still carry
            # every base-representation number.
            # Unchanged in name and shape: the folds the PREDICTIVE
            # hybrid failed to build on. Every space's failures, the
            # geometry one included, are in the key below it.
            'folds_with_no_hybrid': hybrid_failures.get(HYBRID_SPACE, []),
            'folds_with_no_hybrid_by_space': {
                k: list(v) for k, v in hybrid_failures.items()},
            'per_space': {},
        }
        for space in spaces:
            per_arm = {}
            for arm in arms:
                per_arm[arm] = {
                    # Entities taken OUT of training, per fold. Always 0 for
                    # repair, which changes labels instead.
                    'entities_removed_per_fold': removed[space][arm],
                    # Entities kept in training with a CHANGED label, per fold.
                    # Always 0 for every arm but repair.
                    'entities_relabelled_per_fold': relabelled[space][arm],
                    'folds_the_arm_refused': refused[space].get(arm, []),
                    'per_model': {
                        m: {'mean': float(np.mean(v)) if v else float('nan'),
                            'fold_scores': [float(x) for x in v],
                            'n_folds': len(v)}
                        for m, v in acc[space][arm].items()},
                }
            draw_block['per_space'][space] = {'per_arm': per_arm}
        results['per_draw'][str(draw)] = convert_numpy(draw_block)
        print(f"    [{time.time() - t_draw:.0f}s]")

    results['summary'] = _summarise(results, downstream_models, spaces,
                                    metric_name, lower_is_better, arms)
    _print_summary(results['summary'], metric_name)

    # (results, output_dir, experiment_id, dataset) — the order every other
    # experiment in the directory uses. It was (results, experiment_id, dataset,
    # output_dir) here, which wrote to '<cwd>/n17_replicate_removal/<dataset>/
    # <output_dir>.json'. Two things followed: the file landed under the working
    # directory instead of the results directory, and run_all's resume probe
    # looks at '<output_dir>/n17_replicate_removal/<dataset>.json', which never
    # existed, so every run started from nothing.
    save_experiment_results(results, output_dir, 'n17_replicate_removal',
                            dataset_name)
    return results


def _summarise(results, models, spaces, metric_name, lower_is_better,
               arms=ARMS):
    """One row per (space, model): each arm's score, and where the two detector
    arms sit.

    `removal_position` is where an arm sits between floor and ceiling. 0.0 is
    level with the floor -- the gain is training-set size and nothing else. 1.0 is
    level with the ceiling -- the detector found what the repeat measurements say
    is actually worst. It is left undefined when floor and ceiling are within a
    hair of each other, because there is then no distance to be a fraction of.

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
                    if np.isfinite(x):
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
            # Each pick is read against the floor drawn from the same entities it
            # could have picked from. The ceiling reaches only the entities measured
            # twice or more; drop and repair reach all of training.
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

            # The distance the detector arms are measured against. It is only a
            # distance when the ceiling actually beats the floor. When it does not,
            # removing the entities that really disagree most did not help, so there
            # is nothing on the table and no arm can be a fraction of it.
            span = _gain(ceil) - _gain(floor)
            # A span is only a distance when the ceiling and its floor removed
            # DIFFERENT entities. Where the flag count met or passed the number of
            # entities with a disagreement, both removed all of them, so the span is
            # zero by construction and says nothing about the data.
            span_is_real = bool(np.isfinite(span) and span > 0 and not exhausted)
            row = {
                'arm_scores': arm_means,
                'gain_over_no_cleaning': {a: _gain(arm_means[a]['mean_over_draws'])
                                          for a in arms},
                'floor_to_ceiling_span': float(span) if np.isfinite(span) else float('nan'),
                'ceiling_beats_floor': span_is_real,
                'ceiling_exhausted_the_pool': exhausted,
                'nothing_on_the_table': (
                    None if span_is_real else
                    'The flag count met or passed the number of entities that have a '
                    'disagreement at all, so the ceiling and its floor removed the '
                    'identical set. The distance between them is zero by '
                    'construction and says nothing about the data. Lower the flag '
                    'rate, or read this dataset on gain_over_its_own_floor instead.'
                    if exhausted else
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
    base_spaces = [s for s in spaces if s.startswith(BASE_SPACE_PREFIX)]

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
    parser.add_argument('--base-reps', type=str, default=None,
                        help='Comma-separated representations to run as base '
                             'spaces. Default: every representation loaded')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, rep_preset=args.reps)
    kw = {'n_draws': args.n_draws, 'flag_fraction': args.flag_fraction,
          'spread_score': args.spread_score}
    if args.base_reps:
        kw['base_reps'] = [x.strip() for x in args.base_reps.split(',')]
    run(data, config, args.output, args.dataset, **kw)


if __name__ == '__main__':
    main()
