#!/usr/bin/env python3
"""N20: does the cleaning have to come BEFORE the recipe search?

Paper: Noise.

THE QUESTION. Every cleanup in this suite is applied after the recipe is fixed.
n07, n11, n12, n13 and n18 all pick a recipe first, rebuild the features, and
only then drop, reweight or overwrite a training label. So the cleaning never
reaches the feature selection. `create_hybrid` ranks features and splits the
budget against whatever labels it is handed, and it is always handed the
uncleaned ones. Nothing has measured what that ordering costs.

FOUR WAYS OF RUNNING IT, and the point is the shape.

  arm                             features chosen from   model fitted on
  search_asis__fit_asis           the labels as they are  the labels as they are
  search_asis__fit_cleaned        the labels as they are  the cleaned labels
  search_cleaned__fit_cleaned     the cleaned labels      the cleaned labels
  search_cleaned__fit_asis        the cleaned labels      the labels as they are

`search_asis__fit_cleaned` is what the whole suite does today. The difference
between it and `search_cleaned__fit_cleaned` is what the ordering costs. The
fourth arm separates the feature choice from the fit: a gain there is the
cleaning helping the feature choice with no cleaned label ever reaching a model.

"Features chosen from" covers both halves of the choice. It is the label column
handed to the recipe search, AND the label column handed to the per-fold
ranking and budget split. Handing one of those the cleaned labels and the other
the uncleaned ones would leave an arm that is neither row of the table.

THE SEARCHES, SINCE 2026-10-05 (HANDOFF_EXPERIMENT_ALIGNMENT.md 2.4, chat 5).
Both sides run the per-fold search every experiment now runs: the dataset's
decided grid (DECISIONS.md: its two models, budgets, importance methods,
allocation methods, evaluators, representation counts, augmentation), pruning
off, on the fold's training rows only. Each side selects, per decided model,
its 2 best base representations and its 2 best hybrids
(_shared.fold_standard_selection), so every arm is run in 2 models x 4 = 8
configurations. The labels a side is handed are written into the training
rows of a copy of the label column (_with_train_labels); a cleaned search
runs on the kept rows for `remove`. A smoother's cleaned search TRAINS on the
cleaned column (train_labels=) and scores every candidate on the labels as
they are (decided 2026-10-06, SEARCH_SCORES_ON = 'asis'). Each different label column or row set is
filed under its own condition suffix, so two never read each other's stored
rows. On the dataset's own labels (as_they_come, injected level 0) the side
searched on the labels as they are IS the dataset's standard per-fold search.
Until 2026-10-05 both sides called find_best_hybrid with `train_labels=` on
the fast grid, five models, all eight budgets, tuning and calibrated pruning.

WHICH CLEANUPS CAN REACH A FEATURE SELECTION AT ALL. Nine of the eleven n07
offers (as of 2026-10-05, after f36 swapped quantile_clip for constant_lift),
and the other two are refused rather than faked.

`cleaned_training_set` hands back a dict, and three of its entries decide this.
`label_column` is a full-length cleaned label column and is what the cleaned
search is handed in its training rows. `keep` is a mask over the training rows and
is what the cleaned search runs on. `sample_weight` is a per-row weight and is
read by the two FIT arms only. A cleanup that returns neither a label column
nor a mask has no cleaned search, so the pair this file compares does not exist
for it.

  label_smooth   full-length cleaned label column -> the cleaned search
  knn_smooth     the same
  prox_smooth    the same; the neighbours come from a forest's leaves rather
                 than from Euclidean distance in the single representation.
                 Since f36 (2026-10-05), on a regression label every label
                 moves a quarter of the way toward the mean label of its 5
                 forest neighbours and no suspicion score is read
  oof_smooth     the same; the target is an out-of-fold LightGBM prediction
  gp_smooth      the same; the target is a Gaussian process posterior mean
  js_shrink      the same; how far each label moves is derived from two
                 variances rather than chosen
  constant_lift  the same; only the labels sitting at the ceiling the
                 training labels themselves show are raised, by one constant
                 (n30's lift). It reads no suspicion score, and it declines on
                 a fold whose labels show no ceiling. Replaced quantile_clip
                 on 2026-10-05 (f36, owner's decision in n07)
  remove         drops rows, so the search and the ranking run on the kept rows
  downweight     sets a per-row weight, and `sample_weight` appears zero times
                 in src/kirby/hybrid.py and zero times in
                 src/kirby/find_best_hybrid.py (checked 2026-09-23). There is
                 nothing for a weight to reach. Its two searched-on-cleaned
                 arms are left undefined and named in the output.
  huber_weight   NOT RUN. Its dict is `sample_weight` and nothing else:
                 `keep` is None and `label_column` is None
                 (n07_remediation.py:682-683), so the branch that builds the
                 cleaned search never fires and both arms this file exists to
                 compare are undefined for it. `downweight` has the identical
                 return value and stays only because it is already scored on
                 the two fit-side arms and the smoke reads it there. Whether
                 Huber's flat-then-falling weight beats downweight's
                 1/(1 + strength*score) is a question about the two weight
                 shapes, which is n07's comparison and not this file's.
  invvar_weight  NOT RUN, for that reason and one more. Its dict is
                 `sample_weight` and nothing else too. It also weights by how
                 many times each molecule was measured, and that count reaches
                 `cleaned_training_set` as `measurement_counts`, which n20
                 never has: it would decline on every fold and write a row of
                 refusals.

FOUR OF THE NINE ARE NOT DEFINED ON A CLASS LABEL. oof_smooth, gp_smooth,
constant_lift and js_shrink each move a label to a NUMBER, and
`cleaned_training_set` raises rather than approximating one on a nominal class
label (n07_remediation.py:879-882). `cleanups_for` drops them on a
classification dataset and the result file names them under `cleanups_not_run`.

The cleanups themselves are n07's, called rather than copied:
`n07_remediation.cleaned_training_set` is where they live and is what
`n07._evaluate_fold` calls.

ONE CLEANING DECISION PER (FOLD, SINGLE REPRESENTATION), MADE BEFORE ANY HYBRID
EXISTS. The detector has to run somewhere, and the searched-on-cleaned arms need
the cleaning before there are any features to run it in. So the suspicion score
is computed in a single representation, and the same decision serves the search
side and the fit side of all four arms.

WHICH SINGLE REPRESENTATION. In each fold, the 2 best single representations
of CLEANING_RANK_MODEL (default: the first of the dataset's two decided
models) in that fold's search on the labels as they are. The detector and
every cleanup run once in each, as `base_1` and `base_2`, shared by every
model and configuration. Until 2026-10-05 this was rf's 2 best (owner's
decision 2026-09-25); rf is no longer searched on most datasets. No fallback:
a fold whose search ranks no single representation runs no cleanup, and
`base_reps_by_fold` says why. The ranking is read off the fold's training
molecules only, so no held-out molecule enters it. That is a real constraint
rather than a choice: a cleaning that reaches the feature selection cannot have
been decided in a feature space that does not yet exist. It also means the four
arms flag the identical molecules, so the only things differing between two
arms are the two named in the table.

THE CONTROL THAT CHECKS THE TWO SEARCHES ARE ONE CODE PATH. `no_cleaning` is run
as a cleanup: its cleaned column IS the uncleaned one. The two searches are then
handed byte-identical label columns, so they must produce the same recipe and
the four arms must produce four identical scores. A gap there is the two
searches choosing from lists of different lengths, which is the defect n15
carries at NOISE_AUDIT_2026_09_10.md section 4, n15 defect 3. It is reported as
`search_path_identity`, not hidden inside a tolerance.

BOTH REGIMES, because they answer different questions.

  injected       one decided kind of damage (default outlier_p05; a job may
                 name another of the six) at the decided levels for the
                 dataset group (noise_helpers.decided_levels: 0, 0.5, 1.0,
                 1.5 of the clean label spread on full_knn, 0 and 1.0 on
                 fast_knn; classification keeps its flip kinds and levels).
                 Three draws per non-zero level; level 0 is the labels as
                 delivered and runs once. Where a corrupted subset exists
                 `noise_helpers.detection_scores` is defined and is written.
  as_they_come   nothing is corrupted. n18's regime. The training labels are
                 what the loader produced and the held-out labels are graded
                 untouched, so a win is a win on the data as it arrived.

WHAT THE HELD-OUT MOLECULES ARE GRADED AGAINST. `labels[val_idx]`, untouched, in
every arm and both regimes. No cleaned label and no corrupted label is ever
scored against. The folds are noise_helpers.noise_outer_folds (the outer folds
the per-fold searches are filed under; the count is the run's config).

THE HEADLINE IS A COUNT OF FOLDS, not a mean, for n18's reason at :33-37 and
n19's at :823-833. Every comparison is PAIRED: the same fold, the same held-out
molecules, two arms. What varies between folds is common to both and cancels.

THE CLEANING'S NEIGHBOURS (decided 2026-10-06). label_smooth and knn_smooth
find their neighbours in the fold's top screened setting (n07.screen_settings
kept[0]: neighbour space and distance, k, counting scheme), screened in the
cleaning representation on a held-back fifth of the training molecules. That
one setting feeds all four arms and every cleaned search.

COST. One per-fold search per fold and pass on the labels as they are (plus
the distance and similarity searches its neighbour spaces need), and one per
different cleaned label column. Each is the dataset's full decided grid,
unpruned. Both sides get identical settings, so the comparison is like for
like.

⛔ A SEARCH WITH ONE CANDIDATE CANNOT ANSWER THIS FILE'S QUESTION. The first
result n20 ever produced, on flab_thermo_tresanco on 2026-09-20, came back with
its identity control passing exactly: both searches scored the same number of
candidates on 2 of 2 held-out folds and picked the same recipe in 2 of 2
(fold, model) cells, and the four arms sat 0.0 apart. That number of candidates
was ONE. The local iteration settings cut the search to one allocation method,
one importance scorer, one internal evaluator and one budget, and of the two
representation counts they left only the whole pool survived.

With a single configuration in the list the search picks the same recipe
whatever label column it is handed, so the half of this experiment that is
about WHICH REPRESENTATION gets selected cannot move. What still moved in that
run is the other half: the per-fold feature ranking and budget split, which are
recomputed from the label column handed to `hybrid_for_fold`, so the two arms
built different features out of the same recipe. Both halves are named in the
docstring above as "features chosen from"; only the second one was live.

Anything that narrows the search must leave the representation counts, the
allocation methods and the importance scorers more than one value between them.
That is the axis the question rides on.

ONE THING TO KNOW BEFORE READING THE FLAG-RATE AXIS. On a classification
dataset four of the nine read it, and on a regression dataset only two do
(n07.reads_flag_share, read 2026-10-05). The flag rate sets a cut-off at the
(1 - rate) quantile of the suspicion score, so a cleanup that never thresholds
that score never sees the rate, and its three flag rates are one number
repeated:

  reads it          remove and knn_smooth threshold at that quantile on every
                    label; label_smooth and prox_smooth do too, but only on a
                    CLASSIFICATION label
  does not read it  on a regression label, label_smooth, oof_smooth, gp_smooth
                    and js_shrink move every label toward a target in
                    proportion to that molecule's suspicion score
                    (`noise_estimation.smooth_by_scores`, method
                    'proportional'; js_shrink derives its own fraction from two
                    variances). prox_smooth moves every label a fixed quarter
                    of the way and reads no score (f36, 2026-10-05).
                    downweight turns the score into a weight with no cut-off.
                    constant_lift reads neither the rate nor the suspicion
                    score at all -- it raises only the labels at the ceiling.

n07 itself runs fewer rates than this file since f36 (knn_smooth at 0.05
only; the three neighbour cleanups at 0.10 on a class label). n20 still sweeps
all three FLAG_RATES on every cleanup that reads one (2026-10-05).

A cleanup that does not read the rate hands back the same cleaned column at all
three rates. `_label_column_key` is what stops that costing three searches:
identical columns share one cache namespace. The rate is still swept in full on
the ones that read it.

Usage:
    python n20_clean_before_search.py --dataset logd --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import time
import numpy as np

from kirby.scaling import FeatureScaler
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_domain_metric, get_model_factories, as_rep_dtype,
    fold_standard_selection, fold_searches,
)
from experiments.decided_grid import decided_models
from experiments.dataset_policy import noise_groups_for
from experiments.noise.noise_helpers import (
    detection_scores, detection_is_defined, hybrid_for_fold,
    noise_outer_folds, fold_spread, fold_configurations, fold_model_inputs,
    configuration_record, decided_conditions, decided_levels,
    inject_condition, level_unit, CONFIGURATION_SPACES, draws_for,
)
# The cleanups are n07's. `cleaned_training_set` is where they live and is what
# n07's own `_evaluate_fold` calls, so nothing here is a second copy.
#
# ⛔ REGRESSION_ONLY_CLEANUPS IS IMPORTED, NOT RETYPED. n18 wrote its own copy
# of that tuple on 2026-09-18 with the three names it then knew, and js_shrink
# was added to n07's on 2026-09-23 without reaching it. A copy here would go
# stale the same way.
from experiments.noise.n07_remediation import (
    cleaned_training_set, _compute_noise_scores, _predict_for_metric,
    correction_targets, targets_needed, REGRESSION_ONLY_CLEANUPS,
)
# The neighbour settings of the kNN record (2026-10-05): the dataset group's
# settings, the screen that keeps the best 3, the spaces and the lists. n07's
# shared functions, called rather than copied.
from experiments.noise.n07_remediation import (
    sweep_settings as n07_sweep_settings, sweep_ks as n07_sweep_ks,
    sweep_schemes as n07_sweep_schemes, screen_settings,
    fold_similarity_spaces, fold_neighbours,
)
# The D-versus-D' rows, the controls and the per-fold checkpoint, added
# 2026-09-25 (plan sections 1-3).
from experiments.noise import _sim_axis_n15_n22 as sx

# The name the rows file and the checkpoint are filed under.
EXP_FULL_NAME = 'n20_clean_before_search'

# What a default-path row's similarity_setting says when the cleanup did NOT
# read the fold's top screened setting: label_smooth / knn_smooth in the single
# representation the cleaning is decided in, under the process-wide distance
# (sx.TODAY); prox_smooth on a forest fitted on that same matrix (n07:904-913).
#
# DECIDED 2026-10-06 (Adelaide delegated): on the default path, which feeds
# every cleaned search and every main-table number, label_smooth and
# knn_smooth (and their random-flag twins) find their neighbours in the fold's
# top screened setting: n07.screen_settings' kept[0] (neighbour space and
# distance, k, and how much each neighbour counts), screened in this fold's
# cleaning representation, scored on a held-back fifth of the training
# molecules (inner=). prox_smooth reads it only when that setting is
# forest_proximity. One setting per (fold, single representation), so the
# number of cleaned searches does not change. These names are used only when
# the screen kept nothing (or no setting is swept); the reason is written to
# cleaning_neighbours_by_fold.
_DEFAULT_SIM = {'label_smooth': sx.TODAY, 'knn_smooth': sx.TODAY,
                'prox_smooth': 'forest_proximity|none|forest|k=default'}

# Which cleanup is applied to the training rows. 'no_cleaning' is not a cleanup:
# its cleaned column is the uncleaned one, so it is the control that checks the
# two searches are one code path. See the docstring.
IDENTITY_CONTROL = 'no_cleaning'

# Every cleanup n07 offers that n20 can hand to a recipe search, plus
# `downweight`, which it cannot and which is scored on its two fit-side arms
# with the refusal recorded.
#
# ⛔ WIDENED FROM FOUR TO NINE ON 2026-09-23. n07.CLEANUPS names eleven that
# day. Of those, seven hand back a full-length cleaned label column and one
# hands back a row mask, and every one of those eight can be given to the
# search -- as `train_labels=` for a column, as the kept rows for a mask. n20
# was running four of them.
#
# THE TWO LEFT OUT ARE `huber_weight` AND `invvar_weight`, and the reason is
# the value they return. Each one's dict is `sample_weight` and nothing else:
# `keep` is None and `label_column` is None (n07_remediation.py:682-683 and
# :729-730). The branch at the cleaned search reads exactly those two entries
# -- `if col is not None or keep is not None` -- so it never fires, and
# `search_cleaned__fit_cleaned` and `search_cleaned__fit_asis`, the two arms
# this file exists to compare, are undefined for them. `invvar_weight` also
# weights by a per-molecule measurement count that arrives as
# `measurement_counts`, which n20 never has, so it would decline on every
# fold. `downweight` returns the same three values as huber_weight and stays
# only because it is already scored on the two fit-side arms and the smoke
# reads it there.
#
# Not left out for being expected to lose. Anything here can be added back by
# passing `cleanups=` to run().
CLEANUPS = (
    # Hands back a row mask.
    'remove',
    # Hands back a full-length cleaned label column.
    'label_smooth', 'knn_smooth', 'prox_smooth',
    # The same, and each is REGRESSION ONLY -- `cleanups_for` drops these four
    # on a classification dataset, because `cleaned_training_set` raises on a
    # nominal class label rather than approximating a number.
    # 2026-10-05 (f36): quantile_clip cut, constant_lift added.
    'oof_smooth', 'gp_smooth', 'constant_lift', 'js_shrink',
    # Hands back a per-row weight and nothing the search can read. Two arms of
    # its four, and the absence is recorded rather than left blank.
    'downweight',
)
# DECIDED 2026-10-06 (Adelaide delegated): the random-flag controls and the
# flag rates stay as they are. The random-flag twins (RANDOM_FLAG_CLEANUPS
# below) are not in this tuple, so they run only when a job names them in
# `cleanups=`.


def cleanups_for(is_clf, cleanups=None):
    """The cleanups this task type can actually run, and what it dropped.

    Returns (run, dropped). A regression-only cleanup on a classification
    dataset is dropped here rather than raised inside the fold loop:
    `cleaned_training_set` raises ValueError on it (n07_remediation.py:879-882)
    and n20's own try/except would then file one error row per (fold, flag
    rate) for a cleanup that was never going to run.

    A random-flag name is dropped when the cleanup it runs is dropped.

    knn_smooth is dropped on a class label too (2026-10-05, dirty-run bug 13):
    there it replaces the flagged share with the majority class of the same
    neighbours label_smooth reads, so the two are one computation
    (n07_remediation.cleaned_training_set) and the dirty run gave identical
    numbers for both. label_smooth is the one kept.
    """
    asked = tuple(cleanups if cleanups is not None else CLEANUPS)
    if not is_clf:
        return asked, ()
    run, dropped = [], []
    for c in asked:
        strategy = RANDOM_FLAG_CLEANUPS.get(c, c)
        (dropped if (strategy in REGRESSION_ONLY_CLEANUPS
                     or strategy in CLASS_LABEL_DUPLICATES)
         else run).append(c)
    return tuple(run), tuple(dropped)


# Cleanups that compute exactly what another kept cleanup computes on a class
# label. See cleanups_for.
CLASS_LABEL_DUPLICATES = ('knn_smooth',)

# THE RANDOM-FLAG CONTROL. Each name on the left runs the cleanup on the right,
# with the detector's per-molecule suspicion score replaced by a seeded
# permutation of itself. The same number of training molecules are flagged, the
# same cleanup is applied to them, and the only thing that changes is WHICH
# molecules get flagged.
#
# Why a permuted FLAG SET and not a random DROP. n20's headline is not a
# cleaning-against-no-cleaning count: it is a count of held-out folds where
# searching the recipe on the cleaned labels beat searching it on the labels as
# they are. Both sides of that pair flag the same molecules and differ only in
# which label column the search was handed. So the control that matches it
# feeds a random flag set to BOTH sides, and one run of it covers the ordering
# gain and the fourth arm at once.
#
# ⛔ BOTH NAMES SHARE ONE PERMUTATION PER FOLD, seeded from the regime, the
# level, the draw and the fold index. That is what the detector-driven pair
# does: one suspicion score per fold serves every cleanup. Two permutations
# would make the pair differ in a second way.
# WHICH CLEANUPS GET ONE, and the rule is the same for all nine: a cleanup gets
# a random-flag twin when the suspicion score is what decides what it does to
# the labels AND it has a cleaned search for the shuffled score to reach. Two
# do not qualify, and neither is a cut:
#
#   constant_lift  reads no suspicion score at all: it raises the labels at
#                  the ceiling the labels show, from the labels and an
#                  out-of-fold prediction. n07 names it in
#                  SCORE_FREE_CLEANUPS. Shuffling the score would hand back
#                  the identical column, so its twin would be the same fit run
#                  twice. Degenerate. (quantile_clip, which it replaced on
#                  2026-10-05, was left out for the same reason.)
#   downweight     has no searched-on-cleaned arm, and the note below says the
#                  control feeds one random flag set to BOTH sides of the pair.
#                  With one side missing there is no pair to feed. Undefined.
#
# `label_smooth` had no twin before 2026-09-23 and qualifies under the same
# rule as the rest, so it has one now.
RANDOM_FLAG_CLEANUPS = {
    'remove_random_flag': 'remove',
    'knn_smooth_random_flag': 'knn_smooth',
    'label_smooth_random_flag': 'label_smooth',
    # prox_smooth_random_flag was here until 2026-10-05. Since f36 the
    # regression prox_smooth moves every label and reads no score, so its
    # twin would be the same fit run twice (the constant_lift case above).
    'oof_smooth_random_flag': 'oof_smooth',
    'gp_smooth_random_flag': 'gp_smooth',
    'js_shrink_random_flag': 'js_shrink',
}

# Fixed so the permutation is the same on a re-run. Changing it re-runs every
# random-flag search from cold: the flagged rows change, so the cleaned label
# column changes, so its cache identity changes.
RANDOM_FLAG_SEED = 20260923

# How many training rows in every hundred are treated as suspect. n18's, for
# n18's reason: n07 sets this to the share it corrupted, and nobody can do that
# here because the searched-on-cleaned arms have to run in the regime where
# nothing was corrupted too.
FLAG_RATES = (0.05, 0.10, 0.20)

# The four ways of running it. The name says what the features were chosen from
# and what the model was fitted on, in that order.
ARMS = ('search_asis__fit_asis',
        'search_asis__fit_cleaned',
        'search_cleaned__fit_cleaned',
        'search_cleaned__fit_asis')

# The arm the suite runs today, and the arm the question is about. The
# difference between them is what the ordering costs.
ARM_TODAY = 'search_asis__fit_cleaned'
ARM_ORDERED = 'search_cleaned__fit_cleaned'

# Which regimes run. Both, because they answer different questions.
REGIMES = ('injected', 'as_they_come')

# The condition the injected regime runs under when the caller names none. It
# must give a corrupted SUBSET, which is what makes
# `noise_helpers.detection_scores` defined -- under gaussian and
# grouped_shifted every training molecule moves and precision is 1.0 by
# construction. outlier_p10 until 2026-10-05; it is one of the sixteen kinds
# the noise grid cut (KNN_SEARCH_SPACE_RECORD, Adelaide 2026-10-05), and
# outlier_p05 is the kept member of its family.
# DECIDED 2026-10-06 (Adelaide delegated): the default stays outlier_p05; a
# job may name another of the six decided kinds.
DEFAULT_NOISE_CONDITION = 'outlier_p05'

# How many times the corruption is re-drawn at each level. Three, matching n07,
# n09, n11, n12, n13, n15 and n17.
NOISE_DRAWS = (0, 1, 2)

# The detector whose per-molecule suspicion score decides what gets flagged.
# n07's own, because these are n07's cleanups.
#
# ⛔ THE NAME WAS WRONG IN THE RESULT FILE UNTIL 2026-09-20. The file wrote
# 'activity_cliff_aware' and 'detector_k': 10. The call is
# `_compute_noise_scores(X_base_tr, y0, is_clf=is_clf)` with no `detector=`, so
# it takes n07's own default, which is `cross_val` — n07_remediation.py:274.
# That resolves to CrossValResidualEstimator, whose __init__ takes n_folds,
# normalize, groups and random_state and no neighbour count at all, so the
# `k: 10` that _compute_noise_scores offers is dropped by its `accepts` filter
# (n07_remediation.py:300-302). Two things written to the result disagreed with
# what ran: the detector's name, and a neighbour count the estimator never saw.
# The name is now passed explicitly, so the label and the call cannot drift
# apart again.
DETECTOR = 'cross_val'

# What every candidate inside the cleaned search is SCORED on.
# DECIDED 2026-10-06 (Adelaide delegated): 'asis'. The cleaned search TRAINS on
# the cleaned column (passed as train_labels= to the per-fold search helpers)
# and is SCORED on the training labels the practitioner holds: the injected
# labels under the injected regime, the labels as delivered under as_they_come
# (passed as the label column). 'cleaned' (train and score on the cleaned
# column, the default from 2026-10-05 to 2026-10-06) can still be named.
# Only the smoothers see this -- `remove` drops rows and its search is scored
# and trained on the kept rows' labels either way.
SEARCH_SCORES_ON = 'asis'

# Whose 2 best single representations, in each fold's search on the labels as
# they are, the detector and the cleanups run in. 'rf' until 2026-10-05 (owner's
# decision 2026-09-25); rf is not one of the dataset's two models on most
# datasets now, and the per-fold search does not score it.
# DECIDED 2026-10-06 (Adelaide delegated): None = the first of the dataset's
# two decided models (decided_grid.decided_models order), as before.
CLEANING_RANK_MODEL = None

# How many single representations the cleaning is decided in, per fold.
NOISE_BASE_REP_COUNT = 2

# Bumped whenever what the checkpoint and the rows hold changes shape, so an
# old snapshot is never resumed. 2026-10-05 (chat 5): 8 configurations per fold
# from per-fold searches on the decided grid, keyed <model>__<configuration>.
# 2026-10-06: cleaned searches scored on the labels as they are, default-path
# neighbour cleanups read the fold's top screened setting, one draw under
# censoring, and the checkpoint holds cleaning_neighbours_by_fold.
RESULT_LAYOUT = 'decided_2026_10_06'


def _levels_and_draws(data, dataset_name, condition, levels=None, draws=None):
    """(level, draw) pairs for the injected regime.

    The decided levels for this kind and dataset group
    (noise_helpers.decided_levels; classification keeps its flip levels).
    Level 0 is the labels as delivered on every task type since 2026-10-05
    (noise_spec, dirty-run bug 6), so it runs once, not once per draw; every
    other level runs NOISE_DRAWS draws.

    `levels` and `draws` exist for the smoke, which checks the machinery rather
    than the grid. A run that narrows either writes what it ran to the result
    file, so a short grid reads as a short grid and not as the shared one.
    """
    levels = (decided_levels(data, dataset_name, condition)
              if levels is None else levels)
    # One draw under censoring (noise_helpers.draws_for, 2026-10-06).
    NOISE_DRAWS_HERE = draws_for(condition,
                                 NOISE_DRAWS if draws is None else draws)
    out = []
    for lvl in levels:
        here = (NOISE_DRAWS_HERE if float(lvl) > 0 else NOISE_DRAWS_HERE[:1])
        for d in here:
            out.append((float(lvl), int(d)))
    return out


def _fit_score(factory, X_tr, y_tr, weights, X_va, y_va, metric_fn,
               metric_name):
    """One fit, one held-out score against the untouched labels, or (None, why)."""
    try:
        m = factory()
        if weights is None:
            m.fit(X_tr, y_tr)
        else:
            m.fit(X_tr, y_tr, sample_weight=weights)
        return float(metric_fn(y_va, _predict_for_metric(m, X_va, metric_name))), None
    except TypeError as exc:
        if weights is not None and 'sample_weight' in str(exc):
            return None, 'refuses_sample_weight'
        return None, f'{type(exc).__name__}: {exc}'
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'


def _label_column_key(y):
    """8 hex characters of the training label column this search will see.

    Added 2026-09-18. The cleaned search used to be namespaced on the flag rate.
    Two of n20's four cleanups ignore the rate and hand back an identical column
    at all three of theirs, so three identical searches filed under three
    namespaces and each paid in full. Keying on the column itself collapses them
    to one and leaves every genuinely different column on its own namespace.

    float64 and C-order before hashing, so a column that is equal is also
    byte-equal and two runs agree.
    """
    import hashlib
    a = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    return hashlib.sha1(a.tobytes()).hexdigest()[:8]


def _permuted_suspicion(suspicion, cell_key):
    """The same suspicion scores, in a shuffled order over the training rows.

    Every cleanup in n07 turns the score into a flag by thresholding it at the
    (1 - flag rate) quantile, so a permutation of the score flags exactly as
    many training molecules as the real one does and flags a different set.
    Nothing else about the cleanup changes.

    `cell_key` names the regime, the level, the corruption draw and the
    held-out fold. It seeds the permutation, so the two random-flag cleanups
    inside one fold receive the same shuffled order, a re-run reproduces it,
    and two folds do not share one.
    """
    import hashlib
    h = hashlib.sha1(f'{RANDOM_FLAG_SEED}|{cell_key}'.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], 'big'))
    order = rng.permutation(len(suspicion))
    return np.asarray(suspicion)[order], order


def _fingerprint(best):
    """The parts of a recipe that say what it selected. n15's, at :97-106."""
    if not best:
        return None
    bp = best.get('blueprint')
    if bp is None:
        return None
    return {
        'reps': sorted(getattr(bp, 'reps', []) or []),
        'allocation_method': getattr(bp, 'allocation_method', None),
        'importance_method': getattr(bp, 'importance_method', None),
        'budget': getattr(bp, 'budget', None),
        'internal_evaluator': getattr(bp, 'internal_evaluator', None),
        'rep_count': getattr(bp, 'rep_count', None),
    }


def _rank_stored(rows, lower_is_better, top_n):
    """The best `top_n` stored configurations, in the dataset's own direction.

    ⛔ DIRECTION. `score_mean` is a number IN the dataset's metric. flab is
    spearman and higher wins; herg_ki and caco2 are rmse and lower wins.
    Sorting the wrong way round would take the WORST configurations and call
    them the top, and every feature count built on them would be inverted.
    """
    ordered = sorted(rows, key=lambda r: float(r['score_mean']),
                     reverse=not lower_is_better)
    return ordered[:int(top_n)]


def _rep_scores_from_search(search):
    """The per-representation scores every STORED blueprint lost.

    ⛔ WHAT IS BROKEN, and it is not in this file. `find_best_hybrid` attaches
    `rep_scores` to each blueprint at find_best_hybrid.py:9312. It writes its
    rows to the result store at :7245, :7403, :7442 and :9215. All four writes
    happen before that line, and the blueprint is pickled at the write. So
    every blueprint in every `result_db/*.db` has `rep_scores = None`.

    What that costs: `build_fold_hybrid` refuses to rebuild a recipe whose
    `rep_count` names fewer representations than the pool unless something
    says WHICH ones (tests/experiments/_shared.py:5095). It refuses rather
    than rebuilding on the whole pool and filing the result under the
    winner's name, which is the right refusal. But read back from the store,
    EVERY such recipe raises. Measured on this dataset 2026-09-21: of the 5
    configurations stored across the fold-0 and fold-1 searches, the ones
    carrying rep_count 3 and rep_count 5 could not be rebuilt at all, and
    those are the configurations that make n20's question answerable.

    The scores are a property of the SEARCH and not of one configuration --
    each representation's best baseline over every row the search ran on --
    and every blueprint in one search carries the same representation list.
    So one dict serves all of that search's stored rows, and it is lifted
    from the in-memory result rather than recomputed from the baselines.

    Returns the dict, or None when this search's own blueprints have none
    either, in which case nothing here can supply it.
    """
    if search is None:
        return None
    seen = list(search.all_results or [])
    for by_model in (getattr(search, 'best_by_model', None) or {}).values():
        if isinstance(by_model, dict):
            seen.append(by_model)
    for r in seen:
        bp = r.get('blueprint') if isinstance(r, dict) else None
        scores = getattr(bp, 'rep_scores', None)
        if scores:
            return dict(scores)
    return None


def _feature_frequency(top_rows, train_reps, val_reps, y_col, groups_train,
                       model_factory, metric_fn, lower_is_better,
                       rep_scores=None):
    """How many of these hybrids selected each feature.

    A feature is the PAIR (representation, column index inside that
    representation), because `selected_indices` are LOCAL to each
    representation's own matrix (hybrid.py:2236). saprot's column 7 and
    onehot's column 7 are two features.

    Returns (counts, n_built, widths, n_refused). `counts` maps the pair to
    how many of the hybrids that BUILT selected it. `n_built` is how many of
    the configurations handed in produced a hybrid at all, which is the
    denominator every frequency below is out of. `n_refused` is how many were
    not built because the recipe named a representation count and nothing
    could say which representations it meant.
    """
    counts = {}
    widths = {}
    n_built = 0
    n_refused_for_missing_rep_scores = 0
    for row in top_rows:
        bp = row['blueprint']
        # The store lost them; see _rep_scores_from_search. Without this the
        # build raises for every recipe that used fewer than all the pool's
        # representations, which is the axis the question rides on.
        if (getattr(bp, 'rep_count', None) is not None
                and not getattr(bp, 'rep_scores', None)):
            if rep_scores:
                bp.rep_scores = dict(rep_scores)
            else:
                n_refused_for_missing_rep_scores += 1
                continue
        out = hybrid_for_fold(train_reps, val_reps, y_col, bp,
                              groups_train, model_factory(), metric_fn,
                              lower_is_better, want_feature_info=True)
        feat_info = out[2] if len(out) == 3 else None
        if feat_info is None:
            continue
        n_built += 1
        for rep_name, info in feat_info.items():
            if rep_name == 'allocation' or not isinstance(info, dict):
                continue
            idxs = info.get('selected_indices')
            if idxs is None:
                continue
            for i in np.asarray(idxs).ravel().tolist():
                counts[(rep_name, int(i))] = counts.get((rep_name, int(i)), 0) + 1
    for rep_name, mat in train_reps.items():
        widths[rep_name] = int(np.asarray(mat).shape[1])
    return counts, n_built, widths, n_refused_for_missing_rep_scores


def _compare_feature_tables(counts_asis, n_asis, counts_cleaned, n_cleaned,
                            widths, n_named=15):
    """The two frequency tables read against each other.

    Every count names what it counts. Nothing here is a bare number.
    """
    sel_a = set(counts_asis)
    sel_c = set(counts_cleaned)
    both = sel_a & sel_c
    only_a = sel_a - sel_c
    only_c = sel_c - sel_a
    total_width = int(sum(widths.values()))

    def _named(pair):
        return {'representation': pair[0],
                'index_within_that_representation': int(pair[1])}

    moved = []
    for pair in sel_a | sel_c:
        a = counts_asis.get(pair, 0)
        c = counts_cleaned.get(pair, 0)
        moved.append((abs(a - c), pair, a, c))
    moved.sort(key=lambda t: (-t[0], t[1][0], t[1][1]))

    def _top_list(counts, n_built, k):
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
        return [dict(_named(pair),
                     n_top_hybrids_that_selected_it=int(c),
                     n_top_hybrids_built=int(n_built))
                for pair, c in ordered[:k]]

    most_a = _top_list(counts_asis, n_asis, n_named)
    most_c = _top_list(counts_cleaned, n_cleaned, n_named)
    set_most_a = {(d['representation'], d['index_within_that_representation'])
                  for d in most_a}
    set_most_c = {(d['representation'], d['index_within_that_representation'])
                  for d in most_c}

    return {
        'n_features_the_representations_hold_between_them': total_width,
        'n_features_per_representation': dict(sorted(widths.items())),
        'n_top_hybrids_built_searching_on_the_labels_as_they_are': int(n_asis),
        'n_top_hybrids_built_searching_on_the_cleaned_labels': int(n_cleaned),
        'n_features_selected_by_at_least_one_top_hybrid_asis': len(sel_a),
        'n_features_selected_by_at_least_one_top_hybrid_cleaned': len(sel_c),
        'n_features_both_searches_selected': len(both),
        'n_features_only_the_search_on_the_labels_as_they_are_selected': len(only_a),
        'n_features_only_the_search_on_the_cleaned_labels_selected': len(only_c),
        'features_whose_selection_frequency_moved_most': [
            dict(_named(pair),
                 n_top_hybrids_that_selected_it_asis=int(a),
                 n_top_hybrids_that_selected_it_cleaned=int(c),
                 n_top_hybrids_built_asis=int(n_asis),
                 n_top_hybrids_built_cleaned=int(n_cleaned))
            for _, pair, a, c in moved[:n_named]],
        'most_selected_searching_on_the_labels_as_they_are': most_a,
        'most_selected_searching_on_the_cleaned_labels': most_c,
        'n_of_the_most_selected_features_that_are_in_both_lists':
            len(set_most_a & set_most_c),
        'n_features_in_each_most_selected_list': n_named,
    }


def _with_train_labels(data, rows, y_rows, is_clf):
    """`data` with the labels at `rows` replaced by `y_rows`. Added 2026-10-05
    (chat 5).

    The per-fold searches (noise_helpers.fold_configurations,
    _shared.fold_standard_selection) read `data['labels'][train_idx]` and
    nothing else of the label column, so this is how a search is handed the
    labels n20 holds for one fold: the injected ones, or a cleaned column, or
    the kept rows' labels. The held-out rows keep their delivered labels; no
    search reads them. Shallow copy: the representations are shared, not
    copied."""
    lab = np.array(data['labels'], copy=True)
    if not is_clf:
        lab = lab.astype(np.float64)
    lab[np.asarray(rows)] = np.asarray(y_rows)
    out = dict(data)
    out['labels'] = lab
    return out


def _rows_key(rows):
    """8 hex characters of the training rows a search ran on."""
    import hashlib
    a = np.ascontiguousarray(np.asarray(rows, dtype=np.int64))
    return hashlib.sha1(a.tobytes()).hexdigest()[:8]


def _general_search(data_x, dataset_name, *, fold_i, train_idx, n_folds,
                    condition_suffix, train_labels=None):
    """The per-fold general search behind a selection, for the candidate
    counts and the per-feature table. Same arguments as the selection's own
    call, so it is the in-process memo's entry, not a second search.
    train_labels: full-length column the search trains on while it is scored
    on data_x['labels'] (None: trains on those too)."""
    try:
        got = fold_searches(
            data_x['reps'], np.asarray(data_x['labels']), data_x.get('groups'),
            train_idx=np.asarray(train_idx), fold_i=int(fold_i),
            n_outer_folds=int(n_folds), dataset_name=dataset_name,
            augmentations=data_x.get('augmentations'),
            task_type=data_x.get('task_type') or 'auto',
            condition_suffix=condition_suffix, train_labels=train_labels)
    except Exception as exc:
        print(f"      general search unreadable: {type(exc).__name__}: {exc}")
        return None
    return got.get('none')


def _selection_entries(data_x, dataset_name, *, fold_i, train_idx, n_folds,
                       condition_suffix, train_labels=None):
    """Each decided model's 2 base representations and 2 hybrids for one fold,
    in the form noise_helpers.fold_model_inputs reads. Added 2026-10-05
    (chat 5).

    fold_standard_selection on the dataset's decided grid, pruning off. Used
    where only the 4 fitted configurations per model are needed (every cleaned
    search, and the uncleaned one when no neighbour setting is swept), so the
    distance and similarity searches fold_configurations adds for the
    neighbour spaces are not run once per cleaned label column.

    train_labels: a full-length column the search trains on while every
    candidate is scored on data_x['labels'] (2026-10-06: the cleaned column,
    scored on the labels as they are). None trains on data_x['labels']."""
    sel = fold_standard_selection(
        data_x['reps'], np.asarray(data_x['labels']), data_x.get('groups'),
        train_idx=np.asarray(train_idx), fold_i=int(fold_i),
        n_outer_folds=int(n_folds), dataset_name=dataset_name,
        augmentations=data_x.get('augmentations'),
        task_type=data_x.get('task_type') or 'auto',
        condition_suffix=condition_suffix, train_labels=train_labels)
    out = {}
    for m, s_m in sel.items():
        if s_m is None:
            out[m] = None
            continue
        recipes, unavailable = {}, {}
        for space, i in (('hybrid', 0), ('hybrid__general2', 1)):
            if len(s_m['hybrids']) > i:
                h = s_m['hybrids'][i]
                recipes[space] = (h['blueprint'], h.get('aug_condition', 'none'))
            else:
                unavailable[space] = (f'the per-fold general search gave this '
                                      f'model fewer than {i + 1} hybrids with '
                                      f'distinct features')
        out[m] = {
            'base': s_m['base'],
            'hybrids': s_m['hybrids'],
            'hybrids_skipped_identical_features':
                s_m.get('hybrids_skipped_identical_features', []),
            'fit_configurations': (list(s_m['base'])
                                   + [h['label'] for h in s_m['hybrids']]),
            'recipes': recipes,
            'unavailable': unavailable,
            'selection_scope': 'fold_train_rows',
        }
    return out


def _base_scores(search, model, base_names):
    """Each chosen base representation's score in the per-fold search: its
    mean over the search's inner folds of this fold's training rows."""
    out = {}
    bl = getattr(search, 'baselines', None) or {}
    for rn in base_names or []:
        v = (bl.get(rn) or {}).get(model) if isinstance(bl.get(rn), dict) else None
        try:
            out[rn] = None if v is None else float(v)
        except (TypeError, ValueError):
            out[rn] = None
    return out


def _selection_fingerprint(entry):
    """What one model's selection picked: its 2 base representations and what
    each of its 2 hybrids selected. Two searches handed the same labels must
    give the same value."""
    if entry is None:
        return None
    return {'base': list(entry.get('base') or []),
            'hybrids': [_fingerprint({'blueprint': h.get('blueprint')})
                        for h in (entry.get('hybrids') or [])]}


def _search_configurations(search, model):
    """Every configuration one per-fold search scored for one downstream model,
    from its all_results. Since 2026-10-05 (chat 5) this replaces the read of
    the search's own result file: the per-fold searches are filed by fold and
    condition suffix, not under an n20 namespace, and the engine appends every
    stored row the call asked for to all_results (find_best_hybrid.py:7551),
    so nothing is missing from it."""
    if search is None:
        return [], 'no search'
    rows = [r for r in (search.all_results or [])
            if str(r.get('model')) == str(model)
            and r.get('score_mean') is not None
            and r.get('blueprint') is not None]
    return rows, (f'{len(rows)} configurations this per-fold search scored '
                  f'for this downstream model')


def _resolve_condition(data, asked):
    """The damage kind the injected regime runs under. Added 2026-10-05
    (chat 5).

    Regression: one of the six decided kinds (noise_helpers.
    decided_conditions); a cut kind raises. outlier_p10, this file's default
    until 2026-10-05, is one of the sixteen cut, so the default is now
    outlier_p05, the kept kind of the same family (KNN_SEARCH_SPACE_RECORD,
    the noise grid: "only a subset is corrupted, so detection is defined").
    Classification: a flip kind; anything else is replaced by the default flip
    kind, which is what noise_spec_for did silently before, and the result file
    names both."""
    from kirby.noise_spec import CLASSIFICATION_STRATEGIES, DEFAULT_STRATEGY_CLF
    if data.get('task_type') == 'classification':
        return asked if asked in CLASSIFICATION_STRATEGIES else DEFAULT_STRATEGY_CLF
    cond = asked or DEFAULT_NOISE_CONDITION
    if cond not in decided_conditions(data):
        raise ValueError(
            f"n20: noise condition {cond!r} is not one of the six decided kinds "
            f"{decided_conditions(data)} (KNN_SEARCH_SPACE_RECORD, the noise "
            f"grid, Adelaide 2026-10-05).")
    return cond


def run(data, config, output_dir, dataset_name, **kwargs):
    """N20: does the cleaning have to come before the recipe search."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N20: DOES THE CLEANING HAVE TO COME BEFORE THE SEARCH")
    print("=" * 70)

    labels = np.asarray(data['labels'])
    reps = data['reps']
    groups = data.get('groups')
    groups_arr = None if groups is None else np.asarray(groups)
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    metric = (metric_name, metric_fn, lower_is_better)

    condition_asked = kwargs.get('noise_condition') or None
    condition = _resolve_condition(data, condition_asked)
    regimes = tuple(kwargs.get('regimes') or REGIMES)
    # Whatever the caller asked for is filtered by task type, not only the
    # default list: a caller naming oof_smooth on a classification dataset is
    # naming something `cleaned_training_set` refuses.
    cleanups, cleanups_dropped = cleanups_for(
        is_clf, kwargs.get('cleanups') or None)
    flag_rates = tuple(float(r) for r in (kwargs.get('flag_rates') or FLAG_RATES))
    # 2026-10-05 (chat 5): the searches are the per-fold searches on the
    # decided grid (DECISIONS.md), pruning off. A caller's budgets, inner fold
    # count, search settings or pruning mode no longer reach them.
    for _k in ('budgets', 'search_n_folds', 'search_settings', 'pruning_mode',
               'cache', 'cache_dir'):
        if kwargs.get(_k) is not None:
            print(f"  n20: {_k}={kwargs[_k]!r} ignored; the per-fold searches "
                  f"run on the dataset's decided grid, unpruned")
    # HOW MANY of each search's configurations the per-feature table is built
    # over. 0 turns the whole per-feature measurement off, which is what every
    # run before 2026-09-21 did. It is off by default because each
    # configuration in the list has to be BUILT to find out which features it
    # selected -- a Blueprint does not carry them -- and a build is a feature
    # ranking plus an accept/reject pass per representation.
    per_feature_top_n = int(kwargs.get('per_feature_top_n') or 0)
    search_scores_on = str(kwargs.get('search_scores_on') or SEARCH_SCORES_ON)
    if search_scores_on not in ('asis', 'cleaned'):
        # DECIDED 2026-10-06 (Adelaide delegated): 'asis' is the default. The
        # per-fold search helpers pass train_labels= through since 84e6ce2, so
        # the cleaned search trains on the cleaned column and every candidate
        # is scored on the labels as they are.
        raise ValueError(
            f"n20: search_scores_on={search_scores_on!r}; must be 'asis' "
            f"(default) or 'cleaned'.")

    # Models: the dataset's two decided models (DECISIONS.md Step 1), nothing
    # else (2026-10-05, chat 5). knn is not one: it is fitted only on the
    # distance search, which n20 reads as a neighbour space only.
    models = [m for m in decided_models(dataset_name, 'general') if m != 'knn']
    if kwargs.get('models'):
        asked = list(kwargs['models'])
        models = [m for m in models if m in asked]
        if not models:
            raise RuntimeError(
                f"n20: none of {asked} is one of {dataset_name}'s decided "
                f"models {decided_models(dataset_name, 'general')}.")
    model_factories = get_model_factories(
        models, task_type=task_type, n_samples=len(labels))
    models_without_factory = [m for m in models if m not in model_factories]
    models = [m for m in models if m in model_factories]
    if not models:
        raise RuntimeError(
            f"n20: no decided model could be built for {dataset_name}: "
            f"{models_without_factory}.")
    cleaning_rank_model = CLEANING_RANK_MODEL or models[0]

    noise_groups = noise_groups_for(data)[0]
    folds, n_folds = noise_outer_folds(data, config)

    smiles = data.get('smiles')
    has_smiles = smiles is not None and len(smiles) > 0
    n_molecules = len(set(smiles)) if has_smiles else len(labels)
    sweep = n07_sweep_settings(dataset_name, n_molecules, has_smiles=has_smiles)

    # The control runs first, so a run that stops early still carries the check
    # that the two searches are one code path.
    run_cleanups = (IDENTITY_CONTROL,) + tuple(cleanups)

    # Which cleanups can reach a feature selection at all, and how. Written to
    # the result so an absent arm reads as a recorded refusal.
    reach = {
        IDENTITY_CONTROL: 'identity: the cleaned column IS the uncleaned one',
        'remove': 'drops rows; the search and the ranking run on the kept rows',
        'downweight': (
            'sets a per-row weight, and sample_weight appears zero times in '
            'src/kirby/hybrid.py and zero times in '
            'src/kirby/find_best_hybrid.py (checked 2026-09-17). Nothing in '
            'the feature selection reads a weight, so its two '
            'searched-on-cleaned arms are undefined, not zero.'),
        'label_smooth': 'produces a full-length cleaned label column',
        'knn_smooth': 'produces a full-length cleaned label column',
        'prox_smooth': (
            'produces a full-length cleaned label column. The neighbours a '
            'flagged label is rewritten from are the molecules a forest keeps '
            'putting in the same leaves, not the nearest in Euclidean '
            'distance'),
        'oof_smooth': (
            'produces a full-length cleaned label column. Every label moves '
            'toward what a LightGBM trained on the other training molecules '
            'predicts for it, in proportion to its suspicion score'),
        'gp_smooth': (
            'produces a full-length cleaned label column, aiming at a '
            'Gaussian process posterior mean. Refused, and the fold recorded '
            'as refused, above GP_SMOOTH_MAX_ROWS training molecules'),
        'js_shrink': (
            'produces a full-length cleaned label column, aiming at the same '
            'out-of-fold prediction oof_smooth uses. How far each label moves '
            'is the ratio of two variances rather than a chosen fraction'),
        'huber_weight': (
            'NOT RUN. Returns sample_weight and nothing else -- keep is None '
            'and label_column is None -- so no cleaned search can be built '
            'and both searched-on-cleaned arms are undefined. Named here so '
            'its absence reads as a recorded refusal rather than an '
            'oversight.'),
        'invvar_weight': (
            'NOT RUN, for that reason and one more: it weights by a '
            'per-molecule measurement count, which n20 does not have, and it '
            'declines rather than returning a flat weight.'),
        'remove_random_flag': (
            'drops rows, exactly as `remove` does and exactly as many, chosen '
            'by a shuffled suspicion score instead of the detector'),
        'knn_smooth_random_flag': (
            'produces a full-length cleaned label column, exactly as '
            '`knn_smooth` does and moving exactly as many labels, chosen by a '
            'shuffled suspicion score instead of the detector'),
        'label_smooth_random_flag': (
            'produces a full-length cleaned label column, exactly as '
            '`label_smooth` does, with each label moved by a shuffled '
            "molecule's suspicion score instead of its own"),
        'constant_lift': (
            'produces a full-length cleaned label column. The labels sitting '
            'at the ceiling the labels themselves show are raised by one '
            'constant; nothing else moves, and it declines where the labels '
            'show no ceiling. It reads no suspicion score, so it has no '
            'random-flag twin (f36, 2026-10-05)'),
        'oof_smooth_random_flag': (
            'produces a full-length cleaned label column, exactly as '
            '`oof_smooth` does and aiming at the same out-of-fold prediction, '
            "with each label moved by a shuffled molecule's suspicion score"),
        'gp_smooth_random_flag': (
            'produces a full-length cleaned label column, exactly as '
            '`gp_smooth` does and aiming at the same posterior mean, with '
            "each label moved by a shuffled molecule's suspicion score"),
        'js_shrink_random_flag': (
            'produces a full-length cleaned label column, exactly as '
            '`js_shrink` does and aiming at the same out-of-fold prediction, '
            "with each label's shrinkage derived from a shuffled molecule's "
            'suspicion score'),
    }

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'selection_scope': 'fold_train_rows',
        'n_folds': int(n_folds),
        'folds_run': [int(f) for f, _, _ in folds],
        'regimes': list(regimes),
        'noise_condition': condition,
        'noise_condition_asked': condition_asked,
        'noise_level_unit': level_unit(condition, is_clf),
        'cleanups': list(run_cleanups),
        'cleanups_not_run': list(cleanups_dropped),
        'cleanups_not_run_note': (
            'Named on a classification dataset only. oof_smooth, gp_smooth, '
            'js_shrink and constant_lift each move a label to a NUMBER, and '
            'cleaned_training_set raises on a nominal class label rather than '
            'approximating one (n07 REGRESSION_ONLY_CLEANUPS, imported). '
            'knn_smooth (and its random-flag twin) is left out on a class '
            'label because there it computes exactly what label_smooth '
            'computes: the flagged share replaced by the majority class of '
            'the same neighbours (n07_remediation.cleaned_training_set; '
            'dirty-run bug 13, 2026-10-05).'),
        'cleanups_n07_offers_that_n20_does_not_run': {
            'huber_weight': (
                'Returns sample_weight and nothing else: keep is None and '
                'label_column is None, so the cleaned recipe search is never '
                'built and the two arms this experiment compares -- '
                'search_cleaned__fit_cleaned against search_asis__fit_cleaned '
                '-- do not exist for it. downweight has the identical return '
                'value and is kept only because it was already scored on the '
                'two fit-side arms.'),
            'invvar_weight': (
                'The same, and one more: it weights by how many times each '
                'molecule was measured, which reaches cleaned_training_set as '
                'measurement_counts. n20 has no such count, and the cleanup '
                'declines rather than returning a flat weight, so it would '
                'produce nothing on every fold.'),
        },
        'identity_control': IDENTITY_CONTROL,
        'flag_rates': list(flag_rates),
        'arms': list(ARMS),
        'arm_the_suite_runs_today': ARM_TODAY,
        'arm_the_question_is_about': ARM_ORDERED,
        'configurations_note': (
            'Per held-out fold, per side (searched on the labels as they are, '
            'searched on the cleaned labels), each decided model is fitted in '
            'its 2 best base representations and its 2 best hybrids, all '
            'ranked by a per-fold search on that side\'s training rows and '
            'labels (fold_standard_selection, decided grid, pruning off): '
            '2 models x 4 = 8 configurations. Each is run in all four arms. '
            'Keys in per_regime are <model>__<configuration>. The distance '
            'hybrid and the similarity hybrid are neighbour spaces for the '
            'similarity sweep only, never fitted (2026-10-05).'),
        'models': list(models),
        'models_without_a_factory': models_without_factory,
        'cleaning_rank_model': cleaning_rank_model,
        'base_reps_note': (
            f'The detector and every cleanup ran once in each of '
            f"{cleaning_rank_model}'s 2 best single representations, ranked "
            f"by its score in each fold's per-fold search on the labels as "
            f'they are. base_reps_by_fold names them per fold, or says why a '
            f'fold had none. per_regime and summary are keyed by base_1 / '
            f'base_2 under each level.'),
        'detector': DETECTOR,
        'detector_sees_the_folds_groups': True,
        'detector_groups_note': (
            "The fold's training groups are handed to the detector. Without "
            "them CrossValResidualEstimator falls back to a plain KFold and a "
            "molecule's own cluster trains the out-of-fold forest that "
            "predicts it. Changed 2026-09-20 after measuring both ways."),
        'detector_takes_no_neighbour_count': (
            'CrossValResidualEstimator takes n_folds, normalize, groups and '
            'random_state. There is no neighbour count to record.'),
        'search_settings': {
            'grid': "the dataset's decided grid (decided_grid.json)",
            'pruning': 'off',
            'helper': ('noise_helpers.fold_configurations (searched on the '
                       'labels as they are, when neighbour settings are '
                       'swept) or _shared.fold_standard_selection'),
            'condition_suffix_rule': (
                "None when the training labels are the dataset's own "
                "(as_they_come, injected level 0), so the dataset's standard "
                "per-fold search is reused; 'n20_asis_lb<label hash>' for "
                "injected labels; 'n20_<cleanup>_r<rows hash>_lb<label hash>' "
                "for every cleaned search, the identity control included, "
                "with '_sc<hash of the labels it is scored on>' appended "
                "when it trains on the cleaned column and is scored on the "
                "labels as they are."),
        },
        'search_scores_on': search_scores_on,
        'search_scores_on_note': (
            "What every candidate inside the CLEANED search was scored on. "
            "'asis' (default, decided 2026-10-06): it trains on the cleaned "
            "column (train_labels=) and is scored on the training labels the "
            "practitioner holds -- the injected labels under the injected "
            "regime, the labels as delivered under as_they_come. 'cleaned': "
            "trained and scored on the cleaned column (2026-10-05 only). "
            "remove has one column either way: the kept rows' labels."),
        'cleanup_reaches_feature_selection': reach,
        'random_flag_cleanups': dict(RANDOM_FLAG_CLEANUPS),
        'random_flag_seed': RANDOM_FLAG_SEED,
        'random_flag_note': (
            'A cleanup named here ran with the detector output shuffled over '
            'the training rows and nothing else changed. It flags the same '
            'number of training molecules as its detector-driven twin and '
            'flags different ones. Read it against that twin, on the same '
            'held-out folds and the same corruption draws: the claim that the '
            'detector is doing the work survives only if the twin wins more '
            'folds than this does.'),
        'claim_scope': (
            'Measures what it costs to apply a cleanup AFTER the recipe is '
            'fixed rather than before it. Every score is against held-out '
            'labels that were never cleaned and never corrupted. A gain says '
            'the ordering cost something on this data; it does not say the '
            'cleanup found a wrong label.'),
        'feature_selection_labels_note': (
            "An arm's 'search' half is BOTH the label column the recipe search "
            "trained on and the label column the per-fold ranking and budget "
            "split were handed. Nothing splits those two."),
        'cleaning_decided_in_note': (
            'The suspicion score is computed once per (fold, single '
            'representation) and the same decision serves the search side and '
            'the fit side of all four arms. A cleaning that reaches the '
            'feature selection cannot have been decided in a feature space '
            'that does not yet exist.'),
        'label_smooth_ignores_flag_rate_on_regression': not is_clf,
        'models_refusing_sample_weight_note': (
            'A model named in models_refusing_sample_weight cannot take a '
            'per-molecule weight (gp), so the arms that fit on a weight '
            '(downweight) are not run for it; those cells are absent, not '
            'zero.'),
        'cleaning_neighbours_note': (
            "label_smooth and knn_smooth (and their random-flag twins) find "
            "their neighbours in the fold's top screened setting "
            "(n07.screen_settings kept[0], scored on a held-back fifth of the "
            "training molecules), in every arm and every cleaned search; "
            "prox_smooth only when that setting is forest_proximity. "
            "cleaning_neighbours_by_fold names it per (fold pass, cleaning "
            "representation), or says why none was used. Decided 2026-10-06."),
        'per_regime': {},
        'detection': {},
    }

    passes = {'as_they_come': [(None, 0)]}
    if 'injected' in regimes:
        passes['injected'] = _levels_and_draws(
            data, dataset_name, condition,
            levels=kwargs.get('levels'), draws=kwargs.get('draws'))
        results['noise_levels'] = sorted({l for l, _ in passes['injected']})
        results['noise_draws_per_level'] = {
            f'{l:g}': sorted({d for ll, d in passes['injected'] if ll == l})
            for l in results['noise_levels']}

    n_cols = 1 + len(run_cleanups) * len(flag_rates)
    n_passes = sum(len(passes[r]) for r in regimes)
    print(f"  {len(labels)} molecules, {len(folds)} of {n_folds} folds, "
          f"models {models}; detector and cleanups in "
          f"{cleaning_rank_model}'s 2 best single representations")
    print(f"  condition {condition}, cleanups {list(run_cleanups)}, "
          f"flag rates {list(flag_rates)}")
    if cleanups_dropped:
        print(f"  not run on a classification label: {list(cleanups_dropped)}")
    print("  not run at all: huber_weight and invvar_weight. Each returns a "
          "per-row\n  weight and no label column and no row mask, so neither "
          "has a cleaned search")
    print(f"  {n_passes} (level, draw) passes over {len(regimes)} regimes; "
          f"up to {n_cols} label columns per fold and single representation, "
          f"so up to {n_passes * len(folds) * (1 + 2 * (n_cols - 1))} per-fold "
          f"general searches on the decided grid, unpruned")

    # acc[regime][level_key][base_sp][cleanup][rate][<model>__<config>][arm][fold_key]
    acc = {}
    errors = {}
    detection_acc = {}
    # Under the identity control the two searches are handed byte-identical
    # label columns. What is recorded here is what they did with them: how long
    # each one's candidate list was, and what each one selected. That is the
    # check read directly rather than inferred from a score.
    identity_probe = {}
    # THE SAME THING, FOR THE CLEANUPS THAT ARE NOT THE CONTROL: does cleaning
    # first make the search select something different.
    recipe_probe = {}
    # THE PER-FEATURE VIEW. One row here is one FEATURE, a (representation,
    # index inside that representation) pair, and the number is how many of
    # the top configurations of each search selected it.
    per_feature = {}
    refuses_weight = set()
    # What each side selected per fold and model, for the result file
    # (handoff 2.4 item 11). configurations: searched on the labels as they
    # are; configurations_cleaned: one per cleaned search.
    configurations = {}
    configurations_cleaned = {}
    # The neighbour-setting screen per (fold pass, cleaning space).
    screen_tables = {}

    # ── The rows, the similarity axis and the checkpoint ──
    # ROWS. D is `search_asis__fit_asis`; D' is each of the other three arms,
    # one d_vs_dprime row per (regime, level, draw, fold, single
    # representation, cleanup, flag rate, arm, model, configuration), built
    # from the scores the loop already computes -- no refit. score_clean,
    # under the injected regime, is the configuration fitted on the CLEAN
    # training labels in the searched-on-as-it-is features.
    #
    # THE SIMILARITY SWEEP, FIT SIDE ONLY (kNN record, 2026-10-05).
    # label_smooth and knn_smooth, and prox_smooth under forest_proximity,
    # re-find their neighbours in the dataset group's settings
    # (n07.sweep_settings): the cleaning-rank model's 2 best single
    # representations, each plus augmented features, its 2 hybrids, the
    # distance hybrid and the similarity hybrid of this fold's searches, ECFP4
    # Tanimoto, forest proximity. Every (setting, neighbour count 5/10/20,
    # counting scheme rank/inverse_square) is scored with no fit
    # (n07.screen_settings) and only the best 3 are refitted, in the
    # `search_asis__fit_cleaned` arm of all 8 configurations.
    # The top screened setting (kept[0]) is what the default path cleans with,
    # so it reaches all four arms and every cleaned search; the 2nd and 3rd
    # are refitted in `search_asis__fit_cleaned` only.
    # DECIDED 2026-10-06 (Adelaide delegated): the number of cleaned searches
    # is not changed -- one per cleaned column, from the top setting only.
    rows = sx.Rows(output_dir, EXP_FULL_NAME, dataset_name,
                   condition=condition if 'injected' in regimes else None)
    results['rows'] = rows.summary()
    results['similarity_settings_swept'] = [s_.name for s_ in sweep]
    results['similarity_ks_swept'] = list(n07_sweep_ks()) if sweep else []
    results['similarity_schemes_swept'] = (list(n07_sweep_schemes())
                                           if sweep else [])
    # The neighbours used when a fold had no top screened setting; which one
    # each fold used is in cleaning_neighbours_by_fold.
    results['similarity_setting_of_the_default_path'] = dict(_DEFAULT_SIM)
    results['similarity_arms_swept'] = ['search_asis__fit_cleaned']
    sim_unavailable = {}
    # Which neighbours the default path's label_smooth / knn_smooth (and
    # prox_smooth under forest_proximity) read, per (fold pass, cleaning
    # representation): the fold's top screened setting, or why none.
    cleaning_neighbours_by_fold = {}
    ck = sx.FoldCheckpoint(
        output_dir, EXP_FULL_NAME, dataset_name,
        sx.run_signature(data, config, {
            'layout': RESULT_LAYOUT,
            'models': models, 'cleanups': run_cleanups,
            'flag_rates': flag_rates, 'regimes': regimes,
            'condition': condition, 'passes': passes,
            'n_folds': n_folds, 'folds_run': [f for f, _, _ in folds],
            'search_scores_on': search_scores_on,
            'cleaning_rank_model': cleaning_rank_model,
            'sweep': [s_.name for s_ in sweep],
            'per_feature_top_n': per_feature_top_n}),
        condition=condition if 'injected' in regimes else None)
    if ck.state is not None:
        (acc, errors, detection_acc, identity_probe, recipe_probe,
         per_feature, _rw, base_reps_by_fold, sim_unavailable,
         configurations, configurations_cleaned, screen_tables,
         cleaning_neighbours_by_fold) = ck.state
        refuses_weight = set(_rw)
    else:
        base_reps_by_fold = {}

    def _on_hybrid(cleaned, X_tr_a, y0):
        # The cleaning decided in the single representation, applied to the
        # rows of a searched-on-as-it-is configuration -- what the
        # search_asis__fit_cleaned arm fits.
        if cleaned is None:
            return None
        keep, col = cleaned['keep'], cleaned['label_column']
        if keep is not None:
            return {'X': X_tr_a[keep], 'y': y0[keep], 'sample_weight': None,
                    'keep': keep, 'label_column': None}
        return {'X': X_tr_a, 'y': y0 if col is None else col,
                'sample_weight': cleaned['sample_weight'], 'keep': None,
                'label_column': col}

    def _ident(regime, level, draw, fold_i, cleanup, rate, base_sp, base_rn,
               arm, mname_input, sim):
        return sx.identity(
            dataset=dataset_name, experiment=EXP_FULL_NAME, method=cleanup,
            method_setting=(f'arm={arm}|flag_rate={rate:g}|'
                            f'cleaned_in={base_sp}:{base_rn}|'
                            f'detector={DETECTOR}|'
                            f'search_scores_on={search_scores_on}'),
            similarity_setting=sim, model_input=mname_input,
            noise_condition=(condition if regime == 'injected'
                             else 'as_they_come'),
            level=None if level is None else float(level),
            draw=int(draw) if regime == 'injected' else None,
            fold=int(fold_i))

    def _config_record(model, entry, names, ins, unav, search):
        rec = configuration_record(entry, names)
        if entry is not None:
            rec['base_training_row_scores'] = _base_scores(
                search, model, entry.get('base'))
        rec['configurations_unavailable'] = {
            sp: why for sp, why in (unav or {}).items()
            if sp in CONFIGURATION_SPACES}
        rec['n_features'] = {sp: int(np.shape(X[0])[1])
                             for sp, X in (ins or {}).items()}
        rec['metric'] = metric_name
        rec['lower_is_better'] = bool(lower_is_better)
        return rec

    def _cleaned_side(cleanup, keep, col, *, y0, train_idx, val_idx, fold_i,
                      memo, unit_key):
        """One cleaned search and the 8 configurations it selects, built on
        the cleaned rows and labels. Memoised per (cleanup, rows, labels) in
        the fold: two flag rates that give the same column share it."""
        if keep is not None:
            tr_c = np.asarray(train_idx)[np.asarray(keep)]
            y_c = np.asarray(y0)[np.asarray(keep)]
            # The held-out rows AND every training row are transformed through
            # the hybrid built on the kept rows, so the fit-on-the-labels-as-
            # they-are arm can still train on every row.
            val_c = np.concatenate([np.asarray(train_idx), np.asarray(val_idx)])
        else:
            tr_c = np.asarray(train_idx)
            y_c = np.asarray(col)
            val_c = np.asarray(val_idx)
        suffix_c = f'n20_{cleanup}_r{_rows_key(tr_c)}_lb{_label_column_key(y_c)}'
        # DECIDED 2026-10-06 (Adelaide delegated): the cleaned search trains on
        # the cleaned column (train_labels=) and every candidate is scored on
        # the labels as they are (y0, written into the label column). A row
        # mask has no second column: the kept rows' labels are both.
        train_labels_c = None
        if keep is None and search_scores_on == 'asis':
            data_c = _with_train_labels(data, tr_c, y0, is_clf)
            train_labels_c = _with_train_labels(data, tr_c, y_c,
                                                is_clf)['labels']
            # The labels it is scored on are in the suffix too, so a search
            # scored on the cleaned column (2026-10-05) is never read back.
            suffix_c += f'_sc{_label_column_key(y0)}'
        else:
            data_c = _with_train_labels(data, tr_c, y_c, is_clf)
        if suffix_c in memo:
            return memo[suffix_c]
        side = {'suffix': suffix_c, 'train_idx': tr_c, 'y': y_c,
                'val_idx': val_c, 'entries': {}, 'search': None, 'inputs': {},
                'data': data_c}
        try:
            entries_c = _selection_entries(
                data_c, dataset_name, fold_i=fold_i, train_idx=tr_c,
                n_folds=n_folds, condition_suffix=suffix_c,
                train_labels=train_labels_c)
        except Exception as exc:
            errors[f'{unit_key}|{cleanup}|{suffix_c}|search_cleaned'] = (
                f'{type(exc).__name__}: {exc}')
            print(f"      cleaned search failed ({suffix_c}): "
                  f"{type(exc).__name__}: {exc}")
            memo[suffix_c] = None
            return None
        side['entries'] = entries_c
        side['search'] = _general_search(
            data_c, dataset_name, fold_i=fold_i, train_idx=tr_c,
            n_folds=n_folds, condition_suffix=suffix_c,
            train_labels=train_labels_c)
        recs = {}
        for m in models:
            entry = entries_c.get(m)
            if entry is None:
                errors[f'{unit_key}|{cleanup}|{suffix_c}|{m}|search_cleaned'] = (
                    'the cleaned per-fold search gave this model no selection')
            # The two hybrids come back standardised by n07._scale_pair
            # (StandardScaler, 0/1 features included); only base_1/base_2 go
            # through FeatureScaler. Shared code, not fixed here (2026-10-05).
            ins, unav, names = fold_model_inputs(
                data_c, dataset_name, m, entry, train_idx=tr_c, val_idx=val_c,
                y_train=y_c, factory=model_factories[m], metric=metric)
            side['inputs'][m] = ins
            recs[m] = _config_record(m, entry, names, ins, unav,
                                     side['search'])
        configurations_cleaned[f'{unit_key}|{cleanup}|{suffix_c}'] = {
            'searched_on': 'cleaned labels', 'cleanup': cleanup,
            'scored_on': ('labels as they are' if train_labels_c is not None
                          else 'the labels it trained on'),
            'condition_suffix': suffix_c,
            'n_training_rows_searched': int(len(tr_c)), 'models': recs}
        memo[suffix_c] = side
        return side

    def _screen_base(fold_i, X_base_tr, y0, g_tr, train_idx, val_idx,
                     base_sp, entry_cr, unit_key):
        """The neighbour settings screened in one cleaning representation.

        Returns {'nbrs', 'kept'} (kept best first, kept[0] the fold's top
        screened setting), or None with the reason in `errors`. The screen
        scores each (setting, k, scheme) on a held-back fifth of the training
        molecules (inner=, 2026-10-06), never on the training molecules
        themselves."""
        try:
            sp_tr, _sp_va, unav = fold_similarity_spaces(
                dataset_name, sweep, model_input_tr=X_base_tr,
                model_input_va=None, reps=reps, train_idx=train_idx,
                val_idx=val_idx, y_train=y0, groups_train=g_tr,
                base_reps=entry_cr.get('base') or [],
                recipes=entry_cr.get('recipes') or {},
                augmentations=data.get('augmentations'),
                model_name=cleaning_rank_model,
                model_factory=model_factories.get(cleaning_rank_model),
                metric=metric, task_type=task_type)
        except Exception as exc:
            errors[f'{unit_key}|{base_sp}|similarity_spaces'] = (
                f'{type(exc).__name__}: {exc}')
            return
        unav = dict(unav)
        for sp_, why in (entry_cr.get('unavailable') or {}).items():
            unav.setdefault(sp_, why)
        here = []
        for s_ in sweep:
            if s_.space in unav:
                sim_unavailable.setdefault(s_.name, unav[s_.space])
            else:
                here.append(s_)
        smiles_tr = sx.smiles_for(data, train_idx)
        # Axis 6: no exclude_groups, a molecule's scaffold-mates stay in its
        # neighbour list.
        nbrs = fold_neighbours(dataset_name, here, sp_tr, None, y0,
                               smiles_train=smiles_tr, is_clf=is_clf)
        for name, L in nbrs.items():
            if not L or not L.get('available'):
                sim_unavailable.setdefault(name, (L or {}).get('reason'))
        kept, table = screen_settings(
            here, nbrs, y0, metric=metric, is_clf=is_clf,
            inner={'dataset_name': dataset_name, 'spaces_train': sp_tr,
                   'y_train': y0, 'groups_train': g_tr,
                   'smiles_train': smiles_tr, 'is_clf': is_clf})
        screen_tables[f'{unit_key}|{base_sp}'] = table
        return {'nbrs': nbrs, 'kept': list(kept)}

    def _sweep_base(regime, level, draw, fold_i, X_base_tr, y0, suspicion,
                    targets, g_tr, base_sp, base_rn, ins_a, D_by, clean_by,
                    y_val, scr):
        """The fit-side arm under the screen's best 3 neighbour settings.

        kept[0] is the setting the default path already cleaned with, so its
        D-versus-D' rows were written there; only its controls are written
        here."""
        nbrs, kept = scr['nbrs'], scr['kept']
        for i_kept, (setting, k, sch) in enumerate(kept):
            nb = nbrs[setting.name]
            sim = f'{sx.sim_name(setting, k)}|scheme={sch}'
            for cleanup in run_cleanups:
                if (cleanup == IDENTITY_CONTROL
                        or cleanup in RANDOM_FLAG_CLEANUPS
                        or not sx.strategy_reads_setting(cleanup, setting)):
                    continue
                if i_kept == 0 and not sx.controls_enabled():
                    # Its rows came from the default path; no controls asked.
                    continue
                for rate in flag_rates:
                    def _clean(sc_, rate_, _c=cleanup, _k=k, _s=sch, _nb=nb):
                        return cleaned_training_set(
                            _c, X_base_tr, y0, sc_, rate_, is_clf=is_clf,
                            groups=g_tr, targets=targets,
                            similarity_neighbours=_nb, similarity_k=_k,
                            similarity_scheme=_s)
                    try:
                        cleaned = _clean(suspicion, rate)
                    except Exception as exc:
                        print(f"      {cleanup} under {sim} failed: "
                              f"{type(exc).__name__}: {exc}")
                        cleaned = None
                    for mname in models:
                        for sp, (X_tr_a, X_va_a) in (ins_a.get(mname) or {}).items():
                            ident = _ident(regime, level, draw, fold_i,
                                           cleanup, rate, base_sp, base_rn,
                                           ARM_TODAY, f'{sp}|asis|{mname}',
                                           sim)
                            if rows.is_done(ident, [mname]) and not \
                                    sx.controls_enabled():
                                continue
                            _D = D_by.get((mname, sp))
                            _C = (clean_by.get(mname) or {}).get(sp)
                            _kw = dict(
                                X_tr=X_tr_a, y_D=y0, X_va=X_va_a, y_va=y_val,
                                metric=metric,
                                factories={mname: model_factories[mname]},
                                score_D=({mname: _D} if _D is not None
                                         else None),
                                score_clean=({mname: _C} if _C is not None
                                             else None))
                            if i_kept > 0:
                                sx.dprime_rows(rows, dataset_name, ident,
                                               _on_hybrid(cleaned, X_tr_a, y0),
                                               **_kw)
                            sx.control_rows(
                                rows, dataset_name, ident,
                                _on_hybrid(cleaned, X_tr_a, y0),
                                seed=sx.seed_of(regime, level, draw, fold_i,
                                                base_sp, cleanup, rate, mname,
                                                sp, sim),
                                is_clf=is_clf,
                                rerun=(lambda sc_, r_, _X=X_tr_a, _cl=_clean:
                                       _on_hybrid(_cl(sc_, r_), _X, y0)),
                                **_kw)

    def _write_partial():
        # Dirty-run bug 12 and n20's "marked done with no file" (2026-10-05):
        # what is finished is written after every fold, to its own
        # directory, so a killed run leaves its finished folds readable and a
        # status script never mistakes it for the result file.
        if output_dir is None:
            return
        try:
            save_experiment_results(
                {'result_status': 'partial', 'has_computed_results': False,
                 'units_finished': list(ck.done_units),
                 'metric': metric_name, 'lower_is_better': lower_is_better,
                 'noise_condition': condition, 'n_folds': int(n_folds),
                 'per_regime': convert_numpy(acc),
                 'configurations': convert_numpy(configurations),
                 'configurations_cleaned': convert_numpy(configurations_cleaned),
                 '_errors': dict(errors)},
                output_dir, f'{EXP_FULL_NAME}/_partial', dataset_name,
                metric_name=metric_name, lower_is_better=lower_is_better,
                filename_suffix=condition if 'injected' in regimes else None)
        except Exception as exc:
            print(f"    partial result not written: {type(exc).__name__}: "
                  f"{exc}")

    t0 = time.time()

    for regime in regimes:
        if regime not in passes:
            raise ValueError(f"n20: unknown regime {regime!r}")
        # setdefault, not a fresh dict, since 2026-09-25: a resumed run
        # restores acc from the checkpoint and must not wipe it here.
        acc.setdefault(regime, {})
        for level, draw in passes[regime]:
            lvl_key = 'none' if level is None else f'{level:g}'
            tag = (f'{regime}' if level is None
                   else f'{regime}/{condition}/level {level:g}/draw {draw}')
            print(f"\n  --- {tag} ---")
            acc[regime].setdefault(lvl_key, {})

            for fold_i, train_idx, val_idx in folds:
                _unit = f'{regime}|{lvl_key}|d{draw}|f{fold_i}'
                if ck.is_done(_unit):
                    print(f"    fold {fold_i}: already in the checkpoint, "
                          f"skipped")
                    continue
                fold_rows = []
                fold_key = f'l{lvl_key}d{draw}f{fold_i}'
                unit_key = f'{regime}|{lvl_key}|{fold_key}'
                y_val = labels[val_idx]
                y_clean_tr = np.asarray(labels[train_idx])
                g_tr = None if groups_arr is None else groups_arr[train_idx]

                # The labels a practitioner holds for this fold's training
                # molecules. Damaged under the injected regime (the decided
                # kind and level; level 0 is the labels as delivered);
                # exactly what the loader produced under the other.
                cmask = None
                if regime == 'injected':
                    try:
                        y0, cmask, _prov = inject_condition(
                            data, dataset_name, condition, level, draw,
                            train_idx=train_idx, groups=noise_groups)
                        y0 = np.asarray(y0)
                        cmask = np.asarray(cmask, dtype=bool)
                    except Exception as exc:
                        errors[f'{_unit}|inject'] = (
                            f'{type(exc).__name__}: {exc}')
                        print(f"    fold {fold_i}: injection failed: "
                              f"{type(exc).__name__}: {exc}")
                        continue
                else:
                    y0 = y_clean_tr.copy()

                # ── THE SEARCH ON THE LABELS AS THEY ARE ──
                # One per fold, shared by every cleanup and every flag rate --
                # they all have to be read against the same selection or the
                # comparison is not paired. On the dataset's own labels it is
                # the dataset's standard per-fold search (no suffix).
                asis_suffix = (None if np.array_equal(y0, y_clean_tr)
                               else f'n20_asis_lb{_label_column_key(y0)}')
                # The dataset as loaded when its own labels are held, so the
                # stored standard per-fold search is the one read.
                data_asis = (data if asis_suffix is None else
                             _with_train_labels(data, train_idx, y0, is_clf))
                try:
                    if sweep:
                        entries_a = fold_configurations(
                            data_asis, dataset_name, fold_i=fold_i,
                            train_idx=train_idx, n_folds=n_folds,
                            condition_suffix=asis_suffix)
                    else:
                        entries_a = _selection_entries(
                            data_asis, dataset_name, fold_i=fold_i,
                            train_idx=train_idx, n_folds=n_folds,
                            condition_suffix=asis_suffix)
                except Exception as exc:
                    errors[f'{_unit}|search_asis'] = (
                        f'{type(exc).__name__}: {exc}')
                    print(f"    fold {fold_i}: search on the labels as they "
                          f"are failed: {type(exc).__name__}: {exc}")
                    continue
                if not any(entries_a.get(m) for m in models):
                    errors[f'{_unit}|search_asis'] = (
                        'the per-fold search on the labels as they are gave '
                        'no model a selection')
                    continue
                search_asis = _general_search(
                    data_asis, dataset_name, fold_i=fold_i,
                    train_idx=train_idx, n_folds=n_folds,
                    condition_suffix=asis_suffix)

                # The 8 configurations chosen from the labels as they are.
                # Built once per (fold, model) and shared by the two arms
                # whose search half is 'asis'.
                ins_a, recs_a = {}, {}
                for m in models:
                    entry = entries_a.get(m)
                    ins, unav, names = fold_model_inputs(
                        data_asis, dataset_name, m, entry,
                        train_idx=train_idx, val_idx=val_idx, y_train=y0,
                        factory=model_factories[m], metric=metric)
                    ins_a[m] = ins
                    recs_a[m] = _config_record(m, entry, names, ins, unav,
                                               search_asis)
                    print(f"    fold {fold_i} {m}: base {recs_a[m].get('base')}, "
                          f"hybrids {[h.get('label') for h in recs_a[m].get('hybrids') or []]}")
                configurations[unit_key] = {
                    'searched_on': 'labels as they are',
                    'condition_suffix': asis_suffix,
                    'n_training_rows': int(len(train_idx)),
                    'n_held_out_rows': int(len(val_idx)),
                    'models': recs_a}

                # Each configuration fitted on the CLEAN training labels in
                # the same features, the row's score_clean. Injected regime
                # only: under the other there is no clean column.
                clean_by = {}
                if regime == 'injected':
                    for m, by_sp in ins_a.items():
                        for sp, (X_tr_a, X_va_a) in by_sp.items():
                            clean_by.setdefault(m, {})[sp] = sx.fit_score(
                                dataset_name, m, model_factories[m], X_tr_a,
                                y_clean_tr, X_va_a, y_val, metric)

                # ── THE SINGLE REPRESENTATIONS THE CLEANING IS DECIDED IN ──
                # The cleaning-rank model's 2 best base representations in the
                # search just run on this fold's labels as they are. The
                # detector and every cleanup run once in each, shared by every
                # model and configuration. No fallback: a fold with none runs
                # no cleanup, and the reason is written.
                entry_cr = entries_a.get(cleaning_rank_model) or {}
                fold_base = {f'base_{_k}': _rn for _k, _rn in enumerate(
                    list(entry_cr.get('base') or [])[:NOISE_BASE_REP_COUNT],
                    start=1)}
                base_reps_by_fold[unit_key] = (
                    dict(fold_base) if fold_base else
                    {'none': (f'{cleaning_rank_model} has no ranked single '
                              f'representation in this fold\'s search')})
                if not fold_base:
                    errors[f'{_unit}|base_reps'] = (
                        f'no single representation for {cleaning_rank_model}. '
                        f'No detector and no cleanup ran on this fold.')
                    print(f"    fold {fold_i}: no single representation for "
                          f"{cleaning_rank_model}; fold skipped")
                    continue

                cleaned_memo = {}
                for base_sp, base_rn in fold_base.items():
                    cell = acc[regime][lvl_key].setdefault(base_sp, {})
                    # This single representation: where the detector runs and
                    # where the cleaning is decided, once per (fold, single
                    # representation) for all four arms. 0/1 features left
                    # unscaled, float32 kept (2026-10-05).
                    X_base_tr = FeatureScaler().fit_transform(
                        as_rep_dtype(np.asarray(reps[base_rn])[train_idx]))
                    # ⛔ THE FOLD'S GROUPS GO TO THE DETECTOR. Without them
                    # CrossValResidualEstimator falls back to a plain KFold, so a
                    # molecule's own scaffold or homology cluster trains the
                    # out-of-fold forest that predicts it and its residual is small
                    # for a reason that has nothing to do with its label. The
                    # estimator's own docstring says so
                    # (kirby/utils/noise_estimation.py:558-561) and n18 passes them
                    # at n18:400; this file did not until 2026-09-20.
                    #
                    # MEASURED on flab_thermo_tresanco, 672 antibodies, in
                    # esm2_cdr, at an injected level of 0.5 of the clean label
                    # spread under outlier_p10, against the training antibodies the
                    # corruption really moved:
                    #   held-out fold 0, 29 corrupted of 336 training antibodies:
                    #     0.2845 detection average precision withheld, 0.3322 given
                    #   held-out fold 1, 31 corrupted of 336 training antibodies:
                    #     0.2056 withheld, 0.2387 given
                    # Giving the detector the groups won both paired passes. Two
                    # passes, one corruption draw, one dataset.
                    suspicion = _compute_noise_scores(X_base_tr, y0, is_clf=is_clf,
                                                      detector=DETECTOR,
                                                      groups=g_tr)
                    if suspicion is None:
                        errors[f'{_unit}|{base_sp}|detector'] = (
                            'the detector would not fit on this fold')
                        print(f"    fold {fold_i} {base_sp} ({base_rn}): "
                              f"detector failed, dropped")
                        continue

                    # ⛔ THE MODEL-BASED CLEANUPS' TARGETS, ONCE PER FOLD. They
                    # depend on the features, the labels and the groups and on
                    # nothing else, so one fit of each serves every flag rate
                    # and every cleanup that wants it. Same hoist n18 does.
                    #
                    # ⛔ AN EMPTY WANT-LIST IS NOT "NOTHING". correction_targets
                    # reads `want = set(want or ('oof', 'gp', 'quantile'))`,
                    # so a run whose cleanups need no target must not call it.
                    _want = targets_needed(
                        [RANDOM_FLAG_CLEANUPS.get(c, c) for c in run_cleanups])
                    _targets = (correction_targets(X_base_tr, y0, groups=g_tr,
                                                   want=_want)
                                if _want else {})

                    # Did the detector find the molecules that really moved, before
                    # any cleanup ran? Only defined where a corrupted subset exists.
                    if (cmask is not None and cmask.any()
                            and detection_is_defined(condition)):
                        det = detection_scores(cmask, suspicion, condition)
                        if det is not None:
                            (detection_acc.setdefault(base_sp, {})
                             .setdefault(lvl_key, []).append(det))

                    # ── THE CLEANING'S OWN NEIGHBOURS ──
                    # DECIDED 2026-10-06 (Adelaide delegated): the fold's top
                    # screened setting (kept[0]), screened in this cleaning
                    # representation on a held-back fifth of the training
                    # molecules. label_smooth and knn_smooth (and their
                    # random-flag twins) read it on every arm and in every
                    # cleaned search; prox_smooth only when it is
                    # forest_proximity. One setting, so one cleaned search per
                    # cleaned column, as before.
                    scr = (_screen_base(fold_i, X_base_tr, y0, g_tr, train_idx,
                                        val_idx, base_sp, entry_cr, unit_key)
                           if sweep else None)
                    top = None
                    _nkey = f'{unit_key}|{base_sp}'
                    if scr and scr['kept']:
                        _s0, _k0, _sch0 = scr['kept'][0]
                        top = {'setting': _s0, 'k': int(_k0), 'scheme': _sch0,
                               'lists': dict(scr['nbrs'][_s0.name],
                                             scheme=_sch0),
                               'sim': f'{sx.sim_name(_s0, _k0)}|scheme={_sch0}'}
                        cleaning_neighbours_by_fold[_nkey] = {
                            'setting': _s0.name, 'space': _s0.space,
                            'k': int(_k0), 'scheme': _sch0,
                            'similarity_setting': top['sim']}
                    else:
                        if not sweep:
                            _why = 'no neighbour setting is swept on this dataset'
                        elif scr is None:
                            _why = ('the neighbour spaces could not be built '
                                    '(see _errors)')
                        else:
                            _why = 'the screen kept no (setting, k, scheme)'
                        cleaning_neighbours_by_fold[_nkey] = {'none': (
                            _why + "; label_smooth / knn_smooth used n07's "
                                   "default neighbours (_DEFAULT_SIM)")}

                    def _uses_top(strategy, _top=top):
                        return (_top is not None and
                                sx.strategy_reads_setting(strategy,
                                                          _top['setting']))

                    for cleanup in run_cleanups:
                        for rate in flag_rates:
                            if cleanup == IDENTITY_CONTROL:
                                cleaned = {'X': X_base_tr, 'y': y0,
                                           'sample_weight': None, 'keep': None,
                                           'label_column': y0}
                            else:
                                # A random-flag name runs its own cleanup with the
                                # suspicion score shuffled over the training rows.
                                # Same cleanup, same number flagged, different
                                # molecules flagged.
                                strategy = RANDOM_FLAG_CLEANUPS.get(cleanup,
                                                                    cleanup)
                                if strategy != cleanup:
                                    score_used, _order = _permuted_suspicion(
                                        suspicion, _unit)
                                else:
                                    score_used = suspicion
                                _nb_kw = ({} if not _uses_top(strategy) else
                                          {'similarity_neighbours': top['lists'],
                                           'similarity_k': top['k'],
                                           'similarity_scheme': top['scheme']})
                                try:
                                    cleaned = cleaned_training_set(
                                        strategy, X_base_tr, y0, score_used, rate,
                                        is_clf=is_clf, groups=g_tr,
                                        targets=_targets, **_nb_kw)
                                except Exception as exc:
                                    errors[f'{_unit}|{base_sp}|{cleanup}|{rate}|'
                                           f'clean'] = (
                                        f'{type(exc).__name__}: {exc}')
                                    continue
                                if cleaned is None:
                                    # The cleanup declined this fold. Several
                                    # things in cleaned_training_set return None
                                    # and the message cannot tell them apart from
                                    # here; each one prints its own reason.
                                    errors[f'{_unit}|{base_sp}|{cleanup}|{rate}|'
                                           f'clean'] = (
                                        f'{strategy} declined this fold: remove '
                                        f'leaves under 10 training molecules, '
                                        f'gp_smooth and prox_smooth refuse above '
                                        f'their row caps, js_shrink refuses '
                                        f'without an out-of-fold prediction or '
                                        f'without spread to shrink, '
                                        f'constant_lift where the labels show no '
                                        f'ceiling. The reason was printed by the '
                                        f'cleanup itself.')
                                    continue

                            keep = cleaned['keep']
                            col = cleaned['label_column']
                            w = cleaned['sample_weight']
                            # What the fit is handed on the cleaned side, in
                            # whatever feature space the arm produced. The row
                            # selection, the label column and the weight all come
                            # from the one cleaning decision above.
                            y_fit_cleaned = (y0 if col is None else col)

                            # The search on the cleaned labels. Undefined when the
                            # cleaning is a weight: nothing in the feature
                            # selection reads one. `remove` searches on the kept
                            # rows and their labels; a smoother on every row and
                            # the cleaned column. Each cleaned column gets its own
                            # condition suffix (cleanup, rows, label hash), so two
                            # flag rates that hand back the same column share one
                            # search and two different columns never do.
                            side_c = None
                            if col is not None or keep is not None:
                                side_c = _cleaned_side(
                                    cleanup, keep, col, y0=y0,
                                    train_idx=train_idx, val_idx=val_idx,
                                    fold_i=fold_i, memo=cleaned_memo,
                                    unit_key=f'{unit_key}|{base_sp}')

                            if side_c is not None:
                                s_c = side_c['search']
                                probe_row = {
                                    'label_columns_identical': bool(
                                        col is not None and keep is None
                                        and np.array_equal(y0, col)),
                                    'n_training_rows_the_search_saw_asis':
                                        int(len(y0)),
                                    'n_training_rows_the_search_saw_cleaned':
                                        int(len(side_c['y'])),
                                    'n_candidates_searched_asis':
                                        (len(search_asis.all_results or [])
                                         if search_asis is not None else None),
                                    'n_candidates_searched_cleaned':
                                        (len(s_c.all_results or [])
                                         if s_c is not None else None),
                                    'recipe_by_model': {
                                        m: {'asis': _selection_fingerprint(
                                                entries_a.get(m)),
                                            'cleaned': _selection_fingerprint(
                                                side_c['entries'].get(m))}
                                        for m in models},
                                }
                                recipe_probe[f'{unit_key}/{base_sp}|'
                                             f'{cleanup}|{rate:g}'] = probe_row
                                if cleanup == IDENTITY_CONTROL:
                                    identity_probe[
                                        f'{unit_key}/{base_sp}'] = probe_row

                            # ── WHICH FEATURES, not which recipe ──
                            # Both searches' configurations are ranked in the
                            # dataset's own metric and own direction, and the
                            # best `per_feature_top_n` of each are BUILT so the
                            # features they selected can be counted. Under the
                            # identity control the two tables must come out
                            # equal; that is the check that the counting itself
                            # is not what moves.
                            if (per_feature_top_n and side_c is not None
                                    and side_c['search'] is not None):
                                tr_c = side_c['train_idx']
                                train_reps = {k: as_rep_dtype(np.asarray(v)[train_idx])
                                              for k, v in reps.items()}
                                train_reps_c = {k: as_rep_dtype(np.asarray(v)[tr_c])
                                                for k, v in reps.items()}
                                val_reps = {k: as_rep_dtype(np.asarray(v)[val_idx])
                                            for k, v in reps.items()}
                                g_c = (None if groups_arr is None
                                       else groups_arr[tr_c])
                                for mname in models:
                                    rows_a, note_a = _search_configurations(
                                        search_asis, mname)
                                    rows_c, note_c = _search_configurations(
                                        side_c['search'], mname)
                                    top_a = _rank_stored(rows_a, lower_is_better,
                                                         per_feature_top_n)
                                    top_c = _rank_stored(rows_c, lower_is_better,
                                                         per_feature_top_n)
                                    rs_a = _rep_scores_from_search(search_asis)
                                    rs_c = _rep_scores_from_search(side_c['search'])
                                    cnt_a, built_a, widths, ref_a = \
                                        _feature_frequency(
                                            top_a, train_reps, val_reps, y0, g_tr,
                                            model_factories[mname], metric_fn,
                                            lower_is_better, rep_scores=rs_a)
                                    cnt_c, built_c, _w2, ref_c = \
                                        _feature_frequency(
                                            top_c, train_reps_c, val_reps,
                                            side_c['y'], g_c,
                                            model_factories[mname], metric_fn,
                                            lower_is_better, rep_scores=rs_c)
                                    fcell = _compare_feature_tables(
                                        cnt_a, built_a, cnt_c, built_c, widths)
                                    fcell['n_configurations_the_search_held_asis'] = \
                                        len(rows_a)
                                    fcell['n_configurations_the_search_held_cleaned'] = \
                                        len(rows_c)
                                    fcell['search_note_asis'] = note_a
                                    fcell['search_note_cleaned'] = note_c
                                    fcell['budgets_among_the_top_asis'] = sorted(
                                        {str(r.get('budget')) for r in top_a})
                                    fcell['budgets_among_the_top_cleaned'] = sorted(
                                        {str(r.get('budget')) for r in top_c})
                                    fcell['is_the_identity_control'] = bool(
                                        cleanup == IDENTITY_CONTROL)
                                    fcell['n_not_built_because_the_stored_recipe_'
                                         'lost_its_rep_scores_asis'] = int(ref_a)
                                    fcell['n_not_built_because_the_stored_recipe_'
                                         'lost_its_rep_scores_cleaned'] = int(ref_c)
                                    per_feature[
                                        f'{unit_key}|{base_sp}|{cleanup}|'
                                        f'{rate:g}/{mname}'] = fcell

                            rate_cell = (cell.setdefault(cleanup, {})
                                             .setdefault(f'{rate:g}', {}))

                            for mname in models:
                              for sp in CONFIGURATION_SPACES:
                                ckey = f'{mname}__{sp}'
                                arm_cell = rate_cell.setdefault(ckey, {})

                                # ── the two arms whose features came from the
                                #    labels as they are ──
                                pair_a = (ins_a.get(mname) or {}).get(sp)
                                if pair_a is not None:
                                    X_tr_a, X_va_a = pair_a
                                    for arm, yv, wv in (
                                            ('search_asis__fit_asis', y0, None),
                                            ('search_asis__fit_cleaned',
                                             y_fit_cleaned, w)):
                                        Xt, yt = X_tr_a, yv
                                        if arm.endswith('fit_cleaned') and keep is not None:
                                            Xt, yt = X_tr_a[keep], y0[keep]
                                        if wv is not None and mname in refuses_weight:
                                            continue
                                        s, err = _fit_score(
                                            model_factories[mname], Xt, yt, wv,
                                            X_va_a, y_val, metric_fn, metric_name)
                                        if err == 'refuses_sample_weight':
                                            refuses_weight.add(mname)
                                            continue
                                        if s is None:
                                            errors[f'{_unit}|{base_sp}|{cleanup}|'
                                                   f'{rate}|{ckey}|{arm}'] = err
                                            continue
                                        arm_cell.setdefault(arm, {})[fold_key] = s

                                # ── the two arms whose features came from the
                                #    cleaned labels ──
                                if side_c is None:
                                    continue
                                pair_c = (side_c['inputs'].get(mname) or {}).get(sp)
                                if pair_c is None:
                                    continue
                                if keep is not None:
                                    # Built on the kept rows; the dropped
                                    # training rows and the held-out rows were
                                    # transformed through it (_cleaned_side).
                                    n_tr = len(train_idx)
                                    X_keep, X_rest = pair_c
                                    X_all = X_rest[:n_tr]
                                    X_va_c = X_rest[n_tr:]
                                    pairs = (
                                        ('search_cleaned__fit_cleaned',
                                         X_keep, y0[keep], None),
                                        ('search_cleaned__fit_asis',
                                         X_all, y0, None),
                                    )
                                else:
                                    X_c, X_va_c = pair_c
                                    pairs = (
                                        ('search_cleaned__fit_cleaned', X_c,
                                         y_fit_cleaned, w),
                                        ('search_cleaned__fit_asis', X_c, y0, None),
                                    )

                                for arm, Xt, yt, wv in pairs:
                                    if wv is not None and mname in refuses_weight:
                                        continue
                                    s, err = _fit_score(
                                        model_factories[mname], Xt, yt, wv,
                                        X_va_c, y_val, metric_fn, metric_name)
                                    if err == 'refuses_sample_weight':
                                        refuses_weight.add(mname)
                                        continue
                                    if s is None:
                                        errors[f'{_unit}|{base_sp}|{cleanup}|'
                                               f'{rate}|{ckey}|{arm}'] = err
                                        continue
                                    arm_cell.setdefault(arm, {})[fold_key] = s

                            # ── This (cleanup, rate)'s rows ──
                            # Against search_asis__fit_asis on the same fold.
                            # An arm that is undefined for this cleanup (the
                            # searched-on-cleaned pair of downweight) writes
                            # no row; one that declined writes a nan.
                            if cleanup == IDENTITY_CONTROL:
                                continue
                            _strat = RANDOM_FLAG_CLEANUPS.get(cleanup, cleanup)
                            _sim = (top['sim'] if _uses_top(_strat) else
                                    _DEFAULT_SIM.get(_strat, sx.NO_SIMILARITY))
                            for mname in models:
                              for sp in CONFIGURATION_SPACES:
                                _arms = rate_cell.get(f'{mname}__{sp}', {})
                                _D = _arms.get('search_asis__fit_asis',
                                               {}).get(fold_key)
                                if _D is None:
                                    continue
                                for arm in ARMS[1:]:
                                    _P = _arms.get(arm, {}).get(fold_key)
                                    if _P is None and arm != ARM_TODAY:
                                        continue
                                    _mi = (f'{sp}|asis|{mname}'
                                           if arm.startswith('search_asis')
                                           else f'{sp}|searched_on_cleaned|{mname}')
                                    fold_rows.append(sx.row(
                                        _ident(regime, level, draw, fold_i,
                                               cleanup, rate, base_sp, base_rn,
                                               arm, _mi, _sim),
                                        model=mname, metric=metric,
                                        score_D=_D, score_Dprime=_P,
                                        score_clean=(clean_by.get(mname)
                                                     or {}).get(sp)))
                                # The shared twins of the fit-side arm, under
                                # KIRBY_NOISE_CONTROLS=1, in the first hybrid
                                # only, as before 2026-10-05. n20's own
                                # random-flag control (a permuted suspicion
                                # score fed to BOTH searches) is a different
                                # control and stays. A cleanup that read the
                                # top screened setting gets its twins from the
                                # similarity sweep, in all 8 configurations.
                                if (sp == 'hybrid' and sx.controls_enabled()
                                        and not _uses_top(_strat)
                                        and cleanup not in RANDOM_FLAG_CLEANUPS
                                        and (ins_a.get(mname) or {}).get(sp)
                                        is not None):
                                    _Xa, _Xva = ins_a[mname][sp]
                                    sx.control_rows(
                                        rows, dataset_name,
                                        _ident(regime, level, draw, fold_i,
                                               cleanup, rate, base_sp,
                                               base_rn, ARM_TODAY,
                                               f'{sp}|asis|{mname}', _sim),
                                        _on_hybrid(cleaned, _Xa, y0),
                                        X_tr=_Xa, y_D=y0, X_va=_Xva,
                                        y_va=y_val, metric=metric,
                                        factories={mname:
                                                   model_factories[mname]},
                                        seed=sx.seed_of(regime, level, draw,
                                                        fold_i, base_sp,
                                                        cleanup, rate, mname),
                                        is_clf=is_clf,
                                        score_D={mname: _D},
                                        score_clean=(
                                            {mname: clean_by[mname][sp]}
                                            if sp in (clean_by.get(mname) or {})
                                            else None))

                    # ── The similarity sweep, fit side, in this single
                    #    representation ──
                    if scr and scr['kept']:
                        _D_by = {}
                        for _c in cell.values():
                            for _r in _c.values():
                                for mname in models:
                                    for sp in CONFIGURATION_SPACES:
                                        _v = (_r.get(f'{mname}__{sp}', {})
                                              .get('search_asis__fit_asis', {})
                                              .get(fold_key))
                                        if _v is not None:
                                            _D_by.setdefault((mname, sp), _v)
                        _sweep_base(regime, level, draw, fold_i, X_base_tr, y0,
                                    suspicion, _targets, g_tr, base_sp,
                                    base_rn, ins_a, _D_by, clean_by, y_val,
                                    scr)

                rows.write(fold_rows)
                ck.save(_unit, (acc, errors, detection_acc, identity_probe,
                                recipe_probe, per_feature,
                                sorted(refuses_weight), base_reps_by_fold,
                                sim_unavailable, configurations,
                                configurations_cleaned, screen_tables,
                                cleaning_neighbours_by_fold))
                _write_partial()
                print(f"    fold {fold_i} done [{time.time() - t0:.0f}s]")

    results['per_regime'] = convert_numpy(acc)
    results['base_reps_by_fold'] = base_reps_by_fold
    results['configurations'] = convert_numpy(configurations)
    results['configurations_cleaned'] = convert_numpy(configurations_cleaned)
    results['cleaning_neighbours_by_fold'] = convert_numpy(
        cleaning_neighbours_by_fold)
    if screen_tables:
        results['screen'] = convert_numpy(screen_tables)
    # {base_1 / base_2: {level: reading}}
    results['detection'] = convert_numpy(
        {b: {k: _mean_detection(v) for k, v in by_lvl.items()}
         for b, by_lvl in detection_acc.items()})
    results['models_refusing_sample_weight'] = sorted(refuses_weight)
    results['per_feature_top_n'] = per_feature_top_n
    if per_feature:
        results['per_feature'] = convert_numpy({
            'what_one_row_is': (
                'One FEATURE, named as the representation it lives in and its '
                'column index INSIDE that representation. The indices '
                '`create_hybrid` returns are local to each representation '
                '(hybrid.py:2236), so saprot column 7 and onehot column 7 are '
                'two features, not one.'),
            'what_the_frequency_counts': (
                'How many of the top configurations of the search, once built, '
                'selected that feature. The denominator is '
                'n_top_hybrids_built, which is how many of them produced a '
                'hybrid at all, not how many were asked for.'),
            'how_the_top_was_chosen': (
                'Every configuration the per-fold search scored for this '
                'downstream model, across every budget it scored, ranked by '
                'score_mean in the dataset metric and its own direction '
                f'(lower_is_better={lower_is_better}), best '
                f'{per_feature_top_n} kept.'),
            'per_cell': per_feature,
        })
    if errors:
        results['_errors'] = errors
    results['summary'] = convert_numpy(
        _summarise(acc, [f'{m}__{sp}' for m in models
                         for sp in CONFIGURATION_SPACES],
                   run_cleanups, flag_rates,
                   metric_name, lower_is_better, identity_probe,
                   recipe_probe))
    results['rows'] = rows.summary()
    if sim_unavailable:
        results['similarity_settings_unavailable'] = sim_unavailable
    _print_summary(results, metric_name)
    # n20's result file is written by its caller (run_all or main), not here.
    # The snapshot is cleared on return: a kill between this return and that
    # write costs a re-run of the folds, never a stale resume.
    ck.clear()
    return results


def _mean_detection(rows):
    """Average the detection readings over the (fold, draw) passes."""
    if not rows:
        return None
    keys = ('detection_average_precision', 'detection_precision_at_k',
            'detection_recall_at_k', 'detection_n_corrupted',
            'detection_n_training')
    out = {}
    for k in keys:
        vals = [float(r[k]) for r in rows
                if r.get(k) is not None and np.isfinite(float(r[k]))]
        out[k] = float(np.mean(vals)) if vals else float('nan')
    out['detection_condition'] = rows[0].get('detection_condition')
    out['detection_n_passes'] = len(rows)
    return out


def _paired(a, b, lower_is_better):
    """b against a, on the folds both scored. Positive gain always means better.

    The mean and spread of the per-fold gains come from
    noise_helpers.fold_spread since 2026-10-05: one fold reads 'one fold, no
    spread', never a division by zero."""
    shared = sorted(set(a) & set(b))
    gains = np.array([(a[k] - b[k]) if lower_is_better else (b[k] - a[k])
                      for k in shared], dtype=float)
    gains = gains[np.isfinite(gains)]
    scale = max(abs(float(np.mean(list(a.values())))) if a else 1.0, 1.0)
    tol = 1e-9 * scale
    fs = fold_spread(list(gains))
    return {
        'n_folds_paired': int(gains.size),
        'folds_won': int((gains > tol).sum()),
        'folds_lost': int((gains < -tol).sum()),
        # A large tied count means the two arms are one fit.
        'folds_tied': int((np.abs(gains) <= tol).sum()),
        'tie_tolerance': float(tol),
        'mean_gain': fs['mean'],
        'spread_over_fold_gains': fs['sd'],
        'fold_spread': fs,
        'fold_gains': [float(x) for x in gains],
        # A gain no larger than the spread of the per-fold gains behind it is
        # not a gain. Stated as a comparison of two measured numbers, not a
        # test: the suite's standing rule at this replicate count is raw spread
        # only. None on one fold, where there is no spread.
        'gain_exceeds_its_own_spread': (
            None if fs['sd'] is None else bool(fs['mean'] > fs['sd'])),
    }


def _summarise(acc, models, cleanups, flag_rates, metric_name, lower_is_better,
               identity_probe=None, recipe_probe=None):
    """What the ordering cost, paired per fold, plus the identity control.

    Two comparisons answer the question and one checks the machinery.

      ordering_cost        search_cleaned__fit_cleaned against
                           search_asis__fit_cleaned. The second is what the
                           suite runs today; the difference is what applying
                           the cleanup after the recipe was fixed cost.
      feature_choice_only  search_cleaned__fit_asis against
                           search_asis__fit_asis. A gain here is the cleaning
                           helping the feature choice with no cleaned label
                           ever reaching a model.
      search_path_identity under `no_cleaning` the two searches are handed
                           byte-identical label columns. Three things are read
                           off that: how many candidates each search scored,
                           what recipe each one picked, and how far the four
                           arm scores sit apart on one fold. The first two are
                           the defect stated directly -- n15's third, at
                           NOISE_AUDIT_2026_09_10.md section 4, is two searches
                           choosing from lists of DIFFERENT LENGTHS. The third
                           is the consequence, and near the last bit of a
                           double it is the same fit computed twice rather than
                           a disagreement.
    """
    out = {'metric': metric_name, 'lower_is_better': bool(lower_is_better),
           'headline': 'folds won, paired per fold, not mean gain',
           'per_regime': {}}

    for regime, by_level in acc.items():
        reg_row = {'per_level': {}}
        for lvl, by_base in by_level.items():
          lvl_row = {'per_base': {}}
          for base_sp, by_cleanup in by_base.items():
            base_row = {'per_cleanup': {}}
            lvl_row['per_base'][base_sp] = base_row
            for cleanup in cleanups:
                by_rate = by_cleanup.get(cleanup)
                if not by_rate:
                    continue
                rate_rows = {}
                for rate in flag_rates:
                    rk = f'{rate:g}'
                    by_model = by_rate.get(rk)
                    if not by_model:
                        continue
                    model_rows = {}
                    for m in models:
                        arms = by_model.get(m) or {}
                        row = {
                            'arm_means': {
                                a: fold_spread(list((arms.get(a) or {}).values()))['mean']
                                for a in ARMS},
                            'arm_fold_spread': {
                                a: fold_spread(list((arms.get(a) or {}).values()))
                                for a in ARMS},
                            'arm_n_folds': {a: len(arms.get(a) or {})
                                            for a in ARMS},
                        }
                        if arms.get(ARM_TODAY) and arms.get(ARM_ORDERED):
                            row['ordering_cost'] = _paired(
                                arms[ARM_TODAY], arms[ARM_ORDERED],
                                lower_is_better)
                        if (arms.get('search_asis__fit_asis')
                                and arms.get('search_cleaned__fit_asis')):
                            row['feature_choice_only'] = _paired(
                                arms['search_asis__fit_asis'],
                                arms['search_cleaned__fit_asis'],
                                lower_is_better)
                        if cleanup == IDENTITY_CONTROL:
                            row['search_path_identity'] = _identity(arms)
                        model_rows[m] = row
                    rate_rows[rk] = {'per_model': model_rows}
                base_row['per_cleanup'][cleanup] = {'per_flag_rate': rate_rows}
          reg_row['per_level'][lvl] = lvl_row
        out['per_regime'][regime] = reg_row

    # The control, rolled up: the worst four-way disagreement anywhere.
    worst, worst_at, n_read = 0.0, None, 0
    for regime, reg in out['per_regime'].items():
        for lvl, lrow in reg['per_level'].items():
          for base_sp, brow in lrow['per_base'].items():
            blk = brow['per_cleanup'].get(IDENTITY_CONTROL)
            if not blk:
                continue
            for rk, rrow in blk['per_flag_rate'].items():
                for m, mrow in rrow['per_model'].items():
                    idt = mrow.get('search_path_identity')
                    if not idt or not np.isfinite(idt['max_spread_over_arms']):
                        continue
                    n_read += 1
                    if idt['max_spread_over_arms'] > worst:
                        worst = idt['max_spread_over_arms']
                        worst_at = (f'{regime}/level {lvl}/{base_sp}/{m}/'
                                    f'flag {rk}')
    probe = identity_probe or {}
    same_len = [p for p in probe.values()
                if p['n_candidates_searched_asis']
                == p['n_candidates_searched_cleaned']]
    same_recipe, diff_recipe = 0, []
    for key, p in probe.items():
        for m, pair in p['recipe_by_model'].items():
            if pair['asis'] == pair['cleaned']:
                same_recipe += 1
            else:
                diff_recipe.append(f'{key}/{m}')
    out['search_path_identity'] = {
        'what_it_checks': (
            'Under no_cleaning the two searches are handed byte-identical '
            'label columns, so they must score the same number of candidates, '
            'pick the same recipe, and leave the four arms as four identical '
            'fits. Two searches choosing from lists of DIFFERENT LENGTHS is '
            "n15's third defect, NOISE_AUDIT_2026_09_10.md section 4."),
        # The defect stated directly: how long each candidate list was.
        'n_folds_probed': len(probe),
        'n_folds_where_both_searched_the_same_number_of_candidates':
            len(same_len),
        'n_recipes_that_matched': same_recipe,
        'recipes_that_did_not_match': diff_recipe,
        'per_fold': probe,
        # And the consequence, on the scores.
        'max_spread_over_arms_anywhere': float(worst),
        'worst_cell': worst_at,
        'n_cells_read': n_read,
        'spread_note': (
            'A spread near 1e-16 is NOT a disagreement between the two '
            'searches. The roster model does not predict the same number '
            'twice: measured 2026-09-17, one RandomForestRegressor(n_jobs=-1) '
            'fitted once and asked to predict the same rows forty times gave 0 '
            'of 39 repeats bit-identical to the first, largest difference '
            '8.88e-16; at n_jobs=1 it gave 39 of 39 identical. sklearn sums '
            'each tree into a shared array in thread completion order. So this '
            'number has a floor of about 1e-15 and cannot be read below it. '
            'What IS exact is the candidate count and the chosen recipe, both '
            'above. A real disagreement moves the score in the third or fourth '
            'decimal, which is where the real cleanups sit.'),
    }

    # ── DID THE SEARCH PICK A DIFFERENT RECIPE, per cleanup ──
    # The other half of the question. The scores above say what the ordering
    # cost; this says whether the recipe the search SELECTED moved at all, and
    # a run where it did not move is a run whose gain came entirely from the
    # per-fold feature ranking and budget split.
    rp = recipe_probe or {}
    per_cleanup = {}
    for key, p in rp.items():
        cleanup = key.split('|')[3]
        c = per_cleanup.setdefault(cleanup, {
            'n_cells': 0, 'n_cells_where_the_recipe_moved': 0,
            'cells_where_it_moved': [],
            'candidate_counts_asis': [], 'candidate_counts_cleaned': [],
            'rep_count_asis': [], 'rep_count_cleaned': []})
        for m, pair in p['recipe_by_model'].items():
            c['n_cells'] += 1
            if pair['asis'] != pair['cleaned']:
                c['n_cells_where_the_recipe_moved'] += 1
                c['cells_where_it_moved'].append(f'{key}/{m}')
            c['rep_count_asis'].append(_first_hybrid_rep_count(pair['asis']))
            c['rep_count_cleaned'].append(
                _first_hybrid_rep_count(pair['cleaned']))
        c['candidate_counts_asis'].append(p['n_candidates_searched_asis'])
        c['candidate_counts_cleaned'].append(p['n_candidates_searched_cleaned'])
    out['recipe_the_search_selected'] = {
        'what_it_reports': (
            'One cell is one (regime, level, held-out fold and single '
            'representation, cleanup, flag rate, downstream model). For each, '
            'what the search on the labels as they are selected and what the '
            'search on the cleaned labels selected: the 2 base '
            'representations, and for each of the 2 hybrids its '
            'representations, allocation method, importance method, budget, '
            'internal evaluator and representation count. '
            'n_cells_where_the_recipe_moved is the count this experiment '
            'exists to produce: cleaning first changed what KIRBy selected in '
            'that many cells of that many. rep_count lists are the first '
            "hybrid's."),
        'candidate_count_note': (
            'n_candidates_searched is how many configurations the per-fold '
            'search holds in all_results, stored rows included '
            '(find_best_hybrid.py:7551). Pruning is off in every per-fold '
            'search (2026-10-05).'),
        'per_cleanup': per_cleanup,
    }
    return out


def _first_hybrid_rep_count(fp):
    """The representation count of a selection's first hybrid, or None."""
    hyb = (fp or {}).get('hybrids') or []
    return (hyb[0] or {}).get('rep_count') if hyb else None


def _identity(arms):
    """The four-way spread per fold under the identity control."""
    folds = set()
    for a in ARMS:
        folds |= set(arms.get(a) or {})
    per_fold = {}
    for f in sorted(folds):
        vals = [arms[a][f] for a in ARMS
                if arms.get(a) and f in arms[a]]
        if len(vals) < 2:
            continue
        per_fold[f] = float(max(vals) - min(vals))
    spreads = list(per_fold.values())
    seen = [v for a in ARMS for v in (arms.get(a) or {}).values()]
    scale = max(abs(float(np.mean(seen))) if seen else 1.0, 1.0)
    return {
        'n_folds_with_all_four': int(sum(
            1 for f in folds
            if all(arms.get(a) and f in arms[a] for a in ARMS))),
        'spread_over_arms_by_fold': per_fold,
        'max_spread_over_arms': float(max(spreads)) if spreads else float('nan'),
        'tie_tolerance': float(1e-9 * scale),
    }


def _print_summary(results, metric_name):
    s = results['summary']
    print("\n" + "=" * 70)
    print(f"SUMMARY ({metric_name}; paired per fold)")
    if results.get('models_refusing_sample_weight'):
        print(f"  refused a per-molecule weight, so absent from the "
              f"downweight arms: "
              f"{', '.join(results['models_refusing_sample_weight'])}")
    rss = s.get('recipe_the_search_selected') or {}
    if rss.get('per_cleanup'):
        print("\n  THE RECIPE THE SEARCH SELECTED. One cell is one held-out "
              "fold and one\n  downstream model. A cell counts as moved when "
              "the search on the labels as\n  they are and the search on the "
              "cleaned labels picked different recipes:")
        for cleanup, row in rss['per_cleanup'].items():
            tail = '' if cleanup != IDENTITY_CONTROL else '   (identity control: must be 0)'
            print(f"    {cleanup:13s} the recipe moved in "
                  f"{row['n_cells_where_the_recipe_moved']} of "
                  f"{row['n_cells']} cells{tail}")
            def _rc(v):
                return ', '.join('all ten' if x is None else str(x) for x in v)
            print(f"      representations the recipe used, one entry per cell: "
                  f"searching on the labels as they are {_rc(row['rep_count_asis'])}; "
                  f"searching on the cleaned labels {_rc(row['rep_count_cleaned'])}")

    pf = results.get('per_feature') or {}
    for key, c in (pf.get('per_cell') or {}).items():
        print(f"\n  WHICH FEATURES, cell {key}")
        print(f"    the representations hold "
              f"{c['n_features_the_representations_hold_between_them']} "
              f"features between them")
        print(f"    configurations the per-fold search held, for this downstream model: "
              f"{c['n_configurations_the_search_held_asis']} searching on the "
              f"labels as they are, "
              f"{c['n_configurations_the_search_held_cleaned']} searching on "
              f"the cleaned labels")
        print(f"    hybrids built out of those, one per configuration: "
              f"{c['n_top_hybrids_built_searching_on_the_labels_as_they_are']} "
              f"and "
              f"{c['n_top_hybrids_built_searching_on_the_cleaned_labels']}; "
              f"budgets among them "
              f"{c['budgets_among_the_top_asis']} and "
              f"{c['budgets_among_the_top_cleaned']}")
        print(f"    features at least one of those hybrids selected: "
              f"{c['n_features_selected_by_at_least_one_top_hybrid_asis']} "
              f"searching on the labels as they are, "
              f"{c['n_features_selected_by_at_least_one_top_hybrid_cleaned']} "
              f"searching on the cleaned labels")
        print(f"    of those, {c['n_features_both_searches_selected']} features "
              f"were selected by both, "
              f"{c['n_features_only_the_search_on_the_labels_as_they_are_selected']} "
              f"only by the search on the labels as they are, and "
              f"{c['n_features_only_the_search_on_the_cleaned_labels_selected']} "
              f"only by the search on the cleaned labels")
        print(f"    of the "
              f"{c['n_features_in_each_most_selected_list']} most-selected "
              f"features in each arm, "
              f"{c['n_of_the_most_selected_features_that_are_in_both_lists']} "
              f"are in both lists"
              + ('   (identity control: both counts must match)'
                 if c.get('is_the_identity_control') else ''))
        for f in c['features_whose_selection_frequency_moved_most'][:5]:
            print(f"      {f['representation']} column "
                  f"{f['index_within_that_representation']}: selected by "
                  f"{f['n_top_hybrids_that_selected_it_asis']} of "
                  f"{f['n_top_hybrids_built_asis']} hybrids searching on the "
                  f"labels as they are, "
                  f"{f['n_top_hybrids_that_selected_it_cleaned']} of "
                  f"{f['n_top_hybrids_built_cleaned']} searching on the "
                  f"cleaned labels")

    idt = s['search_path_identity']
    print(f"\n  the two searches are one code path, checked on "
          f"{idt['n_folds_probed']} folds:")
    print(f"    both scored the same number of candidates on "
          f"{idt['n_folds_where_both_searched_the_same_number_of_candidates']} "
          f"of {idt['n_folds_probed']} folds")
    print(f"    both picked the same recipe in {idt['n_recipes_that_matched']} "
          f"of {idt['n_recipes_that_matched'] + len(idt['recipes_that_did_not_match'])} "
          f"(fold, model) cells")
    print(f"    the four arms sit at most "
          f"{idt['max_spread_over_arms_anywhere']:.3g} apart over "
          f"{idt['n_cells_read']} cells"
          + (f" (worst at {idt['worst_cell']})" if idt['worst_cell'] else ''))
    if idt['recipes_that_did_not_match']:
        print("    ^ two identical label columns picked two different recipes. "
              "That is\n      n15's third defect, and nothing about cleaning "
              "can be read until it is fixed.")

    for regime, reg in s['per_regime'].items():
        print(f"\n  regime {regime}:")
        for lvl, lrow in ((f'{l} {b}', br) for l, lr in reg['per_level'].items()
                          for b, br in lr['per_base'].items()):
            print(f"    level {lvl}:")
            for cleanup, crow in lrow['per_cleanup'].items():
                if cleanup == IDENTITY_CONTROL:
                    continue
                for rk, rrow in crow['per_flag_rate'].items():
                    for m, mrow in rrow['per_model'].items():
                        oc = mrow.get('ordering_cost')
                        if not oc or not oc['n_folds_paired']:
                            continue
                        mark = '  >spread' if oc['gain_exceeds_its_own_spread'] else ''
                        _sd = oc['spread_over_fold_gains']
                        print(f"      {cleanup:13s} flag {rk:5s} {m:22s} "
                              f"searching on the cleaned labels won "
                              f"{oc['folds_won']} of {oc['n_folds_paired']} "
                              f"folds, mean {oc['mean_gain']:+.4f} "
                              + (f"(spread {_sd:.4f})" if _sd is not None
                                 else "(one fold, no spread)")
                              + mark)
    print("\n  The count of folds won is the headline. A mean gain no larger "
          "than the\n  spread of the per-fold gains behind it is not a gain.")


def main():
    parser = create_experiment_parser(
        'N20: does the cleaning have to come before the recipe search')
    parser.add_argument('--flag-rates', type=str, default=None,
                        help='Comma-separated shares of rows to treat as suspect')
    parser.add_argument('--noise-condition', type=str, default=None,
                        help=f'Injected condition (default {DEFAULT_NOISE_CONDITION})')
    parser.add_argument('--regimes', type=str, default=None,
                        help='Comma-separated: injected, as_they_come')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, rep_preset=args.reps)
    kw = {}
    if args.flag_rates:
        kw['flag_rates'] = [float(x) for x in args.flag_rates.split(',')]
    if args.noise_condition:
        kw['noise_condition'] = args.noise_condition
    if args.regimes:
        kw['regimes'] = [x.strip() for x in args.regimes.split(',')]
    results = run(data, config, args.output, args.dataset, **kw)
    save_experiment_results(results, args.output, 'n20_clean_before_search',
                            args.dataset,
                            filename_suffix=kw.get('noise_condition'))


if __name__ == '__main__':
    main()
