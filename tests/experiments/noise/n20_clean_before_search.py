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

WHAT CHANGED IN THE ENGINE, AND WHAT DID NOT. `find_best_hybrid` takes
`train_labels=`: a full-length column the search TRAINS on while it is still
SCORED on `labels`. It refuses `label_noise=` alongside it, because both write
the column the search trains on. It refuses `rep_factories` alongside it,
because a fine-tuned embedding is fitted on `labels` and would carry the
uncleaned labels straight back in. The cache identity hashes the actual column,
so two label columns cannot read each other's stored rows and no wipe is needed.
Both arms here pass `train_labels=`, so the two searches go through one code
path and differ only in what that column holds.

WHICH CLEANUPS CAN REACH A FEATURE SELECTION AT ALL. Nine of the eleven n07
offers as of 2026-09-23, and the other two are refused rather than faked.

`cleaned_training_set` hands back a dict, and three of its entries decide this.
`label_column` is a full-length cleaned label column and is what the cleaned
search is given as `train_labels=`. `keep` is a mask over the training rows and
is what the cleaned search runs on. `sample_weight` is a per-row weight and is
read by the two FIT arms only. A cleanup that returns neither a label column
nor a mask has no cleaned search, so the pair this file compares does not exist
for it.

  label_smooth   full-length cleaned label column -> train_labels=
  knn_smooth     the same
  prox_smooth    the same; the neighbours come from a forest's leaves rather
                 than from Euclidean distance in the single representation
  oof_smooth     the same; the target is an out-of-fold LightGBM prediction
  gp_smooth      the same; the target is a Gaussian process posterior mean
  js_shrink      the same; how far each label moves is derived from two
                 variances rather than chosen
  quantile_clip  the same; only the labels sitting outside the 10th-to-90th
                 percentile range of the labels their own leaves carry are
                 rewritten
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
quantile_clip and js_shrink each move a label to a NUMBER, and
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

WHICH SINGLE REPRESENTATION. Owner's decision 2026-09-25: in each fold, the top
NOISE_BASE_REP_COUNT (2) single representations of BASE_RANK_MODEL ('rf') by its
score in that fold's own search on the UNCLEANED training labels -- the
search_asis every arm already reads. The detector and every cleanup run once in
each, as `base_1` and `base_2`, shared by every downstream model. No pool lookup
and no fallback: a fold whose search holds no rf score runs no cleanup, and
`base_reps_by_fold` says why. The ranking is read off the fold's training
molecules only, so no held-out molecule enters it. That is a real constraint rather than a
choice: a cleaning that reaches the feature selection cannot have been decided
in a feature space that does not yet exist. It also means the four arms flag the
identical molecules, so the only things differing between two arms are the two
named in the table.

THE CONTROL THAT CHECKS THE TWO SEARCHES ARE ONE CODE PATH. `no_cleaning` is run
as a cleanup: its cleaned column IS the uncleaned one. The two searches are then
handed byte-identical label columns, so they must produce the same recipe and
the four arms must produce four identical scores. A gap there is the two
searches choosing from lists of different lengths, which is the defect n15
carries at NOISE_AUDIT_2026_09_10.md section 4, n15 defect 3. It is reported as
`search_path_identity`, not hidden inside a tolerance.

BOTH REGIMES, because they answer different questions.

  injected       outlier_p10 on the shared grid (0.0, 0.2, 0.3, 0.5, 0.75,
                 1.0, 1.5 of the clean label spread; `dataset_policy.
                 _DEFAULT_NOISE_LEVELS`, read 2026-09-21). A corrupted subset
                 exists, so
                 `noise_helpers.detection_scores` is defined and is written.
                 Three draws per level, one at level 0.0 on a regression set
                 because the draw delivers exactly zero there.
  as_they_come   nothing is corrupted. n18's regime. The training labels are
                 what the loader produced and the held-out labels are graded
                 untouched, so a win is a win on the data as it arrived.

WHAT THE HELD-OUT MOLECULES ARE GRADED AGAINST. `labels[val_idx]`, untouched, in
every arm and both regimes. No cleaned label and no corrupted label is ever
scored against.

THE HEADLINE IS A COUNT OF FOLDS, not a mean, for n18's reason at :33-37 and
n19's at :823-833. Every comparison is PAIRED: the same fold, the same held-out
molecules, two arms. What varies between folds is common to both and cancels.

COST. One recipe search per fold per label column. `speed='fast'` with
`tune=True` and `tune_knn_k=True`, and the aggressive pruning schedule, matching
what n15 was set to on 2026-09-12 and for the same reason -- n15 also searches
inside the experiment, and that is what the settings are there to absorb. Both
arms get identical settings, so the comparison is like for like.

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

ONE THING TO KNOW BEFORE READING THE FLAG-RATE AXIS. Only four of the nine read
it, and on a regression dataset only three do. The flag rate sets a cut-off at
the (1 - rate) quantile of the suspicion score, so a cleanup that never
thresholds that score never sees the rate, and its three flag rates are one
number repeated:

  reads it          remove, knn_smooth, prox_smooth all threshold at that
                    quantile; label_smooth does too, but only on a
                    CLASSIFICATION label
  does not read it  on a regression label, label_smooth, oof_smooth, gp_smooth
                    and js_shrink move every label toward a target in
                    proportion to that molecule's suspicion score
                    (`noise_estimation.smooth_by_scores`, method
                    'proportional'; js_shrink derives its own fraction from two
                    variances). downweight turns the score into a weight with
                    no cut-off. quantile_clip reads neither the rate nor the
                    suspicion score at all -- it rewrites a label only when the
                    label sits outside the range its own leaves carry.

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
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_model_factories,
    get_best_for_downstream, resolve_pruning_kwargs,
    DOWNSTREAM_MODELS_NOISE, FULL_BUDGETS,
    noise_base_reps_by_model, NOISE_BASE_REP_COUNT,
)
from experiments.dataset_policy import (
    noise_levels, noise_groups_for, noise_dose_units,
)
from experiments.noise.noise_helpers import (
    noise_spec_for, inject_train_labels, detection_scores,
    detection_is_defined, hybrid_for_fold,
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
# The similarity axis, the D-versus-D' rows, the controls, the caches and the
# per-fold checkpoint, added 2026-09-25 (plan sections 1-3), for n20's
# CLEANUPS ONLY. The recipe searches are untouched: owner's instruction
# 2026-09-25, n20's search waits for a discussion (plan section 8).
from experiments.noise import noise_cache as nc
from experiments.noise import _sim_axis_n15_n22 as sx
from experiments.noise.similarity_space import SWEPT_KS

# The name the rows file and the checkpoint are filed under.
EXP_FULL_NAME = 'n20_clean_before_search'

# What a default-path row's similarity_setting says when the cleanup read a
# neighbour list. label_smooth / knn_smooth read the single representation the
# cleaning is decided in, under the process-wide distance (sx.TODAY);
# prox_smooth reads a forest fitted on that same matrix (n07:904-913).
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
    'oof_smooth', 'gp_smooth', 'quantile_clip', 'js_shrink',
    # Hands back a per-row weight and nothing the search can read. Two arms of
    # its four, and the absence is recorded rather than left blank.
    'downweight',
)


def cleanups_for(is_clf, cleanups=None):
    """The cleanups this task type can actually run, and what it dropped.

    Returns (run, dropped). A regression-only cleanup on a classification
    dataset is dropped here rather than raised inside the fold loop:
    `cleaned_training_set` raises ValueError on it (n07_remediation.py:879-882)
    and n20's own try/except would then file one error row per (fold, flag
    rate) for a cleanup that was never going to run.

    A random-flag name is dropped when the cleanup it runs is dropped.
    """
    asked = tuple(cleanups if cleanups is not None else CLEANUPS)
    if not is_clf:
        return asked, ()
    run, dropped = [], []
    for c in asked:
        strategy = RANDOM_FLAG_CLEANUPS.get(c, c)
        (dropped if strategy in REGRESSION_ONLY_CLEANUPS else run).append(c)
    return tuple(run), tuple(dropped)

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
#   quantile_clip  reads no suspicion score at all. It asks whether a label
#                  sits outside the range of label values the training
#                  molecules sharing its leaves carry, which is a question
#                  about the label and the range. n07 says the same by naming
#                  it in SCORE_FREE_CLEANUPS. Shuffling the score would
#                  hand back the identical column, so its twin would be the
#                  same fit run twice. Degenerate.
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
    'prox_smooth_random_flag': 'prox_smooth',
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

# The condition the injected regime runs under. outlier_p10 gives a corrupted
# SUBSET, which is what makes `noise_helpers.detection_scores` defined -- under
# gaussian, the student_t settings, laplace and grouped_shifted every training
# molecule moves and precision is 1.0 by construction.
DEFAULT_NOISE_CONDITION = 'outlier_p10'

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

# What every candidate inside the cleaned search is SCORED on: 'asis' is the
# training labels the practitioner holds, 'cleaned' is the same column the
# search trains on. Only the smoothers see this -- `remove` drops rows and its
# search is already scored and trained on the kept rows' labels either way.
SEARCH_SCORES_ON = 'asis'

# Whose ranking in each fold's own uncleaned search names the single
# representations the detector and the cleanups run in. Owner's decision
# 2026-09-25. See the module docstring.
BASE_RANK_MODEL = 'rf'


def search_settings(n_samples, pruning_mode='aggressive_pruning'):
    """The settings BOTH searches run under. n15's, for n15's reason.

    speed='fast' keeps the fast-tier allocation methods, importance methods,
    internal evaluators and representation counts. tune=True with
    tune_knn_k=True is the strongest tuning this repository offers, and it
    happens per fold on that fold's training rows only. The aggressive pruning
    schedule drops the tail of the candidate list at 25, 50 and 100 evaluations.

    Both arms get this same dict. n15's third defect
    (NOISE_AUDIT_2026_09_10.md section 4) was that its two arms went through
    different call paths and so tried different numbers of allocation methods;
    here there is one call path and one settings dict, and `no_cleaning` checks
    that at run time.
    """
    out = {'speed': 'fast', 'tune': True, 'tune_n_iter': 20,
           'tune_knn_k': True}
    out.update(resolve_pruning_kwargs(pruning_mode, n_samples,
                                      dataset_label='n20'))
    return out


def _levels_and_draws(dataset_name, condition, is_clf, levels=None, draws=None):
    """(level, draw) pairs for the injected regime.

    At level 0.0 on a regression dataset the draw delivers exactly zero, so
    three draws are three identical runs and one is kept. NOT on a
    classification set: the flip calibrator returns a floor rather than zero at
    a target of 0.0, so a few labels flip and a different number of them on
    every draw. Same rule as noise_helpers.noisy_search_specs.

    `levels` and `draws` exist for the smoke, which checks the machinery rather
    than the grid. A run that narrows either writes what it ran to the result
    file, so a short grid reads as a short grid and not as the shared one.
    """
    levels = (noise_levels(dataset_name, strategy=condition, is_clf=is_clf)
              if levels is None else levels)
    NOISE_DRAWS_HERE = tuple(NOISE_DRAWS if draws is None else draws)
    out = []
    for lvl in levels:
        here = (NOISE_DRAWS_HERE if (float(lvl) > 0 or is_clf)
                else NOISE_DRAWS_HERE[:1])
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


def _run_one_search(train_reps, y_score, y_train_col, groups_train, *,
                    dataset_name, task_type, metric_name, models, n_folds,
                    budgets, settings, namespace, cache, cache_dir, verbose):
    """One recipe search over a fold's training rows.

    `y_score` is what every candidate is SCORED on -- the labels as they are, in
    both arms, because the outer grading is against untouched labels too and the
    inner selection has to match it. `y_train_col` is what every candidate is
    FITTED on: the same column in the searched-on-as-it-is arm, the cleaned one
    in the searched-on-cleaned arm. Passing `train_labels=` in both arms is what
    puts them on one code path.

    Returns a SearchResult, or None with the reason printed.
    """
    from kirby.find_best_hybrid import find_best_hybrid
    try:
        return find_best_hybrid(
            base_reps=train_reps,
            labels=y_score,
            train_labels=y_train_col,
            groups=groups_train,
            budgets=list(budgets),
            models=list(models),
            metric=metric_name,
            task_type=task_type,
            n_folds=n_folds,
            dataset_name=dataset_name,
            tier='general',
            cache=cache,
            cache_namespace=namespace,
            verbose=verbose,
            **({'cache_dir': cache_dir} if cache_dir else {}),
            **settings,
        )
    except Exception as exc:
        print(f"      search failed ({namespace}): "
              f"{type(exc).__name__}: {exc}")
        return None


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


def _stored_configurations(dataset_name, namespace, model, cache_dir):
    """Every configuration this one search SCORED and stored, newest schema only.

    Not the winner, and not `all_results`. `all_results` holds what this
    invocation computed: a configuration whose score was already in the store
    is read from there and never re-enters that list, which is why n20's
    candidate count reads 1 on a search that compared more. The store is the
    complete record.

    Each n20 search writes to its own file, because find_best_hybrid appends
    the cache namespace to the dataset name before building the path
    (find_best_hybrid.py:8430). So `result_db/<dataset>__<namespace>.db` holds
    this fold's one label column and nothing else.

    One file can still hold more than one identity -- a different
    representation pool or a different fold count is a different search under
    the same namespace -- so the rows are grouped by identity and the largest
    group is used. How many identities were found is returned, so a file
    holding two does not read as a file holding one.

    Returns (rows, note). `rows` is every stored row for the chosen identity
    whose `model` column is this downstream model, each with its blueprint
    unpickled and its `score_mean` and `budget` read. Empty with a note when
    the file does not exist or cannot be opened.
    """
    import sqlite3
    from kirby.result_cache import (ResultCache, IDENTITY_COLS, db_path_for,
                                    StaleResultDB)
    from kirby.find_best_hybrid import _sanitize_filename_part
    ds = f'{dataset_name}__{_sanitize_filename_part(namespace)}'
    path = db_path_for(ds, cache_dir=cache_dir)
    if not path.exists():
        return [], f'no store at {path.name}'
    try:
        conn = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        cols = ', '.join(IDENTITY_COLS)
        groups = conn.execute(
            f'SELECT {cols}, COUNT(*) FROM results GROUP BY {cols}').fetchall()
        conn.close()
    except Exception as exc:
        return [], f'{type(exc).__name__}: {exc}'
    if not groups:
        return [], 'store is empty'
    groups.sort(key=lambda r: r[-1], reverse=True)
    identity = dict(zip(IDENTITY_COLS, groups[0][:-1]))
    try:
        with ResultCache(ds, cache_dir=cache_dir) as rc:
            rows = rc.fetch_rows(identity)
    except StaleResultDB as exc:
        return [], f'StaleResultDB: {exc}'
    except Exception as exc:
        return [], f'{type(exc).__name__}: {exc}'
    rows = [r for r in rows
            if str(r.get('model')) == str(model)
            and r.get('score_mean') is not None
            and r.get('blueprint') is not None]
    note = (f'{len(rows)} stored configurations for this downstream model, '
            f'over {len(groups)} search identities in the file')
    return rows, note


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


def _stack(a, b):
    """One matrix per representation, this fold's training rows then its held-out ones."""
    return {k: np.vstack([np.asarray(a[k]), np.asarray(b[k])]) for k in a}


def run(data, config, output_dir, dataset_name, **kwargs):
    """N20: does the cleaning have to come before the recipe search."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N20: DOES THE CLEANING HAVE TO COME BEFORE THE SEARCH")
    print("=" * 70)

    labels = np.asarray(data['labels'])
    reps = data['reps']
    groups = data.get('groups')
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'

    condition = kwargs.get('noise_condition') or DEFAULT_NOISE_CONDITION
    regimes = tuple(kwargs.get('regimes') or REGIMES)
    # Whatever the caller asked for is filtered by task type, not only the
    # default list: a caller naming oof_smooth on a classification dataset is
    # naming something `cleaned_training_set` refuses.
    cleanups, cleanups_dropped = cleanups_for(
        is_clf, kwargs.get('cleanups') or None)
    flag_rates = tuple(float(r) for r in (kwargs.get('flag_rates') or FLAG_RATES))
    budgets = tuple(kwargs.get('budgets') or FULL_BUDGETS)
    search_n_folds = int(kwargs.get('search_n_folds', 3))
    cache = bool(kwargs.get('cache', True))
    cache_dir = kwargs.get('cache_dir')
    search_verbose = bool(kwargs.get('search_verbose', False))
    # What the candidates inside the cleaned search are SCORED on. See the
    # comment at the branch that uses it. 'asis' is what ran before 2026-09-20.
    # HOW MANY of the stored configurations the per-feature table is built
    # over, per search. 0 turns the whole per-feature measurement off, which is
    # what every run before 2026-09-21 did. It is off by default because each
    # configuration in the list has to be BUILT to find out which features it
    # selected -- a Blueprint does not carry them -- and a build is a feature
    # ranking plus an accept/reject pass per representation.
    per_feature_top_n = int(kwargs.get('per_feature_top_n') or 0)
    search_scores_on = str(kwargs.get('search_scores_on', SEARCH_SCORES_ON))
    if search_scores_on not in ('asis', 'cleaned'):
        raise ValueError(
            f"n20: search_scores_on must be 'asis' or 'cleaned', "
            f"got {search_scores_on!r}")

    # ⛔ config FIRST, then the kwarg. run_all passes the caller's model list in
    # `config['models']` and never as a `models` kwarg -- the only kwarg it ever
    # forwards is noise_condition (noise_helpers.py:1053). This file read
    # kwargs alone, so the list the runner was asked for was silently dropped
    # and every run fell back to the module default. Eight files in this
    # directory had it; the same files read config['n_splits'] correctly two
    # lines away. Fixed 2026-09-20. An explicit kwarg still wins, which is what
    # a smoke or a direct caller uses.
    downstream_models = list(
        kwargs.get('models') or config.get('models') or DOWNSTREAM_MODELS_NOISE)
    model_factories = get_model_factories(
        downstream_models, task_type=task_type, n_samples=len(labels))
    downstream_models = [m for m in downstream_models if m in model_factories]
    if not downstream_models:
        raise RuntimeError(
            f"n20: no downstream model could be built. Asked for "
            f"{list(kwargs.get('models') or DOWNSTREAM_MODELS_NOISE)}, got "
            f"{sorted(model_factories)}.")

    # The single representations the detector and the cleaning run in are
    # ranked inside each fold, from that fold's own uncleaned search; see the
    # module docstring. {regime|level|fold: {base_1: rep, base_2: rep}}.
    base_reps_by_fold = {}
    settings = search_settings(len(labels),
                               kwargs.get('pruning_mode', 'aggressive_pruning'))
    # A caller may narrow the search, and the smoke does — it checks the
    # machinery, not the cost. What it narrowed is written to the result file
    # under 'search_settings', so a narrowed run reads as a narrowed run.
    settings.update(kwargs.get('search_settings') or {})
    noise_groups = noise_groups_for(data)[0]

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
        'quantile_clip': (
            'produces a full-length cleaned label column. Only the labels '
            'outside the 10th-to-90th percentile range of the labels their '
            'own leaves carry are rewritten; the rest of the column is '
            'untouched. It reads no suspicion score, so it has no '
            'random-flag twin'),
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
        'prox_smooth_random_flag': (
            'produces a full-length cleaned label column, exactly as '
            '`prox_smooth` does and moving exactly as many labels, chosen by '
            'a shuffled suspicion score instead of the detector'),
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
        'regimes': list(regimes),
        'noise_condition': condition,
        'dose_units': noise_dose_units(dataset_name),
        'cleanups': list(run_cleanups),
        'cleanups_not_run': list(cleanups_dropped),
        'cleanups_not_run_note': (
            'Named on a classification dataset only. Each of these moves a '
            'label to a NUMBER, and cleaned_training_set raises on a nominal '
            'class label rather than approximating one. n07 names the same '
            'four in REGRESSION_ONLY_CLEANUPS, which this file imports rather '
            'than copying.'),
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
        'models': downstream_models,
        'base_rank_model': BASE_RANK_MODEL,
        'base_reps_note': (
            f'The detector and every cleanup ran once in each of '
            f"{BASE_RANK_MODEL}'s top {NOISE_BASE_REP_COUNT} single "
            f'representations, ranked by its score in each fold\'s own search '
            f'on the uncleaned training labels. base_reps_by_fold names them '
            f'per fold, or says why a fold had none. per_regime and summary '
            f'are keyed by base_1 / base_2 under each level.'),
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
        'search_settings': {k: v for k, v in settings.items()
                            if k not in ('adaptive_pruning_schedule',
                                         'model_pruning_aggressiveness')},
        'adaptive_pruning_schedule': settings.get('adaptive_pruning_schedule'),
        'search_n_folds': search_n_folds,
        'search_scores_on': search_scores_on,
        'search_scores_on_note': (
            "What every candidate inside the CLEANED search was scored on. "
            "'asis' is the training labels the practitioner holds, which under "
            "the injected regime are the corrupted ones. 'cleaned' is the same "
            "column the search trains on. Only the smoothers read it: `remove` "
            "drops rows and its search is scored and trained on the kept rows' "
            "labels under either setting."),
        'search_budgets': list(budgets),
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
        'per_regime': {},
        'detection': {},
    }

    sample_X = next(iter(reps.values()))
    splits = list(cv_split(groups, n_folds, sample_X, labels, mode=mode))
    passes = {'as_they_come': [(None, 0)]}
    if 'injected' in regimes:
        passes['injected'] = _levels_and_draws(
            dataset_name, condition, is_clf,
            levels=kwargs.get('levels'), draws=kwargs.get('draws'))
        results['noise_levels'] = sorted({l for l, _ in passes['injected']})
        results['noise_draws_per_level'] = {
            f'{l:g}': sorted({d for ll, d in passes['injected'] if ll == l})
            for l in results['noise_levels']}

    n_cols = 1 + len(run_cleanups) * len(flag_rates)
    n_passes = sum(len(passes[r]) for r in regimes)
    print(f"  {len(labels)} molecules, {len(splits)} folds, "
          f"models {downstream_models}")
    print(f"  cleanups {list(run_cleanups)}, flag rates {list(flag_rates)}")
    if cleanups_dropped:
        print(f"  not run on a classification label, each moves a label to a "
              f"number: {list(cleanups_dropped)}")
    print("  not run at all: huber_weight and invvar_weight. Each returns a "
          "per-row\n  weight and no label column and no row mask, so neither "
          "has a cleaned search")
    print(f"  {n_passes} (level, draw) passes over {len(regimes)} regimes; "
          f"up to {n_cols} label columns per fold, so up to "
          f"{n_passes * len(splits) * n_cols} recipe searches")
    print(f"  search: {settings['speed']} tier, tune={settings['tune']}, "
          f"tune_knn_k={settings['tune_knn_k']}, "
          f"{search_n_folds} inner folds, budgets {list(budgets)}")

    # acc[regime][level_key][cleanup][rate][model][arm][fold_key] = score
    acc = {}
    errors = {}
    detection_acc = {}
    # Under the identity control the two searches are handed byte-identical
    # label columns. What is recorded here is what they did with them: how long
    # each one's candidate list was, and what recipe each one picked. That is
    # the check read directly rather than inferred from a score.
    identity_probe = {}
    # THE SAME THING, FOR THE CLEANUPS THAT ARE NOT THE CONTROL, and it is the
    # half of this file's question that nothing was reporting. The control
    # answers "are the two searches one code path". It cannot answer "does
    # cleaning first make the search pick a different recipe", because under
    # the control the two label columns are identical by construction and the
    # recipes MUST match. That question is about `remove` and the smoothers,
    # and their recipes were computed and then thrown away.
    recipe_probe = {}
    # THE PER-FEATURE VIEW. The recipe probe above answers "did the two
    # searches pick different recipes". It cannot answer "did they pick
    # different FEATURES", because two recipes that name the same
    # representations, allocation method, importance method and budget still
    # rank that representation's features against whatever label column they
    # were handed. One row here is one FEATURE, a (representation, index inside
    # that representation) pair, and the number is how many of the top stored
    # configurations selected it.
    per_feature = {}
    refuses_weight = set()

    # ── The main measurement, the similarity axis and the checkpoint ──
    # Added 2026-09-25 (plan sections 1-3), for the cleanups only.
    #
    # ROWS. D is `search_asis__fit_asis`; D' is each of the other three arms,
    # one d_vs_dprime row per (regime, level, draw, fold, single
    # representation, cleanup, flag rate, arm, model), built from the scores
    # the loop already computes -- no refit. score_clean, under the injected
    # regime, is the model fitted on the CLEAN training labels in the
    # searched-on-as-it-is features, one extra fit per (fold, model).
    #
    # THE SIMILARITY SWEEP, FIT SIDE ONLY. label_smooth and knn_smooth, and
    # prox_smooth under forest_proximity, re-find their neighbours in every
    # setting of sx.sweep_settings() at every k in SWEPT_KS -- BASE_RANK_MODEL's
    # top-2 single representations of this fold, each plus augmented features,
    # its searched-on-as-it-is hybrid, ECFP4 Tanimoto, forest proximity -- and
    # only the `search_asis__fit_cleaned` arm is refitted with the result.
    # TODO(similarity-sweep 2026-09-25): the two searched-on-cleaned arms are
    # NOT swept. Each cleaned label column needs its own recipe search, and a
    # similarity setting gives each smoother a new column per (setting, k), so
    # the search count multiplies by the number of settings (plan section 8).
    # That waits for the owner. hybrid_similarity is unavailable here: the
    # fold's own search is run under the predictive objective only.
    #
    # CHECKPOINT. Every accumulator is pickled after every (regime, level,
    # draw, fold), so a killed run resumes at the next one; the recipe
    # searches themselves are already on disk under their namespaces.
    metric = (metric_name, metric_fn, lower_is_better)
    rows = sx.Rows(output_dir, EXP_FULL_NAME, dataset_name,
                   condition=condition if 'injected' in regimes else None)
    sweep = sx.sweep_settings()
    results['rows'] = rows.summary()
    results['similarity_settings_swept'] = [s_.name for s_ in sweep]
    results['similarity_ks_swept'] = list(SWEPT_KS) if sweep else []
    # Which neighbours every number OUTSIDE the rows and the similarity axis was
    # computed with, so each result record names its similarity setting.
    results['similarity_setting_of_the_default_path'] = dict(_DEFAULT_SIM)
    results['similarity_arms_swept'] = ['search_asis__fit_cleaned']
    sim_unavailable = {}
    augmentations_all = data.get('augmentations') or None
    _rank_factory = (model_factories.get(BASE_RANK_MODEL) or get_model_factories(
        [BASE_RANK_MODEL], task_type=task_type,
        n_samples=len(labels)).get(BASE_RANK_MODEL))
    ck = sx.FoldCheckpoint(
        output_dir, EXP_FULL_NAME, dataset_name,
        sx.run_signature(data, config, {
            'models': downstream_models, 'cleanups': run_cleanups,
            'flag_rates': flag_rates, 'regimes': regimes,
            'condition': condition, 'passes': passes, 'budgets': budgets,
            'search_n_folds': search_n_folds, 'settings': settings,
            'search_scores_on': search_scores_on,
            'per_feature_top_n': per_feature_top_n}),
        condition=condition if 'injected' in regimes else None)
    if ck.state is not None:
        (acc, errors, detection_acc, identity_probe, recipe_probe,
         per_feature, _rw, base_reps_by_fold, sim_unavailable) = ck.state
        refuses_weight = set(_rw)

    def _on_hybrid(cleaned, X_tr_a, y0):
        # The cleaning decided in the single representation, applied to the
        # rows of the searched-on-as-it-is hybrid -- what the
        # search_asis__fit_cleaned arm fits (the loop below, 'the two arms
        # whose features came from the labels as they are').
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

    def _sweep_base(regime, level, draw, fold_i, X_base_tr, y0, suspicion,
                    targets, g_tr, train_idx, val_idx, train_reps, val_reps,
                    fold_base, base_sp, base_rn, space_asis, D_by_model,
                    clean_by_model, y_val, search_asis):
        """The fit-side arm under every similarity setting and k."""
        best_rf = get_best_for_downstream(search_asis, BASE_RANK_MODEL,
                                          budget=None)
        recipes = {'hybrid': best_rf['blueprint']} if best_rf else {}
        aug_tr = aug_va = None
        if augmentations_all:
            aug_tr = {a: np.asarray(m_[train_idx], dtype=np.float64)
                      for a, m_ in augmentations_all.items()}
            aug_va = {a: np.asarray(m_[val_idx], dtype=np.float64)
                      for a, m_ in augmentations_all.items()}
        sp_tr, _sp_va, unav = sx.build_spaces(
            dataset_name, sweep, X_model_tr=X_base_tr, train_reps=train_reps,
            val_reps=val_reps, train_idx=train_idx,
            base_names=list(fold_base.values()), recipes=recipes, y_train=y0,
            groups_train=g_tr, model_name=BASE_RANK_MODEL,
            model_factory=_rank_factory, metric=metric, task_type=task_type,
            augmentations_train=aug_tr, augmentations_val=aug_va)
        smiles_tr = sx.smiles_for(data, train_idx)
        for setting in sweep:
            if setting.space in unav:
                sim_unavailable.setdefault(setting.name, unav[setting.space])
                continue
            nb, why = sx.neighbour_lists(dataset_name, setting, sp_tr, y0,
                                         smiles_train=smiles_tr, is_clf=is_clf)
            if nb is None:
                sim_unavailable.setdefault(setting.name, why)
                continue
            for k in sx.ks_held(nb):
                sim = sx.sim_name(setting, k)
                for cleanup in run_cleanups:
                    if (cleanup == IDENTITY_CONTROL
                            or cleanup in RANDOM_FLAG_CLEANUPS
                            or not sx.strategy_reads_setting(cleanup, setting)):
                        continue
                    for rate in flag_rates:
                        def _clean(sc_, rate_, _c=cleanup, _k=k):
                            return cleaned_training_set(
                                _c, X_base_tr, y0, sc_, rate_, is_clf=is_clf,
                                groups=g_tr, targets=targets,
                                similarity_neighbours=nb, similarity_k=_k)
                        try:
                            cleaned = _clean(suspicion, rate)
                        except Exception as exc:
                            print(f"      {cleanup} under {sim} failed: "
                                  f"{type(exc).__name__}: {exc}")
                            cleaned = None
                        for mname, (X_tr_a, X_va_a) in space_asis.items():
                            ident = _ident(regime, level, draw, fold_i,
                                           cleanup, rate, base_sp, base_rn,
                                           ARM_TODAY, f'hybrid_asis:{mname}',
                                           sim)
                            if rows.is_done(ident, [mname]) and not \
                                    sx.controls_enabled():
                                continue
                            _kw = dict(
                                X_tr=X_tr_a, y_D=y0, X_va=X_va_a, y_va=y_val,
                                metric=metric,
                                factories={mname: model_factories[mname]},
                                score_D=({mname: D_by_model[mname]}
                                         if mname in D_by_model else None),
                                score_clean=({mname: clean_by_model[mname]}
                                             if mname in clean_by_model
                                             else None))
                            sx.dprime_rows(rows, dataset_name, ident,
                                           _on_hybrid(cleaned, X_tr_a, y0),
                                           **_kw)
                            sx.control_rows(
                                rows, dataset_name, ident,
                                _on_hybrid(cleaned, X_tr_a, y0),
                                seed=sx.seed_of(regime, level, draw, fold_i,
                                                base_sp, cleanup, rate, mname,
                                                sim),
                                is_clf=is_clf,
                                rerun=(lambda sc_, r_, _X=X_tr_a:
                                       _on_hybrid(_clean(sc_, r_), _X, y0)),
                                **_kw)

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

            spec = None
            if regime == 'injected':
                spec = noise_spec_for(dataset_name, condition, level,
                                      noise_seed=draw, is_clf=is_clf)

            for fold_i, (train_idx, val_idx) in enumerate(splits):
                _unit = f'{regime}|{lvl_key}|d{draw}|f{fold_i}'
                if ck.is_done(_unit):
                    print(f"    fold {fold_i}: already in the checkpoint, "
                          f"skipped")
                    continue
                fold_rows = []
                fold_key = f'l{lvl_key}d{draw}f{fold_i}'
                y_val = labels[val_idx]
                g_tr = groups[train_idx] if groups is not None else None

                # The labels a practitioner holds for this fold's training
                # molecules. Corrupted under the injected regime; exactly what
                # the loader produced under the other.
                cmask = None
                if spec is None:
                    y0 = np.asarray(labels[train_idx])
                else:
                    try:
                        y0, cmask, _prov = inject_train_labels(
                            labels, spec, train_idx=train_idx, is_clf=is_clf,
                            groups=noise_groups, dataset_name=dataset_name)
                    except Exception as exc:
                        errors[f'{regime}|{lvl_key}|d{draw}|f{fold_i}|inject'] = (
                            f'{type(exc).__name__}: {exc}')
                        print(f"    fold {fold_i}: injection failed: "
                              f"{type(exc).__name__}: {exc}")
                        continue

                train_reps = {k: np.asarray(v[train_idx], dtype=np.float64)
                              for k, v in reps.items()}
                val_reps = {k: np.asarray(v[val_idx], dtype=np.float64)
                            for k, v in reps.items()}

                # The search on the labels as they are. One per fold, shared by
                # every cleanup and every flag rate -- they all have to be read
                # against the same recipe or the comparison is not paired.
                ns = f'n20_{regime}_{lvl_key}_d{draw}_f{fold_i}'
                search_asis = _run_one_search(
                    train_reps, y0, y0, g_tr,
                    dataset_name=dataset_name, task_type=task_type,
                    metric_name=metric_name, models=downstream_models,
                    n_folds=search_n_folds, budgets=budgets,
                    settings=settings, namespace=ns, cache=cache,
                    cache_dir=cache_dir, verbose=search_verbose)
                if search_asis is None:
                    errors[f'{regime}|{lvl_key}|d{draw}|f{fold_i}|search_asis'] = (
                        'the search on the labels as they are did not return')
                    continue

                # The features chosen from the labels as they are, per model.
                # Built once per (fold, model) and shared by the two arms whose
                # search half is 'asis'.
                space_asis = {}
                for mname in downstream_models:
                    best = get_best_for_downstream(search_asis, mname,
                                                   budget=None)
                    if not best:
                        continue
                    # Through noise_cache since 2026-09-25 (plan section
                    # 3): the same hybrid_for_fold call, stored under its
                    # inputs, so the similarity sweep's 'hybrid' space for
                    # BASE_RANK_MODEL reads this build back.
                    X_tr_h, X_va_h = nc.cached_hybrid_for_fold(
                        dataset_name, train_reps, val_reps, y0,
                        best['blueprint'], g_tr, model_factories[mname](),
                        metric_fn, lower_is_better, model_name=mname,
                        metric_name=metric_name, train_idx=train_idx)
                    if X_tr_h is None:
                        continue
                    s = StandardScaler()
                    space_asis[mname] = (s.fit_transform(X_tr_h),
                                         s.transform(X_va_h))

                # The model fitted on the CLEAN training labels in those same
                # features, the row's score_clean. Injected regime only: under
                # the other there is no clean column. Added 2026-09-25.
                clean_by_model = {}
                if spec is not None:
                    for mname, (X_tr_a, X_va_a) in space_asis.items():
                        clean_by_model[mname] = sx.fit_score(
                            dataset_name, mname, model_factories[mname],
                            X_tr_a, np.asarray(labels[train_idx]), X_va_a,
                            y_val, metric)

                # ── THE SINGLE REPRESENTATIONS, FROM THIS FOLD'S OWN SEARCH ──
                # Owner's decision 2026-09-25: BASE_RANK_MODEL's top
                # NOISE_BASE_REP_COUNT single representations by its score in
                # the search just run on this fold's UNCLEANED training labels.
                # The detector and every cleanup run once in each, shared by
                # every downstream model as before. No pool lookup and no
                # fallback: a fold whose search holds no score for that model
                # runs no cleanup, and the reason is written.
                _ranked, _why_none = noise_base_reps_by_model(
                    {BASE_RANK_MODEL: search_asis}, [BASE_RANK_MODEL],
                    reps=train_reps, dataset_name=dataset_name, verbose=False)
                fold_base = {f'base_{_k}': _rn for _k, _rn in enumerate(
                    _ranked.get(BASE_RANK_MODEL) or [], start=1)}
                base_reps_by_fold[f'{regime}|{lvl_key}|{fold_key}'] = (
                    dict(fold_base) if fold_base else
                    {'none': _why_none.get(BASE_RANK_MODEL, 'no ranking')})
                if not fold_base:
                    errors[f'{regime}|{lvl_key}|d{draw}|f{fold_i}|base_reps'] = (
                        f'no single representation: '
                        f'{_why_none.get(BASE_RANK_MODEL)}. No detector and no '
                        f'cleanup ran on this fold.')
                    print(f"    fold {fold_i}: no single representation for "
                          f"{BASE_RANK_MODEL} in this fold's search; fold skipped")
                    continue

                for base_sp, base_rn in fold_base.items():
                    cell = acc[regime][lvl_key].setdefault(base_sp, {})
                    # This single representation: where the detector runs and
                    # where the cleaning is decided, once per (fold, single
                    # representation) for all four arms.
                    sc = StandardScaler()
                    X_base_tr = sc.fit_transform(train_reps[base_rn])
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
                        errors[f'{regime}|{lvl_key}|d{draw}|f{fold_i}|'
                               f'{base_sp}|detector'] = (
                            'the detector would not fit on this fold')
                        print(f"    fold {fold_i} {base_sp} ({base_rn}): "
                              f"detector failed, dropped")
                        continue

                    # ⛔ THE MODEL-BASED CLEANUPS' TARGETS, ONCE PER FOLD. oof_smooth,
                    # gp_smooth, js_shrink and quantile_clip each fit a model on
                    # part of this fold's training molecules and read a target off
                    # the rest. The target depends on the features, the labels and
                    # the groups and on nothing else -- not on the suspicion score,
                    # not on the flag rate, not on which cleanup is about to use
                    # it. Left inside cleaned_training_set it is refitted for every
                    # flag rate and every cleanup that wants it: 3 rates x 4
                    # cleanups where 1 fit of each does, and the Gaussian process
                    # is cubic in the training molecules with REPAIR_OOF_SPLITS
                    # inner folds per call. Same hoist n18 does at n18:805 and the
                    # same argument `_compute_noise_scores` carries for the
                    # detector.
                    #
                    # The want-list is derived from n07.CLEANUP_TARGETS so a
                    # cleanup added there cannot be left without a target here.
                    # A target whose fit failed comes back None and that one
                    # cleanup builds its own or declines; the fold is not dropped.
                    #
                    # ⛔ AN EMPTY WANT-LIST IS NOT "NOTHING". correction_targets
                    # reads `want = set(want or ('oof', 'gp', 'quantile'))`
                    # (n07_remediation.py:482), so handing it an empty list asks
                    # for all three -- including the Gaussian process. A run whose
                    # cleanups need no target must not call it at all, and the
                    # smoke is such a run.
                    #
                    # ⛔ THE FOLD'S GROUPS GO IN, for the reason the detector's do:
                    # without them the inner split is a plain KFold, so a
                    # molecule's own scaffold or homology cluster trains the model
                    # that says what its label should have been.
                    _want = targets_needed(
                        [RANDOM_FLAG_CLEANUPS.get(c, c) for c in run_cleanups])
                    _targets = (correction_targets(X_base_tr, y0, groups=g_tr,
                                                   want=_want)
                                if _want else {})

                    # Did the detector find the molecules that really moved, before
                    # any cleanup ran? Only defined where a corrupted subset exists.
                    if cmask is not None and detection_is_defined(spec.strategy):
                        det = detection_scores(cmask, suspicion, spec.strategy)
                        if det is not None:
                            (detection_acc.setdefault(base_sp, {})
                             .setdefault(lvl_key, []).append(det))

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
                                        suspicion,
                                        f'{regime}|{lvl_key}|d{draw}|f{fold_i}')
                                else:
                                    score_used = suspicion
                                try:
                                    cleaned = cleaned_training_set(
                                        strategy, X_base_tr, y0, score_used, rate,
                                        is_clf=is_clf, groups=g_tr,
                                        targets=_targets)
                                except Exception as exc:
                                    errors[f'{regime}|{lvl_key}|{base_sp}|{cleanup}|{rate}|'
                                           f'f{fold_i}|clean'] = (
                                        f'{type(exc).__name__}: {exc}')
                                    continue
                                if cleaned is None:
                                    # The cleanup declined this fold. Four things
                                    # in cleaned_training_set return None and the
                                    # message cannot tell them apart from here, so
                                    # it names all four rather than asserting the
                                    # first: `remove` left under 10 training rows;
                                    # `gp_smooth` was over GP_SMOOTH_MAX_ROWS;
                                    # `prox_smooth` was over PROX_SMOOTH_MAX_ROWS;
                                    # `js_shrink` could not build an out-of-fold
                                    # prediction, or found no spread in the
                                    # residuals or in the suspicion scores. Each
                                    # one prints its own reason when it declines.
                                    errors[f'{regime}|{lvl_key}|{base_sp}|{cleanup}|{rate}|'
                                           f'f{fold_i}|clean'] = (
                                        f'{strategy} declined this fold: remove '
                                        f'leaves under 10 training molecules, '
                                        f'gp_smooth and prox_smooth refuse above '
                                        f'their row caps, js_shrink refuses '
                                        f'without an out-of-fold prediction or '
                                        f'without spread to shrink. The reason '
                                        f'was printed by the cleanup itself.')
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
                            # selection reads one.
                            search_cleaned = None
                            if col is not None or keep is not None:
                                if keep is not None:
                                    tr_for_search = {k: v[keep]
                                                     for k, v in train_reps.items()}
                                    y_s = y0[keep]
                                    y_t = y0[keep]
                                    g_s = g_tr[keep] if g_tr is not None else None
                                else:
                                    tr_for_search = train_reps
                                    # WHAT EVERY CANDIDATE INSIDE THE SEARCH IS
                                    # SCORED ON, and the two cleanup families did
                                    # not get the same treatment.
                                    #
                                    # `remove` drops rows, so the branch above
                                    # scores AND trains its search on the kept
                                    # rows' labels: the cleaning reaches both
                                    # halves of the selection.
                                    #
                                    # A smoother keeps every row, so this branch
                                    # trains on the cleaned column and scored every
                                    # candidate on y0 -- the training labels the
                                    # practitioner holds, which under the injected
                                    # regime are the CORRUPTED ones. The candidate
                                    # that best predicts corrupted labels then won
                                    # the selection, which is the thing the arm
                                    # exists to avoid.
                                    #
                                    # `search_scores_on` is the axis: 'asis' is
                                    # what ran before 2026-09-20, 'cleaned' scores
                                    # on the same column it trains on and puts the
                                    # smoothers on `remove`'s footing. Measured,
                                    # not assumed; the value that ran is written to
                                    # the result file.
                                    y_s = col if search_scores_on == 'cleaned' else y0
                                    y_t = col
                                    g_s = g_tr
                                # ⛔ THE SEARCH IS KEYED ON THE LABEL COLUMN, NOT
                                # THE FLAG RATE. Owner's decision 2026-09-18.
                                #
                                # Two of the four cleanups do not read the flag
                                # rate at all. IDENTITY_CONTROL hands back the
                                # original column whatever the rate (:552-555), and
                                # on a regression dataset label_smooth moves every
                                # label toward its neighbours in proportion to that
                                # molecule's suspicion score and never consults the
                                # rate -- the docstring at :100-105 says so.
                                #
                                # The namespace carried the rate, so those two ran
                                # the SAME search three times and could not hit each
                                # other's cache: 4 wasted searches of the 13 per
                                # (fold, pass), 1,105 searches per dataset instead
                                # of 765. Those two counts are the four-cleanup
                                # list this file ran until 2026-09-23 and are not
                                # re-derived for the nine; what the digest does is
                                # unchanged, and five of the nine hand back one
                                # column at all three rates.
                                #
                                # An 8-hex digest of the training label column
                                # replaces the rate. Two rates that produce the same
                                # column now share one namespace and one cached
                                # search; two that produce different columns still
                                # get different namespaces, so the flag-rate axis is
                                # swept in full on the two cleanups that read it and
                                # no arm is lost.
                                #
                                # Degenerate, not a cut for compute.
                                #
                                # ⛔ EVERY CLEANED SEARCH ALREADY ON DISK IS FILED
                                # UNDER THE OLD SPELLING and will not be read again.
                                # The namespace is part of the cache path, so this
                                # is a cold rebuild of the cleaned arm, not a
                                # migration. The uncleaned arm's namespace is
                                # untouched.
                                # BOTH columns are in the key. The digest used to
                                # cover only the column the search TRAINS on, so
                                # two runs that scored their candidates on
                                # different columns read each other's stored rows
                                # and the second one's setting never reached a fit.
                                _col_key = _label_column_key(y_t)
                                _score_key = _label_column_key(y_s)
                                ns_c = f'{ns}_{cleanup}_{_col_key}_{_score_key}'
                                search_cleaned = _run_one_search(
                                    tr_for_search, y_s, y_t, g_s,
                                    dataset_name=dataset_name,
                                    task_type=task_type, metric_name=metric_name,
                                    models=downstream_models,
                                    n_folds=search_n_folds, budgets=budgets,
                                    settings=settings, namespace=ns_c,
                                    cache=cache, cache_dir=cache_dir,
                                    verbose=search_verbose)

                            if search_cleaned is not None:
                                probe_row = {
                                    'label_columns_identical': bool(
                                        col is not None and keep is None
                                        and np.array_equal(y0, col)),
                                    'n_training_rows_the_search_saw_asis':
                                        int(len(y0)),
                                    'n_training_rows_the_search_saw_cleaned':
                                        int(len(y_s)),
                                    'n_candidates_searched_asis':
                                        len(search_asis.all_results or []),
                                    'n_candidates_searched_cleaned':
                                        len(search_cleaned.all_results or []),
                                    'recipe_by_model': {
                                        m: {
                                            'asis': _fingerprint(
                                                get_best_for_downstream(
                                                    search_asis, m, budget=None)),
                                            'cleaned': _fingerprint(
                                                get_best_for_downstream(
                                                    search_cleaned, m,
                                                    budget=None)),
                                        } for m in downstream_models},
                                }
                                recipe_probe[f'{regime}|{lvl_key}|'
                                             f'{fold_key}/{base_sp}|'
                                             f'{cleanup}|{rate:g}'] = probe_row
                                if cleanup == IDENTITY_CONTROL:
                                    identity_probe[
                                        f'{regime}|{lvl_key}|'
                                        f'{fold_key}/{base_sp}'] = probe_row

                            # ── WHICH FEATURES, not which recipe ──
                            # Both searches' stored configurations are read from
                            # their own files, ranked in the dataset's own metric
                            # and own direction, and the best `per_feature_top_n`
                            # of each are BUILT so the features they selected can
                            # be counted. Under the identity control the two
                            # tables must come out equal; that is the check that
                            # the counting itself is not what moves.
                            if per_feature_top_n and search_cleaned is not None:
                                for mname in downstream_models:
                                    if mname not in model_factories:
                                        continue
                                    rows_a, note_a = _stored_configurations(
                                        dataset_name, ns, mname, cache_dir)
                                    rows_c, note_c = _stored_configurations(
                                        dataset_name, ns_c, mname, cache_dir)
                                    top_a = _rank_stored(rows_a, lower_is_better,
                                                         per_feature_top_n)
                                    top_c = _rank_stored(rows_c, lower_is_better,
                                                         per_feature_top_n)
                                    rs_a = _rep_scores_from_search(search_asis)
                                    rs_c = _rep_scores_from_search(search_cleaned)
                                    cnt_a, built_a, widths, ref_a = \
                                        _feature_frequency(
                                            top_a, train_reps, val_reps, y0, g_tr,
                                            model_factories[mname], metric_fn,
                                            lower_is_better, rep_scores=rs_a)
                                    cnt_c, built_c, _w2, ref_c = \
                                        _feature_frequency(
                                            top_c, tr_for_search, val_reps, y_t,
                                            g_s, model_factories[mname], metric_fn,
                                            lower_is_better, rep_scores=rs_c)
                                    fcell = _compare_feature_tables(
                                        cnt_a, built_a, cnt_c, built_c, widths)
                                    fcell['n_configurations_the_store_held_asis'] = \
                                        len(rows_a)
                                    fcell['n_configurations_the_store_held_cleaned'] = \
                                        len(rows_c)
                                    fcell['store_note_asis'] = note_a
                                    fcell['store_note_cleaned'] = note_c
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
                                        f'{regime}|{lvl_key}|{fold_key}|'
                                        f'{base_sp}|{cleanup}|{rate:g}/{mname}'] = fcell

                            rate_cell = (cell.setdefault(cleanup, {})
                                             .setdefault(f'{rate:g}', {}))

                            for mname in downstream_models:
                                arm_cell = rate_cell.setdefault(mname, {})

                                # ── the two arms whose features came from the
                                #    labels as they are ──
                                if mname in space_asis:
                                    X_tr_a, X_va_a = space_asis[mname]
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
                                            errors[f'{regime}|{lvl_key}|{base_sp}|{cleanup}|'
                                                   f'{rate}|{mname}|{arm}|'
                                                   f'f{fold_i}'] = err
                                            continue
                                        arm_cell.setdefault(arm, {})[fold_key] = s

                                # ── the two arms whose features came from the
                                #    cleaned labels ──
                                if search_cleaned is None:
                                    continue
                                best_c = get_best_for_downstream(search_cleaned,
                                                                 mname, budget=None)
                                if not best_c:
                                    continue
                                if keep is not None:
                                    # The hybrid is built on the kept rows. Both
                                    # the dropped training rows and the held-out
                                    # rows are then transformed through it, so the
                                    # fit-on-the-labels-as-they-are arm can still
                                    # train on every row.
                                    n_tr = len(train_idx)
                                    X_keep, X_rest = hybrid_for_fold(
                                        {k: v[keep] for k, v in train_reps.items()},
                                        _stack(train_reps, val_reps),
                                        y0[keep], best_c['blueprint'],
                                        (g_tr[keep] if g_tr is not None else None),
                                        model_factories[mname](), metric_fn,
                                        lower_is_better)
                                    if X_keep is None:
                                        continue
                                    # Scaled on EVERY training row, as the
                                    # other three arms are (space_asis above,
                                    # and the smoothers' branch below). Fitting
                                    # it on the kept rows alone made the
                                    # scaler a third thing differing between
                                    # the arms, beside the two the table
                                    # names; only svm could feel it. The
                                    # scaler reads features, never a label.
                                    s2 = StandardScaler().fit(X_rest[:n_tr])
                                    X_all = s2.transform(X_rest[:n_tr])
                                    X_va_c = s2.transform(X_rest[n_tr:])
                                    pairs = (
                                        ('search_cleaned__fit_cleaned',
                                         s2.transform(X_keep), y0[keep], None),
                                        ('search_cleaned__fit_asis',
                                         X_all, y0, None),
                                    )
                                else:
                                    X_tr_h, X_va_h = hybrid_for_fold(
                                        train_reps, val_reps, col,
                                        best_c['blueprint'], g_tr,
                                        model_factories[mname](), metric_fn,
                                        lower_is_better)
                                    if X_tr_h is None:
                                        continue
                                    s2 = StandardScaler()
                                    X_c = s2.fit_transform(X_tr_h)
                                    X_va_c = s2.transform(X_va_h)
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
                                        errors[f'{regime}|{lvl_key}|{base_sp}|{cleanup}|'
                                               f'{rate}|{mname}|{arm}|'
                                               f'f{fold_i}'] = err
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
                            _sim = _DEFAULT_SIM.get(_strat, sx.NO_SIMILARITY)
                            for mname in downstream_models:
                                _arms = rate_cell.get(mname, {})
                                _D = _arms.get('search_asis__fit_asis',
                                               {}).get(fold_key)
                                if _D is None:
                                    continue
                                for arm in ARMS[1:]:
                                    _P = _arms.get(arm, {}).get(fold_key)
                                    if _P is None and arm != ARM_TODAY:
                                        continue
                                    _mi = (f'hybrid_asis:{mname}'
                                           if arm.startswith('search_asis')
                                           else f'hybrid_searched_on_cleaned:'
                                                f'{mname}')
                                    fold_rows.append(sx.row(
                                        _ident(regime, level, draw, fold_i,
                                               cleanup, rate, base_sp, base_rn,
                                               arm, _mi, _sim),
                                        model=mname, metric=metric,
                                        score_D=_D, score_Dprime=_P,
                                        score_clean=clean_by_model.get(mname)))
                                # The shared twins of the fit-side arm, under
                                # KIRBY_NOISE_CONTROLS=1. n20's own
                                # random-flag control (a permuted suspicion
                                # score fed to BOTH searches) is a different
                                # control and stays.
                                if (sx.controls_enabled()
                                        and cleanup not in RANDOM_FLAG_CLEANUPS
                                        and mname in space_asis):
                                    _Xa, _Xva = space_asis[mname]
                                    sx.control_rows(
                                        rows, dataset_name,
                                        _ident(regime, level, draw, fold_i,
                                               cleanup, rate, base_sp,
                                               base_rn, ARM_TODAY,
                                               f'hybrid_asis:{mname}', _sim),
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
                                            {mname: clean_by_model[mname]}
                                            if mname in clean_by_model
                                            else None))

                    # ── The similarity sweep, fit side, in this single
                    #    representation ──
                    if sweep and space_asis:
                        _D_by_model = {}
                        for mname in space_asis:
                            for _c in acc[regime][lvl_key].get(base_sp, {}).values():
                                for _r in _c.values():
                                    _v = (_r.get(mname, {})
                                          .get('search_asis__fit_asis', {})
                                          .get(fold_key))
                                    if _v is not None:
                                        _D_by_model.setdefault(mname, _v)
                        _sweep_base(regime, level, draw, fold_i, X_base_tr, y0,
                                    suspicion, _targets, g_tr, train_idx,
                                    val_idx, train_reps, val_reps, fold_base,
                                    base_sp, base_rn, space_asis, _D_by_model,
                                    clean_by_model, y_val, search_asis)

                rows.write(fold_rows)
                ck.save(_unit, (acc, errors, detection_acc, identity_probe,
                                recipe_probe, per_feature,
                                sorted(refuses_weight), base_reps_by_fold,
                                sim_unavailable))
                print(f"    fold {fold_i} done [{time.time() - t0:.0f}s]")

    results['per_regime'] = convert_numpy(acc)
    results['base_reps_by_fold'] = base_reps_by_fold
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
                'How many of the top stored configurations, once built, '
                'selected that feature. The denominator is '
                'n_top_hybrids_built, which is how many of them produced a '
                'hybrid at all, not how many were asked for.'),
            'how_the_top_was_chosen': (
                'Every configuration the search stored for this downstream '
                'model, across every budget it scored, ranked by score_mean '
                'in the dataset metric and its own direction '
                f'(lower_is_better={lower_is_better}), best '
                f'{per_feature_top_n} kept.'),
            'per_cell': per_feature,
        })
    if errors:
        results['_errors'] = errors
    results['summary'] = convert_numpy(
        _summarise(acc, downstream_models, run_cleanups, flag_rates,
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
    """b against a, on the folds both scored. Positive gain always means better."""
    shared = sorted(set(a) & set(b))
    gains = np.array([(a[k] - b[k]) if lower_is_better else (b[k] - a[k])
                      for k in shared], dtype=float)
    gains = gains[np.isfinite(gains)]
    scale = max(abs(float(np.mean(list(a.values())))) if a else 1.0, 1.0)
    tol = 1e-9 * scale
    return {
        'n_folds_paired': int(gains.size),
        'folds_won': int((gains > tol).sum()),
        'folds_lost': int((gains < -tol).sum()),
        # A large tied count means the two arms are one fit.
        'folds_tied': int((np.abs(gains) <= tol).sum()),
        'tie_tolerance': float(tol),
        'mean_gain': float(gains.mean()) if gains.size else float('nan'),
        'spread_over_fold_gains': float(gains.std()) if gains.size else float('nan'),
        'fold_gains': [float(x) for x in gains],
        # A gain no larger than the spread of the per-fold gains behind it is
        # not a gain. Stated as a comparison of two measured numbers, not a
        # test: the suite's standing rule at this replicate count is raw spread
        # only.
        'gain_exceeds_its_own_spread': bool(
            gains.size > 0 and gains.mean() > gains.std()),
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
                                a: (float(np.mean(list(arms[a].values())))
                                    if arms.get(a) else float('nan'))
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
            c['rep_count_asis'].append(
                (pair['asis'] or {}).get('rep_count'))
            c['rep_count_cleaned'].append(
                (pair['cleaned'] or {}).get('rep_count'))
        c['candidate_counts_asis'].append(p['n_candidates_searched_asis'])
        c['candidate_counts_cleaned'].append(p['n_candidates_searched_cleaned'])
    out['recipe_the_search_selected'] = {
        'what_it_reports': (
            'One cell is one (regime, level, held-out fold and single '
            'representation, cleanup, flag rate, downstream model). For each, the recipe the search on the '
            'labels as they are picked and the recipe the search on the '
            'cleaned labels picked. They are compared on the representations, '
            'the allocation method, the importance method, the budget, the '
            'internal evaluator and the representation count. '
            'n_cells_where_the_recipe_moved is the count this experiment '
            'exists to produce: cleaning first changed what KIRBy selected in '
            'that many cells of that many.'),
        'candidate_count_note': (
            'n_candidates_searched is how many configurations the search '
            'returned in all_results. A count of 1 does NOT by itself mean the '
            'search compared one configuration: a configuration whose score is '
            'already in the per-dataset result store is read from there and '
            'never re-enters all_results, and the pruning schedule drops the '
            'tail of the list before the last folds are scored. Read it '
            'together with the representation counts below, which differ '
            'across folds exactly when the axis was live.'),
        'per_cleanup': per_cleanup,
    }
    return out


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
        print(f"    configurations the store held, for this downstream model: "
              f"{c['n_configurations_the_store_held_asis']} searching on the "
              f"labels as they are, "
              f"{c['n_configurations_the_store_held_cleaned']} searching on "
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
                        print(f"      {cleanup:13s} flag {rk:5s} {m:9s} "
                              f"searching on the cleaned labels won "
                              f"{oc['folds_won']} of {oc['n_folds_paired']} "
                              f"folds, mean {oc['mean_gain']:+.4f} "
                              f"(spread {oc['spread_over_fold_gains']:.4f})"
                              f"{mark}")
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
