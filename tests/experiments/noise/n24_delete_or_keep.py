#!/usr/bin/env python3
"""N24: once a detector has flagged molecules, delete them or keep them?

Paper: Noise.

THE QUESTION. Every cleanup result in this suite has raised it and none has
asked it directly. A detector scores every training molecule and a share of them
is flagged. Three things can be done with that flagged share:

  delete it              drop those molecules from the training set
  keep it, rewrite it    leave the molecules in and overwrite their labels
  keep it, weigh it down leave the molecules and their labels in, and give those
                         rows less weight in the fit

n07 runs all three and ranks them against each other. n11 runs the same list
against a grid of detectors. Neither asks whether the ranking between the three
survives the comparison below, and that comparison is this file's whole reason
for existing.

⛔ THE CONTROL IS THE POINT OF THE EXPERIMENT AND IT MUST NOT BE DROPPED. For
every arm that deletes molecules there is a matched arm that deletes THE SAME
NUMBER OF MOLECULES, chosen at random, at several seeds. Without it "the cleanup
beat no cleaning" is not evidence that the detector found anything: a smaller
training set is a different fit, and on a small dataset it is sometimes a better
one.

MEASURED 2026-09-23 on flab_thermo_tresanco, 3 folds, outlier_p10 at level 0.5,
deleting 45 of the 448 training molecules in every arm:

    the best detector-driven deletion   won 3 of 3 folds at +0.0697 spearman
    deleting 45 at random, seed 1       won 3 of 3 folds at +0.0688 spearman
    deleting 45 at random, a third seed won 1 of 3 folds at -0.0628 spearman

So beating no cleaning is not a bar. Beating every random seed is. Every number
below is reported per seed and the spread across seeds is printed; a mean across
seeds is never written, because the three rows above have a mean that says
nothing.

WHAT THE FLAG RATE IS HERE, AND WHY THE GRID IS NOT n18's. n18 sweeps
(0.01, 0.05, 0.20) because nobody knows what share of its real labels is wrong
and the repository's own ChEMBL probe put that share below one molecule in a
hundred. Here the share that was damaged is SET by the condition -- outlier_p05
gives the wider error to five molecules in a hundred, outlier_p10 to ten -- so
the grid spans the shares the conditions inject rather than sitting under them.

⛔ UNDER outlier_p01/p05/p10 THE FLAGGED MOLECULES CARRY ONLY PART OF THE
INJECTED ERROR. Those conditions give the selected share an error three times
wider and STILL MOVE EVERY OTHER LABEL, so deleting exactly the molecules the
injector picked leaves the rest of the damage in the training set. The three
`outlier_only_*` conditions widen the selected share and leave every other label
bit-identical. `kirby/noise_spec.py:99-112` measured the difference on 2,000
Gaussian labels at a dose of half their spread: the widened tenth carried 100.0%
of the squared error under outlier_only_p10 and 47.8% under outlier_p10.

That share is not carried over here as a claim. Every fold of this run measures
it on its own labels and writes it to `injury_carried_by_the_flagged_molecules`,
so a deletion arm that cannot win is readable against how much of the damage
deletion could possibly have removed.

WHICH ARMS TOUCH ONLY THE FLAGGED MOLECULES. Two of them, and the rest do not.
Read out of `n07.cleaned_training_set` in this session:

  remove        deletes the molecules above the (1 - flag rate) quantile of the
                suspicion score                              -- the flagged only
  knn_smooth    overwrites those same molecules' labels with the mean label of
                their ten nearest neighbours                  -- the flagged only
  label_smooth  moves EVERY label toward its neighbours' mean, each one in
                proportion to its own suspicion score         -- every molecule
  oof_smooth    the same, aiming at what a LightGBM trained on the other
                training molecules predicts for it            -- every molecule
  downweight    gives EVERY row a weight of 1/(1 + strength * score)
                                                              -- every molecule
  huber_weight  gives a row inside the scale a weight of exactly 1 and only the
                rows beyond it a smaller one                  -- every molecule

`arms_touch` on the result file says which is which for every arm, because an
arm that rewrites the whole column is answering a different question from one
that rewrites the flagged share, and the two are in the same table.

WHAT IS NOT SWEPT HERE, AND WHY. The strength each rewriting and reweighting arm
leans on the suspicion score is left at the value `cleaned_training_set` uses
when no strength is passed: alpha 0.5 for label_smooth and oof_smooth, 1.0 for
downweight, 1.345 for huber_weight. n18 sweeps exactly that axis over
(0.25, 0.5, 1.0) and (1.0, 3.0, 9.0) on the same cleanups and the same call, and
this file's axis is what is DONE with the flagged set rather than how hard it is
done. `strength_per_arm` records the value every arm ran at.

FOUR ARMS PRODUCE THE SAME CLEANED TRAINING SET AT EVERY FLAG RATE. On a
regression label `downweight`, `huber_weight`, `label_smooth` and `oof_smooth`
read no quantile of the suspicion score at all, so sweeping the rate for them
sweeps nothing. They are FITTED ONCE, at the first rate, and the same per-pass
score is written under the other rates so that every rate's table holds every
arm. `arms_that_read_the_flag_rate` names the two that do and
`rate_blind_rows_are_repeats` records that the others repeat.

THE HEADLINE IS A COUNT OF PASSES, NOT A MEAN. One pass is one (noise draw,
fold). Every comparison is paired: the same pass, the same held-out molecules,
two arms. A mean over passes hides an arm that wins large on one and loses on
the rest.

WHAT THIS CANNOT SAY. Nothing here is a claim about real label error. The noise
is injected, so its shape is known by construction and every number is confined
to the condition named on the file. n18 asks the same three-way question on the
labels as they come, where there is no list of damaged molecules to be right
about.

THE FEATURE SPACES ARE PER MODEL AND PER OUTER FOLD (2026-10-05, chat 5;
HANDOFF_EXPERIMENT_ALIGNMENT section 2.4). The models are the dataset's two
decided models (DECISIONS.md Step 1). For each outer fold, searches on that
fold's training rows (noise_helpers.fold_configurations) give each model its
2 best single representations (base_1, base_2) and its 2 best general hybrids
(hybrid, hybrid__general2): 4 configurations per model, 8 for the two (R5).
Which ones, per fold, is written under `folds`. Until 2026-10-05 the recipes
and the single representations came from the whole-dataset stored search,
which had seen every held-out molecule, and five models were fitted.

The detectors and the out-of-fold and Gaussian-process targets are fitted
inside a feature matrix, once per (configuration, model); the detection and
deletion records are keyed '<configuration>/<model>'.

WHAT THE HELD-OUT MOLECULES ARE GRADED AGAINST. `labels[val_idx]`, untouched, in
every arm. No cleaned label and no corrupted label is ever scored against.

SIMILARITY AXIS, ROWS, CACHES, CHECKPOINT (2026-09-25, plan sections 1-3;
neighbour settings re-aligned 2026-10-05 with KNN_SEARCH_SPACE_RECORD).
Every neighbour the experiment reads comes from the decided neighbour
settings of the dataset's group (n07.sweep_settings: spaces and distances per
group, no importance weighting, k 5, 10 and 20, counting schemes rank and
inverse_square). Since 2026-10-05 nothing reads neighbours in the
configuration's own features: the kNN record cut that space (axis 1, number
10), k=9 and k=10 (axis 4) and equal counting (axis 5). The settings are
screened with no model fit (n07.screen_settings), and only the top 3 re-run
neighbor_residual in that setting (every arm, the random-deletion control
included) and, under cross_val, the two cleanups that read neighbours. Those
refits go into the main table under '<detector>@screen<rank>'; which setting
held each rank on each pass is in `screen`. The main table's plain detector
names (cross_val) hold only the arms that read no neighbours. Detection is
graded under every setting, since that fits no model. The neighbour spaces include the fold's distance hybrid and similarity
hybrid; scaffold-mates stay in every neighbour list. Every arm writes one
D-versus-D' row per pass beside the result file; the accumulators are
checkpointed per pass; the hybrid builds, the scaled representations, the
detector scores, the correction targets and the no-cleaning and clean-label
fits of seeded models go through noise_cache. controls.py's twins run with
KIRBY_NOISE_CONTROLS=1; n24's own random deletion always runs.

Usage:
    python n24_delete_or_keep.py --dataset flab_thermo_tresanco --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import copy
import re
import time
import zlib

import numpy as np

# 2026-10-05 (chat 5): the whole-dataset readers (get_best_search_for_downstream,
# noise_base_reps_by_model, pool_recipe_per_downstream_model,
# pool_recipes_by_family), cv_split, DOWNSTREAM_MODELS and StandardScaler are no
# longer imported. Every configuration comes from searches on the outer fold's
# training rows (noise_helpers.fold_configurations), and the hybrids are
# rebuilt and scaled by noise_helpers.fold_model_inputs.
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_domain_metric, get_model_factories, input_fingerprint,
)
from experiments.decided_grid import decided_models
from experiments.dataset_policy import noise_groups_for, noise_dose_units
from experiments.noise.noise_helpers import (
    detection_scores, detection_is_defined,
    sweep_conditions, noise_outer_folds, fold_spread, fold_configurations,
    fold_model_inputs, configuration_record, inject_condition,
    decided_conditions, decided_levels, dataset_knn_group, level_unit,
    CONFIGURATION_SPACES, draws_for,
)
# The cleanups are n07's, called rather than copied, so an arm here is the same
# code n07 and n18 already measured. `_compute_noise_scores` is n07's detector
# call: it offers the scaffold groups under BOTH names the estimator families
# use and refuses a detector that can take neither, instead of dropping them.
# The neighbour settings, their screen and the neighbour spaces are n07's too
# (2026-10-05, chat 5): sweep_settings, screen_settings, fold_similarity_spaces,
# fold_neighbours.
from experiments.noise.n07_remediation import (
    cleaned_training_set, targets_needed, _predict_for_metric,
    REGRESSION_ONLY_CLEANUPS,
    sweep_settings, sweep_ks, sweep_schemes, screen_settings, screen_top,
    usable_ks, fold_similarity_spaces, fold_neighbours,
)
from kirby.noise_spec import (
    GROUPED_STRATEGIES, OUTLIER_STRATEGIES, CLASSIFICATION_STRATEGIES,
    DEFAULT_STRATEGY_CLF,
)
# The D-versus-D' rows, the per-pass checkpoint, the caches and the controls
# (2026-09-25, plan sections 1-3). They live in n23 for the reason given at the
# top of its shared block.
from experiments.noise import noise_cache as nc
from experiments.noise.controls import controls_enabled
from experiments.noise.n23_uncertainty_cleaning import (
    RowSink, PassCheckpoint, fit_score_cached, control_twins,
    cleanup_reads_setting, correction_targets_cached,
    NO_NEIGHBOURS, TODAY_NEIGHBOUR_CLEANUPS,
)

# ── The arms ───────────────────────────────────────────────────────────────
# One cleanup name per arm, all of them n07's. The three groups are the answer
# the experiment is choosing between.
DELETE_CLEANUP = 'remove'
REWRITE_CLEANUPS = ('label_smooth', 'knn_smooth', 'oof_smooth')
REWEIGHT_CLEANUPS = ('downweight', 'huber_weight')
# Not run on a class label: it fits the same training set as label_smooth there
# (DECIDED 2026-10-06, Adelaide delegated; see cleanups_for).
CLASSIFICATION_SKIPPED_CLEANUPS = ('knn_smooth',)

# What each arm actually touches, written onto the result file. See the
# docstring: only two of the seven cleanups touch the flagged molecules alone.
TOUCHES = {
    'remove': 'the flagged molecules only',
    'knn_smooth': 'the flagged molecules only',
    'label_smooth': 'every training molecule, in proportion to its score',
    'oof_smooth': 'every training molecule, in proportion to its score',
    'downweight': 'every training molecule, in proportion to its score',
    'huber_weight': 'every training molecule outside the score scale',
}

# The strength each arm leans on the suspicion score with. None asks
# cleaned_training_set for its own default, and these are those defaults, read
# out of n07_remediation.py this session so the result file can state them
# without a second implementation deciding them.
STRENGTH_DEFAULTS = {
    'remove': None,
    'knn_smooth': None,
    'label_smooth': 0.5,
    'oof_smooth': 0.5,
    'downweight': 1.0,
    'huber_weight': 1.345,
}

# How many molecules in every hundred are treated as suspect. The share the
# condition damages is 1, 5 or 10 in a hundred, so the grid spans it rather than
# sitting under it -- which is where n18's (0.01, 0.05, 0.20) sits, for its own
# stated reason about real labels.
FLAG_RATES = (0.05, 0.10, 0.20)

# Which per-molecule suspicion score decides what is flagged. n18's two, for
# n18's reason: one is the molecule's label judged against its neighbours'
# labels, the other is how badly a model trained on the OTHER molecules predicts
# it, out of fold on scaffold-disjoint folds. Two kinds of evidence, not one
# kind twice.
DETECTORS = ('neighbor_residual', 'cross_val')

# How many independent random-deletion replicates the detector-driven deletion
# has to beat. FIVE and not three: the three measured on 2026-09-23 already held
# one that won every fold and one that lost most of them, so three seeds is a
# spread read off two points. Every seed is reported on its own and no mean
# across them is written anywhere.
RANDOM_DELETION_SEEDS = (0, 1, 2, 3, 4)

# The conditions n24 runs. It needs a damaged SUBSET, which is what makes a
# flagged set and a detection score mean anything -- under gaussian and
# grouped_shifted every training label moves and the list of damaged molecules
# is the list of all of them.
#
# 2026-10-05 (chat 5): outlier_p10 until then. The noise grid decision
# (KNN_SEARCH_SPACE_RECORD, "DECISION, noise grid", Adelaide 2026-10-05) cut
# outlier_p10 and kept six kinds; four of them damage a subset:
# outlier_p05, censoring, shift_only_p10, sparse_dprime_up_p10. A
# classification dataset keeps its flip kinds (kirby.noise_spec.
# CLASSIFICATION_STRATEGIES); every flip kind flips a subset.
#
# Dirty-run bug, n24 (DIRTY_RUN_RESULTS.md, COST_BREAKDOWN.md section 3 item
# 4): outlier_p10 ran once as the default, written to <dataset>.json, and once
# as a named condition, written to <dataset>__outlier_p10.json, so neither
# skipped the other. run() now goes through noise_helpers.sweep_conditions, so
# a run with no condition and a run naming one both write
# <dataset>__<condition>.json and the second skips the first.
#
# DECIDED 2026-10-06 (Adelaide delegated): a run with no condition named runs
# every kind subset_conditions returns. DEFAULT_NOISE_CONDITION is now only the
# condition a --tag run uses when none is named.
DEFAULT_NOISE_CONDITION = 'outlier_p05'

# The replicate index of the injection. n07, n11 and n13 sweep the same three.
# 3 draws per non-zero level; level 0 is the labels as delivered and runs once.
NOISE_DRAWS = (0, 1, 2)

# What this file writes since chat 5 (2026-10-05). It goes into the rows file
# name and the checkpoint key, so nothing written by the whole-dataset design
# before that date is read back or blocks a new row.
# Bumped the same day when the neighbour arms moved off the configuration's
# own features and into the screened settings ('@screen<rank>' in the main
# table), so partial files of the first layout are not read back.
# Bumped 2026-10-06: the screen predicts a held-back fifth of the training
# molecules, class labels drop knn_smooth, censoring runs one draw.
RESULT_LAYOUT = 'fold_aligned_screened_inner_2026-10-06'

# The four configurations each decided model is FITTED on, per outer fold
# (noise_helpers.CONFIGURATION_SPACES): its 2 best single representations and
# its 2 best general hybrids, all chosen by searches on the fold's training
# rows. 2 models x 4 = 8 (R5). The distance hybrid ('hybrid__distance1') and
# the similarity hybrid ('hybrid_similarity') are neighbour spaces only.
BASE_SPACES = ('base_1', 'base_2')


def subset_conditions(data):
    """The decided kinds n24 can run on this dataset: those where only a subset
    of training labels is damaged. Regression: the decided kinds that
    noise_helpers.detection_is_defined accepts. Classification: the flip kinds
    (noise_helpers.decided_conditions)."""
    if data.get('task_type') == 'classification':
        return list(decided_conditions(data))
    return [c for c in decided_conditions(data) if detection_is_defined(c)]


def _detection(cmask, scores, condition, is_clf):
    """noise_helpers.detection_scores under this run's condition.

    Until 2026-10-05 n24 ran classification sets under the regression name
    outlier_p10, which noise_spec_for turned into a uniform flip, and graded
    them under that name (dirty-run bug: humanness rows labelled outlier_p10
    although the noise was a uniform flip). noise_helpers.detection_is_defined
    now accepts the flip kinds, so a flip kind is graded under its own name
    and the record carries that name in detection_condition.
    """
    # DECIDED 2026-10-06 (Adelaide delegated): the shared
    # noise_helpers.detection_is_defined accepts CLASSIFICATION_STRATEGIES, so
    # the stand-in regression name this function used to pass is gone.
    if detection_is_defined(condition):
        return detection_scores(cmask, scores, condition)
    return None


def _fold_of(pass_key):
    """The outer fold of a pass key 'l<level>d<draw>f<fold>'."""
    m = re.search(r'f(\d+)$', str(pass_key))
    return int(m.group(1)) if m else None


def _by_fold(per_pass):
    """{pass key: score} -> noise_helpers.fold_spread over the per-fold means
    (a fold's number is the mean over its draws). No significance test."""
    by = {}
    for k, v in (per_pass or {}).items():
        if v is None or not np.isfinite(v):
            continue
        by.setdefault(_fold_of(k), []).append(float(v))
    return fold_spread([float(np.mean(v)) for _, v in sorted(by.items())])


def reads_flag_rate(cleanup, is_clf):
    """Does this cleanup's cleaned training set depend on the flag rate?

    Every answer below is what the matching branch of
    `n07.cleaned_training_set` does, read in this session:

      remove        thresholds at the (1 - rate) quantile of the suspicion
                    score and drops what is above it   (n07:561-567)  -- reads it
      knn_smooth    thresholds at the same quantile and overwrites what is
                    above it                           (n07:678-694)  -- reads it
      label_smooth  on a CLASSIFICATION label it thresholds; on a regression
                    label it calls smooth_by_scores over every row
                                                       (n07:654-676)  -- clf only
      oof_smooth    smooths every row in proportion to its score, and is
                    refused outright on a class label  (n07:755-772)  -- no
      downweight    weights_from_scores, which takes no threshold
                                                       (n07:569-574)  -- no
      huber_weight  a weight from the median absolute score, no quantile
                                                       (n07:576-601)  -- no

    An unknown cleanup is assumed to read it: scoring it at every rate wastes
    fits, scoring it once and calling that the whole sweep is wrong.
    """
    if cleanup in ('remove', 'knn_smooth'):
        return True
    if cleanup in ('downweight', 'huber_weight', 'oof_smooth'):
        return False
    if cleanup == 'label_smooth':
        return bool(is_clf)
    return True


def cleanups_for(is_clf):
    """(delete, rewrite, reweight) -- the cleanups this task type can run.

    The regression-only ones are dropped on a classification label rather than
    writing rows of NaN; `cleaned_training_set` raises on them there and n07 and
    n18 gate them the same way.

    DECIDED 2026-10-06 (Adelaide delegated): on a classification label
    knn_smooth is not run. label_smooth and knn_smooth there replace the same
    flagged share with the majority class of the same neighbours
    (n07.cleaned_training_set), so they fit the same training set; label_smooth
    runs and knn_smooth does not (dirty-run bug 13).
    """
    rewrite = tuple(c for c in REWRITE_CLEANUPS
                    if not (is_clf and c in REGRESSION_ONLY_CLEANUPS)
                    and not (is_clf and c in CLASSIFICATION_SKIPPED_CLEANUPS))
    reweight = tuple(c for c in REWEIGHT_CLEANUPS
                     if not (is_clf and c in REGRESSION_ONLY_CLEANUPS))
    return DELETE_CLEANUP, rewrite, reweight


def build_arms(is_clf, seeds):
    """Every arm, as (arm name, family, cleanup, random seed).

    'no_cleaning' is the score every other arm is paired against.
    'delete_at_random/seed<k>' is the control the detector-driven deletion has
    to beat, and it is a `remove` over a suspicion score that carries no
    information at all.
    """
    delete, rewrite, reweight = cleanups_for(is_clf)
    out = [('no_cleaning', 'no_cleaning', None, None),
           (f'delete/{delete}', 'delete', delete, None)]
    out += [(f'keep_rewrite/{c}', 'keep_rewrite', c, None) for c in rewrite]
    out += [(f'keep_reweight/{c}', 'keep_reweight', c, None) for c in reweight]
    out += [(f'delete_at_random/seed{s}', 'delete_at_random', delete, int(s))
            for s in seeds]
    return out


def random_deletion_scores(n_training, n_to_delete, seed):
    """A suspicion score that flags `n_to_delete` molecules chosen at random.

    Returns (scores, rate) to hand straight to `cleaned_training_set('remove',
    ...)`, so the control is deleted by the same code as every other deletion
    and no second removal rule exists in this file.

    HOW THE COUNT IS MADE EXACT. The score is 1.0 on the chosen molecules and
    0.0 on the rest, and the rate is the chosen count over the training count.
    `remove` cuts at `np.percentile(scores, 100 * (1 - rate))`, which for this
    two-valued column interpolates to the chosen share itself, strictly above
    0.0 and no higher than 1.0, so exactly the chosen molecules are dropped.
    Verified over every (training count, deleted count) pair the smoke tries,
    and the count the arm actually deleted is recorded on every row either way.
    """
    rng = np.random.default_rng(int(seed))
    scores = np.zeros(int(n_training), dtype=float)
    scores[rng.choice(int(n_training), int(n_to_delete), replace=False)] = 1.0
    return scores, float(n_to_delete) / float(n_training)


def replicate_seed(seed, level, draw, fold_i, n_training, n_to_delete):
    """The draw for one random-deletion replicate on one pass.

    A SEED IS A REPLICATE OF THE WHOLE PROCEDURE, not one fixed set of molecules
    reused in every fold: each pass draws its own molecules, so a seed that wins
    every fold has won every fold on a different random set. The space is NOT in
    here on purpose -- where the same number of molecules is deleted in the base
    representation and in a hybrid, the same molecules are deleted in both, so
    the feature space is the only thing that differs between those two rows.

    zlib.crc32 and not hash(): Python randomises string hashing per process, so
    hash() would draw different molecules on every run of the same command.
    """
    key = f'{seed}|{level}|{draw}|{fold_i}|{n_training}|{n_to_delete}'
    return int(zlib.crc32(key.encode('utf-8')) & 0x7FFFFFFF)


def _fit_score(factory, cleaned, X_va, y_val, metric_fn, metric_name):
    """One fit on a cleaned training set, one held-out score, or (None, why).

    `sample_weight` is passed only when the cleanup produced one, which is what
    n07's `_evaluate_fold` does with the same dict.
    """
    try:
        m = factory()
        if cleaned.get('sample_weight') is None:
            m.fit(cleaned['X'], cleaned['y'])
        else:
            m.fit(cleaned['X'], cleaned['y'],
                  sample_weight=cleaned['sample_weight'])
        return float(metric_fn(
            y_val, _predict_for_metric(m, X_va, metric_name))), None
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'


def _deleted_count(cleaned, n_training):
    """How many training molecules this arm deleted. 0 when it kept them all."""
    keep = cleaned.get('keep')
    if keep is None:
        return 0
    return int(n_training - int(np.asarray(keep).sum()))


def _levels_and_draws(data, dataset_name, condition, levels=None, draws=None):
    """(level, draw) pairs to inject.

    Levels: noise_helpers.decided_levels (the noise grid decision, by dataset
    group; classification keeps its flip-rate grid), unless the caller names
    its own. Level 0 is the labels as delivered (noise_helpers.
    inject_condition), so it runs once, on classification too: 2026-10-05
    (chat 5), dirty-run bug 6 is closed in noise_spec and inject_condition,
    and a flip rate of 0 now flips nothing.

    DECIDED 2026-10-06 (Adelaide delegated): censoring runs one draw
    (noise_helpers.draws_for). It clips at a fixed share and reads no seed, so
    its draws give the same labels.
    """
    levels = (decided_levels(data, dataset_name, condition)
              if levels is None else levels)
    draws = draws_for(condition, NOISE_DRAWS if draws is None else draws)
    out = []
    for lvl in levels:
        here = draws if float(lvl) > 0 else draws[:1]
        for d in here:
            out.append((float(lvl), int(d)))
    return out


def _injury_carried_by_flagged(y_clean_train, y_noisy_train, mask):
    """How much of the injected damage sits on the molecules the injector picked.

    Returned as counts and sums, never as a bare ratio: the squared distance
    from the corrupted label to the clean one, summed over the molecules the
    injector gave the wider error, and summed over every training molecule.

    Under outlier_p01/p05/p10 the second is larger than the first because every
    other label moved too, and the gap is exactly what a deletion arm cannot
    remove. Under outlier_only_p01/p05/p10 the two are equal.
    """
    err2 = (np.asarray(y_noisy_train, dtype=float)
            - np.asarray(y_clean_train, dtype=float)) ** 2
    mask = np.asarray(mask, dtype=bool)
    total = float(err2.sum())
    flagged = float(err2[mask].sum())
    return {
        'squared_error_on_the_widened_molecules': flagged,
        'squared_error_over_every_training_molecule': total,
        'widened_molecules': int(mask.sum()),
        'training_molecules': int(mask.size),
        'share_of_the_squared_error_on_the_widened_molecules': (
            flagged / total if total > 0 else float('nan')),
    }


def _sim_tag(sim):
    """The error-key prefix of one neighbour setting, '' for today's path."""
    return '' if sim is None else f'{sim[2].name}|k={sim[1]}|{sim[3]}|'


def _cleaned(cleanup, X_tr_s, y0, scores, rate, is_clf, g_tr, targets, sim):
    """n07.cleaned_training_set for one arm. `sim` None is today's call,
    unchanged. `sim` = (neighbour lists, k, setting, counting scheme) hands the
    lists to a cleanup that reads them (cleanup_reads_setting) and None to one
    that does not, which n07 would refuse (n07:722-733). The counting scheme
    (KNN_SEARCH_SPACE_RECORD axis 5, 'rank' or 'inverse_square') decides how
    much each neighbour's label counts (2026-10-05, chat 5)."""
    if sim is not None and cleanup_reads_setting(cleanup, sim[2]):
        return cleaned_training_set(
            cleanup, X_tr_s, y0, scores, rate, is_clf=is_clf, groups=g_tr,
            strength=None, targets=targets, similarity_neighbours=sim[0],
            similarity_k=sim[1], similarity_scheme=sim[3])
    return cleaned_training_set(cleanup, X_tr_s, y0, scores, rate,
                                is_clf=is_clf, groups=g_tr, strength=None,
                                targets=targets)


def _arm_scores(det_cell, detector, scores, sim, only, X_tr_s, X_va_s, y0,
                y_val, g_tr, cell_models, targets, base_score, random_cache,
                lvl_key, level, draw, fold_i, ckey, pass_key, n_tr, cmask,
                arms, flag_rates, is_clf, model_factories, metric_fn,
                metric_name, errors, deleted_acc):
    """Every arm of one detector on one matrix, into det_cell[rate][arm][model].

    Moved out of run() on 2026-09-25 so today's path and the similarity sweep
    run the SAME code; with sim=None and only=None it is the loop that stood
    here, line for line. `only` names the cleanups to run (the sweep, for a
    detector that reads no neighbours, runs only the cleanups that do).
    `deleted_acc` None skips the deletion counts (the sweep writes rows, not
    those counts). Error keys carry the similarity label when sim is given.
    """
    tag = _sim_tag(sim)
    # Per-pass scores for the arms that are fitted once because their cleaned
    # training set does not depend on the rate.
    blind = {}
    for slot, rate in enumerate(flag_rates):
        rate_key = f'{rate:g}'
        rate_cell = det_cell.setdefault(rate_key, {})
        for mname, s in base_score.items():
            (rate_cell.setdefault('no_cleaning', {})
             .setdefault(mname, {})[pass_key]) = s

        n_deleted_here = None
        for arm, family, cleanup, seed in arms:
            if family in ('no_cleaning', 'delete_at_random'):
                continue
            if only is not None and cleanup not in only:
                continue
            blind_key = (detector, cleanup)
            if (not reads_flag_rate(cleanup, is_clf)
                    and blind_key in blind):
                # Already fitted at the first rate; the cleaned training set
                # is the same column, so the same per-pass score is written
                # here rather than the same fit being run again.
                for mname, done in blind[blind_key].items():
                    (rate_cell.setdefault(arm, {})
                     .setdefault(mname, {})[pass_key]) = done
                continue

            try:
                cleaned = _cleaned(cleanup, X_tr_s, y0, scores, rate, is_clf,
                                   g_tr, targets, sim)
            except Exception as exc:
                errors[f'{tag}{lvl_key}|{ckey}|{detector}|'
                       f'{rate_key}|{arm}|f{fold_i}|clean'] = (
                    f'{type(exc).__name__}: {exc}')
                continue
            if cleaned is None:
                errors[f'{tag}{lvl_key}|{ckey}|{detector}|'
                       f'{rate_key}|{arm}|f{fold_i}|clean'] = (
                    'the cleanup declined on this pass')
                continue

            if family == 'delete':
                n_deleted_here = _deleted_count(cleaned, n_tr)
                if deleted_acc is not None:
                    rm = ~np.asarray(cleaned['keep'], dtype=bool)
                    (deleted_acc.setdefault(lvl_key, {})
                     .setdefault(ckey, {})
                     .setdefault(detector, {})
                     .setdefault(rate_key, []).append({
                         'molecules_deleted': int(rm.sum()),
                         'training_molecules': n_tr,
                         'widened_molecules_among_the_deleted':
                             int(np.sum(np.asarray(cmask, dtype=bool) & rm)),
                         'widened_molecules': int(np.sum(cmask))}))

            got = {}
            for mname in cell_models:
                s, err = _fit_score(
                    model_factories[mname], cleaned, X_va_s,
                    y_val, metric_fn, metric_name)
                if s is None:
                    errors[f'{tag}{lvl_key}|{ckey}|{detector}|'
                           f'{rate_key}|{arm}|{mname}|f{fold_i}'] = err
                    continue
                got[mname] = s
                (rate_cell.setdefault(arm, {})
                 .setdefault(mname, {})[pass_key]) = s
            if not reads_flag_rate(cleanup, is_clf):
                blind[blind_key] = got

        # ── The control, on the same count ──
        if only is not None and DELETE_CLEANUP not in only:
            continue
        if not n_deleted_here:
            errors[f'{tag}{lvl_key}|{ckey}|{detector}|{rate_key}|'
                   f'f{fold_i}|control'] = (
                'the deletion arm removed no molecule on this '
                'pass, so there is no count to match at random')
            continue
        for arm, family, cleanup, seed in arms:
            if family != 'delete_at_random':
                continue
            # The random deletion does not depend on the similarity setting,
            # only on the count, so one cache serves every setting.
            ck = (ckey, tuple(cell_models), n_deleted_here, seed)
            if ck not in random_cache:
                rs, rrate = random_deletion_scores(
                    n_tr, n_deleted_here,
                    replicate_seed(seed, level, draw, fold_i,
                                   n_tr, n_deleted_here))
                rcleaned = cleaned_training_set(
                    cleanup, X_tr_s, y0, rs, rrate,
                    is_clf=is_clf, groups=g_tr,
                    strength=None, targets=targets)
                if rcleaned is None:
                    errors[f'{tag}{lvl_key}|{ckey}|{rate_key}|'
                           f'{arm}|f{fold_i}|clean'] = (
                        'the deletion declined on this pass')
                    random_cache[ck] = {}
                else:
                    if deleted_acc is not None:
                        rm = ~np.asarray(rcleaned['keep'], dtype=bool)
                        (deleted_acc.setdefault(lvl_key, {})
                         .setdefault(ckey, {})
                         .setdefault(f'random_seed{seed}', {})
                         .setdefault(rate_key, []).append({
                             'molecules_deleted': int(rm.sum()),
                             'training_molecules': n_tr,
                             'widened_molecules_among_the_deleted':
                                 int(np.sum(np.asarray(
                                     cmask, dtype=bool) & rm)),
                             'widened_molecules': int(
                                 np.sum(cmask))}))
                    out = {}
                    for mname in cell_models:
                        s, err = _fit_score(
                            model_factories[mname], rcleaned,
                            X_va_s, y_val, metric_fn,
                            metric_name)
                        if s is None:
                            errors[
                                f'{tag}{lvl_key}|{ckey}|{rate_key}'
                                f'|{arm}|{mname}|f{fold_i}'] = err
                        else:
                            out[mname] = s
                    random_cache[ck] = out
            for mname, s in random_cache[ck].items():
                (rate_cell.setdefault(arm, {})
                 .setdefault(mname, {})[pass_key]) = s


def _rows_for_rate(rows, rate_cell, arms, detector, rate, label_fn, space,
                   models, base_score, clean_score, condition, level, draw,
                   fold_i, pass_key, prefix_fn=None):
    """The D-versus-D' rows of one (detector, flag rate): one per (arm,
    model) that scored on this pass. score_D is the no-cleaning fit on the
    same matrix. A rate-blind arm's repeated score is written under every
    rate, as the result file writes it (`rate_blind_rows_are_repeats`).

    prefix_fn(cleanup), 2026-10-05 (chat 5): what goes before the detector in
    method_setting. The neighbour-setting rows put 'k=<count>|scheme=<scheme>|'
    there (only 'k=<count>|' for an arm that reads no neighbours), as
    n07.similarity_sweep does; similarity_setting is the setting's name."""
    for arm, family, cleanup, seed in arms:
        if family == 'no_cleaning':
            continue
        for m in models:
            sP = (rate_cell.get(arm) or {}).get(m, {}).get(pass_key)
            if sP is None:
                continue
            setting = ((prefix_fn(cleanup) if prefix_fn else '')
                       + f'{detector}|flag_rate={rate:g}')
            if family == 'delete_at_random':
                setting += f'|seed={seed}'
            rows.add(method=(cleanup if family != 'delete_at_random'
                             else 'delete_at_random'),
                     method_setting=setting,
                     similarity_setting=label_fn(detector, cleanup),
                     model_input=space, model=m, noise_condition=condition,
                     level=float(level), draw=int(draw), fold=fold_i,
                     score_D=base_score.get(m), score_Dprime=sP,
                     score_clean=clean_score.get(m))


def _controls_for(rows, detector, scores, sim, only, X_tr_s, X_va_s, y0,
                  y_val, g_tr, models, targets, base_score, clean_score, arms,
                  flag_rates, is_clf, model_factories, metric_fn, metric_name,
                  label_fn, space, condition, level, draw, fold_i, errors,
                  lvl_key, ckey, prefix_fn=None):
    """The shared random and shuffled twins of every detector-driven arm,
    only when KIRBY_NOISE_CONTROLS=1 (controls.py). n24's own
    delete_at_random stays as it is and always runs: it is this file's design
    and draws its molecules with replicate_seed, which controls.random_twin
    does not reproduce. random_twin is given `rerun`, the arm's own
    cleaned_training_set call on a score that flags the chosen molecules --
    n24's random_deletion_scores rule, generalised by controls.py. A rate-blind
    arm is twinned once, at the first rate."""
    if not controls_enabled():
        return
    tag = _sim_tag(sim)
    for slot, rate in enumerate(flag_rates):
        for arm, family, cleanup, seed in arms:
            if family in ('no_cleaning', 'delete_at_random'):
                continue
            if only is not None and cleanup not in only:
                continue
            if slot > 0 and not reads_flag_rate(cleanup, is_clf):
                continue
            try:
                cleaned = _cleaned(cleanup, X_tr_s, y0, scores, rate, is_clf,
                                   g_tr, targets, sim)
            except Exception:
                continue
            if cleaned is None:
                continue

            def _rerun(sc, r, _c=cleanup):
                return _cleaned(_c, X_tr_s, y0, sc, r, is_clf, g_tr,
                                targets, sim)
            _seed = (int(draw), int(fold_i), int(zlib.crc32(
                f'{detector}|{cleanup}|{rate:g}|{space}|{tag}'.encode())
                & 0x7FFFFFFF))
            for cname, tw, why in control_twins(cleaned, X_tr_s, y0, _seed,
                                                is_clf=is_clf, rerun=_rerun):
                if tw is None:
                    errors[f'{tag}{lvl_key}|{ckey}|{detector}|{rate:g}|{arm}|'
                           f'{cname}|f{fold_i}'] = why
                    continue
                for m in models:
                    sP, err = _fit_score(model_factories[m], tw, X_va_s,
                                         y_val, metric_fn, metric_name)
                    rows.add(method=cleanup,
                             method_setting=(
                                 (prefix_fn(cleanup) if prefix_fn else '')
                                 + f'{detector}|flag_rate={rate:g}'
                                 f'|control={cname}'),
                             similarity_setting=label_fn(detector, cleanup),
                             model_input=space, model=m,
                             noise_condition=condition, level=float(level),
                             draw=int(draw), fold=fold_i,
                             score_D=base_score.get(m), score_Dprime=sP,
                             score_clean=clean_score.get(m))


def run(data, config, output_dir, dataset_name, **kwargs):
    """N24 under one or more injected conditions, each written as it lands.

    2026-10-05 (chat 5): through noise_helpers.sweep_conditions, as n07 and
    n23 do. A caller that names a condition (run_all with NOISE_CONDITION, or
    --noise-condition) runs that one; a caller that names none runs every
    decided kind that damages only some molecules (subset_conditions). Either
    way the file is <dataset>__<condition>.json, so the default run and the
    named run of the same condition are one file and the second skips
    (dirty-run bug: n24 ran outlier_p10 twice under two file names).
    """
    # DECIDED 2026-10-06 (Adelaide delegated): with no condition named, n24
    # runs every decided kind that damages only some molecules
    # (subset_conditions): outlier_p05, censoring, shift_only_p10 and
    # sparse_dprime_up_p10 on regression; the six flip kinds on
    # classification. Until this date it ran one (outlier_p05, or
    # DEFAULT_STRATEGY_CLF on class labels).
    default = subset_conditions(data)
    return sweep_conditions(
        _run_one_condition, 'n24_delete_or_keep', data, config, output_dir,
        dataset_name, default_conditions=default, kwargs=dict(kwargs),
        groups_needed=GROUPED_STRATEGIES)


def _run_one_condition(data, config, output_dir, dataset_name, **kwargs):
    """N24: delete the flagged molecules, rewrite them, or weigh them down.

    CHAT 5 ALIGNMENT, 2026-10-05 (HANDOFF_EXPERIMENT_ALIGNMENT section 2.4).
    What each outer fold does, in order:
      1. Folds are noise_helpers.noise_outer_folds: the per-fold searches' own
         splits, the count from config['n_splits'] or KIRBY_NOISE_N_FOLDS.
      2. noise_helpers.fold_configurations runs this fold's searches on its
         training rows (clean labels, decided grid, pruning off) and gives each
         of the dataset's two decided models its 2 best single
         representations, its 2 best general hybrids, and the recipes of the
         neighbour spaces (the distance hybrid and the similarity hybrid).
         Nothing is read from a whole-dataset search.
      3. At every (level, draw) each model is fitted on its 4 configurations
         (2 models x 4 = 8, R5). The hybrids are rebuilt on the fold's
         training rows from the damaged labels, as before
         (noise_helpers.fold_model_inputs).
      4. Every arm runs in every configuration, as before. The neighbour
         settings (KNN_SEARCH_SPACE_RECORD) are graded for detection with no
         fit, screened with no fit, and only the screen's top 3 (setting,
         neighbour count, counting scheme) are refitted. Scaffold-mates stay
         in every neighbour list (axis 6).
    Every (level, draw, fold) pass is checkpointed as it finishes.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT N24: DELETE THE FLAGGED MOLECULES, OR KEEP THEM")
    print("=" * 70)

    labels = np.asarray(data['labels'])
    groups = data.get('groups')
    groups_arr = None if groups is None else np.asarray(groups)
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    _metric = (metric_name, metric_fn, lower_is_better)

    condition = str(kwargs.get('noise_condition')
                    or (DEFAULT_STRATEGY_CLF if is_clf
                        else DEFAULT_NOISE_CONDITION))
    detectors = list(kwargs.get('detectors') or DETECTORS)
    flag_rates = [float(x) for x in (kwargs.get('flag_rates') or FLAG_RATES)]
    seeds = [int(s) for s in
             (kwargs.get('random_seeds') or RANDOM_DELETION_SEEDS)]

    # ── Which conditions n24 runs (noise grid decision, 2026-10-05) ──
    # A damaged SUBSET is needed: an arm that deletes the flagged molecules is
    # doing nothing distinguishable when every training molecule is damaged by
    # construction. A condition outside subset_conditions is written as a
    # skip with its reason, never run under another name.
    allowed = subset_conditions(data)
    if condition not in allowed:
        if is_clf and condition not in CLASSIFICATION_STRATEGIES:
            why = (f'{condition!r} is a regression kind and {dataset_name} has '
                   f'class labels. noise_spec_for would turn it into a '
                   f'uniform flip filed under {condition!r} (dirty-run bug: '
                   f'humanness rows labelled outlier_p10).')
        elif condition in decided_conditions(data):
            why = (f'{condition!r} moves every training label, so there is no '
                   f'damaged subset for a detector to find and no flagged '
                   f'set to delete or keep.')
        else:
            why = (f'{condition!r} is not one of the six kinds of damage kept '
                   f'by the noise grid decision (KNN_SEARCH_SPACE_RECORD, '
                   f'Adelaide 2026-10-05).')
        print(f"  SKIP n24 under {condition}: {why} n24 runs here under: "
              f"{allowed}")
        return {'skipped': True, 'reason': why, 'noise_condition': condition,
                'conditions_n24_runs_on_this_dataset': allowed,
                'result_layout': RESULT_LAYOUT}

    # ── R1: the dataset's two decided models (DECISIONS.md Step 1) ──
    # Replaces config['models'] / DOWNSTREAM_MODELS (five models) and the rf
    # and svm ensure_dprime_models added to every D-versus-D' row. knn runs
    # only inside the distance search. A caller's `models` can only narrow
    # the decided pair (a smoke); a name outside it is dropped and printed.
    decided = [m for m in decided_models(dataset_name, 'general') if m != 'knn']
    asked = kwargs.get('models')
    if asked:
        dropped = [m for m in asked if m not in decided]
        if dropped:
            print(f"  n24: models {dropped} are not this dataset's decided "
                  f"models {decided}; dropped")
        downstream_models = [m for m in decided if m in asked]
    else:
        downstream_models = list(decided)
    model_factories = get_model_factories(
        downstream_models, task_type=task_type, n_samples=len(labels))
    no_factory = [m for m in downstream_models if m not in model_factories]
    downstream_models = [m for m in downstream_models if m in model_factories]
    if not downstream_models:
        raise RuntimeError(
            f"n24: no downstream model could be built. Decided models "
            f"{decided}, asked {asked}, built {sorted(model_factories)}.")

    noise_groups = noise_groups_for(data)[0]
    folds, n_folds = noise_outer_folds(data, config)

    # ── The neighbour settings (KNN_SEARCH_SPACE_RECORD, 2026-10-05) ──
    # Group from the molecule count (more than 5000 is fast_knn), spaces and
    # distances per group, no model_input setting, no importance weighting.
    # k (5, 10, 20) and the counting scheme (rank, inverse_square) are swept
    # and screened per configuration and pass (n07.screen_settings).
    smiles = data.get('smiles')
    has_smiles = smiles is not None and len(smiles) > 0
    n_molecules = len(set(smiles)) if has_smiles else len(labels)
    knn_grp = dataset_knn_group(data, dataset_name)
    sim_settings = sweep_settings(dataset_name, n_molecules, has_smiles)

    delete_cleanup, rewrite_cleanups, reweight_cleanups = cleanups_for(is_clf)
    cleanups_used = (delete_cleanup,) + rewrite_cleanups + reweight_cleanups
    arms = build_arms(is_clf, seeds)
    arm_names = [a[0] for a in arms]
    rate_reading = [c for c in cleanups_used if reads_flag_rate(c, is_clf)]

    results = {
        'result_layout': RESULT_LAYOUT,
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'noise_condition': condition,
        'level_unit': level_unit(condition, is_clf),
        'dose_units': noise_dose_units(dataset_name),
        'noise_draws': list(draws_for(
            condition, NOISE_DRAWS if kwargs.get('draws') is None
            else kwargs['draws'])),
        'detectors': detectors,
        'flag_rates': flag_rates,
        'random_deletion_seeds': seeds,
        'models': downstream_models,
        'models_decided': decided,
        'models_without_factory': no_factory,
        'n_folds': int(n_folds),
        'folds_run': [int(i) for i, _, _ in folds],
        'selection_scope': 'fold_train_rows',
        'configuration_spaces': list(CONFIGURATION_SPACES),
        'configurations_note': (
            'Per outer fold and decided model: base_1 and base_2 are its 2 '
            'best single representations and hybrid and hybrid__general2 its 2 '
            "best general hybrids, all ranked by searches on that fold's "
            'training rows with clean labels (noise_helpers.'
            'fold_configurations). Which ones, per fold, is under folds. The '
            'distance hybrid and the similarity hybrid are neighbour spaces '
            'only; no model is fitted on them.'),
        'arms': arm_names,
        'arm_to_beat': 'no_cleaning',
        'arm_families': {a[0]: a[1] for a in arms},
        'arm_cleanup': {a[0]: a[2] for a in arms},
        'arms_touch': {a[0]: TOUCHES.get(a[2], 'nothing')
                       if a[1] != 'delete_at_random'
                       else 'molecules chosen at random, nothing else'
                       for a in arms},
        'strength_per_arm': {a[0]: STRENGTH_DEFAULTS.get(a[2])
                             for a in arms if a[2]},
        'strength_note': (
            'How hard each arm leans on the suspicion score is fixed at the '
            "value cleaned_training_set uses when none is passed, and this "
            "file's axis is what is DONE with the flagged set rather than how "
            'hard. n18 sweeps that same strength over (0.25, 0.5, 1.0) for '
            'label_smooth and oof_smooth and (1.0, 3.0, 9.0) for downweight, '
            'on the same call into the same function.'),
        'arms_that_read_the_flag_rate': list(rate_reading),
        'rate_blind_rows_are_repeats': (
            'An arm absent from arms_that_read_the_flag_rate produces the same '
            'cleaned training set at every flag rate, so it is FITTED ONCE at '
            f'flag rate {flag_rates[0]:g} and the same per-pass score is '
            'written under the other rates. Those rows are repeats, not '
            'separate measurements, and fitted_at_flag_rate_per_arm says which '
            'rate each arm was fitted at.'),
        'fitted_at_flag_rate_per_arm': {
            a[0]: ('every rate' if a[2] is None
                   or reads_flag_rate(a[2], is_clf) else flag_rates[0])
            for a in arms},
        'control_note': (
            'delete_at_random/seed<k> deletes the SAME NUMBER of training '
            'molecules as the detector-driven deletion it sits beside, chosen '
            'at random, with a fresh draw on every (draw, fold) pass. It is '
            'the bar: beating no_cleaning says a smaller training set fits '
            'better, and only beating every seed says the detector found '
            'something.'),
        'claim_scope': (
            'The noise is injected, so its shape is known by construction and '
            'every number here is confined to the named condition. Every score '
            'is against held-out labels that were never cleaned and never '
            'corrupted. n18 asks the same three-way question on the labels as '
            'they come.'),
        'flagged_share_note': (
            'Under outlier_p01/p05/p10 the selected share gets an error three times '
            'wider and every other label still moves, so deleting exactly the '
            'selected molecules leaves the rest of the damage in place. This '
            'run measures how much on its own labels and writes it to '
            'injury_carried_by_the_flagged_molecules.'),
        'condition_moves_every_other_label': bool(
            condition in OUTLIER_STRATEGIES),
        'features_are_fixed_note': (
            "Each feature matrix is built once per (draw, fold) from that "
            "pass's training molecules at their CORRUPTED labels, from the "
            "fold's own recipe, and every arm, detector and flag rate shares "
            'it. No cleanup reaches the feature ranking or the budget split; '
            "whether it should is n20's question."),
        'per_level': {},
        'detection': {},
        'injury_carried_by_the_flagged_molecules': {},
        'molecules_deleted': {},
    }
    if is_clf:
        # DECIDED 2026-10-06 (Adelaide delegated): label_smooth only on class
        # labels; knn_smooth fits the same training set there and is not run.
        results['cleanups_not_run_on_class_labels'] = {
            c: ('fits the same training set as label_smooth on class labels '
                '(the majority class of the same neighbours replaces the same '
                'flagged share)')
            for c in CLASSIFICATION_SKIPPED_CLEANUPS}

    passes = _levels_and_draws(data, dataset_name, condition,
                               levels=kwargs.get('levels'),
                               draws=kwargs.get('draws'))
    results['noise_levels'] = sorted({lvl for lvl, _ in passes})

    print(f"  {len(labels)} molecules, {len(folds)} of {n_folds} outer folds, "
          f"decided models {downstream_models}")
    print(f"  condition {condition}, levels {results['noise_levels']} "
          f"({results['level_unit']}), {len(passes)} (level, draw) passes")
    print(f"  detectors {detectors}, flag rates {flag_rates}")
    print(f"  {len(arm_names)} arms: 1 to beat, 1 deletion, "
          f"{len(rewrite_cleanups)} that rewrite labels, "
          f"{len(reweight_cleanups)} that reduce weights, "
          f"{len(seeds)} random-deletion replicates")
    print(f"  configurations per model {list(CONFIGURATION_SPACES)}; "
          f"neighbour settings ({knn_grp}): {len(sim_settings)} x ks "
          f"{list(sweep_ks())} x schemes {list(sweep_schemes())}, top "
          f"{screen_top()} refitted")

    # acc[level][space][detector][flag rate][arm][model][pass] = score
    acc = {}
    errors = {}
    detection_acc = {}
    injury_acc = {}
    deleted_acc = {}
    # sim_detection[level][matrix][model][detector][setting|k] = [readings]
    sim_detection = {}
    sim_unavailable = {}
    fold_records = {}
    screen_acc = []
    provenance = {}
    t0 = time.time()

    # ── Rows, checkpoint ──
    # A new rows file name (RESULT_LAYOUT): rows written by the whole-dataset
    # design carry the same model_input names ('base_1', 'hybrid') and would
    # be skipped as already on disk.
    rows = RowSink(output_dir, 'n24_delete_or_keep', dataset_name,
                   experiment='n24', metric=_metric,
                   condition=f'{condition}__{RESULT_LAYOUT}', data=data,
                   config=config)
    ckpt = PassCheckpoint(
        output_dir, 'n24_delete_or_keep', dataset_name,
        condition=condition, data=data, config=config,
        source_file=__file__,
        key={'layout': RESULT_LAYOUT, 'condition': condition,
             'detectors': detectors, 'flag_rates': flag_rates,
             'seeds': seeds, 'models': list(downstream_models),
             'levels': kwargs.get('levels'), 'draws': kwargs.get('draws'),
             'n_folds': int(n_folds),
             'settings': [s.name for s in sim_settings],
             'screen_top': screen_top()})
    done_passes = set()
    _state = ckpt.load()
    if _state:
        acc, errors = _state['acc'], _state['errors']
        detection_acc, injury_acc = _state['detection_acc'], _state['injury_acc']
        deleted_acc = _state['deleted_acc']
        sim_detection = _state['sim_detection']
        sim_unavailable = _state['sim_unavailable']
        fold_records = _state.get('fold_records', {})
        screen_acc = _state.get('screen', [])
        provenance = _state.get('provenance', {})
        done_passes = set(_state['done'])
        print(f"  resumed from {ckpt.path}: {len(done_passes)} (level, draw, "
              f"fold) passes already done")

    def _save_state():
        ckpt.save({'acc': acc, 'errors': errors,
                   'detection_acc': detection_acc, 'injury_acc': injury_acc,
                   'deleted_acc': deleted_acc, 'sim_detection': sim_detection,
                   'sim_unavailable': sim_unavailable,
                   'fold_records': fold_records, 'screen': screen_acc,
                   'provenance': provenance, 'done': sorted(done_passes)})

    def _sim_label(detector, cleanup):
        """NO_NEIGHBOURS: since 2026-10-05 the main path runs only the
        detectors and the cleanups that read no neighbours (main_only), so
        no row it writes read any. The neighbour rows carry their setting's
        name (_neighbour_settings_pass)."""
        return NO_NEIGHBOURS

    # The cleanups the main path runs: every arm's cleanup except the ones
    # that read neighbours (label_smooth, knn_smooth). Those run only in the
    # screened neighbour settings. 'remove' stays in, so the random-deletion
    # control runs beside it.
    main_only = tuple(sorted({a[2] for a in arms if a[2] is not None
                              and a[2] not in TODAY_NEIGHBOUR_CLEANUPS}))
    main_detectors = [d for d in detectors
                      if d not in nc.NEIGHBOUR_DETECTORS]

    for fold_i, train_idx, val_idx in folds:
        fold_keys = [f'l{lvl:g}d{draw}f{fold_i}' for lvl, draw in passes]
        if all(k in done_passes for k in fold_keys):
            print(f"\n  fold {fold_i}: every pass on disk in the checkpoint")
            continue
        print(f"\n  === fold {fold_i} of {n_folds}: {len(train_idx)} training, "
              f"{len(val_idx)} held-out rows ===")
        # The per-fold searches. One call per fold, shared by every pass; the
        # searches are stored per fold (ocv{n}f{i}s{seed}) and shared with
        # every other experiment on this dataset.
        try:
            entries = fold_configurations(data, dataset_name, fold_i=fold_i,
                                          train_idx=train_idx, n_folds=n_folds)
        except Exception as exc:
            print(f"    per-fold searches failed: {type(exc).__name__}: {exc}")
            fold_records[str(fold_i)] = {
                'error': (f'per-fold searches failed: '
                          f'{type(exc).__name__}: {exc}')}
            _save_state()
            continue
        # A fold whose searches failed on an earlier attempt left
        # {'error': ...} in the checkpoint and none of its passes done. Now
        # that the searches ran, that record is replaced, not reused: it has
        # no 'models' and frec['models'] would raise on every resume.
        frec = fold_records.get(str(fold_i))
        if not isinstance(frec, dict) or 'models' not in frec:
            frec = {'n_train': int(len(train_idx)),
                    'n_val': int(len(val_idx)), 'models': {}}
            fold_records[str(fold_i)] = frec
        for m in downstream_models:
            rec = configuration_record(entries.get(m))
            rec['metric'] = metric_name
            rec['lower_is_better'] = bool(lower_is_better)
            frec['models'].setdefault(m, rec)
            print(f"    {m}: base {rec.get('base')}, hybrids "
                  f"{[h.get('label') for h in rec.get('hybrids') or []]}")

        y_val = labels[val_idx]
        y_clean_train = np.asarray(labels[train_idx], dtype=float)
        g_tr = None if groups_arr is None else groups_arr[train_idx]
        smi_tr = (None if not has_smiles
                  else [smiles[int(i)] for i in np.asarray(train_idx)])
        smi_va = (None if not has_smiles
                  else [smiles[int(i)] for i in np.asarray(val_idx)])

        for level, draw in passes:
            lvl_key = f'{level:g}'
            pass_key = f'l{lvl_key}d{draw}f{fold_i}'
            if pass_key in done_passes:
                print(f"    {pass_key}: on disk in the checkpoint")
                continue
            print(f"\n    --- {condition} / level {level:g} / draw {draw} / "
                  f"fold {fold_i} ---")
            cell = acc.setdefault(lvl_key, {})
            try:
                y0, cmask, _prov = inject_condition(
                    data, dataset_name, condition, level, draw,
                    train_idx=train_idx, groups=noise_groups)
            except Exception as exc:
                errors[f'{lvl_key}|d{draw}|f{fold_i}|inject'] = (
                    f'{type(exc).__name__}: {exc}')
                continue
            provenance[pass_key] = convert_numpy(_prov)
            y0 = np.asarray(y0)
            if not is_clf:
                y0 = y0.astype(float)
            cmask = np.asarray(cmask, dtype=bool)
            n_tr = int(len(y0))
            # The clean labels in the dtype the fit on D gets, for the
            # score_clean column.
            y_clean_fit = (np.asarray(labels[train_idx]) if is_clf
                           else y_clean_train)

            if not is_clf:
                (injury_acc.setdefault(lvl_key, [])
                 .append(_injury_carried_by_flagged(y_clean_train, y0, cmask)))

            for mname in downstream_models:
                entry = entries.get(mname)
                factory = model_factories[mname]
                # The 4 configurations of this model, the hybrids rebuilt on
                # this fold's training rows from THIS pass's damaged labels
                # (the per-fold recipe's own representation scores, dirty-run
                # bug 1), 0/1 features unscaled.
                ins, unav, names = fold_model_inputs(
                    data, dataset_name, mname, entry, train_idx=train_idx,
                    val_idx=val_idx, y_train=y0, factory=factory,
                    metric=_metric)
                mrec = frec['models'][mname]
                if names:
                    mrec.setdefault('configuration_names', dict(names))
                mrec.setdefault('n_features', {})
                for sp, (Xa, _Xb) in ins.items():
                    mrec['n_features'].setdefault(sp, int(np.shape(Xa)[1]))
                for sp in CONFIGURATION_SPACES:
                    if sp not in ins:
                        why = unav.get(sp) or 'not built'
                        (mrec.setdefault('configurations_unavailable', {})
                         .setdefault(sp, why))
                        errors[f'{lvl_key}|d{draw}|f{fold_i}|{sp}/{mname}|'
                               f'build'] = why

                for space in CONFIGURATION_SPACES:
                    if space not in ins:
                        continue
                    X_tr_s, X_va_s = ins[space]
                    cell_models = [mname]
                    # One matrix per (configuration, model): the name in the
                    # detection and deletion records.
                    ckey = f'{space}/{mname}'
                    space_cell = cell.setdefault(space, {})

                    # An arm that aims at a predicted value fits a model to get
                    # there, and that target depends on the features, the
                    # labels and the groups and on nothing else. Hoisted once
                    # per (pass, configuration), which is what n18 does.
                    #
                    # ⛔ THE EMPTY LIST IS NOT PASSED THROUGH. correction_targets
                    # reads `want = set(want or ('oof', 'gp', 'quantile'))`, so
                    # an empty want asks it for all three. On a classification
                    # label oof_smooth is dropped, so the want-list is empty
                    # there and that is exactly when it would have fired.
                    _want = targets_needed(cleanups_used)
                    targets = (correction_targets_cached(
                        dataset_name, X_tr_s, y0, g_tr, _want)
                        if _want else {})

                    # The score to beat, fitted once per (pass, configuration,
                    # model) and shared by every detector and every flag rate.
                    base_score = {}
                    clean_score = {}
                    s, err = fit_score_cached(
                        dataset_name, mname, factory, X_tr_s, y0, X_va_s,
                        y_val, metric_fn, metric_name)
                    if s is None:
                        errors[f'{lvl_key}|{ckey}|{mname}|'
                               f'f{fold_i}|no_cleaning'] = err
                    else:
                        base_score[mname] = s
                    # score_clean for the rows: the same model on the clean
                    # training labels (plan section 2).
                    clean_score[mname], _e = fit_score_cached(
                        dataset_name, mname, factory, X_tr_s, y_clean_fit,
                        X_va_s, y_val, metric_fn, metric_name)

                    random_cache = {}
                    # {detector: scores} of the detectors that read no
                    # neighbours, reused by the neighbour settings.
                    today_scores = {}

                    # 2026-10-05 (chat 5): the main path runs only what reads
                    # no neighbours: cross_val, and under it every cleanup
                    # but label_smooth and knn_smooth. Until this date it
                    # also ran neighbor_residual (10 neighbours) and the two
                    # smoothing cleanups (9 neighbours) in the
                    # configuration's own features with every neighbour
                    # counting equally. The kNN record cut that space (axis 1,
                    # number 10), k=9 and k=10 (axis 4) and equal counting
                    # (axis 5). Those arms now run only in the screened
                    # settings below, and go into this table under
                    # '<detector>@screen<rank>'.
                    for detector in main_detectors:
                        try:
                            scores = nc.cached_detector_scores(
                                dataset_name, X_tr_s, y0, is_clf=is_clf,
                                detector=detector, groups=g_tr)
                        except Exception as exc:
                            errors[f'{lvl_key}|{ckey}|{detector}|f{fold_i}|'
                                   f'detector'] = f'{type(exc).__name__}: {exc}'
                            continue
                        if scores is None:
                            errors[f'{lvl_key}|{ckey}|{detector}|f{fold_i}|'
                                   f'detector'] = 'the estimator fit failed'
                            continue
                        scores = np.asarray(scores, dtype=float)
                        today_scores[detector] = scores

                        # Did the detector rank the damaged molecules at the
                        # top, before any arm ran? It costs no model fit.
                        det = _detection(cmask, scores, condition, is_clf)
                        if det is not None:
                            (detection_acc.setdefault(lvl_key, {})
                             .setdefault(ckey, {})
                             .setdefault(detector, []).append(det))

                        det_cell = space_cell.setdefault(detector, {})
                        _arm_scores(
                            det_cell, detector, scores, None, main_only,
                            X_tr_s,
                            X_va_s, y0, y_val, g_tr, cell_models, targets,
                            base_score, random_cache, lvl_key, level, draw,
                            fold_i, ckey, pass_key, n_tr, cmask, arms,
                            flag_rates, is_clf, model_factories, metric_fn,
                            metric_name, errors, deleted_acc)

                        for rate in flag_rates:
                            rate_cell = det_cell.get(f'{rate:g}', {})
                            _rows_for_rate(
                                rows, rate_cell, arms, detector, rate,
                                _sim_label, space, cell_models, base_score,
                                clean_score, condition, level, draw, fold_i,
                                pass_key)
                        _controls_for(
                            rows, detector, scores, None, main_only, X_tr_s,
                            X_va_s, y0, y_val, g_tr, cell_models, targets,
                            base_score, clean_score, arms, flag_rates, is_clf,
                            model_factories, metric_fn, metric_name,
                            _sim_label, space, condition,
                            level, draw, fold_i, errors, lvl_key, ckey)

                    # ── The decided neighbour settings (2026-10-05) ──
                    if not sim_settings:
                        errors[f'{lvl_key}|{ckey}|f{fold_i}|neighbours'] = (
                            'no neighbour setting for this dataset, so '
                            'neighbor_residual, label_smooth and knn_smooth '
                            'did not run')
                        continue
                    _neighbour_settings_pass(
                        dataset_name, data, sim_settings, entry=entry,
                        model_name=mname, factory=factory, metric=_metric,
                        task_type=task_type, is_clf=is_clf, space=space,
                        ckey=ckey, X_tr_s=X_tr_s, X_va_s=X_va_s, y0=y0,
                        y_val=y_val, g_tr=g_tr, cmask=cmask,
                        train_idx=train_idx, val_idx=val_idx, smi_tr=smi_tr,
                        smi_va=smi_va, detectors=detectors,
                        today_scores=today_scores, targets=targets,
                        base_score=base_score, clean_score=clean_score,
                        arms=arms, flag_rates=flag_rates,
                        model_factories=model_factories, metric_fn=metric_fn,
                        metric_name=metric_name, condition=condition,
                        level=level, draw=draw, fold_i=fold_i,
                        lvl_key=lvl_key, pass_key=pass_key, n_tr=n_tr,
                        rows=rows, errors=errors, sim_detection=sim_detection,
                        sim_unavailable=sim_unavailable,
                        screen_acc=screen_acc, space_cell=space_cell,
                        detection_acc=detection_acc, deleted_acc=deleted_acc,
                        random_cache=random_cache)

            # The pass is finished: its rows to disk, then the accumulators
            # to the checkpoint, so a kill between the two reruns the pass and
            # writes no row twice.
            rows.flush()
            done_passes.add(pass_key)
            _save_state()
            print(f"    level {lvl_key} draw {draw} fold {fold_i} done "
                  f"[{time.time() - t0:.0f}s]")

    results['folds'] = convert_numpy(fold_records)
    results['injection_provenance'] = provenance
    results['per_level'] = convert_numpy(acc)
    results['detection'] = convert_numpy({
        lvl: {sp: {d: _mean_detection(rows_) for d, rows_ in by_det.items()}
              for sp, by_det in by_sp.items()}
        for lvl, by_sp in detection_acc.items()})
    results['injury_carried_by_the_flagged_molecules'] = convert_numpy({
        lvl: _sum_injury(rows_) for lvl, rows_ in injury_acc.items()})
    results['molecules_deleted'] = convert_numpy({
        lvl: {sp: {who: {r: _sum_deleted(rows_) for r, rows_ in by_rate.items()}
                   for who, by_rate in by_who.items()}
              for sp, by_who in by_sp.items()}
        for lvl, by_sp in deleted_acc.items()})
    results['molecules_deleted_note'] = (
        'One entry per (level, feature matrix, who did the deleting, flag '
        'rate). A feature matrix is named <configuration>/<model>. Every '
        'number is a count of molecules summed over the passes of that level, '
        'so the share deleted is molecules_deleted divided by '
        'training_molecules. What the detector did is '
        'widened_molecules_among_the_deleted against the count chance would '
        'take, which is molecules_deleted times widened_molecules divided by '
        'training_molecules -- and the random rows show what chance actually '
        'took rather than what arithmetic says it should.')
    if errors:
        results['_errors'] = errors

    results['summary'] = convert_numpy(_summarise(
        acc, arms, downstream_models, metric_name, lower_is_better, seeds,
        results['detection']))
    results['similarity_sweep'] = {
        'knn_group': knn_grp,
        'settings_swept': [s.name for s in sim_settings],
        'ks': list(sweep_ks()),
        'counting_schemes': list(sweep_schemes()),
        'screen_top': screen_top(),
        'settings_env': os.environ.get('KIRBY_SIMILARITY_SETTINGS'),
        'what_is_swept': (
            'Every neighbour the experiment reads. Per decided model and '
            'configuration, the neighbour settings of this '
            "dataset's group (KNN_SEARCH_SPACE_RECORD, 2026-10-05). Detection "
            'by a neighbour detector is graded under every (setting, '
            'neighbour count), with no model fit. Every (setting, neighbour '
            'count, counting scheme) is then screened with no model fit on '
            "how well the neighbours predict each training molecule's damaged "
            'label (n07.screen_settings), and only the top screen_top are '
            'refitted: a detector that reads neighbours (neighbor_residual) '
            'is recomputed in the setting and drives every arm, the '
            'random-deletion control included; a detector that reads none '
            "(cross_val) keeps its main-path scores and drives only the arms that "
            'read neighbours (label_smooth, knn_smooth), which read them in '
            'the setting under the counting scheme. One neighbour list, '
            "scaffold-mates included (axis 6). The model is fitted on the "
            'configuration matrix the main path fits on. The refits are in '
            "the main table (per_level, detection, summary) under "
            "'<detector>@screen<rank>', rank 1 to screen_top in the screen's "
            'order on that pass; `screen` says which setting held each rank. '
            'Under a second counting scheme of the same (setting, neighbour '
            'count), the arms that read no neighbours repeat the first '
            "scheme's fits. The main table's plain detector names hold only "
            'arms that read no neighbours (since 2026-10-05; before, '
            "neighbor_residual, label_smooth and knn_smooth also ran there in "
            "the configuration's own features with 9 or 10 neighbours counting "
            'equally, which the kNN record cut). On a row, similarity_setting '
            "is the setting's name and method_setting starts "
            "'k=<count>|scheme=<scheme>|' ('k=<count>|' for an arm that reads "
            'no neighbours).'),
        'detection': convert_numpy({
            lvl: {mk: {m: {d: {lab: _mean_detection(rs)
                               for lab, rs in by_lab.items()}
                           for d, by_lab in by_d.items()}
                       for m, by_d in by_m.items()}
                  for mk, by_m in by_mk.items()}
            for lvl, by_mk in sim_detection.items()}),
        'spaces_unavailable': sim_unavailable,
    }
    results['screen'] = convert_numpy(screen_acc)
    results['rows'] = rows.summary()
    results['controls_run'] = bool(controls_enabled())
    _print_summary(results, metric_name)
    print(f"  D-versus-D' rows: {rows.n_written} written this run to "
          f"{rows.path or 'memory'}")
    ckpt.clear()
    return results


def _neighbour_settings_pass(dataset_name, data, sim_settings, *, entry,
                             model_name, factory, metric, task_type, is_clf,
                             space, ckey, X_tr_s, X_va_s, y0, y_val, g_tr,
                             cmask, train_idx, val_idx, smi_tr, smi_va,
                             detectors, today_scores, targets, base_score,
                             clean_score, arms, flag_rates, model_factories,
                             metric_fn, metric_name, condition, level, draw,
                             fold_i, lvl_key, pass_key, n_tr, rows, errors,
                             sim_detection, sim_unavailable, screen_acc,
                             space_cell, detection_acc, deleted_acc,
                             random_cache):
    """The decided neighbour settings for one (configuration, model, pass).

    2026-10-05 (chat 5), KNN_SEARCH_SPACE_RECORD:
      - the spaces are built by n07.fold_similarity_spaces from this fold's
        per-fold recipes (base_1/base_2 and their augmented copies, the two
        general hybrids, the distance hybrid, the similarity hybrid) and the
        fixed spaces (ECFP4 bits, forest leaves);
      - one neighbour list per setting, scaffold-mates included (axis 6;
        until 2026-10-05 the detector read a second list with the molecule's
        scaffold group excluded);
      - detection grading under every (setting, k): scoring only, no fit;
      - n07.screen_settings ranks every (setting, k, scheme) with no fit, and
        only the kept ones (top 3) are refitted;
      - the refits also go into the main table (space_cell, detection_acc,
        deleted_acc) under '<detector>@screen<rank>', rank 1 to 3 in the
        screen's order on this pass. These are the only neighbour arms the
        main table has (2026-10-05). random_cache is the main path's, so a
        random deletion of a count already fitted is not fitted again.
    """
    sp_tr, sp_va, sp_un = fold_similarity_spaces(
        dataset_name, sim_settings, model_input_tr=X_tr_s,
        model_input_va=X_va_s, reps=data['reps'], train_idx=train_idx,
        val_idx=val_idx, y_train=y0, groups_train=g_tr,
        base_reps=(entry or {}).get('base') or [],
        recipes=(entry or {}).get('recipes') or {},
        augmentations=data.get('augmentations'), model_name=model_name,
        model_factory=factory, metric=metric, task_type=task_type)
    for _sp, _w in sp_un.items():
        sim_unavailable.setdefault(_sp, {}).setdefault(model_name, _w)
    nb = fold_neighbours(dataset_name, sim_settings, sp_tr, sp_va, y0,
                         smiles_train=smi_tr, smiles_val=smi_va,
                         is_clf=is_clf)
    for s in sim_settings:
        L = nb.get(s.name)
        if L is None or not L.get('available'):
            sim_unavailable.setdefault(s.space, {}).setdefault(
                model_name, (L or {}).get('reason') or sp_un.get(s.space))

    # Detection grading: every (setting, k), no model fit.
    nd_scores = {}
    for s in sim_settings:
        L = nb.get(s.name)
        for k in usable_ks(L):
            for detector in detectors:
                if detector not in nc.NEIGHBOUR_DETECTORS:
                    continue
                try:
                    sc = nc.cached_detector_scores(
                        dataset_name, X_tr_s, y0, is_clf=is_clf,
                        detector=detector, groups=g_tr,
                        similarity_neighbours=L, similarity_k=k)
                except Exception as exc:
                    errors[f'{lvl_key}|{ckey}|{model_name}|{detector}|'
                           f'{s.name}|k={k}|f{fold_i}|detector'] = (
                        f'{type(exc).__name__}: {exc}')
                    continue
                if sc is None:
                    continue
                sc = np.asarray(sc, dtype=float)
                nd_scores[(detector, s.name, int(k))] = sc
                det = _detection(cmask, sc, condition, is_clf)
                if det is not None:
                    (sim_detection.setdefault(lvl_key, {})
                     .setdefault(ckey, {})
                     .setdefault(model_name, {})
                     .setdefault(detector, {})
                     .setdefault(f'{s.name}|k={k}', []).append(det))

    # The screen: every (setting, k, scheme), no model fit; the top
    # screen_top() are refitted below. DECIDED 2026-10-06 (Adelaide
    # delegated): the screen predicts a held-back fifth of the training
    # molecules (inner=), from the same spaces the lists were built from,
    # 'model_input' included. The refits stay in the main table by screen
    # rank ('<detector>@screen<rank>').
    kept, table = screen_settings(sim_settings, nb, y0, metric=metric,
                                  is_clf=is_clf, y_val=y_val, inner={
                                      'dataset_name': dataset_name,
                                      'spaces_train': sp_tr,
                                      'y_train': y0,
                                      'groups_train': g_tr,
                                      'smiles_train': smi_tr,
                                      'is_clf': is_clf})
    screen_acc.append({
        'configuration': space, 'model': model_name, 'fold': int(fold_i),
        'level': float(level), 'draw': int(draw), 'condition': condition,
        'kept': [f'{s_.name}|k={k_}|{sch}' for s_, k_, sch in kept],
        'table': table})

    if not kept:
        errors[f'{lvl_key}|{ckey}|{model_name}|f{fold_i}|screen'] = (
            'no neighbour setting was available on this pass, so '
            'neighbor_residual, label_smooth and knn_smooth did not run')
        return
    # A neighbour detector reads the list and k but not the counting scheme,
    # so the arms that read no neighbours are fitted once per (setting, k);
    # a second scheme of the same (setting, k) refits only the neighbour
    # cleanups, and its main-table cell copies the other arms' scores from
    # the first (the same fit, so a repeat, not a second measurement).
    first_cell = {}
    for rank, (s, kk, scheme) in enumerate(kept, 1):
        L = dict(nb[s.name], scheme=scheme)
        sim = (L, int(kk), s, scheme)

        def _pfx(cleanup, _s=s, _k=int(kk), _sch=scheme):
            return (f'k={_k}|scheme={_sch}|'
                    if cleanup and cleanup_reads_setting(cleanup, _s)
                    else f'k={_k}|')

        def _lab(d, c, _l=s.name):
            return _l

        for detector in detectors:
            prev = None
            if detector in nc.NEIGHBOUR_DETECTORS:
                s_scores = nd_scores.get((detector, s.name, int(kk)))
                if s_scores is None:
                    continue
                key = (detector, s.name, int(kk))
                prev = first_cell.get(key)
                only = None if prev is None else TODAY_NEIGHBOUR_CLEANUPS
            elif detector in today_scores:
                s_scores = today_scores[detector]
                only = TODAY_NEIGHBOUR_CLEANUPS
            else:
                continue
            # The main-table name of this refit: the detector and the rank
            # its setting took in this pass's screen.
            head = f'{detector}@screen{rank}'
            sw_cell = {}
            _arm_scores(
                sw_cell, head, s_scores, sim, only, X_tr_s, X_va_s, y0,
                y_val, g_tr, [model_name], targets, base_score,
                random_cache, lvl_key, level, draw, fold_i, ckey, pass_key,
                n_tr, cmask, arms, flag_rates, is_clf, model_factories,
                metric_fn, metric_name, errors,
                deleted_acc if only is None else None)
            for rate in flag_rates:
                _rows_for_rate(
                    rows, sw_cell.get(f'{rate:g}', {}), arms, detector, rate,
                    _lab, space, [model_name], base_score, clean_score,
                    condition, level, draw, fold_i, pass_key, prefix_fn=_pfx)
            _controls_for(
                rows, detector, s_scores, sim, only, X_tr_s, X_va_s, y0,
                y_val, g_tr, [model_name], targets, base_score, clean_score,
                arms, flag_rates, is_clf, model_factories, metric_fn,
                metric_name, _lab, space, condition, level, draw, fold_i,
                errors, lvl_key, ckey, prefix_fn=_pfx)

            if detector in nc.NEIGHBOUR_DETECTORS:
                if prev is None:
                    first_cell[key] = copy.deepcopy(sw_cell)
                    head_cell = sw_cell
                else:
                    head_cell = copy.deepcopy(prev)
                    _merge_cell(head_cell, sw_cell)
                det = _detection(cmask, s_scores, condition, is_clf)
                if det is not None:
                    (detection_acc.setdefault(lvl_key, {})
                     .setdefault(ckey, {})
                     .setdefault(head, []).append(det))
            else:
                # cross_val's arms that read no neighbours are under its plain
                # name already; this cell holds the two smoothing cleanups
                # and the no-cleaning score they are paired against.
                head_cell = sw_cell
            _merge_cell(space_cell.setdefault(head, {}), head_cell)


def _merge_cell(dst, src):
    """Merge one pass's {rate: {arm: {model: {pass: score}}}} into dst."""
    for rate, by_arm in src.items():
        d_rate = dst.setdefault(rate, {})
        for arm, by_model in by_arm.items():
            d_arm = d_rate.setdefault(arm, {})
            for m, by_pass in by_model.items():
                d_arm.setdefault(m, {}).update(by_pass)


def _mean_detection(rows):
    """Average the detection readings over the passes of one level."""
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


def _sum_injury(rows):
    """The damage counts of one level, summed over its passes."""
    if not rows:
        return None
    flagged = float(sum(r['squared_error_on_the_widened_molecules']
                        for r in rows))
    total = float(sum(r['squared_error_over_every_training_molecule']
                      for r in rows))
    return {
        'passes': len(rows),
        'squared_error_on_the_widened_molecules': flagged,
        'squared_error_over_every_training_molecule': total,
        'widened_molecules_summed_over_the_passes':
            int(sum(r['widened_molecules'] for r in rows)),
        'training_molecules_summed_over_the_passes':
            int(sum(r['training_molecules'] for r in rows)),
        'share_of_the_squared_error_on_the_widened_molecules': (
            flagged / total if total > 0 else float('nan')),
        'what_it_says': (
            'The share is what deleting exactly the widened molecules could '
            'remove at best. Under outlier_p01/p05/p10 it is below one because '
            'every other label moved too; under outlier_only_p01/p05/p10 it is '
            'one.'),
    }


def _sum_deleted(rows):
    """The deletion counts of one (level, space, who, rate), over its passes."""
    if not rows:
        return None
    return {
        'passes': len(rows),
        'molecules_deleted_summed_over_the_passes':
            int(sum(r['molecules_deleted'] for r in rows)),
        'training_molecules_summed_over_the_passes':
            int(sum(r['training_molecules'] for r in rows)),
        'widened_molecules_among_the_deleted_summed_over_the_passes':
            int(sum(r['widened_molecules_among_the_deleted'] for r in rows)),
        'widened_molecules_summed_over_the_passes':
            int(sum(r['widened_molecules'] for r in rows)),
    }


def _paired(base, got, lower_is_better):
    """One arm against another, on the passes both scored. Positive means better.

    ⛔ THE DIRECTION IS READ OFF THE DATASET, NOT ASSUMED. flab_thermo_tresanco
    is spearman and higher wins; herg_ki, caco2 and herg_ic50 are rmse and lower
    wins. The gain below is (base - got) where lower wins and (got - base) where
    higher does, so a positive gain means the same thing on every dataset and
    nothing downstream has to know which metric it is looking at.
    """
    shared = sorted(set(base) & set(got))
    gain_by_pass = {k: ((base[k] - got[k]) if lower_is_better
                        else (got[k] - base[k])) for k in shared}
    gains = np.array([gain_by_pass[k] for k in shared], dtype=float)
    gains = gains[np.isfinite(gains)]
    scale = max(abs(float(np.mean(list(base.values())))) if base else 1.0, 1.0)
    tol = 1e-9 * scale
    return {
        'n_passes_paired': int(gains.size),
        'passes_won': int((gains > tol).sum()),
        'passes_lost': int((gains < -tol).sum()),
        'passes_tied': int((np.abs(gains) <= tol).sum()),
        'tie_tolerance': float(tol),
        'mean_gain': float(gains.mean()) if gains.size else float('nan'),
        'spread_over_pass_gains': (float(gains.std()) if gains.size
                                   else float('nan')),
        'pass_gains': [float(x) for x in gains],
        # 2026-10-05 (chat 5): the gain per outer fold (mean over that fold's
        # draws) and its spread over folds (noise_helpers.fold_spread; one
        # fold says 'one fold, no spread'). No significance test.
        'gain_by_fold': _by_fold(gain_by_pass),
    }


def _summarise(acc, arms, models, metric_name, lower_is_better, seeds,
               detection):
    """Every arm against no cleaning, and against every random deletion.

    Two comparisons per arm and they answer different questions.

      vs_no_cleaning       did the arm beat training on the corrupted labels.
      vs_random_deletion   did it beat deleting the same number of molecules at
                           random, ONE ENTRY PER SEED. `beat_every_random_seed`
                           is true only where the arm won a majority of passes
                           against every seed that ran.

    A pass is one (noise draw, fold). Nothing here averages across seeds: the
    per-seed rows and their own spread are what is written.
    """
    random_arms = [a[0] for a in arms if a[1] == 'delete_at_random']
    other_arms = [a[0] for a in arms
                  if a[1] not in ('no_cleaning', 'delete_at_random')]
    family_of = {a[0]: a[1] for a in arms}

    out = {'metric': metric_name, 'lower_is_better': bool(lower_is_better),
           'headline': ('passes won against no cleaning, and against deleting '
                        'the same number of molecules at random, paired per '
                        'pass; a pass is one (noise draw, fold)'),
           'random_deletion_seeds': list(seeds),
           'per_level': {}}

    # THREE buckets, not two, and every one of them says what it is. Until
    # 2026-09-23 the headline bucket was filled by `beat_every_random_seed`
    # alone, with the no-cleaning comparison reached only in the `elif`. So a
    # setting that beat all three random seeds AND LOST TO NOT CLEANING AT ALL
    # was counted as the arm winning. The one number this experiment exists to
    # produce was counting cases that argue against it.
    beat_control = []                 # beat every random seed AND no cleaning
    beat_nothing_but_no_cleaning = [] # beat no cleaning, not every random seed
    beat_random_but_lost_to_no_cleaning = []

    for lvl, by_space in acc.items():
        lrow = {'per_space': {}}
        for space, by_det in by_space.items():
            srow = {'per_detector': {}}
            for detector, by_rate in by_det.items():
                drow = {'per_flag_rate': {}}
                for rate, by_arm in by_rate.items():
                    base_by_model = by_arm.get('no_cleaning') or {}
                    per_arm = {}
                    for arm in other_arms + random_arms:
                        per_model = {}
                        for m in models:
                            b = base_by_model.get(m) or {}
                            g = (by_arm.get(arm) or {}).get(m) or {}
                            if not b or not g:
                                continue
                            entry = {'vs_no_cleaning':
                                     _paired(b, g, lower_is_better)}
                            if family_of[arm] != 'delete_at_random':
                                vs = {}
                                for r_arm in random_arms:
                                    rg = (by_arm.get(r_arm) or {}).get(m) or {}
                                    if not rg:
                                        continue
                                    vs[r_arm] = _paired(rg, g, lower_is_better)
                                entry['vs_random_deletion'] = vs
                                entry['beat_every_random_seed'] = bool(
                                    vs and all(
                                        v['passes_won'] * 2 > v['n_passes_paired']
                                        for v in vs.values()))
                                entry['random_seeds_compared'] = len(vs)
                                won_nc = (entry['vs_no_cleaning']['passes_won'] * 2
                                          > entry['vs_no_cleaning']['n_passes_paired'])
                                key = (f'{lvl}|{space}|{detector}|{rate}|{arm}'
                                       f'|{m}')
                                entry['beat_no_cleaning'] = bool(won_nc)
                                if entry['beat_every_random_seed'] and won_nc:
                                    beat_control.append(key)
                                elif won_nc:
                                    beat_nothing_but_no_cleaning.append(key)
                                elif entry['beat_every_random_seed']:
                                    # Beat the random deletions and still lost
                                    # to leaving the labels alone. That is the
                                    # random control being WEAK here, not the
                                    # arm being strong, and it used to be
                                    # counted as a win.
                                    beat_random_but_lost_to_no_cleaning.append(key)
                            per_model[m] = entry
                        if per_model:
                            per_arm[arm] = {'per_model': per_model}

                    # The control's own gain against no cleaning, per seed.
                    # Never averaged: on 2026-09-23 two seeds won every fold
                    # and a third lost most of them, and their mean says
                    # nothing about either.
                    control_spread = {}
                    for m in models:
                        b = base_by_model.get(m) or {}
                        rows = {}
                        for r_arm in random_arms:
                            rg = (by_arm.get(r_arm) or {}).get(m) or {}
                            if b and rg:
                                p = _paired(b, rg, lower_is_better)
                                rows[r_arm] = {
                                    'passes_won': p['passes_won'],
                                    'n_passes_paired': p['n_passes_paired'],
                                    'mean_gain': p['mean_gain']}
                        if rows:
                            vals = [v['mean_gain'] for v in rows.values()
                                    if np.isfinite(v['mean_gain'])]
                            control_spread[m] = {
                                'per_seed': rows,
                                'best_seed_mean_gain': (max(vals) if vals
                                                        else float('nan')),
                                'worst_seed_mean_gain': (min(vals) if vals
                                                         else float('nan')),
                                'seeds_read': len(rows),
                                'note': ('one row per seed and no mean across '
                                         'them; the best and the worst are the '
                                         'spread this arm has to clear')}
                    drow['per_flag_rate'][rate] = {
                        'per_arm': per_arm,
                        'random_deletion_against_no_cleaning': control_spread,
                        'no_cleaning_mean_by_model': {
                            m: (float(np.mean(list(v.values()))) if v
                                else float('nan'))
                            for m, v in base_by_model.items()},
                        # 2026-10-05 (chat 5): the same score per outer fold
                        # and its spread over folds (fold_spread).
                        'no_cleaning_by_fold_by_model': {
                            m: _by_fold(v)
                            for m, v in base_by_model.items()},
                    }
                # One reading per feature matrix in this space. The key is the
                # space where every model shares one matrix, and
                # '<space>/<first model>' where the models have their own.
                drow['detection_by_matrix'] = {
                    k: v.get(detector)
                    for k, v in detection.get(lvl, {}).items()
                    if (k == space or k.startswith(space + '/'))
                    and v.get(detector) is not None}
                drow['detection'] = detection.get(lvl, {}).get(space, {}).get(
                    detector)
                srow['per_detector'][detector] = drow
            lrow['per_space'][space] = srow
        out['per_level'][lvl] = lrow

    out['settings_where_the_arm_beat_every_random_seed_and_no_cleaning'] = beat_control
    out['settings_where_it_beat_only_no_cleaning'] = beat_nothing_but_no_cleaning
    out['settings_where_it_beat_the_seeds_but_lost_to_no_cleaning'] = \
        beat_random_but_lost_to_no_cleaning
    out['counts'] = {
        'settings_beating_every_random_seed_AND_no_cleaning': len(beat_control),
        'settings_beating_no_cleaning_but_not_every_random_seed':
            len(beat_nothing_but_no_cleaning),
        'settings_beating_every_random_seed_but_LOSING_to_no_cleaning':
            len(beat_random_but_lost_to_no_cleaning),
        'what_a_setting_is': ('one (level, space, detector, flag rate, arm, '
                              "downstream model); a detector named "
                              "'<detector>@screen<rank>' is that detector "
                              "or its cleanups in the neighbour setting "
                              "that took that rank in the pass's screen"),
    }
    return out


def _print_summary(results, metric_name):
    s = results['summary']
    print("\n" + "=" * 70)
    print(f"SUMMARY ({metric_name}; paired per pass, a pass is one draw and "
          f"one fold)")
    if not results.get('hybrid_available', True):
        print("  no hybrid recipe was available; single representations only")
    inj = results.get('injury_carried_by_the_flagged_molecules') or {}
    for lvl, row in inj.items():
        if not row:
            continue
        print(f"  level {lvl}: the molecules the injector widened carry "
              f"{row['share_of_the_squared_error_on_the_widened_molecules']:.3f} "
              f"of the squared label error, over "
              f"{row['widened_molecules_summed_over_the_passes']} widened "
              f"molecules of "
              f"{row['training_molecules_summed_over_the_passes']} training "
              f"molecules summed over {row['passes']} passes")

    for lvl, lrow in s['per_level'].items():
        print(f"\n  level {lvl}:")
        for space, srow in lrow['per_space'].items():
            for detector, drow in srow['per_detector'].items():
                for mkey, det in (drow.get('detection_by_matrix')
                                  or {}).items():
                    if not (det and np.isfinite(det.get(
                            'detection_average_precision', float('nan')))):
                        continue
                    print(f"    {mkey}/{detector}: ranked the widened "
                          f"molecules at average precision "
                          f"{det['detection_average_precision']:.3f}, over "
                          f"{det['detection_n_corrupted']:.0f} widened "
                          f"molecules of {det['detection_n_training']:.0f} "
                          f"training molecules")
                for rate, rrow in drow['per_flag_rate'].items():
                    for arm, arow in rrow['per_arm'].items():
                        for m, entry in arow['per_model'].items():
                            nc = entry['vs_no_cleaning']
                            tail = ''
                            if 'beat_every_random_seed' in entry:
                                tail = (f"; beat all "
                                        f"{entry['random_seeds_compared']} "
                                        f"random-deletion seeds: "
                                        f"{entry['beat_every_random_seed']}")
                            print(f"      {space}/{detector}/rate {rate}/"
                                  f"{arm}/{m}: won {nc['passes_won']} of "
                                  f"{nc['n_passes_paired']} passes against no "
                                  f"cleaning, mean {nc['mean_gain']:+.4f} "
                                  f"(spread {nc['spread_over_pass_gains']:.4f})"
                                  f"{tail}")
                    for m, csp in rrow['random_deletion_against_no_cleaning'].items():
                        per_seed = ', '.join(
                            f"{k.split('/')[-1]} {v['passes_won']} of "
                            f"{v['n_passes_paired']} passes at "
                            f"{v['mean_gain']:+.4f}"
                            for k, v in csp['per_seed'].items())
                        print(f"      CONTROL {space}/{detector}/rate {rate}/"
                              f"{m}: {per_seed}")

    c = s['counts']
    print(f"\n  settings where an arm beat every random-deletion seed AND "
          f"beat not cleaning at all: "
          f"{c['settings_beating_every_random_seed_AND_no_cleaning']}")
    print(f"  settings where it beat no cleaning and NOT every random seed: "
          f"{c['settings_beating_no_cleaning_but_not_every_random_seed']}")
    print(f"  settings where it beat every random seed and LOST to not "
          f"cleaning: "
          f"{c['settings_beating_every_random_seed_but_LOSING_to_no_cleaning']}"
          f"  <- the random control was weak here, not the arm strong")
    print(f"  a setting is {c['what_a_setting_is']}.")
    print("  Beating no cleaning is not the bar. Beating every seed is.")


def main():
    parser = create_experiment_parser(
        'N24: delete the flagged molecules, or keep them')
    parser.add_argument('--noise-condition', type=str, default=None,
                        help=f'Injected condition (default: every decided '
                             f'kind that damages a subset; with --tag, '
                             f'{DEFAULT_NOISE_CONDITION}, or '
                             f'{DEFAULT_STRATEGY_CLF} on class labels). One of '
                             f'the decided kinds that damage a subset: '
                             f'outlier_p05, censoring, shift_only_p10, '
                             f'sparse_dprime_up_p10, or a flip kind on class '
                             f'labels; anything else is written as a skip')
    parser.add_argument('--levels', type=str, default=None,
                        help='Comma-separated noise levels. Default: the '
                             'decided levels for this dataset group and '
                             'condition (noise_helpers.decided_levels)')
    parser.add_argument('--detectors', type=str, default=None,
                        help='Comma-separated suspicion scores '
                             f'(default {",".join(DETECTORS)})')
    parser.add_argument('--flag-rates', type=str, default=None,
                        help='Comma-separated shares of training molecules '
                             'treated as suspect '
                             f'(default {",".join(str(r) for r in FLAG_RATES)})')
    parser.add_argument('--random-seeds', type=str, default=None,
                        help='Comma-separated random-deletion replicates '
                             f'(default '
                             f'{",".join(str(s) for s in RANDOM_DELETION_SEEDS)})')
    parser.add_argument('--draws', type=str, default=None,
                        help='Comma-separated injection replicates '
                             f'(default {",".join(str(d) for d in NOISE_DRAWS)})')
    parser.add_argument('--exp-folds', type=int, default=None,
                        help='Outer folds this experiment evaluates over. '
                             'Overrides the 1 or 5 --mode gives; '
                             'KIRBY_NOISE_N_FOLDS overrides both')
    parser.add_argument('--tag', type=str, default=None,
                        help='Extra text on the result filename, so two runs '
                             'of one condition on one dataset land on two files')
    args = parser.parse_args()
    config = get_config(args)
    if args.exp_folds:
        config['n_splits'] = int(args.exp_folds)
    data = load_data(args.dataset, rep_preset=args.reps)

    kw = {}
    if args.noise_condition:
        kw['noise_condition'] = args.noise_condition
    if args.levels:
        kw['levels'] = [float(x) for x in args.levels.split(',')]
    if args.detectors:
        kw['detectors'] = [x.strip() for x in args.detectors.split(',')]
    if args.flag_rates:
        kw['flag_rates'] = [float(x) for x in args.flag_rates.split(',')]
    if args.random_seeds:
        kw['random_seeds'] = [int(x) for x in args.random_seeds.split(',')]
    if args.draws:
        kw['draws'] = [int(x) for x in args.draws.split(',')]

    is_clf = data.get('task_type') == 'classification'
    cond = kw.get('noise_condition') or (DEFAULT_STRATEGY_CLF if is_clf
                                         else DEFAULT_NOISE_CONDITION)
    if args.tag:
        # A second run of one condition on one dataset, on its own file:
        # <dataset>__<condition>_<tag>.json. Run directly, not through the
        # sweep, which names files by condition alone.
        kw['noise_condition'] = cond
        res = _run_one_condition(data, config, args.output, args.dataset,
                                 **kw)
        save_experiment_results(res, args.output, 'n24_delete_or_keep',
                                args.dataset,
                                fingerprint=input_fingerprint(data, config),
                                filename_suffix=f'{cond}_{args.tag}')
        return
    # 2026-10-05 (chat 5): run() goes through sweep_conditions, which has
    # already written <dataset>__<condition>.json. Saving the index it returns
    # under the same suffix would replace the scores with a list of file
    # names, so the index goes to <dataset>.json, as run_all saves it.
    res = run(data, config, args.output, args.dataset, **kw)
    save_experiment_results(res, args.output, 'n24_delete_or_keep',
                            args.dataset)


if __name__ == '__main__':
    main()
