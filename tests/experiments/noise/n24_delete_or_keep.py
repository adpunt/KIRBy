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

WHAT THE HELD-OUT MOLECULES ARE GRADED AGAINST. `labels[val_idx]`, untouched, in
every arm. No cleaned label and no corrupted label is ever scored against.

Usage:
    python n24_delete_or_keep.py --dataset flab_thermo_tresanco --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import time
import zlib

import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_model_factories, DOWNSTREAM_MODELS,
    get_best_search_for_downstream, pick_noise_base_rep,
    pool_recipes_by_family, describe_pool_family, POOL_SUBSETS,
)
from experiments.dataset_policy import (
    noise_levels, noise_groups_for, noise_dose_units,
)
from experiments.noise.noise_helpers import (
    noise_spec_for, inject_train_labels, detection_scores,
    detection_is_defined, hybrid_for_fold, DETECTION_IS_DEFINED,
)
# The cleanups are n07's, called rather than copied, so an arm here is the same
# code n07 and n18 already measured. `_compute_noise_scores` is n07's detector
# call: it offers the scaffold groups under BOTH names the estimator families
# use and refuses a detector that can take neither, instead of dropping them.
from experiments.noise.n07_remediation import (
    cleaned_training_set, correction_targets, targets_needed,
    _compute_noise_scores, _predict_for_metric,
    REGRESSION_ONLY_CLEANUPS,
)
from kirby.noise_spec import GROUPED_STRATEGIES, OUTLIER_STRATEGIES

# ── The arms ───────────────────────────────────────────────────────────────
# One cleanup name per arm, all of them n07's. The three groups are the answer
# the experiment is choosing between.
DELETE_CLEANUP = 'remove'
REWRITE_CLEANUPS = ('label_smooth', 'knn_smooth', 'oof_smooth')
REWEIGHT_CLEANUPS = ('downweight', 'huber_weight')

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

# The condition injected by default. outlier_p10 damages a SUBSET, which is what
# makes a flagged set and a detection score mean anything -- under gaussian, the
# student_t settings and laplace every training label moves and the list of
# damaged molecules is the list of all of them.
DEFAULT_NOISE_CONDITION = 'outlier_p10'

# The replicate index of the injection. n07, n11 and n13 sweep the same three.
NOISE_DRAWS = (0, 1, 2)

# Which searches the recipe may come from. Same as n02, n07, n11, n12, n13, n17,
# n18, n19 and n23 since 2026-09-12.
POOL_SUBSET = 'predictive_and_similarity'

# One feature space per family of stored search. 'hybrid' is the predictive
# recipe, so it is named the way every other file in this directory names it.
HYBRID_SPACE = 'hybrid'
SPACE_FOR_FAMILY = {
    'predictive': HYBRID_SPACE,
    'similarity': 'hybrid_similarity',
}


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
    """
    rewrite = tuple(c for c in REWRITE_CLEANUPS
                    if not (is_clf and c in REGRESSION_ONLY_CLEANUPS))
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


def _levels_and_draws(dataset_name, condition, is_clf, levels=None, draws=None):
    """(level, draw) pairs to inject.

    At level 0.0 on a regression dataset the draw delivers exactly zero, so
    three draws are three identical runs and one is kept. NOT on a
    classification set: the flip calibrator returns a floor rather than zero at
    a target of 0.0. Same rule as noise_helpers.noisy_search_specs and n23.
    """
    levels = (noise_levels(dataset_name, strategy=condition, is_clf=is_clf)
              if levels is None else levels)
    draws = tuple(NOISE_DRAWS if draws is None else draws)
    out = []
    for lvl in levels:
        here = draws if (float(lvl) > 0 or is_clf) else draws[:1]
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


def run(data, config, output_dir, dataset_name, **kwargs):
    """N24: delete the flagged molecules, rewrite them, or weigh them down."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N24: DELETE THE FLAGGED MOLECULES, OR KEEP THEM")
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
    detectors = list(kwargs.get('detectors') or DETECTORS)
    flag_rates = [float(x) for x in (kwargs.get('flag_rates') or FLAG_RATES)]
    seeds = [int(s) for s in
             (kwargs.get('random_seeds') or RANDOM_DELETION_SEEDS)]

    # ⛔ config FIRST, then the kwarg. run_all passes the caller's model list in
    # `config['models']` and never as a `models` kwarg. Eight files in this
    # directory read kwargs alone and silently dropped the list the runner was
    # asked for; fixed across them on 2026-09-20. An explicit kwarg still wins,
    # which is what a smoke or a direct caller passes.
    downstream_models = list(
        kwargs.get('models') or config.get('models') or DOWNSTREAM_MODELS)
    model_factories = get_model_factories(
        downstream_models, task_type=task_type, n_samples=len(labels))
    downstream_models = [m for m in downstream_models if m in model_factories]
    if not downstream_models:
        raise RuntimeError(
            f"n24: no downstream model could be built. Asked for "
            f"{list(kwargs.get('models') or DOWNSTREAM_MODELS)}, got "
            f"{sorted(model_factories)}.")

    base_rep_name = (kwargs.get('base_rep')
                     or pick_noise_base_rep(reps, dataset_name))
    noise_groups = noise_groups_for(data)[0]

    # ⛔ A GROUPED CONDITION NEEDS SCAFFOLD GROUPS AND AN ANTIBODY SET HAS NONE.
    # n23's guard, for n23's reason: without it every fold raises inside the
    # injector, each raise is caught and recorded, and the run finishes with a
    # result file that holds no score at all.
    if str(condition) in GROUPED_STRATEGIES and noise_groups is None:
        raise RuntimeError(
            f"n24: condition {condition!r} gives whole scaffold families the "
            f"wider error, and {dataset_name} loads no SMILES, so there are no "
            f"scaffold groups to give it to. noise_groups_for returned None. "
            f"Pick a condition whose damaged molecules are chosen without "
            f"groups: "
            f"{', '.join(c for c in DETECTION_IS_DEFINED if c not in GROUPED_STRATEGIES)}.")

    # The flagged set has to be a SUBSET for any of this to be defined: an arm
    # that deletes the flagged molecules is doing nothing distinguishable when
    # every training molecule is flagged by construction.
    if not detection_is_defined(condition):
        raise ValueError(
            f"n24: condition {condition!r} moves every training label, so there "
            f"is no damaged subset for a detector to find and no flagged set to "
            f"delete or keep. Run it under one of: "
            f"{', '.join(DETECTION_IS_DEFINED)}.")

    delete_cleanup, rewrite_cleanups, reweight_cleanups = cleanups_for(is_clf)
    cleanups_used = (delete_cleanup,) + rewrite_cleanups + reweight_cleanups
    arms = build_arms(is_clf, seeds)
    arm_names = [a[0] for a in arms]
    rate_reading = [c for c in cleanups_used if reads_flag_rate(c, is_clf)]

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'noise_condition': condition,
        'dose_units': noise_dose_units(dataset_name),
        'noise_draws': list(NOISE_DRAWS if kwargs.get('draws') is None
                            else kwargs['draws']),
        'detectors': detectors,
        'flag_rates': flag_rates,
        'random_deletion_seeds': seeds,
        'models': downstream_models,
        'base_rep': base_rep_name,
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
            'Under outlier_p01/p05/p10 the selected share gets an error three '
            'times wider and every other label still moves, so deleting '
            'exactly the selected molecules leaves the rest of the damage in '
            'place. Under outlier_only_p01/p05/p10 the other labels are '
            'bit-identical. This run measures the difference on its own labels '
            'and writes it to injury_carried_by_the_flagged_molecules.'),
        'condition_moves_every_other_label': bool(
            condition in OUTLIER_STRATEGIES),
        'features_are_fixed_note': (
            "Each feature space is built once per (draw, fold) from that "
            "pass's training molecules at their CORRUPTED labels, and every "
            'arm, detector and flag rate shares it. No cleanup reaches the '
            "feature ranking or the budget split; whether it should is n20's "
            'question.'),
        'per_level': {},
        'detection': {},
        'injury_carried_by_the_flagged_molecules': {},
        'molecules_deleted': {},
    }

    sample_X = reps[base_rep_name]
    splits = list(cv_split(groups, n_folds, sample_X, labels, mode=mode))

    # ── The hybrid recipes, fetched once. n18's block, for n18's reason. ──
    bp = None
    lookup_error = None
    _meta = None
    try:
        _search, _meta = get_best_search_for_downstream(
            reps, labels, groups,
            downstream_model=downstream_models[0], budget=None,
            augmentations=data.get('augmentations'),
            dataset_name=dataset_name, task_type=task_type, data=data,
            pool_subset=POOL_SUBSET)
        bp = _meta['best_config']['blueprint'] if _meta else None
    except Exception as exc:
        # What went wrong is recorded, not just that something did. A dataset
        # the precompute never covered and a stored search written under an
        # older identity call for opposite responses and both used to be
        # reported as "the pool held nothing".
        lookup_error = f'{type(exc).__name__}: {exc}'
        print(f"  hybrid recipe lookup failed: {lookup_error}")

    hybrid_recipes, hybrid_unavailable = pool_recipes_by_family(
        _meta if bp is not None else None, SPACE_FOR_FAMILY,
        eligible_families=POOL_SUBSETS.get(POOL_SUBSET))
    recipe_for_space = {r['space']: r['blueprint'] for r in hybrid_recipes}
    results['hybrid_spaces_unavailable'] = hybrid_unavailable
    results['hybrid_lookup_error'] = lookup_error

    if not hybrid_recipes:
        spaces = ('base',)
        results['hybrid_available'] = False
        results['hybrid_unavailable_reason'] = (
            (f'the pool lookup RAISED {lookup_error} -- the pool was never '
             f'read, so this is not a statement about what it holds. Every arm '
             f'ran in the base representation only.')
            if lookup_error else
            (f'no recipe came back from the pool subset {POOL_SUBSET!r}, so '
             f'every arm ran in the base representation only. Dropping the run '
             f'would lose real answers because one of several spaces is '
             f'missing.'))
        print(f"  no hybrid recipe from pool subset {POOL_SUBSET}; "
              f"base representation only")
    else:
        spaces = ('base',) + tuple(r['space'] for r in hybrid_recipes)
        results['hybrid_available'] = True
        results['hybrid_family'] = _meta.get('objective')
        results['hybrid_spaces'] = convert_numpy({
            r['space']: {
                'built_under_objective': r['objective'],
                'features_chosen_by': describe_pool_family(r['objective']),
                'is_the_recipe_the_pool_returned': r['is_pool_winner'],
                'budget': r['budget'],
                'allocation_method': r['allocation_method'],
                'importance_method': r['importance_method'],
                'reps': sorted(getattr(r['blueprint'], 'reps', []) or []),
                'search_score_mean': r['search_score_mean'],
                'search_metric_name': r['search_metric_name'],
            } for r in hybrid_recipes})
        for r in hybrid_recipes:
            print(f"  hybrid space {r['space']}: family {r['objective']!r}, "
                  f"budget {r['budget']} features, allocation "
                  f"{r['allocation_method']}")
    for _sp, _why in hybrid_unavailable.items():
        print(f"  no {_sp} space: {_why}")
    results['spaces_run'] = list(spaces)

    passes = _levels_and_draws(dataset_name, condition, is_clf,
                               levels=kwargs.get('levels'),
                               draws=kwargs.get('draws'))
    results['noise_levels'] = sorted({lvl for lvl, _ in passes})

    print(f"  {len(labels)} molecules, {len(splits)} folds, "
          f"models {downstream_models}")
    print(f"  condition {condition}, levels {results['noise_levels']}, "
          f"{len(passes)} (level, draw) passes")
    print(f"  detectors {detectors}, flag rates {flag_rates}")
    print(f"  {len(arm_names)} arms: 1 to beat, 1 deletion, "
          f"{len(rewrite_cleanups)} that rewrite labels, "
          f"{len(reweight_cleanups)} that reduce weights, "
          f"{len(seeds)} random-deletion replicates")
    print(f"  base representation: {base_rep_name}")

    # acc[level][space][detector][flag rate][arm][model][pass] = score
    acc = {}
    errors = {}
    detection_acc = {}
    injury_acc = {}
    deleted_acc = {}
    t0 = time.time()

    for level, draw in passes:
        lvl_key = f'{level:g}'
        print(f"\n  --- {condition} / level {level:g} / draw {draw} ---")
        spec = noise_spec_for(dataset_name, condition, level,
                              noise_seed=draw, is_clf=is_clf)
        cell = acc.setdefault(lvl_key, {})

        for fold_i, (train_idx, val_idx) in enumerate(splits):
            pass_key = f'l{lvl_key}d{draw}f{fold_i}'
            y_val = labels[val_idx]
            y_clean_train = np.asarray(labels[train_idx], dtype=float)
            g_tr = groups[train_idx] if groups is not None else None

            try:
                y0, cmask, _prov = inject_train_labels(
                    labels, spec, train_idx=train_idx, is_clf=is_clf,
                    groups=noise_groups, dataset_name=dataset_name)
            except Exception as exc:
                errors[f'{lvl_key}|d{draw}|f{fold_i}|inject'] = (
                    f'{type(exc).__name__}: {exc}')
                continue
            y0 = np.asarray(y0)
            if not is_clf:
                y0 = y0.astype(float)
            n_tr = int(len(y0))

            if not is_clf:
                (injury_acc.setdefault(lvl_key, [])
                 .append(_injury_carried_by_flagged(y_clean_train, y0, cmask)))

            train_reps = {k: np.asarray(v[train_idx], dtype=np.float64)
                          for k, v in reps.items()}
            val_reps = {k: np.asarray(v[val_idx], dtype=np.float64)
                        for k, v in reps.items()}

            # The feature spaces for this pass. Built from the CORRUPTED
            # training labels, once, and shared by every arm below.
            fold_spaces = {}
            sc = StandardScaler()
            fold_spaces['base'] = (
                sc.fit_transform(train_reps[base_rep_name]),
                sc.transform(val_reps[base_rep_name]))
            for _sp in spaces:
                _bp = recipe_for_space.get(_sp)
                if _bp is None:
                    continue
                X_tr_h, X_va_h = hybrid_for_fold(
                    train_reps, val_reps, y0, _bp, g_tr,
                    model_factories[downstream_models[0]](),
                    metric_fn, lower_is_better)
                if X_tr_h is None:
                    errors[f'{lvl_key}|d{draw}|f{fold_i}|{_sp}|build'] = (
                        'the hybrid could not be built on this pass')
                else:
                    sh = StandardScaler()
                    fold_spaces[_sp] = (sh.fit_transform(X_tr_h),
                                        sh.transform(X_va_h))

            for space in spaces:
                if space not in fold_spaces:
                    continue
                X_tr_s, X_va_s = fold_spaces[space]
                space_cell = cell.setdefault(space, {})

                # An arm that aims at a predicted value fits a model to get
                # there, and that target depends on the features, the labels
                # and the groups and on nothing else. Hoisted once per (pass,
                # space), which is what n18 does and why correction_targets
                # exists.
                #
                # ⛔ THE EMPTY LIST IS NOT PASSED THROUGH. correction_targets
                # reads `want = set(want or ('oof', 'gp', 'quantile'))`, so an
                # empty want asks it for all three -- a Gaussian process and a
                # quantile forest fitted for arms that do not exist. On a
                # classification label oof_smooth is dropped, so the want-list
                # is empty there and that is exactly when it would have fired.
                _want = targets_needed(cleanups_used)
                targets = (correction_targets(X_tr_s, y0, groups=g_tr,
                                              want=_want) if _want else {})

                # The score to beat, fitted once per (pass, space, model) and
                # shared by every detector and every flag rate -- they all have
                # to beat the same number or the comparison is not paired.
                base_score = {}
                for mname in downstream_models:
                    s, err = _fit_score(
                        model_factories[mname],
                        {'X': X_tr_s, 'y': y0, 'sample_weight': None},
                        X_va_s, y_val, metric_fn, metric_name)
                    if s is None:
                        errors[f'{lvl_key}|{space}|{mname}|'
                               f'f{fold_i}|no_cleaning'] = err
                    else:
                        base_score[mname] = s

                # One random deletion is one fit per (space, model, count,
                # seed) and the same count comes back from both detectors, so
                # the fit is done once and read twice rather than repeated.
                random_cache = {}

                for detector in detectors:
                    try:
                        scores = _compute_noise_scores(
                            X_tr_s, y0, is_clf=is_clf, detector=detector,
                            groups=g_tr)
                    except Exception as exc:
                        errors[f'{lvl_key}|{space}|{detector}|f{fold_i}|'
                               f'detector'] = f'{type(exc).__name__}: {exc}'
                        continue
                    if scores is None:
                        errors[f'{lvl_key}|{space}|{detector}|f{fold_i}|'
                               f'detector'] = 'the estimator fit failed'
                        continue
                    scores = np.asarray(scores, dtype=float)

                    # Did the detector rank the damaged molecules at the top,
                    # before any arm ran? It costs no model fit.
                    det = detection_scores(cmask, scores, condition)
                    if det is not None:
                        (detection_acc.setdefault(lvl_key, {})
                         .setdefault(space, {})
                         .setdefault(detector, []).append(det))

                    det_cell = space_cell.setdefault(detector, {})
                    # Per-pass scores for the arms that are fitted once because
                    # their cleaned training set does not depend on the rate.
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
                            blind_key = (detector, cleanup)
                            if (not reads_flag_rate(cleanup, is_clf)
                                    and blind_key in blind):
                                # Already fitted at the first rate; the cleaned
                                # training set is the same column, so the same
                                # per-pass score is written here rather than
                                # the same fit being run again.
                                for mname, done in blind[blind_key].items():
                                    (rate_cell.setdefault(arm, {})
                                     .setdefault(mname, {})[pass_key]) = done
                                continue

                            try:
                                cleaned = cleaned_training_set(
                                    cleanup, X_tr_s, y0, scores, rate,
                                    is_clf=is_clf, groups=g_tr,
                                    strength=None, targets=targets)
                            except Exception as exc:
                                errors[f'{lvl_key}|{space}|{detector}|'
                                       f'{rate_key}|{arm}|f{fold_i}|clean'] = (
                                    f'{type(exc).__name__}: {exc}')
                                continue
                            if cleaned is None:
                                errors[f'{lvl_key}|{space}|{detector}|'
                                       f'{rate_key}|{arm}|f{fold_i}|clean'] = (
                                    'the cleanup declined on this pass')
                                continue

                            if family == 'delete':
                                n_deleted_here = _deleted_count(cleaned, n_tr)
                                rm = ~np.asarray(cleaned['keep'], dtype=bool)
                                (deleted_acc.setdefault(lvl_key, {})
                                 .setdefault(space, {})
                                 .setdefault(detector, {})
                                 .setdefault(rate_key, []).append({
                                     'molecules_deleted': int(rm.sum()),
                                     'training_molecules': n_tr,
                                     'widened_molecules_among_the_deleted':
                                         int(np.sum(
                                             np.asarray(cmask, dtype=bool) & rm)),
                                     'widened_molecules': int(np.sum(cmask))}))

                            got = {}
                            for mname in downstream_models:
                                s, err = _fit_score(
                                    model_factories[mname], cleaned, X_va_s,
                                    y_val, metric_fn, metric_name)
                                if s is None:
                                    errors[f'{lvl_key}|{space}|{detector}|'
                                           f'{rate_key}|{arm}|{mname}|'
                                           f'f{fold_i}'] = err
                                    continue
                                got[mname] = s
                                (rate_cell.setdefault(arm, {})
                                 .setdefault(mname, {})[pass_key]) = s
                            if not reads_flag_rate(cleanup, is_clf):
                                blind[blind_key] = got

                        # ── The control, on the same count ──
                        if not n_deleted_here:
                            errors[f'{lvl_key}|{space}|{detector}|{rate_key}|'
                                   f'f{fold_i}|control'] = (
                                'the deletion arm removed no molecule on this '
                                'pass, so there is no count to match at random')
                            continue
                        for arm, family, cleanup, seed in arms:
                            if family != 'delete_at_random':
                                continue
                            ck = (space, n_deleted_here, seed)
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
                                    errors[f'{lvl_key}|{space}|{rate_key}|'
                                           f'{arm}|f{fold_i}|clean'] = (
                                        'the deletion declined on this pass')
                                    random_cache[ck] = {}
                                else:
                                    rm = ~np.asarray(rcleaned['keep'],
                                                     dtype=bool)
                                    (deleted_acc.setdefault(lvl_key, {})
                                     .setdefault(space, {})
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
                                    for mname in downstream_models:
                                        s, err = _fit_score(
                                            model_factories[mname], rcleaned,
                                            X_va_s, y_val, metric_fn,
                                            metric_name)
                                        if s is None:
                                            errors[
                                                f'{lvl_key}|{space}|{rate_key}'
                                                f'|{arm}|{mname}|f{fold_i}'] = err
                                        else:
                                            out[mname] = s
                                    random_cache[ck] = out
                            for mname, s in random_cache[ck].items():
                                (rate_cell.setdefault(arm, {})
                                 .setdefault(mname, {})[pass_key]) = s

            print(f"    level {lvl_key} draw {draw} fold {fold_i} done "
                  f"[{time.time() - t0:.0f}s]")

    results['per_level'] = convert_numpy(acc)
    results['detection'] = convert_numpy({
        lvl: {sp: {d: _mean_detection(rows) for d, rows in by_det.items()}
              for sp, by_det in by_sp.items()}
        for lvl, by_sp in detection_acc.items()})
    results['injury_carried_by_the_flagged_molecules'] = convert_numpy({
        lvl: _sum_injury(rows) for lvl, rows in injury_acc.items()})
    results['molecules_deleted'] = convert_numpy({
        lvl: {sp: {who: {r: _sum_deleted(rows) for r, rows in by_rate.items()}
                   for who, by_rate in by_who.items()}
              for sp, by_who in by_sp.items()}
        for lvl, by_sp in deleted_acc.items()})
    results['molecules_deleted_note'] = (
        'One entry per (level, space, who did the deleting, flag rate). Every '
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
    _print_summary(results, metric_name)
    return results


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
    gains = np.array([(base[k] - got[k]) if lower_is_better
                      else (got[k] - base[k]) for k in shared], dtype=float)
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
                    }
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
                              'downstream model)'),
    }
    return out


def _print_summary(results, metric_name):
    s = results['summary']
    print("\n" + "=" * 70)
    print(f"SUMMARY ({metric_name}; paired per pass, a pass is one draw and "
          f"one fold)")
    if not results.get('hybrid_available', True):
        print("  no hybrid recipe was available; base representation only")
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
                det = drow.get('detection')
                if det and np.isfinite(det.get('detection_average_precision',
                                               float('nan'))):
                    print(f"    {space}/{detector}: ranked the widened "
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
                        help=f'Injected condition (default '
                             f'{DEFAULT_NOISE_CONDITION}). Must be one where a '
                             f'damaged subset exists: '
                             f'{", ".join(DETECTION_IS_DEFINED)}')
    parser.add_argument('--levels', type=str, default=None,
                        help='Comma-separated noise levels. Default: the '
                             'shared grid for this dataset and task type')
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
                             'Overrides the 1 or 5 --mode gives')
    parser.add_argument('--base-rep', type=str, default=None,
                        help='Which representation is the base space. Default: '
                             "pick_noise_base_rep's, and it is recorded either way")
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
    if args.base_rep:
        kw['base_rep'] = args.base_rep

    res = run(data, config, args.output, args.dataset, **kw)
    # The condition changes what the run MEASURED while leaving the dataset and
    # the experiment the same, so it is part of the filename; --tag is appended
    # to it for a second run of the same condition.
    suffix = kw.get('noise_condition') or DEFAULT_NOISE_CONDITION
    if args.tag:
        suffix = f'{suffix}_{args.tag}'
    save_experiment_results(res, args.output, 'n24_delete_or_keep',
                            args.dataset, filename_suffix=suffix)


if __name__ == '__main__':
    main()
