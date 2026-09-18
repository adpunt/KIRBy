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
Until then this file read `sorted(reps)[0]` and every number in it came from
whichever representation sorted first, with nothing in the output saying so.

The base representation is still one matrix, now chosen by pick_noise_base_rep,
but it is now written to `base_rep` and read against the hybrid rather than
standing alone.

THE HYBRID IS BUILT ON THE TRAINING MOLECULES OF A FOLD, AT THEIR UNCLEANED
LABELS. It is built ONCE per fold and every cleanup, flag rate and detector
shares it, so the only thing differing between two cleanups is what the
downstream fit was handed. The held-out molecules' labels reach no feature
ranking and no budget split. Whether a recipe searched AFTER the cleaning would
do better is a different question and it is n20's.

The detector is re-run in each space. It reads features, so the molecules it
flags in the hybrid are not the molecules it flags in the base representation --
which is the whole reason a cleanup can move between the two.

A RUN WITH NO RECIPE STILL PRODUCES EVERY BASE NUMBER. If the pool returns
nothing, `hybrid_available` is written False, `spaces_run` says which space ran,
and every cleanup is still measured in the base representation. Dropping the run
would lose real answers because one of two spaces is missing.

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

import time
import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_model_factories, DOWNSTREAM_MODELS,
    get_best_search_for_downstream,
    top_base_reps, top_hybrids, select_model_rep_pairs, hybrid_deltas,
    pick_noise_base_rep,
)
# One per-fold hybrid build, shared with n17 and n19. It lived in n19 until
# 2026-09-17.
from experiments.noise.noise_helpers import hybrid_for_fold
# The four cleanups are n07's, called directly rather than copied. n07 injects
# noise unless it is handed a training label vector, and it is handed one here --
# the labels exactly as the loader produced them. So the same code runs, over the
# real labels, with nothing corrupted.
from experiments.noise.n07_remediation import _evaluate_fold
from kirby.utils.noise_estimation import ESTIMATOR_REGISTRY

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

# Which feature space each cleanup is measured in. Both, always, unless the pool
# returns no recipe -- then the base representation alone, and the result says so.
SPACES = ('base', 'hybrid')

# Which searches the recipe may come from. Same as n02, n06, n07, n11, n12, n13,
# n17 and n19 since 2026-09-12: a search whose features were chosen to predict
# the label, and one whose features were chosen so that the distance they define
# predicts each molecule's activity from its five nearest neighbours. Three of
# the four cleanups here overwrite or reweight a label using its neighbours'
# labels, which is the quantity that second objective optimises.
POOL_SUBSET = 'predictive_and_similarity'


def _suspicion(detector, X_tr_s, y_tr, is_clf, groups=None):
    """One suspicion score per training row, or None if the detector failed.

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
    import inspect
    offered = {'k': DETECTOR_K, 'metric': None, 'groups': groups}
    if is_clf:
        offered['is_clf'] = True
    try:
        cls = ESTIMATOR_REGISTRY[detector]
        accepts = set(inspect.signature(cls.__init__).parameters)
        kwargs = {k: v for k, v in offered.items() if k in accepts}
        est = cls(**kwargs)
        return np.asarray(est.estimate(X_tr_s, y_tr), dtype=float)
    except Exception as exc:
        print(f"      {detector} failed: {type(exc).__name__}: {exc}")
        return None


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

    downstream_models = list(kwargs.get('models') or DOWNSTREAM_MODELS)
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
    # The base space exists to be the thing the hybrid is read against, and it
    # is RECORDED rather than left implicit. Until 2026-09-17 it was the only
    # space and the result file did not name it.
    # pick_noise_base_rep, not sorted(reps)[0]. The alphabetical pick is the
    # fallback that function was written to replace: it makes the control arm a
    # matter of spelling, and it can hand the control slot to a structure-tuned
    # sim_* recipe, which is a treatment this study built. n11, n17 and n21
    # already call it; these four did not. Changed 2026-09-18.
    base_rep_name = (kwargs.get('base_rep')
                     or pick_noise_base_rep(reps, dataset_name))
    X_source = reps[base_rep_name]

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
        'base_rep': base_rep_name,
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
            'The hybrid is built once per fold from that fold\'s training '
            'molecules at their UNCLEANED labels, and every cleanup, flag rate '
            'and detector shares it. No cleanup reaches the feature ranking or '
            'the budget split. Whether a recipe searched after the cleaning '
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

    splits = list(cv_split(groups, n_folds, X_source, labels, mode=mode))
    print(f"  {len(labels)} molecules, {len(splits)} folds, "
          f"models {downstream_models}")
    print(f"  flag rates {flag_rates}, detectors {detectors}")
    print(f"  base representation: {base_rep_name}")

    # ── The hybrid recipe, fetched once ──
    # The pool is keyed on the dataset as the precompute built it, so the lookup
    # is handed the representations, labels and fold grouping exactly as n02,
    # n11, n12, n13, n17 and n19 hand them over. What comes back is a recipe --
    # a representation set, an allocation method, an importance method, a budget
    # and an internal evaluator -- and the features are rebuilt per fold below.
    bp = None
    # Bound before the try so the top-2 block below can test it. Until
    # 2026-09-18 nothing read `search` and an exception here left the name
    # unbound; reading it would have raised UnboundLocalError on the one path
    # this file is built to survive -- a dataset the precompute never covered.
    search = None
    try:
        search, meta = get_best_search_for_downstream(
            reps, labels, groups,
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

    if bp is None:
        # Every cleanup still runs, in the base representation alone. Dropping
        # the run would lose every base-space number because one of two spaces
        # is missing.
        spaces = ('base',)
        results['hybrid_available'] = False
        results['hybrid_unavailable_reason'] = (
            'no recipe came back from the precomputed pool for subset '
            f'{POOL_SUBSET!r}, so the hybrid space could not be built. Every '
            'cleanup ran in the base representation only.')
        print(f"  no hybrid recipe from pool subset {POOL_SUBSET}; "
              f"base representation only")
    else:
        spaces = tuple(SPACES)
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
        print(f"  hybrid recipe: {len(results['hybrid_reps'])} representations, "
              f"budget {results['hybrid_budget']}, searched under objective "
              f"{meta['objective']} (pool subset {meta['pool_subset']})")
    results['spaces_run'] = list(spaces)

    # ── What each downstream model would have picked for itself ──
    # Added 2026-09-18, ADDITIVE. Nothing above or below moves: `base_rep` is
    # still the one pick_noise_base_rep names, `spaces_run` is still the same 2
    # spaces, the hybrid is still the 1 recipe fetched for
    # downstream_models[0], and every fold-win count in 'per_space' is still
    # measured in those 2 spaces. These 4 keys sit beside them.
    #
    # What was absent without them: this file scores 5 downstream models on 1
    # base representation and 1 hybrid recipe, and the result file recorded
    # neither which representations nor which configurations the other 4
    # downstream models rank highest. A reader could not tell whether a model
    # lost because the cleanup did nothing or because it was scored on another
    # model's recipe.
    #
    # 0 new model fits. All four read rows the search already stored:
    # search.baselines for the representations, search.all_results for the
    # configurations.
    results['top2_base_reps'] = {}
    results['top2_hybrids'] = {}
    results['top2_pairs_by_model'] = {}
    results['hybrid_deltas_by_model'] = {}
    if search is None:
        results['top2_unavailable_reason'] = (
            'the pool returned no SearchResult, so there are no stored rows to '
            'rank. Every cleanup number in this file stands.')
    else:
        for _m in downstream_models:
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
        try:
            results['top2_pairs_by_model'] = _drop_blueprint(
                select_model_rep_pairs(search, models=list(downstream_models),
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
    folds_ran = {s: {d: {r: 0 for r in flag_rates} for d in detectors}
                 for s in spaces}
    hybrid_failures = []

    t0 = time.time()
    for fold_i, (train_idx, val_idx) in enumerate(splits):
        y_train = labels[train_idx]
        y_val = labels[val_idx]

        # ── The feature spaces for this fold ──
        # The base representation is scaled on this fold's training molecules.
        # The hybrid is built ONCE, here, from those same training molecules at
        # their UNCLEANED labels, and every cleanup below shares it.
        fold_spaces = {}
        # This fold's training scaffold groups. Bound here and not inside the
        # hybrid branch below, because the detector reads them too and a run
        # with no hybrid recipe still has to reach them.
        g_tr = (np.asarray(groups)[train_idx]
                if groups is not None else None)
        scaler = StandardScaler()
        fold_spaces['base'] = (
            scaler.fit_transform(np.asarray(X_source[train_idx],
                                            dtype=np.float64)),
            scaler.transform(np.asarray(X_source[val_idx], dtype=np.float64)))
        if 'hybrid' in spaces:
            X_tr_h, X_va_h = hybrid_for_fold(
                {k: np.asarray(v[train_idx], dtype=np.float64)
                 for k, v in reps.items()},
                {k: np.asarray(v[val_idx], dtype=np.float64)
                 for k, v in reps.items()},
                y_train, bp, g_tr,
                model_factories[downstream_models[0]](),
                metric_fn, lower_is_better)
            if X_tr_h is None:
                hybrid_failures.append(int(fold_i))
            else:
                sc_h = StandardScaler()
                fold_spaces['hybrid'] = (sc_h.fit_transform(X_tr_h),
                                         sc_h.transform(X_va_h))

        for space in spaces:
            if space not in fold_spaces:
                continue
            X_tr_s, X_va_s = fold_spaces[space]

            # The score to beat, IN THIS SPACE. One fit per (fold, space,
            # model), shared by every cleanup, every flag rate and every
            # detector — they all have to beat the same number or the
            # comparison is not paired.
            base_score = {}
            for mname in downstream_models:
                sc = _evaluate_fold(
                    X_tr_s, X_va_s, y_train, y_val, 'no_noise', 0.0,
                    model_factories[mname], metric_fn, is_clf=is_clf,
                    metric_name=metric_name, dataset_name=dataset_name,
                    precomputed_y_tr=y_train)
                base_score[mname] = float(sc) if sc is not None else float('nan')

            for detector in detectors:
                # Re-run per space. The detector reads features, so the
                # molecules it flags in the hybrid are not the ones it flags in
                # the base representation.
                scores = _suspicion(detector, X_tr_s, y_train, is_clf,
                                    groups=g_tr)
                if scores is None:
                    continue
                for slot, rate in enumerate(flag_rates):
                    folds_ran[space][detector][rate] += 1
                    for mname in downstream_models:
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
                                # ⛔ REQUIRED BY THE THREE MODEL-BASED
                                # CORRECTIONS. oof_smooth, gp_smooth and
                                # quantile_clip each fit a model on part of the
                                # training rows and read the target off the
                                # rest. Without the scaffold groups that inner
                                # split is random, so a molecule's own scaffold
                                # family trains the model that says what its
                                # label should have been -- which is the leak
                                # cleaned_training_set's docstring warns about.
                                train_groups=g_tr)
                            acc[space][detector][rate][mname][cleanup].append(
                                float(sc) if sc is not None else float('nan'))
        print(f"    fold {fold_i}: [{time.time() - t0:.0f}s]")

    if hybrid_failures:
        # Named rather than left as a shorter list of folds. Those folds still
        # carry every base-space number.
        results['folds_with_no_hybrid'] = hybrid_failures
        print(f"  hybrid could not be built on folds {hybrid_failures}; "
              f"their base-representation numbers stand")

    # ── Fold win counts, which are the headline ──
    for space in spaces:
        space_block = {}
        for detector in detectors:
            space_block[detector] = {}
            for rate in flag_rates:
                block = {'n_folds_ran': folds_ran[space][detector][rate],
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

    results['summary'] = _summarise(results, downstream_models, flag_rates,
                                    detectors, spaces, metric_name,
                                    cleanups=CLEANUPS)
    _print_summary(results, metric_name)

    # (results, output_dir, experiment_id, dataset) — see the note in n17; the
    # arguments were rotated here the same way and with the same two effects.
    save_experiment_results(results, output_dir, 'n18_clean_the_real_labels',
                            dataset_name)
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
    if 'hybrid' in spaces and 'base' in spaces:
        out['sweeps_by_space'] = {
            c: {
                'settings_winning_every_fold_in_base':
                    out['per_space']['base']['per_cleanup'][c].get(
                        'settings_winning_every_fold', 0),
                'settings_winning_every_fold_in_hybrid':
                    out['per_space']['hybrid']['per_cleanup'][c].get(
                        'settings_winning_every_fold', 0),
                'settings_scored_in_base':
                    out['per_space']['base']['per_cleanup'][c].get(
                        'settings_scored', 0),
                'settings_scored_in_hybrid':
                    out['per_space']['hybrid']['per_cleanup'][c].get(
                        'settings_scored', 0),
            }
            for c in cleanups}
        out['sweeps_by_space_note'] = (
            'How many settings each cleanup won EVERY fold at, in the base '
            'representation and in the hybrid, out of how many settings were '
            'scored in each. A setting is one (detector, flag rate, model).')
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
    print("\n  One setting winning on one fold is not a result. Read the counts.")
    if 'sweeps_by_space' in s:
        print("  'sweeps_by_space' in the result file holds the sweep counts in "
              "both spaces side by side.")


def main():
    parser = create_experiment_parser(
        'N18: does cleaning help on the labels as they come')
    parser.add_argument('--flag-rates', type=str, default=None,
                        help='Comma-separated shares of rows to treat as suspect')
    parser.add_argument('--base-rep', type=str, default=None,
                        help='Which representation is the base space. Default: '
                             'pick_noise_base_rep\'s, and it is recorded either way')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, rep_preset=args.reps)
    kw = {}
    if args.flag_rates:
        kw['flag_rates'] = [float(x) for x in args.flag_rates.split(',')]
    if args.base_rep:
        kw['base_rep'] = args.base_rep
    run(data, config, args.output, args.dataset, **kw)


if __name__ == '__main__':
    main()
