#!/usr/bin/env python3
"""Iteration-grade settings for a noise METHOD, so a research loop can close.

⛔ WHY THIS EXISTS. A loop is measure -> change -> RE-MEASURE the same thing ->
keep or revert. A configuration that takes twelve hours cannot close one, so a
hypothesis about it can never be tested. n12 and n21 ran for over twelve hours
on 2026-09-19 and produced no number at all.

What this cuts, beyond `_local_run.shrink`:
    folds       5 -> 3        the outer folds an experiment holds out
    levels      4 -> 2        one clean, one clearly damaging
    detectors   kept          which detector wins IS the question in most files
    cleanups    kept          same
    reps        kept          never cut

⛔ ROWS ARE NEVER CUT. Every molecule of the dataset, always. A mitigation
method that finds wrong labels has nothing to find in a subsample: the detector
is scored against the corrupted SUBSET, and shrinking the dataset shrinks that
subset toward nothing. The owner's standing rule -- "use small datasets, but use
the full set", "a successful method will show nothing with N=100".

THE TRAP, and it has been fallen into once already: KIRBY_SMOKE_N is a ROW CAP
inside load_data (`tests/experiments/_shared.py:3574`), not a verbosity flag.
Setting it to 1 in the belief it was a boolean gate silently reduced caco2 to a
single molecule on 2026-09-18. This module refuses to run with it set.

⛔ NOT A RESULTS CONFIG. A setting that wins here is re-measured at the full
configuration before it is believed. The point is to find WHICH constant to
move, cheaply, not to publish the number.
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'tests', 'experiments', 'noise'))

import _local_run as LR


def shrink_for_iteration(models=None, budgets=None):
    """_local_run's cuts, then the iteration-only ones on top."""
    out = LR.shrink(models=models, budgets=budgets or [50], search_folds=2)
    import experiments.dataset_policy as DP
    # One clean level and one that clearly damages. Two is the fewest that can
    # show a method doing anything; the shape of the curve is a full-run
    # question.
    DP._DEFAULT_NOISE_LEVELS = (0.0, 0.5)
    DP.CENSORING_LEVELS_DETECT = (0.0, 0.25)
    out['levels'] = (0.0, 0.5)
    out['exp_folds'] = 3
    return out


def config(models=None, budgets=None, exp_folds=2):
    c = LR.config(models=models, budgets=budgets or [50], exp_folds=exp_folds)
    return c


def force_experiment_models(exp_ids, models):
    """Fit ONE downstream model inside the experiment, and leave the search alone.

    Two different lists, and cutting the wrong one is expensive. LR.shrink's
    `models` is part of the hybrid search's cache key, so cutting it there
    throws away the precomputed search on disk and pays for a new one. The list
    an experiment fits AFTER the detector has scored the molecules is separate:
    n11, n12, n17, n18, n19, n20, n21 and n23 all read it as
    `kwargs.get('models')` and run_all passes none, so wrapping run() is the
    only way in.

    ⛔ n09 IS NOT ONE OF THEM, and it was named here until 2026-09-21. Grep it:
    the only `models` in n09_lnl_survey.py is `models={'rf': _rf_factory()}` on
    the per-fold feature-mix search. n09 has no downstream roster to cut. What
    it fits after the features are built is one plain network and the twelve
    learning-with-noisy-labels methods, and those twelve are the experiment.
    Passing --exp-models for n09 set a kwarg nothing reads, printed
    "EXPERIMENT DOWNSTREAM MODELS FORCED TO: rf", and changed no fit.

    What it costs: the recovery figure is then one model's, not the roster's.
    The detection figure is untouched -- no model is fitted to produce it.
    """
    from experiments.noise import noise_helpers as NH
    for exp_id in exp_ids:
        m = NH._import_experiment(exp_id)
        _orig = m.run

        def _wrapped(data, cfg, output_dir, dataset_name, _o=_orig, **kwargs):
            kwargs.setdefault('models', list(models))
            return _o(data, cfg, output_dir, dataset_name, **kwargs)
        m.run = _wrapped


def _refuse_row_caps():
    """Stop before loading anything if a row cap is in the environment."""
    n = os.environ.get('KIRBY_SMOKE_N', '').strip()
    if n:
        raise SystemExit(
            f"KIRBY_SMOKE_N={n} is set. It is a ROW CAP inside load_data, not a "
            f"speed flag, and this module runs every molecule of the dataset by "
            f"design -- a detector graded against a corrupted subset has "
            f"nothing to find once the rows are cut. Unset it. Cut folds, "
            f"levels or budgets instead; that is what this module is for.")


def main():
    import argparse
    import time
    _refuse_row_caps()
    p = argparse.ArgumentParser()
    p.add_argument('dataset')
    p.add_argument('experiments', nargs='+')
    p.add_argument('--out', default='/Volumes/seagate/local_noise_run/fast')
    p.add_argument('--condition', default=None)
    # The injected levels, as a fraction of the clean label SD. Two is the
    # default here. One is enough while a single constant is being swept, and
    # under every condition in noise_helpers.DETECTION_IS_DEFINED level 0.0
    # moves no label at all -- average precision is then undefined and the
    # clean and corrupted fits are the same number, so the recovery figure
    # divides by zero. Cutting it costs nothing and halves the run.
    p.add_argument('--levels', default=None,
                   help='comma-separated injected levels, e.g. 0.5')
    p.add_argument('--exp-folds', type=int, default=2,
                   help='outer folds the experiment holds out')
    p.add_argument('--exp-models', default=None,
                   help='downstream models the experiment fits after the '
                        'detector has scored, e.g. rf. Does NOT touch the '
                        'hybrid search cache key.')
    a = p.parse_args()

    # ORDER MATTERS, BOTH WAYS, AND THE TWO PULL AGAINST EACH OTHER.
    #
    # shrink_for_iteration has to run FIRST. LR.sweep_cuts imports each
    # experiment module, and n07/n09/n11/n12/n13 bind DOWNSTREAM_MODELS by
    # `from experiments._shared import ...` at import time. LR.shrink rebinds
    # that name inside _shared, so an experiment imported before the shrink
    # keeps the five-model roster (rf, xgboost, svm, dnn, lightgbm) and fits
    # every one of them.
    #
    # But LR.sweep_cuts sets DP._DEFAULT_NOISE_LEVELS to its OWN four levels
    # (0.0, 0.15, 0.3, 0.75), which overwrites the two this module asked for.
    # So the levels are set LAST, after both.
    print('ITERATION SETTINGS:', shrink_for_iteration())
    print('SWEEP CUTS:', LR.sweep_cuts(set(a.experiments)))
    import experiments.dataset_policy as _DP
    _lv = (tuple(float(x) for x in a.levels.split(','))
           if a.levels else (0.0, 0.5))
    _DP._DEFAULT_NOISE_LEVELS = _lv
    _DP.CENSORING_LEVELS_DETECT = _lv
    print('LEVELS IN FORCE:', _DP._DEFAULT_NOISE_LEVELS)
    if a.exp_models:
        force_experiment_models(a.experiments,
                                [m.strip() for m in a.exp_models.split(',')])
        print('EXPERIMENT DOWNSTREAM MODELS FORCED TO:', a.exp_models)
    from experiments.noise import n12_importance_weighted_distance as _n12
    print('n12 DOWNSTREAM MODELS:', _n12.DOWNSTREAM_MODELS)
    print('EXPERIMENT FOLDS:', a.exp_folds)
    from experiments.noise import noise_helpers as NH
    kw = {'noise_condition': a.condition} if a.condition else {}
    t0 = time.time()
    NH.run_all(dataset_name=a.dataset, output_dir=a.out,
               experiments=a.experiments,
               config=config(exp_folds=a.exp_folds),
               force_rerun=True, **kw)
    print(f"\nTOTAL {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
