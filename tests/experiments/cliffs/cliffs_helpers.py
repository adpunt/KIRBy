#!/usr/bin/env python3
"""The activity-cliff category: what it runs on one dataset, in what order.

ONE CATEGORY, four experiments, in the order they depend on each other.

  cliff_sets              which molecules are in a cliff pair, under every
                          definition, per fold. It writes the per-molecule
                          flags the next two read.
  definition_agreement    each definition against a cliff set it did not
                          produce: the standard ECFP4 Tanimoto cutoffs,
                          MoleculeACE's published rule, and a flag from a
                          published model where one is on disk.
  error_by_cliff_status   held-out prediction error on cliff molecules against
                          non-cliff molecules.
  cliff_aware_cleanup     training on D, D' and D'': does knowing which
                          molecules are on cliffs make a cleanup work better.

WHY IT IS A CATEGORY OF ITS OWN. Her outline: "Make this a separate set of AC
experiments". The similarity suite's s04 and s09 stay as they are.

THE SETTINGS ARE NOT HERE. Every one of her seven is in
cliffs/CLIFF_SETTINGS.json, read at the start of each unit. An experiment whose
settings have no value prints which one it waited for and writes a skip marker,
so the ledger holds it and starts it by itself once the file says.

Run as a unit of the papers ledger:
    python tests/experiments/run_dataset.py cliffs herg_ki
    python tests/experiments/run_dataset.py cliffs herg_ki \\
        --experiments cliff_sets definition_agreement
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent.parent
REPO = TESTS.parent
for _p in (REPO / 'src', TESTS, TESTS / 'experiments'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

#: In dependency order. cliff_sets writes the per-molecule flags the two
#: after it read, so it runs first.
SINGLE_DATASET_EXPERIMENTS = [
    'cliff_sets',
    'definition_agreement',
    'error_by_cliff_status',
    'cliff_aware_cleanup',
]

#: Which unit reads which other unit's output, so the ledger can order them.
AFTER = {
    'definition_agreement': ('cliff_sets',),
    'error_by_cliff_status': (),
    'cliff_aware_cleanup': (),
}

MODULES = {name: f'experiments.cliffs.{name}'
           for name in SINGLE_DATASET_EXPERIMENTS}


def _import_experiment(exp_id):
    return importlib.import_module(MODULES[exp_id])


def _experiment_full_name(exp_id):
    """The directory each unit's result file goes in. The experiment's own
    name: there is no number to prefix, because Adelaide has not named these
    and a number invented here would become one."""
    return exp_id


def rep_names_for(dataset_name):
    """The representations a dataset's cliff definitions are computed on.

    DATASET_REPS[dataset]['general'], which is what the framework searched, so
    a hybrid built here is built from the same pool the rest of the suite uses.
    A dataset with no entry is an error and not a guess: without the list,
    which coefficients ran would depend on whatever the loader happened to
    return.
    """
    from experiments.complexity.complexity_helpers import DATASET_REPS
    from experiments.noise.noise_helpers import REP_LIST_FROM
    source = REP_LIST_FROM.get(dataset_name, dataset_name)
    entry = DATASET_REPS.get(source)
    if entry is None:
        raise ValueError(
            f"no representation list for {dataset_name!r}. DATASET_REPS in "
            f"complexity/complexity_helpers.py has no entry for it, and the "
            f"cliff definitions are computed on that list. Add the entry, or "
            f"leave the dataset out of this category.")
    return list(entry['general'])


def _already_done(output_dir, exp_id, dataset_name, fingerprint, force_rerun):
    """Why this unit may be skipped, or None to run it.

    ⛔ WHY A UNIT HERE NEEDS A RESUME AT ALL. cliff_sets on 1,412 molecules and
    13 representations is 15.8 minutes a fold, which is 2.6 hours over five
    folds and two definition scopes; a unit that recomputes that on every claim
    cannot be rerun on its own, and her ledger handoff says a unit must be.

    THREE WAYS A FILE IS NOT A FINISHED RUN, each checked rather than assumed:
    it holds no measurement (an all-failed run, a refusal, a skip marker), it
    will not parse, or it was written under different folds, labels or
    representations. The last is what `input_fingerprint` records;
    `fingerprint_mismatch` says which field moved.
    """
    from experiments._shared import (experiment_result_is_empty,
                                     fingerprint_mismatch)
    if force_rerun:
        return None
    path = Path(output_dir) / _experiment_full_name(exp_id) / f'{dataset_name}.json'
    if not path.exists():
        return None
    try:
        prev = json.loads(path.read_text())
    except (ValueError, OSError) as exc:
        print(f'  RERUN {exp_id}: {path} could not be read ({exc})')
        return None
    if experiment_result_is_empty(prev, classify_unmarked=True):
        print(f'  RERUN {exp_id}: {path} records no measurement')
        return None
    drift = fingerprint_mismatch(prev.get('input_fingerprint'), fingerprint)
    if drift:
        print(f'  RERUN {exp_id}: {drift}')
        return None
    return f'already done, {path}'


def run_all(dataset_name, output_dir, experiments=None, config=None,
            rep_names=None, pruning_mode='default_pruning', force_rerun=False,
            **kwargs):
    """Load the dataset once and run the activity-cliff experiments on it."""
    # ⛔ NOT `import experiments.dataset_policy` HERE. That statement binds the
    # name `experiments` inside this function, which shadows this function's
    # own `experiments` argument, and the next line that reads the argument
    # gets the module instead of the list.
    from experiments._shared import input_fingerprint, load_data
    importlib.import_module('experiments.dataset_policy')   # registers it

    from experiments.cliffs import settings as S

    if experiments is None:
        experiments = list(SINGLE_DATASET_EXPERIMENTS)
    unknown = [e for e in experiments if e not in MODULES]
    if unknown:
        raise ValueError(f'unknown cliff experiments {unknown}; they are '
                         f'{SINGLE_DATASET_EXPERIMENTS}')
    config = dict(config or {})
    # The outer fold count, which decides the splits every unit holds out on.
    # The same splits the per-fold searches are filed under
    # (noise_helpers.noise_outer_folds), so a hybrid searched on a fold's
    # training rows is scored on that fold's held-out rows. KIRBY_NOISE_N_FOLDS
    # overrides it and KIRBY_NOISE_ONLY_FOLDS restricts which folds run, both
    # read inside noise_outer_folds.
    if 'n_splits' not in config:
        from kirby.find_best_hybrid import DEFAULT_N_FOLDS
        config['n_splits'] = int(DEFAULT_N_FOLDS)

    if rep_names is None:
        rep_names = rep_names_for(dataset_name)

    print('\n' + '=' * 70)
    print(f'ACTIVITY CLIFF EXPERIMENTS: {dataset_name.upper()}')
    print('=' * 70)
    print(f'Representations ({len(rep_names)}): {", ".join(rep_names)}')
    print(f'Experiments: {experiments}')
    print(f'Output: {output_dir}')
    stored = S.read()
    miss = S.unset(stored)
    print(f'Settings: {S.path()}')
    if miss:
        print(f'  {len(miss)} setting(s) with no value: {miss}')
        print('  an experiment that needs one of them writes a skip marker '
              'and names it')

    t0 = time.time()
    data = load_data(dataset_name, rep_preset='all', rep_names=rep_names)
    print(f'Loaded {len(np.asarray(data["labels"]).ravel())} molecules and '
          f'{len(data.get("reps") or {})} representations in '
          f'{time.time() - t0:.1f}s')

    # What a resumed run compares an existing result file against, so a file
    # written before the folds, the labels or the representations changed is
    # recomputed rather than kept.
    fingerprint = input_fingerprint(data, config)

    results = {}
    for exp_id in experiments:
        print('\n' + '-' * 70)
        t = time.time()
        skip = _already_done(output_dir, exp_id, dataset_name, fingerprint,
                             force_rerun)
        if skip is not None:
            print(f'  RESUME {exp_id}: {skip}')
            results[exp_id] = {'resumed': True, 'why': skip}
            continue
        try:
            mod = _import_experiment(exp_id)
            results[exp_id] = mod.run(data, config, output_dir, dataset_name,
                                      **kwargs)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            results[exp_id] = {'error': f'{type(exc).__name__}: {exc}',
                               'traceback': traceback.format_exc()}
            print(f'  {exp_id} FAILED: {type(exc).__name__}: {exc}', flush=True)
        print(f'  {exp_id}: {time.time() - t:.1f}s', flush=True)
    return results
