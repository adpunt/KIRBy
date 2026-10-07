"""Build one outer fold's per-fold searches for a dataset, before any
experiment on it runs.

    python tests/experiments/orchestrate/search_unit.py DATASET --fold 2

It loads the dataset the way the suites do (noise_helpers.load_noise_data:
the pruned representations, the NaN guard, the augmentation representations)
and calls _shared.fold_neighbour_space_recipes on that fold's training rows.
That function runs the three per-fold searches every suite reads back
through the result store -- general, general_distance, and general with the
similarity objective -- on the dataset's decided grid, filed under the
namespace ocv{n}f{fold}s{seed} that every experiment asks for. Nothing is
fitted beyond the searches; no result file is written.

Why a unit of its own: the ledger gives each fold to one job, so five jobs
build five folds at once in five result stores, and the experiments that
follow read instead of fit. Two experiments building one fold's search at
the same time would share one store (KIRBY_SEARCH_REVERSE spreads them, but
the store is single-writer).

The outer split is kirby.find_best_hybrid.nested_cv_outer_folds with
DEFAULT_N_FOLDS and seed 42, the split noise_outer_folds and c02 use. The
federated suite searches with its own binarised labels and is not covered.

A dataset in plan.MOLECULE_MEAN_SEARCH_DATASETS (herg_ic50; Adelaide
2026-10-07) is searched on one row per molecule instead of the loader's one
row per measurement: noise_helpers.molecule_mean_data, each molecule's mean
label, its first row's features and scaffold group. The folds are
nested_cv_outer_folds over those molecule rows, so every measurement of a
molecule is on one side of every fold, and the three searches are filed under
noise_helpers.MOLECULE_MEAN_SEARCH_SUFFIX. Those are the rows, folds and name
n17, n19, n26 and n29 search under, so they read these searches back.
"""
import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent.parent
REPO = TESTS.parent
for p in (REPO / 'src', TESTS, TESTS / 'experiments'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('dataset')
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--n-folds', type=int, default=None,
                    help='outer folds (default DEFAULT_N_FOLDS)')
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args(argv)

    import numpy as np
    from kirby.find_best_hybrid import DEFAULT_N_FOLDS, nested_cv_outer_folds
    from experiments.noise.noise_helpers import (
        load_noise_data, molecule_mean_data, MOLECULE_MEAN_SEARCH_SUFFIX)
    from experiments._shared import fold_neighbour_space_recipes
    from experiments.orchestrate.plan import MOLECULE_MEAN_SEARCH_DATASETS

    n = int(a.n_folds or DEFAULT_N_FOLDS)
    if not 0 <= a.fold < n:
        sys.exit(f'--fold must be in 0..{n - 1}')
    t0 = time.time()
    data, rep_names = load_noise_data(a.dataset)
    suffix = None
    if a.dataset in MOLECULE_MEAN_SEARCH_DATASETS:
        n_rows = len(data['labels'])
        data = molecule_mean_data(data, a.dataset)
        if data is None:
            sys.exit(f'{a.dataset}: the loader exposes no repeat structure, '
                     f'so there are no molecule rows to search')
        suffix = MOLECULE_MEAN_SEARCH_SUFFIX
        print(f"{a.dataset}: {n_rows} measurement rows -> {len(data['labels'])} "
              f"molecule rows (mean label); searches filed under {suffix!r}")
    labels = np.asarray(data['labels'])
    folds = nested_cv_outer_folds(data['reps'], labels, data.get('groups'),
                                  n, a.seed, data.get('task_type') or 'regression')
    tr, va = folds[a.fold]
    print(f"\n{a.dataset}: per-fold searches, fold {a.fold} of {n} "
          f"(train {len(tr)}, held out {len(va)}, {len(rep_names)} representations)")
    out = fold_neighbour_space_recipes(
        data['reps'], labels, data.get('groups'),
        train_idx=np.asarray(tr), fold_i=a.fold, n_outer_folds=n,
        dataset_name=a.dataset, augmentations=data.get('augmentations'),
        task_type=data.get('task_type') or 'auto', seed=a.seed, verbose=True,
        condition_suffix=suffix)
    failed = [m for m, e in out.items() if e is None]
    print(f"\n{a.dataset} fold {a.fold}: searches stored in "
          f"{time.time() - t0:.0f}s; models {sorted(out)}; failed {failed}")
    for m, e in out.items():
        if e is None:
            continue
        unavail = e.get('unavailable') or {}
        if unavail:
            print(f"  {m}: unavailable {unavail}")
    if failed and len(failed) == len(out):
        sys.exit(1)


if __name__ == '__main__':
    main()
