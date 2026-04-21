"""Check whether ginkgo_heparin's reps and augmentations contain NaN.

If any do, complexity_helpers.run_all mutates them via np.nan_to_num,
which changes their content hash and therefore the hybrid_eval cache
filename. That would explain why verify_c02 misses cache entries that
direct load_data (no mutation) finds perfectly.

Usage:
    python tests/scripts/check_rep_nans.py [dataset_name]
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
from experiments._shared import load_data
from experiments.complexity.complexity_helpers import DATASET_REPS


def check(dataset: str) -> None:
    ds_reps = DATASET_REPS.get(dataset)
    if ds_reps is None:
        print(f'ERROR: {dataset} not in DATASET_REPS')
        sys.exit(1)

    print(f'=== {dataset} ===\n')
    data = load_data(dataset, rep_preset='all', rep_names=ds_reps['general'])

    any_dirty = False
    print('\nreps:')
    for rname, X in data['reps'].items():
        n_nan = int(np.isnan(X).sum())
        if n_nan > 0:
            pct = 100 * n_nan / X.size
            print(f'  {rname}: shape={X.shape} {n_nan} NaN ({pct:.2f}%) '
                  f'<-- MUTATED by run_all')
            any_dirty = True
        else:
            print(f'  {rname}: shape={X.shape} clean')

    aug_names = ds_reps.get('augmentation', [])
    missing = [r for r in aug_names if r not in data['reps']]
    if missing:
        aug_data = load_data(dataset, rep_preset='all', rep_names=missing)
        augs = dict(aug_data['reps'])
    else:
        augs = {r: data['reps'][r] for r in aug_names}

    print('\naugmentations:')
    for rname, X in augs.items():
        n_nan = int(np.isnan(X).sum())
        if n_nan > 0:
            pct = 100 * n_nan / X.size
            print(f'  {rname}: shape={X.shape} {n_nan} NaN ({pct:.2f}%) '
                  f'<-- MUTATED by run_all')
            any_dirty = True
        else:
            print(f'  {rname}: shape={X.shape} clean')

    print('')
    if any_dirty:
        print('VERDICT: at least one rep has NaN. complexity_helpers.run_all')
        print('         mutates them via np.nan_to_num, which changes the')
        print('         content hash and therefore the cache filename. This')
        print('         is the likely source of verify_c02 missing cache.')
    else:
        print('VERDICT: all reps/augs are clean. NaN cleanup is NOT the')
        print('         cause. Next suspect: parameter drift in meta '
              '(augmentation_threshold, augmentation_budget, etc.)')


if __name__ == '__main__':
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'ginkgo_heparin'
    check(dataset)
