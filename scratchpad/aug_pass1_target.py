#!/usr/bin/env python3
"""Pass 1 for the augmentation-stage workstream: establish THE TARGET.

Scores every rep in a dataset's pool, solo, on the dataset's REAL metric and
the same folds the arms will use. Two jobs:

  1. Fix the best rep INDEPENDENTLY of the folds the deltas are later read on.
     Both prior harnesses picked their target by argmax over the same folds they
     scored it on, which is a selection they then reported as a baseline.
  2. Say whether the dataset is a "signal spread across reps" case or a "one
     dominant rep" case. Nobody has verified that for any of the four test
     datasets, and it decides which scenario each dataset can actually test.

No augmentation, no hybrid, no search — solo reps only. Read-only w.r.t. the
repo; writes one JSON to the scratch dir.

Usage:
    KIRBY_CACHE_DIR=/Volumes/seagate/kirby_cache \
    python scratchpad/aug_pass1_target.py --dataset herg_ki --n-folds 5 --model lightgbm
"""
import argparse
import json
import os
import sys
import time

# LOCAL macOS ONLY — must run BEFORE experiments._shared is imported.
# load_data's import chain grabs an OpenMP runtime that conflicts with the one
# lightgbm loads later, so the first lightgbm fit AFTER load_data segfaults the
# interpreter (exit 139, no traceback, no output). Touching lightgbm's native
# library first makes it win the load and the crash disappears. Verified:
# rdkit-then-fit is fine, lightgbm alone is fine, load_data-then-fit CRASHES,
# warmup-then-load_data-then-fit is fine. OMP_NUM_THREADS=1 and
# KMP_DUPLICATE_LIB_OK=TRUE both fail to fix it. Not a server problem, so this
# stays out of the repo modules.
try:
    import numpy as _np
    import lightgbm as _lgb
    _lgb.LGBMRegressor(n_estimators=2, verbose=-1).fit(
        _np.zeros((8, 2)), _np.arange(8.0))
except Exception as _e:  # pragma: no cover
    print(f"  [warmup] lightgbm warm-up skipped: {_e}")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))

import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_model_factories, cv_split, get_domain_metric,
)
from experiments.complexity.complexity_helpers import DATASET_REPS

N_SEEDS = 3


def _seeded(factory, seed):
    m = factory()
    try:
        if 'random_state' in m.get_params():
            m.set_params(random_state=seed)
    except Exception:
        pass
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--n-folds', type=int, default=5)
    ap.add_argument('--model', default='lightgbm')
    ap.add_argument('--n-seeds', type=int, default=N_SEEDS)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    cache = os.environ.get('KIRBY_CACHE_DIR')
    if not cache or 'seagate' not in cache:
        print(f"REFUSING TO RUN: KIRBY_CACHE_DIR={cache!r}. "
              f"The default ~/.cache/kirby is missing pool reps and would start "
              f"regenerating heavy embeddings. Set it to the seagate cache.")
        return 2
    print(f"KIRBY_CACHE_DIR={cache}")

    pool = DATASET_REPS[args.dataset]['general']
    aug_names = DATASET_REPS[args.dataset].get('augmentation') or []
    print(f"{args.dataset}: pool={len(pool)} reps, augmentation={len(aug_names)} reps")

    # Load base pool AND augmentation reps. load_data auto-filters to 'general'
    # and hard-sets augmentations=None on every branch, so the aug names must be
    # requested explicitly or they are silently absent — the exact failure that
    # made the previous attempt's use-case-A arm a no-op.
    data = load_data(args.dataset, rep_names=pool + aug_names)
    reps = data['reps']
    labels = data['labels']
    groups = data.get('groups')
    task_type = data['task_type']
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    sign = -1 if lower_is_better else 1
    print(f"  metric={metric_name} lower_is_better={lower_is_better} "
          f"n={len(labels)} groups={len(np.unique(groups)) if groups is not None else None}")

    missing = [r for r in pool + aug_names if r not in reps]
    if missing:
        print(f"  ERROR: reps missing after load: {missing}")
        return 2
    assert len(set(pool) & set(aug_names)) == 0, "base/aug overlap"
    for r in aug_names:
        assert reps[r].shape[1] > 0, f"augmentation rep {r} loaded empty"
    print(f"  aug widths: {[(r, reps[r].shape[1]) for r in aug_names]}")

    factory = get_model_factories([args.model], task_type=task_type,
                                  n_samples=len(labels))[args.model]

    X_split = reps[pool[0]]
    splits = list(cv_split(groups, args.n_folds, X_split, labels))
    print(f"  folds: {len(splits)}  sizes={[len(v) for _, v in splits]}")

    rows = {}
    t0 = time.time()
    # Score the augmentation reps solo too. They have never been scored solo on
    # any of these datasets — it is not an admission criterion (the pool is
    # judged by marginal gain, never solo) but it says whether a block carries
    # anything at all.
    for rname in pool + aug_names:
        X = reps[rname]
        fold_means = []
        for tr, va in splits:
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[tr])
            Xva = sc.transform(X[va])
            vals = []
            for s in range(args.n_seeds):
                m = _seeded(factory, 42 + s)
                m.fit(Xtr, labels[tr])
                vals.append(float(metric_fn(labels[va], m.predict(Xva))))
            fold_means.append(float(np.mean(vals)))
        rows[rname] = {
            'mean': float(np.mean(fold_means)),
            'std': float(np.std(fold_means)),
            'folds': fold_means,
            'dim': int(X.shape[1]),
            'tier': 'augmentation' if rname in aug_names else 'base',
        }
        print(f"    {rname:<20s} {np.mean(fold_means):+.4f} "
              f"+-{np.std(fold_means):.4f}  {X.shape[1]:>5d}D  "
              f"[{rows[rname]['tier']}]  ({time.time()-t0:.0f}s)", flush=True)

    base_rows = {k: v for k, v in rows.items() if v['tier'] == 'base'}
    order = sorted(base_rows, key=lambda r: -sign * base_rows[r]['mean'])
    best, second, worst = order[0], order[1], order[-1]
    gap = abs(base_rows[best]['mean'] - base_rows[second]['mean'])
    spread = abs(base_rows[best]['mean'] - base_rows[worst]['mean'])
    # A dominant rep is one whose lead over the runner-up is larger than the
    # fold-to-fold noise of the leader itself. Otherwise the top is flat and the
    # dataset cannot test "one dominant rep slightly improved".
    dominant = gap > base_rows[best]['std']

    print(f"\n  TARGET = {best}  {base_rows[best]['mean']:+.4f} "
          f"({base_rows[best]['dim']}D)")
    print(f"  runner-up {second} {base_rows[second]['mean']:+.4f} -> gap {gap:.4f}, "
          f"leader fold-sd {base_rows[best]['std']:.4f}")
    print(f"  pool spread best..worst = {spread:.4f}")
    print(f"  VERDICT: {'DOMINANT-REP (scenario 2 testable)' if dominant else 'FLAT TOP (scenario 1 case)'}")

    out = {
        'dataset': args.dataset, 'metric': metric_name,
        'lower_is_better': lower_is_better, 'model': args.model,
        'n_folds': len(splits), 'n_seeds': args.n_seeds, 'n_samples': len(labels),
        'reps': rows, 'ranking': order, 'best_rep': best, 'runner_up': second,
        'gap_to_runner_up': gap, 'leader_fold_sd': base_rows[best]['std'],
        'pool_spread': spread, 'dominant_rep': bool(dominant),
    }
    path = args.out or os.path.join(
        os.environ.get('KIRBY_SCRATCH', '/tmp'), f'pass1_{args.dataset}.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  -> {path}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
