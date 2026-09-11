#!/usr/bin/env python3
"""How many different things do the noise estimators measure?

`ESTIMATOR_REGISTRY` holds 19 keys for 15 distinct classes — four pairs of
names are one class each — and n06, n11 and n13 each run a subset of those
keys. Nobody has measured whether two estimators give the same per-molecule
ranking. If four of them agree at 0.98 they are one method occupying four
slots in all three files.

What this does: one dataset, one feature space, one fit per estimator. Every
estimator scores every molecule; the per-molecule score vectors are then
Spearman-correlated against each other and the matrix is printed.

It measures redundancy. It cuts nothing — a measured redundancy is a reason to
ask the owner, not a reason to delete.

    python tests/experiments/noise/_probe_estimator_distinctness.py \
        --dataset herg_ki --rep ecfp4
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('KIRBY_LGBM_THREADS', '1')

import sys
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from experiments._shared import load_data, save_experiment_results  # noqa: E402
from kirby.utils.noise_estimation import (  # noqa: E402
    ESTIMATOR_REGISTRY, get_noise_estimator,
)


def distinct_estimator_names():
    """One name per estimator CLASS, with the aliases it also answers to.

    Four classes carry two keys each: 'distance_weighted' /
    'distance_weighted_residual', 'cross_val' / 'cross_val_residual',
    'ensemble' / 'ensemble_disagreement' and 'hybrid' / 'hybrid_noise'. n06
    used the long spellings and n11 the short ones, so the two result files
    did not join on name even though they ran the same code.
    """
    by_class = {}
    for name, cls in ESTIMATOR_REGISTRY.items():
        by_class.setdefault(cls, []).append(name)
    out = {}
    for cls, names in by_class.items():
        canonical = sorted(names, key=len)[0]
        out[canonical] = sorted(names)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', default='herg_ki')
    ap.add_argument('--rep', default='ecfp4',
                    help='the one representation the scores are computed in')
    ap.add_argument('--subsample', type=int, default=0,
                    help='0 = every molecule')
    ap.add_argument('--output', default='tests/results/noise')
    args = ap.parse_args()

    aliases = distinct_estimator_names()
    names = sorted(aliases)
    print(f"{len(ESTIMATOR_REGISTRY)} registry keys, {len(names)} distinct "
          f"estimator classes")

    data = load_data(args.dataset, rep_preset='all', rep_names=[args.rep])
    X = np.asarray(data['reps'][args.rep], dtype=np.float64)
    y = np.asarray(data['labels'], dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    if args.subsample and len(y) > args.subsample:
        rng = np.random.RandomState(0)
        idx = np.sort(rng.choice(len(y), args.subsample, replace=False))
        X, y = X[idx], y[idx]
    X = StandardScaler().fit_transform(X)
    print(f"  {args.dataset}: {X.shape[0]} molecules, {X.shape[1]} features "
          f"in {args.rep}")

    scores = {}
    timings = {}
    failures = {}
    for name in names:
        print(f"  {name}...", end='', flush=True)
        t0 = time.time()
        try:
            est = get_noise_estimator(name)
            s = np.asarray(est.estimate(X, y), dtype=np.float64)
            if len(s) != len(y):
                raise ValueError(f'{len(s)} scores for {len(y)} molecules')
            scores[name] = s
            timings[name] = time.time() - t0
            print(f" {timings[name]:.1f}s")
        except Exception as exc:
            failures[name] = f'{type(exc).__name__}: {exc}'
            print(f" FAILED: {type(exc).__name__}: {exc}")

    ran = sorted(scores)
    n = len(ran)
    matrix = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            a, b = scores[ran[i]], scores[ran[j]]
            if np.std(a) == 0 or np.std(b) == 0:
                continue
            matrix[i, j] = float(spearmanr(a, b).statistic)

    print(f"\n  Spearman between the per-molecule score vectors "
          f"({X.shape[0]} molecules each)")
    width = max(len(r) for r in ran) + 1
    print(' ' * width + ''.join(f'{r[:7]:>8s}' for r in ran))
    for i, r in enumerate(ran):
        row = ''.join('      --' if np.isnan(matrix[i, j])
                      else f'{matrix[i, j]:8.2f}' for j in range(n))
        print(f'{r:<{width}s}{row}')

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if not np.isnan(matrix[i, j]):
                pairs.append({'a': ran[i], 'b': ran[j],
                              'spearman': float(matrix[i, j])})
    pairs.sort(key=lambda p: -abs(p['spearman']))

    print("\n  The ten most-agreeing pairs:")
    for p in pairs[:10]:
        print(f"    {p['spearman']:+.3f}  {p['a']} / {p['b']}")

    for floor in (0.98, 0.95, 0.90):
        above = [p for p in pairs if abs(p['spearman']) >= floor]
        print(f"  pairs agreeing at |rho| >= {floor}: {len(above)} "
              f"of {len(pairs)}")

    results = {
        'dataset': args.dataset,
        'representation': args.rep,
        'n_molecules': int(X.shape[0]),
        'n_features': int(X.shape[1]),
        'metric': 'spearman_between_estimator_score_vectors',
        'task_type': data['task_type'],
        'n_registry_keys': len(ESTIMATOR_REGISTRY),
        'n_distinct_classes': len(names),
        'aliases': aliases,
        'estimators_ran': ran,
        'failures': failures,
        'seconds_per_estimator': timings,
        'spearman_matrix': {ran[i]: {ran[j]: (None if np.isnan(matrix[i, j])
                                              else float(matrix[i, j]))
                                     for j in range(n)} for i in range(n)},
        'pairs_sorted_by_agreement': pairs,
    }
    save_experiment_results(results, args.output,
                            '_probe_estimator_distinctness', args.dataset)
    print(f"\n  written to {args.output}/_probe_estimator_distinctness/"
          f"{args.dataset}.json")


if __name__ == '__main__':
    main()
