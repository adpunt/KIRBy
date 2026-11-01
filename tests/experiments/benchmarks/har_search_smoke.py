"""Does the hybrid search run on UCI HAR's 18 feature sets with the minibatch
dnn among the models, and what does one dnn fit cost per pass over the rows.

Written 2026-10-28 as a crash check for the first server run, not a result.
Debug settings: one allocation method, one importance method, three folds,
four budgets. The published protocol (b01_uci_har_published.py) is not run
here; the search sees the 7,352 training windows only.

Data: the UCI HAR Dataset folder (override with KIRBY_HAR_DIR, as in b01).
Output: <out>/har_search_smoke.log lines on stdout and <out>/har_search_smoke.json.

    python tests/experiments/benchmarks/har_search_smoke.py --out <dir>
    python tests/experiments/benchmarks/har_search_smoke.py --out <dir> --skip-timing

Locally the search needs KIRBY_LGBM_THREADS=1 and OMP_NUM_THREADS=1 or it dies
with exit 139 in the baseline evaluation (torch and LightGBM carry two OpenMP
runtimes on macOS). On the server setup.sh sets OMP_NUM_THREADS to the core
count, which is what every other search runs under.

Part A: time DNNClassifier.fit on HAR train (7,352 x 561) and on a synthetic
        350,000 x 72 matrix, 3 passes only, and print the device the dnn picked.
Part B: find_best_hybrid on HAR train, groups = person, models rf/lightgbm/dnn.
"""
import argparse
import json
import os
import sys
import time
import faulthandler
from pathlib import Path

import numpy as np

faulthandler.enable()

HAR_DIR = Path(os.environ.get(
    'KIRBY_HAR_DIR',
    '/Volumes/seagate/benchmarks_protocol/uci_har/UCI HAR Dataset'))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_feature_sets():
    X = np.loadtxt(HAR_DIR / 'train' / 'X_train.txt', dtype=np.float32)
    y = np.loadtxt(HAR_DIR / 'train' / 'y_train.txt', dtype=int) - 1
    subj = np.loadtxt(HAR_DIR / 'train' / 'subject_train.txt', dtype=int)
    names = [l.split()[1] for l in open(HAR_DIR / 'features.txt')]
    # The 18 sets of NEW_DATASETS.md: the part of each name before the first
    # '-', with the seven angle(...) features as their own set. Positional.
    sets = {}
    for i, n in enumerate(names):
        k = 'angle' if n.startswith('angle') else n.split('-')[0]
        sets.setdefault(k, []).append(i)
    reps = {k: X[:, idx] for k, idx in sets.items()}
    return X, y, subj, reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--skip-timing', action='store_true')
    ap.add_argument('--n-folds', type=int, default=3)
    ap.add_argument('--budgets', default='25,50,100,250')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    X, y, subj, reps = load_feature_sets()
    log(f"HAR train {X.shape}, {len(reps)} feature sets, "
        f"{len(np.unique(subj))} people, windows per activity {np.bincount(y).tolist()}")

    from kirby.models.dnn import DNNClassifier, _pick_device
    log(f"dnn device: {_pick_device()}")

    timing = {}
    if not args.skip_timing:
        t0 = time.time()
        m = DNNClassifier()
        m.fit(X, y, groups=subj)
        timing['dnn_fit_har_seconds'] = time.time() - t0
        timing['dnn_har_resubstitution_accuracy'] = float((m.predict(X) == y).mean())
        log(f"A(a) dnn fit on 7,352 x 561: {timing['dnn_fit_har_seconds']:.1f}s, "
            f"resubstitution accuracy {timing['dnn_har_resubstitution_accuracy']:.4f}")

        rng = np.random.default_rng(0)
        Xs = rng.standard_normal((350_000, 72), dtype=np.float32)
        ys = Xs[:, :6].argmax(1).astype(int)
        m2 = DNNClassifier(epochs=3, patience=100)
        t0 = time.time()
        m2.fit(Xs, ys)
        timing['dnn_350k_seconds_per_pass'] = (time.time() - t0) / 3
        log(f"A(b) dnn fit on 350,000 x 72, 3 passes: "
            f"{timing['dnn_350k_seconds_per_pass']:.1f}s per pass, batch 256")
        del Xs, ys, m2

    from kirby.find_best_hybrid import find_best_hybrid
    budgets = [int(b) for b in args.budgets.split(',')]
    t0 = time.time()
    res = find_best_hybrid(
        base_reps=reps, labels=y, groups=subj,
        budgets=budgets,
        models=['rf', 'lightgbm', 'dnn'],
        metric='accuracy', task_type='classification',
        allocation_methods=['performance_weighted'],
        importance_methods=['lightgbm_gain'],
        internal_evaluators=['match'],
        rep_counts=[None],
        n_folds=args.n_folds, cache=False, verbose=True, seed=42,
    )
    search_seconds = time.time() - t0
    log(f"B search done in {search_seconds / 60:.1f} min")

    keep = ('score_mean', 'score_std', 'budget', 'allocation_method', 'importance_method',
            'internal_evaluator', 'actual_dim')
    summary = {
        'search_seconds': search_seconds, **timing,
        'budgets': budgets, 'n_folds': args.n_folds,
        'baselines': res.baselines,
        'best_by_model_budget': {
            m: {int(b): {k: c.get(k) for k in keep} for b, c in d.items()}
            for m, d in res.best_by_model_budget.items()},
    }
    json.dump(summary, open(out / 'har_search_smoke.json', 'w'), indent=1, default=str)
    for mname, d in res.best_by_model_budget.items():
        for b, c in sorted(d.items()):
            log(f"  {mname} budget {b}: accuracy {c.get('score_mean')}")
    for r, d in res.baselines.items():
        log(f"  baseline {r} ({reps[r].shape[1]} features): " +
            ", ".join(f"{k} {v:.4f}" for k, v in d.items() if isinstance(v, float)))
    log(f"written {out / 'har_search_smoke.json'}")


if __name__ == '__main__':
    main()
