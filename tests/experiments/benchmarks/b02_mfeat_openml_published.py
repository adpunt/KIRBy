#!/usr/bin/env python3
"""mfeat on OpenML as published: one SVM run per feature set, OpenML's folds.

The six mfeat feature sets are six OpenML datasets, each with its own task.
All six tasks use the same 10 test folds over the same 2,000 rows (checked
2026-10-26: the split files assign every row to the same fold, and the rows
equal the UCI mfeat files in the same order).

For each task this reruns the best published run of OpenML flow 6952 (the
OpenML study 14 SVM pipeline, sklearn 0.18.1, uploaded by Jan van Rijn):
imputer -> one-hot encoder -> VarianceThreshold(0) -> SVC, with that run's
exact hyperparameters and seed, on that task's 10 folds, and compares the
accuracy per fold with the published per-fold accuracy.

The imputer and one-hot encoder do nothing here: mfeat has no missing values
and the encoder's categorical feature list is empty in every run. The imputer
is kept with each run's strategy; the encoder is left out.

Data: /Volumes/seagate/benchmarks_protocol/openml_mfeat/ (override with
KIRBY_OPENML_MFEAT_DIR): mfeat-<view>.arff (OpenML dataset file) and
task<id>_splits.arff (OpenML task split file).

Output: /Volumes/seagate/benchmarks_protocol/results/b02_mfeat_openml.json,
rewritten after each feature set. --views runs only the named feature sets.

On the laptop with scikit-learn 1.6.1 (2026-10-26), factors, fourier and
karhunen finished. morphological had not finished fitting after 40 minutes
and was stopped; pixel and zernike were not reached.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.io import arff
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

DATA_DIR = Path(os.environ.get('KIRBY_OPENML_MFEAT_DIR',
                               '/Volumes/seagate/benchmarks_protocol/openml_mfeat'))
OUT = Path('/Volumes/seagate/benchmarks_protocol/results/b02_mfeat_openml.json')

# task id -> feature set, best flow-6952 run, its settings, published accuracy
# per test fold (fold 0..9). Copied from openml.org/api/v1/json/run/<run>.
RUNS = {
    12: dict(view='factors', run=6052289, impute='most_frequent',
             svc=dict(kernel='poly', C=0.21667044921218367, gamma=1.3104633832737629,
                      degree=2, coef0=0.5349566824252636, tol=0.049522549538275104,
                      shrinking=True, random_state=18802),
             published=[0.975, 0.98, 0.975, 0.97, 0.975, 0.985, 1.0, 0.98, 0.98, 0.98]),
    14: dict(view='fourier', run=5803533, impute='median',
             svc=dict(kernel='rbf', C=3319.7474187009875, gamma=1.6445829573832509,
                      degree=3, coef0=0.0, tol=0.00025306848936662346,
                      shrinking=True, random_state=22021),
             published=[0.895, 0.885, 0.875, 0.935, 0.86, 0.875, 0.885, 0.87, 0.875, 0.85]),
    16: dict(view='karhunen', run=5262068, impute='most_frequent',
             svc=dict(kernel='poly', C=0.8088835052850308, gamma=0.004642817601620335,
                      degree=2, coef0=0.49718355056027086, tol=4.1890041145990105e-05,
                      shrinking=True, random_state=6615),
             published=[0.98, 0.99, 0.985, 0.985, 0.99, 0.995, 0.98, 0.975, 0.99, 0.985]),
    18: dict(view='morphological', run=6042020, impute='most_frequent',
             svc=dict(kernel='poly', C=2.3291230054074092, gamma=5.177598132465959e-05,
                      degree=2, coef0=0.8581730378264689, tol=0.006005734481271572,
                      shrinking=True, random_state=46620),
             published=[0.695, 0.725, 0.7, 0.77, 0.74, 0.765, 0.75, 0.74, 0.765, 0.735]),
    20: dict(view='pixel', run=5294357, impute='most_frequent',
             svc=dict(kernel='poly', C=0.7310753426416944, gamma=0.020013806918255675,
                      degree=2, coef0=-0.9469767283149837, tol=4.3472522262752264e-05,
                      shrinking=False, random_state=63757),
             published=[0.98, 0.98, 0.985, 0.975, 0.985, 0.985, 0.985, 0.98, 0.985, 0.975]),
    22: dict(view='zernike', run=6049712, impute='mean',
             svc=dict(kernel='rbf', C=3.4456951688701047, gamma=3.198494066760803e-05,
                      degree=3, coef0=0.0, tol=0.015450756240662701,
                      shrinking=False, random_state=7817),
             published=[0.935, 0.905, 0.925, 0.94, 0.915, 0.885, 0.9, 0.865, 0.89, 0.91]),
}


def load_view(view):
    data, meta = arff.loadarff(DATA_DIR / f'mfeat-{view}.arff')
    cols = [c for c in meta.names() if c != 'class']
    X = np.array([[r[c] for c in cols] for r in data], dtype=float)
    y = np.array([r['class'].decode() for r in data])
    return X, y


def load_folds(task):
    """Test fold (0-9) of every row, from the OpenML split file."""
    sp, _ = arff.loadarff(DATA_DIR / f'task{task}_splits.arff')
    fold = np.full(2000, -1)
    for r in sp:
        if r['type'].decode() == 'TEST' and int(r['repeat']) == 0:
            fold[int(r['rowid'])] = int(r['fold'])
    assert (fold >= 0).all()
    return fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--views', default=','.join(s['view'] for s in RUNS.values()),
                    help='comma-separated feature sets, default all six')
    views = ap.parse_args().views.split(',')
    unknown = set(views) - {s['view'] for s in RUNS.values()}
    if unknown:
        raise SystemExit(f'unknown feature sets: {sorted(unknown)}')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for task, spec in RUNS.items():
        if spec['view'] not in views:
            continue
        X, y = load_view(spec['view'])
        fold = load_folds(task)
        accs = []
        for k in range(10):
            tr, te = fold != k, fold == k
            m = make_pipeline(SimpleImputer(strategy=spec['impute']),
                              VarianceThreshold(0.0),
                              SVC(probability=True, cache_size=200, **spec['svc']))
            m.fit(X[tr], y[tr])
            accs.append(float((m.predict(X[te]) == y[te]).mean()))
        mine, pub = float(np.mean(accs)), float(np.mean(spec['published']))
        print(f"task {task} {spec['view']:<13} run {spec['run']}: published "
              f"{pub:.4f}, rerun {mine:.4f} accuracy (10 folds of 200 digits)",
              flush=True)
        print(f"   published per fold {spec['published']}")
        print(f"   rerun per fold     {[round(a, 3) for a in accs]}", flush=True)
        results.append({'task': task, 'view': spec['view'], 'run': spec['run'],
                        'published_per_fold': spec['published'],
                        'rerun_per_fold': accs,
                        'published_mean': pub, 'rerun_mean': mine})
        with open(OUT, 'w') as f:
            json.dump(results, f, indent=1)


if __name__ == '__main__':
    main()
