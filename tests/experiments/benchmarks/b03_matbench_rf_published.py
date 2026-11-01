#!/usr/bin/env python3
"""Matbench v0.1 as published: the leaderboard's random forest, 'RF-SCM/Magpie'.

Reruns the leaderboard entry matbench_v0.1_rf (Dunn et al. 2020) on Matbench's
own 5 test folds and compares the MAE per fold with the published MAE per fold.

The published pipeline (benchmarks/matbench_v0.1_rf/run.py and info.json in
github.com/materialsproject/matbench, run inside automatminer 1.0.3):
  features  magpie (matminer ElementProperty 'magpie' preset) on every task,
            plus SineCoulombMatrix(flatten=True) on structure tasks, both
            fitted on the training rows (automatminer preset 'debug')
  cleaning  DataCleaner(max_na_frac=0.01, feature_na_method='mean',
            na_method_fit='drop', na_method_transform='mean'). What that does,
            read from automatminer's DataCleaner.handle_na:
            - infinite values become missing
            - a feature missing in every training compound is dropped
            - a feature missing in more than 1% of training compounds is
              filled with its training mean (not dropped)
            - a training compound still missing any value is dropped
            - a test value still missing is filled with the mean of that
              feature over the TEST compounds (pandas fillna on the
              transformed frame); a feature missing in every test compound
              gets the training mean
            info.json says features over 1% missing were dropped; the code
            fills them. The code is followed here.
  reduction none
  model     RandomForestRegressor(n_estimators=500), every other setting
            sklearn's default; no seed was recorded
The cleaning is reimplemented here from those four settings, not imported
from automatminer (whose pins do not install beside current matminer).

published_rf() is the protocol on any feature matrix, so a hybrid's features
can be scored the same way as the published features.

Data: Matbench json.gz under MATMINER_DATA (default
/Volumes/seagate/matbench/data). Published scores:
/Volumes/seagate/benchmarks_protocol/matbench/rf/results.json.gz.

Output: /Volumes/seagate/benchmarks_protocol/results/b03_matbench_rf.json,
rewritten after each task.

    /Volumes/seagate/matbench/venv/bin/python \
        tests/experiments/benchmarks/b03_matbench_rf_published.py
    ... --tasks matbench_jdft2d --folds 0,1
"""
import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))

PROTOCOL_DIR = Path('/Volumes/seagate/benchmarks_protocol/matbench')
PUBLISHED = PROTOCOL_DIR / 'rf' / 'results.json.gz'
OUT = Path('/Volumes/seagate/benchmarks_protocol/results/b03_matbench_rf.json')
TASKS = ['matbench_jdft2d', 'matbench_phonons', 'matbench_expt_gap',
         'matbench_dielectric', 'matbench_log_gvrh', 'matbench_perovskites']
MAX_NA_FRAC = 0.01
N_ESTIMATORS = 500


def published_mae(task):
    d = json.load(gzip.open(PUBLISHED))
    r = d['tasks'][task]['results']
    return [float(r[f'fold_{k}']['scores']['mae']) for k in range(5)]


def clean(X_tr, y_tr, X_te):
    """automatminer DataCleaner with the leaderboard's four settings.

    Returns (X_tr, y_tr, X_te, features dropped, training compounds dropped).
    """
    keep = ~np.isnan(X_tr).all(axis=0)
    X_tr, X_te = X_tr[:, keep].copy(), X_te[:, keep].copy()
    problem = np.isnan(X_tr).mean(axis=0) > MAX_NA_FRAC
    train_mean = np.nanmean(X_tr, axis=0)
    X_tr[:, problem] = np.where(np.isnan(X_tr[:, problem]),
                                train_mean[problem], X_tr[:, problem])
    rows = ~np.isnan(X_tr).any(axis=1)
    X_tr, y_tr = X_tr[rows], y_tr[rows]
    all_nan_te = np.isnan(X_te).all(axis=0)
    X_te[:, all_nan_te] = train_mean[all_nan_te]
    with np.errstate(all='ignore'):
        test_mean = np.nanmean(X_te, axis=0)
    X_te = np.where(np.isnan(X_te), test_mean, X_te)
    return X_tr, y_tr, X_te, int((~keep).sum()), int((~rows).sum())


def published_rf(X_tr, y_tr, X_te, seed=42, n_jobs=-1):
    """The leaderboard protocol on any feature matrix. NaN = missing.

    Returns (test predictions, features dropped, training compounds dropped).
    """
    from sklearn.ensemble import RandomForestRegressor
    X_tr = np.asarray(X_tr, dtype=float)
    X_te = np.asarray(X_te, dtype=float)
    X_tr[np.isinf(X_tr)] = np.nan
    X_te[np.isinf(X_te)] = np.nan
    X_tr, y_tr, X_te, n_feat_drop, n_row_drop = clean(X_tr, y_tr, X_te)
    m = RandomForestRegressor(n_estimators=N_ESTIMATORS, n_jobs=n_jobs,
                              random_state=seed)
    m.fit(X_tr, y_tr)
    return m.predict(X_te), n_feat_drop, n_row_drop


def magpie_with_nan(compositions):
    from matminer.featurizers.composition import ElementProperty
    from kirby.representations.materials import _featurize
    return _featurize(ElementProperty.from_preset('magpie'), compositions)


def sine_coulomb_with_nan(structures, train_idx):
    """Fitted on the training structures, as automatminer fits featurizers."""
    from matminer.featurizers.structure import SineCoulombMatrix
    from kirby.representations.materials import _featurize
    f = SineCoulombMatrix(flatten=True)
    f.fit([structures[i] for i in train_idx])
    return _featurize(f, structures)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', default=','.join(TASKS))
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    os.environ.setdefault('MATMINER_DATA', '/Volumes/seagate/matbench/data')
    from kirby.datasets.matbench import load_matbench

    OUT.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(OUT.read_text()) if OUT.exists() else []
    folds = [int(f) for f in a.folds.split(',')]
    for task in a.tasks.split(','):
        d = load_matbench(task)
        y, fold_of = d['labels'], d['matbench_fold']
        t0 = time.time()
        X_comp = magpie_with_nan(d['compositions'])
        print(f'{task}: magpie {X_comp.shape[1]} features, '
              f'{time.time() - t0:.0f} s', flush=True)
        pub = published_mae(task)
        rows = []
        for k in folds:
            tr, te = np.where(fold_of != k)[0], np.where(fold_of == k)[0]
            X = X_comp
            if d['structures'] is not None:
                X = np.hstack([X_comp, sine_coulomb_with_nan(d['structures'], tr)])
            t0 = time.time()
            pred, n_fd, n_rd = published_rf(X[tr], y[tr], X[te])
            mae = float(np.mean(np.abs(pred - y[te])))
            rows.append(dict(fold=k, published_mae=pub[k], rerun_mae=mae,
                             n_features=int(X.shape[1]),
                             features_dropped=n_fd,
                             train_compounds_dropped=n_rd,
                             n_train=len(tr), n_test=len(te),
                             seconds=round(time.time() - t0, 1)))
            print(f'  fold {k}: published {pub[k]:.4f}, rerun {mae:.4f} MAE '
                  f'({len(te)} test compounds; {n_fd} features and {n_rd} '
                  f'training compounds dropped)', flush=True)
        results = [r for r in results if r['task'] != task]
        results.append(dict(task=task, folds=rows,
                            published_mean=float(np.mean([r['published_mae'] for r in rows])),
                            rerun_mean=float(np.mean([r['rerun_mae'] for r in rows]))))
        OUT.write_text(json.dumps(results, indent=1))
        print(f'{task}: published {results[-1]["published_mean"]:.4f}, rerun '
              f'{results[-1]["rerun_mean"]:.4f} MAE over folds {folds}',
              flush=True)


if __name__ == '__main__':
    main()
