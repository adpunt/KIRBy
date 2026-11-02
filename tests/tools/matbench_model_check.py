#!/usr/bin/env python3
"""Which downstream models work with the Matbench base representations.

One representation at a time, no hybrid search. For each (task, rep, model):
fit on the Matbench training rows of one fold, predict that fold's Matbench
test rows, record MAE and RMSE and the seconds taken.

Models:
  KIRBy's own default set, the 'fast' and 'full' tiers of DEFAULT_MODELS
  (rf, xgboost, lightgbm, svm, dnn), each run as nested_cv_evaluate runs it:
  features scaled and imputed on the training rows only.

  rf_matbench, the leaderboard's 'RF-SCM/Magpie' learner as its run.py builds
  it: RandomForestRegressor(n_estimators=500), every other setting sklearn's
  default (so every feature is a split candidate). No scaling. Its leaderboard
  pipeline also dropped features missing in more than 1% of compounds and
  filled the rest with the training mean; our representations already set
  missing values to 0, so that step cannot be reproduced here.

  MODNet runs in its own environment: tests/tools/matbench_modnet_check.py.

Local use is a check that every combination runs, not a result: --folds 0 and
--max-train subsample the training rows. On the server leave both at the
full settings (--folds 0,1,2,3,4, no --max-train).

Output: one JSON line per (task, rep, model, fold) in
/Volumes/seagate/matbench/model_check/results.jsonl. A line already there is
not run again.

    /Volumes/seagate/matbench/venv/bin/python tests/tools/matbench_model_check.py \
        --tasks matbench_jdft2d --folds 0 --max-train 2000
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# macOS only: after lightgbm has fitted once in a process, the dnn's first
# fit never returns (0% CPU, tested 2026-10-26 in this env and in the base
# anaconda env). One OpenMP thread avoids it: 500 compounds x 64 features fit
# in 19 s. Set before numpy, torch or lightgbm load. Linux is left alone.
if sys.platform == 'darwin':
    os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

REPS_DIR = Path('/Volumes/seagate/matbench/reps')
OUT_DIR = Path('/Volumes/seagate/matbench/model_check')
LABELS_DIR = OUT_DIR / 'labels'
RESULTS = OUT_DIR / 'results.jsonl'

KIRBY_MODELS = ['rf', 'xgboost', 'lightgbm', 'svm', 'dnn']
ALL_MODELS = KIRBY_MODELS + ['rf_matbench']


def task_labels(task):
    """Labels and Matbench test folds, cached so MODNet's env needs no matbench."""
    path = LABELS_DIR / f'{task}.npz'
    if not path.exists():
        from kirby.datasets.matbench import load_matbench
        d = load_matbench(task)
        LABELS_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(path, labels=d['labels'], matbench_fold=d['matbench_fold'])
    z = np.load(path)
    return z['labels'], z['matbench_fold']


def cached_reps(task):
    return sorted(p.stem for p in (REPS_DIR / task).glob('*.npy')
                  if not p.name.startswith('._'))


def train_test_rows(fold_of, fold, max_train, seed=42):
    train = np.where(fold_of != fold)[0]
    test = np.where(fold_of == fold)[0]
    if max_train and len(train) > max_train:
        rng = np.random.default_rng(seed)
        train = np.sort(rng.choice(train, max_train, replace=False))
    return train, test


def fit_predict(model_name, X_tr, y_tr, X_te):
    if model_name == 'rf_matbench':
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42)
        m.fit(X_tr, y_tr)
        return m.predict(X_te)
    from copy import deepcopy
    from kirby.find_best_hybrid import (
        FeatureScaler, _fit_model, _impute_nan, get_model)
    model = get_model(model_name, 'regression', n_samples=len(y_tr))
    if model is None:
        return None
    scaler = FeatureScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    X_tr, X_te, _ = _impute_nan(X_tr, X_te)
    m = deepcopy(model)
    # One group per compound: the dnn's early-stopping holdout is a random
    # split of the training rows, matching the Matbench folds.
    _fit_model(m, X_tr, y_tr, groups=np.arange(len(y_tr)))
    return np.asarray(m.predict(X_te)).ravel()


def done_keys():
    keys = set()
    if RESULTS.exists():
        for line in RESULTS.read_text().splitlines():
            r = json.loads(line)
            keys.add((r['task'], r['rep'], r['model'], r['fold'],
                      r['max_train']))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', default='matbench_jdft2d,matbench_phonons,'
                    'matbench_expt_gap,matbench_dielectric,matbench_log_gvrh,'
                    'matbench_perovskites')
    ap.add_argument('--reps', default=None, help='default: every cached rep')
    ap.add_argument('--models', default=','.join(ALL_MODELS))
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--max-train', type=int, default=0,
                    help='subsample the training rows; 0 = all')
    a = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = done_keys()
    folds = [int(f) for f in a.folds.split(',')]
    models = a.models.split(',')
    for task in a.tasks.split(','):
        y, fold_of = task_labels(task)
        reps = a.reps.split(',') if a.reps else cached_reps(task)
        for rep in reps:
            X = np.load(REPS_DIR / task / f'{rep}.npy', mmap_mode='r')
            for fold in folds:
                tr, te = train_test_rows(fold_of, fold, a.max_train)
                X_tr = np.asarray(X[tr], dtype=np.float32)
                X_te = np.asarray(X[te], dtype=np.float32)
                for model in models:
                    key = (task, rep, model, fold, a.max_train)
                    if key in done:
                        continue
                    t0 = time.time()
                    err = None
                    try:
                        pred = fit_predict(model, X_tr, y[tr], X_te)
                    except Exception as e:  # recorded, never silent
                        pred, err = None, f'{type(e).__name__}: {e}'
                    sec = time.time() - t0
                    row = dict(task=task, rep=rep, model=model, fold=fold,
                               max_train=a.max_train, n_train=len(tr),
                               n_test=len(te), n_features=int(X.shape[1]),
                               seconds=round(sec, 2))
                    if pred is None:
                        row['error'] = err or 'model skipped (_max_samples)'
                    else:
                        res = pred - y[te]
                        row['mae'] = float(np.mean(np.abs(res)))
                        row['rmse'] = float(np.sqrt(np.mean(res ** 2)))
                    with open(RESULTS, 'a') as fh:
                        fh.write(json.dumps(row) + '\n')
                    print(f"{task} {rep:24s} {model:12s} fold {fold} "
                          f"MAE {row.get('mae', float('nan')):.4f} "
                          f"{sec:7.1f} s {row.get('error', '')}", flush=True)


if __name__ == '__main__':
    main()
