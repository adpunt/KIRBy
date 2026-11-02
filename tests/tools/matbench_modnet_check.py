#!/usr/bin/env python3
"""MODNet on the Matbench base representations, one representation at a time.

The MODNet half of tests/tools/matbench_model_check.py. Same tasks, folds,
training rows, cached representations and results file, so its lines sit
beside the other models'. Separate because MODNet needs tensorflow and its
own pins (python 3.10, pandas<=1.5, scikit-learn 1.3).

What runs is the leaderboard's MODNet v0.1.12 recipe with our representation
in place of MODNet's own matminer features:
  1. MODNet's feature selection (relevance to the target minus redundancy with
     features already chosen, both measured by normalised mutual information),
     fitted on the training rows only.
  2. Hyperparameters chosen by MODNet's genetic search (FitGenetic), on the
     training rows only, refit on all of them, then the test rows predicted.
The leaderboard ran the genetic search with a population of 20 for 10
generations; --ga-pop and --ga-gens default to that. Locally pass smaller
values to check it runs.

    /Volumes/seagate/matbench/env_modnet/bin/python tests/tools/matbench_modnet_check.py \
        --tasks matbench_jdft2d --folds 0 --max-train 2000 --ga-pop 4 --ga-gens 2
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# Shared with the other models, so the training rows are identical.
from matbench_model_check import (  # noqa: E402
    OUT_DIR, REPS_DIR, RESULTS, cached_reps, done_keys, train_test_rows)

LABELS_DIR = OUT_DIR / 'labels'


def modnet_fit_predict(X_tr, y_tr, X_te, ga_pop, ga_gens, n_jobs, n_select):
    import pandas as pd
    from modnet.hyper_opt import FitGenetic
    from modnet.preprocessing import MODData

    cols = [f'f{i}' for i in range(X_tr.shape[1])]
    tr_ids = [f'tr{i}' for i in range(len(y_tr))]
    te_ids = [f'te{i}' for i in range(len(X_te))]
    train = MODData(
        materials=[None] * len(y_tr),
        targets=y_tr.reshape(-1, 1), target_names=['y'],
        structure_ids=tr_ids,
        df_featurized=pd.DataFrame(X_tr, index=tr_ids, columns=cols))
    train.feature_selection(n=n_select, n_jobs=n_jobs)
    test = MODData(
        materials=[None] * len(X_te), structure_ids=te_ids,
        df_featurized=pd.DataFrame(X_te, index=te_ids, columns=cols))
    test.optimal_features = train.optimal_features
    ga = FitGenetic(train)
    model = ga.run(size_pop=ga_pop, num_generations=ga_gens, n_jobs=n_jobs,
                   early_stopping=True, refit=True)
    pred = model.predict(test)
    if isinstance(pred, tuple):
        pred = pred[0]
    return np.asarray(pred.values).ravel(), len(train.optimal_features)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', default='matbench_jdft2d,matbench_phonons,'
                    'matbench_expt_gap,matbench_dielectric,matbench_log_gvrh,'
                    'matbench_perovskites')
    ap.add_argument('--reps', default=None, help='default: every cached rep')
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--max-train', type=int, default=0)
    ap.add_argument('--ga-pop', type=int, default=20)
    ap.add_argument('--ga-gens', type=int, default=10)
    ap.add_argument('--n-select', type=int, default=-1,
                    help='features kept by MODNet selection; -1 = MODNet '
                         'decides (its default)')
    ap.add_argument('--n-jobs', type=int, default=4)
    a = ap.parse_args()

    done = done_keys()
    model = 'modnet'
    for task in a.tasks.split(','):
        z = np.load(LABELS_DIR / f'{task}.npz')
        y, fold_of = z['labels'], z['matbench_fold']
        reps = a.reps.split(',') if a.reps else cached_reps(task)
        for rep in reps:
            X = np.load(REPS_DIR / task / f'{rep}.npy', mmap_mode='r')
            for fold in (int(f) for f in a.folds.split(',')):
                key = (task, rep, model, fold, a.max_train)
                if key in done:
                    continue
                tr, te = train_test_rows(fold_of, fold, a.max_train)
                t0 = time.time()
                row = dict(task=task, rep=rep, model=model, fold=fold,
                           max_train=a.max_train, n_train=len(tr),
                           n_test=len(te), n_features=int(X.shape[1]),
                           ga_pop=a.ga_pop, ga_gens=a.ga_gens)
                try:
                    pred, n_kept = modnet_fit_predict(
                        np.asarray(X[tr], dtype=np.float32), y[tr],
                        np.asarray(X[te], dtype=np.float32),
                        a.ga_pop, a.ga_gens, a.n_jobs, a.n_select)
                    res = pred - y[te]
                    row['mae'] = float(np.mean(np.abs(res)))
                    row['rmse'] = float(np.sqrt(np.mean(res ** 2)))
                    row['n_features_selected'] = n_kept
                except Exception as e:  # recorded, never silent
                    row['error'] = f'{type(e).__name__}: {e}'
                row['seconds'] = round(time.time() - t0, 2)
                with open(RESULTS, 'a') as fh:
                    fh.write(json.dumps(row) + '\n')
                print(f"{task} {rep:24s} modnet fold {fold} "
                      f"MAE {row.get('mae', float('nan')):.4f} "
                      f"{row['seconds']:7.1f} s {row.get('error', '')}",
                      flush=True)


if __name__ == '__main__':
    main()
