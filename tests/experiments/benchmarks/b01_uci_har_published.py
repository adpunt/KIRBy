#!/usr/bin/env python3
"""UCI HAR as published: one-vs-all Gaussian SVM on the fixed train/test split.

Source of the protocol: Anguita, Ghio, Oneto, Parra, Reyes-Ortiz, "A Public
Domain Dataset for Human Activity Recognition Using Smartphones", ESANN 2013,
section 3 and Table 4. Published result: 96% accuracy on the 2,947 test
windows (Table 4 diagonal: 2,840 of 2,947 correct = 96.37%).

What the paper states:
  - multiclass SVM, one-vs-all, Gaussian kernel
  - hyperparameters chosen by 10-fold cross-validation on the training set
  - scored once on the test set (2,947 windows, 9 people)
What it does not state, so is chosen here (see NEW_DATASETS.md):
  - the values searched: C in {1, 10, 100, 1000}, gamma in {1e-4 .. 1e-1}
  - how the 10 folds were drawn: --folds random (stratified, seed 0) or
    --folds subject (each person's windows in one fold), or both

Data: UCI dataset 240, unzipped to
/Volumes/seagate/benchmarks_protocol/uci_har/UCI HAR Dataset/ (override with
KIRBY_HAR_DIR). Features are used as shipped (already scaled to [-1, 1]).

Output: one JSON per fold scheme in
/Volumes/seagate/benchmarks_protocol/results/b01_uci_har_<folds>.json

Each fold scheme takes about 21 minutes on the laptop (all cores).
    python tests/experiments/benchmarks/b01_uci_har_published.py --folds both
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

HAR_DIR = Path(os.environ.get(
    'KIRBY_HAR_DIR',
    '/Volumes/seagate/benchmarks_protocol/uci_har/UCI HAR Dataset'))
OUT_DIR = Path('/Volumes/seagate/benchmarks_protocol/results')

PUBLISHED_CORRECT = 2840
PUBLISHED_TEST_N = 2947
ACTIVITIES = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS',
              'SITTING', 'STANDING', 'LAYING']
GRID = {'estimator__C': [1, 10, 100, 1000],
        'estimator__gamma': [1e-4, 1e-3, 1e-2, 1e-1]}


def load(part):
    X = np.loadtxt(HAR_DIR / part / f'X_{part}.txt')
    y = np.loadtxt(HAR_DIR / part / f'y_{part}.txt', dtype=int) - 1
    subject = np.loadtxt(HAR_DIR / part / f'subject_{part}.txt', dtype=int)
    return X, y, subject


def run(folds, X_tr, y_tr, s_tr, X_te, y_te):
    if folds == 'random':
        cv, groups = StratifiedKFold(10, shuffle=True, random_state=0), None
    else:
        cv, groups = GroupKFold(10), s_tr
    t0 = time.time()
    gs = GridSearchCV(OneVsRestClassifier(SVC(kernel='rbf')), GRID, cv=cv,
                      n_jobs=-1)
    gs.fit(X_tr, y_tr, groups=groups)
    pred = gs.predict(X_te)
    cm = confusion_matrix(y_te, pred)
    out = {
        'benchmark': 'uci_har',
        'folds': folds,
        'grid': GRID,
        'best_params': gs.best_params_,
        'cv_accuracy': float(gs.best_score_),
        'test_accuracy': float(accuracy_score(y_te, pred)),
        'test_correct': int(np.trace(cm)),
        'test_n': int(len(y_te)),
        'published_test_accuracy': PUBLISHED_CORRECT / PUBLISHED_TEST_N,
        'published_test_correct': PUBLISHED_CORRECT,
        'confusion_matrix_rows_true': cm.tolist(),
        'activities': ACTIVITIES,
        'seconds': round(time.time() - t0),
    }
    print(f"{folds}: best {gs.best_params_}, 10-fold accuracy "
          f"{out['cv_accuracy']:.4f}; test {out['test_correct']} of "
          f"{out['test_n']} windows correct = {out['test_accuracy']:.4f} "
          f"(published {PUBLISHED_CORRECT} of {PUBLISHED_TEST_N} = "
          f"{out['published_test_accuracy']:.4f}); {out['seconds']} s",
          flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', choices=['random', 'subject', 'both'],
                    default='both')
    args = ap.parse_args()
    X_tr, y_tr, s_tr = load('train')
    X_te, y_te, _ = load('test')
    assert X_tr.shape == (7352, 561) and X_te.shape == (2947, 561)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for folds in (['random', 'subject'] if args.folds == 'both'
                  else [args.folds]):
        out = run(folds, X_tr, y_tr, s_tr, X_te, y_te)
        with open(OUT_DIR / f'b01_uci_har_{folds}.json', 'w') as f:
            json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
