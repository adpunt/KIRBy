#!/usr/bin/env python3
"""Fits one non-torch model in a process of its own.

On this Mac, torch's OpenMP runtime clashes with the one xgboost and
scikit-learn's thread pools use: imported after torch, an xgboost fit dies
silently; imported before torch, loading Chronos-2 segfaults; and after a
torch training run, a scikit-learn grid search with n_jobs > 1 died silently
too (all seen 2026-10-30). So the screen scripts fit every model except the
torch networks in this subprocess, which never imports torch.

    python _xgb_worker.py <kind> <dir> <seed> [debug]

<dir> holds X_tr.npy, y_tr.npy, X_te.npy; the worker writes pred.npy and
score.npy there (score = probabilities or decision values, where the model
has them). kind: xgbod (pyod XGBOD), xgboost (XGBClassifier), knn, rf,
logreg, svm (the DroneRF screen's settings, see b06_dronerf_views.py).
"""
import os
import sys
from pathlib import Path

import numpy as np

kind, d, seed = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
debug = len(sys.argv) > 4 and sys.argv[4] == 'debug'
X_tr = np.load(d / 'X_tr.npy')
y_tr = np.load(d / 'y_tr.npy')
X_te = np.load(d / 'X_te.npy')
N_JOBS = int(os.environ.get('SLURM_CPUS_PER_TASK') or 8)   # the job's CPUs on the server
info = {}

if kind == 'xgbod':
    from pyod.models.xgbod import XGBOD
    m = XGBOD(random_state=seed).fit(X_tr, y_tr)
    pred, score = m.predict(X_te), m.decision_function(X_te)
elif kind == 'xgboost':
    from xgboost import XGBClassifier
    m = XGBClassifier(random_state=seed, n_jobs=N_JOBS).fit(X_tr, y_tr)
    pred, score = m.predict(X_te), m.predict_proba(X_te)
elif kind == 'knn':
    from sklearn.neighbors import KNeighborsClassifier
    m = KNeighborsClassifier(n_neighbors=5, n_jobs=N_JOBS).fit(X_tr, y_tr)
    pred, score = m.predict(X_te), m.predict_proba(X_te)
elif kind == 'rf':
    from sklearn.ensemble import RandomForestClassifier
    m = RandomForestClassifier(n_estimators=300, random_state=seed,
                               class_weight='balanced', n_jobs=N_JOBS).fit(X_tr, y_tr)
    pred, score = m.predict(X_te), m.predict_proba(X_te)
elif kind == 'logreg':
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(StandardScaler(),
                      LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000)).fit(X_tr, y_tr)
    pred, score = m.predict(X_te), m.predict_proba(X_te)
elif kind == 'svm':
    import json
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    exps = range(-15, 16, 2) if not debug else [-15, -5, 5, 15]
    grid = {'svc__C': [2.0 ** e for e in exps], 'svc__gamma': [2.0 ** e for e in exps]}
    pipe = make_pipeline(StandardScaler(), SVC(kernel='rbf', cache_size=2000))
    gs = GridSearchCV(pipe, grid, cv=StratifiedKFold(3, shuffle=True, random_state=seed), n_jobs=N_JOBS)
    if debug and len(X_tr) > 3000:
        # debug only: choose C and gamma on a 3,000-row subsample, then refit on all
        rng = np.random.RandomState(seed)
        sub = rng.choice(len(X_tr), 3000, replace=False)
        gs.fit(X_tr[sub], y_tr[sub])
        info['grid_subsample'] = 3000
    else:
        gs.fit(X_tr, y_tr)
    m = gs.best_estimator_
    m.fit(X_tr, y_tr)
    info['best'] = {k: float(v) for k, v in gs.best_params_.items()}
    pred, score = m.predict(X_te), m.decision_function(X_te)
    with open(d / 'info.json', 'w') as f:
        json.dump(info, f)
else:
    raise SystemExit(f'unknown kind {kind}')
np.save(d / 'pred.npy', np.asarray(pred))
np.save(d / 'score.npy', np.asarray(score))
