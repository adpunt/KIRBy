#!/usr/bin/env python3
"""Smoke for n43_noise_aware_training.run_unit: every n43 method on one unit
of random data, no per-fold search, no dataset.

What is checked, on 160 random molecules (20 features; 16 training labels
moved by 4 label SD) split 128 training and 32 held out:
  - a regression unit in two feature spaces (base_1, and a random 'hybrid')
    writes rows for the lnl methods, n33's arms and loss swaps, the NGBoost
    pair, n40's arms on the decided models, the stack, the GP arms, the
    molecule-mean GP arms (pairs of rows share a molecule id) and the kNN
    means under one neighbour setting built here (Euclidean distance on the
    features; a stand-in for n07.sweep_settings, which needs a dataset);
  - every method n43 lists either has rows or a recorded reason;
  - every row key is distinct; every row carries all_D, and every row with a
    D' score carries all_Dprime;
  - the residual kurtosis is recorded per space;
  - a classification unit (two classes) writes lnl rows scored in auc and
    records n33 and n40 as regression only.

SUBSET RUN IN THIS SMOKE ONLY (time): the main unit uses lightgbm and rf;
a second unit on one space adds xgboost and dnn for their loss swaps and
xgboost's n40 arms. dnn's n40 tuning and seed averaging (126 and 10
network fits) are not exercised; the classification unit runs one lnl method
(early_learning). The full run fits every one where its parent is decided.

Run:  python tests/experiments/noise/_smoke_n43.py
"""
import os
import sys
import tempfile
import time
import warnings

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'KIRBY_LGBM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
_root = ('/Volumes/seagate' if os.path.isdir('/Volumes/seagate') else
         '/private/tmp/claude-501/-Users-apunt-repos-KIRBy/'
         '62112545-c7bc-4a8e-9acd-c99b58998fb5/scratchpad')
_tmp = tempfile.mkdtemp(prefix='n43_smoke_', dir=_root)
os.environ['KIRBY_CACHE_DIR'] = os.path.join(_tmp, 'cache')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
warnings.filterwarnings('ignore')

import json

import numpy as np
from scipy.spatial.distance import cdist

from experiments._shared import get_model_factories
from experiments.noise import n43_noise_aware_training as na
from experiments.noise import n07_remediation as n07
from experiments.noise import n33_robust_training as n33
from experiments.noise import n42_similarity_noise_sweep as n42
from experiments.noise.d_vs_dprime import row_key

print(f'temp dir {_tmp}')
rng = np.random.RandomState(0)
N, P = 160, 20
X = rng.normal(size=(N, P)).astype(np.float32)
Xh = np.hstack([X[:, :10], rng.normal(size=(N, 10)).astype(np.float32)])
beta = rng.normal(size=P)
y_clean = X @ beta + 0.3 * rng.normal(size=N)
cmask = np.zeros(N, dtype=bool)
cmask[rng.choice(N, 16, replace=False)] = True
y0 = y_clean.copy()
y0[cmask] += 4.0 * y_clean.std() * rng.choice([-1, 1], size=16)
groups = np.arange(N) // 4
rep_ids = np.arange(N) // 2
tr = np.arange(0, 128)
va = np.arange(128, N)
rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

MODELS = ['lightgbm', 'rf']
factories = get_model_factories(MODELS, task_type='regression', n_samples=N)
swaps = {k: v for k, v in n33.build_swaps(na.HUBER_C).items() if v in MODELS}
SIM_KS = (5, 10, 20)


def neighbours(space, X_tr, X_va):
    """One stand-in neighbour setting: Euclidean distance on the features."""
    D_va = cdist(X_va, X_tr)
    idx = np.argsort(D_va, axis=1)[:, :20]
    dist = np.take_along_axis(D_va, idx, axis=1)
    return [{'name': 'smoke_euclidean',
             'blocks': lambda a, b, X_tr=X_tr: cdist(X_tr[a:b], X_tr),
             'idx_val': idx, 'dist_val': dist, 'why': None}]


failures = []


def check(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg)
    if not cond:
        failures.append(msg)


def read_rows(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ── regression unit ──
sink = n07.RowSink(_tmp, na.EXP_ID, 'smoke', 'outlier_p05__smoke')
errors = n42._Errors()
unavailable = {}
u = na._Unit(ds='smoke', sink=sink, metric=('rmse', rmse, True),
             task_type='regression', condition='outlier_p05', level=1.0,
             draw=0, fold=0, y0=y0[tr], y_clean=y_clean[tr], y_val=y_clean[va],
             g_tr=groups[tr], cmask=cmask[tr], errors=errors,
             unavailable=unavailable)
spaces = {'base_1': (X[tr], X[va]), 'hybrid': (Xh[tr], Xh[va])}
t0 = time.time()
na.run_unit(u, spaces, n33_models=MODELS, n40_models=MODELS,
            factories=factories, swaps=swaps, ngb_pair=True, host='lightgbm',
            neighbours=neighbours, sim_ks=SIM_KS,
            schemes=('rank', 'inverse_square'), rep_ids_tr=rep_ids[tr])
dt = time.time() - t0

rows = read_rows(sink.path)
keys = [row_key(r) for r in rows]
methods = sorted({r['method'] for r in rows})
print(f'regression unit: {len(rows)} rows in {dt:.0f}s; {len(methods)} methods')
check(len(keys) == len(set(keys)), 'every row key distinct')
expected = ([f'lnl/{m}' for m in na.LNL_METHODS]
            + [f'n33/{a}' for a in na.N33_LABEL_ARMS]
            + [f'n33/{s}' for s in swaps] + ['n33/ngboost_t']
            + [f'n40/{a}' for a in na.N40_MODEL_ARMS]
            + [f'n40/{a}' for a in na.N40_HOST_ARMS if a != 'lightgbm'])
for m in expected:
    got = [r for r in rows if r['method'] == m]
    finite = [r for r in got if r['score_Dprime'] is not None
              and np.isfinite(r['score_Dprime'])]
    reason = [k for k in unavailable if k.startswith(m)]
    check(bool(finite) or bool(reason),
          f'{m}: {len(finite)} of {len(got)} rows scored'
          + (f'; reason recorded {reason[:1]}' if reason and not finite else ''))
check(all(isinstance(r.get('all_D'), dict) for r in rows), 'every row carries all_D')
check(all(isinstance(r.get('all_Dprime'), dict) for r in rows
          if r['score_Dprime'] is not None and np.isfinite(r['score_Dprime'])),
      "every row with a D' score carries all_Dprime")
check(set(rows[0]['all_D']) == {'rmse', 'mae', 'r2', 'spearman', 'pearson'},
      f"all_D holds {sorted(rows[0]['all_D'])}")
check(all(r['model_input'] in spaces for r in rows), 'model_input is the space name')
check(any(r['similarity_setting'] == 'smoke_euclidean' for r in rows),
      'kNN and Gamma rows carry the neighbour setting')
check(all('residual_kurtosis' in u.record['per_space'][sp] for sp in spaces),
      'residual kurtosis recorded in both spaces: ' + ', '.join(
          f"{sp} {u.record['per_space'][sp]['residual_kurtosis']:.2f}"
          for sp in spaces))
lnl_scored = [r for r in rows if r['method'].startswith('lnl/')
              and np.isfinite(r['score_Dprime'] or np.nan)]
print('       lnl gains on base_1 (rmse, positive = method beat the plain network):')
for r in sorted(lnl_scored, key=lambda r: r['method']):
    if r['model_input'] == 'base_1':
        print(f"         {r['method']:28s} D {r['score_D']:.3f}  D' "
              f"{r['score_Dprime']:.3f}  gain {r['gain']:+.3f}")
if errors.d:
    print('  errors:')
    for k, v in errors.d.items():
        print(f'    {k}: {v["count"]}x {v["first"][:160]}')
check(len(errors.d) == 0, f'{len(errors.d)} distinct errors')

# ── xgboost and dnn on one space: their loss swaps, xgboost's n40 arms ──
M2 = ['xgboost', 'dnn']
f2 = get_model_factories(M2, task_type='regression', n_samples=N)
sw2 = {k: v for k, v in n33.build_swaps(na.HUBER_C).items() if v in M2}
sink2 = n07.RowSink(_tmp, na.EXP_ID, 'smoke2', 'outlier_p05__smoke')
unav2 = {}
e2 = n42._Errors()
u3 = na._Unit(ds='smoke2', sink=sink2, metric=('rmse', rmse, True),
              task_type='regression', condition='outlier_p05', level=1.0,
              draw=0, fold=0, y0=y0[tr], y_clean=y_clean[tr], y_val=y_clean[va],
              g_tr=groups[tr], cmask=cmask[tr], errors=e2, unavailable=unav2)
t2 = time.time()
na.run_unit(u3, {'base_1': spaces['base_1']}, lnl_methods=(),
            n33_models=M2, n40_models=['xgboost'], factories=f2, swaps=sw2,
            ngb_pair=False, host='xgboost', neighbours=None, sim_ks=SIM_KS,
            rep_ids_tr=None)
rows2 = read_rows(sink2.path)
print(f'xgboost and dnn block: {len(rows2)} rows in {time.time() - t2:.0f}s')
for m in [f'n33/{s}' for s in sw2] + ['n40/retuned', 'n40/seed_avg',
                                      'n40/lightgbm', 'n40/lightgbm_plus_gp_residual']:
    got = [r for r in rows2 if r['method'] == m
           and np.isfinite(r['score_Dprime'] or np.nan)]
    check(bool(got), f'{m}: {len(got)} rows scored')
check('n40/knn_mean' in unav2 and 'n40/gp_means' in unav2,
      'no neighbour setting and no repeats: kNN and molecule-mean reasons recorded')
for k, v in e2.d.items():
    print(f'    error {k}: {v["count"]}x {v["first"][:160]}')
check(len(e2.d) == 0, f'{len(e2.d)} distinct errors in the xgboost and dnn block')

# resume: a second call writes nothing
n_before = sink.n_written
u2 = na._Unit(ds='smoke', sink=sink, metric=('rmse', rmse, True),
              task_type='regression', condition='outlier_p05', level=1.0,
              draw=0, fold=0, y0=y0[tr], y_clean=y_clean[tr], y_val=y_clean[va],
              g_tr=groups[tr], cmask=cmask[tr], errors=n42._Errors(),
              unavailable={})
t1 = time.time()
na.run_unit(u2, {'base_1': spaces['base_1']}, lnl_methods=('dividemix',),
            n33_models=[], n40_models=[])
check(sink.n_written == n_before, f'rerun of a finished cell writes no row '
      f'({time.time() - t1:.1f}s)')

# ── classification unit ──
from sklearn.metrics import roc_auc_score
yc = (y_clean > np.median(y_clean)).astype(int)
yc0 = yc.copy()
flip = rng.choice(128, 13, replace=False)
yc0[flip] = 1 - yc0[flip]
sink_c = n07.RowSink(_tmp, na.EXP_ID, 'smoke_clf', 'flip__smoke')
unav_c = {}
uc = na._Unit(ds='smoke_clf', sink=sink_c,
              metric=('auc', lambda a, b: float(roc_auc_score(a, b)), False),
              task_type='classification', condition='uniform', level=0.1,
              draw=0, fold=0, y0=yc0[tr], y_clean=yc[tr], y_val=yc[va],
              g_tr=groups[tr], cmask=None, errors=n42._Errors(),
              unavailable=unav_c, classes_=np.array([0, 1]))
na.run_unit(uc, {'base_1': spaces['base_1']},
            lnl_methods=('early_learning', 'evidential_kernel'))
rows_c = read_rows(sink_c.path)
check(len(rows_c) == 1 and rows_c[0]['metric'] == 'auc'
      and np.isfinite(rows_c[0]['score_Dprime']),
      f'classification: {len(rows_c)} row, early_learning scored in auc')
check('lnl/evidential_kernel' in unav_c and 'n33/*' in unav_c and 'n40/*' in unav_c,
      'classification: evidential_kernel, n33 and n40 recorded as regression only')
check(set(rows_c[0].get('all_D') or {}) == {'auc', 'accuracy', 'mcc', 'f1_weighted'},
      'classification all_D holds auc, accuracy, mcc, f1_weighted')

print(f'\ntotal {time.time() - t0:.0f}s')
if failures:
    print(f'{len(failures)} FAILED')
    sys.exit(1)
print('all passed')
