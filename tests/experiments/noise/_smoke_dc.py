#!/usr/bin/env python3
"""Smoke for dc_detector_by_cleanup.run_unit: the detector x cleanup cross on
one unit of random data, no per-fold search, no dataset.

What is checked: on 160 random molecules (20 features, 16 labels moved by
4 label SD) with rf as the model, a unit with two detector spaces (own and a
random 'hybrid') writes rows for every (detector, cleanup, setting) pair the
pairing rules allow, every row key is distinct, the score-free cleanups run
once, the rounds axis runs, and the detection record holds an AUROC per score
detector. The in-loop network detectors are included (each trains a small DNN
on 160 rows).

Run:  python tests/experiments/noise/_smoke_dc.py
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
_tmp = tempfile.mkdtemp(prefix='dc_smoke_')
os.environ['KIRBY_CACHE_DIR'] = os.path.join(_tmp, 'cache')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
warnings.filterwarnings('ignore')

import numpy as np

from experiments.noise import dc_detector_by_cleanup as dc
from experiments.noise import n07_remediation as n07
from experiments.noise import n42_similarity_noise_sweep as n42
from experiments.noise.d_vs_dprime import row_key

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
tr = np.arange(0, 128)
va = np.arange(128, N)
rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
metric = ('rmse', rmse, True)

sink = n07.RowSink(_tmp, dc.EXP_ID, 'smoke', 'gaussian__smoke')
errors = n42._Errors()
counts, unavailable = {}, {}
u = n42._Unit(ds='smoke', sink=sink, metric=metric, task_type='regression',
              is_clf=False, condition='outlier_p05', level=1.0, draw=0, fold=0,
              model='rf', mi='rf:base_1', X_tr=X[tr], X_va=X[va], y0=y0[tr],
              y_clean=y_clean[tr], y_val=y_clean[va], g_tr=groups[tr],
              train_idx=tr, val_idx=va, prov={}, cliff_train=None,
              rand_train=None, measurement_counts=None, row_models=('rf',),
              takes_weight=True, learner_factories={}, errors=errors,
              unavailable=unavailable, counts=counts, fold_similarity=None)
u.cmask = cmask[tr]
u.detection = {}
spaces = {'own': dc._Space('own', X[tr], X[va], u),
          'hybrid': dc._Space('hybrid', Xh[tr], Xh[va], u)}

t0 = time.time()
dc.run_unit(u, X[tr], X[va], spaces, None, {})
dt = time.time() - t0

failures = []


def check(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg)
    if not cond:
        failures.append(msg)


rows = []
with open(sink.path) as fh:
    import json
    for line in fh:
        rows.append(json.loads(line))
keys = [row_key(r) for r in rows]
methods = sorted({r['method'] for r in rows})
dets = sorted({r['method'].split('/')[1] for r in rows})
cls = sorted({r['method'].split('/')[2] for r in rows})
print(f'{len(rows)} rows in {dt:.0f}s; {len(dets)} detectors, {len(cls)} cleanups')
check(len(keys) == len(set(keys)), 'every row key distinct')
check(len(rows) > 500, f'the cross wrote {len(rows)} rows (over 500)')
check('none' in dets and 'random' in dets, 'score-free (none) and random rows present')
check(any(d.startswith('oracle@precision') for d in dets), 'oracle-precision flag rows present')
check(any(d.startswith('conformal_bh') for d in dets) or counts.get('n41_no_flags'),
      'conformal flag rows present (or none flagged, counted)')
check(all(any(r['method_setting'].startswith(f'dspace={sp}|') for r in rows)
          for sp in ('own', 'hybrid')), 'rows in both detector spaces')
scored = [r for r in rows if r['score_Dprime'] is not None and np.isfinite(r['score_Dprime'])]
check(len(scored) > 0.5 * len(rows), f'{len(scored)} of {len(rows)} rows carry a D\' score')
check(any(r['method'].endswith('/rounds_delete') and np.isfinite(r['score_Dprime'] or np.nan)
          for r in rows), 'rounds_delete scored')
check(any('/score_weight' in r['method'] and np.isfinite(r['score_Dprime'] or np.nan)
          for r in rows), 'score_weight scored under a unit-interval detector')
check(any('/spread_invvar_weight' in r['method'] and np.isfinite(r['score_Dprime'] or np.nan)
          for r in rows), 'spread_invvar_weight scored under a spread detector')
for c in ('network_rewrite', 'ema_rewrite', 'rewrite', 'winsorise', 'exp_downweight',
          'move_confidence', 'bayes_shrink', 'shrink_flagged_per_row'):
    check(any(r['method'].endswith('/' + c) and np.isfinite(r['score_Dprime'] or np.nan)
              for r in rows), f'{c} scored')
det_rec = u.detection.get('own', {})
check(len(det_rec) >= 10 and all('detection_auroc' in v for v in det_rec.values()),
      f'detection graded for {len(det_rec)} detectors in the own space')
best = max(det_rec.items(), key=lambda kv: kv[1].get('detection_auroc', 0))
print(f'       best detector on the moved labels: {best[0]} auroc {best[1]["detection_auroc"]:.3f}')
if errors.d:
    print('  errors:')
    for k, v in errors.d.items():
        print(f'    {k}: {v["count"]}x {v["first"][:160]}')
check(len(errors.d) <= 3, f'{len(errors.d)} distinct errors (3 or fewer)')
print()
if failures:
    print(f'{len(failures)} FAILED'); sys.exit(1)
print('all passed')
