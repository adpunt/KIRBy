#!/usr/bin/env python3
"""Synthetic smoke for f36. No real dataset, no search, nothing from the
precompute.

    KIRBY_LGBM_THREADS=1 OMP_NUM_THREADS=1 python tests/experiments/framework/_smoke_f36.py

Checks:
  1. setting and unit counts;
  2. the combinations not run (residual scale with preserve none or group
     mean) give the same labels as raw scale;
  3. what each preserve option promises;
  4. one regression unit, every setting, through run_unit with the real
     noise_cache and the kernel-once SVM; a rerun writes nothing;
  5. a unit killed part way keeps its rows and the rerun fits only the rest;
  6. one classification unit, SVM on its decision function;
  7. f36_analyse on the rows written.
"""
import io
import json
import os
import sys
import tempfile
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))
sys.path.insert(0, os.path.join(_HERE, '..'))

TMP = tempfile.mkdtemp(prefix='f36_smoke_')
os.environ.setdefault('KIRBY_CACHE_DIR', os.path.join(TMP, 'cache'))

from experiments.framework import f36_label_smoothing_portfolio as f36  # noqa: E402
from experiments._shared import get_model_factories  # noqa: E402
from sklearn.metrics import mean_squared_error, roc_auc_score  # noqa: E402


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg, flush=True)
    if not cond:
        raise SystemExit(1)


def toy(n=90, p=8, seed=0, clf=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = X[:, 0] - 0.5 * X[:, 1] + 0.3 * rng.normal(size=n)
    if clf:
        y = (y > np.median(y)).astype(int)
    return X, y, np.repeat(np.arange(n // 3), 3)[:n]


# 1
n_reg = len(f36.settings(False, 'gaussian'))
want = 18 * (15 * 3 + 15) + 18 * 9 + 13 * 30 * 2 + 5
check(n_reg == want, f'{n_reg} settings per regression unit (= {want})')
check(len(f36.settings(False, 'censoring')) == want + 3,
      'censoring adds the three lifts')
check(len(f36.settings(True, 'uniform')) == 12 * 3 * 3 * 2,
      f"{len(f36.settings(True, 'uniform'))} settings per herg unit")
check(len(f36.units_for(False)) == 16 and len(f36.units_for(True)) == 4,
      'units: 16 per regression dataset, 4 for herg')

# 2
X, y, g = toy()
rng = np.random.default_rng(1)
y0 = y + rng.normal(scale=0.5, size=y.size)
score = np.abs(rng.normal(size=y.size))
p0 = y0 + rng.normal(scale=0.3, size=y.size)
v0 = y0 + rng.normal(scale=0.4, size=y.size)
ctx = {'disagree': np.abs(rng.normal(size=y.size)), 'disagree_s': 1.0}
worst = 0.0
for r in f36.RULES:
    for pr in ('none', 'group_mean'):
        a, _ = f36.smooth(y0, v0, r, pr, 'raw', 'once', score=score, ctx=ctx,
                          groups=g, p0=p0)
        b, _ = f36.smooth(y0, v0, r, pr, 'residual', 'once', score=score,
                          ctx=ctx, groups=g, p0=p0)
        worst = max(worst, float(np.max(np.abs(a - b))))
check(worst < 1e-9, f'residual scale = raw scale under preserve none and '
                    f'group mean, every rule (largest difference {worst:.1e})')

# 3
moved = f36.apply_rule('prop1', y0, v0, score, ctx)
check(abs(np.std(f36.preserve('variance', moved, y0)) - np.std(y0)) < 1e-9,
      'preserve variance keeps the SD')
pg = f36.preserve('group_mean', moved, y0, g)
gm = lambda a: np.bincount(g, weights=a) / np.bincount(g)  # noqa: E731
check(np.max(np.abs(gm(pg) - gm(y0))) < 1e-9, 'preserve group mean keeps it')
check(np.array_equal(np.sort(f36.preserve('rank', moved, y0)), np.sort(y0)),
      'preserve rank keeps the set of label values')

# 4
X, y, g = toy(n=120, seed=3)
Xv, yv, _ = toy(n=40, seed=4)
y0 = y + np.random.default_rng(5).normal(scale=0.6, size=y.size)
rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))  # noqa: E731
svm = get_model_factories(['svm'], 'regression', n_samples=160)['svm']
OUT = os.path.join(TMP, 'out')
errors, unav = f36._Errors(), {}
unit = (0, 'censoring', 0.25, 'base_1')
y0c = np.minimum(y0, np.quantile(y0, 0.8))     # a censored column
kw = dict(ds='f36_smoke', X_tr=X, X_va=Xv, y_clean=y, y_val=yv, g_tr=g,
          metric=('rmse', rmse, True), task_type='regression', factory=svm,
          output_dir=OUT, errors=errors, unavailable=unav)
t0 = time.time()
n1 = f36.run_unit(unit=unit, y0=y0c, **kw)
check(n1 == want + 3, f'one regression unit wrote {n1} rows '
                      f'({time.time() - t0:.0f}s)')
rows = [json.loads(line) for line in
        open(f36.unit_rows_path(OUT, 'f36_smoke', unit))]
nan = [r for r in rows if not np.isfinite(r['score_Dprime'])]
print(f'     {len(nan)} rows with no score; errors {errors.d}')
print(f'     why (first 4): {list(unav.items())[:4]}')
lifts = [r for r in rows if r['method_setting'].startswith('lift')]
check(len(lifts) == 3 and all(np.isfinite(r['score_Dprime']) for r in lifts),
      'the three lifts scored on the censored column')
check(f36.run_unit(unit=unit, y0=y0c, **kw) == 0, 'rerun writes nothing')

# 5
calls = {'n': 0}


def dying(*a, **k):
    calls['n'] += 1
    if calls['n'] > 40:
        raise KeyboardInterrupt('wall clock')
    return svm()


unit_k = (1, 'gaussian', 0.5, 'base_1')
try:
    f36.run_unit(unit=unit_k, y0=y0, flush_seconds=0.0,
                 **dict(kw, factory=dying))
except KeyboardInterrupt:
    pass
pk = f36.unit_rows_path(OUT, 'f36_smoke', unit_k)
kept = sum(1 for _ in open(pk))
n_more = f36.run_unit(unit=unit_k, y0=y0, **kw)
check(kept > 0 and kept + n_more == want,
      f'killed part way: {kept} rows kept, rerun added {n_more}, {want} in all')

# 6
Xc, yc, gc = toy(n=120, seed=6, clf=True)
Xcv, ycv, _ = toy(n=40, seed=7, clf=True)
y0k = yc.copy()
flip = np.random.default_rng(8).random(yc.size) < 0.15
y0k[flip] = 1 - y0k[flip]
svc = get_model_factories(['svm'], 'classification', n_samples=160)['svm']
nc_ = f36.run_unit(ds='f36_smoke_clf', unit=(0, 'uniform', 0.1, 'base_1'),
                   X_tr=Xc, X_va=Xcv, y0=y0k, y_clean=yc, y_val=ycv, g_tr=gc,
                   metric=('auc', roc_auc_score, False),
                   task_type='classification', factory=svc, output_dir=OUT,
                   errors=errors, unavailable=unav)
check(nc_ == 216, f'classification unit wrote {nc_} rows')

# 7
from experiments.framework import f36_analyse as an  # noqa: E402
buf = io.StringIO()
rep = an.analyse(os.path.join(OUT, f36.EXP_ID), out=buf)
print('\n'.join(buf.getvalue().splitlines()[-12:]))
check(rep['n_tasks'] == 3, f"analyse read {rep['n_tasks']} tasks")
print(f'ALL PASS ({TMP})')
