#!/usr/bin/env python3
"""Synthetic smoke for f36. No real dataset, no search, nothing from the
precompute. Run locally with

    KIRBY_LGBM_THREADS=1 OMP_NUM_THREADS=1 python tests/experiments/framework/_smoke_f36.py

Checks:
  1. the combination counts (regression and classification);
  2. the two combinations f36 does not run are the same numbers as the ones it
     does: scale=residual with preserve none or group_mean against scale=raw,
     every v-independent rule, every passes schedule;
  3. what each preserve option promises: variance keeps the SD, group_mean
     keeps every group's mean, rank keeps the set of label values;
  4. one regression unit, every combination under passes=once, through
     run_unit with the real noise_cache, then rerun: nothing is refitted;
  5. passes=fixed and anneal on that unit for every rule and two values;
  6. one classification unit;
  7. f36_analyse on the rows written.
"""
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
os.environ.setdefault('KIRBY_NO_COLD_SEARCH', '1')

from experiments.framework import f36_label_smoothing_portfolio as f36  # noqa: E402


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg, flush=True)
    if not cond:
        raise SystemExit(1)


def toy(n=60, p=6, seed=0, clf=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = X[:, 0] - 0.5 * X[:, 1] + 0.3 * rng.normal(size=n)
    if clf:
        y = (y > np.median(y)).astype(int)
    g = np.repeat(np.arange(n // 4), 4)[:n]
    return X, y, g


# 1 -----------------------------------------------------------------------
reg, skipped = f36.combinations(False)
clf, _ = f36.combinations(True)
check(len(f36.VALUES) == 12 and len(f36.RULES) == 31,
      f'12 values and 31 rules ({len(f36.VALUES)}, {len(f36.RULES)})')
check(len(reg) == 12 * 31 * 10 + 3 * 31 * 10 * 2,
      f'regression runs {len(reg)} combinations = 12 x 31 x 10 single pass '
      f'+ 3 x 31 x 10 x 2 repeated')
check(len(skipped) == 12 * 31 * 2 + 3 * 31 * 2 * 2,
      f'{len(skipped)} combinations not run as identical')
check(len(clf) == 12, f'classification runs {len(clf)} combinations')
n1 = sum(len(f36.stage_units('logd', c, False, 1))
         for c in f36.REGRESSION_CONDITIONS)
n1h = len(f36.stage_units('herg', 'uniform', True, 1))
check(n1 == 108 and n1h == 36,
      f'stage 1 units: logd {n1} (3 conditions x 2 levels x 3 folds x 6), '
      f'herg {n1h}')

# 2 -----------------------------------------------------------------------
X, y, g = toy()
rng = np.random.default_rng(1)
y0 = y + rng.normal(scale=0.5, size=y.size)
score = np.abs(rng.normal(size=y.size))
p0 = y0 + rng.normal(scale=0.3, size=y.size)
v0 = y0 + rng.normal(scale=0.4, size=y.size)
nbrs = np.argsort(np.linalg.norm(X[:, None] - X[None], axis=2), axis=1)[:, 1:10]


def refit(yy):
    return np.mean(np.asarray(yy)[nbrs], axis=1)


ctx = {'unc': np.abs(rng.normal(size=y.size)) + 0.1,
       'disagree': np.abs(rng.normal(size=y.size)), 'disagree_s': 1.0}
worst = 0.0
n_cmp = 0
for r in f36.RULES:
    for pr in ('none', 'group_mean'):
        for p in f36.PASSES:
            a, _ = f36.smooth(y0, v0, r, pr, 'raw', p, score=score, ctx=ctx,
                              groups=g, p0=p0, refit_v=refit)
            b, _ = f36.smooth(y0, v0, r, pr, 'residual', p, score=score,
                              ctx=ctx, groups=g, p0=p0, refit_v=refit)
            worst = max(worst, float(np.max(np.abs(a - b))))
            n_cmp += 1
check(worst < 1e-9, f'scale=residual equals scale=raw under preserve none and '
                    f'group_mean on {n_cmp} rule x preserve x passes '
                    f'combinations (largest difference {worst:.2e})')
# and the ones it does run are not identical
a, _ = f36.smooth(y0, v0, 'prop0.5', 'variance', 'raw', 'once', score=score,
                  ctx=ctx, groups=g, p0=p0)
b, _ = f36.smooth(y0, v0, 'prop0.5', 'variance', 'residual', 'once',
                  score=score, ctx=ctx, groups=g, p0=p0)
check(np.max(np.abs(a - b)) > 1e-6,
      f'scale=residual differs from raw under preserve=variance '
      f'({np.max(np.abs(a - b)):.3g} label units)')

# 3 -----------------------------------------------------------------------
moved = f36.apply_rule('prop1', y0, v0, score, ctx)
pv = f36.preserve('variance', moved, y0)
check(abs(np.std(pv) - np.std(y0)) < 1e-9, 'preserve=variance keeps the SD')
pg = f36.preserve('group_mean', moved, y0, g)
gm = lambda a: np.bincount(g, weights=a) / np.bincount(g)  # noqa: E731
check(np.max(np.abs(gm(pg) - gm(y0))) < 1e-9,
      'preserve=group_mean keeps every group mean')
pr_ = f36.preserve('rank', moved, y0)
check(np.array_equal(np.sort(pr_), np.sort(y0)),
      'preserve=rank keeps the set of label values')
check(f36.preserve('group_mean', moved, y0, None) is None,
      'preserve=group_mean declines without groups')
sc = f36.Scale('rank', np.r_[y0, [y0.max()] * 5])
check(np.allclose(sc.back(sc.fwd(y0)), y0), 'rank scale maps back onto itself')

# 4 -----------------------------------------------------------------------
from experiments._shared import get_model_factories  # noqa: E402
from experiments.noise.d_vs_dprime import done_keys  # noqa: E402
from sklearn.metrics import mean_squared_error  # noqa: E402

X, y, g = toy(n=80, seed=3)
Xva, yva, _ = toy(n=30, seed=4)
y0 = y + np.random.default_rng(5).normal(scale=0.6, size=y.size)
rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))  # noqa: E731
metric = ('rmse', rmse, True)
fac = get_model_factories(['rf'], 'regression', n_samples=len(y))['rf']
OUT = os.path.join(TMP, 'out')
errors, unavailable = f36._Errors(), {}
unit = (0, 0.5, 0, 'rf', 'base_1')
kw = dict(ds='f36_smoke', cond='outlier_p10', unit=unit, X_tr=X, X_va=Xva,
          y0=y0, y_clean=y, y_val=yva, g_tr=g, metric=metric,
          task_type='regression', factory=fac, output_dir=OUT,
          errors=errors, unavailable=unavailable)
t0 = time.time()
n1 = f36.run_unit(passes='once', **kw)
t_once = time.time() - t0
check(n1 == 12 * 31 * 10, f'passes=once wrote {n1} rows, one per combination '
                          f'({t_once:.0f}s)')
path = f36.unit_rows_path(OUT, 'f36_smoke', 'outlier_p10', unit)
import json  # noqa: E402
rows = [json.loads(l) for l in open(path)]
nan = [r for r in rows if not np.isfinite(r['score_Dprime'])]
print(f'     {len(nan)} rows with no D-prime score; errors: '
      f'{json.dumps(errors.d, default=str)[:600]}')
print(f'     unavailable (first 5): {list(unavailable.items())[:5]}')
check(len({r['score_D'] for r in rows}) == 1, 'every row carries one score_D')
t0 = time.time()
n2 = f36.run_unit(passes='once', **kw)
check(n2 == 0, f'rerun writes nothing ({time.time() - t0:.1f}s)')

# 4b: a job killed mid-unit keeps what it fitted and the rerun finishes it.
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
calls = {'n': 0}


def dying():
    calls['n'] += 1
    if calls['n'] > 25:
        raise KeyboardInterrupt('wall clock')
    return KNeighborsRegressor(5)


keep_values = f36.VALUES
f36.VALUES = ('knn9_mean',)
unit_k = (1, 0.5, 0, 'rf', 'base_2')
kw_k = dict(kw, unit=unit_k, factory=dying)
try:
    try:
        f36.run_unit(passes='once', flush_seconds=0.0, **kw_k)
    except KeyboardInterrupt:
        pass
    path_k = f36.unit_rows_path(OUT, 'f36_smoke', 'outlier_p10', unit_k)
    n_kept = sum(1 for _ in open(path_k))
    check(n_kept >= 20, f'killed after {calls["n"] - 1} fits, {n_kept} rows '
                        f'were already on disk')
    calls['n'] = -10**9
    n_more = f36.run_unit(passes='once', **kw_k)
    n_all = sum(1 for _ in open(path_k))
    check(n_all == 31 * 10 and n_more == 31 * 10 - n_kept,
          f'rerun fitted only the {n_more} missing, {n_all} rows in all')
finally:
    f36.VALUES = keep_values

# 5 -----------------------------------------------------------------------
keep_values = f36.VALUES
f36.VALUES = ('knn9_mean', 'pair_mean')
try:
    for p in ('fixed', 'anneal'):
        t0 = time.time()
        n3 = f36.run_unit(passes=p, **kw)
        check(n3 == 2 * 31 * 10, f'passes={p} wrote {n3} rows for 2 values '
                                 f'({time.time() - t0:.0f}s)')
finally:
    f36.VALUES = keep_values
rows = [json.loads(l) for l in open(path)]
diff = {}
for r in rows:
    diff.setdefault(r['method_setting'].replace('passes=once', 'P')
                    .replace('passes=fixed', 'P').replace('passes=anneal', 'P'),
                    []).append(r['score_Dprime'])
n_diff = sum(1 for v in diff.values() if len(v) == 3
             and len({round(x, 12) for x in v if np.isfinite(x)}) > 1)
check(n_diff > 0, f'{n_diff} of {sum(len(v) == 3 for v in diff.values())} '
                  f'combinations score differently across the three passes')

# 6 -----------------------------------------------------------------------
Xc, yc, gc = toy(n=80, seed=6, clf=True)
Xcv, ycv, _ = toy(n=30, seed=7, clf=True)
y0c = yc.copy()
flip = np.random.default_rng(8).random(yc.size) < 0.15
y0c[flip] = 1 - y0c[flip]
from sklearn.metrics import roc_auc_score  # noqa: E402
facc = get_model_factories(['rf'], 'classification', n_samples=len(yc))['rf']
unit_c = (0, 0.1, 0, 'rf', 'base_1')
for p in f36.CLF_PASSES:
    nc_ = f36.run_unit(ds='f36_smoke_clf', cond='uniform', unit=unit_c,
                       passes=p, X_tr=Xc, X_va=Xcv, y0=y0c, y_clean=yc,
                       y_val=ycv, g_tr=gc, metric=('auc', roc_auc_score, False),
                       task_type='classification', factory=facc,
                       output_dir=OUT, errors=errors, unavailable=unavailable)
    check(nc_ == 6, f'classification passes={p}: {nc_} rows')

# 7 -----------------------------------------------------------------------
from experiments.framework import f36_analyse as an  # noqa: E402
rep = an.analyse(os.path.join(OUT, f36.EXP_ID), tolerances=(0.0, 0.01, 0.02),
                 out=sys.stdout)
check(rep['n_tasks'] >= 2, f"analyse read {rep['n_tasks']} tasks")
print(f'\nsmoke output under {TMP}')
print('ALL PASS')
