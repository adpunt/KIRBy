#!/usr/bin/env python3
"""Smoke for the split halves of the in-loop label-noise methods
(kirby.utils.lnl_parts; registry names net_*; n07 SPLIT_CLEANUPS).

What is checked, on 150 random molecules with 20 features and 15 labels
moved by 4 label SD (regression), and a 2-class copy with 15 flipped labels:

  1. every in-loop detector returns one finite score per molecule through
     n07._compute_noise_scores, so the is_clf / groups hand-over works;
  2. net_small_loss, net_loss_mixture and net_josrc_test give the damaged
     labels a higher mean score than the clean ones (regression);
  3. each cleanup honours cleaned_training_set's contract, changes only what
     it says (rewrites touch the flagged labels only; weights are mean 1),
     and declines where it says it does (score_weight on a score outside
     [0, 1]; spread_invvar_weight on a non-positive score).

Cheap: no dataset, no result store. Meta-Weight-Net and the evidential
network train their full loop, so the whole file takes about a minute.

Run:  python tests/experiments/noise/_smoke_lnl_parts.py
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
warnings.filterwarnings('ignore')

import numpy as np

from kirby.utils.noise_estimation import ESTIMATOR_REGISTRY, IN_LOOP_DETECTORS
from experiments.noise.n07_remediation import (
    _compute_noise_scores, cleaned_training_set, SPLIT_CLEANUPS,
    FLAG_SHARE_CLEANUPS, last_decline_reason)

rng = np.random.RandomState(0)
N, P = 150, 20
X = rng.normal(size=(N, P)).astype(np.float32)
beta = rng.normal(size=P)
y_clean = X @ beta + 0.3 * rng.normal(size=N)
damaged = np.zeros(N, dtype=bool)
damaged[rng.choice(N, 15, replace=False)] = True
y = y_clean.copy()
y[damaged] += 4.0 * y_clean.std() * rng.choice([-1, 1], size=15)
groups = np.arange(N) // 5

y_clf_clean = (y_clean > np.median(y_clean)).astype(int)
flipped = np.zeros(N, dtype=bool)
flipped[rng.choice(N, 15, replace=False)] = True
y_clf = y_clf_clean.copy()
y_clf[flipped] = 1 - y_clf[flipped]

failures = []


def check(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg)
    if not cond:
        failures.append(msg)


print('1. detectors, regression')
scores = {}
for name in IN_LOOP_DETECTORS:
    assert name in ESTIMATOR_REGISTRY
    t0 = time.time()
    sc = _compute_noise_scores(X, y, is_clf=False, detector=name, groups=groups)
    dt = time.time() - t0
    check(sc is not None and sc.shape == (N,) and np.all(np.isfinite(sc)),
          f'{name}: one finite score per molecule ({dt:.1f}s)')
    scores[name] = sc
for name in ('net_small_loss', 'net_loss_mixture', 'net_josrc_test'):
    sc = scores[name]
    check(sc[damaged].mean() > sc[~damaged].mean(),
          f'{name}: damaged mean {sc[damaged].mean():.3g} > clean mean '
          f'{sc[~damaged].mean():.3g}')
check(scores['net_meta_weight'].min() >= 0 and scores['net_meta_weight'].max() <= 1,
      'net_meta_weight: in [0, 1]')
check(scores['net_evidential_spread'].min() > 0,
      'net_evidential_spread: positive variances')

print('2. detectors, classification (the regression-only one must refuse)')
for name in IN_LOOP_DETECTORS:
    if name == 'net_evidential_spread':
        # _compute_noise_scores turns a failed fit into None and prints why
        sc = _compute_noise_scores(X, y_clf, is_clf=True, detector=name,
                                   groups=groups)
        check(sc is None, f'{name}: refuses a classification set (None)')
        continue
    sc = _compute_noise_scores(X, y_clf, is_clf=True, detector=name,
                               groups=groups)
    check(sc is not None and sc.shape == (N,) and np.all(np.isfinite(sc)),
          f'{name}: one finite score per molecule')

print('3. cleanups')
sc = scores['net_small_loss']
sc01 = scores['net_josrc_test']
for c in SPLIT_CLEANUPS:
    assert c in SPLIT_CLEANUPS
check(set(('network_rewrite', 'ema_rewrite')) <= set(FLAG_SHARE_CLEANUPS),
      'the two rewrites read the flag share')

r = cleaned_training_set('score_weight', X, y, sc01, 0.1)
check(r is not None and r['sample_weight'] is not None
      and abs(r['sample_weight'].mean() - 1) < 1e-9
      and np.allclose(r['sample_weight'] * (1 - sc01).mean(), 1 - sc01),
      'score_weight: weight = (1 - score) / mean')
r = cleaned_training_set('score_weight', X, y, sc, 0.1)
check(r is None and 'not in [0, 1]' in (last_decline_reason() or ''),
      f'score_weight declines on a loss: {last_decline_reason()}')

ev = scores['net_evidential_spread']
r = cleaned_training_set('spread_invvar_weight', X, y, ev, 0.1)
check(r is not None and abs(r['sample_weight'].mean() - 1) < 1e-9
      and np.allclose(r['sample_weight'] * (1 / ev).mean(), 1 / ev),
      'spread_invvar_weight: weight = (1 / variance) / mean')
r = cleaned_training_set('spread_invvar_weight', X, y, sc01, 0.1)
check(r is None, f'spread_invvar_weight declines on a zero score: '
                 f'{last_decline_reason()}')

for c in ('network_rewrite', 'ema_rewrite'):
    t0 = time.time()
    r = cleaned_training_set(c, X, y, sc, 0.1)
    thr = np.percentile(sc, 90)
    fl = sc >= thr
    ok = (r is not None and r['label_column'] is not None
          and np.array_equal(r['y'][~fl], y[~fl])
          and not np.array_equal(r['y'][fl], y[fl])
          and r['sample_weight'] is None and r['keep'] is None)
    check(ok, f'{c}: changes the {int(fl.sum())} flagged labels only '
              f'({time.time() - t0:.1f}s)')
    if ok:
        closer = np.abs(r['y'][fl] - y_clean[fl]).mean()
        before = np.abs(y[fl] - y_clean[fl]).mean()
        print(f'       flagged labels: {before:.3f} from clean before, '
              f'{closer:.3f} after (support only, 150 random molecules)')
    sc_c = _compute_noise_scores(X, y_clf, is_clf=True,
                                 detector='net_small_loss', groups=groups)
    r = cleaned_training_set(c, X, y_clf, sc_c, 0.1, is_clf=True)
    thr = np.percentile(sc_c, 90)
    fl = sc_c >= thr
    check(r is not None and np.array_equal(r['y'][~fl], y_clf[~fl])
          and set(np.unique(r['y'])) <= {0, 1},
          f'{c}, classification: flagged labels replaced by a class')

print()
if failures:
    print(f'{len(failures)} FAILED:')
    for f in failures:
        print('  ' + f)
    sys.exit(1)
print('all passed')
