#!/usr/bin/env python3
"""Smoke: a search says which way its metric runs, and says it once.

Every search carries two fields that answer the same question. `metric_name`
answers it by name -- 'rmse' means lower is better. `lower_is_better` answers it
as a flag. Anything that reads a search reads one or the other, so they have to
agree, or the reader either refuses the search or ranks it backwards.

They can come apart in one place. Under a blend objective the search still fits
the model with the dataset's own metric, but it picks the winner on a weighted
mean of correlations, which is always higher-is-better, so the flag is False
while the metric is still RMSE. nested_cv_evaluate renames the metric at the
moment it flips the flag. find_best_hybrid did not until 2026-09-17, so a blend
search on an RMSE dataset came back saying metric_name='rmse' and
lower_is_better=False, and build_honest_rescore_input raised
"SearchResult direction contradiction" on it.

Checked on an RMSE dataset, for all four objectives, on the search and on the
blueprint it hands back:

  1. the two fields agree.
  2. build_honest_rescore_input accepts the search and reports the same
     direction the search used.
  3. a blueprint rebuilt through apply_blueprint's own resolution lands on the
     direction the search used, not the opposite one.

The registry is checked too: every dataset's stored direction against
BUILTIN_METRICS.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault('KIRBY_LGBM_THREADS', '1')

# Torch before anything that loads LightGBM. Each loads its own OpenMP runtime
# and whichever loads second kills the process -- seen here as exit 139 with no
# traceback, on the run before this import was added.
try:
    import torch  # noqa: F401
except ImportError:
    pass

import warnings

import numpy as np

warnings.filterwarnings('ignore')

import experiments.dataset_policy  # noqa: F401  (populates the registry)
import kirby.find_best_hybrid as F
from kirby.find_best_hybrid import find_best_hybrid, BUILTIN_METRICS
from kirby.registry import all_datasets
from experiments._shared import build_honest_rescore_input

FAIL = []

# Four objectives. 'predictive' and 'similarity' are what precompute writes;
# the other two are the blends, which are the pair that can come apart.
OBJECTIVES = [
    'predictive',
    'predictive_corr',
    'similarity',
    {'predictive': 1.0, 'similarity': 1.0},
]


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ''))
    if not ok:
        FAIL.append(name)


def test_registry_directions():
    print("registry:")
    ds = all_datasets()
    bad = [(n, i.metric, i.lower_is_better) for n, i in ds.items()
           if i.lower_is_better != BUILTIN_METRICS[i.metric][1]]
    n_lower = sum(1 for i in ds.values() if i.lower_is_better)
    check('every dataset agrees with BUILTIN_METRICS',
          not bad,
          f"{len(ds)} datasets, {n_lower} of them lower-is-better; "
          f"{len(bad)} disagree" + (f" {bad}" if bad else ''))


def make_data(n=150, seed=0):
    rng = np.random.default_rng(seed)
    reps = {'a': rng.normal(size=(n, 8)), 'b': rng.normal(size=(n, 6))}
    y = 2 * reps['a'][:, 0] - reps['b'][:, 1] + rng.normal(0, 0.3, n)
    groups = np.arange(n) // 10
    return reps, y, groups


def direction_apply_blueprint_would_use(bp):
    """apply_blueprint's own resolution order, so a change there fails here."""
    objective = getattr(bp, 'objective', 'predictive')
    if F._blend_needs_model(objective):
        return bool(bp.lower_is_better), 'the blueprint flag'
    if bp.metric in BUILTIN_METRICS:
        return BUILTIN_METRICS[bp.metric][1], 'the metric name'
    return bool(bp.lower_is_better), 'the blueprint flag'


def test_search_direction(objective):
    label = (objective if isinstance(objective, str)
             else 'blend ' + '+'.join(sorted(objective)))
    print(f"objective={label}, metric=rmse:")
    reps, y, groups = make_data()
    res = find_best_hybrid(
        reps, y, groups=groups, metric='rmse', task_type='regression',
        models=['rf'], budgets=[10], allocation_methods=['performance_weighted'],
        importance_methods=['lightgbm_gain'], n_folds=3, objective=objective,
        verbose=False, tune=False)

    name, flag = res.metric_name, bool(res.lower_is_better)
    if name in BUILTIN_METRICS:
        check(f'{label}: metric name and flag agree',
              BUILTIN_METRICS[name][1] == flag,
              f"metric_name={name!r} means lower_is_better="
              f"{BUILTIN_METRICS[name][1]}, the search carries {flag}")
    else:
        check(f'{label}: metric name is not a dataset metric, so nothing '
              f'contradicts the flag',
              True, f"metric_name={name!r}, lower_is_better={flag}")

    try:
        blk = build_honest_rescore_input(res)
        check(f'{label}: build_honest_rescore_input accepts the search',
              bool(blk['lower_is_better']) == flag,
              f"block says lower_is_better={blk['lower_is_better']}, "
              f"the search used {flag}")
    except Exception as exc:
        check(f'{label}: build_honest_rescore_input accepts the search',
              False, f'{type(exc).__name__}: {exc}')

    bp = F.Blueprint.from_dict(res.all_results[0]['blueprint'].to_dict())
    got, via = direction_apply_blueprint_would_use(bp)
    check(f'{label}: a rebuilt blueprint uses the direction the search used',
          got == flag,
          f"blueprint gives lower_is_better={got} from {via}, "
          f"the search used {flag}")


if __name__ == '__main__':
    test_registry_directions()
    print()
    for objective in OBJECTIVES:
        test_search_direction(objective)
        print()
    print("=" * 60)
    if FAIL:
        print(f"SMOKE FAILED ({len(FAIL)}): {FAIL}")
        sys.exit(1)
    print("ALL GREEN")
