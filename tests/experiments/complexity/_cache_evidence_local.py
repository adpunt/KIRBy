"""§V CACHE EVIDENCE driver (complexity lane, local smoke).

Proves the new per-dataset SQLite result store (src/kirby/result_cache.py) is
exercised AND reused by find_best_hybrid:

  RUN 1 (cold): empty result_db -> computes the grid, stores N rows.
  RUN 2 (warm): same KIRBY_CACHE_DIR -> prints
                'ResultCache: N stored configs reused' and NO 'meta drift'.
  RUN 3 (changed model list): keep the old model, ADD one new model -> the old
                rows are reused (SELECT) and ONLY the new model is computed.

This is NOT a server experiment; it trims the search axes in-process so a laptop
can run it in seconds. Server jobs import find_best_hybrid fresh and are
unaffected (these are runtime monkeypatches of imported objects, like
_run_complexity_locally.py).

Usage:
  KIRBY_CACHE_DIR=/tmp/kirby_V_cev python tests/experiments/complexity/_cache_evidence_local.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')

import experiments._shared as _shared  # noqa: E402
from experiments._shared import load_data, get_domain_metric  # noqa: E402
import kirby.find_best_hybrid as _fbh  # noqa: E402
from kirby.find_best_hybrid import find_best_hybrid  # noqa: E402

# Trim the search axes the precompute reads (smoke-only, in-process).
_shared.FULL_BUDGETS = [25]
_shared.INTERNAL_EVALUATORS = ['rf']
_fbh.DEFAULT_INTERNAL_EVALUATORS = ['rf']
_fbh.DEFAULT_ALLOCATION_METHODS = _fbh.DEFAULT_ALLOCATION_METHODS[:1]
_fbh.DEFAULT_IMPORTANCE_METHODS = ['treeshap']

DATASET = os.environ.get('KIRBY_LOCAL_DS', 'buchwald_hartwig')
REPS = os.environ.get('KIRBY_LOCAL_REPS', 'diff_maccs,onehot_components').split(',')


def _make_rf(**kw):
    # find_best_hybrid deep-copies dict values as estimator INSTANCES, so the
    # model dict maps name -> estimator instance (this is what c06's
    # var_factory() yields).
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=50, random_state=0, **kw)


def _search(models, label):
    print('\n' + '=' * 70)
    print(label)
    print('=' * 70)
    import numpy as np
    data = load_data(DATASET, rep_names=REPS, rep_preset='fast')
    # Smoke subsample (keeps the cold run tractable on a laptop). Honors the
    # standard _shared row cap; server runs never set KIRBY_SMOKE_N.
    if os.environ.get('KIRBY_SMOKE_N', '').strip():
        data = _shared._smoke_subsample(data)
    reps = dict(data['reps'])
    # Guarantee >=3 reps + a multi-budget grid so the search produces stored
    # rows (with only 2 reps top_2 is skipped and the grid can collapse to 0
    # configs). The 3rd rep is a deterministic PCA-ish transform of the first —
    # purely to populate the leave-one-out rep_subset grid; this is a cache
    # mechanism test, not a science result.
    if len(reps) < 3:
        first = next(iter(reps.values()))
        rng = np.random.RandomState(0)
        proj = rng.standard_normal((first.shape[1], min(16, first.shape[1])))
        reps['synthetic_proj'] = (first @ proj).astype(np.float32)
    metric_name, _, _ = get_domain_metric(data)
    res = find_best_hybrid(
        base_reps=reps,
        labels=data['labels'],
        groups=data['groups'],
        models=models,
        budgets=[10, 25],
        task_type='regression',
        metric=metric_name,
        speed='full',
        seed=42,
        n_folds=2,
        cache=True,
        test_rep_subsets=True,
        dataset_name='cev_buchwald',
        tier='cache_evidence',
        internal_evaluators=['rf'],
        verbose=True,
    )
    return res


if __name__ == '__main__':
    # Cold + warm with identical model set.
    _search({'rf': _make_rf()}, 'RUN 1 (COLD): empty result_db, should COMPUTE')
    _search({'rf': _make_rf()}, 'RUN 2 (WARM): same cache, should REUSE (no meta drift)')
    # Changed model list: old 'rf' reused, only new 'rf_deep' computed.
    _search({'rf': _make_rf(), 'rf_deep': _make_rf(max_depth=8)},
            'RUN 3 (CHANGED MODELS): reuse rf rows, compute ONLY rf_deep')
