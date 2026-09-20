#!/usr/bin/env python3
"""Whole-run smoke for n11, the estimator x correction cross.

WHY THIS EXISTS. n11 has never written a result file, and the two check files
that already import it both stop at the inner cell. `_smoke_n08_n12.py` runs
`_run_estimator_grid` and checks the classification skip partition;
`_smoke_repair_targets.py` runs it again and checks that the two prefixed
smoothing rules and the clip rule come out under the right keys. Neither one
calls `run()`, which is where the defects that stop a result file living are:
the pool lookup that returns early, the downstream roster read from the wrong
place, the draw count that ignores its config, and the swept lists whose names
a cut-down harness rebinds without hitting a loop.

Nothing here overlaps those two. The grid keys, the clip and the skip partition
are theirs; this file takes `run()` from its first fold to its summary.

Everything is synthetic and in-process. No dataset, no representation cache, no
result store, no find_best_hybrid search. The pool lookup is replaced with a
blueprint built in this file, so a stale or absent result store cannot stop it
running -- which is the state the store was in on 2026-09-20.

Run:
    python tests/experiments/noise/_smoke_n11.py
Exit 0 = all green.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
           'NUMBA_NUM_THREADS', 'KIRBY_LGBM_THREADS', 'KIRBY_RF_JOBS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import types
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')

import numpy as np  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402

from experiments.noise import n11_estimation_remediation_grid as n11  # noqa: E402

FAILS = []


def check(name, cond, detail=''):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ''))
    if not cond:
        FAILS.append(name)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------
def _tiny_data(n=120, d=8, seed=0):
    """A data dict with the keys run() reads, and nothing else.

    No 'smiles', so noise_groups_for returns no groups and the two grouped
    noise conditions are never reachable from here -- which is what we want, the
    injector under test is the plain Gaussian one.

    'blosum' is one of the names pick_noise_base_rep prefers, so the control
    representation is decided here rather than alphabetically.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    y = X[:, 0] * 2.0 + X[:, 1] - 0.5 * X[:, 2] + 0.25 * rng.randn(n)
    return {
        'reps': {'blosum': X, 'other': rng.randn(n, d)},
        'labels': y,
        'groups': np.repeat(np.arange(n // 4), 4)[:n],
        'augmentations': None,
        'task_type': 'regression',
        'dataset_name': 'smoke_n11',
        'domain': 'molecular',
    }


def _blueprint(budget=12):
    return types.SimpleNamespace(
        allocation_method='performance_weighted',
        importance_method='lightgbm_gain',
        internal_evaluator='match',
        budget=budget,
        augmentation_strategy=None,
    )


def _fake_pool(*_a, **_kw):
    """Stand in for get_best_search_for_downstream.

    One family only. run() builds one estimation space per family the pool
    holds, so a second entry here would double the grid and slow the smoke
    without testing anything the first does not.
    """
    meta = {
        'pool_subset': 'predictive_and_similarity',
        'objective': 'predictive',
        'by_objective': {
            'predictive': {
                'tier': 'general', 'aug_condition': 'none',
                'score_mean': 0.5, 'metric_name': 'spearman',
                'lower_is_better': False,
                'best_config': {'blueprint': _blueprint()},
            },
        },
    }
    return object(), meta


def _spearman(a, b):
    from scipy.stats import spearmanr
    r = spearmanr(np.asarray(a), np.asarray(b)).correlation
    return 0.0 if not np.isfinite(r) else float(r)


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


class _Patched:
    """Swap out the pool lookup, the metric, the estimator roster and the sweep.

    WHAT IS CUT AND WHY IT IS ONLY CUT HERE. Which estimator to pair with which
    correction is the question n11 exists to answer, so the cross is never cut
    in a run. These tests ask whether run() reaches its end and fills the fields
    it declares, and one value of each swept knob answers that as well as four
    do. At the full sweep this file took 25 minutes and stopped being a smoke.

        estimators      2 of the 10, one neighbour-based and one model-based
        distances       1 of the 4
        smoothing alpha 1 of the 4, weight strength 1 of the 4
        smoothing/weighting methods and the targets: NOT cut, because the cell
                        count per target is what test 5 and test 6 read.

    Test 5 restores the two swept lists itself -- it is the one test whose
    subject is those lists -- so it is unaffected by the cut here.
    """

    def __init__(self, metric_name, metric_fn, lower_is_better,
                 estimators=('neighbor_residual', 'cross_val')):
        self.metric = (metric_name, metric_fn, lower_is_better)
        self.estimators = list(estimators)
        self._saved = {}

    def __enter__(self):
        for k in ('get_best_search_for_downstream', 'get_domain_metric',
                  'BASE_ESTIMATORS', 'TORCH_ESTIMATORS', 'OPTIONAL_ESTIMATORS',
                  'NEIGHBOR_METRICS', 'SMOOTH_ALPHAS', 'WEIGHT_STRENGTHS',
                  'ALPHAS'):
            self._saved[k] = getattr(n11, k)
        n11.get_best_search_for_downstream = _fake_pool
        n11.get_domain_metric = lambda _d: self.metric
        n11.BASE_ESTIMATORS = self.estimators
        n11.TORCH_ESTIMATORS = []
        n11.OPTIONAL_ESTIMATORS = []
        n11.NEIGHBOR_METRICS = ['manhattan']
        n11.SMOOTH_ALPHAS = [0.5]
        n11.WEIGHT_STRENGTHS = [1.0]
        n11.ALPHAS = n11.SMOOTH_ALPHAS
        return self

    def __exit__(self, *_exc):
        for k, v in self._saved.items():
            setattr(n11, k, v)
        return False


def _cells(sigma_result):
    """Every scored cell of one level, as (space, estimator, remediation, dict)."""
    out = []
    for sp, sd in sigma_result.get('per_space', {}).items():
        for est, ed in sd.items():
            for rk, rv in ed.items():
                if rk.startswith('_') or not isinstance(rv, dict):
                    continue
                if 'denoised_score' not in rv:
                    continue
                out.append((sp, est, rk, rv))
    return out


# ---------------------------------------------------------------------------
# 1. run() reaches the end and writes scored cells.
# ---------------------------------------------------------------------------
def test_run_produces_cells():
    print("n11.run() end to end (spearman, higher is better):")
    data = _tiny_data()
    with _Patched('spearman', _spearman, False):
        res = n11.run(data, {'n_splits': 2, 'models': ['rf']}, None, 'smoke_n11',
                      sigmas=[0.5], noise_seeds=[0])
    check('no early return', 'error' not in res, res.get('error', ''))
    check("the level asked for is the level reported",
          list(res.get('per_sigma', {})) == ['0.5'],
          str(list(res.get('per_sigma', {}))))
    sr = res['per_sigma']['0.5']
    cells = _cells(sr)
    check('at least one scored cell exists', len(cells) > 0, f'{len(cells)} cells')
    check('every cell carries a finite denoised score',
          all(np.isfinite(rv['denoised_score']) for _, _, _, rv in cells))
    check('every cell names its model and its remediation',
          all(rv.get('model') and rv.get('remediation') for _, _, _, rv in cells))
    spaces = {sp for sp, _, _, _ in cells}
    check('the control representation has its own space',
          'blosum_raw' in spaces, str(sorted(spaces)))
    check('the pool family got a space of its own',
          any(s.startswith('hybrid_predictive') for s in spaces),
          str(sorted(spaces)))
    ho = sr.get('heldout_recovery')
    check('a recovery picked without the fold it is read on was computed',
          isinstance(ho, dict) and ho.get('n_configs_ranked', 0) > 0,
          '' if not isinstance(ho, dict) else
          f"{ho.get('n_configs_ranked')} configs ranked")
    return res


# ---------------------------------------------------------------------------
# 2. The downstream roster comes from config['models'], not from the module
#    default. n11 read kwargs alone until 2026-09-20 and silently fitted five
#    models when the runner had asked for one.
# ---------------------------------------------------------------------------
def test_models_come_from_config():
    print("downstream roster reaches run() through config['models']:")
    data = _tiny_data()
    with _Patched('spearman', _spearman, False):
        res = n11.run(data, {'n_splits': 2, 'models': ['rf']}, None, 'smoke_n11',
                      sigmas=[0.5], noise_seeds=[0])
    sr = res['per_sigma']['0.5']
    models = {rv['model'] for _, _, _, rv in _cells(sr)}
    check('only the one model config asked for was fitted',
          models == {'rf'}, str(sorted(models)))
    check('the clean and noisy references are that same one model',
          set(sr['clean_score_by_model']) == {'rf'}
          and set(sr['noisy_score_by_model']) == {'rf'},
          f"clean={sorted(sr['clean_score_by_model'])}")


# ---------------------------------------------------------------------------
# 3. The draw count. run() reads kwargs.get('noise_seeds', [0, 1, 2]) as a
#    literal default with no module constant behind it, so a config alone
#    cannot shrink it and a harness has to pass it as a keyword.
#
#    The observable is results['noise_provenance']. Its key is the spec id of
#    every noise draw that was actually injected, and a spec id carries the
#    draw's seed ('..._s0'), so one key per (level, draw). NOT the detection
#    pass count, which is undefined here: the Gaussian condition moves every
#    training label, so there is no corrupted subset to find and
#    noise_helpers.DETECTION_IS_DEFINED excludes it.
# ---------------------------------------------------------------------------
def test_noise_seeds_must_be_passed():
    print("draw count (one noise_provenance key per level x draw):")
    data = _tiny_data()
    with _Patched('spearman', _spearman, False, estimators=['neighbor_residual']):
        passed = n11.run(data, {'n_splits': 2, 'models': ['rf']}, None,
                         'smoke_n11', sigmas=[0.5], noise_seeds=[0])
        in_config = n11.run(data, {'n_splits': 2, 'models': ['rf'],
                                   'noise_seeds': [0]},
                            None, 'smoke_n11', sigmas=[0.5])
    n_passed = len(passed.get('noise_provenance', {}))
    n_config = len(in_config.get('noise_provenance', {}))
    check('one draw passed as a keyword injects one noise draw at one level',
          n_passed == 1, f'{n_passed} draws injected')
    check('the same list in config alone does NOT shrink it — the literal '
          'default of three draws is injected instead',
          n_config == 3, f'{n_config} draws injected')
    check('the draws are distinct, one seed each',
          len(set(in_config.get('noise_provenance', {}))) == n_config,
          str(sorted(in_config.get('noise_provenance', {}))))


# ---------------------------------------------------------------------------
# 4. Metric direction. The gap is sign-normalised so a positive gap always
#    means the injected noise cost the model something, on both directions.
# ---------------------------------------------------------------------------
def test_gap_sign_both_directions():
    print("clean-to-noisy gap, both metric directions:")
    data = _tiny_data()
    for mname, mfn, lib in (('spearman', _spearman, False), ('rmse', _rmse, True)):
        with _Patched(mname, mfn, lib, estimators=['neighbor_residual']):
            res = n11.run(data, {'n_splits': 2, 'models': ['rf']}, None,
                          'smoke_n11', sigmas=[0.5], noise_seeds=[0])
        sr = res['per_sigma']['0.5']
        c, nz, gap = sr['clean_score'], sr['noisy_score'], sr['gap']
        damaged = (nz > c) if lib else (c > nz)
        check(f'{mname}: gap is positive exactly when the noise did damage',
              (gap > 0) == damaged,
              f'clean={c:.4f} noisy={nz:.4f} gap={gap:+.4f} '
              f'lower_is_better={lib}')
        check(f'{mname}: lower_is_better is recorded on every cell',
              all(rv['lower_is_better'] is lib for _, _, _, rv in _cells(sr)))


# ---------------------------------------------------------------------------
# 5. THE CUT CONTRACT. A harness shrinks n11 by rebinding its module lists.
#    Each list must be the one the loop actually reads. `ALPHAS` is an alias
#    bound once at import and is written into the result file only — rebinding
#    it changes no loop, and a harness that rebinds it has cut nothing.
# ---------------------------------------------------------------------------
def test_swept_lists_are_the_lists_the_loops_read():
    print("the swept lists a harness rebinds:")
    rng = np.random.RandomState(0)
    n = 90
    X = StandardScaler().fit_transform(rng.randn(n, 6))
    y = X[:, 0] + 0.2 * rng.randn(n)
    tr, va = np.arange(0, 70), np.arange(70, n)
    fac = {'rf': lambda: RandomForestRegressor(n_estimators=10, random_state=0)}

    def grid(**kw):
        r = n11._run_estimator_grid('neighbor_residual', 'manhattan',
                                    X[tr], y[tr], X[tr], X[va], y[va],
                                    fac, _rmse, n, is_clf=False, **kw)
        return [k for k in r if not k.startswith('_')]

    saved = (n11.WEIGHT_STRENGTHS, n11.SMOOTH_ALPHAS, n11.ALPHAS)
    try:
        base = grid()
        n_w = len(n11.WEIGHT_STRENGTHS) * len(n11.WEIGHT_METHODS)
        n_s = len(n11.SMOOTH_ALPHAS) * len(n11.SMOOTH_METHODS)
        check(f'the whole grid is {n_w} weighting cells + {n_s} smoothing cells',
              len(base) == n_w + n_s, f'{len(base)} cells')

        n11.ALPHAS = [0.3]
        check('rebinding ALPHAS alone changes NOTHING — it is written into the '
              'result file and read by no loop',
              len(grid()) == len(base), f'{len(grid())} cells')
        n11.ALPHAS = saved[2]

        n11.WEIGHT_STRENGTHS = [3.0]
        half_w = grid()
        check('rebinding WEIGHT_STRENGTHS drops the weighting cells',
              len(half_w) == len(n11.WEIGHT_METHODS) + n_s,
              f'{len(half_w)} cells')
        n11.WEIGHT_STRENGTHS = saved[0]

        n11.SMOOTH_ALPHAS = [0.5]
        half_s = grid()
        check('rebinding SMOOTH_ALPHAS drops the smoothing cells',
              len(half_s) == n_w + len(n11.SMOOTH_METHODS),
              f'{len(half_s)} cells')
    finally:
        n11.WEIGHT_STRENGTHS, n11.SMOOTH_ALPHAS, n11.ALPHAS = saved


# ---------------------------------------------------------------------------
# 6. EXTRA_TARGETS is the list run() declares and writes into the result file.
#    The build loop must read it, or the file names one set of targets while
#    carrying rows from another.
# ---------------------------------------------------------------------------
def test_extra_targets_list_is_honoured():
    print("EXTRA_TARGETS is honoured by run(), not only reported:")
    data = _tiny_data()
    saved = n11.EXTRA_TARGETS
    try:
        n11.EXTRA_TARGETS = ('oof',)
        with _Patched('spearman', _spearman, False,
                      estimators=['neighbor_residual']):
            res = n11.run(data, {'n_splits': 2, 'models': ['rf']}, None,
                          'smoke_n11', sigmas=[0.5], noise_seeds=[0])
    finally:
        n11.EXTRA_TARGETS = saved
    sr = res['per_sigma']['0.5']
    keys = [rk for _, _, rk, _ in _cells(sr)]
    check("the file says the targets it was asked for",
          res['extra_smoothing_targets'] == ['oof'],
          str(res['extra_smoothing_targets']))
    check('a row under the asked-for target is present',
          any('|oof+' in k for k in keys))
    check('NO row under a target that was not asked for',
          not any('|gp+' in k for k in keys),
          str(sorted({k.split('|', 1)[1] for k in keys if '|gp+' in k})[:3]))


def main():
    print("=" * 66)
    print("SMOKE: n11 estimation x remediation grid, whole run")
    print("=" * 66)
    test_run_produces_cells()
    test_models_come_from_config()
    test_noise_seeds_must_be_passed()
    test_gap_sign_both_directions()
    test_swept_lists_are_the_lists_the_loops_read()
    test_extra_targets_list_is_honoured()
    print("=" * 66)
    if FAILS:
        print(f"RESULT: {len(FAILS)} FAILURES: {FAILS}")
        sys.exit(1)
    print("RESULT: ALL GREEN")


if __name__ == '__main__':
    main()
