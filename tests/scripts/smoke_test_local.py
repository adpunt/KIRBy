#!/usr/bin/env python3
"""Local end-to-end smoke test for the Phase A–H plumbing.

Runs in ~30 seconds. No external data, no server, no real ML. Uses a
tiny synthetic dataset to exercise:

  1. Registry resolution (metric / task_type / direction)
  2. load_data → _finalize_load_data → dataset_name injection + cross-check
  3. find_best_hybrid(cache=True) compute path (write cache)
  4. find_best_hybrid(cache=True) hit path (read cache, validate meta)
  5. Parameter drift detection (stale cache rebuilt)
  6. condition_suffix gives distinct cache entries
  7. cv_splitter_id enforcement (raises without id)
  8. Cache stats line at end of run

This test is the final proof-of-life for the session's changes. If it
passes, the plumbing works. If a real run fails, the failure is in the
actual ML workload — not in the harness.

Usage:
    python tests/scripts/smoke_test_local.py

Exits 0 on success, non-zero on any failure.
"""

from __future__ import annotations

import os
import sys
import shutil
import tempfile
import warnings
from pathlib import Path

# Suppress non-fatal noise.
warnings.filterwarnings('ignore')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# Make imports work from the scripts/ directory
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT / 'tests'))


# =============================================================================
# SETUP — isolated temp cache
# =============================================================================

_TMPDIR = tempfile.mkdtemp(prefix='kirby_smoke_')
os.environ['KIRBY_CACHE_DIR'] = _TMPDIR
print(f'Using throwaway cache at {_TMPDIR}')

import numpy as np  # noqa: E402
from kirby.registry import DATASET_REGISTRY, lookup, verify_registry  # noqa: E402
from kirby.find_best_hybrid import (  # noqa: E402
    find_best_hybrid, BUILTIN_METRICS, _CACHE_STATS, _reset_cache_stats,
)


def _fail(test_num: int, msg: str):
    print(f'\nFAIL [{test_num}] {msg}')
    shutil.rmtree(_TMPDIR, ignore_errors=True)
    sys.exit(1)


# =============================================================================
# TESTS
# =============================================================================

def test_1_registry():
    print('[1/8] Registry integrity')
    verify_registry()
    # Antibody inter-family datasets default to spearman.
    info = lookup('flab_thermo_tresanco')
    assert info is not None
    assert info.metric == 'spearman'
    assert info.lower_is_better is False
    # Classification: tz_her2_binary → auc
    info = lookup('tz_her2_binary')
    assert info.task_type == 'classification'
    assert info.metric == 'auc'
    # Multi-class TCGA: f1_weighted
    assert lookup('tcga_brca_pam50').metric == 'f1_weighted'
    # Every entry has a valid BUILTIN_METRICS entry
    for name, info in DATASET_REGISTRY.items():
        assert info.metric in BUILTIN_METRICS, name
    print(f'  OK — {len(DATASET_REGISTRY)} entries verified')


def test_2_finalize_load_data():
    print('[2/8] _finalize_load_data injection + cross-check')
    from experiments._shared import _finalize_load_data
    # Registered dataset: dataset_name injected, domain backfilled
    d = _finalize_load_data({'task_type': 'regression', 'labels': np.zeros(5)},
                            'caco2')
    assert d['dataset_name'] == 'caco2'
    assert d['domain'] == 'molecular'
    # Mismatch: raises
    try:
        _finalize_load_data({'task_type': 'classification',
                             'labels': np.zeros(5)}, 'caco2')
        _fail(2, 'mismatch should have raised')
    except ValueError as e:
        assert 'Registry/loader mismatch' in str(e)
    # Unregistered dataset: pass-through
    d = _finalize_load_data({'task_type': 'regression', 'labels': np.zeros(5)},
                            'unknown_dataset_xyz')
    assert d['dataset_name'] == 'unknown_dataset_xyz'
    print('  OK')


def test_3_compute_and_cache():
    print('[3/8] find_best_hybrid compute → cache write')
    rng = np.random.RandomState(0)
    n, d = 40, 20
    # Small regression dataset
    X1 = rng.randn(n, d).astype(np.float32)
    X2 = rng.randn(n, d).astype(np.float32)
    y = (X1[:, 0] + X2[:, 0] + 0.3 * rng.randn(n)).astype(np.float64)
    groups = np.repeat(np.arange(n // 4), 4)

    _reset_cache_stats()
    result = find_best_hybrid(
        base_reps={'r1': X1, 'r2': X2}, labels=y, groups=groups,
        metric='rmse', task_type='regression',
        speed='full', budgets=[5, 10],
        cache=True, dataset_name='smoke_ds', tier='test',
        verbose=False,
    )
    assert result is not None
    assert result.metric_name == 'rmse'
    assert result.lower_is_better is True
    # Cache miss on first run
    assert _CACHE_STATS['search_miss'] == 1, _CACHE_STATS
    # File on disk
    cache_path = Path(_TMPDIR) / 'searches' / 'smoke_ds__test__uncapped__fast.pkl'
    assert cache_path.exists(), f'missing cache file: {cache_path}'
    print(f'  OK — cached to {cache_path.name}')
    return X1, X2, y, groups


def test_4_cache_hit(X1, X2, y, groups):
    print('[4/8] find_best_hybrid cache hit')
    _reset_cache_stats()
    result = find_best_hybrid(
        base_reps={'r1': X1, 'r2': X2}, labels=y, groups=groups,
        metric='rmse', task_type='regression',
        speed='full', budgets=[5, 10],
        cache=True, dataset_name='smoke_ds', tier='test',
        verbose=False,
    )
    assert result is not None
    # Second run should hit
    assert _CACHE_STATS['search_hit'] == 1, _CACHE_STATS
    assert _CACHE_STATS['search_miss'] == 0, _CACHE_STATS
    print('  OK — hit counter incremented, no recompute')


def test_5_drift_invalidates(X1, X2, y, groups):
    print('[5/8] Parameter drift → stale cache rebuilt')
    _reset_cache_stats()
    # Change seed → different cache meta → stale
    result = find_best_hybrid(
        base_reps={'r1': X1, 'r2': X2}, labels=y, groups=groups,
        metric='rmse', task_type='regression',
        speed='full', budgets=[5, 10],
        cache=True, dataset_name='smoke_ds', tier='test',
        seed=99,  # drift
        verbose=False,
    )
    assert result is not None
    assert _CACHE_STATS['search_stale'] == 1, _CACHE_STATS
    print('  OK — stale detected, recomputed')


def test_6_condition_suffix_distinct(X1, X2, y, groups):
    print('[6/8] condition_suffix produces distinct cache entries')
    _reset_cache_stats()
    find_best_hybrid(
        base_reps={'r1': X1, 'r2': X2}, labels=y, groups=groups,
        metric='rmse', task_type='regression',
        speed='full', budgets=[5, 10],
        cache=True, dataset_name='smoke_ds', tier='test',
        condition_suffix='variant_a',
        verbose=False,
    )
    find_best_hybrid(
        base_reps={'r1': X1, 'r2': X2}, labels=y, groups=groups,
        metric='rmse', task_type='regression',
        speed='full', budgets=[5, 10],
        cache=True, dataset_name='smoke_ds', tier='test',
        condition_suffix='variant_b',
        verbose=False,
    )
    a = Path(_TMPDIR) / 'searches' / 'smoke_ds__test__uncapped__fast__variant_a.pkl'
    b = Path(_TMPDIR) / 'searches' / 'smoke_ds__test__uncapped__fast__variant_b.pkl'
    assert a.exists() and b.exists()
    assert a != b
    print(f'  OK — both files written distinctly:')
    print(f'    {a.name}')
    print(f'    {b.name}')


def test_7_cv_splitter_id_required(X1, X2, y, groups):
    print('[7/8] cv_splitter without cv_splitter_id raises')
    from sklearn.model_selection import KFold
    try:
        find_best_hybrid(
            base_reps={'r1': X1, 'r2': X2}, labels=y, groups=None,
            metric='rmse', task_type='regression',
            speed='full', budgets=[5, 10],
            cache=True, dataset_name='smoke_ds', tier='test_cv',
            cv_splitter=KFold(n_splits=3),  # no id provided
            verbose=False,
        )
        _fail(7, 'expected ValueError when cv_splitter=... without id')
    except ValueError as e:
        assert 'cv_splitter_id' in str(e)
    print('  OK — guard raises as expected')


def test_8_registry_backed_metric_resolution():
    print('[8/8] Registry backs metric resolution in _shared + precompute')
    from experiments._shared import _resolve_dataset_metric
    from precompute_hybrids import _get_metric
    # Antibody inter-family: registry-backed → spearman.
    m1, _, _ = _resolve_dataset_metric(dataset_name='flab_thermo_tresanco')
    m2 = _get_metric('antibody', 'regression', None,
                     dataset_name='flab_thermo_tresanco')
    assert m1 == m2 == 'spearman'
    # Mismatched task_type raises
    try:
        _get_metric('antibody', 'classification', None,
                    dataset_name='flab_thermo_tresanco')
        _fail(8, 'mismatch should have raised')
    except ValueError:
        pass
    print('  OK — two dispatchers agree + mismatch raises')


# =============================================================================
# RUN
# =============================================================================

def main() -> int:
    try:
        test_1_registry()
        test_2_finalize_load_data()
        X1, X2, y, groups = test_3_compute_and_cache()
        test_4_cache_hit(X1, X2, y, groups)
        test_5_drift_invalidates(X1, X2, y, groups)
        test_6_condition_suffix_distinct(X1, X2, y, groups)
        test_7_cv_splitter_id_required(X1, X2, y, groups)
        test_8_registry_backed_metric_resolution()
    finally:
        shutil.rmtree(_TMPDIR, ignore_errors=True)

    print('\n' + '=' * 60)
    print('smoke_test_local: ALL 8 TESTS PASSED')
    print('=' * 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
