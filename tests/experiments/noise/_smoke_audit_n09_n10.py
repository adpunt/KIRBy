#!/usr/bin/env python3
"""Smoke tests for the n09 AUDIT_NOISE fixes (P1#4, importance-weight
cancellation). Fast: pure-numpy property checks, no find_best_hybrid, no server.

Covers:
  - P1#4 (n09) importance weighting: MEAN-1 sqrt(importance) weighting
    preserves relative column weights AND restores ~unit overall scale, whereas
    the old SUM-1 weighting crushed every column toward zero (~1/n_cols).
  - The StandardScaler-after-multiply CANCELLATION no-op: weighting a column
    then z-scoring it returns the plain z-scored column (weighting vanishes) —
    which is exactly why n09 now standardises FIRST, then weight.

Run:  python tests/experiments/noise/_smoke_audit_n09_n10.py
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))


def smoke_standardscaler_cancellation():
    """Proof of the no-op bug the fix avoids: StandardScaler(X * w) == StandardScaler(X).

    A per-column multiply by w followed by a per-column StandardScaler divides
    each column by a std that was itself scaled by w, exactly cancelling w. So
    the OLD 'multiply then standardize' made the weighted space identical to the
    unweighted one. The fix standardizes FIRST, then multiplies, which does NOT
    cancel.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 12)) * rng.uniform(0.5, 5, size=12)  # varied scales
    w = rng.uniform(0.1, 3.0, size=12)

    old_weighted = StandardScaler().fit_transform(X * w)   # buggy order
    plain_std = StandardScaler().fit_transform(X)
    # The bug: these are identical (up to fp noise).
    assert np.allclose(old_weighted, plain_std, atol=1e-9), (
        'expected StandardScaler(X*w) == StandardScaler(X) (the cancellation)')

    new_weighted = StandardScaler().fit_transform(X) * w   # fixed order
    # The fix: standardize-first genuinely changes the geometry.
    assert not np.allclose(new_weighted, plain_std, atol=1e-6), (
        'standardize-then-weight must NOT equal plain standardized')
    # And column variances now reflect w^2.
    col_var = new_weighted.var(axis=0)
    assert np.corrcoef(col_var, w ** 2)[0, 1] > 0.99, (
        'standardize-then-weight column variance should track w^2')
    return ('cancellation confirmed: multiply-then-standardize is a no-op; '
            'standardize-then-multiply changes geometry (var ~ w^2)')


def smoke_mean1_vs_sum1_scale():
    """MEAN-1 sqrt-importance weighting keeps relative weights and ~unit scale;
    SUM-1 crushes the overall scale to ~1/n_cols (the n09 collapse)."""
    rng = np.random.default_rng(1)
    n_cols = 100
    Xs = StandardScaler().fit_transform(rng.normal(size=(300, n_cols)))
    imp = rng.uniform(0, 1, size=n_cols)
    w_raw = np.sqrt(np.maximum(imp, 0))

    w_sum1 = w_raw / (w_raw.sum() + 1e-10)
    w_mean1 = w_raw / (w_raw.mean() + 1e-10)

    # Relative weighting identical (both are scalar multiples of w_raw).
    ratio = w_mean1 / (w_sum1 + 1e-30)
    assert np.allclose(ratio, ratio[0]), 'mean-1 and sum-1 must be proportional'

    rms_sum1 = np.sqrt(np.mean((Xs * w_sum1) ** 2))
    rms_mean1 = np.sqrt(np.mean((Xs * w_mean1) ** 2))
    # SUM-1 collapses scale (~1/n_cols); MEAN-1 stays O(1).
    assert rms_sum1 < 0.05, f'sum-1 should crush scale, got rms={rms_sum1:.4f}'
    assert 0.3 < rms_mean1 < 3.0, f'mean-1 should keep ~unit scale, got {rms_mean1:.4f}'
    assert rms_mean1 > 10 * rms_sum1, 'mean-1 must restore scale vs sum-1'
    return (f'sum-1 rms={rms_sum1:.4f} (collapsed) vs '
            f'mean-1 rms={rms_mean1:.3f} (~unit); relative weights preserved')


SMOKE_TESTS = [
    ('n09 StandardScaler cancellation no-op', smoke_standardscaler_cancellation),
    ('n09 mean-1 vs sum-1 weighting scale', smoke_mean1_vs_sum1_scale),
]


def main():
    print('=' * 70)
    print('SMOKE TESTS — n09 AUDIT_NOISE fixes')
    print('=' * 70)
    failed = 0
    for name, fn in SMOKE_TESTS:
        try:
            print(f'[PASS] {name}\n       {fn()}')
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            print(f'[FAIL] {name}\n       {type(e).__name__}: {e}')
            traceback.print_exc()
    print('-' * 70)
    print(f'{len(SMOKE_TESTS) - failed}/{len(SMOKE_TESTS)} passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
