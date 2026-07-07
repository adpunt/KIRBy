#!/usr/bin/env python3
"""Non-degeneracy smoke test for noise experiment n07.

Built during the 2026-06-03 noise audit. This goes beyond "does run() crash" —
it asserts the output is NOT degenerate in the specific way the audit found:

  * n07: recovery percentages must be finite for at least one strategy, and the
    recovery sign convention must be self-consistent with the metric direction
    (a strategy that moves the score toward `clean` yields positive recovery).

Reuses the validated _smoke_harness machinery (build_data / tiny_config /
precompute) for cheap local reps. Run directly:

    python tests/experiments/noise/_smoke_n04_n07.py            # n07 (reg)
    python tests/experiments/noise/_smoke_n04_n07.py n07 --clf  # clf branch

Exit 0 = all green. Non-zero = a degeneracy assertion fired.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import argparse
import warnings
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')

import numpy as np  # noqa: E402
import experiments.noise._smoke_harness as H  # noqa: E402


OUT = '/tmp/noise_smoke'


def _finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def _prep(clf, subsample):
    data, ds = H.build_data(clf=clf, subsample=subsample)
    config = H.tiny_config()
    data = H.precompute(data, ds, config)
    return data, ds, config


def check_n07(clf=False, subsample=200):
    """n07 recovery must be finite and sign-consistent with metric direction."""
    data, ds, config = _prep(clf, subsample)
    from experiments.noise import n07_remediation as n07
    res = n07.run(data, config, OUT, ds)
    summ = res['summary']
    lower_is_better = res['lower_is_better']
    clean = summ['clean_score']
    noisy = summ['noisy_score']
    rec = summ['recovery_pct']

    finite_rec = {k: v for k, v in rec.items() if '/' not in k and _finite(v)}
    print(f"  [n07] clean={clean:.4f} noisy={noisy:.4f} "
          f"lower_is_better={lower_is_better}")
    print(f"  [n07] hybrid recovery%: "
          f"{ {k: round(v, 1) for k, v in finite_rec.items()} }")
    assert finite_rec, "DEGENERATE: no finite hybrid recovery values."

    # Sign-consistency: reconstruct the per-model scores and verify that a
    # positive recovery corresponds to a score that is BETTER than noisy
    # (lower for RMSE, higher for f1/auc), per the metric direction.
    pm = res['per_model_per_strategy']
    checked = 0
    for m, strats in pm.items():
        m_clean = strats.get('no_noise', {}).get('score', float('nan'))
        m_noisy = strats.get('noisy_baseline', {}).get('score', float('nan'))
        m_rec = summ['per_model_recovery_pct'].get(m, {})
        if not (_finite(m_clean) and _finite(m_noisy)):
            continue
        for strat, rv in m_rec.items():
            s = strats.get(strat, {}).get('score', float('nan'))
            if not (_finite(rv) and _finite(s)):
                continue
            better_than_noisy = (s < m_noisy) if lower_is_better else (s > m_noisy)
            # Allow a tiny epsilon band around 0 recovery for ties.
            if abs(rv) < 0.5:
                continue
            assert (rv > 0) == better_than_noisy, (
                f"SIGN BUG: {m}/{strat} recovery={rv:.1f}% but score {s:.4f} "
                f"vs noisy {m_noisy:.4f} (lower_is_better={lower_is_better}) "
                f"disagrees on direction.")
            checked += 1
    print(f"  [n07] sign-consistency checks passed: {checked}")
    assert checked > 0, "n07: no recovery values available to sign-check."
    print("  [n07] OK")


CHECKS = {'n07': check_n07}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('which', nargs='?', default='all',
                    choices=['all', 'n07'])
    ap.add_argument('--clf', action='store_true')
    ap.add_argument('--subsample', type=int, default=200)
    args = ap.parse_args()

    names = ['n07'] if args.which == 'all' else [args.which]

    failures = []
    for name in names:
        print(f"\n=== smoke {name} ({'clf' if args.clf else 'reg'}) ===")
        try:
            CHECKS[name](clf=args.clf, subsample=args.subsample)
        except AssertionError as e:
            print(f"  [{name}] FAIL: {e}")
            failures.append(name)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  [{name}] ERROR: {type(e).__name__}: {e}")
            failures.append(name)

    print("\n" + "=" * 50)
    if failures:
        print(f"SMOKE FAILED: {failures}")
        sys.exit(1)
    print("ALL SMOKE GREEN")


if __name__ == '__main__':
    main()
