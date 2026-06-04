#!/usr/bin/env python3
"""Non-degeneracy smoke tests for noise experiments n04, n05, n07.

Built during the 2026-06-03 noise audit (n04/n05/n07 group). These go beyond
"does run() crash" — they assert the output is NOT degenerate in the specific
ways the audit found:

  * n04: _identify_cliffs_euclidean must find > 0 cliffs (the absolute
    sim_threshold=0.55 bug made n_cliffs always 0 on standardised high-D reps,
    which silently turned the cliff-false-positive win-rate into a meaningless
    0<=0 tie reporting 1.0). It must ALSO not report a win-rate when cliffs are
    degenerate.
  * n05: at the highest sigma at least one detector must reach F1 > 0.2 in the
    concat space (all-zero F1 everywhere would mean detectors flag the wrong
    samples or the noise_mask ground truth is broken).
  * n07: recovery percentages must be finite for at least one strategy, and the
    recovery sign convention must be self-consistent with the metric direction
    (a strategy that moves the score toward `clean` yields positive recovery).

Reuses the validated _smoke_harness machinery (build_data / tiny_config /
precompute) for cheap local reps. Run directly:

    python tests/experiments/noise/_smoke_n04_n07.py            # all (reg)
    python tests/experiments/noise/_smoke_n04_n07.py n04        # one
    python tests/experiments/noise/_smoke_n04_n07.py n05 --clf  # clf branch

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


def check_n04(clf=False, subsample=200):
    """n04 must find cliffs and must not report a bogus win-rate."""
    assert not clf, "n04 is regression-only; do not run --clf"
    data, ds, config = _prep(clf, subsample)
    from experiments.noise import n04_gp_loo_detection as n04

    # Direct unit check on the cliff identifier (the bug surface).
    reps = data['reps']
    X = np.hstack([reps[k] for k in sorted(reps.keys())])
    y = np.asarray(data['labels'])
    cm = n04._identify_cliffs_euclidean(X, y)
    n_cliffs = int(cm.sum())
    print(f"  [n04] cliffs identified: {n_cliffs}/{len(y)}")
    assert n_cliffs > 0, (
        "DEGENERATE: _identify_cliffs_euclidean found 0 cliffs — the absolute "
        "sim_threshold bug has regressed (sim=1/(1+D) never reaches it on "
        "standardised high-D reps).")

    res = n04.run(data, config, OUT, ds)
    assert not res.get('skipped'), f"n04 unexpectedly skipped: {res}"
    assert res.get('n_cliffs', 0) > 0, f"n04 run reports 0 cliffs: {res.get('n_cliffs')}"
    summ = res.get('summary', {})
    # cliff_FP rates must not be identically zero across all conditions/levels.
    saw_nonzero_cfp = False
    for lvl in res['per_noise_level'].values():
        for cond in lvl.values():
            if isinstance(cond, dict) and cond.get('cliff_fp_rate', 0) > 0:
                saw_nonzero_cfp = True
    assert saw_nonzero_cfp, (
        "DEGENERATE: every cliff_fp_rate is 0 across all conditions — "
        "win-rate comparison is meaningless.")
    # Win-rate must be a real fraction now (not the false 1.0 from 0<=0 ties).
    wr = summ.get('win_rate')
    print(f"  [n04] win_rate={wr} cliffs_degenerate={res.get('cliffs_degenerate')}")
    assert _finite(wr), f"n04 win_rate not finite with cliffs present: {summ}"
    print("  [n04] OK")


def check_n05(clf=False, subsample=200):
    """n05 must have at least one detector with F1>0.2 at the highest sigma."""
    data, ds, config = _prep(clf, subsample)
    from experiments.noise import n05_noise_injection_benchmark as n05
    res = n05.run(data, config, OUT, ds)
    sigmas = sorted(res['per_sigma'].keys(), key=float)
    hi = sigmas[-1]
    methods = res['methods']
    best = -1.0
    best_m = None
    for m in methods:
        f1 = res['per_sigma'][hi].get(m, {}).get('concat', {}).get('f1', -1)
        if f1 is not None and f1 > best:
            best, best_m = f1, m
    print(f"  [n05] best concat F1 at sigma={hi}: {best_m}={best:.3f}")
    assert best > 0.2, (
        f"DEGENERATE: no detector exceeds F1=0.2 at the hardest noise level "
        f"(best={best_m}={best:.3f}) — detectors or noise_mask are broken.")
    print("  [n05] OK")


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


CHECKS = {'n04': check_n04, 'n05': check_n05, 'n07': check_n07}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('which', nargs='?', default='all',
                    choices=['all', 'n04', 'n05', 'n07'])
    ap.add_argument('--clf', action='store_true')
    ap.add_argument('--subsample', type=int, default=200)
    args = ap.parse_args()

    names = ['n04', 'n05', 'n07'] if args.which == 'all' else [args.which]
    # n04 is regression-only; drop it from a --clf run.
    if args.clf and 'n04' in names:
        print("  [skip] n04 is regression-only — not run under --clf")
        names = [n for n in names if n != 'n04']

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
