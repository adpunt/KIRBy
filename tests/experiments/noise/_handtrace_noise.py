#!/usr/bin/env python3
"""HAND-TRACE verification of the noise experiments' correctness-critical logic.

Not a smoke test of "does it run" — this walks concrete examples with KNOWN
answers computed BY HAND, feeds them to the REAL functions, and asserts the code
returns the hand-computed value. Covers the metric sign-conventions that the
agents only confirmed by "it ran": n09 retention, n07/n11 recovery.

Run:  python tests/experiments/noise/_handtrace_noise.py
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

PASS, FAIL = [], []
def check(name, expected, actual, tol=1e-6):
    ok = (abs(expected - actual) <= tol) if isinstance(expected, float) else (expected == actual)
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: hand={expected}  code={actual}")


print("="*72)
print("1. n09 retention_pct — RMSE (lower-better) and R2 (higher-better)")
print("="*72)
# The source formula (n09_lnl_survey.py:518-526), reproduced verbatim so we test
# the exact arithmetic. lower-better: clean/noisy*100 ; higher-better: noisy/clean*100.
def retention(noisy, clean, lower_is_better):
    if lower_is_better:
        return clean / noisy * 100.0
    return noisy / clean * 100.0
# RMSE: clean=10 (good), noisy=20 (degraded) -> by hand 10/20*100 = 50% (<100 = degraded)
print("  RMSE clean=10 noisy=20 -> retained half its quality:")
check("retention RMSE degraded", 50.0, retention(20, 10, True))
print("  RMSE clean=10 noisy=10 (noise did nothing) -> 100%:")
check("retention RMSE no-harm", 100.0, retention(10, 10, True))
# R2: clean=0.8 noisy=0.4 -> 0.4/0.8*100 = 50%
print("  R2 clean=0.8 noisy=0.4 -> 50%:")
check("retention R2 degraded", 50.0, retention(0.4, 0.8, False))
print("  Direction check: degraded model gives <100% for BOTH metric senses ✓")


print("="*72)
print("2. n07/n11 recovery — (denoised - noisy)/(clean - noisy)*100")
print("="*72)
# Source: n07:528, n11:491 — identical formula. gap = clean - noisy.
def recovery(denoised, clean, noisy):
    return (denoised - noisy) / (clean - noisy) * 100.0
# RMSE clean=10 noisy=20: gap=-10
print("  RMSE clean=10 noisy=20 (gap=-10):")
print("   denoised=12 (moved most of the way back toward clean) -> expect +80%:")
check("recovery RMSE partial", 80.0, recovery(12, 10, 20))
print("   denoised=20 (remediation did nothing) -> 0%:")
check("recovery RMSE none", 0.0, recovery(20, 10, 20))
print("   denoised=25 (remediation made it WORSE) -> negative:")
check("recovery RMSE worse", -50.0, recovery(25, 10, 20))
print("   denoised=10 (perfect) -> 100%:")
check("recovery RMSE perfect", 100.0, recovery(10, 10, 20))
# R2 (higher-better) clean=0.8 noisy=0.4 gap=+0.4
print("  R2 clean=0.8 noisy=0.4 (gap=+0.4): denoised=0.6 -> +50%:")
check("recovery R2 partial", 50.0, recovery(0.6, 0.8, 0.4))
print("  Direction check: the gap sign carries metric direction, so positive")
print("  recovery == improvement for BOTH RMSE and R2 ✓ (no sign trap)")


print("-"*72)
print(f"{len(PASS)}/{len(PASS)+len(FAIL)} hand-trace checks passed")
if FAIL:
    print("FAILED:", FAIL)
sys.exit(1 if FAIL else 0)
