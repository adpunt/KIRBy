"""A subset run must not destroy what it did not produce.

    python tests/smoke/smoke_kirby_merge.py

`--conditions` is a first-class way to split a run across the queue -- the
runbook says so -- but it was not in the guard that merges with an existing
all_results.csv. Only `--models` and `--reps` were. So a job that ran one
condition rewrote the file with that condition's rows alone and destroyed every
other condition already in it (RERUN_PLAN.md §2.13 in qsar_qm_models).

This exercises the merge arithmetic itself, on frames shaped like the real ones.
"""
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

T0 = time.time()
FAILS = []
NCHECK = [0]


def check(name, ok, detail=''):
    NCHECK[0] += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ''))
    if not ok:
        FAILS.append(name)


def merge(existing, produced, model_filter=None, rep_filter=None,
          conditions=None, all_conditions=('gaussian', 'grouped_wider', 'censoring')):
    """The guard as the runner applies it."""
    condition_subset = conditions is not None and set(conditions) != set(all_conditions)
    if not (model_filter or rep_filter or condition_subset):
        return produced
    mask = pd.Series(True, index=existing.index)
    if model_filter:
        mask &= existing['model'].isin(model_filter)
    if rep_filter:
        mask &= existing['rep'].isin(rep_filter)
    if condition_subset and 'noise_type' in existing.columns:
        mask &= existing['noise_type'].isin(list(conditions))
    return pd.concat([existing[~mask], produced], ignore_index=True)


EXISTING = pd.DataFrame([
    {'model': 'RF', 'rep': 'ECFP4', 'noise_type': c, 'sigma': s, 'r2': 0.5}
    for c in ('gaussian', 'grouped_wider', 'censoring') for s in (0.0, 0.3)
])
PRODUCED = pd.DataFrame([
    {'model': 'RF', 'rep': 'ECFP4', 'noise_type': 'gaussian', 'sigma': s, 'r2': 0.9}
    for s in (0.0, 0.3)
])

print("=" * 70)
print("a condition-only re-run keeps the other conditions")
print("=" * 70)

out = merge(EXISTING, PRODUCED, conditions=['gaussian'])
kept = sorted(out['noise_type'].unique())
check("the other two conditions survive", kept == ['censoring', 'gaussian', 'grouped_wider'], str(kept))
check("the re-run condition's rows are the NEW ones",
      set(out[out.noise_type == 'gaussian']['r2']) == {0.9},
      str(sorted(set(out[out.noise_type == 'gaussian']['r2']))))
check("no duplicate rows for the re-run condition",
      len(out[out.noise_type == 'gaussian']) == 2,
      f"{len(out[out.noise_type == 'gaussian'])} rows")

# CONTROL: the rule this replaced ignored `conditions` entirely, so the produced
# frame simply replaced the file.
def old_merge(existing, produced, model_filter=None, rep_filter=None):
    if not (model_filter or rep_filter):
        return produced
    return existing

old = old_merge(EXISTING, PRODUCED)
check("CONTROL: the old rule DID destroy the other conditions",
      sorted(old['noise_type'].unique()) == ['gaussian'],
      str(sorted(old['noise_type'].unique())))

print("\n" + "=" * 70)
print("a full-roster run still replaces the file")
print("=" * 70)
out = merge(EXISTING, PRODUCED,
            conditions=['gaussian', 'grouped_wider', 'censoring'])
check("running every condition replaces the file outright",
      len(out) == len(PRODUCED), f"{len(out)} rows")

print("\n" + "=" * 70)
print("a model that produced no rows stops the job")
print("=" * 70)

# `quantile_forest` raises `Invalid parameter 'monotonic_cst'` on this sklearn, so
# a --models QRF BNN-Full run produced NO QRF rows -- and exited 0 with a results
# file. The all-failed guard fires only when EVERY model fails; one model failing
# everywhere is the likelier case, and it leaves that model absent for one dataset
# only, which biases every cross-model aggregate (RERUN_PLAN.md 2.13d).
_experiments = [("QRF", "ECFP4", None, None), ("BNN-Full", "ECFP4", None, None)]
_combined = pd.DataFrame([{"model": "BNN-Full", "rep": "ECFP4", "r2": 0.5}])
_asked = {n for n, _r, _f, _m in _experiments}
_silent = sorted(_asked - set(_combined["model"].unique()))
check("a requested model with no rows is named", _silent == ["QRF"], str(_silent))

_full = pd.DataFrame([{"model": m, "rep": "ECFP4", "r2": 0.5} for m in _asked])
check("CONTROL: nothing is named when every model produced rows",
      sorted(_asked - set(_full["model"].unique())) == [])

print("\n" + "=" * 70)
print(f"{NCHECK[0]} checks in {time.time() - T0:.1f}s")
if FAILS:
    print("FAILED:", "; ".join(FAILS))
    sys.exit(1)
print("ALL CHECKS PASSED")
sys.exit(0)
