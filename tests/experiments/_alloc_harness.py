"""round2_harness — shared primitives for the round-2 selection experiments
(c30 balanced-global, c31 performance-weighted-global).

RECONSTRUCTED 2026-08-11 (the original scratchpad copy was deleted).  Rebuilt
from the exact call surface c30/c31 use plus the confirmed kirby APIs:
  load_data / cv_split / compute_feature_importance  (tests/experiments/_shared.py)
  register_datasets via dataset_policy               (kirby.registry)
  get_model / BUILTIN_METRICS                        (kirby.find_best_hybrid)

HARD RULES honoured: imports lightgbm FIRST (macOS segfault guard — lightgbm
must load before torch/kirby pull OpenMP); full rows always (row counts
asserted in load_full); repo group-aware cv_split (GroupKFold / StratifiedGroup
KFold, deterministic seed=42, NO random fallback); lightgbm_gain importance
FIXED; never edits src/ (import-only).

Public surface used by the experiments:
  load_full(ds)                     -> (reps, labels, groups, mname, mfn, lower, task)
  make_folds(groups, labels, task, n) -> [(train_idx, val_idx), ...]
  rank_columns(X, y, method)        -> per-column importance vector (col-aligned)
  score_matrix(X, y, folds, model, metric_fn, lower, task_type) -> (mean, per_fold, oof)
  get_model                         (re-exported from kirby.find_best_hybrid)
  assert_metric_direction(verbose)  -> dict, RAISES on any direction/monotone failure
  _clean(X)                         -> nan/inf-scrubbed copy
"""
# ---------------------------------------------------------------------------
# lightgbm FIRST (segfault guard) — must precede any torch / kirby import.
# ---------------------------------------------------------------------------
import lightgbm as _lgb  # noqa: F401

import os
import sys
from pathlib import Path

# Cache location: honour an already-set KIRBY_CACHE_DIR (the server's setup.sh
# sets it). Only fall back to the local seagate stage if it exists and nothing
# is set — so this file is safe to commit and run on the server unchanged.
_SEAGATE = "/Volumes/seagate/kirby_cache"
if not os.environ.get("KIRBY_CACHE_DIR") and os.path.isdir(_SEAGATE):
    os.environ["KIRBY_CACHE_DIR"] = _SEAGATE

# Portable path bootstrap: prefer an already-importable layout (server runs
# experiments as modules with tests/ on PYTHONPATH). Fall back to locating the
# repo root from this file's own position — no hardcoded absolute path.
_HERE = Path(__file__).resolve()
_REPO = str(_HERE.parents[2])            # <repo>/tests/experiments/_alloc_harness.py
for _p in (os.path.join(_REPO, "src"),
           os.path.join(_REPO, "tests"),
           os.path.join(_REPO, "tests", "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from kirby.scaling import FeatureScaler  # 0/1 features unscaled (2026-10-05)

# Populate DATASET_REGISTRY (metric + direction source of truth) as a side effect.
import dataset_policy  # noqa: E402,F401
from kirby.registry import require  # noqa: E402
from kirby.find_best_hybrid import get_model, BUILTIN_METRICS  # noqa: E402
from experiments._shared import (  # noqa: E402
    load_data, cv_split, compute_feature_importance,
)

# datasets in scope + their EXPECTED metric direction (the blocking gate check).
EXPECTED_DIRECTION = {
    "herg_ki": ("rmse", True),           # RMSE, lower is better
    "logd": ("rmse", True),
    "buchwald_hartwig": ("rmse", True),
    "flab_thermo_tresanco": ("spearman", False),  # Spearman, higher is better
}


def _clean(X):
    """Scrub NaN/inf so downstream fits never choke; preserves shape/order."""
    X = np.asarray(X, dtype=float)
    X = np.clip(X, -1e10, 1e10)
    return np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)


# ---------------------------------------------------------------------------
# DATA LOADING — full dataset, all rows, staged 'general' rep pool.
# ---------------------------------------------------------------------------
def load_full(ds):
    """Load a dataset at FULL size with its staged representation matrices.

    Returns (reps, labels, groups, metric_name, metric_fn, lower_is_better,
    task_type).  Row counts are asserted equal across every rep, labels and
    groups — a full-size guarantee, no silent subsampling.
    """
    info = require(ds)                       # authoritative metric + direction
    metric_fn, lower = BUILTIN_METRICS[info.metric]
    # rep_names=None -> auto-filter to DATASET_REPS[ds]['general']; preset='all'
    # so any general rep not already cached would be materialised in full.
    data = load_data(ds, rep_preset="all")
    reps = {k: _clean(v) for k, v in data["reps"].items()}
    labels = np.asarray(data["labels"], dtype=float)
    groups = np.asarray(data["groups"])

    n = len(labels)
    assert n == len(groups), f"{ds}: labels {n} != groups {len(groups)}"
    for rn, X in reps.items():
        assert X.shape[0] == n, f"{ds}: rep {rn} has {X.shape[0]} rows, expected {n}"
    # registry direction must match the experiment's stated expectation.
    if ds in EXPECTED_DIRECTION:
        exp_metric, exp_lower = EXPECTED_DIRECTION[ds]
        assert info.metric == exp_metric and bool(lower) == exp_lower, (
            f"{ds}: registry says ({info.metric},{lower}) but experiment "
            f"expects ({exp_metric},{exp_lower})")
    assert info.task_type in ("regression", "classification")
    print(f"  [load_full] {ds}: n={n} reps={len(reps)} metric={info.metric} "
          f"lower={bool(lower)} task={info.task_type}", flush=True)
    return reps, labels, groups, info.metric, metric_fn, bool(lower), info.task_type


# ---------------------------------------------------------------------------
# CV FOLDS — repo group-aware splitter, deterministic, NO random fallback.
# ---------------------------------------------------------------------------
def make_folds(groups, labels, task, n=5):
    """Materialise n deterministic group-aware folds as [(train_idx, val_idx)]."""
    X = np.zeros((len(labels), 1))           # only length matters to the splitter
    folds = [(np.asarray(tr), np.asarray(va))
             for tr, va in cv_split(groups, n, X, labels, seed=42, task_type=task)]
    assert len(folds) == n, f"expected {n} folds, got {len(folds)}"
    # disjoint val coverage sanity
    cov = np.concatenate([va for _, va in folds])
    assert len(np.unique(cov)) == len(cov), "validation folds overlap"
    return folds


# ---------------------------------------------------------------------------
# IMPORTANCE — single-matrix wrapper over compute_feature_importance.
# Returns a per-column, column-aligned importance vector (higher = better).
# ---------------------------------------------------------------------------
def rank_columns(X, y, method="lightgbm_gain"):
    imp = compute_feature_importance({"_": _clean(X)}, np.asarray(y, dtype=float),
                                     method=method)["_"]
    return np.asarray(imp, dtype=float)


# ---------------------------------------------------------------------------
# SCORING — fit `model` per fold on scaled columns, return (mean, per_fold, oof).
# ---------------------------------------------------------------------------
def score_matrix(X, y, folds, model_name, metric_fn, lower, task_type="regression"):
    y = np.asarray(y, dtype=float)
    X = _clean(X)
    oof = np.full(len(y), np.nan)
    per_fold = []
    for tr, va in folds:
        sc = FeatureScaler()
        Xtr = sc.fit_transform(X[tr])
        Xva = sc.transform(X[va])
        m = get_model(model_name, task_type=task_type, n_samples=len(tr))
        if m is None:
            raise RuntimeError(f"get_model returned None for {model_name} n={len(tr)}")
        m.fit(Xtr, y[tr])
        p = m.predict(Xva)
        oof[va] = p
        per_fold.append(float(metric_fn(y[va], p)))
    return float(np.mean(per_fold)), per_fold, oof


# ---------------------------------------------------------------------------
# BLOCKING METRIC-DIRECTION GATE.
# Proves, end-to-end and in BOTH directions, that (a) the registry direction
# for every in-scope dataset equals the experiment's expectation, and (b) the
# metric function actually moves the right way when predictions get better.
# RAISES on any mismatch.
# ---------------------------------------------------------------------------
def assert_metric_direction(verbose=False, datasets=None):
    """Blocking metric-direction gate. Loads NO rep matrices — builds synthetic
    good/bad predictors from a label ladder, so it runs for ANY dataset via the
    registry. `datasets=None` gates the local anchor set (EXPECTED_DIRECTION);
    pass an explicit list (e.g. the dataset a server task is running) otherwise.
    Anchors in EXPECTED_DIRECTION additionally cross-check the registry metric."""
    rng = np.random.default_rng(0)
    y = np.linspace(0.0, 10.0, 200)
    good = y + rng.normal(0, 0.05, size=y.shape)    # near-perfect predictions
    bad = y + rng.normal(0, 5.0, size=y.shape)      # noisy predictions
    out = {}
    check = list(EXPECTED_DIRECTION) if datasets is None else list(datasets)
    for ds in check:
        info = require(ds)
        fn, lower = BUILTIN_METRICS[info.metric]
        # (a) registry vs experiment expectation — only for known anchors
        if ds in EXPECTED_DIRECTION:
            exp_metric, exp_lower = EXPECTED_DIRECTION[ds]
            if not (info.metric == exp_metric and bool(lower) == exp_lower):
                raise AssertionError(
                    f"[gate] {ds}: registry ({info.metric},{lower}) != "
                    f"expected ({exp_metric},{exp_lower})")
        # (b) end-to-end monotonicity in the claimed direction
        s_good, s_bad = fn(y, good), fn(y, bad)
        if lower:                       # lower is better -> good must score LOWER
            monotone = s_good < s_bad
        else:                           # higher is better -> good must score HIGHER
            monotone = s_good > s_bad
        if not monotone:
            raise AssertionError(
                f"[gate] {ds}: metric {info.metric} not monotone in claimed "
                f"direction (good={s_good:.4f} bad={s_bad:.4f} lower={lower})")
        rec = {"direction": "lower" if lower else "higher",
               "monotone": bool(monotone), "metric": info.metric}
        out[ds] = rec
        if verbose:
            print(f"  [gate] {ds:22s} metric={info.metric:9s} "
                  f"dir={'lower' if lower else 'higher':6s} "
                  f"good={s_good:.4f} bad={s_bad:.4f}  PASS", flush=True)
    return out


if __name__ == "__main__":
    print("round2_harness self-test — metric-direction gate:")
    assert_metric_direction(verbose=True)
    print("gate PASSED.")
