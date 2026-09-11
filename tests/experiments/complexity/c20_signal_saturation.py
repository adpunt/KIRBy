#!/usr/bin/env python3
"""C20: Per-rep signal saturation -- where each base rep's useful signal runs out.

Paper: Complexity
Core question (user's words): "where does each base representation's USEFUL
    signal run out?" For each rep ALONE: rank its columns by importance
    (lightgbm_gain), then score the rep at increasing feature counts under
    full-size grouped CV, and find where the metric plateaus. This measures a
    property of each base rep BY ITSELF. Hybrids are NOT involved at any point.

NO hybrids, ever. This file NEVER calls find_best_hybrid() and NEVER edits
find_best_hybrid.py / hybrid.py. It ranks a SINGLE rep's own columns by
lightgbm_gain, truncates to top-k, and scores that single rep alone.

Fold discipline (load-bearing):
  - Splitting is the repo's own cv_split (GroupKFold, deterministic) via the
    shared harness make_folds. NO random splits.
  - Column importance is ranked on the TRAINING rows of each fold only, then
    the same top-k columns are read on the held-out rows. Ranking never sees
    validation rows (no selection leakage).

Metric direction is load-bearing and verified end-to-end by the harness'
BLOCKING assert_metric_direction() gate before any curve is computed:
  herg_ki / logd / buchwald_hartwig = RMSE  (lower_is_better=True)
  flab_thermo_tresanco               = Spearman (lower_is_better=False)

Downstream models: lightgbm and rf (both via the repo's get_model).

Reuses the shared round-2 harness (load_full / make_folds / rank_columns /
score primitives / metric-direction gate). The harness performs the macOS
segfault guard (`import lightgbm` FIRST) and sets KIRBY_CACHE_DIR to the staged
read-only cache; importing it first here preserves that guard.

Usage:
    python c20_signal_saturation.py                       # all 4 staged datasets
    python c20_signal_saturation.py --dataset herg_ki
    python c20_signal_saturation.py --dataset herg_ki --reps ecfp4,maccs --models lightgbm
    python c20_signal_saturation.py --smoke               # ~8 reps/dataset (disclosed)
"""
import sys
import os
import argparse
import json
import time
from pathlib import Path

# Portable path bootstrap: locate the repo root from this file (no hardcoded
# absolute path) and put tests/ + src/ on sys.path so the imports resolve
# whether run locally or on the server.
_REPO = str(Path(__file__).resolve().parents[3])
for _p in (os.path.join(_REPO, "tests"), os.path.join(_REPO, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Shared allocation harness — a committed repo module (imports lightgbm FIRST
# for the macOS segfault guard). PORT 2026-09-10: was a /private/tmp scratchpad
# copy of round2_harness, deleted; same call surface.
from experiments import _alloc_harness as H  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


# Feature-count ladder (user-specified). 'full' is appended per-rep as its
# native width, so any k >= width is redundant with 'full' and skipped.
DEFAULT_KS = [10, 25, 50, 100, 200, 300, 500, 1000]
DEFAULT_MODELS = ["lightgbm", "rf"]

# Disclosed smoke rep pool: a spread of rep families per dataset, chosen to
# cover a fingerprint/descriptor/sequence mix WITHOUT subsampling rows. Only
# reps that are actually staged are used; missing names are silently dropped
# and the ACTUAL pool used is recorded in the results.
SMOKE_REPS = {
    "herg_ki": ["ecfp4", "rdkfp", "maccs", "avalon",
                "atom_pair", "top_torsion", "mordred", "mol2vec"],
    "logd": ["ecfp4", "rdkfp", "maccs", "avalon",
             "atom_pair", "top_torsion", "mordred", "mol2vec"],
    "flab_thermo_tresanco": ["aac", "onehot", "blosum", "esm2",
                             "antiberty", "prot_t5", "structural",
                             "mutation_features"],
    "buchwald_hartwig": ["drfp_2048", "rxnfp", "product_ecfp4",
                         "reactant_ecfp4", "diff_ecfp4", "product_mordred",
                         "cgr_morgan", "ligand_descriptors"],
}

# Plateau tolerance (see _find_plateau docstring). Absolute in metric units,
# with a small relative floor so large-magnitude scores aren't over-strict.
PLATEAU_ABS_TOL = 0.01
PLATEAU_REL_TOL = 0.02


def _find_plateau(ks, scores, full_score, lower_is_better):
    """Smallest feature-count k whose CV score reaches within `tol` of the
    full-rep CV score, in the improving direction.

    Definition (stated so a reader can re-derive it from the raw curve):
        tol = max(PLATEAU_ABS_TOL, PLATEAU_REL_TOL * |full_score|)
        lower_is_better  (RMSE):     plateau = min k with score(k) <= full + tol
        higher_is_better (Spearman): plateau = min k with score(k) >= full - tol
    i.e. plateau = the first k that is no meaningfully-worse than using ALL the
    rep's columns. Returns (plateau_k, tol). plateau_k is None if no k in the
    ladder qualifies (signal keeps rising to full width -> "no plateau below
    full"). `full` itself is always a valid fallback but is reported as None
    here to make "needs full width" explicit.
    """
    if full_score is None or np.isnan(full_score):
        return None, float("nan")
    tol = max(PLATEAU_ABS_TOL, PLATEAU_REL_TOL * abs(full_score))
    for k, s in zip(ks, scores):
        if s is None or np.isnan(s):
            continue
        if lower_is_better and s <= full_score + tol:
            return int(k), tol
        if (not lower_is_better) and s >= full_score - tol:
            return int(k), tol
    return None, tol


def saturation_curve(X, y, folds, model_name, metric_fn, lower_is_better,
                     task_type, ks):
    """Fold-safe importance-truncated learning curve for ONE rep alone.

    Per fold: rank the rep's OWN columns by lightgbm_gain on TRAIN rows only,
    then for each k in `ks` (plus 'full') fit `model_name` on the top-k train
    columns and score the held-out rows. Averages per-fold metric values.

    Returns dict:
        {"ks": [...ints..., width],          # width is the 'full' point
         "width": int,
         "mean": [...per-k mean...],
         "per_fold": [[...per-fold at k0...], ...],
         "full_mean": float, "full_per_fold": [...]}

    NO hybrid, NO cross-rep concatenation: X is a single rep matrix.
    """
    X = H._clean(X)
    y = np.asarray(y, dtype=float)
    width = X.shape[1]

    # Ladder = requested ks that fit strictly under the rep width, then the
    # native width as the explicit 'full' point (always last, always present).
    ladder = [k for k in ks if k < width]
    ladder.append(width)  # 'full'

    per_fold = [[] for _ in ladder]
    for tr, va in folds:
        Xtr_all, Xva_all = X[tr], X[va]
        ytr = y[tr]
        # Rank this rep's own columns on TRAIN ONLY (no val leakage).
        imp = H.rank_columns(Xtr_all, ytr, method="lightgbm_gain")
        order = np.argsort(imp)[::-1]  # descending importance
        for j, k in enumerate(ladder):
            cols = order[:k]
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(Xtr_all[:, cols])
            Xva = scaler.transform(Xva_all[:, cols])
            model = H.get_model(model_name, task_type=task_type,
                                n_samples=len(tr))
            if model is None:
                per_fold[j].append(float("nan"))
                continue
            model.fit(Xtr, ytr)
            pred = model.predict(Xva)
            per_fold[j].append(float(metric_fn(y[va], pred)))

    mean = [float(np.nanmean(f)) if len(f) else float("nan") for f in per_fold]
    return {
        "ks": [int(k) for k in ladder],
        "width": int(width),
        "mean": mean,
        "per_fold": per_fold,
        "full_mean": mean[-1],
        "full_per_fold": per_fold[-1],
    }


def run_dataset(name, rep_filter=None, models=DEFAULT_MODELS, ks=DEFAULT_KS,
                n_folds=5, verbose=True):
    """Compute per-rep saturation curves for one staged dataset at FULL size."""
    reps, labels, groups, mname, mfn, lower, task = H.load_full(name)
    folds = H.make_folds(groups, labels, task, n=n_folds)

    all_names = sorted(reps.keys())
    if rep_filter:
        used = [r for r in rep_filter if r in reps]
        dropped = [r for r in rep_filter if r not in reps]
    else:
        used = all_names
        dropped = []

    if verbose:
        print(f"\n[c20] {name}: metric={mname} (lower_is_better={lower}), "
              f"n={len(labels)}, using {len(used)}/{len(all_names)} reps, "
              f"models={models}")
        if dropped:
            print(f"[c20] {name}: requested-but-not-staged reps dropped: {dropped}")

    ds_result = {
        "dataset": name, "n": int(len(labels)),
        "metric": mname, "lower_is_better": bool(lower),
        "n_folds": n_folds, "ks_requested": list(ks),
        "reps_available": all_names, "reps_used": used,
        "reps_requested_missing": dropped,
        "models": {},
        "plateau_definition": (
            "smallest k with score(k) within tol of the full-rep score in the "
            "improving direction; tol = max(%.3f, %.3f*|full_score|); "
            "plateau=None means signal needs full native width"
            % (PLATEAU_ABS_TOL, PLATEAU_REL_TOL)),
    }

    for model_name in models:
        ds_result["models"][model_name] = {}
        for rep_name in used:
            t0 = time.time()
            curve = saturation_curve(reps[rep_name], labels, folds, model_name,
                                     mfn, lower, task, ks)
            plateau_k, tol = _find_plateau(
                curve["ks"][:-1], curve["mean"][:-1], curve["full_mean"], lower)
            curve["plateau_k"] = plateau_k
            curve["plateau_tol"] = tol
            curve["full_score"] = curve["full_mean"]
            curve["seconds"] = round(time.time() - t0, 1)
            ds_result["models"][model_name][rep_name] = curve
            if verbose:
                pk = plateau_k if plateau_k is not None else "full"
                print(f"    [{model_name}] {rep_name:<12} width={curve['width']:>5} "
                      f"full={curve['full_mean']:.4f} plateau@{pk} "
                      f"({curve['seconds']}s) curve="
                      f"{[round(x,3) for x in curve['mean']]}")

    return ds_result


def _plateau_table(results):
    """Compact rep x (dataset,model) -> (plateau_k, full_score) text table."""
    lines = ["", "=" * 70, "PLATEAU TABLE (rep -> saturation feature-count @ full-rep score)",
             "=" * 70]
    for name, ds in results.items():
        if "error" in ds:
            lines.append(f"\n{name}: ERROR - {ds['error']}")
            continue
        lines.append(f"\n{name}  (metric={ds['metric']}, "
                     f"lower_is_better={ds['lower_is_better']}, n={ds['n']})")
        for model_name, reps in ds["models"].items():
            lines.append(f"  model={model_name}")
            for rep_name, c in reps.items():
                pk = c["plateau_k"] if c["plateau_k"] is not None else "FULL"
                lines.append(f"    {rep_name:<14} plateau@{str(pk):<6} "
                             f"full_score={c['full_score']:.4f} "
                             f"(width={c['width']})")
    return "\n".join(lines)


def run(datasets, rep_map=None, models=DEFAULT_MODELS, ks=DEFAULT_KS,
        n_folds=5, gate=True):
    """Top-level: optional metric-direction gate, then per-dataset curves."""
    if gate:
        print("[c20] Running BLOCKING metric-direction gate...")
        H.assert_metric_direction(verbose=False)

    results = {}
    for name in datasets:
        rep_filter = (rep_map or {}).get(name)
        try:
            results[name] = run_dataset(name, rep_filter=rep_filter,
                                        models=models, ks=ks, n_folds=n_folds)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[name] = {"dataset": name, "error": f"{type(e).__name__}: {e}"}
    print(_plateau_table(results))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="C20: per-rep signal saturation")
    ap.add_argument("--dataset", default=None,
                    help="single staged dataset (default: all 4 staged)")
    ap.add_argument("--reps", default=None,
                    help="comma-separated rep names (default: all staged reps)")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--ks", default=",".join(str(k) for k in DEFAULT_KS))
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--smoke", action="store_true",
                    help="use the disclosed ~8-rep/dataset smoke pool")
    ap.add_argument("--no-gate", action="store_true",
                    help="skip the metric-direction gate (NOT for real runs)")
    ap.add_argument("--out", default=None, help="output dir for results JSON")
    args = ap.parse_args()

    if not args.dataset:
        ap.error("--dataset is required; the shared harness carries no "
                 "dataset roster of its own.")
    ds_list = [args.dataset]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    ks = [int(k) for k in args.ks.split(",") if k.strip()]

    rep_map = None
    if args.smoke:
        rep_map = SMOKE_REPS
    elif args.reps:
        rep_map = {d: [r.strip() for r in args.reps.split(",")] for d in ds_list}

    t0 = time.time()
    results = run(ds_list, rep_map=rep_map, models=models, ks=ks,
                  n_folds=args.n_folds, gate=not args.no_gate)
    results["_elapsed_s"] = round(time.time() - t0, 1)

    if args.out:
        outdir = Path(args.out)
        outdir.mkdir(parents=True, exist_ok=True)
        fp = outdir / "c20_signal_saturation.json"
        fp.write_text(json.dumps(results, indent=2, default=str))
        print(f"\n[c20] wrote {fp}")

    print("\n===== MACHINE-READABLE REPORT =====")
    print(json.dumps(results, indent=2, default=str))
