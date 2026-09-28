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
  - Features are ranked on the TRAINING rows of each fold only, then the same
    top-k features are read on the held-out rows. Ranking never sees validation
    rows (no selection leakage).
  - The ranking reads only the training rows and the labels, never the
    downstream model, so it is computed ONCE per representation per fold and
    both downstream models reuse it: 50 rankings per dataset, not 100.

"Scores as well as all of them" is set from the folds (user decision
2026-09-11): the tolerance is one standard deviation of the full-width
fold scores, averaged over the dataset's representations, per downstream model.
The value used is written into the results as plateau_tol.

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
    python c20_signal_saturation.py --smoke               # first 8 staged reps (disclosed)
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
from experiments.complexity.complexity_helpers import DATASET_REPS  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


# Feature-count ladder (user-specified). 'full' is appended per-rep as its
# native width, so any k >= width is redundant with 'full' and skipped.
DEFAULT_KS = [10, 25, 50, 100, 200, 300, 500, 1000]
DEFAULT_MODELS = ["lightgbm", "rf"]

# Smoke representation pool.
#
# 2026-09-11, user decision: use the dataset's staged representations, never a
# list typed into this file. The pool used to be eight names per dataset typed
# here before the 2026-08-20 pruning, and they had drifted badly: 4 of 8 were
# staged on herg_ki and on logd, 2 of 8 on buchwald_hartwig, and 1 of 8 on
# flab_thermo_tresanco. Missing names are dropped silently by load_data, so a
# smoke run on flab_thermo_tresanco was scoring one representation.
#
# The smoke pool is now the first SMOKE_N_REPS names of the dataset's staged
# list, in that list's own order. Every name is therefore present, and the pool
# actually used is recorded in the results.
SMOKE_N_REPS = 8


def smoke_reps(ds):
    """The smoke pool for one dataset, read from the one place the staged list
    lives.

    complexity_helpers.DATASET_REPS is what c30, c31, c32 and c43 use, and what
    _shared.load_data auto-filters to when a caller passes no representation
    names. Raises rather than substituting when a dataset has no staged list.
    """
    entry = DATASET_REPS.get(ds) or {}
    names = list(entry.get("general") or [])
    if not names:
        raise RuntimeError(
            f"{ds} has no 'general' representation list in "
            f"complexity_helpers.DATASET_REPS, so the smoke pool cannot be "
            f"resolved. Nothing is guessed here.")
    return names[:SMOKE_N_REPS]


# How big a difference still counts as "scores as well as all of them".
#
# 2026-09-11, user decision: set it from the folds, not from a fixed number.
# It was max(0.01 in metric units, 2% of the full-width score), which meant the
# same absolute size on an RMSE dataset and on a Spearman one. It is now one
# standard deviation of the full-width fold scores (one per fold), averaged over the
# dataset's representations, computed separately under each downstream model.
# So "as well as" means "inside this dataset's own fold-to-fold wobble, in its
# own metric". The value actually used is written into the results.
PLATEAU_TOL_DDOF = 1   # the fold scores treated as a sample, not a population


def fold_spread(per_fold_scores):
    """Standard deviation of one representation's full-width fold scores.

    Takes the held-out scores at full width, one per fold, in the dataset's own metric.
    Returns nan when fewer than two folds produced a number.
    """
    vals = np.asarray([v for v in per_fold_scores if v is not None],
                      dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        return float("nan")
    return float(np.std(vals, ddof=PLATEAU_TOL_DDOF))


def plateau_tolerance(curves):
    """One tolerance for a whole dataset under one downstream model.

    `curves` is every representation's saturation curve under that model. Each
    representation contributes the standard deviation of its own
    full-width fold scores; the tolerance is the mean of those, in the
    dataset's own metric units. Returns nan when no representation gave a
    usable spread.
    """
    spreads = [fold_spread(c.get("full_per_fold", [])) for c in curves]
    spreads = [s for s in spreads if np.isfinite(s)]
    if not spreads:
        return float("nan")
    return float(np.mean(spreads))


def _find_plateau(ks, scores, full_score, lower_is_better, tol):
    """Smallest feature count whose mean over all folds reaches within `tol`
    of the full-width mean over all folds, in the improving direction.

    `tol` is in the metric's own units and comes from plateau_tolerance().
        lower_is_better  (RMSE):     first k with score(k) <= full + tol
        higher_is_better (Spearman): first k with score(k) >= full - tol
    i.e. the first feature count that is no meaningfully-worse than using ALL
    of the representation's features. Returns that feature count, or None when
    no count in the ladder qualifies (the signal keeps rising to full width, so
    "needs full width"). None also comes back when the full-width score or the
    tolerance is not a number.
    """
    if full_score is None or not np.isfinite(full_score):
        return None
    if not np.isfinite(tol):
        return None
    for k, s in zip(ks, scores):
        if s is None or not np.isfinite(s):
            continue
        if lower_is_better and s <= full_score + tol:
            return int(k)
        if (not lower_is_better) and s >= full_score - tol:
            return int(k)
    return None


def plateau_status(ks, scores, full_score, tol, plateau_k):
    """Why plateau_n_features has the value it has.

    _find_plateau returns None for five different reasons, and the table used
    to print every one of them as "FULL". Only one means the representation
    needs more features than the ladder tested.
        reached              a tested count scored within tol of full width
        above_largest_tested no tested count did; the signal needs more than
                             max(ks) features (not necessarily all of them)
        no_count_below_width the width is at or under the smallest requested
                             count, so nothing below full width was tested
        ladder_scores_nan    no tested count below full width scored on
                             every fold
        full_score_nan       the full-width mean is not a number
        tol_nan              the tolerance is not a number
    """
    if plateau_k is not None:
        return "reached"
    if full_score is None or not np.isfinite(full_score):
        return "full_score_nan"
    if not np.isfinite(tol):
        return "tol_nan"
    if len(ks) == 0:
        return "no_count_below_width"
    if not any(s is not None and np.isfinite(s) for s in scores):
        return "ladder_scores_nan"
    return "above_largest_tested"


def rank_folds(X, y, folds):
    """Rank one representation's own features on each fold's TRAINING rows.

    Returns one ordering of that representation's feature indices per fold,
    most important first. The ranking is lightgbm_gain on the training rows and
    the labels. It never reads the held-out rows, and it never reads the
    downstream model, so one ordering per fold serves every downstream model.
    """
    X = H._clean(X)
    y = np.asarray(y, dtype=float)
    orders, n_ranked = [], []
    for tr, _va in folds:
        imp = H.rank_columns(X[tr], y[tr], method="lightgbm_gain")
        orders.append(np.argsort(imp)[::-1])   # descending importance
        # Features the ranker gave any gain at all. The rest tie at exactly 0
        # (never split on, or cut by the 2,000-feature variance pre-screen in
        # compute_feature_importance), so past this count the top-k is filled
        # in argsort's arbitrary tie order, not by importance.
        n_ranked.append(int(np.sum(imp > 0)))
    return orders, n_ranked


def saturation_curve(X, y, folds, orders, model_name, metric_fn,
                     task_type, ks):
    """Fold-safe importance-truncated learning curve for ONE rep alone.

    `orders` is that representation's per-fold feature ranking from
    rank_folds(), computed once and handed to every downstream model. Per fold:
    for each k in `ks` (plus 'full') fit `model_name` on the top-k training
    features and score the held-out rows. Averages per-fold metric values.

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

    assert len(orders) == len(folds), (
        f"got {len(orders)} feature rankings for {len(folds)} folds")

    per_fold = [[] for _ in ladder]
    for (tr, va), order in zip(folds, orders):
        Xtr_all, Xva_all = X[tr], X[va]
        ytr = y[tr]
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
            # _predict_for_metric, not .predict (dirty local run 2026-09-28):
            # on an AUROC dataset (humanness_therapeutic) .predict returns hard
            # 0/1 labels and the AUROC collapses to the mean of sensitivity and
            # specificity. Falls through to .predict for rmse/spearman.
            from kirby.find_best_hybrid import _predict_for_metric
            pred = _predict_for_metric(model, Xva, metric_fn)
            per_fold[j].append(float(metric_fn(y[va], pred)))

    # Plain mean over ALL folds: a feature count where any fold failed to score
    # is nan, not a mean over the folds that did. A nanmean there compared a
    # mean over fewer folds against a full-width mean over all of them.
    mean = [float(np.mean(f)) if len(f) else float("nan") for f in per_fold]
    n_finite = [int(np.sum(np.isfinite(np.asarray(f, dtype=float))))
                for f in per_fold]
    return {
        "ks": [int(k) for k in ladder],
        "width": int(width),
        "mean": mean,
        "n_folds_scored": n_finite,
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
        "plateau_tol_by_model": {},
        "plateau_definition": (
            "plateau_n_features = the smallest feature count whose %d-fold "
            "mean sits within plateau_tol of the full-width %d-fold mean, in "
            "the improving direction. plateau_tol is in %s units: it is the "
            "mean over this dataset's representations of the standard "
            "deviation (ddof=%d) of that representation's %d full-width fold "
            "scores, taken separately under each downstream model. "
            "plateau_n_features=None: plateau_status says why. "
            "'above_largest_tested' means more features than the largest "
            "tested count below the width, not necessarily all of them. "
            "k_exceeds_ranked marks counts above the smallest per-fold number "
            "of features with nonzero lightgbm gain; past that the top-k is "
            "filled in arbitrary tie order, not by importance. The full-width "
            "entry is never flagged, because it holds every feature."
            % (n_folds, n_folds, mname, PLATEAU_TOL_DDOF, n_folds)),
    }

    # Feature rankings, once per representation per fold. They read only the
    # training rows and the labels, never the downstream model, so both models
    # below reuse them: one ranking per representation per fold, not one per
    # representation per fold per model.
    orders, n_ranked = {}, {}
    for rep_name in used:
        t0 = time.time()
        orders[rep_name], n_ranked[rep_name] = rank_folds(
            reps[rep_name], labels, folds)
        if verbose:
            print(f"    ranked {rep_name:<14} {reps[rep_name].shape[1]:>5} features "
                  f"x {n_folds} folds ({round(time.time() - t0, 1)}s)")
    if verbose:
        print(f"[c20] {name}: {len(used) * n_folds} feature rankings computed, "
              f"reused by all {len(models)} downstream models")

    for model_name in models:
        ds_result["models"][model_name] = {}
        for rep_name in used:
            t0 = time.time()
            curve = saturation_curve(reps[rep_name], labels, folds,
                                     orders[rep_name], model_name, mfn, task, ks)
            curve["full_score"] = curve["full_mean"]
            curve["full_fold_std"] = fold_spread(curve["full_per_fold"])
            # How far down the ranking is an actual ranking. A feature count
            # above the smallest per-fold count of nonzero-gain features is
            # partly filled with features the ranker never distinguished.
            nr = n_ranked[rep_name]
            curve["n_nonzero_importance_per_fold"] = nr
            # The last entry is the full width. Every feature is in it, so the
            # ranking order cannot matter there and it is never flagged.
            curve["k_exceeds_ranked"] = [
                bool(k > min(nr)) and k < curve["width"] for k in curve["ks"]]
            curve["seconds"] = round(time.time() - t0, 1)
            ds_result["models"][model_name][rep_name] = curve
            if verbose:
                print(f"    [{model_name}] {rep_name:<12} width={curve['width']:>5} "
                      f"full={curve['full_mean']:.4f} "
                      f"fold_std={curve['full_fold_std']:.4f} "
                      f"({curve['seconds']}s) curve="
                      f"{[round(x, 3) for x in curve['mean']]}")

        # One tolerance per dataset per downstream model, set from the folds.
        tol = plateau_tolerance(list(ds_result["models"][model_name].values()))
        ds_result["plateau_tol_by_model"][model_name] = tol
        if verbose:
            print(f"  [{model_name}] tolerance = {tol:.4f} {mname}, the mean over "
                  f"{len(used)} representations of the standard deviation of "
                  f"that representation's {n_folds} full-width fold scores")
        for rep_name, curve in ds_result["models"][model_name].items():
            curve["plateau_tol"] = tol
            curve["plateau_n_features"] = _find_plateau(
                curve["ks"][:-1], curve["mean"][:-1], curve["full_mean"],
                lower, tol)
            curve["plateau_status"] = plateau_status(
                curve["ks"][:-1], curve["mean"][:-1], curve["full_mean"], tol,
                curve["plateau_n_features"])
            pn = curve["plateau_n_features"]
            curve["plateau_beyond_ranked"] = (
                None if pn is None
                else bool(pn > min(curve["n_nonzero_importance_per_fold"])))
            if verbose:
                print(f"    [{model_name}] {rep_name:<12} plateau at "
                      f"{pn if pn is not None else curve['plateau_status']}"
                      f" features (nonzero-gain features per fold: "
                      f"{curve['n_nonzero_importance_per_fold']})")

    return ds_result


def _plateau_table(results):
    """Compact representation x (dataset, model) table: plateau feature
    count, full-width score, and the tolerance that decided the plateau."""
    lines = ["", "=" * 70, "PLATEAU TABLE (rep -> saturation feature-count @ full-rep score)",
             "=" * 70]
    for name, ds in results.items():
        if "error" in ds:
            lines.append(f"\n{name}: ERROR - {ds['error']}")
            continue
        lines.append(f"\n{name}  (metric={ds['metric']}, "
                     f"lower_is_better={ds['lower_is_better']}, n={ds['n']})")
        for model_name, reps in ds["models"].items():
            tol = ds.get("plateau_tol_by_model", {}).get(model_name,
                                                         float("nan"))
            lines.append(f"  model={model_name}  "
                         f"tolerance={tol:.4f} {ds['metric']} "
                         f"(mean fold-to-fold spread at full width)")
            for rep_name, c in reps.items():
                pn = (c["plateau_n_features"]
                      if c["plateau_n_features"] is not None
                      else c.get("plateau_status", "FULL"))
                lines.append(f"    {rep_name:<14} plateau@{str(pn):<6} features  "
                             f"full_score={c['full_score']:.4f} "
                             f"fold_std={c['full_fold_std']:.4f} "
                             f"(width={c['width']} features)")
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
                    help="use the first 8 of the dataset's staged reps")
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
        rep_map = {d: smoke_reps(d) for d in ds_list}
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
